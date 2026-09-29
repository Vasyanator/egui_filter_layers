/*
File: glow_backend/coverage.rs

Purpose:
The glow side of the mask rasterizer: the shape-mask coverage step (F9) and the region-id passes
of a region mask (R1). Holds the two mesh programs (`shaders/mesh.wgsl` translated by naga's GLSL
writer, `glsl::translate_mesh`: `fs_main` for coverage, `fs_region` for ids), their shared vertex
array (egui's 20-byte vertex: two float pairs and an integer colour attribute), the vertex /
index / locals buffers, the RGBA8 coverage target and the RGBA8 region-id target, and the draw that
rasterizes a mask's packed meshes into them.

Key structures:
- `CoverageResources`: created on the first masked layer (`Resources::coverage`), freed with the
  other resources.

Key functions:
- `CoverageResources::create()` / `draw()` / `destroy()`.

Notes:
The coverage draw clears the coverage target, enables the ADDITIVE coverage union (`ONE, ONE` for
colour and alpha, saturating in RGBA8: abutting pieces' complementary feathers sum to 1, so they
leave no seam; the same rule as wgpu's `COVERAGE_BLEND`) over `(0, 0, used)` and draws one
`glDrawElements` per packed draw with its texture on unit 0 (egui_glow's texture object, sampled
through its own parameters). Indices are absolute (`mask::PackedCoverage`), so no base vertex is
needed (GLES 3.0 has none). With `CoverageStep::regions > 0` the region-id target (same window,
same viewport) is then cleared to id 255 (`REGION_NONE`) and every region's draw range is drawn
twice with `fs_region` and blending DISABLED (`plan::REGION_PASS_THRESHOLDS`: pass A then pass B,
regions in id order, each draw with its own `plan::region_locals` re-uploaded by
`glBufferSubData`), so the last region drawn owns a pixel — the same sequence as wgpu's
`coverage.rs`. The locals block goes to `UNIFORM_BINDING_POINT` (re-bound by every later draw of
the run). Both targets are stored bottom-up like every GL texture here: the mesh shader's
clip-space y points up, exactly as the capture blit leaves the capture, so the prelude's `flip_y`
reads them like the capture.
*/

use std::ops::Range;

use glow::HasContext as _;

use super::glsl::{GlslProgram, GlslTarget, MeshEntry, translate_mesh};
use super::resources::{GlError, RenderTarget, TargetFormat, UNIFORM_BINDING_POINT, build_program};
use crate::geometry::MaskTransform;
use crate::mask::{PackedRegions, VERTEX_BYTES};
use crate::plan::{CoverageStep, MESH_LOCALS_SIZE, REGION_PASS_THRESHOLDS, coverage_locals, region_locals};

/// The region-id target's clear value: id 255 (`REGION_NONE`) in `.r` (`255 / 255 = 1`).
const REGION_CLEAR: [f32; 4] = [1.0, 0.0, 0.0, 0.0];

/// The mask rasterizer's GL objects.
#[derive(Debug)]
pub(super) struct CoverageResources {
    /// `mesh.wgsl` `vs_main` + `fs_main` (coverage, additive blend).
    program: glow::Program,
    /// `mesh.wgsl` `vs_main` + `fs_region` (region ids, no blend).
    region_program: glow::Program,
    /// Attribute layout of egui's vertex over `vertices`, with `indices` as its element buffer.
    vertex_array: glow::VertexArray,
    /// Packed vertices (re-specified per draw).
    vertices: glow::Buffer,
    /// Packed `u32` indices (re-specified per draw).
    indices: glow::Buffer,
    /// The `MeshLocals` block.
    locals: glow::Buffer,
    /// The coverage texture (RGBA8, grow-only; the run's window at its origin).
    pub target: RenderTarget,
    /// The region-id texture (RGBA8, grow-only; the run's window at its origin, `.r * 255` = id).
    pub region_target: RenderTarget,
}

/// `VERTEX_BYTES` as the `i32` stride GL takes.
const STRIDE: i32 = 20;

/// Translates and links the mesh program with fragment entry `entry` and binds every stage's
/// `MeshLocals` block name to [`UNIFORM_BINDING_POINT`].
///
/// # Errors
/// [`GlError::Translate`] (a crate bug) or a compile / link error; nothing is left allocated.
///
/// # Safety
/// `gl` must be current on this thread.
unsafe fn mesh_program(gl: &glow::Context, target: GlslTarget, entry: MeshEntry) -> Result<glow::Program, GlError> {
    let mesh = translate_mesh(target, entry)?;
    let glsl = GlslProgram { vertex: mesh.vertex, fragment: mesh.fragment, uniform_block: None, params_block: None, samplers: mesh.samplers };
    // SAFETY: `gl` is current on this thread (this `unsafe fn`'s contract); the program is created here.
    unsafe {
        let program = build_program(gl, &glsl)?;
        // naga names the block per stage: bind every stage's name to the one binding point.
        for block in &mesh.locals_blocks {
            if let Some(index) = gl.get_uniform_block_index(program, block) {
                gl.uniform_block_binding(program, index, UNIFORM_BINDING_POINT);
            }
        }
        Ok(program)
    }
}

/// Creates an RGBA8 target with a 1x1 allocation.
///
/// # Errors
/// As [`RenderTarget::create`] / [`RenderTarget::ensure_size`]; nothing is left allocated.
///
/// # Safety
/// `gl` must be current on this thread.
unsafe fn small_target(gl: &glow::Context) -> Result<RenderTarget, GlError> {
    // SAFETY: forwarded contract; the target is freed again on the error path.
    unsafe {
        RenderTarget::create(gl, TargetFormat::Rgba8).and_then(|mut target| match target.ensure_size(gl, [1, 1], i32::MAX) {
            Ok(()) => Ok(target),
            Err(error) => {
                target.destroy(gl);
                Err(error)
            }
        })
    }
}

impl CoverageResources {
    /// Translates and builds both mesh programs, creates the buffers, the vertex array and 1x1
    /// coverage / region-id targets. Leaves no program, vertex array or buffer bound (the draw
    /// framebuffer changes; the caller restores it).
    ///
    /// # Errors
    /// [`GlError::Translate`] (a crate bug), a compile / link error, or an object the driver
    /// refused; everything created so far is freed.
    ///
    /// # Safety
    /// `gl` must be current on this thread.
    pub(super) unsafe fn create(gl: &glow::Context, target: GlslTarget) -> Result<Self, GlError> {
        debug_assert_eq!(usize::try_from(STRIDE).unwrap_or(0), VERTEX_BYTES);
        // SAFETY: `gl` is current on this thread (this `unsafe fn`'s contract); every object is created here and freed again on the error paths.
        unsafe {
            let program = mesh_program(gl, target, MeshEntry::Coverage)?;
            let region_program = match mesh_program(gl, target, MeshEntry::Region) {
                Ok(region_program) => region_program,
                Err(error) => {
                    gl.delete_program(program);
                    return Err(error);
                }
            };
            let free_programs = |gl: &glow::Context| {
                gl.delete_program(program);
                gl.delete_program(region_program);
            };
            let mut created: Vec<glow::Buffer> = Vec::with_capacity(3);
            for _ in 0..3 {
                match gl.create_buffer() {
                    Ok(buffer) => created.push(buffer),
                    Err(detail) => {
                        for buffer in created {
                            gl.delete_buffer(buffer);
                        }
                        free_programs(gl);
                        return Err(GlError::CreateResource { resource: "mask buffer", detail });
                    }
                }
            }
            let (vertices, indices, locals) = (created[0], created[1], created[2]);
            let free_buffers = |gl: &glow::Context| {
                for buffer in [vertices, indices, locals] {
                    gl.delete_buffer(buffer);
                }
            };
            let vertex_array = match gl.create_vertex_array() {
                Ok(vertex_array) => vertex_array,
                Err(detail) => {
                    free_buffers(gl);
                    free_programs(gl);
                    return Err(GlError::CreateResource { resource: "vertex array", detail });
                }
            };
            gl.bind_vertex_array(Some(vertex_array));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vertices));
            gl.enable_vertex_attrib_array(0);
            gl.vertex_attrib_pointer_f32(0, 2, glow::FLOAT, false, STRIDE, 0);
            gl.enable_vertex_attrib_array(1);
            gl.vertex_attrib_pointer_f32(1, 2, glow::FLOAT, false, STRIDE, 8);
            // The colour is `u32` in the shader (unpacked by hand): an INTEGER attribute.
            gl.enable_vertex_attrib_array(2);
            gl.vertex_attrib_pointer_i32(2, 1, glow::UNSIGNED_INT, STRIDE, 16);
            gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(indices));
            gl.bind_vertex_array(None);
            gl.bind_buffer(glow::ARRAY_BUFFER, None);
            gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, None);
            gl.bind_buffer(glow::UNIFORM_BUFFER, Some(locals));
            gl.buffer_data_size(glow::UNIFORM_BUFFER, i32::try_from(MESH_LOCALS_SIZE).unwrap_or(32), glow::DYNAMIC_DRAW);
            gl.bind_buffer(glow::UNIFORM_BUFFER, None);

            let free_objects = |gl: &glow::Context| {
                gl.delete_vertex_array(vertex_array);
                free_buffers(gl);
                free_programs(gl);
            };
            let target = match small_target(gl) {
                Ok(target) => target,
                Err(error) => {
                    free_objects(gl);
                    return Err(error);
                }
            };
            match small_target(gl) {
                Ok(region_target) => Ok(Self { program, region_program, vertex_array, vertices, indices, locals, target, region_target }),
                Err(error) => {
                    target.destroy(gl);
                    free_objects(gl);
                    Err(error)
                }
            }
        }
    }

    /// Rasterizes `packed` for `step` (`plan::CoverageStep`): grows the coverage target to at
    /// least `step.used_px` (`max` = the context's limit), clears it and draws every packed draw
    /// `i` with `textures[i]` on unit 0 over `(0, 0, used_px)` with `plan::coverage_locals`
    /// (additive union). With `step.regions > 0` (R1) it then grows the region-id target the same
    /// way, clears it to `REGION_NONE` and runs the id passes: for every threshold of
    /// `REGION_PASS_THRESHOLDS`, every region's range of `packed.regions` in id order with
    /// `plan::region_locals`, blending disabled. Leaves blend disabled, and the framebuffer,
    /// program, unit-0 texture and uniform-buffer bindings changed (the caller's cleanup resets
    /// them).
    ///
    /// # Errors
    /// As [`RenderTarget::ensure_size`].
    ///
    /// # Safety
    /// `gl` must be current on this thread with scissoring disabled; `textures` are alive on it,
    /// `textures.len() == packed.packed.draws.len()`, `packed.regions.len() == step.regions` and
    /// every region range lies within the draws ([`CoverageResources::fits`]).
    pub(super) unsafe fn draw(&mut self, gl: &glow::Context, packed: &PackedRegions, textures: &[glow::Texture], transform: &MaskTransform, step: &CoverageStep, max: i32) -> Result<(), GlError> {
        let used_px = step.used_px;
        let regions = step.regions > 0;
        // SAFETY: `gl` is current on this thread (this `unsafe fn`'s contract); every object is ours or one of egui_glow's live textures (caller contract), and every index of `packed` addresses its own vertex upload.
        unsafe {
            self.target.ensure_size(gl, used_px, max)?;
            self.target.clear(gl);
            if regions {
                // Every pass step binds the region ids, so the target is sized and cleared even
                // when no region has a mesh (every pixel then reads `REGION_NONE`).
                self.region_target.ensure_size(gl, used_px, max)?;
                self.region_target.clear_to(gl, REGION_CLEAR);
            }
            if packed.packed.draws.is_empty() {
                return Ok(());
            }
            gl.viewport(0, 0, used_px[0], used_px[1]);
            gl.bind_vertex_array(Some(self.vertex_array));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.vertices));
            gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, &packed.packed.vertices, glow::STREAM_DRAW);
            // The element buffer binding is vertex-array state: `indices` is bound through it.
            gl.buffer_data_u8_slice(glow::ELEMENT_ARRAY_BUFFER, &packed.packed.indices, glow::STREAM_DRAW);
            gl.bind_buffer_base(glow::UNIFORM_BUFFER, UNIFORM_BINDING_POINT, Some(self.locals));
            gl.active_texture(glow::TEXTURE0);

            gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(self.target.framebuffer));
            gl.enable(glow::BLEND);
            gl.blend_equation(glow::FUNC_ADD);
            // Additive union (roadmap §19 D1), identical to wgpu's `COVERAGE_BLEND`.
            gl.blend_func(glow::ONE, glow::ONE);
            gl.use_program(Some(self.program));
            self.upload_locals(gl, &coverage_locals(transform, step));
            draw_range(gl, packed, textures, 0..packed.packed.draws.len());
            // Id passes replace (the last region drawn owns the pixel).
            gl.disable(glow::BLEND);

            if regions {
                gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(self.region_target.framebuffer));
                gl.use_program(Some(self.region_program));
                for threshold in REGION_PASS_THRESHOLDS {
                    for (id, range) in (0_u32..).zip(&packed.regions) {
                        if range.is_empty() {
                            continue;
                        }
                        self.upload_locals(gl, &region_locals(transform, step, id, threshold));
                        draw_range(gl, packed, textures, range.clone());
                    }
                }
            }
            gl.bind_vertex_array(None);
            gl.bind_buffer(glow::ARRAY_BUFFER, None);
        }
        Ok(())
    }

    /// Whether `packed` and `textures` meet [`CoverageResources::draw`]'s shape precondition for
    /// `step`: one texture per draw, one range per region, every range within the draws.
    pub(super) fn fits(packed: &PackedRegions, textures: &[glow::Texture], step: &CoverageStep) -> bool {
        let draws = packed.packed.draws.len();
        textures.len() == draws && usize::try_from(step.regions).is_ok_and(|count| count == packed.regions.len()) && packed.regions.iter().all(|range| range.start <= range.end && range.end <= draws)
    }

    /// Writes one `MeshLocals` block into the locals buffer (already bound at the binding point).
    ///
    /// # Safety
    /// `gl` must be current on this thread.
    unsafe fn upload_locals(&self, gl: &glow::Context, locals: &[u8; MESH_LOCALS_SIZE]) {
        // SAFETY: `gl` is current on this thread (this `unsafe fn`'s contract); the buffer is ours, allocated with `MESH_LOCALS_SIZE` bytes.
        unsafe {
            gl.bind_buffer(glow::UNIFORM_BUFFER, Some(self.locals));
            gl.buffer_sub_data_u8_slice(glow::UNIFORM_BUFFER, 0, locals);
        }
    }

    /// Frees every object.
    ///
    /// # Safety
    /// `gl` must be current on this thread and own these objects.
    pub(super) unsafe fn destroy(self, gl: &glow::Context) {
        // SAFETY: `gl` is current on this thread (this `unsafe fn`'s contract) and owns every object (caller contract); `self` is consumed.
        unsafe {
            gl.delete_program(self.program);
            gl.delete_program(self.region_program);
            gl.delete_vertex_array(self.vertex_array);
            for buffer in [self.vertices, self.indices, self.locals] {
                gl.delete_buffer(buffer);
            }
            self.target.destroy(gl);
            self.region_target.destroy(gl);
        }
    }
}

/// Draws the packed draws `range` (each with its texture on unit 0, `TEXTURE0` active) with the
/// bound program, vertex array and framebuffer. A range outside the draws draws nothing (excluded
/// by [`CoverageResources::fits`]).
///
/// # Safety
/// `gl` must be current on this thread; `textures` are alive on it; the upload of `packed` is bound.
unsafe fn draw_range(gl: &glow::Context, packed: &PackedRegions, textures: &[glow::Texture], range: Range<usize>) {
    let (Some(draws), Some(textures)) = (packed.packed.draws.get(range.clone()), textures.get(range)) else {
        return;
    };
    // SAFETY: `gl` is current on this thread (this `unsafe fn`'s contract); every index addresses the bound vertex upload (`mask::PackedCoverage` contract) and every texture is alive (caller contract).
    unsafe {
        for (draw, texture) in draws.iter().zip(textures) {
            gl.bind_texture(glow::TEXTURE_2D, Some(*texture));
            let (Ok(count), Ok(offset)) = (i32::try_from(draw.index_count), i32::try_from(u64::from(draw.first_index) * 4)) else {
                continue;
            };
            gl.draw_elements(glow::TRIANGLES, count, glow::UNSIGNED_INT, offset);
        }
    }
}
