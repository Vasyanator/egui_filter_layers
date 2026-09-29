/*
File: glow_backend/coverage.rs

Purpose:
The glow side of the shape-mask coverage step (F9): the mesh program (`shaders/mesh.wgsl`
translated by naga's GLSL writer, `glsl::translate_mesh`), its vertex array (egui's 20-byte
vertex: two float pairs and an integer colour attribute), the vertex / index / locals buffers and
the RGBA8 coverage target, and the draw that rasterizes a mask's packed meshes into it.

Key structures:
- `CoverageResources`: created on the first shape-masked layer (`Resources::coverage`), freed
  with the other resources.

Key functions:
- `CoverageResources::create()` / `draw()` / `destroy()`.

Notes:
The draw clears the target, enables the ADDITIVE coverage union (`ONE, ONE` for colour and alpha,
saturating in RGBA8: abutting pieces' complementary feathers sum to 1, so they leave no seam;
the same rule as wgpu's `COVERAGE_BLEND`) over `(0, 0, used)` and draws one `glDrawElements` per packed draw
with its texture on unit 0 (egui_glow's texture object, sampled through its own parameters).
Indices are absolute (`mask::PackedCoverage`), so no base vertex is needed (GLES 3.0 has none).
The locals block goes to `UNIFORM_BINDING_POINT` (re-bound by every later draw of the run). The
target is stored bottom-up like every GL texture here: the mesh shader's clip-space y points up,
exactly as the capture blit leaves the capture, so the prelude's `flip_y` reads both alike.
*/

use glow::HasContext as _;

use super::glsl::{GlslProgram, GlslTarget, translate_mesh};
use super::resources::{GlError, RenderTarget, TargetFormat, UNIFORM_BINDING_POINT, build_program};
use crate::mask::{PackedCoverage, VERTEX_BYTES};
use crate::plan::MESH_LOCALS_SIZE;

/// The coverage rasterizer's GL objects.
#[derive(Debug)]
pub(super) struct CoverageResources {
    /// `mesh.wgsl` `vs_main` + `fs_main`.
    program: glow::Program,
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
}

/// `VERTEX_BYTES` as the `i32` stride GL takes.
const STRIDE: i32 = 20;

impl CoverageResources {
    /// Translates and builds the mesh program, creates the buffers, the vertex array and a 1x1
    /// coverage target. Leaves no program, vertex array or buffer bound (the draw framebuffer
    /// changes; the caller restores it).
    ///
    /// # Errors
    /// [`GlError::Translate`] (a crate bug), a compile / link error, or an object the driver
    /// refused; everything created so far is freed.
    ///
    /// # Safety
    /// `gl` must be current on this thread.
    pub(super) unsafe fn create(gl: &glow::Context, target: GlslTarget) -> Result<Self, GlError> {
        debug_assert_eq!(usize::try_from(STRIDE).unwrap_or(0), VERTEX_BYTES);
        let mesh = translate_mesh(target)?;
        let glsl = GlslProgram { vertex: mesh.vertex, fragment: mesh.fragment, uniform_block: None, params_block: None, samplers: mesh.samplers };
        // SAFETY: `gl` is current on this thread (this `unsafe fn`'s contract); every object is created here and freed again on the error paths.
        unsafe {
            let program = build_program(gl, &glsl)?;
            // naga names the block per stage: bind every stage's name to the one binding point.
            for block in &mesh.locals_blocks {
                if let Some(index) = gl.get_uniform_block_index(program, block) {
                    gl.uniform_block_binding(program, index, UNIFORM_BINDING_POINT);
                }
            }
            let mut created: Vec<glow::Buffer> = Vec::with_capacity(3);
            for _ in 0..3 {
                match gl.create_buffer() {
                    Ok(buffer) => created.push(buffer),
                    Err(detail) => {
                        for buffer in created {
                            gl.delete_buffer(buffer);
                        }
                        gl.delete_program(program);
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
                    gl.delete_program(program);
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

            let target = RenderTarget::create(gl, TargetFormat::Rgba8).and_then(|mut target| match target.ensure_size(gl, [1, 1], i32::MAX) {
                Ok(()) => Ok(target),
                Err(error) => {
                    target.destroy(gl);
                    Err(error)
                }
            });
            match target {
                Ok(target) => Ok(Self { program, vertex_array, vertices, indices, locals, target }),
                Err(error) => {
                    gl.delete_vertex_array(vertex_array);
                    free_buffers(gl);
                    gl.delete_program(program);
                    Err(error)
                }
            }
        }
    }

    /// Rasterizes `packed` into the coverage target: grows it to at least `used_px` (`max` = the
    /// context's limit), clears it, and draws draw `i` with `textures[i]` on unit 0 over `(0, 0,
    /// used_px)` with `locals` (`plan::coverage_locals`). Leaves blend disabled, and the framebuffer, program,
    /// unit-0 texture and uniform-buffer bindings changed (the caller's cleanup resets them).
    ///
    /// # Errors
    /// As [`RenderTarget::ensure_size`].
    ///
    /// # Safety
    /// `gl` must be current on this thread with scissoring disabled; `textures` are alive on it
    /// and `textures.len() == packed.draws.len()`.
    pub(super) unsafe fn draw(&mut self, gl: &glow::Context, packed: &PackedCoverage, textures: &[glow::Texture], locals: &[u8; MESH_LOCALS_SIZE], used_px: [i32; 2], max: i32) -> Result<(), GlError> {
        // SAFETY: `gl` is current on this thread (this `unsafe fn`'s contract); every object is ours or one of egui_glow's live textures (caller contract), and every index of `packed` addresses its own vertex upload.
        unsafe {
            self.target.ensure_size(gl, used_px, max)?;
            self.target.clear(gl);
            if packed.draws.is_empty() {
                return Ok(());
            }
            gl.viewport(0, 0, used_px[0], used_px[1]);
            gl.enable(glow::BLEND);
            gl.blend_equation(glow::FUNC_ADD);
            // Additive union (roadmap §19 D1), identical to wgpu's `COVERAGE_BLEND`.
            gl.blend_func(glow::ONE, glow::ONE);
            gl.use_program(Some(self.program));
            gl.bind_buffer(glow::UNIFORM_BUFFER, Some(self.locals));
            gl.buffer_sub_data_u8_slice(glow::UNIFORM_BUFFER, 0, locals);
            gl.bind_buffer_base(glow::UNIFORM_BUFFER, UNIFORM_BINDING_POINT, Some(self.locals));
            gl.bind_vertex_array(Some(self.vertex_array));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.vertices));
            gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, &packed.vertices, glow::STREAM_DRAW);
            // The element buffer binding is vertex-array state: `indices` is bound through it.
            gl.buffer_data_u8_slice(glow::ELEMENT_ARRAY_BUFFER, &packed.indices, glow::STREAM_DRAW);
            gl.active_texture(glow::TEXTURE0);
            for (draw, texture) in packed.draws.iter().zip(textures) {
                gl.bind_texture(glow::TEXTURE_2D, Some(*texture));
                let (Ok(count), Ok(offset)) = (i32::try_from(draw.index_count), i32::try_from(u64::from(draw.first_index) * 4)) else {
                    continue;
                };
                gl.draw_elements(glow::TRIANGLES, count, glow::UNSIGNED_INT, offset);
            }
            gl.bind_vertex_array(None);
            gl.bind_buffer(glow::ARRAY_BUFFER, None);
            gl.disable(glow::BLEND);
        }
        Ok(())
    }

    /// Frees every object.
    ///
    /// # Safety
    /// `gl` must be current on this thread and own these objects.
    pub(super) unsafe fn destroy(self, gl: &glow::Context) {
        // SAFETY: `gl` is current on this thread (this `unsafe fn`'s contract) and owns every object (caller contract); `self` is consumed.
        unsafe {
            gl.delete_program(self.program);
            gl.delete_vertex_array(self.vertex_array);
            for buffer in [self.vertices, self.indices, self.locals] {
                gl.delete_buffer(buffer);
            }
            self.target.destroy(gl);
        }
    }
}
