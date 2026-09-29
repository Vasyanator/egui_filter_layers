/*
File: wgpu_backend/coverage.rs

Purpose:
The wgpu mesh machinery shared by the backdrop re-render (UI) and the shape-mask coverage
rasterizer (F9, UI and offline): the mesh pipeline over `shaders/mesh.wgsl` (egui's vertex
layout, egui-wgpu's texture group layout; egui's blend for the backdrop, an additive one for
coverage), and the coverage step of a chain run — the mask's packed meshes (`mask::PackedCoverage`) drawn into an RGBA8
texture whose alpha the composite reads, and for a region mask (R1) the two id passes into the
region-id texture every pass reads. egui-free like `chain.rs` (only bytes and wgpu objects cross
this boundary), so the offline renderer runs it unchanged.

Key structures:
- `CoverageRasterizer`: per-device objects (locals layouts, the coverage and region-id pipelines,
  a white texture group for untextured meshes).
- `CoverageTargets`: per-run coverage and region-id textures and buffers (grow-only).
- `CoverageJob`: one run's packed meshes, their transform, a texture bind group per draw and the
  draw range of every region.
- `CoverageOutput`: the textures a recorded coverage step leaves (bound at `Slot::Coverage` /
  `Slot::RegionId`).

Key functions:
- `mesh_pipeline()`: a pipeline drawing egui meshes into a format with a blend (backdrop:
  `EGUI_BLEND`; coverage: the additive `COVERAGE_BLEND`).
- `CoverageRasterizer::record()`: the coverage step of a plan.

Notes:
Coverage is the ADDITIVE union of the meshes (`COVERAGE_BLEND`: `One, One`, saturating in the
RGBA8 target) into a target cleared to transparent: two pieces meeting edge to edge have
complementary feathers that sum to 1 on the shared edge, so abutting pieces leave no seam (egui's
"over" blend would give 1 - (1 - a)(1 - b) = 0.75 there). Overlapping pieces therefore sum their
alphas (translucent fills included). The backdrop re-render keeps egui's blend (`EGUI_BLEND`).
No MSAA: the tessellator's feathering is the anti-aliasing. The target holds the plan's WINDOW at
its origin.
Region ids (R1): after the coverage draw, the region-id texture (RGBA8, the window at its origin)
is cleared to id 255 (`REGION_NONE`) and every region's draws run twice (`plan::
REGION_PASS_THRESHOLDS`: pass A then pass B, regions in id order) with `mesh.wgsl`'s `fs_region`
and NO blend, so the last region drawn owns a pixel. Each (pass, region) has its own `MeshLocals`
block in one buffer at the device's uniform offset alignment, selected by a dynamic offset.
*/

use super::chain::ChainRecording;
use super::gpu::{GrowBuffer, Target, begin_pass, shader, uniform_entry};
use std::ops::Range;

use crate::geometry::{MaskTransform, px_f32};
use crate::mask::{MAX_REGIONS, PackedCoverage, VERTEX_BYTES};
use crate::plan::{CoverageStep, MESH_LOCALS_SIZE, REGION_PASS_THRESHOLDS, coverage_locals, region_locals};

/// Format of the coverage texture: RGBA8 renders and filters everywhere (WebGL2 included);
/// the coverage is its alpha.
pub(crate) const COVERAGE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Format of the region-id texture: `.r * 255` = the id, exact in 8-bit unorm.
pub(crate) const REGION_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// The region-id texture's clear value: id 255 (`REGION_NONE`) in `.r`.
const REGION_CLEAR: wgpu::Color = wgpu::Color { r: 1.0, g: 0.0, b: 0.0, a: 0.0 };

/// egui-wgpu's premultiplied-alpha blend state (egui-wgpu 0.36 `renderer.rs`, `Renderer::new`).
pub(crate) const EGUI_BLEND: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha, operation: wgpu::BlendOperation::Add },
    alpha: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::OneMinusDstAlpha, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add },
};

/// The coverage union (roadmap §19 D1): every mesh ADDS its premultiplied colour and alpha, the
/// RGBA8 target saturates at 1. Complementary feathers of abutting pieces sum to full coverage.
pub(crate) const COVERAGE_BLEND: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add },
    alpha: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add },
};

pub(crate) use crate::mask::MESH_WGSL;

/// A pipeline drawing egui meshes (`mesh.wgsl`'s `vs_main` + `fragment_entry`) with `blend`
/// (`None`: replace) into `format`, single-sampled. `layout` = the locals group then egui's
/// texture group.
pub(crate) fn mesh_pipeline(device: &wgpu::Device, layout: &wgpu::PipelineLayout, module: &wgpu::ShaderModule, format: wgpu::TextureFormat, fragment_entry: &str, blend: Option<wgpu::BlendState>, label: &str) -> wgpu::RenderPipeline {
    let stride = u64::try_from(VERTEX_BYTES).unwrap_or(20);
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some("vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[Some(wgpu::VertexBufferLayout { array_stride: stride, step_mode: wgpu::VertexStepMode::Vertex, attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2, 2 => Uint32] })],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some(fragment_entry),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState { format, blend, write_mask: wgpu::ColorWrites::ALL })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

/// The layout of the mesh shader's group 0 (`MeshLocals`).
pub(crate) fn mesh_locals_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("egui_shader_layers mesh locals"), entries: &[uniform_entry(0, Some(MESH_LOCALS_SIZE))] })
}

/// The layout of the region-id passes' group 0: `MeshLocals` at a DYNAMIC offset (one block per
/// pass and region in one buffer).
fn region_locals_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let mut entry = uniform_entry(0, Some(MESH_LOCALS_SIZE));
    if let wgpu::BindingType::Buffer { has_dynamic_offset, .. } = &mut entry.ty {
        *has_dynamic_offset = true;
    }
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("egui_shader_layers region locals"), entries: &[entry] })
}

/// Per-device objects of the coverage rasterizer.
#[derive(Debug)]
pub(crate) struct CoverageRasterizer {
    /// Group 0 layout (`MeshLocals`).
    locals_layout: wgpu::BindGroupLayout,
    /// `mesh.wgsl` `vs_main` + `fs_main` into [`COVERAGE_FORMAT`] with [`COVERAGE_BLEND`].
    pipeline: wgpu::RenderPipeline,
    /// Group 0 layout of the region-id passes (`MeshLocals`, dynamic offset).
    region_locals_layout: wgpu::BindGroupLayout,
    /// `mesh.wgsl` `vs_main` + `fs_region` into [`REGION_FORMAT`], no blend (R1).
    region_pipeline: wgpu::RenderPipeline,
    /// Stride of the region-id passes' locals blocks: `MESH_LOCALS_SIZE` rounded up to the
    /// device's `min_uniform_buffer_offset_alignment`.
    region_locals_stride: usize,
    /// A 1x1 white texture in egui's texture layout: the texture of untextured meshes when the
    /// caller has no egui texture to bind (offline).
    white_group: wgpu::BindGroup,
}

impl CoverageRasterizer {
    /// Builds the pipeline over `texture_layout` (egui-wgpu's texture layout) and the white group
    /// from `white` + `sampler`. Must run inside the caller's error scope.
    pub(crate) fn create(device: &wgpu::Device, texture_layout: &wgpu::BindGroupLayout, white: &wgpu::TextureView, sampler: &wgpu::Sampler) -> Self {
        let locals_layout = mesh_locals_layout(device);
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("egui_shader_layers coverage"), bind_group_layouts: &[Some(&locals_layout), Some(texture_layout)], immediate_size: 0 });
        let module = shader(device, "egui_shader_layers mesh", MESH_WGSL);
        let pipeline = mesh_pipeline(device, &layout, &module, COVERAGE_FORMAT, "fs_main", Some(COVERAGE_BLEND), "egui_shader_layers coverage");
        let region_locals_layout = region_locals_layout(device);
        let region_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("egui_shader_layers region ids"), bind_group_layouts: &[Some(&region_locals_layout), Some(texture_layout)], immediate_size: 0 });
        let region_pipeline = mesh_pipeline(device, &region_layout, &module, REGION_FORMAT, "fs_region", None, "egui_shader_layers region ids");
        let alignment = usize::try_from(device.limits().min_uniform_buffer_offset_alignment.max(1)).unwrap_or(256);
        let region_locals_stride = MESH_LOCALS_SIZE.div_ceil(alignment) * alignment;
        let white_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("egui_shader_layers white mesh texture"),
            layout: texture_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(white) }, wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(sampler) }],
        });
        Self { locals_layout, pipeline, region_locals_layout, region_pipeline, region_locals_stride, white_group }
    }

    /// The white texture group for untextured meshes (offline masks, `Mask::check_offline`).
    #[cfg_attr(not(any(feature = "image", feature = "video")), allow(dead_code))] // The UI binds egui's own texture groups.
    pub(crate) fn white_group(&self) -> &wgpu::BindGroup {
        &self.white_group
    }

    /// Records the coverage step `step` of a run on `rec`: grows `targets.texture` to at least
    /// `step.used_px` (the plan's window), clears it and draws `job`'s meshes into `(0, 0,
    /// step.used_px)`; with `step.regions > 0` (R1) then grows the region-id texture the same way,
    /// clears it to `REGION_NONE` and runs the id passes (every region's draw range, pass A then
    /// pass B). Returns the textures (bound at `Slot::Coverage` / `Slot::RegionId`), or `None`
    /// when `job` does not match the step (`job.textures` not one per draw, a region range outside
    /// the draws, a region count other than `step.regions` or above `MAX_REGIONS`).
    pub(crate) fn record<'t>(&self, rec: &mut ChainRecording<'_>, targets: &'t mut CoverageTargets, job: &CoverageJob<'_>, step: &CoverageStep) -> Option<CoverageOutput<'t>> {
        let draws = job.packed.draws.len();
        let region_count = usize::try_from(step.regions).ok()?;
        let ranges_fit = job.regions.iter().all(|range| range.start <= range.end && range.end <= draws);
        if job.textures.len() != draws || job.regions.len() != region_count || region_count > MAX_REGIONS || !ranges_fit {
            return None;
        }
        let (device, queue) = (rec.device, rec.queue);
        let needed = step.used_px.map(|side| u32::try_from(side).unwrap_or(0));
        let coverage = Target::ensure(&mut targets.texture, device, COVERAGE_FORMAT, None, "egui_shader_layers coverage", needed);
        let locals = targets.locals.write(device, queue, &coverage_locals(&job.transform, step));
        let locals = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("egui_shader_layers coverage locals"),
            layout: &self.locals_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: locals, offset: 0, size: std::num::NonZeroU64::new(u64::try_from(MESH_LOCALS_SIZE).unwrap_or(0)) }) }],
        });
        let vertices = targets.vertices.write(device, queue, &job.packed.vertices);
        let indices = targets.indices.write(device, queue, &job.packed.indices);
        {
            let mut pass = begin_pass(rec.encoder, &coverage.view, "egui_shader_layers coverage", wgpu::Color::TRANSPARENT);
            // Without draws the cleared (zero) coverage masks everything out.
            if !job.packed.draws.is_empty() {
                pass.set_viewport(0.0, 0.0, px_f32(step.used_px[0]), px_f32(step.used_px[1]), 0.0, 1.0);
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &locals, &[]);
                pass.set_vertex_buffer(0, vertices.slice(..));
                pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
                for (draw, &texture) in job.packed.draws.iter().zip(&job.textures) {
                    pass.set_bind_group(1, texture, &[]);
                    let end = draw.first_index.saturating_add(draw.index_count);
                    pass.draw_indexed(draw.first_index..end, 0, 0..1);
                }
            }
        }
        if region_count == 0 {
            return Some(CoverageOutput { coverage, region: None });
        }
        let region = Target::ensure(&mut targets.region, device, REGION_FORMAT, None, "egui_shader_layers region ids", needed);
        // One locals block per (pass, region), pass-major, at the device's offset alignment.
        let stride = self.region_locals_stride;
        let mut bytes = vec![0_u8; stride.checked_mul(REGION_PASS_THRESHOLDS.len() * region_count)?];
        for (pass_index, &threshold) in REGION_PASS_THRESHOLDS.iter().enumerate() {
            for (id, id_u32) in (0..region_count).zip(0_u32..) {
                let at = (pass_index * region_count + id) * stride;
                bytes[at..at + MESH_LOCALS_SIZE].copy_from_slice(&region_locals(&job.transform, step, id_u32, threshold));
            }
        }
        let region_locals = targets.region_locals.write(device, queue, &bytes);
        let region_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("egui_shader_layers region locals"),
            layout: &self.region_locals_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: region_locals, offset: 0, size: std::num::NonZeroU64::new(u64::try_from(MESH_LOCALS_SIZE).unwrap_or(0)) }) }],
        });
        let mut pass = begin_pass(rec.encoder, &region.view, "egui_shader_layers region ids", REGION_CLEAR);
        if !job.packed.draws.is_empty() {
            pass.set_viewport(0.0, 0.0, px_f32(step.used_px[0]), px_f32(step.used_px[1]), 0.0, 1.0);
            pass.set_pipeline(&self.region_pipeline);
            pass.set_vertex_buffer(0, vertices.slice(..));
            pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
            for pass_index in 0..REGION_PASS_THRESHOLDS.len() {
                for (id, range) in job.regions.iter().enumerate() {
                    if range.is_empty() {
                        continue;
                    }
                    // Below `bytes.len()`, which was allocated without overflow above.
                    let offset = u32::try_from((pass_index * region_count + id) * stride).ok()?;
                    pass.set_bind_group(0, &region_group, &[offset]);
                    for (draw, &texture) in job.packed.draws[range.clone()].iter().zip(&job.textures[range.clone()]) {
                        pass.set_bind_group(1, texture, &[]);
                        let end = draw.first_index.saturating_add(draw.index_count);
                        pass.draw_indexed(draw.first_index..end, 0, 0..1);
                    }
                }
            }
        }
        drop(pass);
        Some(CoverageOutput { coverage, region: Some(region) })
    }
}

/// The textures a recorded coverage step leaves for the run's draws.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CoverageOutput<'t> {
    /// The coverage (`Slot::Coverage`).
    pub coverage: &'t Target,
    /// The region ids (`Slot::RegionId`), for a region mask.
    pub region: Option<&'t Target>,
}

/// The coverage and region-id textures and the mesh buffers of ONE run (see `ChainTargets`).
#[derive(Debug)]
pub(crate) struct CoverageTargets {
    /// The coverage (`COVERAGE_FORMAT`), grow-only.
    texture: Option<Target>,
    /// The region ids (`REGION_FORMAT`), grow-only; allocated by the first region run.
    region: Option<Target>,
    /// The region-id passes' `MeshLocals` blocks.
    region_locals: GrowBuffer,
    /// `MeshLocals`.
    locals: GrowBuffer,
    /// Packed vertices.
    vertices: GrowBuffer,
    /// Packed indices.
    indices: GrowBuffer,
}

impl CoverageTargets {
    /// Empty; allocated on the first shape-masked run.
    pub(crate) fn new() -> Self {
        Self {
            texture: None,
            region: None,
            region_locals: GrowBuffer::new(wgpu::BufferUsages::UNIFORM, "egui_shader_layers region locals"),
            locals: GrowBuffer::new(wgpu::BufferUsages::UNIFORM, "egui_shader_layers coverage locals"),
            vertices: GrowBuffer::new(wgpu::BufferUsages::VERTEX, "egui_shader_layers coverage vertices"),
            indices: GrowBuffer::new(wgpu::BufferUsages::INDEX, "egui_shader_layers coverage indices"),
        }
    }
}

/// The mask of one run: its packed meshes, how their layer points map to capture pixels, the
/// texture group of each draw (`textures[i]` for `packed.draws[i]`, egui-wgpu's texture layout),
/// and for a region mask the draw range of every region (`mask::PackedRegions::regions`).
#[derive(Debug)]
pub(crate) struct CoverageJob<'a> {
    /// The packed meshes (layer points).
    pub packed: &'a PackedCoverage,
    /// Layer points -> capture pixels (`geometry::mask_transform`).
    pub transform: MaskTransform,
    /// One texture group per draw.
    pub textures: Vec<&'a wgpu::BindGroup>,
    /// `regions[id]` = region `id`'s range of `packed.draws`; empty for a plain shape mask.
    pub regions: &'a [Range<usize>],
}
