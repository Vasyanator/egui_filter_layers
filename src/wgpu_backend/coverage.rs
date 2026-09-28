/*
File: wgpu_backend/coverage.rs

Purpose:
The wgpu mesh machinery shared by the backdrop re-render (UI) and the shape-mask coverage
rasterizer (F9, UI and offline): the mesh pipeline over `shaders/mesh.wgsl` (egui's vertex
layout, egui-wgpu's premultiplied blend, egui-wgpu's texture group layout), and the coverage
step of a chain run — the mask's packed meshes (`mask::PackedCoverage`) drawn into an RGBA8
texture whose alpha the composite reads. egui-free like `chain.rs` (only bytes and wgpu objects
cross this boundary), so the offline renderer runs it unchanged.

Key structures:
- `CoverageRasterizer`: per-device objects (locals layout, the coverage pipeline, a white texture
  group for untextured meshes).
- `CoverageTargets`: per-run coverage texture and buffers (grow-only).
- `CoverageJob`: one run's packed meshes, their transform and a texture bind group per draw.

Key functions:
- `mesh_pipeline()`: a pipeline drawing egui meshes into a format (backdrop and coverage).
- `CoverageRasterizer::record()`: the coverage step of a plan.

Notes:
Coverage blends like egui (`One, OneMinusSrcAlpha` / `OneMinusDstAlpha, One`) into a target cleared
to transparent, so overlapping meshes accumulate alpha as egui would paint them. No MSAA: the
tessellator's feathering is the anti-aliasing. The target holds the plan's WINDOW at its origin.
*/

use super::chain::ChainRecording;
use super::gpu::{GrowBuffer, Target, begin_pass, shader, uniform_entry};
use crate::geometry::{MaskTransform, px_f32};
use crate::mask::{PackedCoverage, VERTEX_BYTES};
use crate::plan::{CoverageStep, MESH_LOCALS_SIZE, coverage_locals};

/// Format of the coverage texture: RGBA8 renders and filters everywhere (WebGL2 included);
/// the coverage is its alpha.
pub(crate) const COVERAGE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// egui-wgpu's premultiplied-alpha blend state (egui-wgpu 0.36 `renderer.rs`, `Renderer::new`).
pub(crate) const EGUI_BLEND: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha, operation: wgpu::BlendOperation::Add },
    alpha: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::OneMinusDstAlpha, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add },
};

pub(crate) use crate::mask::MESH_WGSL;

/// A pipeline drawing egui meshes (`mesh.wgsl`'s `vs_main` + `fragment_entry`) with egui's
/// blend into `format`, single-sampled. `layout` = the locals group then egui's texture group.
pub(crate) fn mesh_pipeline(device: &wgpu::Device, layout: &wgpu::PipelineLayout, module: &wgpu::ShaderModule, format: wgpu::TextureFormat, fragment_entry: &str, label: &str) -> wgpu::RenderPipeline {
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
            targets: &[Some(wgpu::ColorTargetState { format, blend: Some(EGUI_BLEND), write_mask: wgpu::ColorWrites::ALL })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

/// The layout of the mesh shader's group 0 (`MeshLocals`).
pub(crate) fn mesh_locals_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("egui_filter_layers mesh locals"), entries: &[uniform_entry(0, Some(MESH_LOCALS_SIZE))] })
}

/// Per-device objects of the coverage rasterizer.
#[derive(Debug)]
pub(crate) struct CoverageRasterizer {
    /// Group 0 layout (`MeshLocals`).
    locals_layout: wgpu::BindGroupLayout,
    /// `mesh.wgsl` `vs_main` + `fs_main` into [`COVERAGE_FORMAT`] with egui's blend.
    pipeline: wgpu::RenderPipeline,
    /// A 1x1 white texture in egui's texture layout: the texture of untextured meshes when the
    /// caller has no egui texture to bind (offline).
    white_group: wgpu::BindGroup,
}

impl CoverageRasterizer {
    /// Builds the pipeline over `texture_layout` (egui-wgpu's texture layout) and the white group
    /// from `white` + `sampler`. Must run inside the caller's error scope.
    pub(crate) fn create(device: &wgpu::Device, texture_layout: &wgpu::BindGroupLayout, white: &wgpu::TextureView, sampler: &wgpu::Sampler) -> Self {
        let locals_layout = mesh_locals_layout(device);
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("egui_filter_layers coverage"), bind_group_layouts: &[Some(&locals_layout), Some(texture_layout)], immediate_size: 0 });
        let module = shader(device, "egui_filter_layers mesh", MESH_WGSL);
        let pipeline = mesh_pipeline(device, &layout, &module, COVERAGE_FORMAT, "fs_main", "egui_filter_layers coverage");
        let white_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("egui_filter_layers white mesh texture"),
            layout: texture_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(white) }, wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(sampler) }],
        });
        Self { locals_layout, pipeline, white_group }
    }

    /// The white texture group for untextured meshes (offline masks, `Mask::check_offline`).
    #[cfg_attr(not(any(feature = "image", feature = "video")), allow(dead_code))] // The UI binds egui's own texture groups.
    pub(crate) fn white_group(&self) -> &wgpu::BindGroup {
        &self.white_group
    }

    /// Records the coverage step `step` of a run on `rec`: grows `targets.texture` to at least
    /// `step.used_px` (the plan's window), clears it and draws `job`'s meshes into `(0, 0,
    /// step.used_px)`. Returns the target (bound at `Slot::Coverage`), or `None` when
    /// `job.textures` does not match `job.packed.draws`.
    pub(crate) fn record<'t>(&self, rec: &mut ChainRecording<'_>, targets: &'t mut CoverageTargets, job: &CoverageJob<'_>, step: &CoverageStep) -> Option<&'t Target> {
        if job.textures.len() != job.packed.draws.len() {
            return None;
        }
        let (device, queue) = (rec.device, rec.queue);
        let needed = step.used_px.map(|side| u32::try_from(side).unwrap_or(0));
        let target = Target::ensure(&mut targets.texture, device, COVERAGE_FORMAT, None, "egui_filter_layers coverage", needed);
        let locals = targets.locals.write(device, queue, &coverage_locals(&job.transform, step));
        let locals = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("egui_filter_layers coverage locals"),
            layout: &self.locals_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: locals, offset: 0, size: std::num::NonZeroU64::new(u64::try_from(MESH_LOCALS_SIZE).unwrap_or(0)) }) }],
        });
        let vertices = targets.vertices.write(device, queue, &job.packed.vertices);
        let indices = targets.indices.write(device, queue, &job.packed.indices);
        let mut pass = begin_pass(rec.encoder, &target.view, "egui_filter_layers coverage", wgpu::Color::TRANSPARENT);
        if job.packed.draws.is_empty() {
            // Nothing to draw: the cleared (zero) coverage masks everything out.
            return Some(target);
        }
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
        drop(pass);
        Some(target)
    }
}

/// The coverage texture and the mesh buffers of ONE run (see `ChainTargets`).
#[derive(Debug)]
pub(crate) struct CoverageTargets {
    /// The coverage (`COVERAGE_FORMAT`), grow-only.
    texture: Option<Target>,
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
            locals: GrowBuffer::new(wgpu::BufferUsages::UNIFORM, "egui_filter_layers coverage locals"),
            vertices: GrowBuffer::new(wgpu::BufferUsages::VERTEX, "egui_filter_layers coverage vertices"),
            indices: GrowBuffer::new(wgpu::BufferUsages::INDEX, "egui_filter_layers coverage indices"),
        }
    }
}

/// The mask of one run: its packed meshes, how their layer points map to capture pixels, and the
/// texture group of each draw (`textures[i]` for `packed.draws[i]`, egui-wgpu's texture layout).
#[derive(Debug)]
pub(crate) struct CoverageJob<'a> {
    /// The packed meshes (layer points).
    pub packed: &'a PackedCoverage,
    /// Layer points -> capture pixels (`geometry::mask_transform`).
    pub transform: MaskTransform,
    /// One texture group per draw.
    pub textures: Vec<&'a wgpu::BindGroup>,
}
