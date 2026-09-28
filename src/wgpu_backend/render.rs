/*
File: wgpu_backend/render.rs

Purpose:
The `egui_wgpu::CallbackTrait` of one filter layer: `prepare` re-renders the beneath content
offscreen and hands it to the chain executor (`chain.rs`) on egui's command encoder; `paint`
draws the chain's composite into egui's render pass.

Key structures:
- `FilterCallback`: one painted layer (its `LayerJob` + hand-over slot + install state).

Key functions:
- `FilterCallback::prepare()` / `finish_prepare()` / `paint()`.
- `mesh_locals()`, `scissor_in_capture()`: pure helpers (unit-tested).

Notes:
Order within `prepare`: every pipeline is resolved BEFORE anything is recorded, so a failing
effect skips the layer without leaving half-recorded work. Nothing here panics; a missing plan
(the end-of-pass plugin never saw the layer) or a mismatched generation simply draws nothing
(the missing plan is debug-logged once per install, `Shared::logged`).
The composite BLENDS over egui's pass (`chain::draw_composite_over`): pixels outside the rounded
mask keep what the pass holds, since the re-rendered backdrop lacks paint callbacks and lower
filter layers. On `*Srgb` targets the meshes are rendered through an sRGB view of the capture
(`WgpuResources::srgb_capture`) so they blend in linear like egui's own draws.
*/

use std::sync::Arc;

use egui::epaint::{ClippedPrimitive, Primitive};
use egui_wgpu::{CallbackResources, ScreenDescriptor};

use super::chain::{ChainJob, ChainRecording, ChainSource, ChainWindow, draw_composite_over};
use super::gpu::{Target, begin_pass};
use super::resources::{CAPTURE_FORMAT, CAPTURE_SRGB_VIEW_FORMAT, MESH_LOCALS_SIZE, WgpuResources};
use super::{BeneathPlan, Drawn, LayerSlot, LoggedOnce, Shared};
use crate::backend::LayerJob;
use crate::geometry::{LayerGeometry, PxRect, layer_geometry, px_f32, transform_scale};

/// One painted filter layer, as egui-wgpu sees it.
pub(super) struct FilterCallback {
    /// Install-wide state (options, health latches).
    shared: Arc<Shared>,
    /// Plan in, drawn target out.
    slot: Arc<LayerSlot>,
    /// What to run.
    job: LayerJob,
}

impl FilterCallback {
    /// A callback for `job`, handing over through `slot`.
    pub(super) fn new(shared: Arc<Shared>, slot: Arc<LayerSlot>, job: LayerJob) -> Self {
        Self { shared, slot, job }
    }
}

/// Serializes `MeshLocals` (see `mesh.wgsl`): pixels per point, dithering flag, capture origin
/// and size in pixels.
#[must_use]
pub(super) fn mesh_locals(pixels_per_point: f32, dithering: bool, capture: PxRect) -> [u8; MESH_LOCALS_SIZE] {
    let mut bytes = [0_u8; MESH_LOCALS_SIZE];
    let fields: [[u8; 4]; 6] = [
        pixels_per_point.to_ne_bytes(),
        u32::from(dithering).to_ne_bytes(),
        px_f32(capture.x).to_ne_bytes(),
        px_f32(capture.y).to_ne_bytes(),
        px_f32(capture.w).to_ne_bytes(),
        px_f32(capture.h).to_ne_bytes(),
    ];
    for (chunk, field) in bytes.chunks_exact_mut(4).zip(fields) {
        chunk.copy_from_slice(&field);
    }
    bytes
}

/// The scissor of a primitive clipped by `clip_rect` (points), in capture-texture pixels, or
/// `None` when it misses the capture. Rounds like egui-wgpu's `ScissorRect` so the re-rendered
/// meshes are clipped exactly as egui clips them on screen.
#[must_use]
pub(super) fn scissor_in_capture(clip_rect: egui::Rect, pixels_per_point: f32, screen: [u32; 2], capture: PxRect) -> Option<[u32; 4]> {
    let screen_w = i32::try_from(screen[0]).unwrap_or(i32::MAX);
    let screen_h = i32::try_from(screen[1]).unwrap_or(i32::MAX);
    let round = |value: f32| {
        // Saturating float -> int cast (NaN -> 0); clamped to the screen right after.
        #[allow(clippy::cast_possible_truncation)]
        let rounded = (value * pixels_per_point).round() as i32;
        rounded
    };
    let min_x = round(clip_rect.min.x).clamp(0, screen_w);
    let min_y = round(clip_rect.min.y).clamp(0, screen_h);
    let max_x = round(clip_rect.max.x).clamp(min_x, screen_w);
    let max_y = round(clip_rect.max.y).clamp(min_y, screen_h);
    let left = min_x.max(capture.x) - capture.x;
    let top = min_y.max(capture.y) - capture.y;
    let right = max_x.min(capture.x + capture.w) - capture.x;
    let bottom = max_y.min(capture.y + capture.h) - capture.y;
    if right <= left || bottom <= top {
        return None;
    }
    Some([u32::try_from(left).ok()?, u32::try_from(top).ok()?, u32::try_from(right - left).ok()?, u32::try_from(bottom - top).ok()?])
}

/// Byte ranges of one mesh inside the concatenated vertex / index uploads.
struct MeshRange {
    /// Index into the plan's primitives.
    primitive: usize,
    /// Scissor in capture pixels.
    scissor: [u32; 4],
    /// Byte range in the vertex buffer.
    vertices: std::ops::Range<u64>,
    /// Byte range in the index buffer.
    indices: std::ops::Range<u64>,
    /// Number of indices.
    index_count: u32,
}

/// Concatenates the drawable meshes (non-empty, texture known, scissor inside the capture) into
/// vertex and index byte arrays.
fn pack_meshes(plan: &BeneathPlan, pixels_per_point: f32, screen: [u32; 2], capture: PxRect) -> (Vec<u8>, Vec<u8>, Vec<MeshRange>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut ranges = Vec::new();
    for (index, ClippedPrimitive { clip_rect, primitive }) in plan.primitives.iter().enumerate() {
        let Primitive::Mesh(mesh) = primitive else {
            continue;
        };
        if mesh.indices.is_empty() || !plan.textures.contains_key(&mesh.texture_id) {
            continue;
        }
        let Ok(index_count) = u32::try_from(mesh.indices.len()) else {
            continue;
        };
        let Some(scissor) = scissor_in_capture(*clip_rect, pixels_per_point, screen, capture) else {
            continue;
        };
        let vertex_start = u64::try_from(vertices.len()).unwrap_or(u64::MAX);
        let index_start = u64::try_from(indices.len()).unwrap_or(u64::MAX);
        vertices.extend_from_slice(bytemuck::cast_slice(&mesh.vertices));
        indices.extend_from_slice(bytemuck::cast_slice(&mesh.indices));
        let vertex_end = u64::try_from(vertices.len()).unwrap_or(u64::MAX);
        let index_end = u64::try_from(indices.len()).unwrap_or(u64::MAX);
        ranges.push(MeshRange { primitive: index, scissor, vertices: vertex_start..vertex_end, indices: index_start..index_end, index_count });
    }
    (vertices, indices, ranges)
}

impl FilterCallback {
    /// Gets (creating on first use or after a re-install) this install's resources.
    fn resources<'a>(&self, device: &wgpu::Device, queue: &wgpu::Queue, resources: &'a mut CallbackResources) -> Option<&'a mut WgpuResources> {
        let generation = self.shared.generation;
        if resources.get::<WgpuResources>().is_none_or(|existing| existing.generation != generation) {
            let state = &self.shared.render_state;
            match WgpuResources::create(device, queue, &state.adapter, state.target_format, &self.shared.options, generation) {
                Ok(created) => {
                    log::info!(
                        "egui_filter_layers (wgpu): resources created (target {:?}, MSAA {}, intermediates {:?})",
                        state.target_format,
                        self.shared.options.msaa_samples(),
                        created.chain.intermediate_format()
                    );
                    self.shared.health().float_intermediates = Some(created.chain.float_intermediates());
                    resources.insert(created);
                }
                Err(error) => {
                    self.shared.latch("creating GPU resources", &error);
                    return None;
                }
            }
        }
        resources.get_mut::<WgpuResources>()
    }

    /// Resolves every pass's pipeline, or `None` (layer skipped) when one failed.
    fn pipelines(&self, device: &wgpu::Device, resources: &mut WgpuResources) -> Option<Vec<wgpu::RenderPipeline>> {
        let mut pipelines = Vec::with_capacity(self.job.passes.len());
        for pass in &self.job.passes {
            let module = pass.effect().module();
            if self.shared.effect_failed(module) {
                return None;
            }
            match resources.chain.effect_pipeline(device, module) {
                Ok(pipeline) => pipelines.push(pipeline),
                Err(error) => {
                    self.shared.fail_effect(module, &error);
                    return None;
                }
            }
        }
        Some(pipelines)
    }
}

/// Everything recording one layer needs.
struct Recording<'a> {
    /// The device.
    device: &'a wgpu::Device,
    /// The queue (uniform / mesh uploads).
    queue: &'a wgpu::Queue,
    /// egui's encoder.
    encoder: &'a mut wgpu::CommandEncoder,
    /// Pixel geometry of the layer.
    geometry: LayerGeometry,
    /// egui scale factor.
    pixels_per_point: f32,
    /// Screen size in pixels.
    screen: [u32; 2],
}

impl egui_wgpu::CallbackTrait for FilterCallback {
    fn prepare(&self, device: &wgpu::Device, queue: &wgpu::Queue, screen: &ScreenDescriptor, encoder: &mut wgpu::CommandEncoder, callback_resources: &mut CallbackResources) -> Vec<wgpu::CommandBuffer> {
        let plan = {
            let mut slot = self.slot.lock();
            slot.drawn = None;
            slot.plan.take()
        };
        let Some(plan) = plan else {
            if LoggedOnce::first(&self.shared.logged.missing_plan) {
                log::debug!(
                    "egui_filter_layers (wgpu): a filter layer reached the renderer without its backdrop plan (invisible, or painted \
                     outside an egui pass that ended with Context::end_pass); it draws nothing this frame. Logged once."
                );
            }
            return Vec::new();
        };
        if self.shared.is_failed() || self.job.passes.is_empty() {
            return Vec::new();
        }
        let Some(resources) = self.resources(device, queue, callback_resources) else {
            return Vec::new();
        };
        resources.chain.evict_dropped();
        let Some(pipelines) = self.pipelines(device, resources) else {
            return Vec::new();
        };
        let Some(geometry) = layer_geometry(plan.callback_rect, plan.clip_rect, screen.pixels_per_point, screen.size_in_pixels, self.job.padding_points, self.job.corner_radius, plan.transform_scale) else {
            return Vec::new();
        };
        let mut recording = Recording { device, queue, encoder, geometry, pixels_per_point: screen.pixels_per_point, screen: screen.size_in_pixels };
        let pool_index = resources.next_targets();
        self.record(&mut recording, resources, pool_index, &plan, &pipelines);
        self.slot.lock().drawn = Some(Drawn { generation: resources.generation, pool_index, region: geometry.region });
        Vec::new()
    }

    fn finish_prepare(&self, _device: &wgpu::Device, _queue: &wgpu::Queue, _encoder: &mut wgpu::CommandEncoder, callback_resources: &mut CallbackResources) -> Vec<wgpu::CommandBuffer> {
        // Every layer of this frame has been prepared: the next frame may reuse the pool.
        if let Some(resources) = callback_resources.get_mut::<WgpuResources>() {
            resources.cursor = 0;
        }
        Vec::new()
    }

    fn paint(&self, info: egui::PaintCallbackInfo, render_pass: &mut wgpu::RenderPass<'static>, callback_resources: &CallbackResources) {
        let Some(drawn) = self.slot.lock().drawn else {
            return;
        };
        let Some(resources) = callback_resources.get::<WgpuResources>() else {
            return;
        };
        if resources.generation != drawn.generation {
            return;
        }
        let Some(bind_group) = resources.pool.get(drawn.pool_index).and_then(|targets| targets.composite.as_ref()) else {
            return;
        };
        // egui-wgpu reports the same rect / clip the plugin planned with; a mismatch would mean
        // the composite lands somewhere else than what was captured, so it is skipped instead.
        let transform_scale = transform_scale(self.job.rect, info.viewport);
        let region = layer_geometry(info.viewport, info.clip_rect, info.pixels_per_point, info.screen_size_px, self.job.padding_points, self.job.corner_radius, transform_scale).map(|geometry| geometry.region);
        if region != Some(drawn.region) {
            log::debug!("egui_filter_layers (wgpu): paint region {region:?} differs from the prepared {:?}; layer skipped this frame", drawn.region);
            return;
        }
        draw_composite_over(render_pass, &resources.composite, bind_group, drawn.region);
    }
}

impl FilterCallback {
    /// Records the backdrop re-render, then the chain (every filter pass) on the capture, and
    /// stores the composite bind group `paint` uses in the layer's pool entry.
    fn record(&self, rec: &mut Recording<'_>, resources: &mut WgpuResources, pool_index: usize, plan: &BeneathPlan, pipelines: &[wgpu::RenderPipeline]) {
        let capture = rec.geometry.capture;
        let size = capture.size().map(|extent| u32::try_from(extent).unwrap_or(1));
        let Some(targets) = resources.pool.get_mut(pool_index) else {
            return;
        };
        let render_view_format = resources.srgb_capture.then_some(CAPTURE_SRGB_VIEW_FORMAT);
        let capture_target = Target::ensure(&mut targets.capture, rec.device, CAPTURE_FORMAT, render_view_format, "egui_filter_layers capture", size);

        let locals_buffer = targets.mesh_locals.write(rec.device, rec.queue, &mesh_locals(rec.pixels_per_point, self.shared.options.dithering(), capture));
        let locals = rec.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("egui_filter_layers mesh locals"),
            layout: &resources.mesh_locals_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: locals_buffer, offset: 0, size: std::num::NonZeroU64::new(u64::try_from(MESH_LOCALS_SIZE).unwrap_or(0)) }),
            }],
        });
        let (vertex_bytes, index_bytes, ranges) = pack_meshes(plan, rec.pixels_per_point, rec.screen, capture);
        let buffers = MeshBuffers { vertices: targets.vertices.write(rec.device, rec.queue, &vertex_bytes).clone(), indices: targets.indices.write(rec.device, rec.queue, &index_bytes).clone(), ranges };
        self.render_backdrop(rec, &resources.mesh_pipeline, &locals, capture_target, plan, &buffers);

        let mut chain_rec = ChainRecording { device: rec.device, queue: rec.queue, encoder: rec.encoder };
        let job = ChainJob {
            source: ChainSource { view: &capture_target.view, allocated: capture_target.size },
            geometry: rec.geometry,
            // The whole capture; its pixel (0, 0) is at the capture's screen position.
            window: ChainWindow::whole(&rec.geometry, [capture.x, capture.y]),
            time: self.job.time,
            // The pointer relative to this capture, at the callback's pixels per point.
            input: self.job.frame_input.resolve(rec.pixels_per_point, rec.geometry.capture),
            passes: &self.job.passes,
            pipelines,
        };
        targets.composite = resources.chain.record(&mut chain_rec, &mut targets.chain, &job);
    }

    /// Re-renders the planned meshes into `capture` (through its render view: the sRGB view when
    /// `WgpuResources::srgb_capture`), cleared first to what egui's pass shows where nothing was
    /// painted (`capture_clear_color`).
    fn render_backdrop(&self, rec: &mut Recording<'_>, pipeline: &wgpu::RenderPipeline, locals: &wgpu::BindGroup, capture: &Target, plan: &BeneathPlan, buffers: &MeshBuffers) {
        // Through the sRGB view wgpu encodes the clear value itself, exactly as for egui's pass.
        let through_srgb_view = capture.render_view_format().is_some_and(|format| format.is_srgb());
        let encode = self.shared.render_state.target_format.is_srgb() && !through_srgb_view;
        let clear = capture_clear_color(plan.clear_color, encode);
        let mut pass = begin_pass(rec.encoder, capture.render_view(), "egui_filter_layers backdrop", clear);
        pass.set_viewport(0.0, 0.0, px_f32(rec.geometry.capture.w), px_f32(rec.geometry.capture.h), 0.0, 1.0);
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, locals, &[]);
        for range in &buffers.ranges {
            let Some(Primitive::Mesh(mesh)) = plan.primitives.get(range.primitive).map(|primitive| &primitive.primitive) else {
                continue;
            };
            let Some(texture) = plan.textures.get(&mesh.texture_id) else {
                continue;
            };
            let [x, y, width, height] = range.scissor;
            pass.set_scissor_rect(x, y, width, height);
            pass.set_bind_group(1, texture, &[]);
            pass.set_vertex_buffer(0, buffers.vertices.slice(range.vertices.clone()));
            pass.set_index_buffer(buffers.indices.slice(range.indices.clone()), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..range.index_count, 0, 0..1);
        }
    }
}

/// The clear value of the capture that reproduces how egui's pass clears a target.
///
/// eframe hands its premultiplied gamma-space clear colour to `LoadOp::Clear` unchanged. On a
/// `*Srgb` target wgpu takes clear values as linear and stores their sRGB encoding, so the screen
/// shows `gamma_from_linear(rgb)`. When the capture is cleared through its sRGB view the hardware
/// does the same (`encode = false`); when it is cleared through its unorm view on an sRGB target
/// (the no-view-formats fallback), the value is encoded here (`encode = true`). Alpha is never
/// encoded.
#[must_use]
fn capture_clear_color(clear_color: [f32; 4], encode: bool) -> wgpu::Color {
    let [r, g, b, a] = clear_color;
    let encode = |channel: f32| if encode { egui::ecolor::gamma_from_linear(channel) } else { channel };
    wgpu::Color { r: f64::from(encode(r)), g: f64::from(encode(g)), b: f64::from(encode(b)), a: f64::from(a) }
}

/// The uploaded beneath meshes of one layer.
struct MeshBuffers {
    /// Vertex buffer holding every mesh back to back.
    vertices: wgpu::Buffer,
    /// Index buffer holding every mesh back to back (indices relative to their own mesh).
    indices: wgpu::Buffer,
    /// Where each drawable mesh lives in the two buffers.
    ranges: Vec<MeshRange>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::{Rect, pos2};

    const CAPTURE: PxRect = PxRect { x: 100, y: 50, w: 200, h: 100 };

    #[test]
    fn scissors_are_rounded_like_egui_and_moved_into_capture_space() {
        let clip = Rect::from_min_max(pos2(60.2, 20.0), pos2(120.0, 40.4));
        // ppp 2: x 120..240, y 40..81 -> capture-local x 20..140, y 0..31.
        assert_eq!(scissor_in_capture(clip, 2.0, [1000, 1000], CAPTURE), Some([20, 0, 120, 31]));
        assert_eq!(scissor_in_capture(Rect::EVERYTHING, 1.0, [1000, 1000], CAPTURE), Some([0, 0, 200, 100]));
    }

    #[test]
    fn scissors_outside_the_capture_or_screen_are_rejected() {
        let outside = Rect::from_min_max(pos2(0.0, 0.0), pos2(90.0, 40.0));
        assert_eq!(scissor_in_capture(outside, 1.0, [1000, 1000], CAPTURE), None);
        let beyond_screen = Rect::from_min_max(pos2(500.0, 500.0), pos2(600.0, 600.0));
        assert_eq!(scissor_in_capture(beyond_screen, 1.0, [400, 400], CAPTURE), None);
    }

    #[test]
    fn the_fallback_clears_the_capture_to_the_encoded_colour() {
        let clear = [0.2, 0.5, 0.0, 0.7];
        let unorm = capture_clear_color(clear, false);
        assert!((unorm.r - 0.2).abs() < 1e-6 && (unorm.g - 0.5).abs() < 1e-6 && (unorm.a - 0.7).abs() < 1e-6);
        let srgb = capture_clear_color(clear, true);
        // sRGB encoding of linear 0.2 / 0.5; alpha untouched.
        assert!((srgb.r - 0.484_529).abs() < 1e-4 && (srgb.g - 0.735_357).abs() < 1e-4, "{srgb:?}");
        assert!(srgb.b.abs() < 1e-6 && (srgb.a - 0.7).abs() < 1e-6);
    }

    #[test]
    fn mesh_locals_follow_the_wgsl_layout() {
        let bytes = mesh_locals(1.5, true, CAPTURE);
        let read = |offset: usize| f32::from_ne_bytes([bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]]);
        assert!((read(0) - 1.5).abs() < f32::EPSILON);
        assert_eq!(u32::from_ne_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]), 1);
        assert!((read(8) - 100.0).abs() < f32::EPSILON && (read(12) - 50.0).abs() < f32::EPSILON);
        assert!((read(16) - 200.0).abs() < f32::EPSILON && (read(20) - 100.0).abs() < f32::EPSILON);
    }
}
