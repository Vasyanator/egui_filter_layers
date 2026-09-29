/*
File: wgpu_backend/render.rs

Purpose:
The `egui_wgpu::CallbackTrait` of one shader layer: `prepare` re-renders the beneath content
offscreen and hands it to the chain executor (`chain.rs`) on egui's command encoder; `paint`
draws the chain's composite into egui's render pass.

Key structures:
- `FilterCallback`: one painted layer (its `LayerJob` + hand-over slot + install state).
- `EntryRun`, `BackdropDraw`, `MeshBuffers`: one entry's run inputs, one re-render's inputs, the
  layer's beneath meshes uploaded once.

Key functions:
- `FilterCallback::prepare()` / `finish_prepare()` / `paint()`.
- `mesh_locals()`, `scissor_in_capture()`: pure helpers (unit-tested).

Notes:
Order within `prepare`: every entry's geometry, state format and pipelines are resolved BEFORE
anything is recorded, so a failing effect skips the layer without leaving half-recorded work.
Then the ops of `plan::schedule` (pristine source) run in order (R3): the beneath meshes are
packed and uploaded once (into the first visible entry's pool slot); `Capture` re-renders them
into that entry's own capture (scissored per mesh), `Run` records its chain (one pool slot per
entry), `HandOff` draws an earlier entry's replace composite into a later capture (`Stacked`; mixing in
linear on `*Srgb` targets, `WgpuResources::linear_handoff`, like the screen composite),
`Composite` is kept for `paint`, which draws them in that order. Nothing here panics; a missing plan
(the end-of-pass plugin never saw the layer) or a mismatched generation simply draws nothing
(the missing plan is debug-logged once per install, `Shared::logged`).
Stateful entries (F7): an entry's state (`WgpuResources::states`, keyed by viewport and the
entry's key `LayerJob::state_key`, outside the per-frame pool) is looked up at its `Run` and
updated after it (the layer's reset ticket is claimed once per paint); a state format
the device cannot render / filter skips the layer (`Shared::fail_state`). Unpainted states are
evicted at every prepare (`StateRegistry::evict`).
A shape mask (F9) or region mask (R1) is packed here and handed to the chain executor with its
layer-points -> capture-pixels transform (`geometry::mask_transform`), the texture groups the
plugin resolved and the regions' draw ranges; the executor rasterizes it into the run's coverage
(and region-id) texture.
The composite BLENDS over egui's pass (`chain::draw_composite_over`): pixels outside the
mask keep what the pass holds, since the re-rendered backdrop lacks paint callbacks and lower
shader layers. On `*Srgb` targets the meshes are rendered through an sRGB view of the capture
(`WgpuResources::srgb_capture`) so they blend in linear like egui's own draws.
*/

use std::sync::Arc;

use egui::epaint::{ClippedPrimitive, Primitive};
use egui_wgpu::{CallbackResources, ScreenDescriptor};

use super::coverage::CoverageJob;
use super::chain::{ChainJob, ChainRecording, ChainSource, ChainState, ChainWindow, HandoffSpace, RecordedRun, draw_composite_over};
use super::gpu::{Target, begin_pass};
use super::resources::{CAPTURE_FORMAT, CAPTURE_SRGB_VIEW_FORMAT, WgpuResources};
use super::{BeneathPlan, Drawn, DrawnEntry, LayerSlot, LoggedOnce, Shared};
use crate::backend::{EntryJob, LayerJob};
use crate::geometry::{LayerGeometry, PxRect, entry_rect, layer_geometry, mask_transform, px_f32, transform_scale};
use crate::plan::{CaptureSource, MESH_LOCALS_SIZE, Op, StatePlanInput, mesh_locals, schedule};

/// One painted shader layer, as egui-wgpu sees it.
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

/// The backdrop re-render's `MeshLocals` (`shaders/mesh.wgsl`, packed by `plan::mesh_locals`):
/// screen points -> capture pixels (`pos * pixels_per_point - capture.xy`) over the capture.
#[must_use]
pub(super) fn backdrop_locals(pixels_per_point: f32, dithering: bool, capture: PxRect) -> [u8; MESH_LOCALS_SIZE] {
    mesh_locals(pixels_per_point, dithering, [px_f32(capture.x), px_f32(capture.y)], [px_f32(capture.w), px_f32(capture.h)], 0.0, 0.0)
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
    /// Index into the plan's primitives (its clip rect gives each entry's scissor).
    primitive: usize,
    /// Byte range in the vertex buffer.
    vertices: std::ops::Range<u64>,
    /// Byte range in the index buffer.
    indices: std::ops::Range<u64>,
    /// Number of indices.
    index_count: u32,
}

/// Concatenates the drawable meshes (non-empty, texture known) into vertex and index byte
/// arrays, once for every entry of the layer; each entry's re-render skips the meshes whose
/// scissor misses its capture.
fn pack_meshes(plan: &BeneathPlan) -> (Vec<u8>, Vec<u8>, Vec<MeshRange>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut ranges = Vec::new();
    for (index, ClippedPrimitive { primitive, .. }) in plan.primitives.iter().enumerate() {
        let Primitive::Mesh(mesh) = primitive else {
            continue;
        };
        if mesh.indices.is_empty() || !plan.textures.contains_key(&mesh.texture_id) {
            continue;
        }
        let Ok(index_count) = u32::try_from(mesh.indices.len()) else {
            continue;
        };
        let vertex_start = u64::try_from(vertices.len()).unwrap_or(u64::MAX);
        let index_start = u64::try_from(indices.len()).unwrap_or(u64::MAX);
        vertices.extend_from_slice(bytemuck::cast_slice(&mesh.vertices));
        indices.extend_from_slice(bytemuck::cast_slice(&mesh.indices));
        let vertex_end = u64::try_from(vertices.len()).unwrap_or(u64::MAX);
        let index_end = u64::try_from(indices.len()).unwrap_or(u64::MAX);
        ranges.push(MeshRange { primitive: index, vertices: vertex_start..vertex_end, indices: index_start..index_end, index_count });
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
                        "egui_shader_layers (wgpu): resources created (target {:?}, MSAA {}, intermediates {:?})",
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

    /// Resolves every pass's pipeline of `entry` (state passes in its state format), or `None`
    /// (layer skipped) when one failed.
    fn pipelines(&self, device: &wgpu::Device, resources: &mut WgpuResources, entry: &EntryJob) -> Option<Vec<wgpu::RenderPipeline>> {
        if entry.passes.iter().any(|pass| self.shared.effect_failed(pass.effect().module())) {
            return None;
        }
        match resources.chain.pass_pipelines(device, &entry.passes, entry.state.format()) {
            Ok(pipelines) => Some(pipelines),
            Err((module, error)) => {
                self.shared.fail_effect(&module, &error);
                None
            }
        }
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
    /// egui scale factor.
    pixels_per_point: f32,
    /// Screen size in pixels.
    screen: [u32; 2],
}

/// What recording one entry needs besides the layer's `Recording` and resources.
struct EntryRun<'a> {
    /// Index of the entry in the job.
    index: usize,
    /// Its `WgpuResources::pool` slot.
    pool_index: usize,
    /// Its geometry (screen pixels).
    geometry: LayerGeometry,
    /// Its passes' pipelines.
    pipelines: &'a [wgpu::RenderPipeline],
    /// Where its composite is handed off (`plan::Op::HandOff::origin`, in op order).
    handoff_origins: &'a [[i32; 2]],
    /// The layer's claimed `reset_state` request (one claim per paint, every stateful entry).
    reset: bool,
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
                    "egui_shader_layers (wgpu): a shader layer reached the renderer without its backdrop plan (invisible, an attached \
                     texture that cannot be resolved (see failed_textures), or painted outside an egui pass that ended with \
                     Context::end_pass); it draws nothing this frame. Logged once."
                );
            }
            return Vec::new();
        };
        if self.shared.is_failed() || self.job.entries.is_empty() {
            return Vec::new();
        }
        let Some(resources) = self.resources(device, queue, callback_resources) else {
            return Vec::new();
        };
        resources.chain.evict_dropped();
        // Unpainted states are freed here (every prepare checks), before this layer's own lookup.
        drop(resources.states.evict(self.job.viewport, self.job.frame_input.frame, f64::from(self.job.time)));
        self.shared.state_count.store(resources.states.len(), std::sync::atomic::Ordering::Relaxed);
        // Every entry's geometry in screen pixels (`None`: invisible). A single entry's rect is
        // the callback rect, so its geometry is what a single-entry layer always had.
        let geometries: Vec<Option<LayerGeometry>> = self
            .job
            .entries
            .iter()
            .map(|entry| layer_geometry(entry_rect(self.job.rect, plan.callback_rect, entry.rect), plan.clip_rect, screen.pixels_per_point, screen.size_in_pixels, entry.padding_points, entry.mask.corner_radius(), plan.transform_scale))
            .collect();
        // State formats and pipelines of every visible entry BEFORE anything is recorded: one
        // failing entry skips the whole layer (one layer, one decision) without half-recorded work.
        let mut pipelines = Vec::with_capacity(self.job.entries.len());
        for (index, (entry, geometry)) in self.job.entries.iter().zip(&geometries).enumerate() {
            if geometry.is_none() {
                pipelines.push(None);
                continue;
            }
            if self.job.state_key(index).is_some()
                && let Some(problem) = resources.chain.state_format_problem(entry.state.format())
            {
                self.shared.fail_state(entry.state.format(), problem);
                return Vec::new();
            }
            let Some(entry_pipelines) = self.pipelines(device, resources, entry) else {
                return Vec::new();
            };
            pipelines.push(Some(entry_pipelines));
        }
        let ops = schedule(&geometries, self.job.overlap, CaptureSource::Pristine);
        if ops.is_empty() {
            return Vec::new();
        }
        let mut recording = Recording { device, queue, encoder, pixels_per_point: screen.pixels_per_point, screen: screen.size_in_pixels };
        let drawn = self.record(&mut recording, resources, &plan, &geometries, &pipelines, &ops);
        self.shared.state_count.store(resources.states.len(), std::sync::atomic::Ordering::Relaxed);
        self.slot.lock().drawn = drawn.map(|entries| Drawn { generation: resources.generation, entries });
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
        let Some(drawn) = self.slot.lock().drawn.clone() else {
            return;
        };
        let Some(resources) = callback_resources.get::<WgpuResources>() else {
            return;
        };
        if resources.generation != drawn.generation {
            return;
        }
        // egui-wgpu reports the same rect / clip the plugin planned with; a mismatch would mean
        // a composite lands somewhere else than what was captured, so the layer is skipped instead.
        let transform_scale = transform_scale(self.job.rect, info.viewport);
        let mut composites = Vec::with_capacity(drawn.entries.len());
        for drawn_entry in &drawn.entries {
            let Some(entry) = self.job.entries.get(drawn_entry.entry) else {
                return;
            };
            let region = layer_geometry(entry_rect(self.job.rect, info.viewport, entry.rect), info.clip_rect, info.pixels_per_point, info.screen_size_px, entry.padding_points, entry.mask.corner_radius(), transform_scale).map(|geometry| geometry.region);
            if region != Some(drawn_entry.region) {
                log::debug!("egui_shader_layers (wgpu): paint region {region:?} differs from the prepared {:?}; layer skipped this frame", drawn_entry.region);
                return;
            }
            let Some(bind_group) = resources.pool.get(drawn_entry.pool_index).and_then(|targets| targets.composite.as_ref()) else {
                return;
            };
            composites.push((bind_group, drawn_entry.region));
        }
        // In the schedule's (list) order: a later entry lands over an earlier one.
        for (bind_group, region) in composites {
            draw_composite_over(render_pass, &resources.composite, bind_group, region);
        }
    }
}

impl FilterCallback {
    /// Executes `ops` (`plan::schedule` with a pristine source): the beneath meshes are packed and
    /// uploaded once, `Capture` re-renders them into an entry's capture, `Run` records its chain
    /// (with the composite bind groups of its `Stacked` hand-offs), `HandOff` draws one into a
    /// later entry's capture, `Composite` is left for `paint` (returned, in order). `None`: a run
    /// could not be recorded, the layer draws nothing this frame.
    fn record(&self, rec: &mut Recording<'_>, resources: &mut WgpuResources, plan: &BeneathPlan, geometries: &[Option<LayerGeometry>], pipelines: &[Option<Vec<wgpu::RenderPipeline>>], ops: &[Op]) -> Option<Vec<DrawnEntry>> {
        // One pool slot per visible entry (all prepares precede egui's pass, see `resources.rs`).
        let pool: Vec<Option<usize>> = geometries.iter().map(|geometry| geometry.map(|_| resources.next_targets())).collect();
        let first = pool.iter().flatten().copied().next()?;
        let (vertex_bytes, index_bytes, ranges) = pack_meshes(plan);
        let targets = resources.pool.get_mut(first)?;
        let buffers = MeshBuffers { vertices: targets.vertices.write(rec.device, rec.queue, &vertex_bytes).clone(), indices: targets.indices.write(rec.device, rec.queue, &index_bytes).clone(), ranges };
        // The layer's reset request is claimed once per paint, for every stateful entry.
        let stateful = geometries.iter().enumerate().any(|(index, geometry)| geometry.is_some() && self.job.state_key(index).is_some());
        let reset = stateful && self.job.claim_reset();
        let mut handoffs: Vec<std::vec::IntoIter<wgpu::BindGroup>> = geometries.iter().map(|_| Vec::new().into_iter()).collect();
        let mut drawn = Vec::new();
        for op in ops {
            match *op {
                Op::Capture(index) => self.capture(rec, resources, (*pool.get(index)?)?, geometries.get(index).copied().flatten()?.capture, plan, &buffers)?,
                Op::Run(index) => {
                    let origins: Vec<[i32; 2]> = ops.iter().filter_map(|op| if let Op::HandOff { from, origin, .. } = *op { (from == index).then_some(origin) } else { None }).collect();
                    let run = EntryRun { index, pool_index: (*pool.get(index)?)?, geometry: geometries.get(index).copied().flatten()?, pipelines: pipelines.get(index)?.as_deref()?, handoff_origins: &origins, reset };
                    let recorded = self.run_entry(rec, resources, plan, &run)?;
                    *handoffs.get_mut(index)? = recorded.handoffs.into_iter();
                    resources.pool.get_mut(run.pool_index)?.composite = Some(recorded.composite);
                }
                Op::HandOff { from, into, viewport, .. } => {
                    let group = handoffs.get_mut(from)?.next()?;
                    let target = resources.pool.get((*pool.get(into)?)?)?.capture.as_ref()?;
                    // In the space `paint`'s screen composite blends in (linear on `*Srgb`).
                    let space = if resources.linear_handoff { HandoffSpace::Linear } else { HandoffSpace::Gamma };
                    resources.chain.record_handoff(rec.encoder, &group, &target.view, viewport, space);
                }
                Op::Composite(index) => drawn.push(DrawnEntry { entry: index, pool_index: (*pool.get(index)?)?, region: geometries.get(index).copied().flatten()?.region }),
            }
        }
        Some(drawn)
    }

    /// `Op::Capture`: re-renders the beneath meshes into the capture of pool slot `pool_index`
    /// (`capture`, screen pixels), each mesh scissored like egui clips it on screen.
    fn capture(&self, rec: &mut Recording<'_>, resources: &mut WgpuResources, pool_index: usize, capture: PxRect, plan: &BeneathPlan, buffers: &MeshBuffers) -> Option<()> {
        let size = capture.size().map(|extent| u32::try_from(extent).unwrap_or(1));
        let render_view_format = resources.srgb_capture.then_some(CAPTURE_SRGB_VIEW_FORMAT);
        let targets = resources.pool.get_mut(pool_index)?;
        let capture_target = Target::ensure(&mut targets.capture, rec.device, CAPTURE_FORMAT, render_view_format, "egui_shader_layers capture", size);
        let locals_buffer = targets.mesh_locals.write(rec.device, rec.queue, &backdrop_locals(rec.pixels_per_point, self.shared.options.dithering(), capture));
        let locals = rec.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("egui_shader_layers mesh locals"),
            layout: &resources.mesh_locals_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: locals_buffer, offset: 0, size: std::num::NonZeroU64::new(u64::try_from(MESH_LOCALS_SIZE).unwrap_or(0)) }),
            }],
        });
        self.render_backdrop(rec, &BackdropDraw { pipeline: &resources.mesh_pipeline, locals: &locals, target: capture_target, capture }, plan, buffers);
        Some(())
    }

    /// `Op::Run`: records entry `run.index`'s chain on its capture (a stateful entry over its state
    /// entry, updated with where the latest state now is) and returns the recorded run.
    fn run_entry(&self, rec: &mut Recording<'_>, resources: &mut WgpuResources, plan: &BeneathPlan, run: &EntryRun<'_>) -> Option<RecordedRun> {
        let entry = self.job.entries.get(run.index)?;
        let capture = run.geometry.capture;
        let user_textures = plan.user_textures.get(run.index)?;
        // A shape mask (F9): its meshes in layer points, mapped through the layer transform the
        // callback rect reveals; every texture they sample was resolved by the plugin.
        // A region mask (R1) packs every region into the same upload plus its draw ranges.
        let packed = entry.mask.pack();
        let coverage = match &packed {
            Some(packed) => {
                let textures: Option<Vec<&wgpu::BindGroup>> = packed.packed.draws.iter().map(|draw| plan.textures.get(&draw.texture)).collect();
                Some(CoverageJob { packed: &packed.packed, transform: mask_transform(self.job.rect, plan.callback_rect, rec.pixels_per_point, capture), textures: textures?, regions: &packed.regions })
            }
            None => None,
        };
        let (frame, time) = (self.job.frame_input.frame, f64::from(self.job.time));
        let format = entry.state.format();
        // The entry's replaced targets (a format change) are simply dropped (wgpu frees them
        // once no recorded command uses them).
        let mut state = self.job.state_key(run.index).map(|key| resources.states.begin(key, format, frame, time, run.reset, || [None, None]));
        let state_input = state.as_ref().map(|(_, state_run, _)| StatePlanInput { scale: entry.state.scale(), current: state_run.current, previous_px: state_run.previous, step: state_run.step });
        let targets = resources.pool.get_mut(run.pool_index)?;
        let (Some(capture_target), chain_targets) = (targets.capture.as_ref(), &mut targets.chain) else {
            return None;
        };
        let job = ChainJob {
            source: ChainSource { view: &capture_target.view, allocated: capture_target.size },
            geometry: run.geometry,
            // The whole capture; its pixel (0, 0) is at the capture's screen position.
            window: ChainWindow::whole(&run.geometry, [capture.x, capture.y]),
            time: self.job.time,
            // The pointer relative to this capture, at the callback's pixels per point.
            input: self.job.frame_input.resolve(rec.pixels_per_point, capture),
            passes: &entry.passes,
            pipelines: run.pipelines,
            user_textures: user_textures.each_ref().map(Option::as_ref),
            state: state_input,
            coverage,
            handoff_origins: run.handoff_origins,
        };
        let mut chain_rec = ChainRecording { device: rec.device, queue: rec.queue, encoder: rec.encoder };
        let recorded = {
            let mut chain_state = state.as_mut().map(|(state_entry, _, _)| ChainState { targets: &mut state_entry.targets, format });
            resources.chain.record(&mut chain_rec, chain_targets, &job, chain_state.as_mut())
        };
        if let Some((state_entry, state_run, _)) = state {
            match recorded.as_ref().and_then(|recorded| recorded.state.as_ref()) {
                Some(state_plan) => state_entry.finish(state_run, frame, state_plan.final_index, state_plan.final_px),
                // Nothing (or not everything) was recorded: start over from zero next time.
                None => state_entry.invalidate(),
            }
        }
        recorded
    }

    /// Re-renders the planned meshes into `draw.target` (through its render view: the sRGB view
    /// when `WgpuResources::srgb_capture`), cleared first to what egui's pass shows where nothing
    /// was painted (`capture_clear_color`).
    fn render_backdrop(&self, rec: &mut Recording<'_>, draw: &BackdropDraw<'_>, plan: &BeneathPlan, buffers: &MeshBuffers) {
        let (pipeline, locals, capture, capture_px) = (draw.pipeline, draw.locals, draw.target, draw.capture);
        // Through the sRGB view wgpu encodes the clear value itself, exactly as for egui's pass.
        let through_srgb_view = capture.render_view_format().is_some_and(|format| format.is_srgb());
        let encode = self.shared.render_state.target_format.is_srgb() && !through_srgb_view;
        let clear = capture_clear_color(plan.clear_color, encode);
        let mut pass = begin_pass(rec.encoder, capture.render_view(), "egui_shader_layers backdrop", clear);
        pass.set_viewport(0.0, 0.0, px_f32(capture_px.w), px_f32(capture_px.h), 0.0, 1.0);
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, locals, &[]);
        for range in &buffers.ranges {
            let Some(ClippedPrimitive { clip_rect, primitive: Primitive::Mesh(mesh) }) = plan.primitives.get(range.primitive) else {
                continue;
            };
            let Some(texture) = plan.textures.get(&mesh.texture_id) else {
                continue;
            };
            // Meshes outside this entry's capture (another entry's) are skipped.
            let Some([x, y, width, height]) = scissor_in_capture(*clip_rect, rec.pixels_per_point, rec.screen, capture_px) else {
                continue;
            };
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

/// One entry's backdrop re-render: the mesh pipeline, its locals and where it draws.
struct BackdropDraw<'a> {
    /// The mesh pipeline (`WgpuResources::mesh_pipeline`).
    pipeline: &'a wgpu::RenderPipeline,
    /// The `MeshLocals` bind group of this capture (`backdrop_locals`).
    locals: &'a wgpu::BindGroup,
    /// The entry's capture texture.
    target: &'a Target,
    /// The capture's rect in screen pixels (viewport size and scissors).
    capture: PxRect,
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
        let bytes = backdrop_locals(1.5, true, CAPTURE);
        let read = |offset: usize| f32::from_ne_bytes([bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]]);
        assert!((read(0) - 1.5).abs() < f32::EPSILON);
        assert_eq!(u32::from_ne_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]), 1);
        assert!((read(8) - 100.0).abs() < f32::EPSILON && (read(12) - 50.0).abs() < f32::EPSILON);
        assert!((read(16) - 200.0).abs() < f32::EPSILON && (read(20) - 100.0).abs() < f32::EPSILON);
    }
}
