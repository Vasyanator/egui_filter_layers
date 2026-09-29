/*
File: offline/layer.rs

Purpose:
Offline multi-entry layers (R3): `OfflineEntry` / `OfflineLayer`, the offline counterparts of
`Entry` / `ShaderLayer::from_entries`, and `OfflineRenderer::process_layer*`, which computes what
such a layer shows over a frame in the UI.

Key structures:
- `OfflineEntry` (public): rect in frame points + chain + mask + `OfflineTexture`s + state options.
- `OfflineLayer` (public): the entries and their `Overlap` mode.
- `LayerResources`: the renderer's objects for layer runs (frame source / output textures, per
  entry a capture, chain targets, texture uploads and persistent state), kept across calls.

Key functions:
- `OfflineRenderer::process_layer()` / `process_layer_into()` (native, blocking) /
  `process_layer_async()`; `run_layer()` the shared implementation.

Notes:
The frame is the screen: every entry's geometry is `layer_geometry(entry rect, the frame)`, as the
UI computes it for a layer whose entries lie on a screen the size of the frame. The order of
captures, runs, hand-offs and composites is `plan::schedule`'s (pristine source, like the wgpu
UI): every capture is a copy of the entry's capture rect from the uploaded (unfiltered) frame, a
`Stacked` hand-off is `ChainResources::record_handoff` (the code the UI uses), and every entry is
composited over the output (a copy of the frame) with the UI's BLENDED composite, so UI == offline
bit for bit in both modes. Layer runs are never tiled (roadmap D6): a frame above the tile limit
is `OfflineError::NotTileable`. Per-entry states are kept by entry index; `reset_state` clears them.
*/

use egui::{CornerRadius, Rect};

use super::frame::{convert_output, frame_bytes};
use super::readback::{RowLayout, map_read};
use super::{FRAME_FORMAT, FrameRef, OfflineError, OfflineOptions, OfflineRenderer, OfflineState, OfflineTexture, RgbaFrame, UploadedTexture, copy_rows, frame_rect, offline_gpu_error, px_u32};
use crate::backend::TextureError;
use crate::effect::TEXTURE_SLOTS;
use crate::entry::Overlap;
use crate::geometry::{LayerGeometry, PxRect, layer_geometry, mask_transform};
use crate::mask::{Mask, PackedRegions, tessellate, tessellate_regions};
use crate::pass::FilterChain;
use crate::plan::{CaptureSource, Op, StatePlan, StatePlanInput, check_textures, schedule};
use crate::state::{StateError, StateOptions};
use crate::wgpu_backend::chain::{ChainJob, ChainRecording, ChainSource, ChainState, ChainTargets, ChainWindow, draw_composite_over};
use crate::wgpu_backend::coverage::CoverageJob;
use crate::wgpu_backend::gpu::{ErrorScopes, begin_pass_load};

/// One filtered area of an [`OfflineLayer`]: what an [`Entry`](crate::Entry) is in the UI, with
/// the rect in FRAME points (frame pixels / `OfflineOptions::with_pixels_per_point`) and
/// [`OfflineTexture`]s instead of egui texture ids.
#[derive(Clone, Debug)]
pub struct OfflineEntry {
    /// The filtered rect, in frame points.
    rect: Rect,
    /// The passes to run.
    chain: FilterChain,
    /// The mask (rounded rect by default).
    mask: Mask,
    /// The texture attached at each user texture slot.
    textures: [Option<OfflineTexture>; TEXTURE_SLOTS],
    /// The first out-of-range slot given to [`OfflineEntry::texture`].
    invalid_slot: Option<usize>,
    /// How the persistent state is stored (stateful chains only).
    state: StateOptions,
}

impl OfflineEntry {
    /// An entry filtering `rect` (frame points) through `chain`, with the plain rect as mask.
    #[must_use]
    pub fn new(rect: Rect, chain: impl Into<FilterChain>) -> Self {
        Self { rect, chain: chain.into(), mask: Mask::default(), textures: [const { None }; TEXTURE_SLOTS], invalid_slot: None, state: StateOptions::default() }
    }

    /// Restricts the entry to `mask` (frame points; offline-checked: no text, untextured meshes).
    #[must_use]
    pub fn mask(mut self, mask: Mask) -> Self {
        self.mask = mask;
        self
    }

    /// Restricts the entry to a rounded rect: shorthand for `mask(Mask::RoundedRect(..))`.
    #[must_use]
    pub fn corner_radius(self, corner_radius: impl Into<CornerRadius>) -> Self {
        self.mask(Mask::RoundedRect(corner_radius.into()))
    }

    /// Attaches `texture` at user texture `slot` (`0..TEXTURE_SLOTS`) of this entry's effects. An
    /// out-of-range slot makes `process_layer` fail with [`OfflineError::Texture`].
    #[must_use]
    pub fn texture(mut self, slot: usize, texture: OfflineTexture) -> Self {
        match self.textures.get_mut(slot) {
            Some(entry) => *entry = Some(texture),
            None => {
                self.invalid_slot.get_or_insert(slot);
            }
        }
        self
    }

    /// How this entry's persistent state is stored (stateful chains only).
    #[must_use]
    pub fn state(mut self, options: StateOptions) -> Self {
        self.state = options;
        self
    }

    /// The filtered rect, in frame points.
    #[must_use]
    pub fn rect(&self) -> Rect {
        self.rect
    }

    /// The chain this entry runs.
    #[must_use]
    pub fn chain(&self) -> &FilterChain {
        &self.chain
    }

    /// The entry's mask.
    #[must_use]
    pub fn layer_mask(&self) -> &Mask {
        &self.mask
    }

    /// The texture attached at `slot`, if any.
    #[must_use]
    pub fn texture_at(&self, slot: usize) -> Option<&OfflineTexture> {
        self.textures.get(slot).and_then(Option::as_ref)
    }

    /// The state options.
    #[must_use]
    pub fn state_options(&self) -> StateOptions {
        self.state
    }

    /// Whether the entry draws anything (a non-empty chain over a positive rect).
    fn is_drawable(&self) -> bool {
        !self.chain.is_empty() && self.rect.is_positive()
    }

    /// Whether the entry has a state pass.
    fn is_stateful(&self) -> bool {
        self.chain.passes().iter().any(|pass| pass.effect().is_state())
    }

    /// The checks of entry `index` before any GPU work: texture slots, the textures the chain
    /// reads, the mask (offline rules).
    fn check(&self, index: usize) -> Result<(), OfflineError> {
        if let Some(slot) = self.invalid_slot {
            return Err(OfflineError::Texture(TextureError::InvalidSlot { entry: index, slot }));
        }
        check_textures(self.chain.passes(), self.textures.each_ref().map(Option::is_some), index)?;
        self.mask.check_offline()?;
        Ok(())
    }
}

/// A multi-entry layer for [`OfflineRenderer::process_layer`]: what
/// [`ShaderLayer::from_entries`](crate::ShaderLayer::from_entries) paints in the UI.
#[derive(Clone, Debug, Default)]
pub struct OfflineLayer {
    /// The entries, in paint order.
    entries: Vec<OfflineEntry>,
    /// How overlapping entries see each other.
    overlap: Overlap,
}

impl OfflineLayer {
    /// A layer of `entries`, painted in this order, [`Overlap::Independent`].
    #[must_use]
    pub fn new(entries: impl IntoIterator<Item = OfflineEntry>) -> Self {
        Self { entries: entries.into_iter().collect(), overlap: Overlap::default() }
    }

    /// Appends `entry` (painted after the existing ones).
    #[must_use]
    pub fn with_entry(mut self, entry: OfflineEntry) -> Self {
        self.entries.push(entry);
        self
    }

    /// How overlapping entries see each other.
    #[must_use]
    pub fn overlap(mut self, overlap: Overlap) -> Self {
        self.overlap = overlap;
        self
    }

    /// The overlap mode.
    #[must_use]
    pub fn overlap_mode(&self) -> Overlap {
        self.overlap
    }

    /// The entries, in paint order.
    #[must_use]
    pub fn entries(&self) -> &[OfflineEntry] {
        &self.entries
    }
}

/// The renderer's objects for layer runs, kept across calls (a stream of frames reuses them).
#[derive(Debug, Default)]
pub(super) struct LayerResources {
    /// The frame's source and output textures (exactly the frame size).
    frame: Option<FrameTextures>,
    /// Per entry index.
    entries: Vec<EntryResources>,
}

impl LayerResources {
    /// Starts every entry's persistent state over (their textures are kept).
    pub(super) fn reset_states(&mut self) {
        for state in self.entries.iter_mut().filter_map(|entry| entry.state.as_mut()) {
            state.used = None;
        }
    }

    /// Drops everything that may be invalid after a GPU error (textures, states).
    fn drop_after_error(&mut self) {
        *self = Self::default();
    }
}

/// The source and output textures of a layer run.
#[derive(Debug)]
struct FrameTextures {
    /// Allocated size (the frame).
    alloc: [u32; 2],
    /// The prepared frame (captures are copied out of it).
    source: wgpu::Texture,
    /// The output: a copy of the frame, the entries composited over it.
    output: wgpu::Texture,
    /// Render view of `output`.
    output_view: wgpu::TextureView,
}

/// One entry's objects.
#[derive(Debug)]
struct EntryResources {
    /// Its capture (exactly the capture size, like a freshly allocated UI capture) and view.
    capture: Option<([u32; 2], wgpu::Texture, wgpu::TextureView)>,
    /// Its chain targets (ping-pong, uniforms, coverage).
    chain: ChainTargets,
    /// Its user texture uploads.
    textures: [Option<UploadedTexture>; TEXTURE_SLOTS],
    /// Its persistent state (stateful chains).
    state: Option<OfflineState>,
    /// The composite bind group of its last recorded run (drawn by the `Composite` op).
    composite: Option<wgpu::BindGroup>,
}

impl EntryResources {
    /// Nothing allocated yet.
    fn new() -> Self {
        Self { capture: None, chain: ChainTargets::new(), textures: [const { None }; TEXTURE_SLOTS], state: None, composite: None }
    }
}

/// One visible entry of a run: its geometry and pipelines.
struct PreparedEntry {
    /// Pixel geometry (the frame is the screen).
    geometry: LayerGeometry,
    /// Its passes' pipelines.
    pipelines: Vec<wgpu::RenderPipeline>,
    /// A shape / region mask, packed, with its transform.
    coverage: Option<(PackedRegions, crate::geometry::MaskTransform)>,
    /// The state before this run (stateful chains).
    state: Option<StatePlanInput>,
}

/// What recording a layer run leaves: the readback layout, the future of the run's first GPU
/// error (await it before mapping) and each entry's state plan (by entry index).
type RecordedLayer<F> = (RowLayout, F, Vec<Option<StatePlan>>);

impl OfflineRenderer {
    /// Filters `frame` through the multi-entry `layer` and returns the result (blocking until the
    /// GPU is done): exactly what [`ShaderLayer::from_entries`](crate::ShaderLayer::from_entries)
    /// shows over that frame in the UI (conditions as for [`OfflineRenderer::process`]), in both
    /// [`Overlap`] modes. The options give the frame-wide settings (pixels per point, time,
    /// per-frame input, background, output alpha, tile limit); masks, textures and state options
    /// are per [`OfflineEntry`] — setting them on the options is refused. Every stateful entry
    /// keeps its own state across calls ([`OfflineRenderer::reset_state`] clears them all).
    ///
    /// # Errors
    /// As [`OfflineRenderer::process`]; [`OfflineError::NotTileable`] when the frame exceeds the
    /// tile limit (layer runs are never tiled: raise it with `with_max_tile_size`);
    /// [`OfflineError::InvalidOption`] for a mask, texture or state set on the options.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn process_layer(&mut self, frame: &FrameRef<'_>, layer: &OfflineLayer, options: &OfflineOptions) -> Result<RgbaFrame, OfflineError> {
        let mut pixels = Vec::new();
        self.process_layer_into(frame, layer, options, &mut pixels)?;
        Ok(RgbaFrame::from_parts(frame.width(), frame.height(), options.output_alpha, pixels))
    }

    /// [`OfflineRenderer::process_layer`] into a reusable buffer.
    ///
    /// # Errors
    /// As [`OfflineRenderer::process_layer`].
    #[cfg(not(target_arch = "wasm32"))]
    pub fn process_layer_into(&mut self, frame: &FrameRef<'_>, layer: &OfflineLayer, options: &OfflineOptions, output: &mut Vec<u8>) -> Result<(), OfflineError> {
        super::readback::block_on(self.run_layer(frame, layer, options, output, true))
    }

    /// Async [`OfflineRenderer::process_layer`] (does not poll the device, like
    /// [`OfflineRenderer::process_async`]).
    ///
    /// # Errors
    /// As [`OfflineRenderer::process_layer`].
    pub async fn process_layer_async(&mut self, frame: &FrameRef<'_>, layer: &OfflineLayer, options: &OfflineOptions) -> Result<RgbaFrame, OfflineError> {
        let mut pixels = Vec::new();
        self.run_layer(frame, layer, options, &mut pixels, false).await?;
        Ok(RgbaFrame::from_parts(frame.width(), frame.height(), options.output_alpha, pixels))
    }

    /// The shared implementation of the layer entry points (the scratch buffer is moved out for
    /// the run, like [`OfflineRenderer::run`]).
    async fn run_layer(&mut self, frame: &FrameRef<'_>, layer: &OfflineLayer, options: &OfflineOptions, output: &mut Vec<u8>, block: bool) -> Result<(), OfflineError> {
        let mut pixels = std::mem::take(&mut self.scratch);
        let result = self.run_layer_prepared(frame, layer, options, output, block, &mut pixels).await;
        self.scratch = pixels;
        result
    }

    /// [`OfflineRenderer::run_layer`] with the prepared-input buffer passed in: checks, CPU
    /// preparation, per-entry geometry / pipelines / textures / states, allocations, one recorded
    /// submission of the whole schedule, readback.
    async fn run_layer_prepared(&mut self, frame: &FrameRef<'_>, layer: &OfflineLayer, options: &OfflineOptions, output: &mut Vec<u8>, block: bool, pixels: &mut Vec<u8>) -> Result<(), OfflineError> {
        options.validate()?;
        if let Some(option) = options.set_entry_option() {
            return Err(OfflineError::InvalidOption { option, reason: "process_layer takes masks, textures and state options per OfflineEntry; set it there".to_owned() });
        }
        for (index, entry) in layer.entries.iter().enumerate() {
            entry.check(index)?;
        }
        frame.premultiplied_rows_into(options.background, pixels)?;
        let pixels: &[u8] = pixels;
        let size = [frame.width(), frame.height()];
        output.clear();
        if !layer.entries.iter().any(OfflineEntry::is_drawable) {
            output.extend_from_slice(pixels);
            convert_output(output, options.output_alpha);
            return Ok(());
        }
        output.resize(frame_bytes(size[0], size[1])?, 0);
        let device_limit = self.max_texture_side.min(self.max_readback_side);
        let limit = options.max_tile_size.map_or(device_limit, |cap| cap.min(device_limit));
        if size[0] > limit || size[1] > limit {
            return Err(OfflineError::NotTileable { reason: format!("a layer of entries (process_layer) needs the whole frame; the frame is {}x{} and tiles are at most {limit} pixels", size[0], size[1]) });
        }
        let prepared = self.prepare_entries(layer, options, size)?;
        self.ensure_layer_resources(layer, &prepared, size).await?;
        let (layout, errors, state_plans) = self.record_layer(layer, &prepared, options, pixels, size)?;
        if let Some(error) = errors.await {
            self.layer.drop_after_error();
            self.readback = None;
            return Err(offline_gpu_error(&error));
        }
        for (entry, state_plan) in self.layer.entries.iter_mut().zip(state_plans) {
            if let (Some(state), Some(state_plan)) = (entry.state.as_mut(), state_plan) {
                state.current = state_plan.final_index;
                state.used = Some(state_plan.final_px);
            }
        }
        let Some((buffer, buffer_size)) = self.readback.take() else {
            return Err(OfflineError::Gpu("the readback buffer is missing (crate bug)".to_owned()));
        };
        let slice = buffer.slice(..layout.buffer_size());
        let mapped = map_read(&slice);
        if block {
            self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).map_err(|error| OfflineError::Readback(error.to_string()))?;
        }
        mapped.await?;
        let whole = PxRect { x: 0, y: 0, w: i32::try_from(size[0]).unwrap_or(i32::MAX), h: i32::try_from(size[1]).unwrap_or(i32::MAX) };
        let copied = copy_rows(&slice, layout, whole, size[0], output);
        buffer.unmap();
        copied?;
        self.readback = Some((buffer, buffer_size));
        convert_output(output, options.output_alpha);
        Ok(())
    }

    /// Per entry: its geometry on a screen the size of the frame (`None`: not drawn), its state
    /// format check, pipelines, packed mask and state input.
    ///
    /// # Errors
    /// [`OfflineError::State`] for an unsupported state format; [`OfflineError::Effect`] for a
    /// pipeline the device rejects; [`OfflineError::SizeOverflow`] for an unaddressable frame.
    fn prepare_entries(&mut self, layer: &OfflineLayer, options: &OfflineOptions, size: [u32; 2]) -> Result<Vec<Option<PreparedEntry>>, OfflineError> {
        let ppp = options.pixels_per_point;
        let frame_points = frame_rect(size, ppp)?;
        self.chain.evict_dropped();
        let mut prepared = Vec::with_capacity(layer.entries.len());
        for (index, entry) in layer.entries.iter().enumerate() {
            let corner_radius = match &entry.mask {
                Mask::RoundedRect(radius) => *radius,
                Mask::Shape(_) | Mask::Regions(_) => CornerRadius::ZERO,
            };
            let geometry = if entry.is_drawable() { layer_geometry(entry.rect, Rect::EVERYTHING, ppp, size, entry.chain.padding_points(), corner_radius, 1.0) } else { None };
            let Some(geometry) = geometry else {
                prepared.push(None);
                continue;
            };
            let format = entry.state.format();
            if entry.is_stateful()
                && let Some(reason) = self.chain.state_format_problem(format)
            {
                return Err(OfflineError::State(StateError::UnsupportedFormat { format, reason }));
            }
            let pipelines = self.chain.pass_pipelines(&self.device, entry.chain.passes(), format).map_err(|(module, message)| OfflineError::Effect { label: module.label.to_string(), message })?;
            // Tessellated like `process` (frame points, default options); the frame is the layer
            // rect as painted and as rendered, so the transform maps points to frame pixels.
            let transform = mask_transform(frame_points, frame_points, ppp, geometry.capture);
            let coverage = match &entry.mask {
                Mask::RoundedRect(_) => None,
                Mask::Shape(shape) => Some((PackedRegions { packed: tessellate(shape.clone(), entry.rect, ppp, egui::epaint::TessellationOptions::default(), [1, 1]).pack(), regions: Vec::new() }, transform)),
                Mask::Regions(regions) => Some((tessellate_regions(regions.clone(), entry.rect, ppp, egui::epaint::TessellationOptions::default(), [1, 1]).pack(), transform)),
            };
            let state = entry.is_stateful().then(|| {
                let current = self.layer.entries.get(index).and_then(|resources| resources.state.as_ref()).filter(|state| state.format == format);
                StatePlanInput { scale: entry.state.scale(), current: current.map_or(0, |state| state.current), previous_px: current.and_then(|state| state.used), step: true }
            });
            prepared.push(Some(PreparedEntry { geometry, pipelines, coverage, state }));
        }
        Ok(prepared)
    }

    /// Makes the per-entry resources match `layer` (one per entry index), each entry's texture
    /// uploads and state objects match its entry, and allocates the frame textures, the captures
    /// and the readback buffer, inside awaited [`ErrorScopes`].
    ///
    /// # Errors
    /// As [`OfflineRenderer::ensure_user_textures`]; [`OfflineError::OutOfMemory`] /
    /// [`OfflineError::Gpu`] for the allocations; [`OfflineError::SizeOverflow`].
    async fn ensure_layer_resources(&mut self, layer: &OfflineLayer, prepared: &[Option<PreparedEntry>], size: [u32; 2]) -> Result<(), OfflineError> {
        self.layer.entries.truncate(layer.entries.len());
        while self.layer.entries.len() < layer.entries.len() {
            self.layer.entries.push(EntryResources::new());
        }
        for (index, entry) in layer.entries.iter().enumerate() {
            let mut cache = std::mem::replace(&mut self.layer.entries[index].textures, [const { None }; TEXTURE_SLOTS]);
            let result = self.sync_textures(&mut cache, &entry.textures).await;
            self.layer.entries[index].textures = cache;
            result?;
            // A stateful entry keeps its state while its format stays; a new one starts at zero.
            let resources = &mut self.layer.entries[index];
            let format = entry.state.format();
            if entry.is_stateful() && resources.state.as_ref().is_none_or(|state| state.format != format) {
                resources.state = Some(OfflineState { targets: [None, None], current: 0, used: None, format });
            }
        }
        let needed = RowLayout::new(size[0], size[1])?.buffer_size();
        let scopes = ErrorScopes::push(&self.device);
        let texture = |label: &str, size: [u32; 2], usage: wgpu::TextureUsages| {
            self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: FRAME_FORMAT,
                usage,
                view_formats: &[],
            })
        };
        if self.layer.frame.as_ref().is_none_or(|frame| frame.alloc != size) {
            self.layer.frame = None;
            let source = texture("egui_shader_layers offline layer source", size, wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST);
            let output = texture("egui_shader_layers offline layer output", size, wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST);
            let output_view = output.create_view(&wgpu::TextureViewDescriptor::default());
            self.layer.frame = Some(FrameTextures { alloc: size, source, output, output_view });
        }
        // Captures exactly at their size (a changed size recreates one), like a fresh UI capture.
        for (resources, prepared) in self.layer.entries.iter_mut().zip(prepared) {
            let Some(prepared) = prepared else {
                continue;
            };
            let capture = [px_u32(prepared.geometry.capture.w)?, px_u32(prepared.geometry.capture.h)?];
            if resources.capture.as_ref().is_none_or(|(alloc, _, _)| *alloc != capture) {
                resources.capture = None;
                let created = texture("egui_shader_layers offline entry capture", capture, wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::RENDER_ATTACHMENT);
                let view = created.create_view(&wgpu::TextureViewDescriptor::default());
                resources.capture = Some((capture, created, view));
            }
        }
        if self.readback.as_ref().is_none_or(|(_, size)| *size != needed) {
            self.readback = None;
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("egui_shader_layers offline readback"), size: needed, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            self.readback = Some((buffer, needed));
        }
        if let Some(error) = scopes.pop().await {
            self.layer.drop_after_error();
            self.readback = None;
            log::warn!("egui_shader_layers (offline): allocating a {}x{} layer run failed: {error}", size[0], size[1]);
            return Err(offline_gpu_error(&error));
        }
        Ok(())
    }

    /// Records the whole schedule on one encoder and submits it, inside [`ErrorScopes`]: the frame
    /// upload, the output as a copy of it, then `plan::schedule`'s ops — `Capture` copies an
    /// entry's capture rect of the frame, `Run` records its chain (with its hand-off composites),
    /// `HandOff` draws one into a later capture, `Composite` blends the entry over the output —
    /// and the output's readback copy. Returns the readback layout, the future of the first GPU
    /// error and each entry's state plan.
    fn record_layer(&mut self, layer: &OfflineLayer, prepared: &[Option<PreparedEntry>], options: &OfflineOptions, pixels: &[u8], size: [u32; 2]) -> Result<RecordedLayer<impl Future<Output = Option<wgpu::Error>> + use<>>, OfflineError> {
        let layout = RowLayout::new(size[0], size[1])?;
        let geometries: Vec<Option<LayerGeometry>> = prepared.iter().map(|entry| entry.as_ref().map(|entry| entry.geometry)).collect();
        let ops = schedule(&geometries, layer.overlap, CaptureSource::Pristine);
        let (device, queue, chain) = (&self.device, &self.queue, &self.chain);
        let (Some(frame), Some((readback, _))) = (self.layer.frame.as_ref(), self.readback.as_ref()) else {
            return Err(OfflineError::Gpu("offline layer targets are missing (crate bug)".to_owned()));
        };
        let entries = &mut self.layer.entries;
        let extent = wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 };
        let mut state_plans: Vec<Option<StatePlan>> = vec![None; entries.len()];
        let mut handoffs: Vec<std::vec::IntoIter<wgpu::BindGroup>> = entries.iter().map(|_| Vec::new().into_iter()).collect();
        let crate_bug = |what: &str| OfflineError::Gpu(format!("{what} (crate bug)"));

        let scopes = ErrorScopes::push(device);
        let recorded: Result<(), OfflineError> = (|| {
            queue.write_texture(frame.source.as_image_copy(), pixels, wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(size[0].saturating_mul(4)), rows_per_image: Some(size[1]) }, extent);
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("egui_shader_layers offline layer") });
            // The output starts as the frame: entries are composited over it like over the screen.
            encoder.copy_texture_to_texture(frame.source.as_image_copy(), frame.output.as_image_copy(), extent);
            for op in &ops {
                match *op {
                    Op::Capture(index) => {
                        let (Some(Some(prepared)), Some(resources)) = (prepared.get(index), entries.get(index)) else {
                            return Err(crate_bug("a scheduled entry was not prepared"));
                        };
                        let Some((capture_size, capture, _)) = resources.capture.as_ref() else {
                            return Err(crate_bug("an entry capture is missing"));
                        };
                        let rect = prepared.geometry.capture;
                        let origin = wgpu::Origin3d { x: px_u32(rect.x)?, y: px_u32(rect.y)?, z: 0 };
                        encoder.copy_texture_to_texture(
                            wgpu::TexelCopyTextureInfo { texture: &frame.source, mip_level: 0, origin, aspect: wgpu::TextureAspect::All },
                            capture.as_image_copy(),
                            wgpu::Extent3d { width: capture_size[0], height: capture_size[1], depth_or_array_layers: 1 },
                        );
                    }
                    Op::Run(index) => {
                        let (Some(Some(prepared)), Some(resources), Some(entry)) = (prepared.get(index), entries.get_mut(index), layer.entries.get(index)) else {
                            return Err(crate_bug("a scheduled entry was not prepared"));
                        };
                        let origins: Vec<[i32; 2]> = ops.iter().filter_map(|op| if let Op::HandOff { from, origin, .. } = *op { (from == index).then_some(origin) } else { None }).collect();
                        let EntryResources { capture, chain: targets, textures, state, composite } = resources;
                        let Some((capture_size, _, capture_view)) = capture.as_ref() else {
                            return Err(crate_bug("an entry capture is missing"));
                        };
                        let geometry = prepared.geometry;
                        let job = ChainJob {
                            source: ChainSource { view: capture_view, allocated: *capture_size },
                            geometry,
                            // The whole capture; its pixel (0, 0) lies at the capture's frame position.
                            window: ChainWindow::whole(&geometry, [geometry.capture.x, geometry.capture.y]),
                            time: options.time,
                            input: options.input_uniforms_in(size, geometry.capture),
                            passes: entry.chain.passes(),
                            pipelines: &prepared.pipelines,
                            user_textures: textures.each_ref().map(|uploaded| uploaded.as_ref().map(|uploaded| &uploaded.bind_group)),
                            state: prepared.state,
                            // Untextured meshes only (`Mask::check_offline`): the white group.
                            coverage: prepared.coverage.as_ref().map(|(packed, transform)| CoverageJob { packed: &packed.packed, transform: *transform, textures: vec![chain.coverage().white_group(); packed.packed.draws.len()], regions: &packed.regions }),
                            handoff_origins: &origins,
                        };
                        let mut chain_state = state.as_mut().filter(|_| prepared.state.is_some()).map(|state| ChainState { targets: &mut state.targets, format: state.format });
                        let Some(run) = chain.record(&mut ChainRecording { device, queue, encoder: &mut encoder }, targets, &job, chain_state.as_mut()) else {
                            return Err(crate_bug("the chain executor rejected an entry"));
                        };
                        if let Some(slot) = state_plans.get_mut(index) {
                            *slot = run.state;
                        }
                        if let Some(slot) = handoffs.get_mut(index) {
                            *slot = run.handoffs.into_iter();
                        }
                        // Kept until the entry's `Composite` op.
                        *composite = Some(run.composite);
                    }
                    Op::HandOff { from, into, viewport, .. } => {
                        let group = handoffs.get_mut(from).and_then(Iterator::next).ok_or_else(|| crate_bug("a hand-off has no recorded composite"))?;
                        let Some((_, _, view)) = entries.get(into).and_then(|resources| resources.capture.as_ref()) else {
                            return Err(crate_bug("an entry capture is missing"));
                        };
                        chain.record_handoff(&mut encoder, &group, view, viewport);
                    }
                    Op::Composite(index) => {
                        let (Some(Some(prepared)), Some(group)) = (prepared.get(index), entries.get(index).and_then(|resources| resources.composite.as_ref())) else {
                            return Err(crate_bug("a composite has no recorded run"));
                        };
                        let mut pass = begin_pass_load(&mut encoder, &frame.output_view, "egui_shader_layers offline entry composite");
                        draw_composite_over(&mut pass, &self.composite_over, group, prepared.geometry.region);
                    }
                }
            }
            encoder.copy_texture_to_buffer(
                frame.output.as_image_copy(),
                wgpu::TexelCopyBufferInfo { buffer: readback, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(layout.padded), rows_per_image: Some(size[1]) } },
                extent,
            );
            queue.submit(std::iter::once(encoder.finish()));
            Ok(())
        })();
        let errors = scopes.pop();
        recorded.map(|()| (layout, errors, state_plans))
    }
}
