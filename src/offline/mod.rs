/*
File: offline/mod.rs

Purpose:
Offline processing: run a `FilterChain` over an ordinary RGBA8 frame (an image, a video frame)
on wgpu and read the result back, computing exactly what a `FilterLayer` covering that frame
shows in the UI (conditions in the crate docs / README). Chain execution is NOT implemented
here: every GPU pass is recorded by `wgpu_backend::chain` (`ChainResources::record`,
`draw_composite`), the same code the UI callback uses.

Key structures:
- `OfflineRenderer` (public): device, queue, chain objects, frame textures and readback buffer,
  reused across calls (a video processes every frame through one renderer).
- `OfflineOptions`, `PointerState` (public, `options.rs`): the settings of a run.
- `OfflineTexture` (public, `texture.rs`): an image effects read as a user texture (F8).

Key functions:
- `OfflineRenderer::new()` / `from_render_state()` / `headless_async()` / `headless()` (native).
- `OfflineRenderer::process()` / `process_into()` (native, blocking) / `process_async()`.
- `OfflineRenderer::run()`: the shared implementation (CPU prep, tiles, record, readback).

Submodules: `video/` (feature `video`) builds frame streams on `run`.

Notes:
The frame is the layer: `layer_geometry` is called with the image as the screen and a layer rect
covering it, so the capture clamps at the image edge exactly as the UI clamps at the screen edge.
Tiles are windows of that one geometry (see `tiles.rs`). Blocking entry points must not run on a
GUI thread. All per-frame GPU work runs in awaited `ErrorScopes` (out-of-memory is a typed error);
the readback buffer is taken out of the renderer while mapped so cancellation cannot leave it
mapped. User textures are checked against the chain (`plan::check_textures`, the UI's check)
before any GPU work, uploaded once per attached `OfflineTexture` (kept per slot while the same
texture stays attached) into `Rgba8Unorm` textures bound through egui-wgpu's sampler mapping, and
are independent of tiles (their uv is their own).
Persistent state (F7): a chain with a state pass keeps its state in the renderer (`OfflineState`,
two state textures) from one run to the next (a video processor's frames), configured by
`OfflineOptions::with_state`, zeroed by `reset_state`; such a chain is never tiled
(`OfflineError::NotTileable`), and a failed run forgets the state.
*/

//! Offline processing: apply a [`FilterChain`] to an ordinary image or video frame on wgpu and
//! get the pixels back — the same result a [`FilterLayer`](crate::FilterLayer) covering that
//! frame shows in the UI.
//!
//! ```no_run
//! # fn main() -> Result<(), egui_filter_layers::OfflineError> {
//! use egui_filter_layers::{AlphaMode, FrameRef, OfflineOptions, OfflineRenderer, presets};
//! let (width, height) = (640, 480);
//! let pixels = vec![255_u8; 640 * 480 * 4]; // straight-alpha RGBA8, e.g. from a decoder
//! let mut renderer = OfflineRenderer::headless()?; // or ::from_render_state(&render_state)
//! let chain = presets::gaussian_blur(4.0).then(presets::saturation(1.3));
//! let options = OfflineOptions::default().with_pixels_per_point(1.0).with_time(0.0);
//! let result = renderer.process(&FrameRef::new(&pixels, width, height, AlphaMode::Straight), &chain, &options)?;
//! assert_eq!(result.pixels().len(), pixels.len());
//! # Ok(()) }
//! ```
//!
//! **Bit-identical to the UI** when all of these hold: the renderer runs on the UI's device
//! ([`OfflineRenderer::from_render_state`]); the UI shows the frame 1:1 (one frame pixel per
//! physical pixel, pixel-aligned) with the layer covering exactly the frame and its capture
//! clamped to the frame (e.g. the frame fills the viewport); dithering is off
//! (`WgpuOptions::with_dithering(false)` and egui-wgpu's own); the surface is not `*Srgb`; the frame is opaque
//! or [`OfflineOptions::with_background`] gives the colour beneath it. The 1-px antialiased edge
//! of a rounded mask is the exception: the UI blends the filter over the screen by coverage there
//! (up to one step; in linear on an sRGB surface, where it differs by tens of steps — up to 70
//! measured on a high-contrast blur + invert). Otherwise results agree
//! within rounding (about one 8-bit step for dithering or an sRGB surface); a layer over a
//! frame shown at another zoom filters a resampled image and is only a preview of the export.
//! Near the frame border the offline run clamps sampling at the frame edge, as the UI does at
//! the screen edge.
//!
//! Large frames are processed in tiles: above [`DEFAULT_MAX_TILE_SIZE`] pixels per side by
//! default ([`OfflineOptions::with_max_tile_size`]; always within the device's texture and buffer
//! limits). Tiles give identical results for effects sampling at texel centres, and at most two
//! 8-bit steps on a few values for effects sampling between them (refraction, a large blur),
//! provided every effect reads no further than its sample radius
//! ([`Effect::with_sample_radius`](crate::Effect::with_sample_radius)). GPU out-of-memory is
//! [`OfflineError::OutOfMemory`], never a panic.
//! The [`lens`](crate::presets::lens) preset reads anywhere inside the layer and declares no
//! radius; give it one (`Pass::with_sample_radius`) before tiling it.
//! A chain with a reduced-resolution pass ([`Pass::with_scale`](crate::Pass::with_scale), the
//! [`kawase_blur`](crate::presets::kawase_blur) and [`bloom`](crate::presets::bloom) presets)
//! cannot be tiled exactly: a frame that would need tiles is refused with
//! [`OfflineError::NotTileable`]; process it whole (`with_max_tile_size(None)`).
//!
//! The pipeline is 8-bit like the UI's capture: 16-bit input is converted to 8 bits first.
//!
//! **Persistent state** (a chain with a state pass, `Effect::new_state`): the renderer keeps the
//! state from one `process` call to the next — frame `n` continues frame `n - 1`, like a layer in
//! the UI — sized from the frame (`OfflineOptions::with_state` for the resolution and format);
//! [`OfflineRenderer::reset_state`] starts over from zero. A new frame size stretches the previous
//! state like a resized layer. Stateful chains cannot be tiled ([`OfflineError::NotTileable`]).
//!
//! User textures ([`OfflineOptions::with_texture`] with an [`OfflineTexture`]) are what
//! `FilterLayer::texture` attaches in the UI; [`OfflineTexture::from_color_image`] holds exactly
//! the bytes egui uploads for that image, so an effect reading them agrees with the UI.
//!
//! Video (feature `video`): see the `video` module — frame streams, source / sink traits and
//! ffmpeg / image-sequence adapters, all running through the same renderer.

mod error;
mod frame;
mod options;
#[cfg(feature = "image")]
mod image;
mod readback;
mod texture;
mod tiles;
#[cfg(feature = "video")]
pub mod video;

use egui::epaint::TessellationOptions;
use egui::{CornerRadius, Rect, pos2};

pub use error::OfflineError;
pub use frame::{AlphaMode, FrameRef, RgbaFrame};
pub use options::{OfflineOptions, PointerState};
pub use texture::OfflineTexture;

use crate::effect::{TEXTURE_SLOTS, USER_SAMPLER_BINDING, USER_TEXTURE_BINDING};
use crate::geometry::{LayerGeometry, MaskTransform, PxRect, layer_geometry, mask_transform, padding_px};
use crate::mask::{Mask, PackedCoverage, tessellate};
use crate::pass::{FilterChain, InputUniforms};
use crate::plan::{StatePlan, StatePlanInput, check_textures};
use crate::state::{StateError, StateFormat};
use crate::wgpu_backend::coverage::CoverageJob;
use crate::wgpu_backend::chain::{ChainJob, ChainRecording, ChainResources, ChainSource, ChainState, ChainTargets, ChainWindow, draw_composite};
use crate::wgpu_backend::gpu::{ErrorScopes, Target, begin_pass, egui_sampler_descriptor};
use frame::{MAX_SIDE, convert_output, frame_bytes};
use readback::{RowLayout, map_read};
use tiles::{Tile, TilePlan};

/// Format of the offline input and output textures: 8-bit gamma values stored as-is, like the
/// UI's capture and a non-sRGB surface.
const FRAME_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Extra overlap of tiles beyond the chain padding: the bilinear footprint of a sample taken at
/// exactly the declared reach.
const TILE_OVERLAP_MARGIN_PX: u32 = 1;

/// Default largest texture side of a run ([`OfflineOptions::with_max_tile_size`]). It bounds the
/// GPU memory of one run to about 450 MiB (RGBA8 source and output, two RGBA16F intermediates, the
/// readback buffer) instead of scaling with the device limit (about 7 GiB at 16384 pixels).
pub const DEFAULT_MAX_TILE_SIZE: u32 = 4096;

/// Runs filter chains over frames on a wgpu device and reads the results back.
///
/// Create one per device and reuse it: pipelines, textures and buffers persist between calls.
/// Processing blocks (native `process`) or awaits the GPU, so never call it on a GUI thread.
#[derive(Debug)]
pub struct OfflineRenderer {
    /// The device everything runs on.
    device: wgpu::Device,
    /// Its queue.
    queue: wgpu::Queue,
    /// The chain executor's objects (the same code the UI uses).
    chain: ChainResources,
    /// Composite pipeline writing `FRAME_FORMAT`.
    composite: wgpu::RenderPipeline,
    /// `max_texture_dimension_2d` of the device.
    max_texture_side: u32,
    /// Largest side `s` whose `s x s` readback fits the device's `max_buffer_size`.
    max_readback_side: u32,
    /// Textures of the current allocation (recreated when it changes).
    targets: Option<FrameTargets>,
    /// Staging buffer of the readback and its size, allocated for the largest tile of the current
    /// plan (recreated when the plan needs another size, so it also shrinks). Taken out while it
    /// is mapped: a failed or cancelled readback drops it instead of leaving it mapped.
    readback: Option<(wgpu::Buffer, u64)>,
    /// The prepared (premultiplied, packed) input of the current run; kept between runs so a
    /// frame stream reuses its capacity.
    scratch: Vec<u8>,
    /// The GPU copy of the texture attached at each user texture slot by the last run (reused
    /// while the same `OfflineTexture` stays attached, dropped when the slot empties).
    user_textures: [Option<UploadedTexture>; TEXTURE_SLOTS],
    /// Samplers by egui texture options (egui-wgpu's mapping), created on first use.
    samplers: std::collections::HashMap<egui::TextureOptions, wgpu::Sampler>,
    /// The persistent state of stateful chains (F7), carried from run to run.
    state: Option<OfflineState>,
}

/// The persistent state of an offline renderer: two state textures and where the latest state is.
#[derive(Debug)]
struct OfflineState {
    /// `Slot::StateA` / `Slot::StateB` (grown by the chain executor, keeping content).
    targets: [Option<Target>; 2],
    /// Which target holds the latest state.
    current: u8,
    /// Its content's size; `None` = no content yet (cleared by the next run).
    used: Option<[u32; 2]>,
    /// The format the targets hold.
    format: StateFormat,
}

/// The GPU copy of an attached [`OfflineTexture`].
#[derive(Debug)]
struct UploadedTexture {
    /// The texture it was uploaded from (identity: `OfflineTexture::same`).
    source: OfflineTexture,
    /// Texture + sampler in `ChainResources::user_layout` (the bind group keeps the GPU texture
    /// alive: wgpu resources are reference-counted by what uses them).
    bind_group: wgpu::BindGroup,
}

/// The textures of one allocation size: every run uses all of them at exactly `alloc`.
#[derive(Debug)]
struct FrameTargets {
    /// Allocated size.
    alloc: [u32; 2],
    /// The (premultiplied) input window.
    source: wgpu::Texture,
    /// View of `source`.
    source_view: wgpu::TextureView,
    /// The composite's output.
    output: wgpu::Texture,
    /// View of `output`.
    output_view: wgpu::TextureView,
    /// Ping-pong intermediates and uniforms (allocated at `alloc` too).
    chain: ChainTargets,
}

impl OfflineRenderer {
    /// A renderer on an existing device. Pass the adapter the device came from: it decides the
    /// intermediate format (RGBA16F where renderable and filterable), exactly as for the UI.
    ///
    /// # Errors
    /// [`OfflineError::Gpu`] when the device rejects the crate's pipelines.
    pub fn new(adapter: &wgpu::Adapter, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<Self, OfflineError> {
        let chain = ChainResources::create(device, queue, adapter).map_err(OfflineError::Gpu)?;
        let composite = chain.composite_pipeline(device, FRAME_FORMAT, wgpu::MultisampleState::default(), None).map_err(OfflineError::Gpu)?;
        let limits = device.limits();
        let max_readback_side = readback_side_limit(limits.max_buffer_size);
        log::info!(
            "egui_filter_layers (offline): renderer created (intermediates {:?}, max texture side {}, max readback side {max_readback_side})",
            chain.intermediate_format(),
            limits.max_texture_dimension_2d
        );
        Ok(Self {
            device: device.clone(),
            queue: queue.clone(),
            chain,
            composite,
            max_texture_side: limits.max_texture_dimension_2d,
            max_readback_side,
            targets: None,
            readback: None,
            scratch: Vec::new(),
            user_textures: [const { None }; TEXTURE_SLOTS],
            samplers: std::collections::HashMap::new(),
            state: None,
        })
    }

    /// Starts the persistent state (F7) over: the next run of a stateful chain begins from zero.
    /// The state textures are kept (reused by the next run).
    pub fn reset_state(&mut self) {
        if let Some(state) = self.state.as_mut() {
            state.used = None;
        }
    }

    /// A renderer on egui-wgpu's device: the same adapter, device and shader compiler as the
    /// UI's filter layers, the precondition for bit-identical results.
    ///
    /// # Errors
    /// As [`OfflineRenderer::new`].
    pub fn from_render_state(render_state: &egui_wgpu::RenderState) -> Result<Self, OfflineError> {
        Self::new(&render_state.adapter, &render_state.device, &render_state.queue)
    }

    /// A renderer on its own device: the default adapter of a new `wgpu::Instance` (backends
    /// from `WGPU_BACKEND` etc.), with the adapter's full limits (fewer tiles for large images).
    ///
    /// # Errors
    /// [`OfflineError::NoAdapter`], [`OfflineError::RequestDevice`], or as [`OfflineRenderer::new`].
    pub async fn headless_async() -> Result<Self, OfflineError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await.map_err(|error| OfflineError::NoAdapter(error.to_string()))?;
        let descriptor = wgpu::DeviceDescriptor { label: Some("egui_filter_layers offline"), required_limits: adapter.limits(), ..Default::default() };
        let (device, queue) = adapter.request_device(&descriptor).await.map_err(|error| OfflineError::RequestDevice(error.to_string()))?;
        log::info!("egui_filter_layers (offline): headless device on {:?}", adapter.get_info());
        Self::new(&adapter, &device, &queue)
    }

    /// Blocking [`OfflineRenderer::headless_async`] (native only; no async runtime needed).
    ///
    /// # Errors
    /// As [`OfflineRenderer::headless_async`].
    #[cfg(not(target_arch = "wasm32"))]
    pub fn headless() -> Result<Self, OfflineError> {
        readback::block_on(Self::headless_async())
    }

    /// The device this renderer runs on.
    #[must_use]
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Filters `frame` through `chain` and returns the result (blocking until the GPU is done).
    /// An empty chain returns the frame unfiltered (after the background, like the UI).
    ///
    /// # Errors
    /// Invalid frame or options, an effect the device rejects, sizes the pipeline cannot
    /// address, GPU or readback failures; see [`OfflineError`].
    #[cfg(not(target_arch = "wasm32"))]
    pub fn process(&mut self, frame: &FrameRef<'_>, chain: &FilterChain, options: &OfflineOptions) -> Result<RgbaFrame, OfflineError> {
        let mut pixels = Vec::new();
        self.process_into(frame, chain, options, &mut pixels)?;
        Ok(RgbaFrame::from_parts(frame.width(), frame.height(), options.output_alpha, pixels))
    }

    /// [`OfflineRenderer::process`] into a reusable buffer (resized to `width * height * 4`),
    /// for frame streams that should not allocate per frame.
    ///
    /// # Errors
    /// As [`OfflineRenderer::process`].
    #[cfg(not(target_arch = "wasm32"))]
    pub fn process_into(&mut self, frame: &FrameRef<'_>, chain: &FilterChain, options: &OfflineOptions, output: &mut Vec<u8>) -> Result<(), OfflineError> {
        readback::block_on(self.run(frame, chain, options, output, true))
    }

    /// Async [`OfflineRenderer::process`], usable on the web. It does not poll the device: on
    /// native, the readback completes when something polls it (eframe does every frame through
    /// its submissions; otherwise call `device().poll(..)`), so prefer `process` there.
    ///
    /// # Errors
    /// As [`OfflineRenderer::process`].
    pub async fn process_async(&mut self, frame: &FrameRef<'_>, chain: &FilterChain, options: &OfflineOptions) -> Result<RgbaFrame, OfflineError> {
        let mut pixels = Vec::new();
        self.run(frame, chain, options, &mut pixels, false).await?;
        Ok(RgbaFrame::from_parts(frame.width(), frame.height(), options.output_alpha, pixels))
    }

    /// The shared implementation: CPU preparation, one GPU run per tile, readback, output alpha.
    /// `block` polls the device (native, `PollType::Wait`) before awaiting each readback.
    /// Every allocation (textures, readback buffer, the prepared input, `output`'s capacity) is
    /// reused by the next run of the same size.
    async fn run(&mut self, frame: &FrameRef<'_>, chain: &FilterChain, options: &OfflineOptions, output: &mut Vec<u8>, block: bool) -> Result<(), OfflineError> {
        // The scratch buffer is moved out for the run (it is borrowed alongside `&mut self`) and
        // put back whatever the outcome, keeping its capacity.
        let mut pixels = std::mem::take(&mut self.scratch);
        let result = self.run_prepared(frame, chain, options, output, block, &mut pixels).await;
        self.scratch = pixels;
        result
    }

    /// [`OfflineRenderer::run`] with the prepared-input buffer passed in.
    async fn run_prepared(&mut self, frame: &FrameRef<'_>, chain: &FilterChain, options: &OfflineOptions, output: &mut Vec<u8>, block: bool, pixels: &mut Vec<u8>) -> Result<(), OfflineError> {
        options.validate()?;
        check_textures(chain.passes(), options.attached_textures())?;
        frame.premultiplied_rows_into(options.background, pixels)?;
        let pixels: &[u8] = pixels;
        let size = [frame.width(), frame.height()];
        output.clear();
        if chain.is_empty() {
            output.extend_from_slice(pixels);
            convert_output(output, options.output_alpha);
            return Ok(());
        }
        output.resize(frame_bytes(size[0], size[1])?, 0);

        let geometry = frame_geometry(size, chain.padding_points(), options)?;
        let stateful = chain.passes().iter().any(|pass| pass.effect().is_state());
        let state_format = options.state.format();
        if stateful && let Some(reason) = self.chain.state_format_problem(state_format) {
            return Err(OfflineError::State(StateError::UnsupportedFormat { format: state_format, reason }));
        }
        self.chain.evict_dropped();
        let pipelines = self.chain.pass_pipelines(&self.device, chain.passes(), state_format).map_err(|(module, message)| OfflineError::Effect { label: module.label.to_string(), message })?;
        let overlap = u32::try_from(padding_px(chain.padding_points(), geometry.pixels_per_layer_point)).unwrap_or(0).saturating_add(TILE_OVERLAP_MARGIN_PX);
        let device_limit = self.max_texture_side.min(self.max_readback_side);
        let limit = options.max_tile_size.map_or(device_limit, |cap| cap.min(device_limit));
        let plan = tiles::plan(size, overlap, limit)?;
        if plan.tiles.len() > 1
            && let Some((index, pass)) = chain.passes().iter().enumerate().find(|(_, pass)| pass.is_scaled())
        {
            // A scaled pass renders ceil(capture * scale) pixels of the WHOLE frame: a tile's
            // window would round its edges differently, so tiles would not match the whole-frame
            // (= UI) result. Refuse instead of approximating.
            return Err(OfflineError::NotTileable { reason: format!("pass {index} runs at scale {} (a reduced-resolution pass needs the whole frame; the frame is {}x{} and tiles are at most {limit} pixels)", pass.scale(), size[0], size[1]) });
        }
        if plan.tiles.len() > 1
            && let Some(index) = chain.passes().iter().position(|pass| pass.effect().is_state())
        {
            // The state spans the whole frame and continues across runs; a tile would step it
            // once per tile over a partial window.
            return Err(OfflineError::NotTileable { reason: format!("pass {index} is a state pass (persistent state needs the whole frame; the frame is {}x{} and tiles are at most {limit} pixels)", size[0], size[1]) });
        }
        if stateful && self.state.as_ref().is_none_or(|state| state.format != state_format) {
            self.state = Some(OfflineState { targets: [None, None], current: 0, used: None, format: state_format });
        }
        let state_input = self.state.as_ref().filter(|_| stateful).map(|state| StatePlanInput { scale: options.state.scale(), current: state.current, previous_px: state.used, step: true });
        if plan.tiles.len() > 1 {
            log::debug!("egui_filter_layers (offline): {}x{} processed in {} tiles of {:?}", size[0], size[1], plan.tiles.len(), plan.alloc);
        }
        self.ensure_allocations(&plan).await?;
        self.ensure_user_textures(options).await?;
        // A shape mask (F9), tessellated once per run like the UI does per paint: the frame is the
        // layer (layer points = frame pixels / ppp, no transform) and the screen.
        let coverage = match &options.mask {
            Mask::RoundedRect(_) => None,
            // `OfflineOptions::validate` refused it already; kept explicit so no run ever treats
            // regions as a plain mask.
            Mask::Regions(_) => return Err(options::regions_unsupported()),
            Mask::Shape(shape) => {
                let frame_rect = frame_rect(size, options.pixels_per_point)?;
                let meshes = tessellate(shape.clone(), frame_rect, options.pixels_per_point, TessellationOptions::default(), [1, 1]);
                Some((meshes.pack(), mask_transform(frame_rect, frame_rect, options.pixels_per_point, geometry.capture)))
            }
        };

        for tile in &plan.tiles {
            let run = TileRun { tile: *tile, geometry, image_width: size[0], time: options.time, input: options.input_uniforms(&geometry), state: state_input, coverage: coverage.as_ref() };
            let (layout, errors, state_plan) = self.record_tile(&run, pixels, chain, &pipelines)?;
            if let Some(error) = errors.await {
                // Textures or the readback buffer may now be invalid objects: recreate them next run.
                self.drop_allocations();
                self.state = None;
                return Err(offline_gpu_error(&error));
            }
            // The state step ran: the next run continues from its result.
            if let (Some(state), Some(state_plan)) = (self.state.as_mut(), state_plan) {
                state.current = state_plan.final_index;
                state.used = Some(state_plan.final_px);
            }
            // Taken out while mapped: an error below, or dropping this future (cancelling
            // `process_async`), drops the buffer instead of leaving it mapped for the next run.
            let Some((buffer, buffer_size)) = self.readback.take() else {
                return Err(OfflineError::Gpu("the readback buffer is missing (crate bug)".to_owned()));
            };
            let slice = buffer.slice(..layout.buffer_size());
            let mapped = map_read(&slice);
            if block {
                self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).map_err(|error| OfflineError::Readback(error.to_string()))?;
            }
            mapped.await?;
            let copied = copy_rows(&slice, layout, tile.output, size[0], output);
            buffer.unmap();
            copied?;
            self.readback = Some((buffer, buffer_size));
        }
        convert_output(output, options.output_alpha);
        Ok(())
    }

    /// Drops the frame textures and the readback buffer (after a GPU error they may be invalid).
    fn drop_allocations(&mut self) {
        self.targets = None;
        self.readback = None;
    }

    /// Makes the textures match the plan's allocation exactly and the readback buffer hold
    /// exactly the plan's largest tile, inside [`ErrorScopes`]: an allocation the device refuses
    /// (out of memory, a limit) is a typed error and leaves nothing half-created behind.
    ///
    /// # Errors
    /// [`OfflineError::OutOfMemory`] / [`OfflineError::Gpu`] from the device;
    /// [`OfflineError::SizeOverflow`] for an unaddressable readback.
    async fn ensure_allocations(&mut self, plan: &TilePlan) -> Result<(), OfflineError> {
        let mut needed = 0;
        for tile in &plan.tiles {
            needed = needed.max(RowLayout::new(px_u32(tile.output.w)?, px_u32(tile.output.h)?)?.buffer_size());
        }
        let scopes = ErrorScopes::push(&self.device);
        self.ensure_targets(plan);
        if self.readback.as_ref().is_none_or(|(_, size)| *size != needed) {
            // Release the old buffer first so both never exist at once.
            self.readback = None;
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("egui_filter_layers offline readback"), size: needed, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            self.readback = Some((buffer, needed));
        }
        if let Some(error) = scopes.pop().await {
            self.drop_allocations();
            log::warn!("egui_filter_layers (offline): allocating {:?} textures / a {needed}-byte readback failed: {error}", plan.alloc);
            return Err(offline_gpu_error(&error));
        }
        Ok(())
    }

    /// Makes the GPU copies match the textures `options` attaches: a slot keeps its upload while
    /// the same `OfflineTexture` stays attached, uploads a new one otherwise (inside
    /// [`ErrorScopes`], awaited), and drops it when the slot is empty.
    ///
    /// # Errors
    /// [`OfflineError::TextureTooLarge`] for a texture beyond the device's limit (before any GPU
    /// work); [`OfflineError::OutOfMemory`] / [`OfflineError::Gpu`] from the device.
    async fn ensure_user_textures(&mut self, options: &OfflineOptions) -> Result<(), OfflineError> {
        let mut uploads = Vec::new();
        for (slot, cached) in self.user_textures.iter_mut().enumerate() {
            match options.texture(slot) {
                None => *cached = None,
                Some(texture) if cached.as_ref().is_some_and(|uploaded| uploaded.source.same(texture)) => {}
                Some(texture) => {
                    let [width, height] = texture.size();
                    if width > self.max_texture_side || height > self.max_texture_side {
                        return Err(OfflineError::TextureTooLarge { slot, width, height, max: self.max_texture_side });
                    }
                    *cached = None;
                    uploads.push((slot, texture.clone()));
                }
            }
        }
        if uploads.is_empty() {
            return Ok(());
        }
        let scopes = ErrorScopes::push(&self.device);
        for (slot, texture) in uploads {
            let uploaded = self.upload_texture(texture);
            if let Some(entry) = self.user_textures.get_mut(slot) {
                *entry = Some(uploaded);
            }
        }
        if let Some(error) = scopes.pop().await {
            self.user_textures = [const { None }; TEXTURE_SLOTS];
            log::warn!("egui_filter_layers (offline): uploading a user texture failed: {error}");
            return Err(offline_gpu_error(&error));
        }
        Ok(())
    }

    /// Creates the GPU copy of `texture` (`Rgba8Unorm`, like egui-wgpu's textures) and its bind
    /// group with egui-wgpu's sampler for its options. Call inside [`ErrorScopes`].
    fn upload_texture(&mut self, texture: OfflineTexture) -> UploadedTexture {
        let [width, height] = texture.size();
        let extent = wgpu::Extent3d { width, height, depth_or_array_layers: 1 };
        let gpu = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("egui_filter_layers offline user texture"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: FRAME_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(gpu.as_image_copy(), texture.pixels(), wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(width.saturating_mul(4)), rows_per_image: Some(height) }, extent);
        let device = &self.device;
        let sampler = self.samplers.entry(texture.options()).or_insert_with(|| device.create_sampler(&egui_sampler_descriptor(texture.options())));
        let view = gpu.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("egui_filter_layers offline user texture"),
            layout: self.chain.user_layout(),
            entries: &[wgpu::BindGroupEntry { binding: USER_TEXTURE_BINDING, resource: wgpu::BindingResource::TextureView(&view) }, wgpu::BindGroupEntry { binding: USER_SAMPLER_BINDING, resource: wgpu::BindingResource::Sampler(sampler) }],
        });
        UploadedTexture { source: texture, bind_group }
    }

    /// Makes `targets` match the plan's allocation exactly (a different size recreates them, so
    /// every run samples textures of exactly the planned size).
    fn ensure_targets(&mut self, plan: &TilePlan) {
        if self.targets.as_ref().is_some_and(|targets| targets.alloc == plan.alloc) {
            return;
        }
        // Release the old textures before creating the new ones (lower peak memory).
        self.targets = None;
        let texture = |label: &str, usage: wgpu::TextureUsages| {
            self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: plan.alloc[0], height: plan.alloc[1], depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: FRAME_FORMAT,
                usage,
                view_formats: &[],
            })
        };
        let source = texture("egui_filter_layers offline source", wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST);
        let output = texture("egui_filter_layers offline output", wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC);
        let source_view = source.create_view(&wgpu::TextureViewDescriptor::default());
        let output_view = output.create_view(&wgpu::TextureViewDescriptor::default());
        self.targets = Some(FrameTargets { alloc: plan.alloc, source, source_view, output, output_view, chain: ChainTargets::new() });
    }

    /// Uploads one tile's window, records the chain (`chain.rs`, which may grow its
    /// intermediates) and the composite into the output texture, copies the output rect into the
    /// readback buffer and submits — all inside [`ErrorScopes`]. Returns the readback layout of
    /// the tile, the future of the first GPU error of that work (await it before mapping) and, for
    /// a stateful run, where the latest state is afterwards.
    fn record_tile(&mut self, run: &TileRun<'_>, pixels: &[u8], chain: &FilterChain, pipelines: &[wgpu::RenderPipeline]) -> Result<(RowLayout, impl Future<Output = Option<wgpu::Error>> + use<>, Option<StatePlan>), OfflineError> {
        let (window, out) = (run.tile.window, run.tile.output);
        let [out_w, out_h] = [px_u32(out.w)?, px_u32(out.h)?];
        let layout = RowLayout::new(out_w, out_h)?;
        let upload = window_upload(pixels, run.image_width, window)?;
        let (Some(targets), Some((readback, _))) = (self.targets.as_mut(), self.readback.as_ref()) else {
            return Err(OfflineError::Gpu("offline targets are missing (crate bug)".to_owned()));
        };
        let (device, queue, resources, composite) = (&self.device, &self.queue, &self.chain, &self.composite);
        let user_textures = self.user_textures.each_ref().map(|uploaded| uploaded.as_ref().map(|uploaded| &uploaded.bind_group));
        let mut state = self.state.as_mut().filter(|_| run.state.is_some()).map(|state| ChainState { targets: &mut state.targets, format: state.format });
        let mut state_plan = None;

        let scopes = ErrorScopes::push(device);
        let recorded = 'record: {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo { texture: &targets.source, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                upload.bytes,
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(upload.bytes_per_row), rows_per_image: Some(upload.extent.height) },
                upload.extent,
            );
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("egui_filter_layers offline") });
            let job = ChainJob {
                source: ChainSource { view: &targets.source_view, allocated: targets.alloc },
                geometry: run.geometry,
                // The tile's output rect lands at the output texture's origin.
                window: ChainWindow { rect: window, composite_origin: [-out.x, -out.y], min_target_size: targets.alloc },
                time: run.time,
                input: run.input,
                passes: chain.passes(),
                pipelines,
                user_textures,
                state: run.state,
                // Untextured meshes only (`Mask::check_offline`): the white group stands in for
                // egui's white atlas pixel.
                // No region ranges: region masks are refused offline before any run.
                coverage: run.coverage.map(|(packed, transform)| CoverageJob { packed, transform: *transform, textures: vec![resources.coverage().white_group(); packed.draws.len()], regions: &[] }),
            };
            let recorded = resources.record(&mut ChainRecording { device, queue, encoder: &mut encoder }, &mut targets.chain, &job, state.as_mut());
            let Some((bind_group, planned_state)) = recorded else {
                break 'record false;
            };
            state_plan = planned_state;
            {
                let mut pass = begin_pass(&mut encoder, &targets.output_view, "egui_filter_layers offline composite", wgpu::Color::TRANSPARENT);
                draw_composite(&mut pass, composite, &bind_group, PxRect { x: 0, y: 0, w: out.w, h: out.h });
            }
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo { texture: &targets.output, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                wgpu::TexelCopyBufferInfo { buffer: readback, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(layout.padded), rows_per_image: Some(out_h) } },
                wgpu::Extent3d { width: out_w, height: out_h, depth_or_array_layers: 1 },
            );
            queue.submit(std::iter::once(encoder.finish()));
            true
        };
        let errors = scopes.pop();
        if recorded { Ok((layout, errors, state_plan)) } else { Err(OfflineError::Gpu("the chain executor rejected the tile (crate bug)".to_owned())) }
    }
}

/// What one tile's GPU run needs besides the renderer (a shape mask is tessellated and packed
/// once per `run`, each tile rasterizes its own window of it).
struct TileRun<'a> {
    /// The tile.
    tile: Tile,
    /// The whole frame's geometry (every tile uses the same).
    geometry: LayerGeometry,
    /// Width of the frame (row length of the prepared pixels).
    image_width: u32,
    /// `fl.time`.
    time: f32,
    /// `fl.delta_time`, `fl.frame`, `fl.pointer` (the whole frame's; every tile uses the same).
    input: InputUniforms,
    /// The renderer's state before this run (stateful chains, never tiled).
    state: Option<StatePlanInput>,
    /// A shape mask's packed meshes and their layer-points -> frame-pixels transform.
    coverage: Option<&'a (PackedCoverage, MaskTransform)>,
}

/// The layer geometry of a whole `size` frame: the layer rect covers the frame (in points at
/// `options.pixels_per_point`) and the frame is the screen, so the capture is the frame itself.
///
/// # Errors
/// [`OfflineError::TooLarge`] beyond `MAX_SIDE`, [`OfflineError::InvalidOption`] when the
/// geometry is degenerate (a non-positive scale) or does not map the layer back onto exactly
/// the frame (`f32` rounding of `width / ppp * ppp` at sides of millions of pixels).
fn frame_geometry(size: [u32; 2], padding_points: f32, options: &OfflineOptions) -> Result<LayerGeometry, OfflineError> {
    if size[0] > MAX_SIDE || size[1] > MAX_SIDE {
        return Err(OfflineError::TooLarge { width: size[0], height: size[1], max: MAX_SIDE });
    }
    let ppp = options.pixels_per_point;
    let rect = frame_rect(size, ppp)?;
    let corner_radius = match &options.mask {
        Mask::RoundedRect(radius) => *radius,
        Mask::Shape(_) | Mask::Regions(_) => CornerRadius::ZERO,
    };
    let geometry = layer_geometry(rect, Rect::EVERYTHING, ppp, size, padding_points, corner_radius, 1.0)
        .ok_or_else(|| OfflineError::InvalidOption { option: "pixels_per_point", reason: format!("{ppp} leaves no visible pixel of a {}x{} frame", size[0], size[1]) })?;
    // Tiles, uploads and the readback address the frame through `capture`: it must be the frame,
    // and the layer rect (`rect_uv`, the mask) must end exactly at its edges.
    let frame = PxRect { x: 0, y: 0, w: points_px(size[0])?, h: points_px(size[1])? };
    let edges_exact = geometry.mask_rect_px[2].to_bits() == crate::geometry::px_f32(frame.w).to_bits() && geometry.mask_rect_px[3].to_bits() == crate::geometry::px_f32(frame.h).to_bits();
    if geometry.capture != frame || geometry.region != frame || !edges_exact {
        return Err(OfflineError::InvalidOption {
            option: "pixels_per_point",
            reason: format!("{ppp} does not map a {}x{} frame onto whole pixels exactly (f32 rounding); use a power of two or a narrower frame", size[0], size[1]),
        });
    }
    Ok(geometry)
}

/// The layer rect of a whole `size` frame in points at `pixels_per_point` (the frame is the layer
/// and the screen): `(0, 0)..(size / ppp)`. Shape masks are in these points.
///
/// # Errors
/// [`OfflineError::SizeOverflow`] for a side beyond `i32`.
fn frame_rect(size: [u32; 2], pixels_per_point: f32) -> Result<Rect, OfflineError> {
    let points = |pixels: u32| i32::try_from(pixels).map(crate::geometry::px_f32).map_err(|_| OfflineError::SizeOverflow { what: "an image side" });
    Ok(Rect::from_min_max(pos2(0.0, 0.0), pos2(points(size[0])? / pixels_per_point, points(size[1])? / pixels_per_point)))
}

/// An image side as `i32`.
fn points_px(side: u32) -> Result<i32, OfflineError> {
    i32::try_from(side).map_err(|_| OfflineError::SizeOverflow { what: "an image side" })
}

/// The bytes of one window of the tightly packed frame, and how to upload them.
struct WindowUpload<'a> {
    /// Starts at the window's top-left pixel.
    bytes: &'a [u8],
    /// The frame's row length in bytes.
    bytes_per_row: u32,
    /// The window's size.
    extent: wgpu::Extent3d,
}

/// Locates `window` inside the tightly packed `pixels` of a frame `image_width` pixels wide.
///
/// # Errors
/// [`OfflineError::SizeOverflow`] on arithmetic overflow; [`OfflineError::Gpu`] when the
/// window lies outside the buffer (a planning bug).
fn window_upload(pixels: &[u8], image_width: u32, window: PxRect) -> Result<WindowUpload<'_>, OfflineError> {
    let overflow = || OfflineError::SizeOverflow { what: "a tile upload" };
    let bytes_per_row = image_width.checked_mul(4).ok_or_else(overflow)?;
    let start = usize::try_from(px_u32(window.y)?)
        .ok()
        .and_then(|y| y.checked_mul(usize::try_from(bytes_per_row).ok()?))
        .and_then(|row| row.checked_add(usize::try_from(px_u32(window.x).ok()?).ok()?.checked_mul(4)?))
        .ok_or_else(overflow)?;
    let bytes = pixels.get(start..).ok_or_else(|| OfflineError::Gpu(format!("tile window {window:?} lies outside the frame (crate bug)")))?;
    Ok(WindowUpload { bytes, bytes_per_row, extent: wgpu::Extent3d { width: px_u32(window.w)?, height: px_u32(window.h)?, depth_or_array_layers: 1 } })
}

/// Copies a mapped tile (`layout`, padded rows) into its `rect` of the tightly packed `output`
/// of a frame `image_width` pixels wide.
///
/// # Errors
/// [`OfflineError::Readback`] when the range cannot be viewed; [`OfflineError::SizeOverflow`] /
/// [`OfflineError::Gpu`] on inconsistent sizes (a planning bug).
fn copy_rows(slice: &wgpu::BufferSlice<'_>, layout: RowLayout, rect: PxRect, image_width: u32, output: &mut [u8]) -> Result<(), OfflineError> {
    let view = slice.get_mapped_range().map_err(|error| OfflineError::Readback(error.to_string()))?;
    let overflow = || OfflineError::SizeOverflow { what: "a tile readback" };
    let [padded, tight] = [layout.padded, layout.tight].map(|bytes| usize::try_from(bytes).unwrap_or(usize::MAX));
    let frame_row = usize::try_from(image_width).ok().and_then(|width| width.checked_mul(4)).ok_or_else(overflow)?;
    let [x, y] = [px_u32(rect.x)?, px_u32(rect.y)?].map(|value| usize::try_from(value).unwrap_or(usize::MAX));
    for (row, source) in view.chunks(padded).take(usize::try_from(layout.rows).unwrap_or(0)).enumerate() {
        let start = y.checked_add(row).and_then(|line| line.checked_mul(frame_row)).and_then(|at| at.checked_add(x.checked_mul(4)?)).ok_or_else(overflow)?;
        let target = start.checked_add(tight).and_then(|end| output.get_mut(start..end));
        let (Some(target), Some(source)) = (target, source.get(..tight)) else {
            return Err(OfflineError::Gpu(format!("tile {rect:?} does not fit the frame (crate bug)")));
        };
        target.copy_from_slice(source);
    }
    Ok(())
}

/// A non-negative pixel count as `u32`.
fn px_u32(value: i32) -> Result<u32, OfflineError> {
    u32::try_from(value).map_err(|_| OfflineError::SizeOverflow { what: "a pixel coordinate" })
}

/// The typed offline error of a device error: out-of-memory separately (the caller can retry
/// with a smaller [`OfflineOptions::with_max_tile_size`]), everything else as [`OfflineError::Gpu`].
fn offline_gpu_error(error: &wgpu::Error) -> OfflineError {
    match error {
        wgpu::Error::OutOfMemory { .. } => OfflineError::OutOfMemory(error.to_string()),
        wgpu::Error::Validation { .. } | wgpu::Error::Internal { .. } => OfflineError::Gpu(error.to_string()),
    }
}

/// The largest side `s` (at least 1) whose `s x s` RGBA8 readback, rows padded to
/// `COPY_BYTES_PER_ROW_ALIGNMENT`, fits `max_buffer_size` bytes. Any tile or untiled frame with
/// both sides `<= s` then fits too (the padded row and the row count only grow with the side).
fn readback_side_limit(max_buffer_size: u64) -> u32 {
    let fits = |side: u32| RowLayout::new(side, side).is_ok_and(|layout| layout.buffer_size() <= max_buffer_size);
    // sqrt(max / 4) is an upper bound (4 bytes per pixel, padding only adds); step down past the
    // row padding (at most a few dozen steps).
    // A row must also fit `u32` bytes (`RowLayout`), hence the `u32::MAX / 4` cap.
    let mut side = u32::try_from((max_buffer_size / 4).isqrt()).unwrap_or(u32::MAX).clamp(1, u32::MAX / 4);
    while side > 1 && !fits(side) {
        side -= 1;
    }
    side
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_frame_is_the_whole_layer_and_the_capture() {
        let options = OfflineOptions::default().with_pixels_per_point(2.0).with_corner_radius(4);
        let Ok(geometry) = frame_geometry([301, 200], 10.0, &options) else { panic!("valid") };
        let whole = PxRect { x: 0, y: 0, w: 301, h: 200 };
        assert_eq!((geometry.region, geometry.capture), (whole, whole), "padding clamps at the frame edge");
        assert!((geometry.pixels_per_layer_point - 2.0).abs() < f32::EPSILON);
        assert!((geometry.rect_uv[2] - 1.0).abs() < 1e-6 && geometry.rect_uv[0].abs() < 1e-6);
        assert!((geometry.mask_radii_px[0] - 8.0).abs() < f32::EPSILON);
        assert!(matches!(frame_geometry([10, 10], 0.0, &OfflineOptions::default().with_pixels_per_point(0.0)), Err(OfflineError::InvalidOption { .. })));
        assert!(matches!(frame_geometry([MAX_SIDE + 1, 10], 0.0, &OfflineOptions::default()), Err(OfflineError::TooLarge { .. })));
    }

    #[test]
    fn a_frame_the_scale_cannot_map_back_exactly_is_rejected() {
        // `width / 1.1 * 1.1` rounds to another whole pixel in f32 there: the geometry would not
        // be the frame, so tiles and the readback would disagree with it.
        let at = |width, ppp| frame_geometry([width, 1], 0.0, &OfflineOptions::default().with_pixels_per_point(ppp));
        // 4 613 737 maps one pixel wider (the layer rect overshoots the frame), 11 534 339 one
        // pixel narrower (the capture would miss the last column).
        for width in [4_613_737, 11_534_339] {
            assert!(matches!(at(width, 1.1), Err(OfflineError::InvalidOption { option: "pixels_per_point", .. })), "{width}");
        }
        assert!(at(4_613_736, 1.1).is_ok());
        // Powers of two are exact at every side.
        for ppp in [0.5, 1.0, 2.0] {
            let Ok(geometry) = at(MAX_SIDE, ppp) else { panic!("ppp {ppp} is exact") };
            assert_eq!(geometry.capture.w, i32::try_from(MAX_SIDE).unwrap_or(0));
        }
    }

    #[test]
    fn the_readback_side_fits_the_buffer_limit() {
        // wgpu's default limit (256 MiB) holds exactly an 8192 x 8192 readback.
        assert_eq!(readback_side_limit(256 << 20), 8192);
        for max in [1, 255, 256, 1 << 20, (1 << 20) + 7, 256 << 20, 2 << 30, u64::MAX] {
            let limit = readback_side_limit(max);
            let bytes = |side| RowLayout::new(side, side).map_or(u64::MAX, |layout| layout.buffer_size());
            assert!(limit == 1 || bytes(limit) <= max, "{max}: {limit}");
            // At `u64::MAX` the row length (`u32` bytes) is the limit instead.
            assert!(bytes(limit + 1) > max || max == u64::MAX, "{max}: {limit} is not the largest");
        }
    }

    #[test]
    fn the_default_options_bound_the_tile_size() {
        assert_eq!(OfflineOptions::default().max_tile_size, Some(DEFAULT_MAX_TILE_SIZE));
        assert_eq!(OfflineOptions::default().with_max_tile_size(None).max_tile_size, None);
    }

    #[test]
    fn options_are_validated() {
        assert!(OfflineOptions::default().validate().is_ok());
        assert!(OfflineOptions::default().with_pixels_per_point(f32::NAN).validate().is_err());
        assert!(OfflineOptions::default().with_pixels_per_point(-1.0).validate().is_err());
        assert!(OfflineOptions::default().with_time(f32::INFINITY).validate().is_err());
    }

    #[test]
    fn windows_are_located_in_the_packed_frame() {
        let pixels: Vec<u8> = (0..=255).cycle().take(10 * 6 * 4).collect();
        let Ok(upload) = window_upload(&pixels, 10, PxRect { x: 3, y: 2, w: 4, h: 3 }) else { panic!("inside") };
        assert_eq!(upload.bytes_per_row, 40);
        assert_eq!(upload.bytes.first(), pixels.get(2 * 40 + 3 * 4));
        assert_eq!((upload.extent.width, upload.extent.height), (4, 3));
        assert!(window_upload(&pixels, 10, PxRect { x: 0, y: 7, w: 1, h: 1 }).is_err());
        assert!(window_upload(&pixels, 10, PxRect { x: -1, y: 0, w: 1, h: 1 }).is_err());
    }
}
