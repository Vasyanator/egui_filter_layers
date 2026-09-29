/*
File: wgpu_backend/mod.rs

Purpose:
The `egui-wgpu` backend: public `install_wgpu` / `WgpuOptions`, the `Backend` implementation, and
the state shared between the GUI thread (painting, the end-of-pass plugin) and the paint
callbacks running inside egui-wgpu's renderer.

Key structures:
- `WgpuOptions` (public builder, private fields): what the backend must know about egui's render
  pass that neither `egui_wgpu::RenderState` nor egui-wgpu's `Renderer` expose (MSAA,
  depth/stencil, dithering, clear colour); setters mirror eframe's option names,
  `From<&egui_wgpu::RendererOptions>` converts a custom integration's options exactly. A mismatch
  is undetectable here (wgpu rejects egui's encoder at `finish`, outside the crate).
- `Shared`: install-wide state — options, the `RenderState`, pending layers, health latches
  (backend, per effect, per unresolvable user texture), log-once latches (`LoggedOnce`; per
  install, so a re-install may log each message again).
- `LayerSlot`: one painted shader layer's hand-over cell (plan from the plugin -> `prepare`,
  drawn target from `prepare` -> `paint`).

Key functions:
- `install_wgpu()` (public).
- `WgpuBackend::paint_layer()`: adds the `egui_wgpu::Callback` and registers the layer as pending.

Notes:
Pipeline of one frame: `paint_layer` (GUI thread) -> `plugin::ShaderLayersPlugin::output_hook`
(end of the egui pass: selects + tessellates the beneath content, snapshots texture bind groups)
-> `render::FilterCallback::prepare` (offscreen re-render, then the egui-free chain executor in
`chain.rs` runs the filter passes on egui's encoder) ->
`paint` (composite into egui's render pass). The content under a filter is therefore drawn
TWICE per frame on wgpu.
*/

mod beneath;
pub(crate) mod coverage;
// The chain executor and its wgpu helpers hold no egui state; crate-visible so a runner on its
// own device can drive the same executor.
pub(crate) mod chain;
pub(crate) mod gpu;
mod plugin;
mod render;
mod resources;
mod textures;

use std::any::Any;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};

use egui::epaint::ClippedPrimitive;
use egui::{Rect, TextureId};

use crate::backend::{Backend, BackendKind, BackendStatus, EffectFailure, InstallError, LayerJob, TextureFailure, register};
use crate::effect::{CompiledModule, TEXTURE_SLOTS};
use crate::state::{StateError, StateFailure, StateFormat};
use crate::geometry::PxRect;
use plugin::ShaderLayersPlugin;
use render::FilterCallback;
use textures::TextureCopies;

/// How egui-wgpu's main render pass is configured, which the filter's pipelines must match.
///
/// Neither `egui_wgpu::RenderState` nor egui-wgpu's `Renderer` expose this, so pass the SAME
/// values your integration gave egui-wgpu. The setters mirror eframe's option names:
///
/// | eframe (`NativeOptions` / `WebOptions`) | here |
/// |---|---|
/// | `multisampling` (native only) | [`WgpuOptions::with_multisampling`] |
/// | `depth_buffer` | [`WgpuOptions::with_depth_buffer`] |
/// | `stencil_buffer` (native only; the web has none) | [`WgpuOptions::with_stencil_buffer`] |
/// | `dithering` | [`WgpuOptions::with_dithering`] |
/// | `App::clear_color` | [`WgpuOptions::with_clear_color`] |
///
/// A custom integration that builds its own `egui_wgpu::RendererOptions` converts them exactly
/// with `WgpuOptions::from(&renderer_options)` (plus [`WgpuOptions::with_clear_color`]).
///
/// **A mismatch cannot be detected by this crate and is not recoverable.** The filter's
/// pipelines are valid on their own; wgpu only rejects them when they are recorded into egui's
/// pass, and it reports that when egui's command encoder is finished — outside this crate. Under
/// eframe that is a wgpu validation error with no handler (neither eframe nor egui-wgpu installs
/// one): wgpu-core's default handler **panics** — natively and in WebGL2 builds, which also run
/// wgpu-core — while a browser WebGPU backend logs an uncaptured error to the console and the
/// frame's commands are dropped. The options MUST match `NativeOptions` / `WebOptions`.
///
/// egui-wgpu's `RendererOptions::predictable_texture_filtering` is not mirrored: the backdrop
/// re-render always uses hardware texture filtering.
#[derive(Clone, Copy, Debug)]
pub struct WgpuOptions {
    /// MSAA sample count of egui's render pass as given (`0` and `1` both mean no MSAA).
    msaa_samples: u32,
    /// Depth/stencil attachment of egui's render pass.
    depth_stencil: DepthStencil,
    /// Whether egui-wgpu dithers (`RendererOptions::dithering`).
    dithering: bool,
    /// Premultiplied, gamma-space RGBA the frame is cleared to before egui draws.
    clear_color: [f32; 4],
}

/// How the depth/stencil attachment was specified: as eframe's bit counts (resolved with
/// `egui_wgpu::depth_format_from_bits`, the mapping eframe itself uses, so the two setters compose
/// in any order) or as an explicit format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DepthStencil {
    /// eframe's `depth_buffer` / `stencil_buffer` bits.
    Bits {
        /// Depth bits.
        depth: u8,
        /// Stencil bits.
        stencil: u8,
    },
    /// An explicit format (`RendererOptions::depth_stencil_format`).
    Format(Option<wgpu::TextureFormat>),
}

impl Default for WgpuOptions {
    /// eframe's defaults: no MSAA, no depth or stencil buffer, dithering on, and eframe's default
    /// clear colour. Equal to `WgpuOptions::from(&egui_wgpu::RendererOptions::default())`.
    fn default() -> Self {
        Self { msaa_samples: 0, depth_stencil: DepthStencil::Bits { depth: 0, stencil: 0 }, dithering: true, clear_color: egui::Color32::from_rgba_unmultiplied(12, 12, 12, 180).to_normalized_gamma_f32() }
    }
}

impl PartialEq for WgpuOptions {
    /// Equal when they describe the same render pass: effective sample count, resolved
    /// depth/stencil format, dithering and clear colour (however they were specified).
    fn eq(&self, other: &Self) -> bool {
        self.msaa_samples() == other.msaa_samples() && self.depth_stencil_format() == other.depth_stencil_format() && self.dithering == other.dithering && self.clear_color == other.clear_color
    }
}

impl From<&egui_wgpu::RendererOptions> for WgpuOptions {
    /// The render pass egui-wgpu builds from `options` (MSAA, depth/stencil format, dithering),
    /// with eframe's default clear colour. `predictable_texture_filtering` is not mirrored (see
    /// [`WgpuOptions`]).
    fn from(options: &egui_wgpu::RendererOptions) -> Self {
        Self { msaa_samples: options.msaa_samples, depth_stencil: DepthStencil::Format(options.depth_stencil_format), dithering: options.dithering, ..Self::default() }
    }
}

impl WgpuOptions {
    /// eframe's `NativeOptions::multisampling`: MSAA sample count of egui's pass (`0` and `1` mean
    /// none). eframe on the web never multisamples; leave it at the default there.
    #[must_use]
    pub fn with_multisampling(mut self, samples: u16) -> Self {
        self.msaa_samples = u32::from(samples);
        self
    }

    /// eframe's `depth_buffer` bits (`0` = no depth buffer). Combined with
    /// [`WgpuOptions::with_stencil_buffer`] in any order, exactly as eframe maps them
    /// (`egui_wgpu::depth_format_from_bits`; a combination it does not know means no attachment).
    /// Replaces an explicit [`WgpuOptions::with_depth_stencil_format`] (stencil bits reset to 0).
    #[must_use]
    pub fn with_depth_buffer(mut self, bits: u8) -> Self {
        let stencil = match self.depth_stencil {
            DepthStencil::Bits { stencil, .. } => stencil,
            DepthStencil::Format(_) => 0,
        };
        self.depth_stencil = DepthStencil::Bits { depth: bits, stencil };
        self
    }

    /// eframe's `NativeOptions::stencil_buffer` bits (`0` = none; eframe on the web has none). See
    /// [`WgpuOptions::with_depth_buffer`].
    #[must_use]
    pub fn with_stencil_buffer(mut self, bits: u8) -> Self {
        let depth = match self.depth_stencil {
            DepthStencil::Bits { depth, .. } => depth,
            DepthStencil::Format(_) => 0,
        };
        self.depth_stencil = DepthStencil::Bits { depth, stencil: bits };
        self
    }

    /// An explicit depth/stencil attachment format of egui's pass
    /// (`egui_wgpu::RendererOptions::depth_stencil_format`), for integrations that do not use
    /// eframe's bit counts. Replaces earlier `with_depth_buffer` / `with_stencil_buffer` calls.
    #[must_use]
    pub fn with_depth_stencil_format(mut self, format: Option<wgpu::TextureFormat>) -> Self {
        self.depth_stencil = DepthStencil::Format(format);
        self
    }

    /// eframe's `dithering` (egui-wgpu's `RendererOptions::dithering`); mirrored when the
    /// backdrop is re-rendered so filtered and unfiltered pixels quantize alike.
    #[must_use]
    pub fn with_dithering(mut self, dithering: bool) -> Self {
        self.dithering = dithering;
        self
    }

    /// The premultiplied, gamma-space RGBA the frame is cleared to before egui draws (what
    /// `eframe::App::clear_color` returns in your app). It is the backdrop where nothing opaque
    /// was painted.
    ///
    /// Like eframe, it applies to the root and deferred viewports; immediate viewports are
    /// cleared to transparent. On a `*Srgb` target it is interpreted as eframe does (handed to
    /// wgpu's clear unchanged, i.e. as a linear value), so unpainted pixels inside and outside a
    /// filter agree.
    #[must_use]
    pub fn with_clear_color(mut self, clear_color: [f32; 4]) -> Self {
        self.clear_color = clear_color;
        self
    }

    /// The effective MSAA sample count of egui's pass (`0` given means `1`).
    #[must_use]
    pub fn msaa_samples(&self) -> u32 {
        self.msaa_samples.max(1)
    }

    /// The depth/stencil attachment format of egui's pass, if it has one (bit counts resolved like
    /// eframe resolves them).
    #[must_use]
    pub fn depth_stencil_format(&self) -> Option<wgpu::TextureFormat> {
        match self.depth_stencil {
            DepthStencil::Bits { depth, stencil } => egui_wgpu::depth_format_from_bits(depth, stencil),
            DepthStencil::Format(format) => format,
        }
    }

    /// Whether egui-wgpu dithers.
    #[must_use]
    pub fn dithering(&self) -> bool {
        self.dithering
    }

    /// The clear colour (premultiplied, gamma-space RGBA).
    #[must_use]
    pub fn clear_color(&self) -> [f32; 4] {
        self.clear_color
    }
}

/// Source of `Shared::generation`: GPU resources built for one install are never reused by another.
///
/// The one process-wide mutable static of this backend, deliberately: egui-wgpu's
/// `CallbackResources` outlive an install (they belong to the renderer, which a re-install on the
/// same context shares), so the id must be unique across installs, and no per-install owner can
/// provide that. `Arc` addresses would not do: a freed `Shared`'s address can be reused. It is a
/// write-only monotonic id source (never reset, never read as shared state), so it carries no
/// cross-install behaviour beyond uniqueness; `u64` cannot wrap in practice.
static NEXT_GENERATION: AtomicU64 = AtomicU64::new(1);

/// A layer painted this pass whose beneath content the plugin has not selected yet.
struct Pending {
    /// The `egui::PaintCallback::callback` object, matched by address in `FullOutput::shapes`.
    /// Holding the `Weak` keeps the allocation (not the value) alive, so the address cannot be
    /// reused by another callback while this entry exists.
    callback: Weak<dyn Any + Send + Sync>,
    /// Where the plan goes.
    slot: Arc<LayerSlot>,
    /// The layer's rect as painted (before layer transforms; the union of its entries), to
    /// derive the transform scale.
    rect: Rect,
    /// Per entry (R3), in list order: what planning needs of it.
    entries: Vec<PendingEntry>,
    /// The viewport whose pass painted the layer (selects its clear colour).
    viewport: egui::ViewportId,
    /// The textures the entries' shape masks sample (F9; the font atlas for text and
    /// pre-rasterized content, a mesh's own texture), resolved with the beneath textures.
    mask_textures: Vec<TextureId>,
}

/// One entry of a pending layer.
struct PendingEntry {
    /// `EntryJob::rect` (layer points, as painted).
    rect: Rect,
    /// `EntryJob::padding_points`.
    padding_points: f32,
    /// `EntryJob::textures`: resolved to bind groups when the layer is planned.
    textures: [Option<TextureId>; TEXTURE_SLOTS],
}

/// What the plugin hands to `prepare`: the beneath content of one layer, tessellated.
struct BeneathPlan {
    /// The filter's rect in global points (layer transform applied), as egui-wgpu will report it.
    callback_rect: Rect,
    /// The filter's clip rect in global points.
    clip_rect: Rect,
    /// `callback_rect.width() / LayerJob::rect.width()` (padding and radii scale with it).
    transform_scale: f32,
    /// Mesh primitives beneath the filter, clipped to the union of the entries' padded capture
    /// rects (one selection and tessellation for every entry).
    primitives: Vec<ClippedPrimitive>,
    /// A bind group for every texture the primitives reference (missing ones skipped) and for
    /// every texture of the layer's shape mask (all present, else the layer is not planned).
    textures: HashMap<TextureId, wgpu::BindGroup>,
    /// Premultiplied gamma-space colour the layer's viewport is cleared to (see `plugin.rs`).
    clear_color: [f32; 4],
    /// Per entry, the bind group of every attached user texture
    /// (`textures::resolve_user_textures`; a layer with an unresolvable one is not planned).
    user_textures: Vec<[Option<wgpu::BindGroup>; TEXTURE_SLOTS]>,
}

/// Where `prepare` left a layer's result, for `paint`: one composite per drawn entry, in the
/// order `plan::schedule` composites them (list order).
#[derive(Clone, Debug)]
struct Drawn {
    /// `Shared::generation` of the resources that hold it.
    generation: u64,
    /// The composites to draw, in order.
    entries: Vec<DrawnEntry>,
}

/// One entry's composite left for `paint`.
#[derive(Clone, Copy, Debug)]
struct DrawnEntry {
    /// Index of the entry in the layer's job.
    entry: usize,
    /// Index into `WgpuResources::pool`.
    pool_index: usize,
    /// The entry's processed region in screen pixels (the composite viewport).
    region: PxRect,
}

/// The per-layer hand-over cell.
#[derive(Default)]
struct SlotState {
    /// Filled by the plugin at the end of the pass, taken by `prepare`.
    plan: Option<BeneathPlan>,
    /// Filled by `prepare`, read by `paint`.
    drawn: Option<Drawn>,
}

/// One painted layer's cell, shared by its `Pending` entry and its callback.
#[derive(Default)]
struct LayerSlot(Mutex<SlotState>);

impl LayerSlot {
    /// Locks the cell, tolerating poisoning (the state is plain data, valid after any panic).
    fn lock(&self) -> MutexGuard<'_, SlotState> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// An effect whose pipeline failed to build on this device.
struct FailedEffect {
    /// Dies with the last `Effect` clone; the entry is then forgotten.
    module: Weak<CompiledModule>,
    /// What `failed_effects()` reports.
    failure: EffectFailure,
}

/// Health latches, written by the callbacks, read by `status()`.
#[derive(Default)]
struct Health {
    /// Backend-wide failure; once set nothing is drawn again.
    failure: Option<String>,
    /// `Some(float)` once GPU resources exist.
    float_intermediates: Option<bool>,
    /// Per-effect failures by `CompiledModule::id`.
    failed_effects: HashMap<u64, FailedEffect>,
    /// Attached user textures that could not be resolved, by id (egui never reuses an id).
    failed_textures: HashMap<TextureId, TextureFailure>,
    /// State formats the device cannot render / filter (F7), latched per format.
    failed_states: HashMap<StateFormat, StateFailure>,
}

/// Per-install "logged once" latches for debug messages that would otherwise repeat every frame.
///
/// Owned by `Shared`, so each install logs each message at most once; a re-install (a new
/// `Shared`) may log it again. Atomics rather than a `Mutex`: each flag is independent and only
/// ever set, and the callbacks run on egui-wgpu's render thread while the plugin runs on the GUI
/// thread.
#[derive(Debug, Default)]
struct LoggedOnce {
    /// A texture a beneath mesh samples could not be resolved (`textures.rs`).
    skipped_texture: AtomicBool,
    /// Paint callbacks beneath a filter were left out of its backdrop (`plugin.rs`).
    skipped_callbacks: AtomicBool,
    /// A layer reached `prepare` without a plan (`render.rs`).
    missing_plan: AtomicBool,
}

impl LoggedOnce {
    /// Sets `flag` and returns whether this call was the first to set it (i.e. whether to log).
    fn first(flag: &AtomicBool) -> bool {
        !flag.swap(true, Ordering::Relaxed)
    }
}

/// Install-wide state shared by the backend, the plugin and every callback.
pub(crate) struct Shared {
    /// Identifies the GPU resources built for this install.
    generation: u64,
    /// The render pass configuration.
    options: WgpuOptions,
    /// Device, queue, adapter, target format and egui's renderer (for texture bind groups).
    render_state: egui_wgpu::RenderState,
    /// Layers painted but not yet planned.
    pending: Mutex<Vec<Pending>>,
    /// Failure latches and status.
    health: Mutex<Health>,
    /// Textures uploaded in full that egui-wgpu has not applied yet (see `textures.rs`).
    texture_copies: Mutex<TextureCopies>,
    /// Log-once latches of this install.
    logged: LoggedOnce,
    /// Number of layer states the resources hold (mirrored after every prepare for
    /// `state_count()`, which runs on the GUI thread).
    state_count: AtomicUsize,
}

impl Shared {
    /// Locks the pending list (poison-tolerant: plain data).
    fn pending(&self) -> MutexGuard<'_, Vec<Pending>> {
        self.pending.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Locks the health latches (poison-tolerant: plain data).
    fn health(&self) -> MutexGuard<'_, Health> {
        self.health.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Locks the texture copies (poison-tolerant: GPU handles and plain data).
    fn texture_copies(&self) -> MutexGuard<'_, TextureCopies> {
        self.texture_copies.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Whether a backend-wide failure was latched.
    fn is_failed(&self) -> bool {
        self.health().failure.is_some()
    }

    /// Latches a backend-wide failure and logs it once.
    fn latch(&self, what: &str, error: &dyn std::fmt::Display) {
        let mut health = self.health();
        if health.failure.is_some() {
            return;
        }
        let message = format!("{what}: {error}");
        log::error!(
            "egui_shader_layers (wgpu): the filter backend failed and is disabled for this session.\n\
             Error: {message}\n\
             Effect: shader layers draw nothing extra; the UI underneath is unaffected.\n\
             Possible cause: the device rejected a shader, pipeline or texture format (for example \
             a device limit or a driver bug)."
        );
        health.failure = Some(message);
    }

    /// Records that `module` failed to build and logs it once.
    fn fail_effect(&self, module: &Arc<CompiledModule>, error: &str) {
        let mut health = self.health();
        if health.failed_effects.contains_key(&module.id) {
            return;
        }
        log::error!(
            "egui_shader_layers (wgpu): effect `{}` could not be built and is disabled; layers using it are skipped.\n\
             Error: {error}\n\
             Possible cause: the device rejected the effect's WGSL (naga accepted it, so this is a device limit or a driver bug).",
            module.label
        );
        let failure = EffectFailure { label: module.label.to_string(), error: error.to_owned() };
        health.failed_effects.insert(module.id, FailedEffect { module: Arc::downgrade(module), failure });
    }

    /// Records that the attached user texture `id` could not be resolved and logs it once.
    fn fail_texture(&self, id: TextureId) {
        let mut health = self.health();
        if health.failed_textures.contains_key(&id) {
            return;
        }
        let error = "egui-wgpu's renderer has no texture with this id (freed, or never registered with this renderer)".to_owned();
        log::error!(
            "egui_shader_layers (wgpu): a shader layer attaches texture {id:?}, which cannot be resolved; layers attaching it draw nothing extra.\n\
             Error: {error}\n\
             Possible cause: the TextureHandle was dropped (egui freed the texture) while the layer still attaches its id."
        );
        health.failed_textures.insert(id, TextureFailure { id, error });
    }

    /// Records that stateful layers asking for `format` are refused (the device cannot render or
    /// filter it) and logs it once per format.
    fn fail_state(&self, format: StateFormat, reason: String) {
        let mut health = self.health();
        if health.failed_states.contains_key(&format) {
            return;
        }
        let error = StateError::UnsupportedFormat { format, reason };
        log::error!("egui_shader_layers (wgpu): stateful shader layers are skipped (they draw nothing extra).\nError: {error}");
        health.failed_states.insert(format, StateFailure { format, error });
    }

    /// Whether `module` is known to fail on this device.
    fn effect_failed(&self, module: &CompiledModule) -> bool {
        self.health().failed_effects.contains_key(&module.id)
    }
}

/// The wgpu backend, registered with the context by the crate's registry plugin.
pub(crate) struct WgpuBackend {
    /// Shared with the plugin and every callback.
    shared: Arc<Shared>,
}

impl Backend for WgpuBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Wgpu
    }

    fn status(&self) -> BackendStatus {
        let health = self.shared.health();
        if let Some(error) = &health.failure {
            return BackendStatus::Failed { backend: BackendKind::Wgpu, error: error.clone() };
        }
        match health.float_intermediates {
            Some(float_intermediates) => BackendStatus::Active { backend: BackendKind::Wgpu, float_intermediates },
            None => BackendStatus::Installed { backend: BackendKind::Wgpu },
        }
    }

    fn failed_effects(&self) -> Vec<EffectFailure> {
        let mut health = self.shared.health();
        health.failed_effects.retain(|_, entry| entry.module.strong_count() > 0);
        health.failed_effects.values().map(|entry| entry.failure.clone()).collect()
    }

    fn failed_textures(&self) -> Vec<TextureFailure> {
        self.shared.health().failed_textures.values().cloned().collect()
    }

    fn failed_states(&self) -> Vec<StateFailure> {
        self.shared.health().failed_states.values().cloned().collect()
    }

    fn state_count(&self) -> usize {
        self.shared.state_count.load(Ordering::Relaxed)
    }

    fn paint_layer(&self, painter: &egui::Painter, job: LayerJob) {
        if self.shared.is_failed() {
            return;
        }
        let slot = Arc::new(LayerSlot::default());
        let rect = job.rect;
        let entries = job.entries.iter().map(|entry| PendingEntry { rect: entry.rect, padding_points: entry.padding_points, textures: entry.textures }).collect();
        let mut mask_textures: Vec<TextureId> = job.entries.iter().flat_map(|entry| entry.mask.textures()).collect();
        mask_textures.sort_unstable();
        mask_textures.dedup();
        // Read during the pass: after `end_pass` the context reports the parent viewport.
        let viewport = painter.ctx().viewport_id();
        let callback = FilterCallback::new(Arc::clone(&self.shared), Arc::clone(&slot), job);
        let paint_callback = egui_wgpu::Callback::new_paint_callback(rect, callback);
        let weak = Arc::downgrade(&paint_callback.callback);
        {
            let mut pending = self.shared.pending();
            // Entries whose callback died without being planned (discarded pass, other viewport
            // that never ended) are dropped here as well as in the plugin.
            pending.retain(|entry| entry.callback.strong_count() > 0);
            pending.push(Pending { callback: weak, slot, rect, entries, viewport, mask_textures });
        }
        painter.add(paint_callback);
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Installs the wgpu filter backend on `ctx`. Call once at startup, e.g. from the eframe app
/// creator with `cc.wgpu_render_state`.
///
/// **The content under every shader layer is rendered twice per frame on wgpu.** egui-wgpu draws
/// into a surface that cannot be read back mid-frame, so this backend re-renders everything
/// painted beneath each shader layer (clipped to the layer's padded rect) into an offscreen
/// texture, filters that, and draws the result over the region. Budget for the extra
/// tessellation and fill; a one-time warning is logged here.
///
/// Semantics specific to this backend:
/// * Paint callbacks beneath a filter (your own 3D views, other shader layers) cannot be
///   re-rendered and are left out of its backdrop. A filter painted over another filter
///   therefore sees the lower filter's *unfiltered* backdrop. Outside a rounded mask (the corners
///   of the rect) the filter draws nothing, so what is on screen there — including such
///   callbacks — stays visible.
/// * On an `*Srgb` target the backdrop is re-rendered with egui's linear blending where the
///   device supports texture view formats; GL / WebGL2 do not, and there semi-transparent edges
///   under a filter blend in gamma (logged once).
/// * Textures uploaded in full during the current frame (new textures, the font atlas after it
///   grew) are copied once more for the backdrop, since egui-wgpu applies them only after the
///   frame's last pass. The copy is shared by all layers and passes of the frame and follows later
///   partial updates (glyphs rasterised in a later pass of a multi-pass frame).
/// * Only [`egui::Context::run_ui`] / [`egui::Context::end_pass`] integrations are supported
///   (the beneath content is collected by an egui plugin at the end of every pass); eframe does this.
///
/// `options` MUST describe egui's render pass (see [`WgpuOptions`]): a mismatch is not
/// detectable here and makes wgpu reject egui's whole frame when its encoder is finished — a
/// panic natively and on WebGL2, a console error and a lost frame on browser WebGPU.
///
/// # Errors
/// [`InstallError::Unsupported`] when `options` cannot describe a valid render pass (sample
/// count not a power of two, or a depth format that is not a depth/stencil format). The backend
/// is then registered in a failed state so [`crate::status`] explains it.
pub fn install_wgpu(ctx: &egui::Context, render_state: &egui_wgpu::RenderState, options: WgpuOptions) -> Result<(), InstallError> {
    let depth_stencil_format = options.depth_stencil_format();
    let problem = if !options.msaa_samples().is_power_of_two() || options.msaa_samples() > 32 {
        Some(format!("MSAA sample count {} is not a power of two up to 32", options.msaa_samples))
    } else if depth_stencil_format.is_some_and(|format| !format.is_depth_stencil_format()) {
        Some(format!("{depth_stencil_format:?} is not a depth/stencil format"))
    } else {
        None
    };
    let shared = Arc::new(Shared {
        generation: NEXT_GENERATION.fetch_add(1, Ordering::Relaxed),
        options,
        render_state: render_state.clone(),
        pending: Mutex::new(Vec::new()),
        health: Mutex::new(Health { failure: problem.clone(), ..Health::default() }),
        texture_copies: Mutex::new(TextureCopies::default()),
        logged: LoggedOnce::default(),
        state_count: AtomicUsize::new(0),
    });
    if let Some(handle) = ctx.plugin_opt::<ShaderLayersPlugin>() {
        handle.lock().replace_shared(Arc::clone(&shared));
    } else {
        ctx.add_plugin(ShaderLayersPlugin::new(Arc::clone(&shared)));
    }
    register(ctx, Arc::new(WgpuBackend { shared }));
    if let Some(reason) = problem {
        log::error!("egui_shader_layers (wgpu): invalid WgpuOptions, shader layers are disabled: {reason}");
        return Err(InstallError::Unsupported { backend: BackendKind::Wgpu, reason });
    }
    log::warn!(
        "egui_shader_layers (wgpu): on wgpu the content beneath every shader layer is rendered TWICE per frame \
         (once by egui, once offscreen for the filter); target format {:?}, MSAA {}",
        render_state.target_format,
        options.msaa_samples()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depth_and_stencil_bits_compose_in_any_order_like_eframe() {
        let a = WgpuOptions::default().with_depth_buffer(24).with_stencil_buffer(8);
        let b = WgpuOptions::default().with_stencil_buffer(8).with_depth_buffer(24);
        assert_eq!(a.depth_stencil_format(), Some(wgpu::TextureFormat::Depth24PlusStencil8));
        assert_eq!(a, b);
        assert_eq!(WgpuOptions::default().with_depth_buffer(32).depth_stencil_format(), Some(wgpu::TextureFormat::Depth32Float));
        assert_eq!(WgpuOptions::default().with_stencil_buffer(8).depth_stencil_format(), Some(wgpu::TextureFormat::Stencil8));
        // A combination eframe does not know gives no attachment, as in eframe.
        assert_eq!(WgpuOptions::default().with_depth_buffer(12).depth_stencil_format(), None);
    }

    #[test]
    fn the_default_is_eframes_and_egui_wgpus_default() {
        let options = WgpuOptions::default();
        assert_eq!(options, WgpuOptions::from(&egui_wgpu::RendererOptions::default()));
        assert_eq!(options.msaa_samples(), 1);
        assert_eq!(options.depth_stencil_format(), None);
        assert!(options.dithering());
    }

    #[test]
    fn renderer_options_convert_exactly() {
        let renderer = egui_wgpu::RendererOptions { msaa_samples: 4, depth_stencil_format: Some(wgpu::TextureFormat::Depth32FloatStencil8), dithering: false, predictable_texture_filtering: true };
        let options = WgpuOptions::from(&renderer);
        assert_eq!(options.msaa_samples(), 4);
        assert_eq!(options.depth_stencil_format(), Some(wgpu::TextureFormat::Depth32FloatStencil8));
        assert!(!options.dithering());
        let eframe_like = WgpuOptions::default().with_multisampling(4).with_depth_buffer(32).with_stencil_buffer(8).with_dithering(false);
        assert_eq!(options, eframe_like, "bits and an explicit format describing the same pass are equal");
    }

    #[test]
    fn the_last_depth_setter_wins() {
        let format_last = WgpuOptions::default().with_depth_buffer(24).with_depth_stencil_format(Some(wgpu::TextureFormat::Depth16Unorm));
        assert_eq!(format_last.depth_stencil_format(), Some(wgpu::TextureFormat::Depth16Unorm));
        let bits_last = format_last.with_depth_buffer(32);
        assert_eq!(bits_last.depth_stencil_format(), Some(wgpu::TextureFormat::Depth32Float), "stencil bits reset to 0 after an explicit format");
        assert_eq!(WgpuOptions::default().with_depth_stencil_format(None).depth_stencil_format(), None);
    }

    #[test]
    fn multisampling_zero_and_one_both_mean_none_and_clear_colour_is_kept() {
        assert_eq!(WgpuOptions::default().with_multisampling(0), WgpuOptions::default().with_multisampling(1));
        assert_eq!(WgpuOptions::default().with_multisampling(8).msaa_samples(), 8);
        let clear = [0.1, 0.2, 0.3, 0.4];
        assert_eq!(WgpuOptions::default().with_clear_color(clear).clear_color().map(f32::to_bits), clear.map(f32::to_bits));
        assert_ne!(WgpuOptions::default().with_clear_color(clear), WgpuOptions::default());
    }
}
