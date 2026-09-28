/*
File: backend.rs

Purpose:
The crate-internal backend boundary and the per-`egui::Context` backend registry.

Key structures:
- `Backend` (crate-internal trait): what a GPU backend must provide. Each backend module
  (`glow_backend`, and the `wgpu` backend) implements it; core code never names a backend type.
- `LayerJob` (crate-internal): one `FilterLayer::paint` call, resolved for the backend.
- `FrameInput` / `PointerInput` (crate-internal): the per-frame input a layer reads on the GUI
  thread (`fl.delta_time`, `fl.frame`, `fl.pointer`), resolved per capture by `FrameInput::resolve`.
- `BackendKind`, `BackendStatus`, `EffectFailure`, `InstallError` (public).
- `TextureError` (public): a layer's user textures cannot serve its effects (checked before any
  backend work); `TextureFailure` (public): an attached texture a backend could not resolve.

Key functions:
- `register()` / `installed()` / `uninstall()` (crate-internal): the per-context registry.
- `status()`, `failed_effects()`, `failed_textures()` (public, re-exported at the crate root).

Notes:
The registry is one `Arc<dyn Backend>` held by a hook-less egui plugin (`Registry`), so any code
holding the `Context` (including `FilterLayer::paint`) can find it without app plumbing. A plugin
lives in the `Context` itself, not in `Memory`, so resetting egui memory (a common "reset UI
state" idiom) does not silently uninstall the backend.
*/

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use egui::{Context, CornerRadius, Painter, Pos2, Rect, TextureId};

use crate::effect::TEXTURE_SLOTS;
use crate::geometry::{PxRect, pointer_uniform};
use crate::pass::{InputUniforms, Pass};

/// Which GPU backend a status refers to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BackendKind {
    /// `egui_glow` (OpenGL 3.3+, OpenGL ES 3.0+, WebGL2).
    Glow,
    /// `egui-wgpu`.
    Wgpu,
}

impl fmt::Display for BackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Glow => "glow",
            Self::Wgpu => "wgpu",
        })
    }
}

/// The state of the filter backend installed on a context. See [`crate::status`].
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum BackendStatus {
    /// No backend is installed on this context (never installed, or removed by `destroy_glow`):
    /// filter layers draw nothing extra.
    NotInstalled,
    /// Installed; GPU resources are created on the first painted layer.
    Installed {
        /// The installed backend.
        backend: BackendKind,
    },
    /// Resources exist and layers are being filtered.
    Active {
        /// The installed backend.
        backend: BackendKind,
        /// Whether intermediate pass textures are half-float (`true`) or 8-bit (`false`).
        float_intermediates: bool,
    },
    /// The backend failed permanently for this session; layers draw nothing extra.
    Failed {
        /// The installed backend.
        backend: BackendKind,
        /// Why (also logged once at `error` level).
        error: String,
    },
}

impl BackendStatus {
    /// Whether layers can currently be filtered (installed and not failed).
    #[must_use]
    pub fn is_usable(&self) -> bool {
        matches!(self, Self::Installed { .. } | Self::Active { .. })
    }
}

impl fmt::Display for BackendStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotInstalled => f.write_str("no filter backend installed"),
            Self::Installed { backend } => write!(f, "{backend}: installed (resources not created yet)"),
            Self::Active { backend, float_intermediates } => {
                write!(f, "{backend}: active, {} intermediates", if *float_intermediates { "RGBA16F" } else { "RGBA8" })
            }
            Self::Failed { backend, error } => write!(f, "{backend}: FAILED: {error}"),
        }
    }
}

/// An effect the installed backend could not compile or run. Layers using it are skipped.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct EffectFailure {
    /// The effect's label.
    pub label: String,
    /// The backend's error (e.g. the driver's shader log).
    pub error: String,
}

/// Why a layer's user textures cannot serve its effects. Checked before any GPU work:
/// [`crate::FilterLayer::check_textures`] (what `FilterLayer::paint` runs; the layer is then
/// skipped and the error logged once per context) and the offline renderer
/// (`OfflineError::Texture`).
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum TextureError {
    /// A texture was attached at a slot that does not exist.
    #[error("texture slot {slot} does not exist; the slots are 0..{TEXTURE_SLOTS}")]
    InvalidSlot {
        /// The slot given.
        slot: usize,
    },
    /// An effect of the chain reads a slot (`tex`, `tex_size`, `tex_load`; see
    /// [`crate::Effect::reads_texture`]) where nothing is attached.
    #[error("effect `{effect}` (pass {pass}) reads texture slot {slot}, but no texture is attached there")]
    NotAttached {
        /// The effect's label.
        effect: String,
        /// Index of its pass in the chain.
        pass: usize,
        /// The slot it reads.
        slot: usize,
    },
}

/// An attached user texture the installed backend could not resolve when it painted a layer:
/// the `TextureId` is unknown to egui's renderer (freed, or never allocated on this painter).
/// Layers attaching it draw nothing extra. egui never reuses a `TextureId`, so the entry stays
/// for the backend's lifetime.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct TextureFailure {
    /// The texture.
    pub id: TextureId,
    /// Why it could not be used.
    pub error: String,
}

/// Why a backend could not be installed.
#[derive(Clone, Debug, thiserror::Error)]
#[non_exhaustive]
pub enum InstallError {
    /// The graphics context cannot run the backend (e.g. `WebGL1`, OpenGL < 3.3).
    #[error("the {backend} filter backend is not supported by this graphics context: {reason}")]
    Unsupported {
        /// The backend that was being installed.
        backend: BackendKind,
        /// What is missing.
        reason: String,
    },
}

/// One `FilterLayer::paint` call, resolved and handed to the backend.
#[derive(Clone, Debug)]
pub(crate) struct LayerJob {
    /// The layer's rect in points (painter space, before layer transforms).
    pub rect: Rect,
    /// Rounded-corner mask, in points.
    pub corner_radius: CornerRadius,
    /// The passes to run, in order (never empty).
    pub passes: Vec<Pass>,
    /// Total capture padding in points (`FilterChain::padding_points`).
    pub padding_points: f32,
    /// egui input time in seconds at paint time, for `fl.time`.
    pub time: f32,
    /// Per-frame input read at paint time (`fl.delta_time`, `fl.frame`, `fl.pointer`).
    pub frame_input: FrameInput,
    /// The user texture attached at each slot (`FilterLayer::texture`); every slot an effect of
    /// `passes` reads is `Some` (checked by `plan::check_textures` before the job is built).
    pub textures: [Option<TextureId>; TEXTURE_SLOTS],
}

/// The pointer as the GUI thread saw it when the layer was painted.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PointerInput {
    /// Position in screen points (egui's global, untransformed input coordinates).
    pub pos: Pos2,
    /// Whether the primary button is down.
    pub pressed: bool,
}

/// The per-frame input of one layer, captured on the GUI thread (`FilterLayer::paint`) and
/// resolved against the capture at paint time (`resolve`). Per context, not per layer: every
/// layer painted in one frame sees the same values. `Default` = no input (0 s, frame 0, no pointer).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct FrameInput {
    /// Seconds since the previous frame (`fl.delta_time`): egui's `stable_dt`, clamped to
    /// `0..=0.1` (a non-finite value reads as 0).
    pub delta_time: f32,
    /// `Context::cumulative_frame_nr` truncated to its low 32 bits (`fl.frame`, wrapping).
    pub frame: u32,
    /// The pointer, `None` when egui has none (off the window; a released touch).
    pub pointer: Option<PointerInput>,
}

impl FrameInput {
    /// The largest `fl.delta_time` in the UI, seconds: a frame after a long idle sleep would
    /// otherwise make simulations jump.
    pub(crate) const MAX_DELTA_TIME: f32 = 0.1;

    /// The input members of the uniform block for a capture `capture` on a screen of
    /// `pixels_per_point` (the paint callback's), through the one conversion
    /// `geometry::pointer_uniform`.
    #[must_use]
    pub(crate) fn resolve(&self, pixels_per_point: f32, capture: PxRect) -> InputUniforms {
        InputUniforms { delta_time: self.delta_time, frame: self.frame, pointer: pointer_uniform(self.pointer.map(|pointer| (pointer.pos, pointer.pressed)), pixels_per_point, capture) }
    }
}

/// The crate-internal backend boundary.
///
/// A backend turns a [`LayerJob`] into whatever it paints (typically one `egui::Shape::Callback`
/// added through `painter`; a backend that must find the filter in egui's final paint list keys
/// it by that callback's `Arc` identity), and reports its health.
pub(crate) trait Backend: Send + Sync + 'static {
    /// Which backend this is.
    fn kind(&self) -> BackendKind;
    /// Current status (cheap; called by apps every frame).
    fn status(&self) -> BackendStatus;
    /// Effects this backend failed to build.
    fn failed_effects(&self) -> Vec<EffectFailure>;
    /// Attached user textures this backend could not resolve.
    fn failed_textures(&self) -> Vec<TextureFailure>;
    /// Paints one filter layer. Must never panic; failures are latched and logged.
    fn paint_layer(&self, painter: &Painter, job: LayerJob);
    /// Downcasting hook for backend-specific teardown (e.g. `destroy_glow`).
    #[cfg_attr(not(feature = "glow"), allow(dead_code))] // Only `destroy_glow` downcasts; wgpu needs no teardown.
    fn as_any(&self) -> &dyn Any;
}

/// The per-context registry: an egui plugin with no hooks, used only as storage that lives as
/// long as the `Context` and survives `Memory` resets. Its lock is held only for the clone /
/// swap of the `Arc` below, never while a backend method runs.
#[derive(Default)]
struct Registry {
    /// The installed backend, if any.
    backend: Option<Arc<dyn Backend>>,
}

impl egui::plugin::Plugin for Registry {
    fn debug_name(&self) -> &'static str {
        "egui_filter_layers::registry"
    }
}

/// Installs `backend` on `ctx`, replacing (and logging) any previous one.
///
/// Replacing does not free the previous backend's GPU objects; an installer that can free them
/// (`install_glow` replacing a glow backend) does so before calling this.
pub(crate) fn register(ctx: &Context, backend: Arc<dyn Backend>) {
    let kind = backend.kind();
    let previous = ctx.plugin_or_default::<Registry>().lock().backend.replace(backend);
    if let Some(previous) = previous {
        log::info!("egui_filter_layers: replaced the installed {} backend with a {kind} backend", previous.kind());
    } else {
        log::info!("egui_filter_layers: {kind} backend installed");
    }
}

/// The backend installed on `ctx`, if any.
pub(crate) fn installed(ctx: &Context) -> Option<Arc<dyn Backend>> {
    ctx.with_plugin(|registry: &mut Registry| registry.backend.clone()).flatten()
}

/// Removes and returns the backend installed on `ctx`.
#[cfg_attr(not(feature = "glow"), allow(dead_code))] // Only `destroy_glow` calls it; wgpu resources are freed with the renderer.
pub(crate) fn uninstall(ctx: &Context) -> Option<Arc<dyn Backend>> {
    ctx.with_plugin(|registry: &mut Registry| registry.backend.take()).flatten()
}

/// The status of the filter backend installed on `ctx`.
///
/// Show it in a debug overlay or settings page: filter layers never panic, so a context whose
/// backend failed (or was never installed) simply renders unfiltered, and this is where to see why.
#[must_use]
pub fn status(ctx: &Context) -> BackendStatus {
    installed(ctx).map_or(BackendStatus::NotInstalled, |backend| backend.status())
}

/// Effects the installed backend failed to build on the GPU (after they passed naga validation),
/// e.g. because a driver rejected the translated shader. Empty when none failed or none installed.
#[must_use]
pub fn failed_effects(ctx: &Context) -> Vec<EffectFailure> {
    installed(ctx).map_or_else(Vec::new, |backend| backend.failed_effects())
}

/// Attached user textures (`FilterLayer::texture`) the installed backend could not resolve when
/// painting (freed or foreign `TextureId`s); layers using them draw nothing extra. Each is also
/// logged once at `error` level. Empty when none failed or none installed.
#[must_use]
pub fn failed_textures(ctx: &Context) -> Vec<TextureFailure> {
    installed(ctx).map_or_else(Vec::new, |backend| backend.failed_textures())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fresh_context_reports_not_installed() {
        let ctx = Context::default();
        assert_eq!(status(&ctx), BackendStatus::NotInstalled);
        assert!(failed_effects(&ctx).is_empty());
        assert!(failed_textures(&ctx).is_empty());
        assert!(!status(&ctx).is_usable());
    }

    /// A minimal backend for registry tests (never paints).
    struct Dummy;

    impl Backend for Dummy {
        fn kind(&self) -> BackendKind {
            BackendKind::Glow
        }
        fn status(&self) -> BackendStatus {
            BackendStatus::Installed { backend: BackendKind::Glow }
        }
        fn failed_effects(&self) -> Vec<EffectFailure> {
            Vec::new()
        }
        fn failed_textures(&self) -> Vec<TextureFailure> {
            Vec::new()
        }
        fn paint_layer(&self, _painter: &Painter, _job: LayerJob) {}
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    #[test]
    fn the_registry_survives_a_memory_reset_and_uninstall_removes_it() {
        let ctx = Context::default();
        register(&ctx, Arc::new(Dummy));
        assert_eq!(status(&ctx), BackendStatus::Installed { backend: BackendKind::Glow });
        ctx.memory_mut(|memory| *memory = egui::Memory::default());
        ctx.data_mut(egui::util::IdTypeMap::clear);
        assert!(status(&ctx).is_usable(), "a memory reset must not uninstall the backend");
        assert!(uninstall(&ctx).is_some());
        assert_eq!(status(&ctx), BackendStatus::NotInstalled);
        assert!(uninstall(&ctx).is_none());
    }

    #[test]
    fn status_display_mentions_the_backend() {
        let failed = BackendStatus::Failed { backend: BackendKind::Glow, error: "boom".into() };
        assert_eq!(failed.to_string(), "glow: FAILED: boom");
        assert!(BackendStatus::Active { backend: BackendKind::Wgpu, float_intermediates: true }.to_string().contains("RGBA16F"));
    }
}
