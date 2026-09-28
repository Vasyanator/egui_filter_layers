/*
File: backend.rs

Purpose:
The crate-internal backend boundary and the per-`egui::Context` backend registry.

Key structures:
- `Backend` (crate-internal trait): what a GPU backend must provide. Each backend module
  (`glow_backend`, and the `wgpu` backend) implements it; core code never names a backend type.
- `LayerJob` (crate-internal): one `FilterLayer::paint` call, resolved for the backend.
- `BackendKind`, `BackendStatus`, `EffectFailure`, `InstallError` (public).

Key functions:
- `register()` / `installed()` / `uninstall()` (crate-internal): the per-context registry.
- `status()`, `failed_effects()` (public, re-exported at the crate root).

Notes:
The registry is one `Arc<dyn Backend>` held by a hook-less egui plugin (`Registry`), so any code
holding the `Context` (including `FilterLayer::paint`) can find it without app plumbing. A plugin
lives in the `Context` itself, not in `Memory`, so resetting egui memory (a common "reset UI
state" idiom) does not silently uninstall the backend.
*/

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use egui::{Context, CornerRadius, Painter, Rect};

use crate::pass::Pass;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fresh_context_reports_not_installed() {
        let ctx = Context::default();
        assert_eq!(status(&ctx), BackendStatus::NotInstalled);
        assert!(failed_effects(&ctx).is_empty());
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
