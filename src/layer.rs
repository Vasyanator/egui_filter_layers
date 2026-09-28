/*
File: layer.rs

Purpose:
`FilterLayer`, the user-facing shape: a rect (+ optional rounded mask) and a `FilterChain`,
painted into an `egui::Painter`. Resolves the chain into a backend-neutral `LayerJob` and hands
it to whichever backend is installed on the painter's context.

Key structures:
- `FilterLayer`.

Key functions:
- `FilterLayer::paint()`.

Notes:
Never panics and never draws a fake: with no backend installed it logs one warning per
`egui::Context` (latched in the context's temporary memory, see `warn_not_installed_once`) and
paints nothing, leaving the backdrop unfiltered. A failed backend is skipped the same way, and
neither case requests repaints for animated chains.
*/

use egui::{Context, CornerRadius, Id, Painter, Rect};

use crate::backend::{LayerJob, installed};
use crate::pass::FilterChain;

/// Sets `ctx`'s "no backend installed" latch; returns whether it was unset (i.e. whether to warn).
fn first_not_installed_warning(ctx: &Context) -> bool {
    ctx.data_mut(|data| {
        let warned = data.get_temp_mut_or_default::<bool>(Id::new("egui_filter_layers::warned_not_installed"));
        !std::mem::replace(warned, true)
    })
}

/// Logs the "no backend installed" warning the first time it is called for `ctx`.
///
/// The latch lives in `ctx`'s temporary memory (not a process-wide static): there is no install
/// to own it, and each `Context` is its own UI. It is not persisted, so a restored or reset
/// memory (`Memory` replaced by the integration) may warn again.
fn warn_not_installed_once(ctx: &Context) {
    if first_not_installed_warning(ctx) {
        log::warn!(
            "egui_filter_layers: a FilterLayer was painted but no backend is installed on this egui::Context; \
             call egui_filter_layers::install_glow (or the wgpu installer) at startup. Layers draw nothing until then."
        );
    }
}

/// A filter applied to everything painted beneath `rect`.
///
/// "Beneath" means earlier in egui's paint order: lower layers and the shapes painted earlier into
/// the same layer. Anything painted after the layer (later shapes, higher `Order` areas such as
/// popups and tooltips) is drawn on top, unfiltered.
///
/// ```no_run
/// # fn demo(ui: &mut egui::Ui, rect: egui::Rect) {
/// use egui_filter_layers::{FilterLayer, presets};
/// FilterLayer::new(rect, presets::gaussian_blur(6.0))
///     .corner_radius(12)
///     .paint(ui.painter());
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct FilterLayer {
    /// The filtered rect, in points.
    rect: Rect,
    /// The passes to run.
    chain: FilterChain,
    /// Rounded-corner mask; outside it the backdrop is left untouched.
    corner_radius: CornerRadius,
}

impl FilterLayer {
    /// A layer filtering `rect` (points) through `chain` (a `Pass`, `Vec<Pass>` or `FilterChain`).
    #[must_use]
    pub fn new(rect: Rect, chain: impl Into<FilterChain>) -> Self {
        Self { rect, chain: chain.into(), corner_radius: CornerRadius::ZERO }
    }

    /// Restricts the filter to a rounded rect (anti-aliased edge).
    #[must_use]
    pub fn corner_radius(mut self, corner_radius: impl Into<CornerRadius>) -> Self {
        self.corner_radius = corner_radius.into();
        self
    }

    /// The filtered rect, in points.
    #[must_use]
    pub fn rect(&self) -> Rect {
        self.rect
    }

    /// The chain this layer runs.
    #[must_use]
    pub fn chain(&self) -> &FilterChain {
        &self.chain
    }

    /// Adds the layer to `painter` at the current position in its paint list.
    ///
    /// Does nothing for an empty chain or a non-positive rect. Requests a repaint when the chain
    /// is animated. With no backend installed on the painter's context, logs a warning once per
    /// context and paints nothing; with a failed backend, paints nothing (the failure was logged
    /// when it happened). Neither case requests repaints — see [`crate::status`].
    pub fn paint(self, painter: &Painter) {
        if self.chain.is_empty() || !self.rect.is_positive() {
            return;
        }
        let ctx = painter.ctx();
        let Some(backend) = installed(ctx) else {
            warn_not_installed_once(ctx);
            return;
        };
        // A failed backend would ignore the job anyway; skipping here also stops an animated
        // chain from forcing a repaint every frame for nothing.
        if !backend.status().is_usable() {
            return;
        }
        // An effect reading `fl.time` must keep moving in an otherwise idle app.
        if self.chain.is_animated() {
            ctx.request_repaint();
        }
        // f64 seconds -> f32: effects only need a smooth phase, and f32 keeps millisecond
        // precision for ~2.3 hours of uptime, which animated effects tolerate.
        #[allow(clippy::cast_possible_truncation)] // Intentional precision reduction, see above.
        let time = ctx.input(|input| input.time) as f32;
        let padding_points = self.chain.padding_points();
        let job = LayerJob { rect: self.rect, corner_radius: self.corner_radius, passes: self.chain.into_passes(), padding_points, time };
        backend.paint_layer(painter, job);
    }
}

#[cfg(test)]
mod tests {
    use std::any::Any;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use egui::{LayerId, pos2, vec2};

    use super::*;
    use crate::backend::{Backend, BackendKind, BackendStatus, EffectFailure, register};
    use crate::presets;

    /// A backend with a fixed status that counts `paint_layer` calls.
    struct Counting {
        status: BackendStatus,
        painted: AtomicUsize,
    }

    impl Backend for Counting {
        fn kind(&self) -> BackendKind {
            BackendKind::Glow
        }
        fn status(&self) -> BackendStatus {
            self.status.clone()
        }
        fn failed_effects(&self) -> Vec<EffectFailure> {
            Vec::new()
        }
        fn paint_layer(&self, _painter: &Painter, _job: LayerJob) {
            self.painted.fetch_add(1, Ordering::Relaxed);
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    /// Paints one animated layer per pass on a fresh context with `status`, for a few passes
    /// (egui itself repaints during the first ones); returns (paint calls, whether the LAST pass
    /// asked for an immediate repaint).
    fn paint_animated_with(status: BackendStatus) -> (usize, bool) {
        const PASSES: usize = 4;
        let ctx = Context::default();
        let backend = Arc::new(Counting { status, painted: AtomicUsize::new(0) });
        register(&ctx, Arc::clone(&backend) as Arc<dyn Backend>);
        let mut repaint = false;
        for _ in 0..PASSES {
            let mut output = ctx.run_ui(egui::RawInput::default(), |ui| {
                let painter = Painter::new(ui.ctx().clone(), LayerId::background(), Rect::EVERYTHING);
                FilterLayer::new(Rect::from_min_size(pos2(0.0, 0.0), vec2(10.0, 10.0)), presets::ripple(2.0, 10.0, 1.0)).paint(&painter);
            });
            repaint = output.viewport_output.get(&egui::ViewportId::ROOT).is_some_and(|viewport| viewport.repaint_delay.is_zero());
            // The font atlas upload is irrelevant here; epaint asserts deltas are handled.
            output.textures_delta.clear();
        }
        (backend.painted.load(Ordering::Relaxed) / PASSES, repaint)
    }

    #[test]
    fn a_failed_backend_gets_no_jobs_and_no_repaints() {
        let (painted, repaint) = paint_animated_with(BackendStatus::Failed { backend: BackendKind::Glow, error: "test".to_owned() });
        assert_eq!(painted, 0);
        assert!(!repaint, "an animated chain must not spin the event loop on a failed backend");

        let (painted, repaint) = paint_animated_with(BackendStatus::Installed { backend: BackendKind::Glow });
        assert_eq!(painted, 1);
        assert!(repaint);
    }

    /// The "no backend installed" latch is per `Context`: set once per context, independent between contexts.
    #[test]
    fn not_installed_warning_latches_per_context() {
        let first = Context::default();
        let second = Context::default();
        assert!(first_not_installed_warning(&first));
        assert!(!first_not_installed_warning(&first));
        assert!(first_not_installed_warning(&second));
    }
}
