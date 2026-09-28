/*
File: wgpu_backend/plugin.rs

Purpose:
The egui `Plugin` that, at the end of every pass, turns each pending filter layer into a
`BeneathPlan`: the tessellated content painted beneath it plus egui-wgpu's bind groups for the
textures that content samples.

Key structures:
- `FilterLayersPlugin`: registered once per `egui::Context` by `install_wgpu`.

Key functions:
- `FilterLayersPlugin::output_hook()`: runs after egui flattened the pass's shapes.

Notes:
`output_hook` (egui `Context::end_pass` -> `on_output`) sees `FullOutput::shapes`, which is
egui's final paint order with layer transforms applied — so "beneath" is exact for the frame
being painted (no stale layer order), and it runs for `run_ui` and manual `begin_pass` /
`end_pass` integrations alike. The egui context lock is NOT held there (`ctx.tessellate` and
`ctx.fonts` lock it themselves). Texture resolution lives in `textures.rs`; every pass's texture
delta is recorded there, including passes that are not planned.

A pass that requested a discard and that `Context::run_ui` will therefore repeat is not planned
(its shapes are never painted). An integration that drives passes by hand AND paints a pass that
requested a discard gets no filter output for that pass.

The capture is cleared like eframe clears the viewport the layer was painted in: immediate
viewports to transparent, all others to `WgpuOptions::clear_color()`.

The layer's attached user textures (F8) are resolved here too, at the end of the pass that
painted it, exactly like the textures of its beneath meshes (`textures.rs`); a layer attaching an
id that cannot be resolved is not planned (it draws nothing) and the id is reported once
(`Shared::fail_texture`). The plan holds the bind groups (`Arc`-backed handles), so the textures
stay alive until the frame is rendered even if egui frees them afterwards.
*/

use std::sync::Arc;

use egui::{FullOutput, Shape};

use super::beneath::{capture_points, find_callback, select_beneath, strip_callbacks};
use super::textures::{resolve_user_textures, snapshot_textures};
use super::{BeneathPlan, LoggedOnce, Pending, Shared};
use crate::geometry::transform_scale;

/// Collects the beneath content of every pending filter layer at the end of each egui pass.
pub(crate) struct FilterLayersPlugin {
    /// The state of the current install (replaced when `install_wgpu` is called again).
    shared: Arc<Shared>,
}

impl FilterLayersPlugin {
    /// A plugin serving `shared`.
    pub(super) fn new(shared: Arc<Shared>) -> Self {
        Self { shared }
    }

    /// Points the (already registered) plugin at a new install.
    pub(super) fn replace_shared(&mut self, shared: Arc<Shared>) {
        self.shared = shared;
    }
}

impl egui::Plugin for FilterLayersPlugin {
    fn debug_name(&self) -> &'static str {
        "egui_filter_layers"
    }

    fn output_hook(&mut self, ctx: &egui::Context, output: &mut FullOutput) {
        // Every pass's delta is recorded, discarded passes included: a texture uploaded in full
        // there is still missing from egui-wgpu's renderer in the passes that follow.
        self.shared.texture_copies().observe(&self.shared.render_state, &output.textures_delta);
        let matched = take_matched(&self.shared, &output.shapes);
        if matched.is_empty() || pass_is_discarded(output, ctx.options(|options| options.max_passes.get())) {
            // A discarded pass's callbacks die with its shapes, so its layers never reach `prepare`.
            return;
        }
        let pixels_per_point = output.pixels_per_point;
        for (index, pending) in matched {
            let class = output.viewport_output.get(&pending.viewport).map(|viewport| viewport.class);
            let clear_color = viewport_clear_color(self.shared.options.clear_color(), class);
            let plan = plan_layer(ctx, &self.shared, &output.shapes, index, &pending, pixels_per_point, clear_color);
            pending.slot.lock().plan = plan;
        }
    }
}

/// Whether egui will throw this pass's output away and run another pass (`Context::run_ui`
/// repeats a pass that called `request_discard` until `max_passes` passes completed).
/// `output.platform_output.num_completed_passes` counts this pass as well.
fn pass_is_discarded(output: &FullOutput, max_passes: usize) -> bool {
    output.platform_output.requested_discard() && output.platform_output.num_completed_passes < max_passes
}

/// The colour a viewport of `class` is cleared to before egui draws, as eframe does it: immediate
/// viewports are cleared to transparent, every other one to `clear_color` (`App::clear_color`).
fn viewport_clear_color(clear_color: [f32; 4], class: Option<egui::ViewportClass>) -> [f32; 4] {
    match class {
        Some(egui::ViewportClass::Immediate) => [0.0; 4],
        Some(egui::ViewportClass::Root | egui::ViewportClass::Deferred | egui::ViewportClass::EmbeddedWindow) | None => clear_color,
    }
}

/// Removes from the pending list every layer whose callback appears in `shapes`, returning each
/// with its index there. Entries whose callback died are dropped; others (painted in another
/// viewport's pass) stay pending.
fn take_matched(shared: &Shared, shapes: &[egui::epaint::ClippedShape]) -> Vec<(usize, Pending)> {
    let mut pending = shared.pending();
    pending.retain(|entry| entry.callback.strong_count() > 0);
    if pending.is_empty() {
        return Vec::new();
    }
    let mut matched = Vec::new();
    let mut index = 0;
    while index < pending.len() {
        match find_callback(shapes, pending[index].callback.as_ptr().cast::<()>()) {
            Some(position) => matched.push((position, pending.swap_remove(index))),
            None => index += 1,
        }
    }
    matched
}

/// Builds the plan of one layer found at `index` in `shapes`, or `None` when it is invisible.
fn plan_layer(ctx: &egui::Context, shared: &Shared, shapes: &[egui::epaint::ClippedShape], index: usize, pending: &Pending, pixels_per_point: f32, clear_color: [f32; 4]) -> Option<BeneathPlan> {
    let clipped = shapes.get(index)?;
    let Shape::Callback(callback) = &clipped.shape else {
        return None;
    };
    let transform_scale = transform_scale(pending.rect, callback.rect);
    let capture = capture_points(callback.rect, clipped.clip_rect, pending.padding_points, transform_scale, pixels_per_point)?;
    let user_textures = match resolve_user_textures(shared, pending.textures) {
        Ok(resolved) => resolved,
        Err(id) => {
            shared.fail_texture(id);
            return None;
        }
    };

    let selection = select_beneath(shapes.get(..index)?, capture);
    let mut primitives = ctx.tessellate(selection.shapes, pixels_per_point);
    let skipped = selection.skipped_callbacks + strip_callbacks(&mut primitives);
    if skipped > 0 && LoggedOnce::first(&shared.logged.skipped_callbacks) {
        log::debug!(
            "egui_filter_layers (wgpu): {skipped} paint callback(s) lie beneath a filter layer and are left out of its backdrop \
             (callbacks cannot be re-rendered offscreen; a lower filter layer is seen unfiltered). Logged once."
        );
    }

    // The method path does not satisfy `ctx.fonts`'s higher-ranked `FontsView<'_>` bound, so the
    // closure clippy suggests removing is required.
    #[allow(clippy::redundant_closure_for_method_calls)]
    let font_size = ctx.fonts(|fonts| fonts.font_image_size());
    let textures = snapshot_textures(shared, &primitives, font_size);
    Some(BeneathPlan { callback_rect: callback.rect, clip_rect: clipped.clip_rect, transform_scale, primitives, textures, clear_color, user_textures })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_passes_egui_will_repeat_are_discarded() {
        let mut output = FullOutput::default();
        output.platform_output.num_completed_passes = 1;
        assert!(!pass_is_discarded(&output, 2), "no discard requested");
        output.platform_output.request_discard_reasons.push(egui::RepaintCause::new());
        assert!(pass_is_discarded(&output, 2), "first of two passes");
        output.platform_output.num_completed_passes = 2;
        assert!(!pass_is_discarded(&output, 2), "the last allowed pass is painted even if it asked for a discard");
    }

    #[test]
    fn immediate_viewports_are_cleared_to_transparent_like_eframe() {
        let clear = [0.1, 0.2, 0.3, 0.4];
        // The colour is passed through, so bit-exact comparison is the contract.
        let bits = |class| viewport_clear_color(clear, class).map(f32::to_bits);
        assert_eq!(bits(Some(egui::ViewportClass::Immediate)), [0.0_f32; 4].map(f32::to_bits));
        assert_eq!(bits(Some(egui::ViewportClass::Root)), clear.map(f32::to_bits));
        assert_eq!(bits(Some(egui::ViewportClass::Deferred)), clear.map(f32::to_bits));
        assert_eq!(bits(None), clear.map(f32::to_bits));
    }
}
