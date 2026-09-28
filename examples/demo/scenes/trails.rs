/*
File: examples/demo/scenes/trails.rs

Purpose:
Scene "trails": a STATEFUL `presets::trails(decay)` layer on a draggable / resizable rect over
moving content — the backdrop's circles, an orbiting comet and (optionally) a glowing brush under
the pointer, all painted beneath the layer — so everything that moves leaves a fading trace.

Key structures:
- TrailsScene: the rect, the decay and the content toggles.
- ENTRY: the registry entry (`--scene trails`).

Notes:
The trace fades per second (`fl.delta_time`), so the fade does not depend on the frame rate;
the state advances only on painted frames, so the scene requests repaints while its content
moves. While recording, the rect is pinned and the comet follows a closed path in the loop phase
(the brush is off, pointer input is dropped). A repeated paint of one recording step has
`delta_time` 0 and leaves the trace unchanged, so recordings are deterministic; they are not a
seamless loop, since the first frame starts without a trace.
*/

use eframe::egui::{self, Color32, Id, Pos2, Rect, Vec2, vec2};
use egui_filter_layers::{FilterLayer, presets, reset_state};

use super::{DragRect, Scene, SceneEntry, SceneFrame, SceneInit};
use crate::backdrop::CircleMotion;

/// Registry entry.
pub const ENTRY: SceneEntry = SceneEntry { key: "trails", title: "Trails", build };

/// The trails scene.
struct TrailsScene {
    /// The trailed rect (`None` until the first frame places it; `--rect` places it up front).
    rect: Option<DragRect>,
    /// Fraction of a trace left after one second (`0..=1`).
    decay: f32,
    /// Whether the comet orbits beneath the layer.
    comet: bool,
    /// Whether a glowing brush follows the pointer beneath the layer.
    brush: bool,
}

/// Builds the scene; `--rect` places the trailed rect.
fn build(init: &SceneInit<'_>) -> Box<dyn Scene> {
    Box::new(TrailsScene { rect: init.rect.map(|rect| DragRect { rect, corner_radius: 18 }), decay: 0.08, comet: true, brush: true })
}

/// The persistent-state identity of the trails layer (distinct from every other scene's).
fn trails_id() -> Id {
    Id::new("demo-scene-trails")
}

/// The comet's position in `area` at angle `turn` (radians): a closed Lissajous curve (x : y
/// frequencies 2 : 3), so `turn = TAU * phase` is periodic in the recording loop.
fn comet_pos(area: Rect, turn: f32) -> Pos2 {
    area.center() + vec2(0.34 * area.width() * (2.0 * turn).sin(), 0.32 * area.height() * (3.0 * turn + 0.7).sin())
}

/// A soft glowing dot (concentric translucent circles and a white core) at `center`.
fn glow_dot(painter: &egui::Painter, center: Pos2, radius: f32, color: Color32) {
    for (scale, alpha) in [(1.0, 0.18), (0.7, 0.3), (0.45, 0.6)] {
        painter.circle_filled(center, radius * scale, color.gamma_multiply(alpha));
    }
    painter.circle_filled(center, radius * 0.25, Color32::WHITE);
}

impl Scene for TrailsScene {
    fn controls(&mut self, ui: &mut egui::Ui) {
        ui.label("Everything moving under the rect leaves a trace; drag the rect or its corner handle.");
        ui.add(egui::Slider::new(&mut self.decay, 0.0..=0.95).text("decay (left after 1 s)"));
        ui.checkbox(&mut self.comet, "orbiting comet");
        ui.checkbox(&mut self.brush, "brush under the pointer");
        if ui.button("reset the trails").clicked() {
            reset_state(ui.ctx(), trails_id());
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, frame: &SceneFrame<'_>) {
        let area = ui.max_rect();
        let painter = ui.painter_at(area);
        frame.backdrop.paint(&painter, area, CircleMotion::for_frame(frame.time, frame.phase));

        let home = || DragRect { rect: Rect::from_center_size(area.center(), area.size() * Vec2::new(0.8, 0.8)), corner_radius: 18 };
        if frame.phase.is_some() {
            self.rect = Some(home());
        }
        let rect = self.rect.get_or_insert_with(home);
        rect.interact(ui, "trails");

        // Moving content, painted BENEATH the layer so it is trailed.
        if self.comet || frame.phase.is_some() {
            let turn = match frame.phase {
                Some(phase) => std::f32::consts::TAU * phase,
                // A slow orbit; the f64 -> f32 narrowing only loses precision after hours.
                #[allow(clippy::cast_possible_truncation)]
                None => (frame.time * 0.35) as f32,
            };
            glow_dot(&painter, comet_pos(area, turn), 26.0, Color32::from_rgb(120, 220, 255));
        }
        let pointer = ui.input(|input| input.pointer.hover_pos()).filter(|&pos| area.contains(pos) && ui.ctx().layer_id_at(pos) == Some(ui.layer_id()));
        if let Some(pos) = pointer.filter(|_| self.brush && frame.phase.is_none()) {
            glow_dot(&painter, pos, 20.0, Color32::from_rgb(255, 170, 60));
        }

        // The circles and the comet move every frame; the trace only fades on painted frames.
        ui.ctx().request_repaint();
        FilterLayer::new(rect.rect, presets::trails(self.decay)).id(trails_id()).corner_radius(rect.corner_radius).paint(&painter);
        rect.paint_outline(&painter);
    }
}
