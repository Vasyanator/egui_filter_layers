/*
File: examples/demo/scenes/ripples.rs

Purpose:
Scene "ripples": the shared backdrop (and its scroll area of real widgets) under one full-area
layer running `presets::click_ripples`. Every press on the scene spawns a ripple (optionally a
trail of them while dragging); the side panel tunes the ripple shape and the `Impulses`
lifetime / capacity and shows the live impulse count.

Key structures:
- RipplesScene: the `Impulses` collection and the ripple parameters.
- ENTRY: the registry entry (`--scene ripples`).

Key functions:
- scripted_impulses(): the `--record` stand-in for clicks, a pure function of the loop phase
  (also used by the water scene).

Notes:
`click_ripples` reads no `fl.time`, so the scene keeps frames coming with
`Impulses::request_repaint_if_alive` (the demo shell repaints every frame anyway; in an app
without that, an idle scene stops repainting once every ripple has expired).
*/

use eframe::egui::{self, Pos2, Rect};
use egui_filter_layers::{FilterLayer, IMPULSE_SLOTS, Impulses, presets};

use super::{Scene, SceneEntry, SceneFrame, SceneInit};
use crate::backdrop::{CircleMotion, scroll_widgets};

/// Registry entry.
pub const ENTRY: SceneEntry = SceneEntry { key: "ripples", title: "Click ripples", build };

/// Scripted clicks while recording: `(loop phase, position in layer uv)`. The last one sits at
/// 0.6 so that, with the lifetime capped by `scripted_impulses`, nothing is alive at the loop
/// seam.
const RECORD_CLICKS: [(f32, [f32; 2]); 3] = [(0.0, [0.3, 0.42]), (0.3, [0.68, 0.58]), (0.6, [0.52, 0.3])];

/// The click-ripple scene.
struct RipplesScene {
    /// Live ripples (positions in the scene area's uv).
    impulses: Impulses,
    /// Largest displacement, in points.
    amplitude: f32,
    /// Distance between crests, in points.
    wavelength: f32,
    /// Travel speed of the ring, in points per second.
    speed: f32,
    /// Seconds each ripple lives (mirrors `impulses.life()`; edited here, applied by rebuilding).
    life: f32,
    /// Most ripples alive at once (mirrors `impulses.capacity()`).
    capacity: usize,
    /// Whether dragging spawns ripples along the pointer path.
    drag_spawns: bool,
    /// Distance between ripples spawned while dragging, in points.
    drag_spacing: f32,
    /// Where the last drag ripple was spawned (`None` when the button is up).
    last_drag_spawn: Option<Pos2>,
}

/// Builds the scene with default ripple parameters (`--rect` / `--stack` do not apply).
fn build(_init: &SceneInit<'_>) -> Box<dyn Scene> {
    let (life, capacity) = (1.6, IMPULSE_SLOTS);
    Box::new(RipplesScene {
        impulses: Impulses::new(life).with_capacity(capacity),
        amplitude: 12.0,
        wavelength: 26.0,
        speed: 180.0,
        life,
        capacity,
        drag_spawns: true,
        drag_spacing: 60.0,
        last_drag_spawn: None,
    })
}

impl RipplesScene {
    /// Rebuilds `impulses` with the current `life` / `capacity`, keeping the ripples it holds
    /// (their start times are unchanged, so a longer life revives nothing that already expired
    /// and was retired).
    fn rebuild_impulses(&mut self) {
        let mut impulses = Impulses::new(self.life).with_capacity(self.capacity);
        for impulse in self.impulses.iter() {
            impulses.push(impulse.uv, impulse.strength, impulse.started);
        }
        self.impulses = impulses;
    }

    /// Spawns ripples from this frame's pointer input over `area` (the layer rect). Presses that
    /// land on anything above the scene (a popup, a window) are ignored; presses on the scroll
    /// area's widgets still ripple, since those widgets lie beneath the layer.
    fn handle_input(&mut self, ui: &egui::Ui, area: Rect, now: f64) {
        let (pressed, down, pos) = ui.input(|input| (input.pointer.primary_pressed(), input.pointer.primary_down(), input.pointer.interact_pos()));
        let on_scene = pos.filter(|&pos| area.contains(pos) && ui.ctx().layer_id_at(pos) == Some(ui.layer_id()));
        if !down {
            self.last_drag_spawn = None;
        }
        let Some(pos) = on_scene else {
            return;
        };
        if pressed {
            self.impulses.push_point(pos, area, 1.0, now);
            self.last_drag_spawn = Some(pos);
        } else if down && self.drag_spawns && self.last_drag_spawn.is_some_and(|last| last.distance(pos) >= self.drag_spacing) {
            // Weaker than a click so a fast drag reads as a wake rather than a row of splashes.
            self.impulses.push_point(pos, area, 0.6, now);
            self.last_drag_spawn = Some(pos);
        }
    }
}

/// `--record` stand-in for clicks: the impulses alive at loop `phase` (`0..1`, egui time `time`
/// seconds) if a click happened at each `(phase, uv)` of `clicks`, with the given `life`,
/// `capacity` and `strength`. A pure function of the phase — independent of how many frames the
/// recorder paints per step — so the ripples are exact and the same in every run.
///
/// The loop period is `time / phase` (unknown at phase 0, where only a click at phase 0 counts,
/// at age 0). The life is capped to `0.4` periods so a click at phase <= 0.6 has expired by the
/// seam; clicks later than that would pop at the seam.
pub(super) fn scripted_impulses(clicks: &[(f32, [f32; 2])], phase: f32, time: f64, life: f32, capacity: usize, strength: f32) -> Impulses {
    let period = if phase > 0.0 { time / f64::from(phase) } else { 0.0 };
    #[allow(clippy::cast_possible_truncation)] // A loop length of seconds; f32 precision is plenty.
    let life = if period > 0.0 { life.min(0.4 * period as f32) } else { life };
    let mut impulses = Impulses::new(life).with_capacity(capacity);
    for &(at, uv) in clicks.iter().filter(|&&(at, _)| at <= phase) {
        impulses.push(uv, strength, time - f64::from(phase - at) * period);
    }
    impulses
}

impl Scene for RipplesScene {
    fn controls(&mut self, ui: &mut egui::Ui) {
        ui.label("Click anywhere on the scene; drag for a trail of ripples.");
        ui.add(egui::Slider::new(&mut self.amplitude, 0.0..=40.0).text("amplitude (pt)"));
        ui.add(egui::Slider::new(&mut self.wavelength, 4.0..=120.0).text("wavelength (pt)"));
        ui.add(egui::Slider::new(&mut self.speed, 10.0..=800.0).logarithmic(true).text("speed (pt/s)"));
        let life_changed = ui.add(egui::Slider::new(&mut self.life, 0.1..=6.0).text("lifetime (s)")).changed();
        let capacity_changed = ui.add(egui::Slider::new(&mut self.capacity, 1..=IMPULSE_SLOTS).text("capacity")).changed();
        if life_changed || capacity_changed {
            self.rebuild_impulses();
        }
        ui.checkbox(&mut self.drag_spawns, "dragging spawns ripples");
        ui.add_enabled(self.drag_spawns, egui::Slider::new(&mut self.drag_spacing, 10.0..=200.0).text("drag spacing (pt)"));
        let now = ui.input(|input| input.time);
        ui.label(format!("live impulses: {} / {}", self.impulses.pack(now).count, self.impulses.capacity()));
        if ui.button("clear").clicked() {
            self.impulses = Impulses::new(self.life).with_capacity(self.capacity);
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, frame: &SceneFrame<'_>) {
        let area = ui.max_rect();
        let painter = ui.painter_at(area);
        frame.backdrop.paint(&painter, area, CircleMotion::for_frame(frame.time, frame.phase));
        scroll_widgets(ui, area);
        let now = frame.time;
        if let Some(phase) = frame.phase {
            self.impulses = scripted_impulses(&RECORD_CLICKS, phase, now, self.life, self.capacity, 1.0);
        } else {
            self.handle_input(ui, area, now);
        }
        self.impulses.retire(now);
        self.impulses.request_repaint_if_alive(ui.ctx(), now);
        FilterLayer::new(area, presets::click_ripples(&self.impulses, now, self.amplitude, self.wavelength, self.speed)).paint(&painter);
    }
}
