/*
File: examples/demo/scenes/water.rs

Purpose:
Scene "water": the backdrop seen through a water surface — one full-area STATEFUL layer
(`presets::wave_sim` substeps + `presets::wave_display`) — with a menu window floating on it.
The window's rect is handed to the simulation every frame as a rounded `WaveObstacle`, so waves
reflect off the menu and diffract around its corners; dragging the menu drags the obstacle. A
second, optional obstacle (a "rock") is a draggable rect painted after the layer. Presses on the
water drop impulses, holding the button drags a wake (`pointer_strength`), and "rain" drops
pseudo-random impulses.

Key structures:
- WaterScene: impulses, simulation / display parameters, the rock, rain and record state.
- ENTRY: the registry entry (`--scene water`).

Notes:
The menu is an `egui::Window` (a higher egui layer), so it is painted after the shader layer and
stays crisp. The simulation steps once per paint (substeps per paint, not per second). While
recording, clicks are scripted as a pure function of the loop phase and the menu is pinned to
the centre. The recorder paints every step the same number of times (its screenshot handshake),
so the simulation steps identically in every run and recordings are deterministic; they are not
a seamless loop, since the surface carries its waves over the seam.
*/

use eframe::egui::{self, Align2, Color32, CornerRadius, Id, Pos2, Rect, Sense, Stroke, StrokeKind, vec2};
use egui_shader_layers::{FilterChain, ShaderLayer, IMPULSE_SLOTS, Impulses, ParamError, StateOptions, presets, reset_state};
use egui_shader_layers::presets::WaveObstacle;

use super::ripples::scripted_impulses;
use super::{DragRect, Scene, SceneEntry, SceneFrame, SceneInit};
use crate::backdrop::CircleMotion;

/// Registry entry.
pub const ENTRY: SceneEntry = SceneEntry { key: "water", title: "Water around a menu", build };

/// Seconds an impulse pushes the surface (short: a splash, not a press).
const IMPULSE_LIFE: f32 = 0.12;

/// Seconds between rain drops.
const RAIN_INTERVAL: f64 = 0.18;

/// Seconds the scene keeps requesting repaints after the last disturbance (waves take a while
/// to settle under the default damping).
const SETTLE_SECONDS: f64 = 8.0;

/// Scripted clicks while recording: `(loop phase, position in layer uv)`, around the centred
/// menu so their rings hit it from above, the side and below.
const RECORD_CLICKS: [(f32, [f32; 2]); 4] = [(0.02, [0.5, 0.14]), (0.2, [0.16, 0.36]), (0.38, [0.82, 0.74]), (0.56, [0.3, 0.84])];

/// The water scene.
struct WaterScene {
    /// Live splashes (positions in the scene area's uv).
    impulses: Impulses,
    /// Strength of a click splash (velocity pushed per second at the splash centre).
    strength: f32,
    /// Splash radius, in points.
    impulse_radius: f32,
    /// Wake strength while the button is held on the water (`pointer_strength`).
    wake: f32,
    /// Simulation steps per paint.
    substeps: usize,
    /// Velocity kept per step.
    damping: f32,
    /// Squared Courant number (`0..=0.5`): how far a wave travels per step.
    speed: f32,
    /// Refraction, in points of displacement per unit of slope (also the cap).
    refraction: f32,
    /// Highlight strength.
    specular: f32,
    /// State resolution relative to the capture.
    state_scale: f32,
    /// Whether it rains.
    rain: bool,
    /// When the next rain drop falls (egui time, seconds).
    next_rain: f64,
    /// Rain drops so far (seeds the drop positions).
    rain_drops: u32,
    /// The second obstacle (`None` until the first frame places it relative to the scene area).
    rock: Option<DragRect>,
    /// Whether the rock is in the water.
    rock_enabled: bool,
    /// Keep repainting until this egui time (seconds).
    active_until: f64,
    /// Whether a parameter error was already logged.
    warned: bool,
}

/// Builds the scene with default water parameters (`--rect` / `--stack` do not apply).
fn build(_init: &SceneInit<'_>) -> Box<dyn Scene> {
    Box::new(WaterScene {
        impulses: Impulses::new(IMPULSE_LIFE).with_capacity(IMPULSE_SLOTS),
        strength: 60.0,
        impulse_radius: 7.0,
        wake: 15.0,
        substeps: 2,
        damping: 0.996,
        speed: 0.35,
        refraction: 20.0,
        specular: 0.9,
        state_scale: 1.0,
        rain: false,
        next_rain: 0.0,
        rain_drops: 0,
        rock: None,
        rock_enabled: true,
        active_until: 0.0,
        warned: false,
    })
}

/// The persistent-state identity of the water layer (distinct from every other scene's).
fn water_id() -> Id {
    Id::new("demo-scene-water")
}

/// A deterministic pseudo-random value in `0..1` for `n` (an integer hash: rain positions).
fn hash01(n: u32, salt: u32) -> f32 {
    let mut x = n.wrapping_mul(0x9E37_79B9) ^ salt.wrapping_mul(0x85EB_CA6B);
    x ^= x >> 16;
    x = x.wrapping_mul(0x7FEB_352D);
    x ^= x >> 15;
    // The top 16 bits always fit a u16.
    f32::from(u16::try_from(x >> 16).unwrap_or(0)) / 65536.0
}

/// What the menu window asked for this frame.
#[derive(Clone, Copy, Debug, Default)]
struct MenuActions {
    /// "Drop a stone" was clicked.
    stone: bool,
    /// "Calm the water" was clicked.
    calm: bool,
}

impl WaterScene {
    /// The wave chain for this frame: `substeps` configured `wave_sim` passes and the display.
    ///
    /// # Errors
    /// A `ParamError` if the built-in `wave_sim` rejected a parameter (a crate bug).
    fn chain(&self, now: f64, obstacles: &[WaveObstacle], wake_on: bool) -> Result<FilterChain, ParamError> {
        let sim = presets::wave_sim(&self.impulses, now, obstacles)
            .with("speed", self.speed)?
            .with("damping", self.damping)?
            .with("impulse_radius", self.impulse_radius)?
            .with("pointer_strength", if wake_on { self.wake } else { 0.0 })?;
        let display = presets::wave_display(self.refraction, self.specular);
        Ok(std::iter::repeat_n(sim, self.substeps).chain(std::iter::once(display)).collect())
    }

    /// Shows the menu window (pinned to the centre of `area` while recording) and returns its
    /// rect (`None` when egui did not show it) and the actions clicked in it.
    fn menu(&mut self, ctx: &egui::Context, area: Rect, recording: bool) -> (Option<Rect>, MenuActions) {
        let mut actions = MenuActions::default();
        let mut window = egui::Window::new("Menu").id(Id::new("demo-water-menu")).pivot(Align2::CENTER_CENTER).collapsible(false).resizable(false).constrain_to(area);
        // `fixed_pos` (not `current_pos`) while recording: egui 0.36 restores a title-bar-draggable
        // window's remembered position after `current_pos` (window.rs, `pivot_pos_before_begin`),
        // and `fixed_pos` also makes it immovable, which a recording needs anyway.
        window = if recording { window.fixed_pos(area.center()) } else { window.default_pos(area.center()) };
        let response = window.show(ctx, |ui| {
            ui.set_width(240.0);
            ui.label("This window floats on the water: waves reflect off it and bend around its corners. Drag it by its title bar.");
            ui.separator();
            actions.stone = ui.button("Drop a stone").clicked();
            ui.checkbox(&mut self.rain, "Rain");
            actions.calm = ui.button("Calm the water").clicked();
        });
        (response.map(|response| response.response.rect), actions)
    }

    /// Pushes a splash at `pos` (points) and keeps the scene repainting while it settles.
    fn splash(&mut self, pos: Pos2, area: Rect, strength: f32, now: f64) {
        if self.impulses.push_point(pos, area, strength, now) {
            self.active_until = now + SETTLE_SECONDS;
        }
    }
}

impl Scene for WaterScene {
    fn controls(&mut self, ui: &mut egui::Ui) {
        ui.label("Click the water to drop a splash; hold and drag for a wake. Drag the menu or the rock: the waves follow the obstacles.");
        ui.add(egui::Slider::new(&mut self.substeps, 1..=8).text("substeps per frame"));
        ui.add(egui::Slider::new(&mut self.speed, 0.02..=0.5).text("wave speed"));
        ui.add(egui::Slider::new(&mut self.damping, 0.95..=1.0).text("damping"));
        ui.add(egui::Slider::new(&mut self.refraction, 0.0..=60.0).text("refraction (pt)"));
        ui.add(egui::Slider::new(&mut self.specular, 0.0..=4.0).text("specular"));
        ui.add(egui::Slider::new(&mut self.strength, 5.0..=200.0).logarithmic(true).text("splash strength"));
        ui.add(egui::Slider::new(&mut self.impulse_radius, 1.0..=30.0).text("splash radius (pt)"));
        ui.add(egui::Slider::new(&mut self.wake, 0.0..=80.0).text("wake strength"));
        ui.add(egui::Slider::new(&mut self.state_scale, 0.25..=1.0).text("simulation resolution"));
        ui.checkbox(&mut self.rain, "rain");
        ui.checkbox(&mut self.rock_enabled, "rock (second obstacle)");
        if ui.button("reset the water").clicked() {
            reset_state(ui.ctx(), water_id());
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, frame: &SceneFrame<'_>) {
        let area = ui.max_rect();
        let painter = ui.painter_at(area);
        let now = frame.time;
        frame.backdrop.paint(&painter, area, CircleMotion::for_frame(frame.time, frame.phase));

        // The menu first: its rect of THIS frame is the obstacle (it is a higher egui layer, so
        // it is still painted after the water).
        let (menu_rect, actions) = self.menu(ui.ctx(), area, frame.phase.is_some());
        let radius = f32::from(ui.visuals().window_corner_radius.nw);
        if actions.calm {
            reset_state(ui.ctx(), water_id());
        }
        if let Some(menu) = menu_rect.filter(|_| actions.stone) {
            self.splash(menu.center_bottom() + vec2(0.0, 50.0), area, self.strength, now);
        }

        // Hit testing: the water first, the rock after it (so the rock wins where they overlap).
        // The rock starts (and, while recording, stays) at a fixed place relative to the area.
        let water = ui.interact(area, ui.id().with("water-surface"), Sense::click_and_drag());
        let rock_home = || DragRect { rect: Rect::from_center_size(area.lerp_inside(vec2(0.8, 0.24)), vec2(110.0, 70.0)), corner_radius: 30 };
        if frame.phase.is_some() || self.rock.is_none() {
            self.rock = Some(rock_home());
        }
        let rock = self.rock.as_mut().filter(|_| self.rock_enabled);
        if let Some(rock) = rock {
            rock.interact(ui, "water-rock");
        }
        let rock = self.rock.filter(|_| self.rock_enabled);
        let wake_on = frame.phase.is_none() && water.is_pointer_button_down_on();
        if let Some(phase) = frame.phase {
            self.impulses = scripted_impulses(&RECORD_CLICKS, phase, now, IMPULSE_LIFE, IMPULSE_SLOTS, self.strength);
            self.active_until = now + SETTLE_SECONDS;
        } else {
            let pressed = water.is_pointer_button_down_on() && ui.input(|input| input.pointer.primary_pressed());
            if let Some(pos) = water.interact_pointer_pos().filter(|_| pressed) {
                self.splash(pos, area, self.strength, now);
            }
            if wake_on || water.dragged() {
                self.active_until = now + SETTLE_SECONDS;
            }
            if self.rain && now >= self.next_rain {
                let uv = [hash01(self.rain_drops, 1), hash01(self.rain_drops, 2)];
                let strength = self.strength * (0.3 + 0.4 * hash01(self.rain_drops, 3));
                if self.impulses.push(uv, strength, now) {
                    self.active_until = now + SETTLE_SECONDS;
                }
                self.rain_drops = self.rain_drops.wrapping_add(1);
                self.next_rain = now + RAIN_INTERVAL;
            }
        }
        self.impulses.retire(now);
        // The simulation advances only on painted frames: keep them coming while it settles.
        if now < self.active_until {
            ui.ctx().request_repaint();
        }

        let mut obstacles = Vec::with_capacity(2);
        obstacles.extend(menu_rect.and_then(|menu| WaveObstacle::from_rect(menu, area, radius)));
        if let Some(rock) = rock {
            obstacles.extend(WaveObstacle::from_rect(rock.rect, area, f32::from(rock.corner_radius)));
        }
        match self.chain(now, &obstacles, wake_on) {
            Ok(chain) => ShaderLayer::new(area, chain).id(water_id()).state(StateOptions::default().with_scale(self.state_scale)).paint(&painter),
            Err(error) => {
                if !self.warned {
                    log::error!("demo: water: the built-in wave_sim rejected a parameter: {error}");
                    self.warned = true;
                }
            }
        }

        // Painted after the layer: the rock stays crisp on top of the water.
        if let Some(rock) = self.rock.filter(|_| self.rock_enabled) {
            let corner = CornerRadius::same(rock.corner_radius);
            painter.rect_filled(rock.rect, corner, Color32::from_rgb(74, 68, 62));
            painter.rect_stroke(rock.rect.shrink(4.0), corner, Stroke::new(2.0, Color32::from_rgb(104, 96, 88)), StrokeKind::Inside);
            rock.paint_outline(&painter);
        }
    }
}
