/*
File: examples/demo/scenes/bloom.rs

Purpose:
Scene "bloom": bright orbs, neon lines and text on a dark backdrop under one draggable filter rect,
showing bloom and the fast blurs (F1 reduced-resolution passes, F6 dual-Kawase). Three modes:
a custom bloom chain (own bright pass at an adjustable `Pass::with_scale`, then Gaussian or Kawase
blur, then an own combine reading `original()`), the built-in `presets::bloom`, and the plain blur
alone (Gaussian vs Kawase at equal visual radius). The side panel shows an averaged frame time.

Key structures:
- BloomScene: rect, mode, blur kind, the sliders, the compiled custom effects, the frame timer.
- Mode / BlurKind: what the layer runs.
- ENTRY: the registry entry (`--scene bloom`).

Key functions:
- BloomScene::chain(): builds the layer's chain from the controls (every frame; cheap).
- paint_dark_backdrop(): the scene's own dark content (animated, periodic in the record phase).

Notes:
The custom effects live in `bloom_bright.wgsl` / `bloom_combine.wgsl` (compiled once, in the
constructor; a compile error is shown in the side panel and the layer is skipped). The frame time
is the shared `FrameTimer` (CPU+GPU wall time between frames, capped by vsync), so the
Gaussian / Kawase difference shows at large radii.
*/

use std::f32::consts::TAU;

use eframe::egui::{self, Color32, FontId, Pos2, Rect, Stroke, pos2, vec2};
use egui_shader_layers::{Effect, EffectError, FilterChain, ShaderLayer, Pass, presets};

use super::{DragRect, FrameTimer, Scene, SceneEntry, SceneFrame, SceneInit};

/// Registry entry.
pub const ENTRY: SceneEntry = SceneEntry { key: "bloom", title: "Bloom / fast blur", build };

/// What the layer runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    /// Own bright pass (scaled) -> selected blur -> own combine.
    CustomBloom,
    /// `presets::bloom(threshold, intensity, radius)`.
    PresetBloom,
    /// The selected blur alone.
    BlurOnly,
}

/// Which blur the custom bloom and the blur-only mode use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BlurKind {
    /// `presets::gaussian_blur(radius / 3)`: reaches `radius` points, taps grow with it.
    Gaussian,
    /// `presets::kawase_blur(radius)`: a pyramid of cheap scaled passes.
    Kawase,
}

/// The bloom / fast-blur scene.
struct BloomScene {
    /// The filter rect; `None` until the first frame knows the scene area (unless `--rect`).
    rect: Option<DragRect>,
    /// What the layer runs.
    mode: Mode,
    /// Blur used by `CustomBloom` and `BlurOnly`.
    blur: BlurKind,
    /// Bright-pass luma threshold, `0..1`.
    threshold: f32,
    /// Glow strength.
    intensity: f32,
    /// Blur reach in points (Gaussian sigma = radius / 3).
    radius: f32,
    /// Output scale of the custom bright pass (and of the Gaussian passes after it).
    pass_scale: f32,
    /// Glow tint of the custom combine.
    glow_color: Color32,
    /// The custom bright and combine effects, compiled once.
    effects: Result<(Effect, Effect), EffectError>,
    /// The averaged frame-time readout.
    frame_timer: FrameTimer,
}

/// Builds the scene (compiles the two custom effects: cheap naga validation).
fn build(init: &SceneInit<'_>) -> Box<dyn Scene> {
    let effects = Effect::new("demo_bloom_bright", include_str!("bloom_bright.wgsl")).and_then(|bright| Ok((bright, Effect::new("demo_bloom_combine", include_str!("bloom_combine.wgsl"))?)));
    if let Err(error) = &effects {
        log::error!("demo: the bloom scene's custom effects did not compile: {error}");
    }
    Box::new(BloomScene {
        rect: init.rect.map(|rect| DragRect { rect, corner_radius: 16 }),
        mode: Mode::CustomBloom,
        blur: BlurKind::Kawase,
        threshold: 0.55,
        intensity: 1.4,
        radius: 36.0,
        pass_scale: 0.5,
        glow_color: Color32::WHITE,
        effects,
        frame_timer: FrameTimer::default(),
    })
}

impl BloomScene {
    /// The selected blur at the current radius (reach `radius` points either way).
    fn blur_chain(&self) -> FilterChain {
        match self.blur {
            BlurKind::Gaussian => presets::gaussian_blur(self.radius / 3.0),
            BlurKind::Kawase => presets::kawase_blur(self.radius),
        }
    }

    /// The layer's chain from the current controls. `Err` carries a message for the side panel
    /// (a custom effect failed to compile, or rejected a named parameter).
    fn chain(&self) -> Result<FilterChain, String> {
        match self.mode {
            Mode::PresetBloom => Ok(presets::bloom(self.threshold, self.intensity, self.radius)),
            Mode::BlurOnly => Ok(self.blur_chain()),
            Mode::CustomBloom => {
                let (bright, combine) = self.effects.as_ref().map_err(ToString::to_string)?;
                let scale = self.pass_scale;
                // The four taps reach a quarter output pixel = 0.25 / scale capture pixels; one
                // extra pixel covers the bilinear upsample (points = pixels at >= 1 px/pt).
                let bright = Pass::new(bright).with_scale(scale).with_sample_radius(0.25 / scale + 1.0).with("threshold", self.threshold).and_then(|pass| pass.with("knee", 0.08)).map_err(|error| error.to_string())?;
                let mut blur = self.blur_chain();
                if self.blur == BlurKind::Gaussian {
                    // The Gaussian steps in CAPTURE pixels, so it blurs the same at any scale:
                    // running it at the bright pass's scale only makes it cheaper.
                    for pass in blur.passes_mut() {
                        pass.set_scale(scale);
                    }
                }
                let glow = self.glow_color.to_normalized_gamma_f32();
                let combine = Pass::new(combine).with_sample_radius(0.0).with("glow_color", glow).and_then(|pass| pass.with("intensity", self.intensity)).map_err(|error| error.to_string())?;
                Ok(FilterChain::new().then(bright).then(blur).then(combine))
            }
        }
    }
}

impl Scene for BloomScene {
    fn controls(&mut self, ui: &mut egui::Ui) {
        ui.label("mode");
        ui.horizontal_wrapped(|ui| {
            ui.selectable_value(&mut self.mode, Mode::CustomBloom, "custom bloom");
            ui.selectable_value(&mut self.mode, Mode::PresetBloom, "presets::bloom");
            ui.selectable_value(&mut self.mode, Mode::BlurOnly, "blur only");
        });
        ui.add_enabled_ui(self.mode != Mode::PresetBloom, |ui| {
            ui.horizontal(|ui| {
                ui.label("blur");
                ui.selectable_value(&mut self.blur, BlurKind::Gaussian, "gaussian");
                ui.selectable_value(&mut self.blur, BlurKind::Kawase, "kawase");
            });
        });
        ui.add(egui::Slider::new(&mut self.radius, 2.0..=160.0).logarithmic(true).text("radius (pt)"));
        ui.add_enabled_ui(self.mode != Mode::BlurOnly, |ui| {
            ui.add(egui::Slider::new(&mut self.threshold, 0.0..=1.0).text("threshold"));
            ui.add(egui::Slider::new(&mut self.intensity, 0.0..=4.0).text("intensity"));
        });
        ui.add_enabled_ui(self.mode == Mode::CustomBloom, |ui| {
            ui.add(egui::Slider::new(&mut self.pass_scale, 0.125..=1.0).text("bright pass scale"));
            ui.horizontal(|ui| {
                ui.label("glow colour");
                ui.color_edit_button_srgba(&mut self.glow_color);
            });
        });
        ui.separator();
        match self.chain() {
            Ok(chain) => {
                let scales = chain.passes().iter().map(|pass| format!("{:.3}", pass.scale())).collect::<Vec<_>>().join(" ");
                ui.label(format!("{} passes, scales: {scales}", chain.len()));
            }
            Err(error) => {
                ui.colored_label(ui.visuals().error_fg_color, format!("chain unavailable: {error}"));
            }
        }
        self.frame_timer.label(ui);
        ui.weak("Wall time between frames (vsync caps it). Compare gaussian and kawase at a large radius.");
    }

    fn ui(&mut self, ui: &mut egui::Ui, frame: &SceneFrame<'_>) {
        let area = ui.max_rect();
        self.frame_timer.record(ui);

        let painter = ui.painter_at(area);
        paint_dark_backdrop(&painter, area, frame.time, frame.phase);
        let rect = self.rect.get_or_insert_with(|| DragRect { rect: area.shrink2(vec2(area.width() * 0.08, area.height() * 0.1)), corner_radius: 16 });
        if frame.phase.is_some() {
            // Recording: the layer covers the scene; the content moves beneath it.
            rect.rect = area.shrink(24.0);
        }
        rect.interact(ui, "bloom");
        let rect = *rect;
        // A chain error is shown in the side panel (and a compile error logged once at build).
        if let Ok(chain) = self.chain() {
            ShaderLayer::new(rect.rect, chain).corner_radius(rect.corner_radius).paint(&painter);
        }
        rect.paint_outline(&painter);
    }
}

/// An animation angle (radians): `laps` turns per recording loop while recording (so the loop is
/// seamless), else `laps * 0.25` turns per second of egui time.
fn angle(time: f64, phase: Option<f32>, laps: i8) -> f32 {
    let laps_f = f32::from(laps);
    match phase {
        Some(phase) => TAU * phase * laps_f,
        None => {
            // Reduced modulo a full turn in f64 first, so the f32 keeps its precision.
            #[allow(clippy::cast_possible_truncation)]
            let turns = (time * 0.25 * f64::from(laps)).rem_euclid(1.0) as f32;
            TAU * turns
        }
    }
}

/// The scene's own content: a near-black backdrop, a dim grid, fixed "stars", orbiting bright
/// orbs, a neon sine line and large text. Every motion is periodic in the record `phase`.
fn paint_dark_backdrop(painter: &egui::Painter, area: Rect, time: f64, phase: Option<f32>) {
    painter.rect_filled(area, 0.0, Color32::from_rgb(8, 9, 18));
    let grid = Stroke::new(1.0, Color32::from_rgb(24, 28, 46));
    let mut x = area.left();
    while x < area.right() {
        painter.vline(x, area.y_range(), grid);
        x += 48.0;
    }
    let mut y = area.top();
    while y < area.bottom() {
        painter.hline(area.x_range(), y, grid);
        y += 48.0;
    }
    // Fixed pseudo-random stars (a small LCG, same every frame).
    let mut seed = 0x2545_f491_u32;
    let mut next = || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        // The top 16 bits (always < 2^16, so the conversion cannot fail), exact in f32.
        f32::from(u16::try_from(seed >> 16).unwrap_or(u16::MAX)) / 65_535.0
    };
    for _ in 0..70 {
        let pos = pos2(area.left() + next() * area.width(), area.top() + next() * area.height());
        painter.circle_filled(pos, 0.8 + next() * 1.4, Color32::from_gray(200 + (next() * 55.0) as u8));
    }

    let center = area.center();
    let orbit = vec2(area.width() * 0.34, area.height() * 0.3);
    let orbs: [(Color32, f32, i8, f32); 6] = [
        (Color32::from_rgb(255, 80, 60), 26.0, 1, 0.0),
        (Color32::from_rgb(80, 200, 255), 20.0, -1, 1.1),
        (Color32::from_rgb(255, 220, 80), 16.0, 2, 2.3),
        (Color32::from_rgb(160, 255, 120), 22.0, 1, 3.4),
        (Color32::from_rgb(255, 90, 220), 18.0, -2, 4.4),
        (Color32::from_rgb(255, 255, 255), 12.0, 1, 5.5),
    ];
    for (index, (color, radius, laps, offset)) in orbs.into_iter().enumerate() {
        let a = angle(time, phase, laps) + offset;
        // Alternate orbits squashed differently so the orbs cross each other.
        let squash = if index % 2 == 0 { 1.0 } else { 0.55 };
        let pos = center + vec2(orbit.x * a.cos(), orbit.y * squash * a.sin());
        painter.circle_filled(pos, radius, color);
        painter.circle_filled(pos, radius * 0.45, Color32::WHITE);
    }

    // A neon sine line whose phase travels once (or twice) per loop.
    let shift = angle(time, phase, 2);
    let points: Vec<Pos2> = (0..=96_u8)
        .map(|step| {
            let t = f32::from(step) / 96.0;
            let x = area.left() + t * area.width();
            pos2(x, area.bottom() - area.height() * 0.2 + (t * TAU * 3.0 + shift).sin() * area.height() * 0.07)
        })
        .collect();
    painter.add(egui::Shape::line(points, Stroke::new(3.0, Color32::from_rgb(90, 255, 230))));

    painter.text(pos2(center.x, area.top() + area.height() * 0.18), egui::Align2::CENTER_CENTER, "BLOOM", FontId::proportional(area.height() * 0.16), Color32::from_rgb(255, 245, 225));
    painter.text(pos2(center.x, area.top() + area.height() * 0.3), egui::Align2::CENTER_CENTER, "dim text stays dim", FontId::proportional(22.0), Color32::from_rgb(90, 96, 120));
}
