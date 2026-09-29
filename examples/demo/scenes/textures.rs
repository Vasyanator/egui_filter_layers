/*
File: examples/demo/scenes/textures.rs

Purpose:
Scene "textures": user textures read by custom effects (F8). Three textures are generated
procedurally when the scene is built (no binary assets) and attached to one draggable layer over
the shared backdrop; a selector switches between three custom WGSL effects, each reading its own
slot: tiling-noise displacement (`tex_displace.wgsl`, slot 0), normal-map glass refraction with a
specular highlight (`tex_refract.wgsl`, slot 1) and a 3D-LUT colour grade (`tex_lut.wgsl`, slot 2,
exact `tex_load` taps).

Key structures:
- TexturesScene: rect, selected effect, sliders, the three texture handles, compiled effects.
- EffectKind / Grade: the effect selector and the LUT's grading look.
- ENTRY: the registry entry (`--scene textures`).

Key functions:
- noise_image(), normal_map_image(), lut_image(): the procedural textures (periodic where tiled).
- TexturesScene::pass(): the selected effect's pass with its named parameters.

Notes:
Generation is a few hundred thousand cheap float operations: acceptable on the GUI thread at
scene build. `failed_textures()` and the layer's `check_textures()` are shown in the side panel.
The handles are kept for the scene's lifetime (a freed handle would make the layer skip).
*/

use std::f32::consts::TAU;

use eframe::egui::{self, Color32, ColorImage, Rect, TextureHandle, TextureOptions, pos2, vec2};
use egui_shader_layers::{Effect, EffectError, ShaderLayer, Pass};

use super::{DragRect, Scene, SceneEntry, SceneFrame, SceneInit, record_rect};
use crate::backdrop::CircleMotion;

/// Registry entry.
pub const ENTRY: SceneEntry = SceneEntry { key: "textures", title: "Texture-driven effects", build };

/// Side of the square noise and normal-map textures, in texels.
const TILE_TEXELS: usize = 256;
/// LUT resolution per colour axis (16^3 entries, flattened to 256 x 16).
const LUT_SIDE: usize = 16;

/// Which custom effect the layer runs (also: which texture slot it reads).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EffectKind {
    /// Noise displacement, slot 0.
    Displace,
    /// Normal-map refraction + specular, slot 1.
    Refract,
    /// 3D LUT grade, slot 2.
    Lut,
}

/// The look baked into the LUT.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Grade {
    /// Teal shadows, orange highlights, a gentle S-curve.
    TealOrange,
    /// Lifted blacks, warm and slightly desaturated.
    WarmFilm,
    /// Mostly desaturated, cold, contrasty.
    ColdNoir,
}

impl Grade {
    /// The graded colour of gamma-space `rgb` (each `0..=1`).
    fn apply(self, [r, g, b]: [f32; 3]) -> [f32; 3] {
        let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        let s_curve = |c: f32| c + 0.35 * (c * c * (3.0 - 2.0 * c) - c);
        let out = match self {
            Self::TealOrange => {
                let (shadow, light) = (1.0 - luma, luma);
                [s_curve(r) - 0.06 * shadow + 0.10 * light, s_curve(g) + 0.03 * shadow + 0.02 * light, s_curve(b) + 0.09 * shadow - 0.10 * light]
            }
            Self::WarmFilm => {
                let desat = |c: f32| luma + 0.8 * (c - luma);
                let lift = |c: f32| 0.07 + 0.9 * c;
                [lift(desat(r) * 1.06), lift(desat(g) * 1.0), lift(desat(b) * 0.86)]
            }
            Self::ColdNoir => {
                let desat = |c: f32| s_curve(s_curve(luma + 0.2 * (c - luma)));
                [desat(r) * 0.92, desat(g) * 0.98, desat(b) * 1.08 + 0.03]
            }
        };
        out.map(|c| c.clamp(0.0, 1.0))
    }
}

/// The texture-driven effects scene.
struct TexturesScene {
    /// The filter rect.
    rect: DragRect,
    /// The effect shown.
    kind: EffectKind,
    /// Displacement: largest offset, in points.
    displace_strength: f32,
    /// Displacement / refraction tile size in points.
    tile: f32,
    /// Refraction: largest offset, in points.
    refract_strength: f32,
    /// Highlight strength.
    specular: f32,
    /// Highlight exponent.
    shininess: f32,
    /// LUT: blend amount (0 = original).
    lut_amount: f32,
    /// Layer-relative x right of which the LUT effect shows the original.
    lut_split: f32,
    /// The look currently baked into `lut`.
    grade: Grade,
    /// Slot 0: tiling noise (`LINEAR_REPEAT`).
    noise: TextureHandle,
    /// Slot 1: normal map (`LINEAR_REPEAT`).
    normals: TextureHandle,
    /// Slot 2: the flattened LUT (`NEAREST`; read with exact `tex_load` anyway).
    lut: TextureHandle,
    /// The three effects (displace, refract, lut), compiled once.
    effects: Result<[Effect; 3], EffectError>,
}

/// Builds the scene: generates and uploads the textures, compiles the effects.
fn build(init: &SceneInit<'_>) -> Box<dyn Scene> {
    let grade = Grade::TealOrange;
    let noise = init.ctx.load_texture("demo-noise", noise_image(), TextureOptions::LINEAR_REPEAT);
    let normals = init.ctx.load_texture("demo-normals", normal_map_image(), TextureOptions::LINEAR_REPEAT);
    let lut = init.ctx.load_texture("demo-lut", lut_image(grade), TextureOptions::NEAREST);
    let effects = compile_effects();
    if let Err(error) = &effects {
        log::error!("demo: the textures scene's effects did not compile: {error}");
    }
    let rect = init.rect.unwrap_or_else(|| Rect::from_min_size(pos2(380.0, 120.0), vec2(460.0, 340.0)));
    Box::new(TexturesScene {
        rect: DragRect { rect, corner_radius: 20 },
        kind: EffectKind::Refract,
        displace_strength: 10.0,
        tile: 180.0,
        refract_strength: 14.0,
        specular: 0.8,
        shininess: 40.0,
        lut_amount: 1.0,
        lut_split: 0.5,
        grade,
        noise,
        normals,
        lut,
        effects,
    })
}

/// Compiles the three custom effects.
fn compile_effects() -> Result<[Effect; 3], EffectError> {
    Ok([
        Effect::new("demo_tex_displace", include_str!("tex_displace.wgsl"))?,
        Effect::new("demo_tex_refract", include_str!("tex_refract.wgsl"))?,
        Effect::new("demo_tex_lut", include_str!("tex_lut.wgsl"))?,
    ])
}

impl TexturesScene {
    /// The selected effect's pass. `pan` offsets the noise (tile units); `light` points towards
    /// the light (xyz). `Err` carries a message for the side panel.
    fn pass(&self, pan: [f32; 2], light: [f32; 3]) -> Result<Pass, String> {
        let [displace, refract, lut] = self.effects.as_ref().map_err(ToString::to_string)?;
        let pass = match self.kind {
            EffectKind::Displace => Pass::new(displace)
                .with_sample_radius(self.displace_strength)
                .with("strength", self.displace_strength)
                .and_then(|pass| pass.with("tile", self.tile))
                .and_then(|pass| pass.with("pan", pan)),
            EffectKind::Refract => Pass::new(refract)
                .with_sample_radius(self.refract_strength)
                .with("strength", self.refract_strength)
                .and_then(|pass| pass.with("tile", self.tile))
                .and_then(|pass| pass.with("specular", self.specular))
                .and_then(|pass| pass.with("shininess", self.shininess))
                .and_then(|pass| pass.with("light", [light[0], light[1], light[2], 0.0])),
            EffectKind::Lut => Pass::new(lut).with("amount", self.lut_amount).and_then(|pass| pass.with("split", self.lut_split)),
        };
        pass.map_err(|error| error.to_string())
    }

    /// The layer with all three textures attached (each effect reads only its own slot).
    fn layer(&self, pass: Pass) -> ShaderLayer {
        ShaderLayer::new(self.rect.rect, pass).corner_radius(self.rect.corner_radius).texture(0, self.noise.id()).texture(1, self.normals.id()).texture(2, self.lut.id())
    }
}

impl Scene for TexturesScene {
    fn controls(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.selectable_value(&mut self.kind, EffectKind::Displace, "noise displacement");
            ui.selectable_value(&mut self.kind, EffectKind::Refract, "normal-map glass");
            ui.selectable_value(&mut self.kind, EffectKind::Lut, "3D LUT grade");
        });
        match self.kind {
            EffectKind::Displace => {
                ui.add(egui::Slider::new(&mut self.displace_strength, 0.0..=40.0).text("strength (pt)"));
                ui.add(egui::Slider::new(&mut self.tile, 40.0..=600.0).text("tile (pt)"));
            }
            EffectKind::Refract => {
                ui.add(egui::Slider::new(&mut self.refract_strength, 0.0..=40.0).text("strength (pt)"));
                ui.add(egui::Slider::new(&mut self.tile, 40.0..=600.0).text("tile (pt)"));
                ui.add(egui::Slider::new(&mut self.specular, 0.0..=2.0).text("specular"));
                ui.add(egui::Slider::new(&mut self.shininess, 1.0..=200.0).logarithmic(true).text("shininess"));
                ui.weak("The light follows the pointer over the scene.");
            }
            EffectKind::Lut => {
                let before = self.grade;
                ui.horizontal_wrapped(|ui| {
                    ui.selectable_value(&mut self.grade, Grade::TealOrange, "teal & orange");
                    ui.selectable_value(&mut self.grade, Grade::WarmFilm, "warm film");
                    ui.selectable_value(&mut self.grade, Grade::ColdNoir, "cold noir");
                });
                if self.grade != before {
                    // 4096 texels: regenerating and re-uploading is instant.
                    self.lut.set(lut_image(self.grade), TextureOptions::NEAREST);
                }
                ui.add(egui::Slider::new(&mut self.lut_amount, 0.0..=1.0).text("amount"));
                ui.add(egui::Slider::new(&mut self.lut_split, 0.0..=1.0).text("split (graded left)"));
            }
        }
        ui.horizontal(|ui| {
            for (name, handle) in [("noise", &self.noise), ("normals", &self.normals), ("LUT", &self.lut)] {
                ui.vertical(|ui| {
                    ui.small(name);
                    let size = handle.size_vec2();
                    let shown = if size.y < size.x * 0.25 { vec2(96.0, 96.0 * size.y / size.x) } else { vec2(64.0, 64.0) };
                    ui.image((handle.id(), shown));
                });
            }
        });
        match self.pass([0.0; 2], [0.0, 0.0, 1.0]) {
            Ok(pass) => {
                if let Err(error) = self.layer(pass).check_textures() {
                    ui.colored_label(ui.visuals().error_fg_color, format!("layer refused: {error}"));
                }
            }
            Err(error) => {
                ui.colored_label(ui.visuals().error_fg_color, format!("effect unavailable: {error}"));
            }
        }
        for failure in egui_shader_layers::failed_textures(ui.ctx()) {
            ui.colored_label(ui.visuals().error_fg_color, format!("texture {:?} failed: {}", failure.id, failure.error));
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, frame: &SceneFrame<'_>) {
        let area = ui.max_rect();
        let painter = ui.painter_at(area);
        frame.backdrop.paint(&painter, area, CircleMotion::for_frame(frame.time, frame.phase));
        let (pan, light_angle) = match frame.phase {
            // Recording: a circular pan and a full light turn per loop (seamless).
            Some(phase) => {
                let turn = TAU * phase;
                ([0.25 * turn.cos(), 0.25 * turn.sin()], turn)
            }
            None => {
                // Reduced in f64 before narrowing so long sessions keep precision (the textures
                // repeat, so only the fractional pan matters).
                #[allow(clippy::cast_possible_truncation)]
                let pan = [(frame.time * 0.03).rem_euclid(1.0) as f32, (frame.time * 0.017).rem_euclid(1.0) as f32];
                #[allow(clippy::cast_possible_truncation)]
                let turn = (frame.time * 0.5).rem_euclid(f64::from(TAU)) as f32;
                (pan, turn)
            }
        };
        if let Some(phase) = frame.phase {
            self.rect.rect = record_rect(area, phase);
        }
        self.rect.interact(ui, "textures");
        // Light towards the pointer when it is over the scene (interactive only), else circling.
        let pointer = if frame.phase.is_none() { ui.input(|input| input.pointer.hover_pos()).filter(|pos| area.contains(*pos)) } else { None };
        let light = match pointer {
            Some(pos) => {
                let d = (pos - self.rect.rect.center()) / (0.5 * self.rect.rect.size().max_elem().max(1.0));
                [d.x, d.y, 0.9]
            }
            None => [0.7 * light_angle.cos(), 0.7 * light_angle.sin(), 0.9],
        };
        if let Ok(pass) = self.pass(pan, light) {
            self.layer(pass).paint(&painter);
        }
        if self.kind == EffectKind::Lut {
            let x = self.rect.rect.left() + self.lut_split * self.rect.rect.width();
            painter.vline(x, self.rect.rect.y_range(), egui::Stroke::new(1.0, Color32::from_white_alpha(160)));
        }
        self.rect.paint_outline(&painter);
    }
}

/// A 32-bit integer hash of a lattice point and a seed, as a value in `0..1`.
fn hash01(x: usize, y: usize, seed: u32) -> f32 {
    // Lattice coordinates are below 256 here; truncation to u32 cannot occur.
    let (x, y) = (u32::try_from(x).unwrap_or(0), u32::try_from(y).unwrap_or(0));
    let mut h = x.wrapping_mul(0x8da6_b343) ^ y.wrapping_mul(0xd816_3841) ^ seed.wrapping_mul(0xcb1a_b31f);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2c1b_3c6d);
    h ^= h >> 12;
    // The top 16 bits, exact in f32.
    f32::from(u16::try_from(h >> 16).unwrap_or(u16::MAX)) / 65_535.0
}

/// Periodic value noise at texel (`x`, `y`) of a `TILE_TEXELS` tile with `cells` lattice cells per
/// side (`cells` divides `TILE_TEXELS`), in `0..1`. Wraps at the tile edge, so the texture tiles.
fn value_noise(x: usize, y: usize, cells: usize, seed: u32) -> f32 {
    let cell = TILE_TEXELS / cells;
    let (cx, cy) = (x / cell, y / cell);
    // Fractions inside the cell; `cell` is at most 256, exact in f32.
    #[allow(clippy::cast_precision_loss)]
    let (fx, fy) = ((x % cell) as f32 / cell as f32, (y % cell) as f32 / cell as f32);
    let smooth = |t: f32| t * t * (3.0 - 2.0 * t);
    let (sx, sy) = (smooth(fx), smooth(fy));
    let at = |i: usize, j: usize| hash01(i % cells, j % cells, seed);
    let top = at(cx, cy) + (at(cx + 1, cy) - at(cx, cy)) * sx;
    let bottom = at(cx, cy + 1) + (at(cx + 1, cy + 1) - at(cx, cy + 1)) * sx;
    top + (bottom - top) * sy
}

/// Converts `0..=1` to a byte (clamped).
fn byte(value: f32) -> u8 {
    // Clamped to 0..=255 and rounded first; float-to-int `as` saturates anyway.
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// The tiling noise: red and green are independent three-octave value noises (blue a third),
/// opaque.
fn noise_image() -> ColorImage {
    let fbm = |x: usize, y: usize, seed: u32| (value_noise(x, y, 8, seed) * 0.57 + value_noise(x, y, 16, seed + 1) * 0.29 + value_noise(x, y, 32, seed + 2) * 0.14).clamp(0.0, 1.0);
    let pixels = (0..TILE_TEXELS * TILE_TEXELS)
        .map(|index| {
            let (x, y) = (index % TILE_TEXELS, index / TILE_TEXELS);
            Color32::from_rgb(byte(fbm(x, y, 11)), byte(fbm(x, y, 29)), byte(fbm(x, y, 47)))
        })
        .collect();
    ColorImage::new([TILE_TEXELS, TILE_TEXELS], pixels)
}

/// Height (texels) of the glass surface at texel (`x`, `y`): flattened spherical domes on a
/// staggered grid (4 x 4 cells of 64 texels, odd rows shifted by half a cell) plus a faint
/// ripple. Periodic over the tile.
fn height(x: usize, y: usize) -> f32 {
    const CELL: f32 = 64.0;
    const RADIUS: f32 = 30.0;
    let tile = 256.0_f32;
    // Texel coordinates are below 256: exact in f32.
    #[allow(clippy::cast_precision_loss)]
    let (px, py) = (x as f32, y as f32);
    let mut best = 0.0_f32;
    for row in -1..=4_i8 {
        let shift = if row.rem_euclid(2) == 1 { CELL * 0.5 } else { 0.0 };
        for column in -1..=4_i8 {
            let center = (f32::from(column) * CELL + CELL * 0.5 + shift, f32::from(row) * CELL + CELL * 0.5);
            // Shortest distance on the torus, so domes crossing the tile edge wrap.
            let wrap = |d: f32| d - tile * (d / tile).round();
            let (dx, dy) = (wrap(px - center.0), wrap(py - center.1));
            best = best.max((RADIUS * RADIUS - dx * dx - dy * dy).max(0.0).sqrt() * 0.6);
        }
    }
    best + 1.5 * (TAU * 3.0 * px / tile).sin() * (TAU * 2.0 * py / tile).cos()
}

/// The normal map of `height`: rgb = normal * 0.5 + 0.5 (x right, y down, z out), opaque.
fn normal_map_image() -> ColorImage {
    let heights: Vec<f32> = (0..TILE_TEXELS * TILE_TEXELS).map(|index| height(index % TILE_TEXELS, index / TILE_TEXELS)).collect();
    let at = |x: usize, y: usize| heights[(y % TILE_TEXELS) * TILE_TEXELS + x % TILE_TEXELS];
    let pixels = (0..TILE_TEXELS * TILE_TEXELS)
        .map(|index| {
            let (x, y) = (index % TILE_TEXELS, index / TILE_TEXELS);
            // Central differences with wrap-around (`+ TILE_TEXELS - 1` = one step back).
            let gx = (at(x + 1, y) - at(x + TILE_TEXELS - 1, y)) * 0.5;
            let gy = (at(x, y + 1) - at(x, y + TILE_TEXELS - 1)) * 0.5;
            let length = (gx * gx + gy * gy + 1.0).sqrt();
            let (nx, ny, nz) = (-gx / length, -gy / length, 1.0 / length);
            Color32::from_rgb(byte(nx * 0.5 + 0.5), byte(ny * 0.5 + 0.5), byte(nz * 0.5 + 0.5))
        })
        .collect();
    ColorImage::new([TILE_TEXELS, TILE_TEXELS], pixels)
}

/// The 16^3 LUT of `grade`, flattened: texel (r + 16 * b, g) holds grade((r, g, b) / 15).
fn lut_image(grade: Grade) -> ColorImage {
    let width = LUT_SIDE * LUT_SIDE;
    let pixels = (0..width * LUT_SIDE)
        .map(|index| {
            let (x, g) = (index % width, index / width);
            let (r, b) = (x % LUT_SIDE, x / LUT_SIDE);
            // Indices are below 16: exact in f32.
            #[allow(clippy::cast_precision_loss)]
            let unit = |i: usize| i as f32 / (LUT_SIDE - 1) as f32;
            let [gr, gg, gb] = grade.apply([unit(r), unit(g), unit(b)]);
            Color32::from_rgb(byte(gr), byte(gg), byte(gb))
        })
        .collect();
    ColorImage::new([width, LUT_SIDE], pixels)
}
