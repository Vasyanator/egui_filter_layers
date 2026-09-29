/*
File: examples/demo/scenes/grid.rs

Purpose:
Scene "grid": the many-elements benchmark (roadmap §19.5, R2) made visible. N small frosted glass
tiles, each with its own tint, over the shared backdrop, painted in a selectable `GridMode`: N
separate shader layers, or ONE layer whose `Mask::Regions` has one region per tile (the tint is
picked by `region()` in the shader). The side panel shows the shared `FrameTimer` readout and how
many shader layers (= egui paint callbacks) the mode paints per frame.

Key structures:
- GridScene: controls (mode, N, blur, tint strength), the compiled tint effect, the tint pass
  cached per (N, strength), the frame timer.
- GridMode: how the tiles are painted (the extension point for the entry modes of phase R2b).
- ENTRY: the registry entry (`--scene grid`).

Key functions:
- tile_rects(): the grid layout, tiles pulsing (periodic in the record phase).
- tile_tint(): the per-tile tint colour.

Notes:
Both modes run the same chain (a Gaussian blur, then `TINT_WGSL`): a layer without regions reads
`region() == REGION_NONE` and falls back to its own `p.index`, so the modes differ only in how
many layers are painted. N is capped at `MAX_REGIONS` (region ids are one byte).
*/

use std::f32::consts::TAU;

use eframe::egui::{self, Color32, CornerRadius, Rect, Stroke, StrokeKind, vec2};
use egui_shader_layers::{Effect, EffectError, FilterChain, MAX_REGIONS, Pass, Region, ShaderLayer, presets};

use super::{FrameTimer, Scene, SceneEntry, SceneFrame, SceneInit};
use crate::backdrop::CircleMotion;

/// Registry entry.
pub const ENTRY: SceneEntry = SceneEntry { key: "grid", title: "Many elements (grid)", build };

/// Tint slots of `TINT_WGSL` (ids `0..MAX_REGIONS` fit).
const TINT_SLOTS: usize = 256;

/// Tile corner radius, points.
const TILE_CORNER: u8 = 6;

/// The tint pass: mixes the blurred backdrop towards `p.tint[i].rgb` by `p.tint[i].a`, `i` = the
/// pixel's region id, or `p.index` on a layer without regions.
const TINT_WGSL: &str = "struct Params { tint: array<vec4<f32>, 256>, index: u32 }
fn effect(uv: vec2<f32>) -> vec4<f32> {
    let r = region();
    let i = min(select(r, p.index, r == REGION_NONE), 255u);
    let t = p.tint[i];
    let c = unpremultiply(src(uv));
    return premultiply(vec4<f32>(mix(c.rgb, t.rgb, t.a), c.a));
}";

/// How the tiles are painted. Phase R2b adds the entry modes (one layer, one entry per tile,
/// independent / stacked) here; `GridScene::paint_tiles` is the one place that dispatches on it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GridMode {
    /// One `ShaderLayer` per tile.
    Layers,
    /// One `ShaderLayer` over the tiles' union, one region per tile.
    Regions,
}

impl GridMode {
    /// Every mode, in selector order, with its label.
    const ALL: [(Self, &'static str); 2] = [(Self::Layers, "N layers"), (Self::Regions, "1 layer, N regions")];

    /// Shader layers (= egui paint callbacks) painted per frame for `count` tiles.
    fn layers_painted(self, count: usize) -> usize {
        match self {
            Self::Layers => count,
            Self::Regions => 1,
        }
    }
}

/// The grid scene.
struct GridScene {
    /// How the tiles are painted.
    mode: GridMode,
    /// Tile count, `4..=MAX_REGIONS`.
    count: usize,
    /// Blur sigma, points.
    sigma: f32,
    /// Tint strength, `0..1`.
    tint_amount: f32,
    /// The tint effect, compiled once.
    effect: Result<Effect, EffectError>,
    /// The tint pass with every tile's tint set, and the (count, strength) it was built for.
    tint_pass: Option<(usize, f32, Pass)>,
    /// The averaged frame-time readout.
    frame_timer: FrameTimer,
}

/// Builds the scene (compiles the tint effect: cheap naga validation).
fn build(_init: &SceneInit<'_>) -> Box<dyn Scene> {
    let effect = Effect::new("demo_grid_tint", TINT_WGSL);
    if let Err(error) = &effect {
        log::error!("demo: the grid scene's tint effect did not compile: {error}");
    }
    Box::new(GridScene { mode: GridMode::Layers, count: 64, sigma: 4.0, tint_amount: 0.35, effect, tint_pass: None, frame_timer: FrameTimer::default() })
}

impl GridScene {
    /// The tint pass for the current count and strength (`p.index` = 0), rebuilt only when one of
    /// them changed. `Err` carries a message for the side panel.
    fn tint_pass(&mut self) -> Result<&Pass, String> {
        let stale = self.tint_pass.as_ref().is_none_or(|(count, amount, _)| *count != self.count || amount.to_bits() != self.tint_amount.to_bits());
        if stale {
            let effect = self.effect.as_ref().map_err(ToString::to_string)?;
            let tints: Vec<[f32; 4]> = (0..self.count.min(TINT_SLOTS)).map(|tile| tile_tint(tile, self.count, self.tint_amount)).collect();
            let mut pass = Pass::new(effect);
            pass.set_slice("tint", &tints).map_err(|error| error.to_string())?;
            self.tint_pass = Some((self.count, self.tint_amount, pass));
        }
        self.tint_pass.as_ref().map(|(_, _, pass)| pass).ok_or_else(|| "tint pass missing".to_owned())
    }

    /// Paints the tiles' glass in the current mode. The one dispatch on `GridMode`.
    fn paint_tiles(&mut self, painter: &egui::Painter, tiles: &[Rect]) -> Result<(), String> {
        let (mode, sigma) = (self.mode, self.sigma);
        let tint = self.tint_pass()?.clone();
        match mode {
            GridMode::Layers => {
                for (index, tile) in tiles.iter().enumerate() {
                    let mut pass = tint.clone();
                    pass.set("index", u32::try_from(index).unwrap_or(0)).map_err(|error| error.to_string())?;
                    ShaderLayer::new(*tile, presets::gaussian_blur(sigma).then(pass)).corner_radius(TILE_CORNER).paint(painter);
                }
            }
            GridMode::Regions => {
                let union = tiles.iter().fold(Rect::NOTHING, |union, tile| union.union(*tile));
                let chain: FilterChain = presets::gaussian_blur(sigma).then(tint);
                ShaderLayer::new(union, chain).regions(tiles.iter().map(|tile| Region::rounded_rect(*tile, TILE_CORNER))).paint(painter);
            }
        }
        Ok(())
    }
}

impl Scene for GridScene {
    fn controls(&mut self, ui: &mut egui::Ui) {
        ui.label("mode");
        ui.horizontal_wrapped(|ui| {
            for (mode, label) in GridMode::ALL {
                ui.selectable_value(&mut self.mode, mode, label);
            }
        });
        ui.add(egui::Slider::new(&mut self.count, 4..=MAX_REGIONS).logarithmic(true).text("elements (N)"));
        ui.add(egui::Slider::new(&mut self.sigma, 0.5..=12.0).text("blur sigma (pt)"));
        ui.add(egui::Slider::new(&mut self.tint_amount, 0.0..=1.0).text("tint strength"));
        ui.separator();
        ui.label(format!("shader layers (egui callbacks) per frame: {}", self.mode.layers_painted(self.count)));
        self.frame_timer.label(ui);
        if let Err(error) = self.tint_pass() {
            ui.colored_label(ui.visuals().error_fg_color, format!("tint pass unavailable: {error}"));
        }
        ui.weak("Same chain in both modes. Compare the frame time at large N (vsync caps it).");
    }

    fn ui(&mut self, ui: &mut egui::Ui, frame: &SceneFrame<'_>) {
        self.frame_timer.record(ui);
        let area = ui.max_rect();
        let painter = ui.painter_at(area);
        frame.backdrop.paint(&painter, area, CircleMotion::for_frame(frame.time, frame.phase));
        let tiles = tile_rects(area.shrink(24.0), self.count, frame.time, frame.phase);
        // An error is shown in the side panel (and a compile error logged once at build).
        if self.paint_tiles(&painter, &tiles).is_ok() {
            let stroke = Stroke::new(1.0, Color32::from_white_alpha(90));
            for tile in &tiles {
                painter.rect_stroke(*tile, CornerRadius::same(TILE_CORNER), stroke, StrokeKind::Outside);
            }
        }
    }
}

/// `count` tiles in a grid over `area` (about as many columns per row as the area is wide per
/// high), each pulsing in size: `phase` (recording) makes the pulse periodic in the loop,
/// otherwise it follows egui `time`.
fn tile_rects(area: Rect, count: usize, time: f64, phase: Option<f32>) -> Vec<Rect> {
    let aspect = (area.width() / area.height().max(1.0)).max(0.1);
    // Small positive integers (count <= 255): exact in f32 / usize.
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let rows = ((count as f32 / aspect).sqrt().round() as usize).max(1);
    let cols = count.div_ceil(rows);
    #[allow(clippy::cast_precision_loss)]
    let cell = vec2(area.width() / cols as f32, area.height() / rows as f32);
    // One pulse per recording loop, or one per ~3 s interactively; reduced modulo a turn in f64.
    #[allow(clippy::cast_possible_truncation)]
    let turn = phase.unwrap_or_else(|| (time / 3.0).rem_euclid(1.0) as f32) * TAU;
    (0..count)
        .map(|tile| {
            #[allow(clippy::cast_precision_loss)]
            let (col, row) = ((tile % cols) as f32, (tile / cols) as f32);
            let pulse = 0.5 + 0.5 * (turn + 0.45 * (col + row)).sin();
            let inset = (cell.min_elem() * (0.08 + 0.12 * pulse)).max(1.0);
            Rect::from_min_size(area.min + vec2(col * cell.x, row * cell.y), cell).shrink(inset)
        })
        .collect()
}

/// The tint of tile `index` of `count`: hues spread around the colour wheel (lightened), alpha =
/// `amount`.
fn tile_tint(index: usize, count: usize, amount: f32) -> [f32; 4] {
    // Both are at most 255: exact in f32.
    #[allow(clippy::cast_precision_loss)]
    let hue = index as f32 / count.max(1) as f32;
    let channel = |offset: f32| {
        let phase = (hue + offset).rem_euclid(1.0);
        (1.0 - (phase * 6.0 - 3.0).abs().min(3.0) / 1.5).clamp(0.0, 1.0) * 0.8 + 0.2
    };
    [channel(0.0), channel(2.0 / 3.0), channel(1.0 / 3.0), amount]
}
