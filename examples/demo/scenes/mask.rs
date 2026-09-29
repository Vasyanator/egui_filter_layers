/*
File: examples/demo/scenes/mask.rs

Purpose:
Scene "mask": shape-mask glass (F9). Three draggable pieces of glass over the shared backdrop,
each its own shader layer masked with `Mask::Shape`: a five-pointed star built from convex pieces
(egui fills convex paths only), a circle and a text galley ("GLASS"). The side panel toggles each
shape and picks the effect (frosted glass, lens refraction under a light frost, or pixelation).

Key structures:
- MaskScene: the three `GlassShape`s and the effect controls.
- GlassShape / ShapeKind: one draggable masked layer (kind, centre offset, visibility).
- Look: the effect selector.
- ENTRY: the registry entry (`--scene mask`).

Key functions:
- star_pieces(): the star as overlapping convex kites (seam-free coverage, see its comment).
- GlassShape::mask_shape(): the shape in scene points, opaque white (full coverage).

Notes:
Each layer's rect is its shape's bounding rect (plus a margin): the mask is always intersected
with the layer rect, and a tight rect keeps the capture small. Layer ids are not needed (no state
passes). While recording, the shapes swing on paths periodic in the loop phase.
*/

use std::f32::consts::{PI, TAU};
use std::sync::Arc;

use eframe::egui::{self, Color32, FontId, Galley, Pos2, Sense, Shape, Stroke, Vec2, vec2};
use egui_shader_layers::{FilterChain, ShaderLayer, Mask, presets};

use super::{Scene, SceneEntry, SceneFrame, SceneInit};
use crate::backdrop::CircleMotion;

/// Registry entry.
pub const ENTRY: SceneEntry = SceneEntry { key: "mask", title: "Shape-mask glass", build };

/// Star tip radius, in points.
const STAR_RADIUS: f32 = 115.0;
/// Circle radius, in points.
const CIRCLE_RADIUS: f32 = 90.0;
/// Font size of the text mask, in points.
const TEXT_SIZE: f32 = 150.0;

/// Which shape a piece of glass has.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ShapeKind {
    /// A five-pointed star from convex pieces.
    Star,
    /// A circle.
    Circle,
    /// The text "GLASS" (a galley: glyph coverage from egui's font atlas).
    Text,
}

/// The effect under every mask.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Look {
    /// `presets::frosted_glass`.
    Frosted,
    /// `presets::lens` after a light blur.
    Lens,
    /// `presets::pixelate`.
    Pixelate,
}

/// One draggable masked layer.
#[derive(Clone, Copy, Debug)]
struct GlassShape {
    /// Its shape.
    kind: ShapeKind,
    /// Centre relative to the scene area's centre, in points (moved by dragging).
    offset: Vec2,
    /// Whether it is shown.
    visible: bool,
}

impl GlassShape {
    /// The mask shape centred at `center`, opaque white (full-strength filter). `galley` is the
    /// laid-out text, used by `ShapeKind::Text` only.
    fn mask_shape(self, center: Pos2, galley: &Arc<Galley>) -> Shape {
        match self.kind {
            ShapeKind::Star => Shape::Vec(star_pieces(center, STAR_RADIUS).into_iter().map(|points| Shape::convex_polygon(points, Color32::WHITE, Stroke::NONE)).collect()),
            ShapeKind::Circle => Shape::circle_filled(center, CIRCLE_RADIUS, Color32::WHITE),
            ShapeKind::Text => Shape::galley(center - galley.size() * 0.5, Arc::clone(galley), Color32::WHITE),
        }
    }

    /// A faint outline over the filtered shape (painted after the layer).
    fn outline(self, center: Pos2, galley: &Arc<Galley>) -> Shape {
        let stroke = Stroke::new(1.5, Color32::from_white_alpha(170));
        match self.kind {
            ShapeKind::Star => Shape::closed_line(star_outline(center, STAR_RADIUS), stroke),
            ShapeKind::Circle => Shape::circle_stroke(center, CIRCLE_RADIUS, stroke),
            ShapeKind::Text => Shape::galley_with_override_text_color(center - galley.size() * 0.5, Arc::clone(galley), Color32::from_white_alpha(28)),
        }
    }

    /// While recording: a swing periodic in `phase`, different per shape.
    fn record_swing(self, phase: f32) -> Vec2 {
        let turn = TAU * phase;
        match self.kind {
            ShapeKind::Star => vec2(40.0 * turn.sin(), 25.0 * (2.0 * turn).sin()),
            ShapeKind::Circle => vec2(-60.0 * turn.cos(), 40.0 * turn.sin()),
            ShapeKind::Text => vec2(70.0 * turn.sin(), 0.0),
        }
    }
}

/// The shape-mask glass scene.
struct MaskScene {
    /// Star, circle, text (fixed order; drawn in that order, so text is on top).
    shapes: [GlassShape; 3],
    /// Effect under every mask.
    look: Look,
    /// Frost blur sigma, points.
    sigma: f32,
    /// Frost tint and its strength.
    tint: Color32,
    /// Tint strength, `0..1`.
    tint_amount: f32,
    /// Lens magnification.
    magnification: f32,
    /// Pixel cell size, points.
    cell: f32,
}

/// Builds the scene (no GPU resources; the galley is laid out per frame from egui's cache).
fn build(_init: &SceneInit<'_>) -> Box<dyn Scene> {
    Box::new(MaskScene {
        shapes: [
            GlassShape { kind: ShapeKind::Star, offset: vec2(-220.0, -70.0), visible: true },
            GlassShape { kind: ShapeKind::Circle, offset: vec2(190.0, -85.0), visible: true },
            GlassShape { kind: ShapeKind::Text, offset: vec2(0.0, 140.0), visible: true },
        ],
        look: Look::Frosted,
        sigma: 7.0,
        tint: Color32::from_rgb(235, 245, 255),
        tint_amount: 0.18,
        magnification: 1.6,
        cell: 10.0,
    })
}

impl MaskScene {
    /// The chain every masked layer runs.
    fn chain(&self) -> FilterChain {
        match self.look {
            Look::Frosted => presets::frosted_glass(self.sigma, self.tint, self.tint_amount),
            Look::Lens => presets::gaussian_blur(self.sigma * 0.3).then(presets::lens(self.magnification)).then(presets::tint(self.tint, self.tint_amount)),
            Look::Pixelate => FilterChain::from(presets::pixelate(self.cell)),
        }
    }
}

impl Scene for MaskScene {
    fn controls(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            for shape in &mut self.shapes {
                let name = match shape.kind {
                    ShapeKind::Star => "star",
                    ShapeKind::Circle => "circle",
                    ShapeKind::Text => "text",
                };
                ui.checkbox(&mut shape.visible, name);
            }
        });
        ui.horizontal_wrapped(|ui| {
            ui.selectable_value(&mut self.look, Look::Frosted, "frosted");
            ui.selectable_value(&mut self.look, Look::Lens, "lens");
            ui.selectable_value(&mut self.look, Look::Pixelate, "pixelate");
        });
        match self.look {
            Look::Frosted | Look::Lens => {
                ui.add(egui::Slider::new(&mut self.sigma, 0.0..=24.0).text("frost sigma (pt)"));
                ui.horizontal(|ui| {
                    ui.label("tint");
                    ui.color_edit_button_srgba(&mut self.tint);
                });
                ui.add(egui::Slider::new(&mut self.tint_amount, 0.0..=1.0).text("tint amount"));
                if self.look == Look::Lens {
                    ui.add(egui::Slider::new(&mut self.magnification, 0.5..=3.0).text("magnification"));
                }
            }
            Look::Pixelate => {
                ui.add(egui::Slider::new(&mut self.cell, 2.0..=40.0).text("cell (pt)"));
            }
        }
        ui.weak("Drag a shape to move it. The star is five overlapping convex kites.");
    }

    fn ui(&mut self, ui: &mut egui::Ui, frame: &SceneFrame<'_>) {
        let area = ui.max_rect();
        let painter = ui.painter_at(area);
        frame.backdrop.paint(&painter, area, CircleMotion::for_frame(frame.time, frame.phase));
        let galley = painter.layout_no_wrap("GLASS".to_owned(), FontId::proportional(TEXT_SIZE), Color32::WHITE);
        let chain = self.chain();
        for (index, shape) in self.shapes.iter_mut().enumerate() {
            if !shape.visible {
                continue;
            }
            let swing = frame.phase.map_or(Vec2::ZERO, |phase| shape.record_swing(phase));
            let center = area.center() + shape.offset + swing;
            let mask = shape.mask_shape(center, &galley);
            let bounds = mask.visual_bounding_rect().expand(2.0);
            // Drag anywhere in the shape's bounds (a simple, forgiving hit area).
            let response = ui.interact(bounds, ui.id().with(("glass", index)), Sense::drag()).on_hover_cursor(egui::CursorIcon::Grab);
            if response.dragged() {
                shape.offset += response.drag_delta();
            }
            ShaderLayer::new(bounds, chain.clone()).mask(Mask::Shape(mask)).paint(&painter);
            painter.add(shape.outline(center, &galley));
        }
    }
}

/// The star's outline, clockwise from the top tip: tip, notch, tip, ... (tip radius `radius`,
/// notch radius `radius / 2`).
fn star_outline(center: Pos2, radius: f32) -> Vec<Pos2> {
    (0..10_u8)
        .map(|index| {
            let angle = -PI / 2.0 + f32::from(index) * PI / 5.0;
            let r = if index % 2 == 0 { radius } else { radius * 0.5 };
            center + vec2(angle.cos(), angle.sin()) * r
        })
        .collect()
}

/// The star as five convex kites, one per tip, each clockwise: (back, notch before, tip, notch
/// after). The back vertex sits 0.3 x the notch radius PAST the centre, opposite the tip, which
/// widens each kite's angle at its notches (about 126 degrees) beyond half the star's interior
/// angle there (about 118): neighbouring kites overlap in a thin wedge, so every seam lies inside
/// a neighbour at full coverage. Kites that only met along shared edges would leave a line of
/// about 75 % coverage (two feathered edges blended "over" each other). The notch radius (half the
/// tip radius) keeps the kites convex and inside the star.
fn star_pieces(center: Pos2, radius: f32) -> Vec<Vec<Pos2>> {
    let notch_radius = radius * 0.5;
    let at = |angle: f32, r: f32| center + vec2(angle.cos(), angle.sin()) * r;
    (0..5_u8)
        .map(|tip| {
            let angle = -PI / 2.0 + f32::from(tip) * TAU / 5.0;
            let back = at(angle + PI, 0.3 * notch_radius);
            vec![back, at(angle - PI / 5.0, notch_radius), at(angle, radius), at(angle + PI / 5.0, notch_radius)]
        })
        .collect()
}
