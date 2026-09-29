/*
File: examples/demo/backdrop.rs

Purpose:
The busy content scenes paint BENEATH their shader layers: a gradient, stripes, moving circles,
large text, a generated image, and a scroll area of real widgets. Shared by every scene so filters
always have detail to act on.

Key structures:
- Backdrop: owns the generated image texture; `paint` draws the shapes.
- CircleMotion: how the circles move (steady rotation, or a periodic swing while recording).

Key functions:
- Backdrop::paint(), scroll_widgets(), generate_pattern().
*/

use eframe::egui::{self, Color32, FontId, Pos2, Rect, Stroke, pos2, vec2};

/// How the backdrop circles move.
#[derive(Clone, Copy, Debug)]
pub enum CircleMotion {
    /// Interactive: rotate steadily with egui time (seconds).
    Rotate(f64),
    /// Recording: swing along the same curve and back once per loop; loop phase `0..1`.
    Swing(f32),
}

impl CircleMotion {
    /// The motion for a frame at `time`: a swing when a recording loop `phase` is given.
    pub fn for_frame(time: f64, phase: Option<f32>) -> Self {
        phase.map_or(Self::Rotate(time), Self::Swing)
    }
}

/// The shared backdrop: its one GPU resource is the generated test image.
pub struct Backdrop {
    /// The generated image shown in the backdrop.
    texture: egui::TextureHandle,
}

impl Backdrop {
    /// Uploads the generated test image.
    pub fn new(ctx: &egui::Context) -> Self {
        Self { texture: ctx.load_texture("demo-pattern", generate_pattern(256), egui::TextureOptions::LINEAR) }
    }

    /// Paints the busy backdrop into `area`: a gradient, stripes, moving circles, large text and
    /// the generated image.
    pub fn paint(&self, painter: &egui::Painter, area: Rect, motion: CircleMotion) {
        let mut mesh = egui::Mesh::default();
        for (pos, color) in [
            (area.left_top(), Color32::from_rgb(28, 30, 80)),
            (area.right_top(), Color32::from_rgb(120, 30, 90)),
            (area.right_bottom(), Color32::from_rgb(230, 140, 40)),
            (area.left_bottom(), Color32::from_rgb(20, 120, 110)),
        ] {
            mesh.colored_vertex(pos, color);
        }
        mesh.add_triangle(0, 1, 2);
        mesh.add_triangle(0, 2, 3);
        painter.add(mesh);

        // Stripes and circles give blur / refraction something to act on.
        for i in 0..24_u8 {
            let x = area.left() + f32::from(i) * 48.0;
            painter.line_segment([pos2(x, area.top()), pos2(x + 200.0, area.bottom())], Stroke::new(3.0, Color32::from_white_alpha(40)));
        }
        let shift = match motion {
            #[allow(clippy::cast_possible_truncation)] // A demo animation phase; precision is irrelevant.
            CircleMotion::Rotate(time) => time as f32 * 0.4,
            // Periodic for any curve shape, unlike the rotation (whose x / y rates are 1 : 1.3).
            CircleMotion::Swing(phase) => 0.8 * (std::f32::consts::TAU * phase).sin(),
        };
        for i in 0..10_u8 {
            let phase = f32::from(i) * 0.63 + shift;
            let center = area.center() + vec2(phase.cos() * 280.0, (phase * 1.3).sin() * 180.0);
            let hue = f32::from(i) / 10.0;
            painter.circle_filled(center, 28.0 + f32::from(i) * 3.0, Color32::from(egui::ecolor::Hsva::new(hue, 0.8, 0.95, 1.0)));
        }
        painter.text(area.left_top() + vec2(40.0, 40.0), egui::Align2::LEFT_TOP, "Shader layers", FontId::proportional(64.0), Color32::WHITE);
        painter.text(
            area.left_top() + vec2(44.0, 120.0),
            egui::Align2::LEFT_TOP,
            "Everything painted before the layer is filtered.\nThe outline and handle are painted after it.",
            FontId::proportional(20.0),
            Color32::from_rgb(240, 240, 200),
        );
        let image_rect = Rect::from_min_size(area.left_top() + vec2(460.0, 200.0), vec2(256.0, 256.0));
        painter.image(self.texture.id(), image_rect, Rect::from_min_max(Pos2::ZERO, pos2(1.0, 1.0)), Color32::WHITE);
    }
}

/// Real widgets beneath the filter (a framed scroll area of labelled rows in the lower left of
/// `area`), to show a layer filters whatever egui drew earlier, widgets included.
pub fn scroll_widgets(ui: &mut egui::Ui, area: Rect) {
    let scroll_rect = Rect::from_min_size(area.min + vec2(24.0, 330.0), vec2(360.0, (area.height() - 360.0).max(120.0)));
    let mut child = ui.new_child(egui::UiBuilder::new().max_rect(scroll_rect));
    egui::Frame::group(child.style()).show(&mut child, |ui| {
        egui::ScrollArea::vertical().id_salt("scene-scroll").show(ui, |ui| {
            for i in 0..60 {
                ui.horizontal(|ui| {
                    ui.label(format!("Row {i:02}: the quick brown fox jumps over the lazy dog."));
                    if ui.button("button").clicked() {
                        log::info!("demo: row {i} clicked");
                    }
                });
            }
        });
    });
}

/// A procedural `size` x `size` test image: hue gradient, checkerboard and rings.
fn generate_pattern(size: usize) -> egui::ColorImage {
    let mut pixels = Vec::with_capacity(size * size);
    let scale = 1.0 / f32::from(u16::try_from(size).unwrap_or(u16::MAX));
    for y in 0..size {
        for x in 0..size {
            let fx = f32::from(u16::try_from(x).unwrap_or(0)) * scale;
            let fy = f32::from(u16::try_from(y).unwrap_or(0)) * scale;
            let checker = ((x / 16) + (y / 16)) % 2 == 0;
            let ring = (((fx - 0.5).hypot(fy - 0.5) * 24.0).sin() * 0.5 + 0.5) * 0.35;
            let value = if checker { 0.95 } else { 0.6 } - ring;
            pixels.push(egui::ecolor::Hsva::new(fx, 0.35 + 0.6 * fy, value, 1.0).into());
        }
    }
    egui::ColorImage::new([size, size], pixels)
}
