/*
File: examples/demo/scenes/stack.rs

Purpose:
Scene "stack": the preset stack editor over one draggable / resizable filter rect above the
shared backdrop and a scroll area of real widgets. Honours `--stack` and `--rect`; while
recording, the rect follows `record_rect`.

Key structures:
- StackScene: the rect and the `StackEditor`.
- ENTRY: the registry entry (`--scene stack`).
*/

use eframe::egui::{self, Rect, pos2, vec2};
use egui_filter_layers::FilterLayer;

use super::{DragRect, Scene, SceneEntry, SceneFrame, SceneInit, record_rect};
use crate::backdrop::{CircleMotion, scroll_widgets};
use crate::stack::StackEditor;

/// Registry entry.
pub const ENTRY: SceneEntry = SceneEntry { key: "stack", title: "Preset stack", build };

/// The preset-stack scene.
struct StackScene {
    /// The filter rect and its corner radius.
    rect: DragRect,
    /// The presets applied to it.
    editor: StackEditor,
}

/// Builds the scene from `--rect` / `--stack` (defaults otherwise).
fn build(init: &SceneInit<'_>) -> Box<dyn Scene> {
    let rect = init.rect.unwrap_or_else(|| Rect::from_min_size(pos2(420.0, 140.0), vec2(420.0, 300.0)));
    Box::new(StackScene { rect: DragRect { rect, corner_radius: 18 }, editor: StackEditor::new(init.stack.map(<[_]>::to_vec)) })
}

impl Scene for StackScene {
    fn controls(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("corner radius");
            ui.add(egui::Slider::new(&mut self.rect.corner_radius, 0..=80));
        });
        self.editor.ui(ui);
    }

    fn ui(&mut self, ui: &mut egui::Ui, frame: &SceneFrame<'_>) {
        let area = ui.max_rect();
        let painter = ui.painter_at(area);
        frame.backdrop.paint(&painter, area, CircleMotion::for_frame(frame.time, frame.phase));
        if let Some(phase) = frame.phase {
            self.rect.rect = record_rect(area, phase);
        }
        scroll_widgets(ui, area);
        self.rect.interact(ui, "filter");
        FilterLayer::new(self.rect.rect, self.editor.chain()).corner_radius(self.rect.corner_radius).paint(&painter);
        self.rect.paint_outline(&painter);
    }
}
