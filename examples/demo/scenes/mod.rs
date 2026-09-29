/*
File: examples/demo/scenes/mod.rs

Purpose:
The scene registry and the contract every demo scene implements, plus small helpers several
scenes share (a draggable / resizable filter rect, the recording path of a moving rect).

Key structures:
- Scene: the trait (side-panel controls + the central-panel scene).
- SceneEntry / SCENES: the registry (command-line key, title, constructor), in selector order.
- SceneInit: what a constructor gets; SceneFrame: what a scene gets every frame.
- DragRect: a filter rect the user moves by dragging and resizes from its corner.
- FrameTimer: an averaged frame-time readout (egui `unstable_dt` over a window of frames).

Key functions:
- find(): registry lookup by key; record_rect(): the closed recording path.

Notes:
Adding a scene is a two-line change here: `mod name;` below and `name::ENTRY,` in `SCENES`.
The scene module defines `pub const ENTRY: SceneEntry` and its `Scene` implementation.
*/

mod bloom;
mod grid;
mod hot;
mod mask;
mod ripples;
mod stack;
mod textures;
mod trails;
mod water;

use std::collections::VecDeque;

use eframe::egui::{self, Color32, CornerRadius, Rect, Sense, Stroke, StrokeKind, Vec2, pos2, vec2};

use crate::backdrop::Backdrop;
use crate::stack::Entry;

/// Every scene, in selector order; the first one is the default.
pub const SCENES: &[SceneEntry] = &[stack::ENTRY, hot::ENTRY, ripples::ENTRY, water::ENTRY, trails::ENTRY, bloom::ENTRY, textures::ENTRY, mask::ENTRY, grid::ENTRY];

/// A registered scene: how to name it and how to build it.
#[derive(Clone, Copy, Debug)]
pub struct SceneEntry {
    /// Command-line name (`--scene`): lower-case, no spaces.
    pub key: &'static str,
    /// Selector label.
    pub title: &'static str,
    /// Builds the scene. Called lazily, the first time the scene is shown, on the GUI thread
    /// (so it must stay cheap: no file I/O or heavy work; start workers instead).
    pub build: fn(&SceneInit<'_>) -> Box<dyn Scene>,
}

/// The registered scene with command-line name `key`.
pub fn find(key: &str) -> Option<usize> {
    SCENES.iter().position(|entry| entry.key == key)
}

/// Command-line state a scene constructor may use. Options that do not apply to a scene are
/// ignored by it.
#[derive(Debug)]
pub struct SceneInit<'a> {
    /// The egui context (texture uploads, repaint handles for workers).
    pub ctx: &'a egui::Context,
    /// `--rect`: the initial filter rect, in points.
    pub rect: Option<Rect>,
    /// `--stack`: the initial preset stack.
    pub stack: Option<&'a [Entry]>,
}

/// What a scene gets every frame.
pub struct SceneFrame<'a> {
    /// The shared backdrop to paint beneath the scene's layers.
    pub backdrop: &'a Backdrop,
    /// egui time in seconds (pinned to `step / fps` while recording).
    pub time: f64,
    /// `Some(loop phase in 0..1)` while `--record` runs: every animation the scene drives itself
    /// (moving rects, pointer stand-ins, effect clocks) must then be periodic in this phase so
    /// the loop is seamless; `None` in interactive runs.
    pub phase: Option<f32>,
}

/// One demo scene. The app owns at most one instance per registry entry, built on first show.
pub trait Scene {
    /// Scene-specific controls in the side panel, below the shared status block. Not called
    /// while recording (no side panel then). Default: none.
    fn controls(&mut self, _ui: &mut egui::Ui) {}

    /// Paints the scene into the central panel: `ui.max_rect()` is the whole scene area. Called
    /// every frame the scene is selected; the app already requests a repaint every frame.
    fn ui(&mut self, ui: &mut egui::Ui, frame: &SceneFrame<'_>);
}

/// A filter rect the user drags to move and resizes from its bottom-right handle.
#[derive(Clone, Copy, Debug)]
pub struct DragRect {
    /// The rect, in points.
    pub rect: Rect,
    /// Rounded mask radius (points).
    pub corner_radius: u8,
}

impl DragRect {
    /// Size of the resize handle, in points.
    const HANDLE: f32 = 16.0;

    /// Smallest width / height the resize keeps, in points.
    const MIN_SIZE: f32 = 40.0;

    /// Handles dragging (move) and the corner handle (resize); `id_salt` keeps several rects
    /// in one scene apart.
    pub fn interact(&mut self, ui: &mut egui::Ui, id_salt: &str) {
        let move_response = ui.interact(self.rect, ui.id().with((id_salt, "move")), Sense::drag()).on_hover_cursor(egui::CursorIcon::Grab);
        if move_response.dragged() {
            self.rect = self.rect.translate(move_response.drag_delta());
        }
        let handle = Rect::from_center_size(self.rect.max, Vec2::splat(Self::HANDLE));
        let resize_response = ui.interact(handle, ui.id().with((id_salt, "resize")), Sense::drag()).on_hover_cursor(egui::CursorIcon::ResizeNwSe);
        if resize_response.dragged() {
            let max = self.rect.max + resize_response.drag_delta();
            self.rect.max = pos2(max.x.max(self.rect.min.x + Self::MIN_SIZE), max.y.max(self.rect.min.y + Self::MIN_SIZE));
        }
    }

    /// The outline and resize handle. Paint AFTER the shader layer so they stay crisp.
    pub fn paint_outline(&self, painter: &egui::Painter) {
        let stroke = Stroke::new(1.5, Color32::from_white_alpha(200));
        painter.rect_stroke(self.rect, CornerRadius::same(self.corner_radius), stroke, StrokeKind::Outside);
        painter.circle_filled(self.rect.max, Self::HANDLE * 0.4, Color32::WHITE);
    }
}

/// A rect's closed path in `--record` mode at loop phase `phase` (`0..1`): its centre is a sum of
/// sinusoids with integer frequencies (fixed "random" amplitudes and phases), so it returns
/// exactly to its start at phase 1 and, for a 280 x 190 rect, stays inside an 800 x 500 `area`.
pub fn record_rect(area: Rect, phase: f32) -> Rect {
    let size = vec2(280.0, 190.0);
    // (frequency, x amplitude, x phase, y amplitude, y phase); amplitudes sum to less than the
    // free half-space (260 x 155 points in the 800x500 window).
    let terms = [(1.0, 150.0, 0.3, 85.0, 1.9), (2.0, 60.0, 2.4, 40.0, 0.7), (3.0, 30.0, 4.1, 20.0, 5.2)];
    let turn = std::f32::consts::TAU * phase;
    let offset = terms.iter().fold(Vec2::ZERO, |sum, &(k, ax, px, ay, py)| sum + vec2(ax * (k * turn + px).sin(), ay * (k * turn + py).sin()));
    Rect::from_center_size(area.center() + offset, size)
}

/// An averaged frame-time readout: egui's `unstable_dt` (CPU + GPU wall time between frames,
/// capped by vsync) over the last [`FrameTimer::WINDOW`] frames.
#[derive(Debug, Default)]
pub struct FrameTimer {
    /// Recent frame durations in seconds, newest last; at most `WINDOW`.
    times: VecDeque<f32>,
}

impl FrameTimer {
    /// How many frames the average spans.
    pub const WINDOW: usize = 60;

    /// Records this frame's duration; call once per frame (from `Scene::ui`).
    pub fn record(&mut self, ui: &egui::Ui) {
        if self.times.len() == Self::WINDOW {
            self.times.pop_front();
        }
        self.times.push_back(ui.input(|input| input.unstable_dt));
    }

    /// The average of the recorded frame times, in milliseconds (`None` before the first frame).
    pub fn average_ms(&self) -> Option<f32> {
        let count = self.times.len();
        // `count` is at most WINDOW (60): exact as f32.
        #[allow(clippy::cast_precision_loss)]
        let count_f = count as f32;
        (count > 0).then(|| self.times.iter().sum::<f32>() / count_f * 1000.0)
    }

    /// The readout as a side-panel label: average milliseconds, frames averaged and fps.
    pub fn label(&self, ui: &mut egui::Ui) {
        match self.average_ms() {
            Some(ms) => ui.label(format!("frame time: {ms:.2} ms (avg of {} frames, {:.0} fps)", self.times.len(), 1000.0 / ms.max(0.001))),
            None => ui.label("frame time: -"),
        };
    }
}
