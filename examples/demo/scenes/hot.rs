/*
File: examples/demo/scenes/hot.rs

Purpose:
Scene "hot": a filter layer whose effect follows `examples/demo/scenes/hot.wgsl` on disk through
`HotEffect` (feature `hot-reload`), with an in-app WGSL editor that saves to that file, compile
errors shown on the layer and in the side panel, and the named-parameter editor whose values
survive reloads (`rebind`).

Key structures:
- HotScene: the watcher, the pass built on its current effect, the editor buffer, the pending
  file job.
- FileJob / FileOutcome: one file read or write running on a worker thread.
- ENTRY: the registry entry (`--scene hot`).

Notes:
- The path is resolved at COMPILE time from `CARGO_MANIFEST_DIR` (`HOT_WGSL_PATH`), so the scene
  edits the source tree's file whatever the working directory; a demo binary moved to another
  machine watches a path that may not exist there (the scene then shows "missing").
- No file I/O on the GUI thread: the initial read, every save and the re-read after an outside
  change run on short-lived std threads and report through a channel. A read is applied only if
  the buffer was not edited since it was requested (`edits` counter), so it never eats typing.
- The editor sits in the left part of the scene area (hidden while recording); the layer rect is
  draggable on the right. While recording, the rect follows `record_rect`; the effect is compiled
  asynchronously, so the first recorded frames may precede the first compile.
*/

use std::sync::mpsc::{self, Receiver, TryRecvError};

use eframe::egui::{self, Align2, Color32, CornerRadius, FontId, Key, KeyboardShortcut, Modifiers, Rect, Stroke, StrokeKind, UiBuilder, pos2, vec2};
use egui_filter_layers::{Effect, FilterLayer, HotEffect, HotOptions, HotReloadError, HotReloadEvent, ParamType, Pass, rebind};

use super::{DragRect, Scene, SceneEntry, SceneFrame, SceneInit, record_rect};
use crate::backdrop::CircleMotion;
use crate::params_ui::pass_params_ui;

/// Registry entry.
pub const ENTRY: SceneEntry = SceneEntry { key: "hot", title: "Hot-reload editor", build };

/// The watched WGSL file: the source tree's `examples/demo/scenes/hot.wgsl`, resolved from
/// `CARGO_MANIFEST_DIR` when the demo is compiled.
const HOT_WGSL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/demo/scenes/hot.wgsl");

/// How far the starter effect samples (amplitude + split at their largest useful values), in
/// points. The file holds only WGSL, so this is set through `HotOptions::with_setup`.
const SAMPLE_RADIUS: f32 = 24.0;

/// Starting values of the starter effect's named parameters, applied to the first pass only
/// (later passes are rebound and keep the user's values). Names the current file does not
/// declare are skipped. `speed` is pi: a 2 s period, so the default 4 s recording loops.
const STARTER_VALUES: &[(&str, [f32; 4])] = &[("amplitude", [6.0; 4]), ("frequency", [3.0; 4]), ("speed", [std::f32::consts::PI; 4]), ("split", [2.0; 4]), ("tint_color", [0.0, 0.06, 0.14, 0.18])];

/// Fraction of the scene area's width given to the editor.
const EDITOR_FRACTION: f32 = 0.45;

/// Save shortcut (Ctrl+S, Cmd+S on macOS).
const SAVE_SHORTCUT: KeyboardShortcut = KeyboardShortcut::new(Modifiers::COMMAND, Key::S);

/// What a worker thread reports back.
#[derive(Debug)]
enum FileOutcome {
    /// The file's text (or the error message of the read).
    Read(Result<String, String>),
    /// The write finished (or failed with this message); carries the text written.
    Written(Result<String, String>),
}

/// One file operation in flight.
#[derive(Debug)]
struct FileJob {
    /// Where the worker reports.
    outcome: Receiver<FileOutcome>,
    /// `HotScene::edits` when the job started (a read is applied only if it is unchanged).
    edits: u64,
}

/// The hot-reload scene.
struct HotScene {
    /// The file watcher, or why it could not start.
    hot: Result<HotEffect, HotReloadError>,
    /// The pass built on the watcher's current effect (`None` before the first compile).
    pass: Option<Pass>,
    /// Named parameters the last reload could not carry over (renamed, retyped or removed).
    dropped: Vec<String>,
    /// The filter rect.
    rect: DragRect,
    /// The editor buffer.
    source: String,
    /// The text last read from or written to the file (`None` until the first read lands).
    on_disk: Option<String>,
    /// Bumped on every edit of `source`.
    edits: u64,
    /// The file operation in flight, if any (one at a time).
    job: Option<FileJob>,
    /// Outcome of the last file operation, for the side panel.
    file_status: Result<String, String>,
    /// Used by workers to request a repaint when they finish.
    ctx: egui::Context,
}

/// Starts watching `HOT_WGSL_PATH` and reading it into the editor (both off the GUI thread).
fn build(init: &SceneInit<'_>) -> Box<dyn Scene> {
    let options = HotOptions::default().with_repaint(init.ctx).with_setup(|effect| effect.with_sample_radius(SAMPLE_RADIUS));
    let hot = HotEffect::watch_with("hot", HOT_WGSL_PATH, options);
    if let Err(error) = &hot {
        log::error!("demo: could not watch {HOT_WGSL_PATH}: {error}");
    }
    let rect = init.rect.unwrap_or_else(|| Rect::from_min_size(pos2(560.0, 120.0), vec2(360.0, 300.0)));
    let mut scene = HotScene {
        hot,
        pass: None,
        dropped: Vec::new(),
        rect: DragRect { rect, corner_radius: 14 },
        source: String::new(),
        on_disk: None,
        edits: 0,
        job: None,
        file_status: Ok("reading the file...".to_owned()),
        ctx: init.ctx.clone(),
    };
    scene.start_read();
    Box::new(scene)
}

impl HotScene {
    /// Whether the editor holds changes not yet written to the file.
    fn dirty(&self) -> bool {
        self.on_disk.as_deref() != Some(self.source.as_str())
    }

    /// Runs `work` on a worker thread as the scene's file job; a job already in flight wins
    /// (the caller retries later). A thread that cannot be spawned is reported in the panel.
    fn start_job(&mut self, name: &str, work: impl FnOnce() -> FileOutcome + Send + 'static) {
        if self.job.is_some() {
            return;
        }
        let (sender, outcome) = mpsc::channel();
        let ctx = self.ctx.clone();
        let spawned = std::thread::Builder::new().name(format!("demo-hot-{name}")).spawn(move || {
            // The receiver is gone only when the scene was dropped: nobody to report to.
            if sender.send(work()).is_ok() {
                ctx.request_repaint();
            }
        });
        match spawned {
            Ok(handle) => {
                // Detached: the worker reports through the channel and exits on its own.
                drop(handle);
                self.job = Some(FileJob { outcome, edits: self.edits });
            }
            Err(error) => {
                log::error!("demo: could not start the {name} thread for {HOT_WGSL_PATH}: {error}");
                self.file_status = Err(format!("could not start a worker thread: {error}"));
            }
        }
    }

    /// Reads the file into the editor (applied only if the buffer is not edited meanwhile).
    fn start_read(&mut self) {
        self.start_job("read", || FileOutcome::Read(std::fs::read_to_string(HOT_WGSL_PATH).map_err(|error| error.to_string())));
    }

    /// Writes the editor buffer to the file.
    fn start_save(&mut self) {
        let text = self.source.clone();
        self.start_job("write", move || FileOutcome::Written(std::fs::write(HOT_WGSL_PATH, &text).map(|()| text).map_err(|error| error.to_string())));
    }

    /// Collects a finished file job (never blocks).
    fn poll_job(&mut self) {
        let Some(job) = &self.job else {
            return;
        };
        let outcome = match job.outcome.try_recv() {
            Ok(outcome) => outcome,
            Err(TryRecvError::Empty) => return,
            Err(TryRecvError::Disconnected) => {
                // The worker panicked before reporting.
                self.job = None;
                self.file_status = Err("the file worker ended without a result".to_owned());
                return;
            }
        };
        let untouched = job.edits == self.edits;
        self.job = None;
        match outcome {
            FileOutcome::Read(Ok(text)) => {
                if untouched {
                    self.source.clone_from(&text);
                    self.file_status = Ok("loaded from disk".to_owned());
                } else if self.on_disk.as_deref() != Some(text.as_str()) {
                    // Typed while the read ran and the file really changed: keep the typing.
                    self.file_status = Ok("the file changed on disk; the editor keeps your edits (Save overwrites it)".to_owned());
                }
                self.on_disk = Some(text);
            }
            FileOutcome::Read(Err(message)) => {
                log::warn!("demo: could not read {HOT_WGSL_PATH}: {message}");
                self.file_status = Err(format!("read failed: {message}"));
            }
            FileOutcome::Written(Ok(text)) => {
                self.on_disk = Some(text);
                self.file_status = Ok("saved; recompiling in the background".to_owned());
            }
            FileOutcome::Written(Err(message)) => {
                log::error!("demo: could not write {HOT_WGSL_PATH}: {message}");
                self.file_status = Err(format!("save failed: {message}"));
            }
        }
    }

    /// Takes the watcher's news: rebuilds the pass on a new effect (keeping the named values
    /// that still fit) and, after any change on disk, re-reads the file into a clean editor.
    fn poll_hot(&mut self) {
        let Ok(hot) = &mut self.hot else {
            return;
        };
        let event = hot.poll();
        if event == HotReloadEvent::Reloaded
            && let Some(effect) = hot.current()
        {
            self.pass = Some(match &self.pass {
                Some(old) => {
                    let rebound = rebind(old, effect);
                    self.dropped = rebound.dropped;
                    rebound.pass
                }
                None => starter_pass(effect),
            });
        }
        // Something changed on disk (an outside editor, or our own save): keep a clean editor in
        // sync. A dirty buffer is left alone; its status says so when the read lands.
        if event != HotReloadEvent::Unchanged && !self.dirty() {
            self.start_read();
        }
    }

    /// The editor column: path, buttons, and the code editor filling the rest.
    fn editor_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let busy = self.job.is_some();
            if ui.add_enabled(!busy && self.dirty(), egui::Button::new("Save (Ctrl+S)")).clicked() {
                self.start_save();
            }
            if ui.add_enabled(!busy, egui::Button::new("Reload from disk")).on_hover_text("Discards unsaved edits").clicked() {
                // Replaces the buffer when it lands, unless the user types meanwhile.
                self.start_read();
            }
            ui.weak(if self.dirty() { "modified" } else { "saved" });
        });
        ui.weak(HOT_WGSL_PATH);
        egui::ScrollArea::vertical().id_salt("hot-source").auto_shrink([false, false]).show(ui, |ui| {
            let response = ui.add(egui::TextEdit::multiline(&mut self.source).code_editor().desired_width(f32::INFINITY).desired_rows(30));
            if response.changed() {
                self.edits += 1;
            }
        });
    }
}

/// The first pass on `effect`, with the starter file's defaults for the members it declares.
fn starter_pass(effect: &Effect) -> Pass {
    let mut pass = Pass::new(effect);
    for (name, value) in STARTER_VALUES {
        let Some(info) = effect.param(name) else {
            continue;
        };
        let result = if info.ty == ParamType::Vec4 { pass.set(name, *value) } else { pass.set(name, value[0]) };
        // A member of another type (the user retyped it): it simply starts at zero.
        if let Err(error) = result {
            log::info!("demo: starter value for `{name}` not applied: {error}");
        }
    }
    pass
}

impl Scene for HotScene {
    fn controls(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("corner radius");
            ui.add(egui::Slider::new(&mut self.rect.corner_radius, 0..=80));
        });
        ui.label("Edit the WGSL in the scene (or in any editor) and save: it recompiles in the background, and named parameter values survive the reload.");
        match &self.file_status {
            Ok(message) => ui.weak(format!("file: {message}")),
            Err(message) => ui.colored_label(ui.visuals().error_fg_color, format!("file: {message}")),
        };
        ui.separator();
        match &self.hot {
            Ok(hot) => match hot.error() {
                Some(_) => {
                    let color = ui.visuals().error_fg_color;
                    egui::ScrollArea::both().id_salt("hot-diagnostic").max_height(220.0).show(ui, |ui| {
                        ui.label(egui::RichText::new(hot.status_text()).monospace().color(color));
                    });
                }
                None => {
                    ui.label(hot.status_text());
                }
            },
            Err(error) => {
                ui.colored_label(ui.visuals().error_fg_color, format!("watcher not running: {error}"));
            }
        }
        if !self.dropped.is_empty() {
            ui.weak(format!("reset by the last reload (name or type changed): {}", self.dropped.join(", ")));
        }
        ui.separator();
        ui.label("Named parameters");
        match &mut self.pass {
            Some(pass) => {
                pass_params_ui(ui, pass);
            }
            None => {
                ui.weak("no effect compiled yet");
            }
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, frame: &SceneFrame<'_>) {
        self.poll_job();
        self.poll_hot();
        let area = ui.max_rect();
        let painter = ui.painter_at(area);
        frame.backdrop.paint(&painter, area, CircleMotion::for_frame(frame.time, frame.phase));
        if let Some(phase) = frame.phase {
            self.rect.rect = record_rect(area, phase);
        } else {
            // The editor column: the scene's own widget, so it sits below the layer's rect in
            // the scene but above the backdrop.
            if ui.input_mut(|input| input.consume_shortcut(&SAVE_SHORTCUT)) && self.dirty() {
                self.start_save();
            }
            let column = Rect::from_min_size(area.min, vec2(area.width() * EDITOR_FRACTION, area.height())).shrink(12.0);
            ui.scope_builder(UiBuilder::new().max_rect(column), |ui| {
                egui::Frame::window(ui.style()).show(ui, |ui| {
                    ui.set_min_size(ui.available_size());
                    self.editor_ui(ui);
                });
            });
            self.rect.interact(ui, "hot-filter");
        }
        if let Some(pass) = &self.pass {
            FilterLayer::new(self.rect.rect, pass.clone()).corner_radius(self.rect.corner_radius).paint(&painter);
        }
        self.paint_overlay(&painter);
        self.rect.paint_outline(&painter);
    }
}

impl HotScene {
    /// The error overlay on the layer rect: the watcher's status (with naga's diagnostic) while
    /// the file is broken or missing, a short note while the first compile is pending.
    fn paint_overlay(&self, painter: &egui::Painter) {
        let text = match &self.hot {
            Ok(hot) if hot.error().is_some() => hot.status_text(),
            Ok(hot) if hot.current().is_none() => "compiling...".to_owned(),
            Ok(_) => return,
            Err(error) => format!("watcher not running: {error}"),
        };
        let rect = self.rect.rect;
        let painter = painter.with_clip_rect(rect);
        let galley = painter.layout(text, FontId::monospace(11.0), Color32::from_rgb(255, 170, 160), rect.width() - 16.0);
        let height = (galley.size().y + 16.0).min(rect.height());
        let panel = Rect::from_min_size(pos2(rect.min.x, rect.max.y - height), vec2(rect.width(), height));
        painter.rect(panel, CornerRadius::ZERO, Color32::from_black_alpha(215), Stroke::new(1.0, Color32::from_rgb(220, 70, 60)), StrokeKind::Inside);
        painter.galley(panel.min + vec2(8.0, 8.0), galley, Color32::WHITE);
        painter.text(rect.left_top() + vec2(8.0, 6.0), Align2::LEFT_TOP, "hot.wgsl", FontId::monospace(11.0), Color32::from_white_alpha(160));
    }
}
