/*
File: examples/demo/main.rs

Purpose:
Interactive demo of `egui_filter_layers`: a set of scenes (see `scenes/`) over a shared busy
backdrop, a side panel with the backend status, a scene selector and the scene's own controls.
This file owns the command line, the stderr logger, eframe start-up on either renderer, and the
app shell that dispatches to the selected scene.

Usage:
    cargo run --example demo --features hot-reload -- [--renderer glow|wgpu] [--msaa N] [--verbose] [--scene NAME]
                                                    [--stack blur,lens:2.5,...] [--rect x,y,w,h] [--screenshot out.ppm]
                                                    [--record DIR [--frames N] [--fps F]]

`--scene` picks the scene (names: `scenes::SCENES` keys; default the first). `--stack` replaces the
initial filter stack of the "stack" scene (names: see `stack::Kind::key`; `name:v1:v2` overrides
the preset's slider values in order); `--rect` places the filter rect (screen points, top-left x/y
and size). `--screenshot` saves the 10th frame as a binary PPM and exits — a scripted check that
the backend produced an image.

`--record DIR` renders a seamless loop of the selected scene for GIFs: an 800x500 window at 1 pixel
per point without the side panel, egui time driven as `step / fps` (not the wall clock), pointer
input dropped. Every animation is periodic with the loop length `N / F` seconds (defaults: 100
frames at 25 fps = 4 s); scenes receive the loop phase in `SceneFrame::phase`. Frames are saved as
`DIR/frame_0000.ppm`, ... (see `record.rs`). Animated effects must pick speeds whose period
divides the loop (e.g. `ripple:6:40:0.5`).

Key structures:
- Options / RendererChoice: the parsed command line.
- DemoApp: the eframe app (status panel, scene selector, lazily built scenes).

Notes:
The side panel shows the renderer and `egui_filter_layers::status()`, so a missing or failed
backend is visible rather than silently drawing nothing.
*/

mod backdrop;
mod params_ui;
mod record;
mod scenes;
mod stack;

use std::path::PathBuf;
use std::process::ExitCode;

use eframe::egui::{self, Rect, pos2, vec2};

use backdrop::Backdrop;
use record::{RECORD_SIZE, RecordOptions, Recorder, ScriptedScreenshot};
use scenes::{SCENES, Scene, SceneFrame, SceneInit};
use stack::Entry;

/// Which renderer the demo runs on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RendererChoice {
    /// `egui_glow`.
    Glow,
    /// `egui-wgpu`.
    Wgpu,
}

/// Parsed command line.
#[derive(Debug)]
struct Options {
    /// Renderer to run.
    renderer: RendererChoice,
    /// MSAA sample count (exercises the glow resolve path).
    msaa: u16,
    /// Log `egui_filter_layers` at debug level.
    verbose: bool,
    /// Index into `SCENES` of the scene shown first.
    scene: usize,
    /// Initial filter stack of the "stack" scene (`None` = its default).
    stack: Option<Vec<Entry>>,
    /// Save a screenshot here and exit.
    screenshot: Option<PathBuf>,
    /// Initial filter rect (`None` = the scene's default placement).
    rect: Option<Rect>,
    /// Seamless-loop recording (`--record`), if requested.
    record: Option<RecordOptions>,
}

/// Parses `std::env::args`; `Err` carries the message to print (empty for `--help`).
fn parse_args() -> Result<Options, String> {
    let mut options = Options { renderer: RendererChoice::Glow, msaa: 0, verbose: false, scene: 0, stack: None, screenshot: None, rect: None, record: None };
    let (mut frames, mut fps) = (100_u32, 25_u32);
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--renderer" => {
                options.renderer = match args.next().as_deref() {
                    Some("glow") => RendererChoice::Glow,
                    Some("wgpu") => RendererChoice::Wgpu,
                    other => return Err(format!("--renderer expects `glow` or `wgpu`, got {other:?}")),
                };
            }
            "--msaa" => {
                let value = args.next().ok_or("--msaa expects a sample count")?;
                options.msaa = value.parse().map_err(|error| format!("--msaa {value:?}: {error}"))?;
            }
            "--verbose" => options.verbose = true,
            "--scene" => {
                let name = args.next().ok_or("--scene expects a scene name")?;
                let known = || SCENES.iter().map(|entry| entry.key).collect::<Vec<_>>().join(",");
                options.scene = scenes::find(&name).ok_or_else(|| format!("unknown scene {name:?}; known: {}", known()))?;
            }
            "--stack" => {
                let list = args.next().ok_or("--stack expects a comma-separated list of filters")?;
                options.stack = Some(stack::parse_stack(&list)?);
            }
            "--rect" => {
                let value = args.next().ok_or("--rect expects x,y,w,h")?;
                let numbers = value.split(',').map(|n| n.trim().parse::<f32>().map_err(|error| format!("--rect {value:?}: {error}"))).collect::<Result<Vec<_>, _>>()?;
                let [x, y, w, h] = numbers[..] else {
                    return Err(format!("--rect expects four numbers, got {value:?}"));
                };
                options.rect = Some(Rect::from_min_size(pos2(x, y), vec2(w, h)));
            }
            "--screenshot" => options.screenshot = Some(PathBuf::from(args.next().ok_or("--screenshot expects a file path")?)),
            "--record" => options.record = Some(RecordOptions { dir: PathBuf::from(args.next().ok_or("--record expects a directory")?), frames: 0, fps: 0 }),
            "--frames" | "--fps" => {
                let value = args.next().ok_or_else(|| format!("{arg} expects a positive integer"))?;
                let number: u32 = value.parse().map_err(|error| format!("{arg} {value:?}: {error}"))?;
                if number == 0 {
                    return Err(format!("{arg} must be positive"));
                }
                if arg == "--frames" { frames = number } else { fps = number }
            }
            "--help" | "-h" => return Err(String::new()),
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    // `--frames` / `--fps` may come before or after `--record`.
    if let Some(record) = &mut options.record {
        record.frames = frames;
        record.fps = fps;
    }
    Ok(options)
}

/// Minimal stderr logger (the demo avoids extra dependencies).
struct StderrLogger {
    /// Level for `egui_filter_layers` records; everything else logs at `Warn` and above.
    crate_level: log::LevelFilter,
}

impl log::Log for StderrLogger {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        if metadata.target().starts_with("egui_filter_layers") || metadata.target().starts_with("demo") { metadata.level() <= self.crate_level } else { metadata.level() <= log::Level::Warn }
    }

    fn log(&self, record: &log::Record<'_>) {
        if self.enabled(record.metadata()) {
            eprintln!("[{} {}] {}", record.level(), record.target(), record.args());
        }
    }

    fn flush(&self) {}
}

fn main() -> ExitCode {
    let options = match parse_args() {
        Ok(options) => options,
        Err(message) => {
            if !message.is_empty() {
                eprintln!("error: {message}");
            }
            let scenes = SCENES.iter().map(|entry| entry.key).collect::<Vec<_>>().join("|");
            eprintln!("usage: demo [--renderer glow|wgpu] [--msaa N] [--verbose] [--scene {scenes}] [--stack blur,lens:2.5,...] [--rect x,y,w,h] [--screenshot out.ppm] [--record DIR [--frames N] [--fps F]]");
            return ExitCode::from(2);
        }
    };
    let crate_level = if options.verbose { log::LevelFilter::Debug } else { log::LevelFilter::Info };
    let logger: &'static StderrLogger = Box::leak(Box::new(StderrLogger { crate_level }));
    if let Err(error) = log::set_logger(logger) {
        eprintln!("warning: could not install the demo logger: {error}");
    }
    log::set_max_level(log::LevelFilter::Debug);

    let native_options = eframe::NativeOptions {
        renderer: match options.renderer {
            RendererChoice::Glow => eframe::Renderer::Glow,
            RendererChoice::Wgpu => eframe::Renderer::Wgpu,
        },
        multisampling: options.msaa,
        viewport: if options.record.is_some() {
            // Recording: a small fixed window (1 pixel per point is forced in `DemoApp::new`).
            egui::ViewportBuilder::default().with_inner_size(RECORD_SIZE).with_resizable(false).with_title("egui_filter_layers demo (recording)")
        } else {
            egui::ViewportBuilder::default().with_inner_size([1280.0, 800.0]).with_title("egui_filter_layers demo")
        },
        ..Default::default()
    };
    let result = eframe::run_native("egui_filter_layers demo", native_options, Box::new(move |cc| Ok(Box::new(DemoApp::new(cc, options)))));
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: the demo window could not be run: {error}");
            ExitCode::FAILURE
        }
    }
}

/// The demo application state.
struct DemoApp {
    /// For `destroy_glow` in `on_exit` (which receives no context).
    ctx: egui::Context,
    /// The renderer eframe runs.
    renderer: RendererChoice,
    /// Why installing the backend failed, if it did.
    install_error: Option<String>,
    /// The shared backdrop.
    backdrop: Backdrop,
    /// Index into `SCENES` of the scene shown.
    current: usize,
    /// Built scenes, parallel to `SCENES` (`None` until first shown).
    scenes: Vec<Option<Box<dyn Scene>>>,
    /// `--stack`, handed to scene constructors.
    stack: Option<Vec<Entry>>,
    /// `--rect`, handed to scene constructors.
    rect: Option<Rect>,
    /// `--screenshot` state, until the capture is saved.
    screenshot: Option<ScriptedScreenshot>,
    /// `--record` state, if recording.
    recorder: Option<Recorder>,
}

impl DemoApp {
    /// Installs the backend matching the renderer (with the MSAA count given to eframe) and
    /// builds the initial state from the command line.
    fn new(cc: &eframe::CreationContext<'_>, options: Options) -> Self {
        let Options { renderer, msaa, scene, stack, screenshot, rect, record, .. } = options;
        if record.is_some() {
            // Frames are exactly the window size in pixels.
            cc.egui_ctx.set_pixels_per_point(1.0);
        }
        let install_error = match (renderer, cc.gl.as_ref()) {
            (RendererChoice::Glow, Some(gl)) => egui_filter_layers::install_glow(&cc.egui_ctx, gl).err().map(|error| error.to_string()),
            (RendererChoice::Glow, None) => Some("eframe provided no glow context".to_owned()),
            (RendererChoice::Wgpu, _) => match cc.wgpu_render_state.as_ref() {
                // Must mirror the NativeOptions given to eframe (see `main`).
                Some(render_state) => egui_filter_layers::install_wgpu(&cc.egui_ctx, render_state, egui_filter_layers::WgpuOptions::default().with_multisampling(msaa)).err().map(|error| error.to_string()),
                None => Some("eframe provided no wgpu render state".to_owned()),
            },
        };
        if let Some(error) = &install_error {
            log::warn!("demo: {error}");
        }
        Self {
            ctx: cc.egui_ctx.clone(),
            renderer,
            install_error,
            backdrop: Backdrop::new(&cc.egui_ctx),
            current: scene,
            scenes: SCENES.iter().map(|_| None).collect(),
            stack,
            rect,
            screenshot: screenshot.map(ScriptedScreenshot::new),
            recorder: record.map(Recorder::new),
        }
    }

    /// The selected scene, built on first use.
    fn scene(&mut self) -> Option<&mut Box<dyn Scene>> {
        let entry = SCENES.get(self.current)?;
        let slot = self.scenes.get_mut(self.current)?;
        if slot.is_none() {
            log::info!("demo: building scene `{}`", entry.key);
            let init = SceneInit { ctx: &self.ctx, rect: self.rect, stack: self.stack.as_deref() };
            *slot = Some((entry.build)(&init));
        }
        slot.as_mut()
    }

    /// The side panel: status, scene selector, then the scene's controls.
    fn controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("egui_filter_layers");
        ui.label(format!("renderer: {:?}", self.renderer));
        let status = egui_filter_layers::status(ui.ctx());
        let color = if status.is_usable() { ui.visuals().text_color() } else { ui.visuals().error_fg_color };
        ui.colored_label(color, format!("backend: {status}"));
        if let Some(error) = &self.install_error {
            ui.colored_label(ui.visuals().error_fg_color, error);
        }
        if self.renderer == RendererChoice::Wgpu {
            ui.colored_label(ui.visuals().warn_fg_color, "wgpu: the content under the filter is rendered twice per frame");
        }
        for failure in egui_filter_layers::failed_effects(ui.ctx()) {
            ui.colored_label(ui.visuals().error_fg_color, format!("effect `{}` failed: {}", failure.label, failure.error));
        }
        ui.separator();

        let title = SCENES.get(self.current).map_or("?", |entry| entry.title);
        egui::ComboBox::from_id_salt("scene-selector").selected_text(title).show_ui(ui, |ui| {
            for (index, entry) in SCENES.iter().enumerate() {
                ui.selectable_value(&mut self.current, index, entry.title);
            }
        });
        ui.separator();
        if let Some(scene) = self.scene() {
            scene.controls(ui);
        }
    }

    /// The central panel: the selected scene.
    fn central(&mut self, ui: &mut egui::Ui) {
        // Scenes animate from `input.time`; without this eframe repaints only on input and the
        // animations would stutter whenever the mouse is idle.
        ui.ctx().request_repaint();
        let time = ui.input(|input| input.time);
        let phase = self.recorder.as_ref().map(|recorder| recorder.phase(time));
        // Split borrows: the frame borrows the backdrop while the scene is borrowed mutably. The
        // scene was built by `controls` (or, while recording, by `ui`) earlier this frame.
        let Some(scene) = self.scenes.get_mut(self.current).and_then(Option::as_mut) else {
            return;
        };
        scene.ui(ui, &SceneFrame { backdrop: &self.backdrop, time, phase });
    }
}

impl eframe::App for DemoApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        if let Some(recorder) = &mut self.recorder {
            recorder.drive(ui.ctx());
            // No side panel while recording: build the scene here.
            self.scene();
            egui::CentralPanel::no_frame().show(ui, |ui| self.central(ui));
            return;
        }
        if self.screenshot.as_mut().is_some_and(|screenshot| screenshot.drive(ui.ctx())) {
            self.screenshot = None;
        }
        egui::Panel::left("controls").resizable(true).default_size(340.0).show(ui, |ui| {
            egui::ScrollArea::vertical().id_salt("controls-scroll").show(ui, |ui| self.controls(ui));
        });
        egui::CentralPanel::no_frame().show(ui, |ui| self.central(ui));
    }

    fn raw_input_hook(&mut self, _ctx: &egui::Context, raw_input: &mut egui::RawInput) {
        if let Some(recorder) = &mut self.recorder {
            recorder.process_input(raw_input);
        }
    }

    fn on_exit(&mut self, gl: Option<&eframe::glow::Context>) {
        if let Some(gl) = gl {
            egui_filter_layers::destroy_glow(&self.ctx, gl);
        }
    }
}
