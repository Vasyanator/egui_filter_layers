/*
File: examples/demo.rs

Purpose:
Interactive demo of `egui_filter_layers`: a busy backdrop (gradient, shapes, text, a generated
image, a scroll area) with a draggable / resizable filter rect, a side panel to stack presets and
edit their parameters, and a live-compiled custom WGSL effect showing the extension API.

Usage:
    cargo run --example demo -- [--renderer glow|wgpu] [--msaa N] [--verbose]
                                [--stack blur,lens:2.5,...] [--rect x,y,w,h] [--screenshot out.ppm]
                                [--record DIR [--frames N] [--fps F]]

`--stack` replaces the initial filter stack (names: see `Kind::key`; `name:v1:v2` overrides the
preset's slider values in order); `--rect` places the filter rect (screen points, top-left x/y and
size). `--screenshot` saves the 10th frame as a binary PPM and exits — a scripted check that the
backend produced an image.

`--record DIR` renders a seamless loop for GIFs: an 800x500 window at 1 pixel per point without
the side panel, egui time driven as `step / fps` (not the wall clock) through
`App::raw_input_hook`, the filter rect moving along a closed pseudo-random path. Every animation
is periodic with the loop length `N / F` seconds (defaults: 100 frames at 25 fps = 4 s), so frame
N would equal frame 0 and is not written. Frames are saved as `DIR/frame_0000.ppm`, ...; the time
only advances when the screenshot of the current step has arrived, so each file is exactly one
step. Animated effects must pick speeds whose period divides the loop (e.g. `ripple:6:40:0.5`).

Notes:
The side panel shows the renderer and `egui_filter_layers::status()`, so a missing or failed
backend is visible rather than silently drawing nothing.
*/

use std::path::PathBuf;
use std::process::ExitCode;

use eframe::egui::{self, Color32, CornerRadius, FontId, Pos2, Rect, Sense, Stroke, StrokeKind, Vec2, pos2, vec2};
use egui_filter_layers::{Effect, EffectError, FilterChain, FilterLayer, Pass, presets};

/// The custom effect shown on start: an animated chromatic wave. Edit it live in the demo.
const CUSTOM_EFFECT: &str = "// params(0).x = amplitude (points), params(0).y = frequency
fn effect(uv: vec2<f32>) -> vec4<f32> {
    let p = params(0u);
    let amp = p.x * fl.pixels_per_point;
    let wave = sin(uv.y * p.y + fl.time * 3.0) * amp;
    let r = src_px(vec2<f32>(wave + 2.0, 0.0)).r;
    let g = src_px(vec2<f32>(wave, 0.0)).g;
    let b = src_px(vec2<f32>(wave - 2.0, 0.0)).b;
    let a = src_px(vec2<f32>(wave, 0.0)).a;
    return vec4<f32>(r, g, b, a);
}
";

/// Window size (points = pixels) in `--record` mode.
const RECORD_SIZE: [f32; 2] = [800.0, 500.0];

/// Frames rendered before the first recorded one, so the window has settled at its final size.
const RECORD_WARMUP_FRAMES: u64 = 10;

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
    /// Initial filter stack (`None` = the default blur + saturation).
    stack: Option<Vec<Entry>>,
    /// Save a screenshot here and exit.
    screenshot: Option<PathBuf>,
    /// Initial filter rect (`None` = the default placement).
    rect: Option<Rect>,
    /// Seamless-loop recording (`--record`), if requested.
    record: Option<RecordOptions>,
}

/// `--record DIR [--frames N] [--fps F]`.
#[derive(Debug)]
struct RecordOptions {
    /// Directory receiving `frame_NNNN.ppm`.
    dir: PathBuf,
    /// Frames in one loop (the loop lasts `frames / fps` seconds).
    frames: u32,
    /// Time steps per second.
    fps: u32,
}

impl RecordOptions {
    /// Loop length in seconds.
    fn period(&self) -> f64 {
        f64::from(self.frames) / f64::from(self.fps)
    }
}

/// Parses one `--stack` item: `name` or `name:v1:v2...` (values override the sliders in order).
fn parse_stack_entry(item: &str) -> Result<Entry, String> {
    let mut parts = item.trim().split(':');
    let name = parts.next().unwrap_or_default();
    let kind = Kind::ALL.into_iter().find(|kind| kind.key() == name).ok_or_else(|| format!("unknown filter {name:?}; known: {}", Kind::ALL.map(Kind::key).join(",")))?;
    let mut entry = Entry::new(kind);
    for (index, value) in parts.enumerate() {
        let slot = entry.values.get_mut(index).ok_or_else(|| format!("{name:?} takes at most {} values", kind.sliders().len()))?;
        *slot = value.parse().map_err(|error| format!("{name:?} value {value:?}: {error}"))?;
    }
    Ok(entry)
}

/// Parses `std::env::args`; `Err` carries the message to print.
fn parse_args() -> Result<Options, String> {
    let mut options = Options { renderer: RendererChoice::Glow, msaa: 0, verbose: false, stack: None, screenshot: None, rect: None, record: None };
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
            "--stack" => {
                let list = args.next().ok_or("--stack expects a comma-separated list of filters")?;
                options.stack = Some(list.split(',').map(parse_stack_entry).collect::<Result<Vec<_>, _>>()?);
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
        if metadata.target().starts_with("egui_filter_layers") { metadata.level() <= self.crate_level } else { metadata.level() <= log::Level::Warn }
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
            eprintln!("usage: demo [--renderer glow|wgpu] [--msaa N] [--verbose] [--stack blur,lens:2.5,...] [--rect x,y,w,h] [--screenshot out.ppm] [--record DIR [--frames N] [--fps F]]");
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

/// A preset the side panel can add to the stack.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Blur,
    BrightnessContrast,
    Saturation,
    Grayscale,
    Invert,
    Tint,
    Sharpen,
    Pixelate,
    Ripple,
    Lens,
    Vignette,
    Custom,
}

impl Kind {
    /// Every kind, in menu order.
    const ALL: [Kind; 12] = [
        Kind::Blur,
        Kind::BrightnessContrast,
        Kind::Saturation,
        Kind::Grayscale,
        Kind::Invert,
        Kind::Tint,
        Kind::Sharpen,
        Kind::Pixelate,
        Kind::Ripple,
        Kind::Lens,
        Kind::Vignette,
        Kind::Custom,
    ];

    /// Command-line name (`--stack`).
    fn key(self) -> &'static str {
        match self {
            Kind::Blur => "blur",
            Kind::BrightnessContrast => "brightness_contrast",
            Kind::Saturation => "saturation",
            Kind::Grayscale => "grayscale",
            Kind::Invert => "invert",
            Kind::Tint => "tint",
            Kind::Sharpen => "sharpen",
            Kind::Pixelate => "pixelate",
            Kind::Ripple => "ripple",
            Kind::Lens => "lens",
            Kind::Vignette => "vignette",
            Kind::Custom => "custom",
        }
    }

    /// Menu / list label.
    fn name(self) -> &'static str {
        match self {
            Kind::Blur => "Gaussian blur",
            Kind::BrightnessContrast => "Brightness / contrast",
            Kind::Saturation => "Saturation",
            Kind::Grayscale => "Grayscale",
            Kind::Invert => "Invert",
            Kind::Tint => "Tint",
            Kind::Sharpen => "Sharpen",
            Kind::Pixelate => "Pixelate",
            Kind::Ripple => "Ripple (animated)",
            Kind::Lens => "Lens",
            Kind::Vignette => "Vignette",
            Kind::Custom => "Custom WGSL (animated)",
        }
    }

    /// `(label, min, max, default)` of each numeric parameter.
    fn sliders(self) -> &'static [(&'static str, f32, f32, f32)] {
        match self {
            Kind::Blur => &[("sigma (pt)", 0.0, 30.0, 6.0)],
            Kind::BrightnessContrast => &[("brightness", -1.0, 1.0, 0.1), ("contrast", 0.0, 3.0, 1.3)],
            Kind::Saturation => &[("factor", 0.0, 3.0, 1.8)],
            Kind::Grayscale | Kind::Invert => &[("amount", 0.0, 1.0, 1.0)],
            Kind::Tint => &[("amount", 0.0, 1.0, 0.3)],
            Kind::Sharpen => &[("amount", 0.0, 3.0, 1.0)],
            Kind::Pixelate => &[("cell (pt)", 1.0, 40.0, 10.0)],
            Kind::Ripple => &[("amplitude (pt)", 0.0, 20.0, 4.0), ("wavelength (pt)", 5.0, 100.0, 30.0), ("speed", 0.0, 3.0, 0.8)],
            Kind::Lens => &[("magnification", 1.0, 4.0, 2.0)],
            Kind::Vignette => &[("strength", 0.0, 1.0, 0.8), ("inner", 0.0, 1.0, 0.3), ("softness", 0.0, 1.0, 0.6)],
            Kind::Custom => &[("amplitude (pt)", 0.0, 20.0, 5.0), ("frequency", 1.0, 80.0, 30.0)],
        }
    }
}

/// One entry of the filter stack.
#[derive(Clone, Debug)]
struct Entry {
    /// Which preset.
    kind: Kind,
    /// Whether it participates in the chain.
    enabled: bool,
    /// Slider values, parallel to `kind.sliders()`.
    values: Vec<f32>,
    /// Colour for `Kind::Tint`.
    color: Color32,
}

impl Entry {
    /// An entry with default values.
    fn new(kind: Kind) -> Self {
        Self { kind, enabled: true, values: kind.sliders().iter().map(|s| s.3).collect(), color: Color32::from_rgb(120, 170, 255) }
    }

    /// Slider value `i` (0 if out of range, which cannot happen for entries built by `new`).
    fn value(&self, i: usize) -> f32 {
        self.values.get(i).copied().unwrap_or(0.0)
    }

    /// The passes of this entry, or `None` when it cannot run (custom effect failed to compile).
    fn chain(&self, custom: Option<&Effect>) -> Option<FilterChain> {
        let v = |i| self.value(i);
        Some(match self.kind {
            Kind::Blur => presets::gaussian_blur(v(0)),
            Kind::BrightnessContrast => presets::brightness_contrast(v(0), v(1)).into(),
            Kind::Saturation => presets::saturation(v(0)).into(),
            Kind::Grayscale => presets::grayscale(v(0)).into(),
            Kind::Invert => presets::invert(v(0)).into(),
            Kind::Tint => presets::tint(self.color, v(0)).into(),
            Kind::Sharpen => presets::sharpen(v(0)).into(),
            Kind::Pixelate => presets::pixelate(v(0)).into(),
            Kind::Ripple => presets::ripple(v(0), v(1), v(2)).into(),
            Kind::Lens => presets::lens(v(0)).into(),
            Kind::Vignette => presets::vignette(v(0), v(1), v(2)).into(),
            Kind::Custom => {
                let effect = custom?;
                let pass = Pass::new(effect).with_param(0, [v(0), v(1), 0.0, 0.0]).ok()?;
                // The wave shifts samples by up to `amplitude` points plus a 2 px channel split.
                pass.with_sample_radius(v(0) + 2.0).into()
            }
        })
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
    /// The filter rect, in points.
    filter_rect: Rect,
    /// Rounded mask radius (points).
    corner_radius: u8,
    /// The preset stack, applied top to bottom.
    stack: Vec<Entry>,
    /// Editable source of the custom effect.
    custom_source: String,
    /// Last compile result of `custom_source`.
    custom_effect: Result<Effect, EffectError>,
    /// The generated image shown in the backdrop.
    texture: egui::TextureHandle,
    /// Where to save the scripted screenshot, if requested.
    screenshot: Option<PathBuf>,
    /// `--record` state, if recording.
    recorder: Option<Recorder>,
}

/// Progress of a `--record` run.
#[derive(Debug)]
struct Recorder {
    /// What to record.
    options: RecordOptions,
    /// The time step currently rendered (`time = step / fps`); advances only when its screenshot arrives.
    step: u32,
    /// Whether the screenshot of `step` has been requested.
    requested: bool,
}

impl Recorder {
    /// Takes the screenshot events out of `raw_input`, saves the one for the current step and
    /// advances; then drops every other input event (no pointer or keyboard influence) and pins
    /// the input time to the current step.
    fn process_input(&mut self, raw_input: &mut egui::RawInput) {
        for event in raw_input.events.drain(..) {
            let egui::Event::Screenshot { user_data, image, .. } = event else {
                continue;
            };
            let tag = user_data.data.as_ref().and_then(|data| data.downcast_ref::<u32>()).copied();
            if tag != Some(self.step) || !self.requested {
                log::warn!("demo: ignoring an unexpected screenshot (tag {tag:?}, current step {})", self.step);
                continue;
            }
            // Synchronous on purpose: a recording tool, and ~1 MB per frame is written in milliseconds.
            let path = self.options.dir.join(format!("frame_{:04}.ppm", self.step));
            if let Err(error) = write_ppm(&path, &image) {
                log::error!("demo: could not save frame {} to {}: {error}", self.step, path.display());
            }
            self.step += 1;
            self.requested = false;
        }
        raw_input.time = Some(f64::from(self.step) / f64::from(self.options.fps));
        raw_input.predicted_dt = 1.0 / f32::from(u16::try_from(self.options.fps).unwrap_or(u16::MAX));
    }

    /// Requests the screenshot of the current step (after the warm-up, once the window is exactly
    /// `RECORD_SIZE` pixels), or closes the window once every step is saved.
    fn drive(&mut self, ctx: &egui::Context) {
        ctx.request_repaint();
        // The builder's inner size is scaled by the display's native factor; with 1 pixel per
        // point forced, re-requesting it in points yields exactly that many pixels.
        let size_ok = ctx.viewport_rect().size() == Vec2::from(RECORD_SIZE) && ctx.pixels_per_point() == 1.0;
        if !size_ok && !self.requested {
            ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(Vec2::from(RECORD_SIZE)));
            return;
        }
        if self.step >= self.options.frames {
            log::info!("demo: recorded {} frames to {}", self.step, self.options.dir.display());
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        } else if !self.requested && ctx.cumulative_frame_nr() >= RECORD_WARMUP_FRAMES {
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::new(self.step)));
            self.requested = true;
        }
    }

    /// Loop phase in `0..1` of `time`.
    fn phase(&self, time: f64) -> f32 {
        #[allow(clippy::cast_possible_truncation)] // A phase in 0..1; f32 precision is plenty.
        let phase = (time / self.options.period()).rem_euclid(1.0) as f32;
        phase
    }
}

/// The filter rect's closed path in `--record` mode at loop phase `phase` (`0..1`): its centre is
/// a sum of sinusoids with integer frequencies (fixed "random" amplitudes and phases), so it
/// returns exactly to its start at phase 1 and stays inside `area`.
fn record_rect(area: Rect, phase: f32) -> Rect {
    let size = vec2(280.0, 190.0);
    // (frequency, x amplitude, x phase, y amplitude, y phase); amplitudes sum to less than the
    // free half-space (260 x 155 points in the 800x500 window).
    let terms = [(1.0, 150.0, 0.3, 85.0, 1.9), (2.0, 60.0, 2.4, 40.0, 0.7), (3.0, 30.0, 4.1, 20.0, 5.2)];
    let turn = std::f32::consts::TAU * phase;
    let offset = terms.iter().fold(Vec2::ZERO, |sum, &(k, ax, px, ay, py)| sum + vec2(ax * (k * turn + px).sin(), ay * (k * turn + py).sin()));
    Rect::from_center_size(area.center() + offset, size)
}

impl DemoApp {
    /// Installs the backend matching the renderer (with the MSAA count given to eframe) and
    /// builds the initial state from the command line.
    fn new(cc: &eframe::CreationContext<'_>, options: Options) -> Self {
        let Options { renderer, msaa, stack, screenshot, rect, record, .. } = options;
        if let Some(record) = &record {
            if let Err(error) = std::fs::create_dir_all(&record.dir) {
                log::error!("demo: could not create {}: {error}", record.dir.display());
            }
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
        let texture = cc.egui_ctx.load_texture("demo-pattern", generate_pattern(256), egui::TextureOptions::LINEAR);
        Self {
            ctx: cc.egui_ctx.clone(),
            renderer,
            install_error,
            filter_rect: rect.unwrap_or_else(|| Rect::from_min_size(pos2(420.0, 140.0), vec2(420.0, 300.0))),
            corner_radius: 18,
            stack: stack.unwrap_or_else(|| vec![Entry::new(Kind::Blur), Entry::new(Kind::Saturation)]),
            custom_source: CUSTOM_EFFECT.to_owned(),
            custom_effect: compile_custom(CUSTOM_EFFECT),
            texture,
            screenshot,
            recorder: record.map(|options| Recorder { options, step: 0, requested: false }),
        }
    }

    /// Drives `--screenshot`: requests one at frame 10, saves it when it arrives, then closes.
    fn scripted_screenshot(&mut self, ctx: &egui::Context) {
        let Some(path) = self.screenshot.clone() else {
            return;
        };
        let image = ctx.input(|input| {
            input.events.iter().find_map(|event| match event {
                egui::Event::Screenshot { image, .. } => Some(image.clone()),
                _ => None,
            })
        });
        if let Some(image) = image {
            match write_ppm(&path, &image) {
                Ok(()) => log::info!("demo: screenshot saved to {}", path.display()),
                Err(error) => log::error!("demo: could not save the screenshot to {}: {error}", path.display()),
            }
            self.screenshot = None;
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        } else if ctx.cumulative_frame_nr() == 10 {
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::default()));
        }
        ctx.request_repaint();
    }

    /// The side panel: status, stack editor, custom effect editor.
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

        ui.horizontal(|ui| {
            ui.label("corner radius");
            ui.add(egui::Slider::new(&mut self.corner_radius, 0..=80));
        });
        ui.menu_button("Add filter", |ui| {
            for kind in Kind::ALL {
                if ui.button(kind.name()).clicked() {
                    self.stack.push(Entry::new(kind));
                    ui.close();
                }
            }
        });
        ui.label("Applied top to bottom. Drag the rect in the scene; resize from its corner.");

        let mut action: Option<(usize, i8)> = None;
        for (index, entry) in self.stack.iter_mut().enumerate() {
            ui.push_id(index, |ui| {
                ui.group(|ui| {
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut entry.enabled, entry.kind.name());
                        if ui.small_button("^").clicked() {
                            action = Some((index, -1));
                        }
                        if ui.small_button("v").clicked() {
                            action = Some((index, 1));
                        }
                        if ui.small_button("x").clicked() {
                            action = Some((index, 0));
                        }
                    });
                    for (value, (label, min, max, _)) in entry.values.iter_mut().zip(entry.kind.sliders()) {
                        ui.add(egui::Slider::new(value, *min..=*max).text(*label));
                    }
                    if entry.kind == Kind::Tint {
                        ui.horizontal(|ui| {
                            ui.label("colour");
                            ui.color_edit_button_srgba(&mut entry.color);
                        });
                    }
                });
            });
        }
        match action {
            Some((index, 0)) => {
                self.stack.remove(index);
            }
            Some((index, -1)) if index > 0 => self.stack.swap(index, index - 1),
            Some((index, 1)) if index + 1 < self.stack.len() => self.stack.swap(index, index + 1),
            _ => {}
        }

        ui.separator();
        ui.label("Custom effect (WGSL). Add \"Custom WGSL\" to the stack to use it.");
        egui::ScrollArea::vertical().id_salt("custom-source").max_height(260.0).show(ui, |ui| {
            ui.add(egui::TextEdit::multiline(&mut self.custom_source).code_editor().desired_width(f32::INFINITY));
        });
        if ui.button("Compile").clicked() {
            self.custom_effect = compile_custom(&self.custom_source);
        }
        match &self.custom_effect {
            Ok(_) => {
                ui.label("compiled OK");
            }
            Err(error) => {
                ui.colored_label(ui.visuals().error_fg_color, error.to_string());
            }
        }
    }

    /// The scene: backdrop, the filter layer, and the rect's handles on top.
    fn scene(&mut self, ui: &mut egui::Ui) {
        // The backdrop animates from `input.time`; without this eframe repaints only on input and
        // the circles would stutter whenever the mouse is idle.
        ui.ctx().request_repaint();
        let area = ui.max_rect();
        let painter = ui.painter_at(area);
        let time = ui.input(|i| i.time);
        // Recording swings the circles back and forth once per loop instead of rotating them.
        let motion = self.recorder.as_ref().map_or(CircleMotion::Rotate(time), |recorder| CircleMotion::Swing(recorder.phase(time)));
        paint_backdrop(&painter, area, &self.texture, motion);
        if let Some(recorder) = &self.recorder {
            self.filter_rect = record_rect(area, recorder.phase(time));
        }

        // Real widgets beneath the filter, to show it filters whatever egui drew earlier.
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

        // Drag / resize handling for the filter rect.
        let handle_size = 16.0;
        let move_response = ui.interact(self.filter_rect, ui.id().with("filter-move"), Sense::drag()).on_hover_cursor(egui::CursorIcon::Grab);
        if move_response.dragged() {
            self.filter_rect = self.filter_rect.translate(move_response.drag_delta());
        }
        let handle = Rect::from_center_size(self.filter_rect.max, Vec2::splat(handle_size));
        let resize_response = ui.interact(handle, ui.id().with("filter-resize"), Sense::drag()).on_hover_cursor(egui::CursorIcon::ResizeNwSe);
        if resize_response.dragged() {
            let max = self.filter_rect.max + resize_response.drag_delta();
            self.filter_rect.max = pos2(max.x.max(self.filter_rect.min.x + 40.0), max.y.max(self.filter_rect.min.y + 40.0));
        }

        let custom = self.custom_effect.as_ref().ok();
        let chain: FilterChain = self.stack.iter().filter(|entry| entry.enabled).filter_map(|entry| entry.chain(custom)).fold(FilterChain::new(), FilterChain::then);
        FilterLayer::new(self.filter_rect, chain).corner_radius(self.corner_radius).paint(&painter);

        // Painted AFTER the filter, so it stays crisp (not filtered).
        let stroke = Stroke::new(1.5, Color32::from_white_alpha(200));
        painter.rect_stroke(self.filter_rect, CornerRadius::same(self.corner_radius), stroke, StrokeKind::Outside);
        painter.circle_filled(self.filter_rect.max, handle_size * 0.4, Color32::WHITE);
    }
}

impl eframe::App for DemoApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        if let Some(recorder) = &mut self.recorder {
            recorder.drive(ui.ctx());
            egui::CentralPanel::no_frame().show(ui, |ui| self.scene(ui));
            return;
        }
        self.scripted_screenshot(&ui.ctx().clone());
        egui::Panel::left("controls").resizable(true).default_size(340.0).show(ui, |ui| {
            egui::ScrollArea::vertical().id_salt("controls-scroll").show(ui, |ui| self.controls(ui));
        });
        egui::CentralPanel::no_frame().show(ui, |ui| self.scene(ui));
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

/// Writes `image` as a binary PPM (P6), dropping alpha.
fn write_ppm(path: &std::path::Path, image: &egui::ColorImage) -> std::io::Result<()> {
    let mut bytes = format!("P6\n{} {}\n255\n", image.width(), image.height()).into_bytes();
    for pixel in &image.pixels {
        bytes.extend_from_slice(&[pixel.r(), pixel.g(), pixel.b()]);
    }
    std::fs::write(path, bytes)
}

/// Compiles the custom effect (animated automatically when it reads `fl.time`).
fn compile_custom(source: &str) -> Result<Effect, EffectError> {
    Effect::new("custom", source)
}

/// How the backdrop circles move.
#[derive(Clone, Copy, Debug)]
enum CircleMotion {
    /// Interactive: rotate steadily with egui time (seconds).
    Rotate(f64),
    /// Recording: swing along the same curve and back once per loop; loop phase `0..1`.
    Swing(f32),
}

/// Paints the busy backdrop: a gradient, moving circles, large text and the generated image.
fn paint_backdrop(painter: &egui::Painter, area: Rect, texture: &egui::TextureHandle, motion: CircleMotion) {
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
    painter.text(area.left_top() + vec2(40.0, 40.0), egui::Align2::LEFT_TOP, "Filter layers", FontId::proportional(64.0), Color32::WHITE);
    painter.text(
        area.left_top() + vec2(44.0, 120.0),
        egui::Align2::LEFT_TOP,
        "Everything painted before the layer is filtered.\nThe outline and handle are painted after it.",
        FontId::proportional(20.0),
        Color32::from_rgb(240, 240, 200),
    );
    let image_rect = Rect::from_min_size(area.left_top() + vec2(460.0, 200.0), vec2(256.0, 256.0));
    painter.image(texture.id(), image_rect, Rect::from_min_max(Pos2::ZERO, pos2(1.0, 1.0)), Color32::WHITE);
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
