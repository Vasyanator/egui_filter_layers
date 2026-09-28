/*
File: examples/demo/record.rs

Purpose:
The demo's scripted capture machinery: `--record` (a seamless loop of frames at deterministic
time) and `--screenshot` (one frame, then exit). Scene-independent: a scene only sees the loop
phase the recorder derives from the pinned time.

Key structures:
- RecordOptions: `--record DIR [--frames N] [--fps F]`.
- Recorder: drives one recording (input hook + per-frame screenshot requests).
- ScriptedScreenshot: drives `--screenshot`.

Key functions:
- write_ppm(): binary PPM (P6) writer shared by both modes.

Notes:
Recording pins `RawInput::time` to `step / fps` and advances a step only when that step's tagged
screenshot has arrived, so each `frame_NNNN.ppm` is exactly one step. Frames are written
synchronously on the GUI thread on purpose: a recording tool, ~1 MB per frame, milliseconds.
*/

use std::path::{Path, PathBuf};

use eframe::egui::{self, Vec2};

/// Window size (points = pixels) in `--record` mode.
pub const RECORD_SIZE: [f32; 2] = [800.0, 500.0];

/// Frames rendered before the first recorded one, so the window has settled at its final size.
const RECORD_WARMUP_FRAMES: u64 = 10;

/// Frame number at which `--screenshot` requests its capture.
const SCREENSHOT_FRAME: u64 = 10;

/// `--record DIR [--frames N] [--fps F]`.
#[derive(Debug)]
pub struct RecordOptions {
    /// Directory receiving `frame_NNNN.ppm`.
    pub dir: PathBuf,
    /// Frames in one loop (the loop lasts `frames / fps` seconds); positive.
    pub frames: u32,
    /// Time steps per second; positive.
    pub fps: u32,
}

impl RecordOptions {
    /// Loop length in seconds.
    fn period(&self) -> f64 {
        f64::from(self.frames) / f64::from(self.fps)
    }
}

/// Progress of a `--record` run.
#[derive(Debug)]
pub struct Recorder {
    /// What to record.
    options: RecordOptions,
    /// The time step currently rendered (`time = step / fps`); advances only when its screenshot arrives.
    step: u32,
    /// Whether the screenshot of `step` has been requested.
    requested: bool,
}

impl Recorder {
    /// A recorder at step 0. Creates the output directory (a failure is logged; every frame
    /// save then fails and is logged too).
    pub fn new(options: RecordOptions) -> Self {
        if let Err(error) = std::fs::create_dir_all(&options.dir) {
            log::error!("demo: could not create {}: {error}", options.dir.display());
        }
        Self { options, step: 0, requested: false }
    }

    /// Takes the screenshot events out of `raw_input`, saves the one for the current step and
    /// advances; then drops every other input event (no pointer or keyboard influence) and pins
    /// the input time to the current step.
    pub fn process_input(&mut self, raw_input: &mut egui::RawInput) {
        for event in raw_input.events.drain(..) {
            let egui::Event::Screenshot { user_data, image, .. } = event else {
                continue;
            };
            let tag = user_data.data.as_ref().and_then(|data| data.downcast_ref::<u32>()).copied();
            if tag != Some(self.step) || !self.requested {
                log::warn!("demo: ignoring an unexpected screenshot (tag {tag:?}, current step {})", self.step);
                continue;
            }
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
    pub fn drive(&mut self, ctx: &egui::Context) {
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

    /// Loop phase in `0..1` of `time` (seconds): every recorded animation must be periodic in it.
    pub fn phase(&self, time: f64) -> f32 {
        #[allow(clippy::cast_possible_truncation)] // A phase in 0..1; f32 precision is plenty.
        let phase = (time / self.options.period()).rem_euclid(1.0) as f32;
        phase
    }
}

/// `--screenshot out.ppm`: requests one screenshot at frame 10, saves it, then closes the window.
#[derive(Debug)]
pub struct ScriptedScreenshot {
    /// Where to save the capture.
    path: PathBuf,
}

impl ScriptedScreenshot {
    /// A pending screenshot to `path`.
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }

    /// Call once per frame. Returns `true` once the capture has been handled (saved or failed,
    /// both logged) and the window close was requested; the caller then drops `self`.
    pub fn drive(&self, ctx: &egui::Context) -> bool {
        let image = ctx.input(|input| {
            input.events.iter().find_map(|event| match event {
                egui::Event::Screenshot { image, .. } => Some(image.clone()),
                _ => None,
            })
        });
        ctx.request_repaint();
        if let Some(image) = image {
            match write_ppm(&self.path, &image) {
                Ok(()) => log::info!("demo: screenshot saved to {}", self.path.display()),
                Err(error) => log::error!("demo: could not save the screenshot to {}: {error}", self.path.display()),
            }
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            return true;
        }
        if ctx.cumulative_frame_nr() == SCREENSHOT_FRAME {
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::default()));
        }
        false
    }
}

/// Writes `image` as a binary PPM (P6), dropping alpha.
///
/// # Errors
/// The file system error of writing `path`.
pub fn write_ppm(path: &Path, image: &egui::ColorImage) -> std::io::Result<()> {
    let mut bytes = format!("P6\n{} {}\n255\n", image.width(), image.height()).into_bytes();
    for pixel in &image.pixels {
        bytes.extend_from_slice(&[pixel.r(), pixel.g(), pixel.b()]);
    }
    std::fs::write(path, bytes)
}
