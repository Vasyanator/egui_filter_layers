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

- FrameWriter: the file-writing worker thread both modes hand their captures to.

Key functions:
- write_ppm(): binary PPM (P6) writer, run on the writer thread.

Notes:
Recording pins `RawInput::time` to `step / fps` and advances a step only when that step's tagged
screenshot has arrived, so each `frame_NNNN.ppm` is exactly one step. The GUI thread never
touches the file system: captures go to a `FrameWriter` thread over a bounded queue (which also
creates the output directory), and the window closes only once the writer has drained and exited
(polled, never joined blocking). When the disk falls `WRITE_QUEUE` frames behind, handing over a
frame waits for the writer: deliberate backpressure in these scripted capture modes, because
rendering on while a frame waits would paint the pinned step extra times and make recordings of
stateful scenes depend on disk speed.
Both modes need eframe to keep redrawing. On native Wayland a script-launched window stops after
two frames (winit waits for a frame callback the compositor withholds; a bare eframe app behaves
the same), so run scripted captures under XWayland (`WAYLAND_DISPLAY=`).
*/

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{SendError, SyncSender, sync_channel};
use std::thread::JoinHandle;

use eframe::egui::{self, Vec2};

/// Window size (points = pixels) in `--record` mode.
pub const RECORD_SIZE: [f32; 2] = [800.0, 500.0];

/// Frames rendered before the first recorded one, so the window has settled at its final size.
const RECORD_WARMUP_FRAMES: u64 = 10;

/// Frame number at which `--screenshot` requests its capture.
const SCREENSHOT_FRAME: u64 = 10;

/// Captures that may wait for the writer thread before handing one over blocks (~1 MB each).
const WRITE_QUEUE: usize = 8;

/// One capture to save: the target path and the image.
type WriteJob = (PathBuf, Arc<egui::ColorImage>);

/// Saves captures as PPM files on a worker thread, so the GUI thread does no file I/O. The queue
/// is bounded ([`WRITE_QUEUE`]): `write` blocks only while it is full (see the file notes).
#[derive(Debug)]
struct FrameWriter {
    /// The queue into the worker; `None` once finished (or when the thread could not start).
    sender: Option<SyncSender<WriteJob>>,
    /// The worker; it returns how many writes failed (each is logged).
    worker: Option<JoinHandle<usize>>,
}

impl FrameWriter {
    /// Starts the worker; it first creates `dir` when given (a failure is logged, and every write
    /// into it then fails and is logged too). A thread that cannot start is logged, and every
    /// `write` then drops its capture with an error.
    fn spawn(dir: Option<PathBuf>) -> Self {
        let (sender, receiver) = sync_channel::<WriteJob>(WRITE_QUEUE);
        let worker = std::thread::Builder::new().name("demo-frame-writer".to_owned()).spawn(move || {
            if let Some(dir) = dir
                && let Err(error) = std::fs::create_dir_all(&dir)
            {
                log::error!("demo: could not create {}: {error}", dir.display());
            }
            let mut failed = 0_usize;
            // Ends when the sender is dropped (`finish`) and the queue is drained.
            for (path, image) in receiver {
                if let Err(error) = write_ppm(&path, &image) {
                    log::error!("demo: could not save {}: {error}", path.display());
                    failed += 1;
                }
            }
            failed
        });
        match worker {
            Ok(worker) => Self { sender: Some(sender), worker: Some(worker) },
            Err(error) => {
                log::error!("demo: could not start the frame writer thread: {error}; captures will not be saved");
                Self { sender: None, worker: None }
            }
        }
    }

    /// Queues `job`, waiting only while [`WRITE_QUEUE`] frames are already queued. A capture
    /// that cannot be queued (no worker, or it exited) is dropped with an error.
    fn write(&self, job: WriteJob) {
        let Some(sender) = &self.sender else {
            log::error!("demo: {} not saved: the frame writer is not running", job.0.display());
            return;
        };
        if let Err(SendError((path, _))) = sender.send(job) {
            log::error!("demo: {} not saved: the frame writer thread has exited", path.display());
        }
    }

    /// Closes the queue and, without blocking, reports whether the worker is done: `Some(failed
    /// writes)` once it has drained and exited, `None` while it is still writing (call again).
    fn finish(&mut self) -> Option<usize> {
        self.sender = None;
        let Some(worker) = self.worker.take_if(|worker| worker.is_finished()) else {
            return if self.worker.is_some() { None } else { Some(0) };
        };
        // `is_finished` was true: this join returns at once.
        Some(worker.join().unwrap_or_else(|_| {
            log::error!("demo: the frame writer thread panicked; some captures may be missing");
            1
        }))
    }
}

impl Drop for FrameWriter {
    /// Lets the worker drain what is queued when the app exits early (window closed mid-recording):
    /// a blocking join, but only at shutdown, so no queued capture is lost.
    fn drop(&mut self) {
        self.sender = None;
        if let Some(worker) = self.worker.take()
            && worker.join().is_err()
        {
            log::error!("demo: the frame writer thread panicked; some captures may be missing");
        }
    }
}

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
    /// Saves the frames off the GUI thread (and creates the output directory).
    writer: FrameWriter,
}

impl Recorder {
    /// A recorder at step 0. Its writer thread creates the output directory (a failure is logged;
    /// every frame save then fails and is logged too).
    pub fn new(options: RecordOptions) -> Self {
        let writer = FrameWriter::spawn(Some(options.dir.clone()));
        Self { options, step: 0, requested: false, writer }
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
            self.writer.write((path, image));
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
            // Close only once every frame is on disk (the writer drained and exited).
            if let Some(failed) = self.writer.finish() {
                log::info!("demo: recorded {} frames to {} ({failed} failed to save)", self.step, self.options.dir.display());
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
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

/// `--screenshot out.ppm`: requests one screenshot at frame 10, saves it (on the writer thread),
/// then closes the window.
#[derive(Debug)]
pub struct ScriptedScreenshot {
    /// Where to save the capture.
    path: PathBuf,
    /// Saves the capture off the GUI thread.
    writer: FrameWriter,
    /// Whether the capture arrived (and was handed to the writer).
    captured: bool,
}

impl ScriptedScreenshot {
    /// A pending screenshot to `path`.
    pub fn new(path: PathBuf) -> Self {
        Self { path, writer: FrameWriter::spawn(None), captured: false }
    }

    /// Call once per frame. Returns `true` once the capture has been written (or its failure
    /// logged) and the window close was requested; the caller then drops `self`.
    pub fn drive(&mut self, ctx: &egui::Context) -> bool {
        ctx.request_repaint();
        if !self.captured {
            let image = ctx.input(|input| {
                input.events.iter().find_map(|event| match event {
                    egui::Event::Screenshot { image, .. } => Some(Arc::clone(image)),
                    _ => None,
                })
            });
            if let Some(image) = image {
                self.captured = true;
                self.writer.write((self.path.clone(), image));
            } else if ctx.cumulative_frame_nr() == SCREENSHOT_FRAME {
                ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::default()));
            }
            if !self.captured {
                return false;
            }
        }
        match self.writer.finish() {
            Some(failed) => {
                if failed == 0 {
                    log::info!("demo: screenshot saved to {}", self.path.display());
                }
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                true
            }
            None => false,
        }
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
