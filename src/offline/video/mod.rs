/*
File: offline/video/mod.rs

Purpose:
Video processing on top of the offline core (feature `video`): a frame-stream processor that
keeps the GPU state across frames, the source / sink model and the `process_video` driver.
Adapters live in submodules: ffmpeg / ffprobe command-line programs (`ffmpeg/`, native) and image
sequences (`sequence.rs`, native, with the `image` feature).

Key structures:
- `VideoProcessor` (public): an `OfflineRenderer` + the chain and options + a reused output
  buffer; `process_frame(frame, time)` per frame.
- `VideoProgress` (public): what the progress callback sees.

Key functions:
- `process_video()` (native, blocking) / `process_video_async()`: source -> processor -> sink,
  with progress and cancellation. Both run `drive()`.

Notes:
Every frame goes through `OfflineRenderer::run` (and so through `wgpu_backend/chain.rs`); this
module adds no GPU work. `fl.time` of a frame is the source's time for it (seconds from the first
frame), converted to `f32`.
*/

//! Video: filter frame streams with the same chains as the UI.
//!
//! Four ways to use it, from the lowest level up:
//!
//! 1. **Your own decode loop** — [`VideoProcessor::process_frame`] takes a frame and its time and
//!    returns the filtered frame; the renderer, pipelines, textures and buffers persist across
//!    frames.
//! 2. **Your own source / sink** — implement [`VideoSource`] and [`VideoSink`] (e.g. over
//!    ffmpeg-next or `GStreamer`; the trait docs show how) and call [`process_video`].
#![cfg_attr(
    not(target_arch = "wasm32"),
    doc = "3. **The ffmpeg command-line programs** — [`ffmpeg::FfmpegTranscode`] (or [`ffmpeg::FfmpegSource`] + [`ffmpeg::FfmpegSink`]) decode and encode through `ffmpeg` / `ffprobe` found on `PATH`, with audio passthrough."
)]
#![cfg_attr(target_arch = "wasm32", doc = "3. **The ffmpeg command-line programs** — `ffmpeg::FfmpegTranscode` (native targets only).")]
#![cfg_attr(
    all(feature = "image", not(target_arch = "wasm32")),
    doc = "4. **Image sequences** — [`ImageSequenceSource`] / [`ImageSequenceSink`] read and write numbered frames (`frame_%04d.png`) through the `image` crate."
)]
#![cfg_attr(
    not(all(feature = "image", not(target_arch = "wasm32"))),
    doc = "4. **Image sequences** — `ImageSequenceSource` / `ImageSequenceSink` (native, with the `image` feature too)."
)]
//!
//! ```no_run
//! # #[cfg(not(target_arch = "wasm32"))]
//! # fn main() -> Result<(), egui_filter_layers::offline::video::VideoError> {
//! use egui_filter_layers::{OfflineOptions, OfflineRenderer, presets};
//! use egui_filter_layers::offline::video::{VideoProcessor, ffmpeg::{FfmpegTranscode, VideoCodec}};
//!
//! let renderer = OfflineRenderer::headless()?;
//! let chain = presets::ripple(6.0, 40.0, 0.5).into(); // animated through `fl.time`
//! let mut processor = VideoProcessor::new(renderer, chain, OfflineOptions::default());
//! let frames = FfmpegTranscode::new("in.mp4", "out.mp4")
//!     .with_codec(VideoCodec::H264 { crf: 18 })
//!     .run(&mut processor, |progress| {
//!         eprintln!("frame {} ({:.1} s)", progress.frames, progress.time);
//!         std::ops::ControlFlow::Continue(())
//!     })?;
//! # let _ = frames; Ok(()) }
//! # #[cfg(target_arch = "wasm32")] fn main() {}
//! ```
//!
//! **Time and frame rate.** A frame's `fl.time` is its presentation time minus the first frame's.
//! The ffmpeg adapters produce **constant-frame-rate** output at the input's average frame rate
//! (`avg_frame_rate`): a variable-frame-rate input is resampled to it (frames duplicated or
//! dropped by ffmpeg, as `-fps_mode cfr` does) and a video stream that starts after the file's
//! first audio is padded with its first frame, which keeps the passed-through audio in sync.
//! Frame `i` then has `fl.time = i / frame_rate`.

mod error;
#[cfg(not(target_arch = "wasm32"))]
pub mod ffmpeg;
#[cfg(all(feature = "image", not(target_arch = "wasm32")))]
mod sequence;
mod stream;

use std::ops::ControlFlow;

pub use error::VideoError;
#[cfg(all(feature = "image", not(target_arch = "wasm32")))]
pub use sequence::{ImageSequenceSink, ImageSequenceSource};
pub use stream::{FrameRate, VideoInfo, VideoSink, VideoSource};

use super::{AlphaMode, FrameRef, OfflineError, OfflineOptions, OfflineRenderer};
use crate::pass::FilterChain;

/// Filters a stream of frames with one chain and one set of options, keeping everything that
/// can be reused between frames: the renderer (device, pipelines, textures, readback buffer, the
/// prepared-input buffer) and the output buffer.
///
/// The frame's time replaces [`OfflineOptions::time`] for every frame. Like the renderer, it
/// blocks or awaits the GPU: never use it on a GUI thread.
#[derive(Debug)]
pub struct VideoProcessor {
    /// The renderer, reused for every frame.
    renderer: OfflineRenderer,
    /// The chain applied to every frame.
    chain: FilterChain,
    /// Options of every frame (its time is replaced per frame).
    options: OfflineOptions,
    /// The last frame's result.
    output: Vec<u8>,
}

impl VideoProcessor {
    /// A processor applying `chain` with `options` on `renderer`.
    #[must_use]
    pub fn new(renderer: OfflineRenderer, chain: FilterChain, options: OfflineOptions) -> Self {
        Self { renderer, chain, options, output: Vec::new() }
    }

    /// The renderer.
    #[must_use]
    pub fn renderer(&self) -> &OfflineRenderer {
        &self.renderer
    }

    /// Takes the renderer back (e.g. to build a processor with another chain).
    #[must_use]
    pub fn into_renderer(self) -> OfflineRenderer {
        self.renderer
    }

    /// The chain.
    #[must_use]
    pub fn chain(&self) -> &FilterChain {
        &self.chain
    }

    /// Replaces the chain from the next frame on (keyframed parameters: set a new chain per frame;
    /// compiled pipelines of effects still in use are kept).
    pub fn set_chain(&mut self, chain: FilterChain) {
        self.chain = chain;
    }

    /// The options.
    #[must_use]
    pub fn options(&self) -> &OfflineOptions {
        &self.options
    }

    /// Replaces the options from the next frame on.
    pub fn set_options(&mut self, options: OfflineOptions) {
        self.options = options;
    }

    /// Filters `frame`, shown at `time` seconds (`fl.time`), and returns a view of the result,
    /// valid until the next call. Blocks until the GPU is done (native only).
    ///
    /// # Errors
    /// As [`OfflineRenderer::process`]; a non-finite `time` is [`OfflineError::InvalidOption`].
    #[cfg(not(target_arch = "wasm32"))]
    pub fn process_frame(&mut self, frame: &FrameRef<'_>, time: f64) -> Result<FrameRef<'_>, OfflineError> {
        super::readback::block_on(self.run_frame(frame, time, None, true))
    }

    /// [`VideoProcessor::process_frame`] into the caller's buffer (resized to `width * height * 4`).
    ///
    /// # Errors
    /// As [`VideoProcessor::process_frame`].
    #[cfg(not(target_arch = "wasm32"))]
    pub fn process_frame_into(&mut self, frame: &FrameRef<'_>, time: f64, output: &mut Vec<u8>) -> Result<(), OfflineError> {
        let options = frame_options(&self.options, time, None)?;
        super::readback::block_on(self.renderer.run(frame, &self.chain, &options, output, true))
    }

    /// Async [`VideoProcessor::process_frame`] (on the web, or where the device is polled by
    /// someone else; it does not poll the device, see [`OfflineRenderer::process_async`]).
    ///
    /// # Errors
    /// As [`VideoProcessor::process_frame`].
    pub async fn process_frame_async(&mut self, frame: &FrameRef<'_>, time: f64) -> Result<FrameRef<'_>, OfflineError> {
        self.run_frame(frame, time, None, false).await
    }

    /// The shared per-frame implementation: `alpha` overrides the options' output alpha.
    async fn run_frame(&mut self, frame: &FrameRef<'_>, time: f64, alpha: Option<AlphaMode>, block: bool) -> Result<FrameRef<'_>, OfflineError> {
        let options = frame_options(&self.options, time, alpha)?;
        self.renderer.run(frame, &self.chain, &options, &mut self.output, block).await?;
        Ok(FrameRef::new(&self.output, frame.width(), frame.height(), options.output_alpha))
    }
}

/// `options` with `fl.time = time` (and the output alpha overridden when given).
///
/// # Errors
/// [`OfflineError::InvalidOption`] for a non-finite time or one beyond `f32`.
fn frame_options(options: &OfflineOptions, time: f64, alpha: Option<AlphaMode>) -> Result<OfflineOptions, OfflineError> {
    if !time.is_finite() || time.abs() > f64::from(f32::MAX) {
        return Err(OfflineError::InvalidOption { option: "time", reason: format!("a frame time must be finite, got {time}") });
    }
    // `fl.time` is an `f32` uniform, like egui's own time in the UI; in range (checked above), so
    // the conversion only rounds.
    #[allow(clippy::cast_possible_truncation)]
    let seconds = time as f32;
    let mut options = options.clone().with_time(seconds);
    if let Some(alpha) = alpha {
        options = options.with_output_alpha(alpha);
    }
    Ok(options)
}

/// Where [`process_video`] is, reported after every frame.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct VideoProgress {
    /// Frames written so far.
    pub frames: u64,
    /// Time of the last frame written, seconds.
    pub time: f64,
    /// Expected number of frames (the source's, possibly estimated), when known.
    pub frame_count: Option<u64>,
    /// The source's length in seconds, when known.
    pub duration: Option<f64>,
}

impl VideoProgress {
    /// Completed fraction `0..=1`, when the length is known (by frame count, else by time).
    #[must_use]
    pub fn fraction(&self) -> Option<f64> {
        // Frame counts far below 2^53 convert exactly.
        #[allow(clippy::cast_precision_loss)]
        let by_frames = self.frame_count.filter(|&count| count > 0).map(|count| self.frames as f64 / count as f64);
        by_frames.or_else(|| self.duration.filter(|&duration| duration > 0.0).map(|duration| self.time / duration)).map(|fraction| fraction.clamp(0.0, 1.0))
    }
}

/// Reads every frame of `source`, filters it with `processor` and writes it to `sink`, blocking
/// until done (native only; run it on a worker thread). Returns the number of frames written.
///
/// Calls `sink.begin` with the source's info, then per frame `write_frame` (straight alpha,
/// whatever the processor's output alpha option says) and `progress`; returning
/// `ControlFlow::Break` from `progress` cancels. On success `sink.finish` completes the output.
/// When reading, filtering or writing a frame fails, or on cancellation, `sink.abort` is called
/// instead; an error of `sink.begin` or `sink.finish` itself is returned without `abort`.
///
/// # Errors
/// The source's, processor's or sink's errors; [`VideoError::FrameBytesMismatch`] when the
/// source delivers a frame of the wrong length; [`VideoError::Cancelled`] when `progress` stops
/// it.
#[cfg(not(target_arch = "wasm32"))]
pub fn process_video<S, K>(processor: &mut VideoProcessor, source: &mut S, sink: &mut K, progress: impl FnMut(&VideoProgress) -> ControlFlow<()>) -> Result<u64, VideoError>
where
    S: VideoSource + ?Sized,
    K: VideoSink + ?Sized,
{
    super::readback::block_on(drive(processor, source, sink, progress, true))
}

/// Async [`process_video`]. It does not poll the GPU device (see
/// [`OfflineRenderer::process_async`]), and the source and sink are called synchronously.
///
/// # Errors
/// As [`process_video`].
pub async fn process_video_async<S, K>(processor: &mut VideoProcessor, source: &mut S, sink: &mut K, progress: impl FnMut(&VideoProgress) -> ControlFlow<()>) -> Result<u64, VideoError>
where
    S: VideoSource + ?Sized,
    K: VideoSink + ?Sized,
{
    drive(processor, source, sink, progress, false).await
}

/// The shared driver: `begin`, the frame loop, then `finish` or `abort`.
async fn drive<S, K>(processor: &mut VideoProcessor, source: &mut S, sink: &mut K, progress: impl FnMut(&VideoProgress) -> ControlFlow<()>, block: bool) -> Result<u64, VideoError>
where
    S: VideoSource + ?Sized,
    K: VideoSink + ?Sized,
{
    let info = source.info().clone();
    log::info!("egui_filter_layers (video): processing {}x{} at {} fps ({} frames expected)", info.width(), info.height(), info.frame_rate(), info.frame_count().map_or_else(|| "unknown".to_owned(), |count| count.to_string()));
    sink.begin(&info)?;
    match frame_loop(processor, source, sink, &info, progress, block).await {
        Ok(frames) => {
            sink.finish()?;
            log::info!("egui_filter_layers (video): wrote {frames} frames");
            Ok(frames)
        }
        Err(error) => {
            log::warn!("egui_filter_layers (video): stopped: {error}");
            sink.abort();
            Err(error)
        }
    }
}

/// Reads, filters and writes frames until the source ends; returns the number written.
async fn frame_loop<S, K>(processor: &mut VideoProcessor, source: &mut S, sink: &mut K, info: &VideoInfo, mut progress: impl FnMut(&VideoProgress) -> ControlFlow<()>, block: bool) -> Result<u64, VideoError>
where
    S: VideoSource + ?Sized,
    K: VideoSink + ?Sized,
{
    let expected_bytes = info.frame_bytes()?;
    let mut input = Vec::new();
    let mut frames = 0_u64;
    while let Some(time) = source.read_frame(&mut input)? {
        if input.len() != expected_bytes {
            return Err(VideoError::FrameBytesMismatch { frame: frames, expected_bytes, actual_bytes: input.len(), stderr_tail: String::new() });
        }
        let frame = FrameRef::new(&input, info.width(), info.height(), AlphaMode::Straight);
        let filtered = processor.run_frame(&frame, time, Some(AlphaMode::Straight), block).await?;
        sink.write_frame(&filtered, time)?;
        frames += 1;
        let report = VideoProgress { frames, time, frame_count: info.frame_count(), duration: info.duration() };
        if progress(&report).is_break() {
            return Err(VideoError::Cancelled { frames });
        }
    }
    Ok(frames)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_times_become_fl_time() {
        let base = OfflineOptions::default().with_time(9.0).with_output_alpha(AlphaMode::Premultiplied);
        let Ok(options) = frame_options(&base, 1.25, None) else { panic!("finite time") };
        assert!((options.time() - 1.25).abs() < f32::EPSILON);
        assert_eq!(options.output_alpha, AlphaMode::Premultiplied);
        let Ok(options) = frame_options(&base, 0.5, Some(AlphaMode::Straight)) else { panic!("finite time") };
        assert_eq!(options.output_alpha, AlphaMode::Straight);
        for bad in [f64::NAN, f64::INFINITY, 1e300] {
            assert!(matches!(frame_options(&base, bad, None), Err(OfflineError::InvalidOption { option: "time", .. })), "{bad}");
        }
    }

    #[test]
    fn progress_fraction_prefers_frames_then_time() {
        let progress = VideoProgress { frames: 5, time: 1.0, frame_count: Some(20), duration: Some(2.0) };
        assert_eq!(progress.fraction(), Some(0.25));
        assert_eq!(VideoProgress { frame_count: None, ..progress }.fraction(), Some(0.5));
        assert_eq!(VideoProgress { frame_count: None, duration: None, ..progress }.fraction(), None);
        assert_eq!(VideoProgress { frames: 30, ..progress }.fraction(), Some(1.0), "estimates are clamped");
    }
}
