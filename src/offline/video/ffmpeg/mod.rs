/*
File: offline/video/ffmpeg/mod.rs

Purpose:
Adapters to the `ffmpeg` and `ffprobe` command-line programs (feature `video`, native only):
program locations (`FfmpegTools`), a source and a sink over pipes, codec presets, and the
one-call `FfmpegTranscode`.

Key structures:
- `FfmpegTools`: where `ffmpeg` / `ffprobe` are (default: looked up on `PATH`).
- `FfmpegTranscode`: input file -> `VideoProcessor` -> output file, audio passed through.
- Re-exports: `FfmpegSource`, `FfmpegSink`, `VideoCodec`, `AudioMode`.

Notes:
No Rust dependency: the programs run through `std::process`. They must be installed separately
(ffmpeg 5.1 or newer, checked before use). Which codecs exist, and under which licence, depends on
that build.
*/

//! Video through the `ffmpeg` / `ffprobe` programs (native only).
//!
//! The programs are **not** bundled: install [FFmpeg](https://ffmpeg.org/download.html) 5.1 or
//! newer and put it on `PATH`, or point [`FfmpegTools`] at it. They run as child processes;
//! frames travel as raw RGBA over pipes. The codecs available — and the licence terms that apply
//! to what you encode (e.g. `libx264` is GPL) — are those of the ffmpeg build the user installs;
//! this crate links none of it.
//!
//! Output is at a **constant frame rate** (the input's average); see the
//! [`video`](super) module docs for how variable-rate input and `fl.time` map.
//!
//! **Colour.** Decoding converts to sRGB with the input's own YCbCr matrix and range tags
//! (untagged: BT.709 for HD, BT.601 for SD, as players assume); HDR and wide-gamut inputs are
//! rejected ([`VideoError::UnsupportedColor`]). Encoding converts explicitly with the BT.709
//! matrix to limited range and tags the stream BT.709, so players show the colours the filters
//! produced (see [`VideoCodec`]).
//!
//! **Output files** are written to a temporary file beside the output and renamed over it only
//! when encoding succeeds; the output may never be the input ([`VideoError::OutputIsInput`]).

mod args;
mod color;
mod paths;
mod probe;
mod process;
mod sink;
mod source;
mod version;

use std::ops::ControlFlow;
use std::path::{Path, PathBuf};

pub use args::{AudioMode, VideoCodec};
pub use sink::FfmpegSink;
pub use source::FfmpegSource;

use super::{FrameRate, VideoError, VideoProcessor, VideoProgress, process_video};

/// Where the `ffmpeg` and `ffprobe` programs are.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FfmpegTools {
    /// The `ffmpeg` program.
    ffmpeg: PathBuf,
    /// The `ffprobe` program.
    ffprobe: PathBuf,
}

impl Default for FfmpegTools {
    /// `ffmpeg` and `ffprobe`, looked up on `PATH`.
    fn default() -> Self {
        Self { ffmpeg: PathBuf::from("ffmpeg"), ffprobe: PathBuf::from("ffprobe") }
    }
}

impl FfmpegTools {
    /// Explicit program paths.
    #[must_use]
    pub fn new(ffmpeg: impl Into<PathBuf>, ffprobe: impl Into<PathBuf>) -> Self {
        Self { ffmpeg: ffmpeg.into(), ffprobe: ffprobe.into() }
    }

    /// Both programs in `dir` (`ffmpeg.exe` / `ffprobe.exe` on Windows).
    #[must_use]
    pub fn in_dir(dir: impl AsRef<Path>) -> Self {
        let dir = dir.as_ref();
        let program = |name: &str| dir.join(format!("{name}{}", std::env::consts::EXE_SUFFIX));
        Self { ffmpeg: program("ffmpeg"), ffprobe: program("ffprobe") }
    }

    /// The `ffmpeg` program.
    #[must_use]
    pub fn ffmpeg(&self) -> &Path {
        &self.ffmpeg
    }

    /// The `ffprobe` program.
    #[must_use]
    pub fn ffprobe(&self) -> &Path {
        &self.ffprobe
    }
}

/// Filters a video file into another in one call: [`FfmpegSource`] -> [`process_video`] ->
/// [`FfmpegSink`], with the input's audio handled by [`AudioMode`] (copied by default).
///
/// ```no_run
/// # fn run(processor: &mut egui_filter_layers::offline::video::VideoProcessor) -> Result<(), egui_filter_layers::offline::video::VideoError> {
/// use egui_filter_layers::offline::video::ffmpeg::{AudioMode, FfmpegTranscode, VideoCodec};
/// FfmpegTranscode::new("clip.mov", "clip.webm")
///     .with_codec(VideoCodec::Vp9 { crf: 31 })
///     .with_audio(AudioMode::Encode(vec!["-c:a".into(), "libopus".into()]))
///     .run(processor, |_| std::ops::ControlFlow::Continue(()))?;
/// # Ok(()) }
/// ```
#[derive(Clone, Debug)]
pub struct FfmpegTranscode {
    /// The file read.
    input: PathBuf,
    /// The file written (replaced on success).
    output: PathBuf,
    /// Output codec.
    codec: VideoCodec,
    /// What happens to the input's audio.
    audio: AudioMode,
    /// Program locations.
    tools: FfmpegTools,
    /// Frame rate override.
    frame_rate: Option<FrameRate>,
}

impl FfmpegTranscode {
    /// Reads `input`, writes `output` with the default codec (H.264, crf 18) and copied audio.
    #[must_use]
    pub fn new(input: impl Into<PathBuf>, output: impl Into<PathBuf>) -> Self {
        Self { input: input.into(), output: output.into(), codec: VideoCodec::default(), audio: AudioMode::default(), tools: FfmpegTools::default(), frame_rate: None }
    }

    /// Sets the output codec.
    #[must_use]
    pub fn with_codec(mut self, codec: VideoCodec) -> Self {
        self.codec = codec;
        self
    }

    /// Sets what happens to the input's audio.
    #[must_use]
    pub fn with_audio(mut self, audio: AudioMode) -> Self {
        self.audio = audio;
        self
    }

    /// Uses these program locations.
    #[must_use]
    pub fn with_tools(mut self, tools: FfmpegTools) -> Self {
        self.tools = tools;
        self
    }

    /// Decodes and encodes at `frame_rate` instead of the input's average rate.
    #[must_use]
    pub fn with_frame_rate(mut self, frame_rate: Option<FrameRate>) -> Self {
        self.frame_rate = frame_rate;
        self
    }

    /// Runs it (blocking; use a worker thread). Returns the number of frames written.
    ///
    /// # Errors
    /// [`VideoError::OutputIsInput`] when the output is the input file (also through links);
    /// as [`FfmpegSource::open_with`], [`process_video`] and [`FfmpegSink`] otherwise.
    pub fn run(&self, processor: &mut VideoProcessor, progress: impl FnMut(&VideoProgress) -> ControlFlow<()>) -> Result<u64, VideoError> {
        if paths::same_file(&self.input, &self.output) {
            return Err(VideoError::OutputIsInput { path: self.output.clone() });
        }
        let mut source = FfmpegSource::open_with(&self.input, &self.tools, self.frame_rate)?;
        let mut sink = FfmpegSink::new(&self.output, self.codec.clone()).with_tools(self.tools.clone());
        if source.has_audio() {
            sink = sink.with_audio_from(&self.input, self.audio.clone());
        }
        process_video(processor, &mut source, &mut sink, progress)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tools_default_to_path_lookup_or_a_directory() {
        let tools = FfmpegTools::default();
        assert_eq!((tools.ffmpeg(), tools.ffprobe()), (Path::new("ffmpeg"), Path::new("ffprobe")));
        let tools = FfmpegTools::in_dir("/opt/ffmpeg/bin");
        assert_eq!(tools.ffmpeg(), Path::new(&format!("/opt/ffmpeg/bin/ffmpeg{}", std::env::consts::EXE_SUFFIX)));
    }
}
