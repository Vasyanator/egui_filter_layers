/*
File: offline/video/ffmpeg/sink.rs

Purpose:
`FfmpegSink`: a `VideoSink` that encodes raw RGBA frames written to the stdin pipe of an
`ffmpeg` process, with a codec preset and optional audio taken from another file (usually the
source video: audio passthrough).

Key structures:
- `FfmpegSink` (public).

Notes:
`begin` checks ffmpeg's version, reserves a temporary file beside the output and spawns the
encoder writing it; `write_frame` checks size, alpha mode and opacity and writes the frame;
`finish` closes stdin, checks the exit status and renames the temporary file over the output;
`abort` / drop / any failure kill the encoder and delete the temporary file, so the output path
only ever holds a complete file (an existing one is replaced only on success). A write that fails
because ffmpeg exited reports `BrokenPipe` with ffmpeg's stderr. Frames that are not opaque are
rejected for codecs without alpha — never silently flattened.
*/

use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::process::ChildStdin;

use super::super::super::{AlphaMode, FrameRef};
use super::super::error::VideoError;
use super::super::stream::{VideoInfo, VideoSink};
use super::FfmpegTools;
use super::args::{AudioMode, EncodeCommand, VideoCodec, encode_args};
use super::paths::{TemporaryFile, same_file};
use super::process::ToolProcess;
use super::version::check_ffmpeg;

/// Encodes frames through the `ffmpeg` program (native only).
///
/// The encoder writes a temporary file beside the output (`<stem>.partial-<pid>-<n>.<ext>`),
/// renamed over the output when [`finish`](VideoSink::finish) succeeds and deleted on failure,
/// [`abort`](VideoSink::abort) or drop: the output is never left half-written, and an existing
/// file there is replaced only by a complete one. The output directory must be writable.
///
/// Colours are stored as [`VideoCodec`] describes (BT.709 limited-range YCbCr, tagged, for the
/// YCbCr presets); the stream's sample aspect ratio ([`VideoInfo::sample_aspect_ratio`]) is
/// stored with it.
#[derive(Debug)]
pub struct FfmpegSink {
    /// The file written (replaced on success).
    output: PathBuf,
    /// Codec preset.
    codec: VideoCodec,
    /// Audio input and treatment.
    audio: Option<(PathBuf, AudioMode)>,
    /// Program locations.
    tools: FfmpegTools,
    /// The running encoder, between `begin` and `finish`.
    running: Option<Running>,
}

/// The encoder of one stream.
#[derive(Debug)]
struct Running {
    /// The process. Declared (so dropped) before `stdin` and `temporary`: a dropped sink kills
    /// the encoder first, then deletes its unfinished file.
    process: ToolProcess,
    /// Its stdin (`None` once closed).
    stdin: Option<ChildStdin>,
    /// The stream being written.
    info: VideoInfo,
    /// Frames written.
    frames: u64,
    /// The file the encoder writes; renamed over the output on success.
    temporary: TemporaryFile,
}

impl FfmpegSink {
    /// A sink writing `output` with `codec`, without audio, using `ffmpeg` from `PATH`.
    #[must_use]
    pub fn new(output: impl Into<PathBuf>, codec: VideoCodec) -> Self {
        Self { output: output.into(), codec, audio: None, tools: FfmpegTools::default(), running: None }
    }

    /// Takes the audio of `input` (typically the source video) with `mode`: passthrough with
    /// [`AudioMode::Copy`]. An input without audio gives an output without audio. `input` must
    /// not be the output file ([`VideoError::OutputIsInput`] from `begin`).
    #[must_use]
    pub fn with_audio_from(mut self, input: impl Into<PathBuf>, mode: AudioMode) -> Self {
        self.audio = Some((input.into(), mode));
        self
    }

    /// Uses these program locations.
    #[must_use]
    pub fn with_tools(mut self, tools: FfmpegTools) -> Self {
        self.tools = tools;
        self
    }

    /// The output file.
    #[must_use]
    pub fn output(&self) -> &Path {
        &self.output
    }

    /// The codec preset.
    #[must_use]
    pub fn codec(&self) -> &VideoCodec {
        &self.codec
    }

    /// Turns a failed write into the most useful error: ffmpeg's own report when it exited.
    fn write_error(running: &mut Running, source: std::io::Error) -> VideoError {
        running.stdin = None;
        if source.kind() != ErrorKind::BrokenPipe {
            return VideoError::Io { context: "writing frames to ffmpeg".to_owned(), source };
        }
        match running.process.wait_status() {
            Ok((status, stderr_tail)) => VideoError::BrokenPipe { program: running.process.program().to_owned(), status: status.to_string(), stderr_tail },
            Err(error) => error,
        }
    }
}

impl VideoSink for FfmpegSink {
    fn begin(&mut self, info: &VideoInfo) -> Result<(), VideoError> {
        if self.running.is_some() {
            return Err(VideoError::InvalidState("`begin` was called twice"));
        }
        if self.codec.needs_even_size() && (!info.width().is_multiple_of(2) || !info.height().is_multiple_of(2)) {
            return Err(VideoError::UnsupportedSize {
                codec: self.codec.name(),
                width: info.width(),
                height: info.height(),
                reason: "4:2:0 chroma subsampling needs even sides; crop or pad the frames, or use ProRes 4444 / FFV1",
            });
        }
        if let Some((input, _)) = &self.audio
            && same_file(input, &self.output)
        {
            return Err(VideoError::OutputIsInput { path: self.output.clone() });
        }
        check_ffmpeg(self.tools.ffmpeg())?;
        if !self.codec.supports_audio() && self.audio.as_ref().is_some_and(|(_, mode)| *mode != AudioMode::Drop) {
            log::info!("egui_shader_layers (video): {} carries no audio; the input's audio is left out", self.codec.name());
        }
        let temporary = TemporaryFile::reserve(&self.output)?;
        let command = EncodeCommand {
            width: info.width(),
            height: info.height(),
            rate: info.frame_rate(),
            sample_aspect_ratio: info.sample_aspect_ratio(),
            codec: &self.codec,
            audio: self.audio.as_ref().map(|(input, mode)| (input.as_path(), mode)),
            output: temporary.path(),
        };
        log::info!("egui_shader_layers (video): encoding {} with {} (via {})", self.output.display(), self.codec.name(), temporary.path().display());
        // A spawn error drops `temporary`, deleting the reserved file.
        let mut process = ToolProcess::spawn(self.tools.ffmpeg(), &encode_args(&command), true, false)?;
        let stdin = process.take_stdin();
        self.running = Some(Running { process, stdin, info: info.clone(), frames: 0, temporary });
        Ok(())
    }

    fn write_frame(&mut self, frame: &FrameRef<'_>, _time: f64) -> Result<(), VideoError> {
        let Some(running) = self.running.as_mut() else {
            return Err(VideoError::InvalidState("`write_frame` was called before `begin` or after `finish`"));
        };
        let (width, height) = (running.info.width(), running.info.height());
        if (frame.width(), frame.height()) != (width, height) {
            return Err(VideoError::FrameSizeMismatch { frame: running.frames, expected_width: width, expected_height: height, actual_width: frame.width(), actual_height: frame.height() });
        }
        if frame.alpha() != AlphaMode::Straight {
            return Err(VideoError::NotStraightAlpha { alpha: frame.alpha() });
        }
        let pixels = frame.packed()?;
        if !self.codec.supports_alpha() && pixels.chunks_exact(4).any(|pixel| pixel[3] != u8::MAX) {
            return Err(VideoError::AlphaNotSupported { output: self.codec.name(), frame: running.frames });
        }
        let Some(stdin) = running.stdin.as_mut() else {
            return Err(VideoError::InvalidState("the encoder's input is already closed"));
        };
        if let Err(source) = stdin.write_all(&pixels) {
            return Err(Self::write_error(running, source));
        }
        running.frames += 1;
        Ok(())
    }

    fn finish(&mut self) -> Result<(), VideoError> {
        let Some(mut running) = self.running.take() else {
            return Err(VideoError::InvalidState("`finish` was called before `begin`"));
        };
        if let Some(mut stdin) = running.stdin.take() {
            if let Err(source) = stdin.flush() {
                return Err(Self::write_error(&mut running, source));
            }
            // Closing stdin is ffmpeg's end of input.
            drop(stdin);
        }
        // On an error `running` drops here: the partial file is deleted.
        running.process.wait()?;
        let Running { temporary, frames, .. } = running;
        temporary.persist(&self.output)?;
        log::info!("egui_shader_layers (video): wrote {} ({frames} frames)", self.output.display());
        Ok(())
    }

    fn abort(&mut self) {
        if let Some(mut running) = self.running.take() {
            running.process.kill();
            running.stdin = None;
            log::warn!("egui_shader_layers (video): encoding {} aborted after {} frames; nothing was written there", self.output.display(), running.frames);
            // Dropping `running` deletes the partial file.
        }
    }
}
