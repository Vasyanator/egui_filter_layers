/*
File: offline/video/error.rs

Purpose:
`VideoError`, the error type of video processing (feature `video`): the offline core's errors,
stream-contract violations (frame sizes, frame rates), the ffmpeg / ffprobe adapters (program
missing, non-zero exit with the tail of its stderr, malformed probe output, broken pipes), image
sequences, missing alpha support, cancellation, and errors of user-written sources and sinks.

Notes:
Messages are user-facing sentences; variants carry the data needed for diagnosis (paths, sizes,
exit status, the last lines ffmpeg printed) and never pixel data.
*/

use std::path::PathBuf;

use super::super::{AlphaMode, OfflineError};

/// Why processing a video (or opening / writing one) failed.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum VideoError {
    /// Filtering a frame failed (GPU, invalid options, ...).
    #[error(transparent)]
    Offline(#[from] OfflineError),
    /// A frame rate is zero, negative or unparsable.
    #[error("invalid frame rate `{text}`: expected a positive `numerator/denominator` or integer")]
    InvalidFrameRate {
        /// The rejected value.
        text: String,
    },
    /// A frame does not have the stream's size.
    #[error("frame {frame} is {actual_width}x{actual_height} pixels, but the stream is {expected_width}x{expected_height}")]
    FrameSizeMismatch {
        /// Index of the frame (0-based).
        frame: u64,
        /// Width announced by the stream.
        expected_width: u32,
        /// Height announced by the stream.
        expected_height: u32,
        /// Width of the frame.
        actual_width: u32,
        /// Height of the frame.
        actual_height: u32,
    },
    /// A frame buffer has the wrong length for the stream's size.
    #[error("frame {frame} holds {actual_bytes} bytes, but the stream's frames are {expected_bytes} bytes (RGBA8, tightly packed){}", stderr_suffix(stderr_tail))]
    FrameBytesMismatch {
        /// Index of the frame (0-based).
        frame: u64,
        /// `width * height * 4`.
        expected_bytes: usize,
        /// Bytes delivered.
        actual_bytes: usize,
        /// Last lines of the decoder's stderr, when a decoder process produced the frame.
        stderr_tail: String,
    },
    /// The ffmpeg or ffprobe program could not be found.
    #[error(
        "`{}` was not found; install FFmpeg (it provides both `ffmpeg` and `ffprobe`, https://ffmpeg.org/download.html) \
         and put it on PATH, or give its location with `FfmpegTools::new` / `FfmpegTools::in_dir`",
        program.display()
    )]
    ToolNotFound {
        /// The program that was looked up.
        program: PathBuf,
    },
    /// Starting a program failed for another reason (permissions, resources).
    #[error("could not start `{}`: {source}", program.display())]
    Spawn {
        /// The program.
        program: PathBuf,
        /// The operating system's error.
        source: std::io::Error,
    },
    /// A program exited with an error.
    #[error("`{}` failed ({status}){}", program.display(), stderr_suffix(stderr_tail))]
    ProcessFailed {
        /// The program.
        program: PathBuf,
        /// Its exit status, as printed by the OS.
        status: String,
        /// The last lines it wrote to stderr.
        stderr_tail: String,
    },
    /// The encoder closed its input before all frames were written (it usually exited on an
    /// error: an unsupported codec, container or option).
    #[error("`{}` stopped reading frames ({status}){}", program.display(), stderr_suffix(stderr_tail))]
    BrokenPipe {
        /// The program.
        program: PathBuf,
        /// Its exit status, as printed by the OS.
        status: String,
        /// The last lines it wrote to stderr.
        stderr_tail: String,
    },
    /// ffprobe's output could not be understood.
    #[error("could not read the video properties of `{}` from ffprobe: {reason}", input.display())]
    MalformedProbe {
        /// The probed input.
        input: PathBuf,
        /// What was missing or invalid.
        reason: String,
    },
    /// The input has no video stream.
    #[error("`{}` has no video stream", input.display())]
    NoVideoStream {
        /// The probed input.
        input: PathBuf,
    },
    /// Reading or writing a pipe or file failed.
    #[error("{context}: {source}")]
    Io {
        /// What was being done.
        context: String,
        /// The operating system's error.
        source: std::io::Error,
    },
    /// The output cannot store transparency, and a frame is not opaque.
    #[error(
        "{output} has no alpha channel, but frame {frame} is not opaque; set a background colour \
         (`OfflineOptions::with_background`) or choose an output with alpha"
    )]
    AlphaNotSupported {
        /// The output format or codec.
        output: String,
        /// Index of the first non-opaque frame.
        frame: u64,
    },
    /// The output codec cannot encode frames of this size.
    #[error("{codec} cannot encode {width}x{height} frames: {reason}")]
    UnsupportedSize {
        /// The codec.
        codec: String,
        /// Frame width.
        width: u32,
        /// Frame height.
        height: u32,
        /// Why.
        reason: &'static str,
    },
    /// An image-sequence file-name pattern is invalid.
    #[error("invalid file-name pattern `{pattern}`: {reason}")]
    InvalidPattern {
        /// The pattern.
        pattern: String,
        /// Why it is rejected.
        reason: &'static str,
    },
    /// An image sequence has no frame at its start.
    #[error("no image `{}` starts the sequence", first.display())]
    EmptySequence {
        /// The first file looked for.
        first: PathBuf,
    },
    /// A source or sink was used out of order (`write_frame` before `begin`, after `finish`, a
    /// read after the stream failed, ...).
    #[error("a video source or sink is used out of order: {0}")]
    InvalidState(&'static str),
    /// A sink that takes straight-alpha frames received a frame with another alpha mode.
    #[error("the video sink takes straight-alpha frames, but got a {alpha:?} frame; convert it or let `process_video` produce straight frames")]
    NotStraightAlpha {
        /// The alpha mode of the rejected frame.
        alpha: AlphaMode,
    },
    /// The output file is the input file (writing it would destroy the input while it is read).
    #[error("the output `{}` is the input file; choose another output path", path.display())]
    OutputIsInput {
        /// The path given as output.
        path: PathBuf,
    },
    /// The installed ffmpeg is too old for the command lines this crate builds.
    #[error("`{}` is FFmpeg {version}; FFmpeg 5.1 or newer is required (https://ffmpeg.org/download.html)", program.display())]
    UnsupportedFfmpeg {
        /// The program.
        program: PathBuf,
        /// The version it reported.
        version: String,
    },
    /// The installed ffmpeg lacks a decoder needed for this input.
    #[error("`{}` has no `{decoder}` decoder, which is needed to {purpose}; install an FFmpeg build with it", program.display())]
    DecoderNotAvailable {
        /// The program.
        program: PathBuf,
        /// The missing decoder.
        decoder: String,
        /// What it is needed for.
        purpose: &'static str,
    },
    /// The input's colour encoding cannot be converted to the sRGB frames the filters work on.
    #[error("`{}` uses an unsupported colour encoding: {reason}", input.display())]
    UnsupportedColor {
        /// The probed input.
        input: PathBuf,
        /// Which colour property is unsupported, and why.
        reason: String,
    },
    /// The progress callback cancelled processing.
    #[error("video processing was cancelled after {frames} frames")]
    Cancelled {
        /// Frames written before the cancellation.
        frames: u64,
    },
    /// An error of a user-implemented [`VideoSource`](super::VideoSource) or
    /// [`VideoSink`](super::VideoSink).
    ///
    /// Display and [`source`](std::error::Error::source) are forwarded to the wrapped error.
    #[error(transparent)]
    Custom(Box<dyn std::error::Error + Send + Sync + 'static>),
}

/// `": <tail>"` on a new line when a program printed something, else nothing.
fn stderr_suffix(tail: &str) -> String {
    if tail.trim().is_empty() { String::new() } else { format!(":\n{}", tail.trim_end()) }
}
