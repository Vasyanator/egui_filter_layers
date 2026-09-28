/*
File: offline/video/ffmpeg/source.rs

Purpose:
`FfmpegSource`: a `VideoSource` that probes a file with `ffprobe` and decodes its first real
video stream with `ffmpeg` to raw RGBA frames read from a stdout pipe, at a constant frame rate,
converted to sRGB with the stream's own colour tags (`color.rs`).

Key structures:
- `FfmpegSource` (public).

Notes:
Frame `i` has time `i / frame_rate` (the constant-rate output's pts minus the first pts). The
decoder is killed and reaped when the source is dropped before the end. A pipe that ends in the
middle of a frame is a `FrameBytesMismatch` (the decoded size differs from the probed one). After
an error every further read is an error too (never a disguised end of stream).
*/

use std::io::{ErrorKind, Read};
use std::path::{Path, PathBuf};
use std::process::ChildStdout;

use super::super::error::VideoError;
use super::super::stream::{FrameRate, VideoInfo, VideoSource};
use super::FfmpegTools;
use super::args::{DecodeCommand, decode_args};
use super::color::decode_color;
use super::probe::{Probe, probe};
use super::process::ToolProcess;
use super::version::{check_ffmpeg, require_decoder};

/// Decodes a video file through the `ffmpeg` program (native only).
#[derive(Debug)]
pub struct FfmpegSource {
    /// The decoder (`None` once it exited).
    process: Option<ToolProcess>,
    /// Its stdout.
    stdout: Option<ChildStdout>,
    /// The stream's properties.
    info: VideoInfo,
    /// What ffprobe reported.
    probe: Probe,
    /// The decoded file.
    input: PathBuf,
    /// Bytes of one frame.
    frame_bytes: usize,
    /// Index of the next frame.
    next: u64,
    /// Whether a read failed (then every later read fails too).
    failed: bool,
}

impl FfmpegSource {
    /// Opens `input` with `ffmpeg` / `ffprobe` from `PATH`, at its own average frame rate.
    ///
    /// # Errors
    /// As [`FfmpegSource::open_with`].
    pub fn open(input: impl AsRef<Path>) -> Result<Self, VideoError> {
        Self::open_with(input, &FfmpegTools::default(), None)
    }

    /// Opens `input` with `tools`, decoding at `frame_rate` (default: the input's average frame
    /// rate; required when ffprobe reports none or only an implausible guess).
    ///
    /// The first video stream that is not an attached picture (cover art) is decoded, with its
    /// sample aspect ratio kept in the [`VideoInfo`]. Colours are converted to sRGB with the
    /// stream's tagged YCbCr matrix and range; an untagged stream is taken as BT.709 when it is
    /// HD (at least 1280 wide or above 576 high) and BT.601 otherwise, as common players do (a
    /// warning is logged). VP8/VP9 alpha stored as a side channel is decoded with libvpx.
    ///
    /// # Errors
    /// [`VideoError::ToolNotFound`], [`VideoError::UnsupportedFfmpeg`] (older than 5.1),
    /// [`VideoError::ProcessFailed`] (an unreadable or unsupported file),
    /// [`VideoError::NoVideoStream`], [`VideoError::MalformedProbe`] (also when no frame rate is
    /// known), [`VideoError::UnsupportedColor`] (HDR or wide-gamut input),
    /// [`VideoError::DecoderNotAvailable`] (VP8/VP9 alpha without libvpx), [`VideoError::Spawn`].
    pub fn open_with(input: impl AsRef<Path>, tools: &FfmpegTools, frame_rate: Option<FrameRate>) -> Result<Self, VideoError> {
        let input = input.as_ref();
        check_ffmpeg(tools.ffmpeg())?;
        let probe = probe(tools.ffprobe(), input)?;
        let color = decode_color(&probe).map_err(|reason| VideoError::UnsupportedColor { input: input.to_owned(), reason })?;
        if color.guessed {
            log::warn!(
                "egui_filter_layers (video): {} has no colour matrix tag; decoding it as {} (players differ for untagged video)",
                input.display(),
                color.matrix.unwrap_or("RGB")
            );
        }
        let decoder = probe.alpha_decoder();
        if let Some(decoder) = decoder {
            require_decoder(tools.ffmpeg(), decoder, "keep the alpha channel of VP8/VP9 video")?;
        }
        let rate = frame_rate.or(probe.frame_rate).ok_or_else(|| {
            let note = probe.frame_rate_note.as_deref().map_or_else(String::new, |note| format!(" ({note})"));
            VideoError::MalformedProbe { input: input.to_owned(), reason: format!("the video stream has no reliable frame rate{note}; give one to `FfmpegSource::open_with`") }
        })?;
        let (sar_numerator, sar_denominator) = probe.sample_aspect_ratio.unwrap_or((1, 1));
        let info = VideoInfo::new(probe.width, probe.height, rate)
            .with_duration(probe.duration)
            .with_frame_count(probe.duration.and_then(|seconds| rate.frames_in(seconds)))
            .with_alpha(probe.has_alpha())
            .with_sample_aspect_ratio(sar_numerator, sar_denominator);
        let frame_bytes = info.frame_bytes()?;
        log::info!(
            "egui_filter_layers (video): decoding {} (stream {}, {} {}x{}, {}, colour {}/{}, {} fps, audio: {})",
            input.display(),
            probe.stream_index,
            probe.codec_name,
            probe.width,
            probe.height,
            probe.pixel_format,
            probe.color_space,
            probe.color_range,
            rate,
            probe.has_audio
        );
        let filter = color.filter();
        let command = DecodeCommand { input, stream_index: probe.stream_index, rate, decoder, filter: &filter };
        let mut process = ToolProcess::spawn(tools.ffmpeg(), &decode_args(&command), false, true)?;
        let stdout = process.take_stdout();
        Ok(Self { process: Some(process), stdout, info, probe, input: input.to_owned(), frame_bytes, next: 0, failed: false })
    }

    /// The decoded file (e.g. for [`FfmpegSink::with_audio_from`](super::FfmpegSink::with_audio_from)).
    #[must_use]
    pub fn input(&self) -> &Path {
        &self.input
    }

    /// Whether the file has an audio stream.
    #[must_use]
    pub fn has_audio(&self) -> bool {
        self.probe.has_audio
    }

    /// The video stream's pixel format as ffprobe names it (e.g. `yuv420p`).
    #[must_use]
    pub fn pixel_format(&self) -> &str {
        &self.probe.pixel_format
    }

    /// The video stream's YCbCr matrix tag as ffprobe names it (`bt709`, `smpte170m`,
    /// `unknown`, ...); see [`FfmpegSource::open_with`] for how untagged streams are decoded.
    #[must_use]
    pub fn color_space(&self) -> &str {
        &self.probe.color_space
    }

    /// The video stream's colour range tag (`tv`, `pc` or `unknown`).
    #[must_use]
    pub fn color_range(&self) -> &str {
        &self.probe.color_range
    }

    /// Ends the stream: waits for the decoder and checks its exit status.
    fn finish(&mut self) -> Result<(), VideoError> {
        self.stdout = None;
        match self.process.take() {
            Some(mut process) => process.wait(),
            None => Ok(()),
        }
    }
}

impl VideoSource for FfmpegSource {
    fn info(&self) -> &VideoInfo {
        &self.info
    }

    fn read_frame(&mut self, pixels: &mut Vec<u8>) -> Result<Option<f64>, VideoError> {
        if self.failed {
            return Err(VideoError::InvalidState("the ffmpeg source already failed; open the file again"));
        }
        let result = self.read_next(pixels);
        self.failed = result.is_err();
        result
    }
}

impl FfmpegSource {
    /// One read of [`VideoSource::read_frame`], without the failure latch.
    fn read_next(&mut self, pixels: &mut Vec<u8>) -> Result<Option<f64>, VideoError> {
        let Some(stdout) = self.stdout.as_mut() else {
            return Ok(None);
        };
        pixels.resize(self.frame_bytes, 0);
        let filled = read_full(stdout, pixels).map_err(|source| VideoError::Io { context: "reading decoded frames from ffmpeg".to_owned(), source })?;
        if filled == self.frame_bytes {
            let time = self.info.frame_rate().frame_time(self.next);
            self.next += 1;
            return Ok(Some(time));
        }
        // End of the pipe: a clean end falls on a frame boundary and a successful exit.
        let status = self.finish();
        if filled == 0 {
            status?;
            log::info!("egui_filter_layers (video): decoded {} frames of {}", self.next, self.input.display());
            return Ok(None);
        }
        let stderr_tail = match status {
            Err(VideoError::ProcessFailed { stderr_tail, .. }) => stderr_tail,
            Err(other) => return Err(other),
            Ok(()) => String::new(),
        };
        Err(VideoError::FrameBytesMismatch { frame: self.next, expected_bytes: self.frame_bytes, actual_bytes: filled, stderr_tail })
    }
}

/// Fills `buffer` from `reader` until it is full or the stream ends; returns the bytes read.
fn read_full(reader: &mut impl Read, buffer: &mut [u8]) -> std::io::Result<usize> {
    let mut filled = 0;
    while let Some(rest) = buffer.get_mut(filled..).filter(|rest| !rest.is_empty()) {
        match reader.read(rest) {
            Ok(0) => break,
            Ok(read) => filled += read,
            Err(error) if error.kind() == ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
    Ok(filled)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_are_read_whole_across_short_reads() {
        /// Returns at most 3 bytes per read.
        struct Trickle(std::io::Cursor<Vec<u8>>);
        impl Read for Trickle {
            fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
                let len = buffer.len().min(3);
                self.0.read(&mut buffer[..len])
            }
        }
        let mut reader = Trickle(std::io::Cursor::new((0..20).collect()));
        let mut frame = [0_u8; 8];
        assert_eq!(read_full(&mut reader, &mut frame).ok(), Some(8));
        assert_eq!(frame, [0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(read_full(&mut reader, &mut frame).ok(), Some(8));
        assert_eq!(read_full(&mut reader, &mut frame).ok(), Some(4), "a truncated frame");
        assert_eq!(read_full(&mut reader, &mut frame).ok(), Some(0), "the end");
    }
}
