/*
File: offline/video/sequence.rs

Purpose:
Image sequences as video (features `video` + `image`, native only): `ImageSequenceSource` reads
numbered files (`frame_%04d.png`) from a folder at a given frame rate, `ImageSequenceSink` writes
them. Decoding and encoding go through the `image` crate.

Key structures:
- `ImageSequenceSource`, `ImageSequenceSink` (public).
- `Pattern`: a parsed printf-style file-name pattern with one `%d` / `%0Nd` (unit-tested).

Notes:
The source starts at the given number, else at 0 or 1 (whichever exists, like ffmpeg's image2
demuxer) and ends at the first missing number; it counts the files up front (progress). Every
frame must have the first frame's size. The sink numbers from 1 by default, creates the folder,
and rejects non-opaque frames for formats without alpha (JPEG, ...), like `process_file`.
*/

use std::path::{Path, PathBuf};

use image::{DynamicImage, ImageFormat, RgbaImage};

use super::super::{FrameRef, OfflineError, image::format_has_alpha};
use super::error::VideoError;
use super::stream::{FrameRate, VideoInfo, VideoSink, VideoSource};

/// A file-name pattern with exactly one frame-number placeholder.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Pattern {
    /// Text before the number (`%%` already unescaped).
    prefix: String,
    /// Minimum digits (zero-padded); 0 = no padding.
    digits: usize,
    /// Text after the number (`%%` already unescaped).
    suffix: String,
}

impl Pattern {
    /// Parses `frame_%04d.png`, `%d.jpg`, `100%%_%03d.png`.
    ///
    /// # Errors
    /// [`VideoError::InvalidPattern`] without exactly one `%d` / `%0Nd` (`N` at most 9), with
    /// another `%` sequence, or with a path separator (the folder is given separately).
    fn parse(pattern: &str) -> Result<Self, VideoError> {
        let invalid = |reason| VideoError::InvalidPattern { pattern: pattern.to_owned(), reason };
        if pattern.contains(['/', '\\']) {
            return Err(invalid("give the folder separately; the pattern is a file name"));
        }
        let (mut prefix, mut suffix) = (String::new(), String::new());
        let mut digits = None;
        let mut chars = pattern.chars().peekable();
        while let Some(c) = chars.next() {
            let target = if digits.is_some() { &mut suffix } else { &mut prefix };
            if c != '%' {
                target.push(c);
                continue;
            }
            if chars.next_if_eq(&'%').is_some() {
                target.push('%');
                continue;
            }
            let mut width = String::new();
            while let Some(digit) = chars.next_if(char::is_ascii_digit) {
                width.push(digit);
            }
            if chars.next() != Some('d') {
                return Err(invalid("only `%d`, `%0Nd` and `%%` are supported"));
            }
            if digits.is_some() {
                return Err(invalid("the frame number may appear only once"));
            }
            if !(width.is_empty() || (width.starts_with('0') && width.len() == 2)) {
                return Err(invalid("the number width must be `%0N` with N from 1 to 9"));
            }
            digits = Some(width.get(1..).and_then(|n| n.parse::<usize>().ok()).unwrap_or(0));
        }
        let digits = digits.ok_or_else(|| invalid("no `%d` / `%0Nd` frame-number placeholder"))?;
        Ok(Self { prefix, digits, suffix })
    }

    /// The file name of frame number `number`.
    fn name(&self, number: u64) -> String {
        format!("{}{number:0width$}{}", self.prefix, self.suffix, width = self.digits)
    }
}

/// Reads a numbered image sequence as a video (native only).
#[derive(Debug)]
pub struct ImageSequenceSource {
    /// The folder.
    dir: PathBuf,
    /// File names.
    pattern: Pattern,
    /// Number of the first file.
    start: u64,
    /// Index of the next frame.
    next: u64,
    /// The stream's properties (exact frame count).
    info: VideoInfo,
}

impl ImageSequenceSource {
    /// Opens `dir/pattern` (e.g. `frame_%04d.png`) played at `frame_rate`, starting at number 0
    /// or 1, whichever exists.
    ///
    /// # Errors
    /// As [`ImageSequenceSource::open_from`].
    pub fn open(dir: impl Into<PathBuf>, pattern: &str, frame_rate: FrameRate) -> Result<Self, VideoError> {
        Self::open_from(dir, pattern, frame_rate, None)
    }

    /// Opens `dir/pattern` starting at number `start` (default: 0 or 1, whichever exists). The
    /// first image decides the size and whether the stream has alpha.
    ///
    /// # Errors
    /// [`VideoError::InvalidPattern`], [`VideoError::EmptySequence`] when the first file is
    /// missing, [`VideoError::Offline`] (`Image`) when it cannot be decoded.
    pub fn open_from(dir: impl Into<PathBuf>, pattern: &str, frame_rate: FrameRate, start: Option<u64>) -> Result<Self, VideoError> {
        let dir = dir.into();
        let pattern = Pattern::parse(pattern)?;
        let path = |number: u64| dir.join(pattern.name(number));
        let start = start.or_else(|| [0, 1].into_iter().find(|&number| path(number).is_file())).unwrap_or(0);
        if !path(start).is_file() {
            return Err(VideoError::EmptySequence { first: path(start) });
        }
        let count = (start..u64::MAX).take_while(|&number| path(number).is_file()).count();
        let first = decode(&path(start))?;
        let has_alpha = first.color().has_alpha();
        let frames = u64::try_from(count).ok();
        let info = VideoInfo::new(first.width(), first.height(), frame_rate)
            .with_frame_count(frames)
            .with_duration(frames.map(|frames| frame_rate.frame_time(frames)))
            .with_alpha(has_alpha);
        log::info!("egui_filter_layers (video): image sequence {} ({count} frames from {start}, {}x{})", dir.join(pattern.name(start)).display(), first.width(), first.height());
        Ok(Self { dir, pattern, start, next: 0, info })
    }
}

impl VideoSource for ImageSequenceSource {
    fn info(&self) -> &VideoInfo {
        &self.info
    }

    fn read_frame(&mut self, pixels: &mut Vec<u8>) -> Result<Option<f64>, VideoError> {
        let Some(number) = self.start.checked_add(self.next) else {
            return Ok(None);
        };
        let path = self.dir.join(self.pattern.name(number));
        if !path.is_file() {
            return Ok(None);
        }
        let image = decode(&path)?.to_rgba8();
        if (image.width(), image.height()) != (self.info.width(), self.info.height()) {
            return Err(VideoError::FrameSizeMismatch { frame: self.next, expected_width: self.info.width(), expected_height: self.info.height(), actual_width: image.width(), actual_height: image.height() });
        }
        pixels.clear();
        pixels.extend_from_slice(image.as_raw());
        let time = self.info.frame_rate().frame_time(self.next);
        self.next += 1;
        Ok(Some(time))
    }
}

/// Decodes `path` (format from its content).
fn decode(path: &Path) -> Result<DynamicImage, VideoError> {
    let reader = image::ImageReader::open(path).map_err(|source| VideoError::Io { context: format!("opening {}", path.display()), source })?;
    let reader = reader.with_guessed_format().map_err(|source| VideoError::Io { context: format!("reading {}", path.display()), source })?;
    reader.decode().map_err(|error| VideoError::Offline(OfflineError::Image(error)))
}

/// Writes frames as numbered images (native only). The format comes from the pattern's
/// extension (PNG and JPEG built in; more through the `image` crate's features).
#[derive(Debug)]
pub struct ImageSequenceSink {
    /// The folder (created by `begin`).
    dir: PathBuf,
    /// File names.
    pattern: Pattern,
    /// Encoder format.
    format: ImageFormat,
    /// Number of the first file.
    start: u64,
    /// The stream, between `begin` and `finish`.
    info: Option<VideoInfo>,
    /// Frames written.
    frames: u64,
}

impl ImageSequenceSink {
    /// Writes `dir/pattern` (e.g. `frame_%04d.png`), numbering from 1. Existing files are
    /// overwritten.
    ///
    /// # Errors
    /// [`VideoError::InvalidPattern`], also when the extension names no image format. A format
    /// whose encoder is not compiled in fails when the first frame is written.
    pub fn new(dir: impl Into<PathBuf>, pattern: &str) -> Result<Self, VideoError> {
        let pattern_parsed = Pattern::parse(pattern)?;
        let extension = pattern_parsed.suffix.rsplit_once('.').map(|(_, extension)| extension);
        let format = extension.and_then(ImageFormat::from_extension).ok_or_else(|| VideoError::InvalidPattern { pattern: pattern.to_owned(), reason: "the file extension names no image format the `image` crate knows" })?;
        Ok(Self { dir: dir.into(), pattern: pattern_parsed, format, start: 1, info: None, frames: 0 })
    }

    /// Numbers files from `start` instead of 1.
    #[must_use]
    pub fn with_start_number(mut self, start: u64) -> Self {
        self.start = start;
        self
    }
}

impl VideoSink for ImageSequenceSink {
    fn begin(&mut self, info: &VideoInfo) -> Result<(), VideoError> {
        std::fs::create_dir_all(&self.dir).map_err(|source| VideoError::Io { context: format!("creating {}", self.dir.display()), source })?;
        self.info = Some(info.clone());
        self.frames = 0;
        Ok(())
    }

    fn write_frame(&mut self, frame: &FrameRef<'_>, _time: f64) -> Result<(), VideoError> {
        let Some(info) = &self.info else {
            return Err(VideoError::InvalidState("`write_frame` was called before `begin` or after `finish`"));
        };
        if (frame.width(), frame.height()) != (info.width(), info.height()) {
            return Err(VideoError::FrameSizeMismatch { frame: self.frames, expected_width: info.width(), expected_height: info.height(), actual_width: frame.width(), actual_height: frame.height() });
        }
        if frame.alpha() != super::super::AlphaMode::Straight {
            return Err(VideoError::NotStraightAlpha { alpha: frame.alpha() });
        }
        let pixels = frame.packed()?.into_owned();
        let opaque = pixels.chunks_exact(4).all(|pixel| pixel[3] == u8::MAX);
        let image = RgbaImage::from_raw(frame.width(), frame.height(), pixels).ok_or(OfflineError::SizeOverflow { what: "an output image" })?;
        let image = if format_has_alpha(self.format) {
            DynamicImage::ImageRgba8(image)
        } else if opaque {
            DynamicImage::ImageRgb8(DynamicImage::ImageRgba8(image).to_rgb8())
        } else {
            return Err(VideoError::AlphaNotSupported { output: format!("{:?}", self.format), frame: self.frames });
        };
        let number = self.start.checked_add(self.frames).ok_or(OfflineError::SizeOverflow { what: "a frame number" })?;
        let path = self.dir.join(self.pattern.name(number));
        image.save_with_format(&path, self.format).map_err(|error| VideoError::Offline(OfflineError::Image(error)))?;
        self.frames += 1;
        Ok(())
    }

    fn finish(&mut self) -> Result<(), VideoError> {
        if self.info.take().is_none() {
            return Err(VideoError::InvalidState("`finish` was called before `begin`"));
        }
        log::info!("egui_filter_layers (video): wrote {} images to {}", self.frames, self.dir.display());
        Ok(())
    }

    fn abort(&mut self) {
        self.info = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patterns_parse_and_format() {
        let pattern = Pattern::parse("frame_%04d.png").unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(pattern.name(7), "frame_0007.png");
        assert_eq!(pattern.name(123_456), "frame_123456.png");
        assert_eq!(Pattern::parse("%d.jpg").map(|pattern| pattern.name(42)).ok().as_deref(), Some("42.jpg"));
        assert_eq!(Pattern::parse("100%%_%03d.png").map(|pattern| pattern.name(5)).ok().as_deref(), Some("100%_005.png"));
        for bad in ["frame.png", "%d_%d.png", "%s.png", "%4d.png", "%010d.png", "a/%d.png", "%", "x%0"] {
            assert!(matches!(Pattern::parse(bad), Err(VideoError::InvalidPattern { .. })), "{bad}");
        }
    }

    #[test]
    fn a_sequence_round_trips_through_the_folder() {
        let dir = std::env::temp_dir().join(format!("egui_filter_layers_sequence_{}", std::process::id()));
        let rate = FrameRate::new(10, 1).unwrap_or_else(|error| panic!("{error}"));
        let info = VideoInfo::new(3, 2, rate);
        let frames: Vec<Vec<u8>> = (0..3_u8).map(|index| (0..24).map(|byte: u8| if byte % 4 == 3 { 255 } else { byte.wrapping_mul(7).wrapping_add(index * 40) }).collect()).collect();
        let mut sink = ImageSequenceSink::new(&dir, "f_%03d.png").unwrap_or_else(|error| panic!("{error}")).with_start_number(0);
        assert!(sink.begin(&info).is_ok());
        for (index, frame) in frames.iter().enumerate() {
            let time = rate.frame_time(u64::try_from(index).unwrap_or(0));
            assert!(sink.write_frame(&FrameRef::new(frame, 3, 2, super::super::super::AlphaMode::Straight), time).is_ok());
        }
        assert!(sink.finish().is_ok());
        assert!(dir.join("f_000.png").is_file() && dir.join("f_002.png").is_file());

        let mut source = ImageSequenceSource::open(&dir, "f_%03d.png", rate).unwrap_or_else(|error| panic!("{error}"));
        assert_eq!((source.info().width(), source.info().height(), source.info().frame_count()), (3, 2, Some(3)));
        let mut pixels = Vec::new();
        for (index, expected) in frames.iter().enumerate() {
            let time = source.read_frame(&mut pixels).unwrap_or_else(|error| panic!("{error}"));
            assert_eq!(time, Some(rate.frame_time(u64::try_from(index).unwrap_or(0))));
            assert_eq!(&pixels, expected);
        }
        assert!(matches!(source.read_frame(&mut pixels), Ok(None)));

        let mut jpeg = ImageSequenceSink::new(&dir, "j_%d.jpg").unwrap_or_else(|error| panic!("{error}"));
        assert!(jpeg.begin(&info).is_ok());
        let translucent = vec![10_u8; 24];
        assert!(matches!(jpeg.write_frame(&FrameRef::new(&translucent, 3, 2, super::super::super::AlphaMode::Straight), 0.0), Err(VideoError::AlphaNotSupported { frame: 0, .. })));
        assert!(matches!(ImageSequenceSource::open(&dir, "missing_%d.png", rate), Err(VideoError::EmptySequence { .. })));
        assert!(std::fs::remove_dir_all(&dir).is_ok());
    }
}
