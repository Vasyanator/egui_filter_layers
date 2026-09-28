/*
File: offline/video/stream.rs

Purpose:
The stream model of video processing: `FrameRate` (an exact rational), `VideoInfo` (stream
metadata), and the two small object-safe traits every source and sink implements, `VideoSource`
and `VideoSink`.

Key structures:
- `FrameRate`: `numerator / denominator` frames per second, reduced; frame index -> time.
- `VideoInfo`: size, frame rate, optional duration and frame count, alpha presence, sample
  aspect ratio.

Key traits:
- `VideoSource`: `info()` + `read_frame()` (straight RGBA8 into a reused buffer, with its time).
- `VideoSink`: `begin()` / `write_frame()` / `finish()` / `abort()`.

Notes:
The traits carry no codec or library types, so they can wrap ffmpeg-next, GStreamer, a camera or
an in-memory buffer; the trait docs show how. Frames are straight-alpha RGBA8, tightly packed,
top row first, the stream's size.
*/

use std::fmt;
use std::str::FromStr;

use super::super::FrameRef;
use super::error::VideoError;

/// A frame rate as an exact fraction (`30000/1001` for NTSC's 29.97), always positive and
/// reduced to lowest terms.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FrameRate {
    /// Frames per `denominator` seconds (> 0).
    numerator: u32,
    /// Seconds per `numerator` frames (> 0).
    denominator: u32,
}

impl FrameRate {
    /// `numerator / denominator` frames per second.
    ///
    /// # Errors
    /// [`VideoError::InvalidFrameRate`] when either part is zero.
    pub fn new(numerator: u32, denominator: u32) -> Result<Self, VideoError> {
        if numerator == 0 || denominator == 0 {
            return Err(VideoError::InvalidFrameRate { text: format!("{numerator}/{denominator}") });
        }
        let divisor = gcd(numerator, denominator);
        Ok(Self { numerator: numerator / divisor, denominator: denominator / divisor })
    }

    /// Frames per `denominator` seconds.
    #[must_use]
    pub fn numerator(&self) -> u32 {
        self.numerator
    }

    /// Seconds per `numerator` frames.
    #[must_use]
    pub fn denominator(&self) -> u32 {
        self.denominator
    }

    /// Frames per second as a float (for display; use the fraction for timing).
    #[must_use]
    pub fn as_f64(&self) -> f64 {
        f64::from(self.numerator) / f64::from(self.denominator)
    }

    /// Presentation time of frame `index` of a constant-rate stream starting at 0, in seconds:
    /// `index * denominator / numerator`, computed in integers and exact to `f64` rounding.
    #[must_use]
    pub fn frame_time(&self, index: u64) -> f64 {
        let ticks = u128::from(index) * u128::from(self.denominator);
        let numerator = u128::from(self.numerator);
        let (whole, rest) = (ticks / numerator, ticks % numerator);
        // `whole` fits `f64` exactly below 2^53 seconds and `rest < numerator <= u32::MAX`;
        // the conversions only round beyond any real video length.
        #[allow(clippy::cast_precision_loss)]
        let seconds = whole as f64 + rest as f64 / f64::from(self.numerator);
        seconds
    }

    /// Number of frames a constant-rate stream of `seconds` holds, rounded to the nearest frame
    /// (for progress estimates). `None` for a negative or non-finite duration.
    #[must_use]
    pub fn frames_in(&self, seconds: f64) -> Option<u64> {
        let frames = (seconds * self.as_f64()).round();
        // `u64::MAX as f64` rounds up to 2^64, which `<` excludes, so the cast below is exact.
        #[allow(clippy::cast_precision_loss)]
        let limit = u64::MAX as f64;
        if !(frames.is_finite() && frames >= 0.0 && frames < limit) {
            return None;
        }
        // Finite, non-negative, below 2^64 and integral (rounded above): the cast is exact.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let count = frames as u64;
        Some(count)
    }
}

impl fmt::Display for FrameRate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.numerator, self.denominator)
    }
}

impl FromStr for FrameRate {
    type Err = VideoError;

    /// Parses `"30000/1001"` (ffprobe's format) or an integer `"25"`.
    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let invalid = || VideoError::InvalidFrameRate { text: text.to_owned() };
        let (numerator, denominator) = text.trim().split_once('/').unwrap_or((text.trim(), "1"));
        let numerator = numerator.trim().parse::<u32>().map_err(|_| invalid())?;
        let denominator = denominator.trim().parse::<u32>().map_err(|_| invalid())?;
        Self::new(numerator, denominator).map_err(|_| invalid())
    }
}

/// Greatest common divisor (Euclid); `gcd(a, 0) = a`.
fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// What a [`VideoSource`] delivers and a [`VideoSink`] receives.
#[derive(Clone, Debug, PartialEq)]
pub struct VideoInfo {
    /// Frame width in pixels.
    width: u32,
    /// Frame height in pixels.
    height: u32,
    /// Constant frame rate of the stream.
    frame_rate: FrameRate,
    /// Length in seconds, when known.
    duration: Option<f64>,
    /// Number of frames, when known or estimated.
    frame_count: Option<u64>,
    /// Whether the frames can be translucent.
    has_alpha: bool,
    /// Sample (pixel) aspect ratio, reduced; `(1, 1)` for square pixels.
    sample_aspect_ratio: (u32, u32),
}

impl VideoInfo {
    /// A stream of `width x height` frames at `frame_rate`, opaque, of unknown length.
    #[must_use]
    pub fn new(width: u32, height: u32, frame_rate: FrameRate) -> Self {
        Self { width, height, frame_rate, duration: None, frame_count: None, has_alpha: false, sample_aspect_ratio: (1, 1) }
    }

    /// Sets the length in seconds (progress reporting).
    #[must_use]
    pub fn with_duration(mut self, seconds: Option<f64>) -> Self {
        self.duration = seconds;
        self
    }

    /// Sets the number of frames, exact or estimated (progress reporting only).
    #[must_use]
    pub fn with_frame_count(mut self, frames: Option<u64>) -> Self {
        self.frame_count = frames;
        self
    }

    /// Declares whether frames can be translucent (a codec / pixel format with alpha).
    #[must_use]
    pub fn with_alpha(mut self, has_alpha: bool) -> Self {
        self.has_alpha = has_alpha;
        self
    }

    /// Sets the sample (pixel) aspect ratio `numerator : denominator` of anamorphic video: a
    /// pixel is displayed `numerator / denominator` times as wide as it is tall. A zero in
    /// either term means "unknown", which is square (ffmpeg's convention).
    ///
    /// Frames are always processed at their stored size, so effect distances are in stored
    /// pixels (a round effect looks stretched once the video is displayed at its aspect). Sinks
    /// that can store the ratio (ffmpeg) keep it; image sequences cannot and ignore it.
    #[must_use]
    pub fn with_sample_aspect_ratio(mut self, numerator: u32, denominator: u32) -> Self {
        self.sample_aspect_ratio = if numerator == 0 || denominator == 0 {
            (1, 1)
        } else {
            let divisor = gcd(numerator, denominator);
            (numerator / divisor, denominator / divisor)
        };
        self
    }

    /// Frame width in pixels.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Frame height in pixels.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Frame rate.
    #[must_use]
    pub fn frame_rate(&self) -> FrameRate {
        self.frame_rate
    }

    /// Length in seconds, when known.
    #[must_use]
    pub fn duration(&self) -> Option<f64> {
        self.duration
    }

    /// Number of frames, when known (possibly an estimate from the duration).
    #[must_use]
    pub fn frame_count(&self) -> Option<u64> {
        self.frame_count
    }

    /// Whether frames can be translucent.
    #[must_use]
    pub fn has_alpha(&self) -> bool {
        self.has_alpha
    }

    /// Sample (pixel) aspect ratio `(numerator, denominator)`, reduced; `(1, 1)` for square
    /// pixels.
    #[must_use]
    pub fn sample_aspect_ratio(&self) -> (u32, u32) {
        self.sample_aspect_ratio
    }

    /// `width * height * 4`: bytes of one frame.
    ///
    /// # Errors
    /// [`VideoError::Offline`] (`SizeOverflow`) when it does not fit `usize`.
    pub fn frame_bytes(&self) -> Result<usize, VideoError> {
        Ok(super::super::frame::frame_bytes(self.width, self.height)?)
    }
}

/// A stream of decoded frames.
///
/// Frames are straight-alpha (what decoders produce) RGBA8, tightly packed, top row first, all of
/// `info()`'s size. The trait is object-safe (`&mut dyn VideoSource`).
///
/// Wrapping a decoder library: with [ffmpeg-next](https://crates.io/crates/ffmpeg-next), open the
/// input, take the best video stream, and in `read_frame` receive decoded frames, convert them
/// with a `software::scaling::Context` to `Pixel::RGBA` and copy the rows (the frame's stride is
/// usually larger than `width * 4`) into `pixels`; the time is
/// `(frame.pts() - first_pts) * stream.time_base()`. With `GStreamer`, end the pipeline in an
/// `appsink` with caps `video/x-raw,format=RGBA`, pull samples in `read_frame`, copy the mapped
/// buffer's rows, and take the time from the buffer's PTS minus the first one. Report
/// `Ok(None)` at end of stream.
pub trait VideoSource {
    /// The stream's properties; constant for the whole stream.
    fn info(&self) -> &VideoInfo;

    /// Reads the next frame into `pixels` (resized to `width * height * 4`; its capacity is
    /// reused between calls) and returns its presentation time in seconds, counted from the
    /// first frame (`fl.time` of that frame). `Ok(None)` at the end of the stream.
    ///
    /// # Errors
    /// Whatever the source hits; custom sources wrap their errors in [`VideoError::Custom`].
    fn read_frame(&mut self, pixels: &mut Vec<u8>) -> Result<Option<f64>, VideoError>;
}

/// A consumer of filtered frames (an encoder, a file sequence, a preview).
///
/// [`process_video`](super::process_video) calls `begin` once, `write_frame` for every frame in
/// order, then `finish` after the last one — or, once `begin` succeeded, `abort` instead when
/// reading, filtering or writing a frame fails or processing is cancelled. A failing `begin` or
/// `finish` is not followed by `abort`: they clean up after themselves. The trait is object-safe (`&mut dyn VideoSink`).
///
/// Wrapping an encoder library: with ffmpeg-next, set up the output context and encoder in
/// `begin` (size and `frame_rate` from `info`, time base `1 / frame_rate`), convert each RGBA
/// frame to the encoder's pixel format with a scaling context in `write_frame`, set its pts to
/// the frame index, send it and write the received packets; in `finish` flush the encoder (send
/// EOF, drain packets) and write the trailer. With `GStreamer`, push buffers into an `appsrc`
/// (caps `video/x-raw,format=RGBA`, with `framerate`) whose PTS is `time`, and send EOS in
/// `finish`.
pub trait VideoSink {
    /// Prepares for a stream of `info` frames (spawns an encoder, creates a directory, ...).
    ///
    /// # Errors
    /// Whatever the sink hits; custom sinks wrap their errors in [`VideoError::Custom`].
    fn begin(&mut self, info: &VideoInfo) -> Result<(), VideoError>;

    /// Writes the next frame. `frame` is straight-alpha RGBA8 of the stream's size; `time` is
    /// its presentation time in seconds from the first frame.
    ///
    /// # Errors
    /// As [`VideoSink::begin`].
    fn write_frame(&mut self, frame: &FrameRef<'_>, time: f64) -> Result<(), VideoError>;

    /// Completes the output after the last frame (flushes, finalises the container).
    ///
    /// # Errors
    /// As [`VideoSink::begin`].
    fn finish(&mut self) -> Result<(), VideoError>;

    /// Stops after a failure or a cancellation that followed a successful `begin`, releasing
    /// resources without completing the output (whatever was already written may remain,
    /// depending on the sink). The default does nothing.
    fn abort(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_rates_parse_reduce_and_reject_zero() {
        let ntsc: FrameRate = "30000/1001".parse().unwrap_or_else(|error| panic!("{error}"));
        assert_eq!((ntsc.numerator(), ntsc.denominator()), (30000, 1001));
        assert_eq!("25".parse::<FrameRate>().ok(), FrameRate::new(25, 1).ok());
        assert_eq!(FrameRate::new(50, 2).ok(), FrameRate::new(25, 1).ok(), "reduced");
        assert_eq!(FrameRate::new(24, 1).map(|rate| rate.to_string()).ok().as_deref(), Some("24/1"));
        for bad in ["0/0", "25/0", "0/1", "", "abc", "-25/1", "25/x", "1.5"] {
            assert!(matches!(bad.parse::<FrameRate>(), Err(VideoError::InvalidFrameRate { .. })), "{bad}");
        }
    }

    #[test]
    fn frame_times_are_exact_fractions() {
        let ntsc = FrameRate::new(30000, 1001).unwrap_or_else(|error| panic!("{error}"));
        assert!(ntsc.frame_time(0).abs() < f64::EPSILON);
        assert!((ntsc.frame_time(30) - 1.001).abs() < 1e-12);
        // One hour of NTSC frames: 107892 frames = 3599.9964 s, no drift.
        assert!((ntsc.frame_time(107_892) - 107_892.0 * 1001.0 / 30000.0).abs() < 1e-9);
        let pal = FrameRate::new(25, 1).unwrap_or_else(|error| panic!("{error}"));
        assert!((pal.frame_time(3) - 0.12).abs() < 1e-12);
        assert!(pal.frame_time(u64::MAX).is_finite());
        assert_eq!(pal.frames_in(2.0), Some(50));
        assert_eq!(ntsc.frames_in(1.001), Some(30));
        assert_eq!(pal.frames_in(-1.0), None);
        assert_eq!(pal.frames_in(f64::NAN), None);
        assert_eq!(pal.frames_in(f64::MAX), None);
    }

    #[test]
    fn video_info_frame_bytes_are_checked() {
        let rate = FrameRate::new(25, 1).unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(VideoInfo::new(4, 3, rate).frame_bytes().ok(), Some(48));
        assert!(VideoInfo::new(u32::MAX, u32::MAX, rate).frame_bytes().is_err());
    }

    #[test]
    fn sample_aspect_ratios_are_reduced_and_zero_means_square() {
        let rate = FrameRate::new(25, 1).unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(VideoInfo::new(4, 3, rate).sample_aspect_ratio(), (1, 1));
        assert_eq!(VideoInfo::new(4, 3, rate).with_sample_aspect_ratio(64, 48).sample_aspect_ratio(), (4, 3));
        assert_eq!(VideoInfo::new(4, 3, rate).with_sample_aspect_ratio(0, 1).sample_aspect_ratio(), (1, 1));
        assert_eq!(VideoInfo::new(4, 3, rate).with_sample_aspect_ratio(3, 0).sample_aspect_ratio(), (1, 1));
    }
}
