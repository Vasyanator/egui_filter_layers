/*
File: offline/video/ffmpeg/probe.rs

Purpose:
Reads a video file's properties with `ffprobe`: the first real video stream's (attached pictures
such as cover art are skipped) index, codec, displayed size (after the rotation ffmpeg applies
when decoding), sample aspect ratio, frame rate, duration, colour tags, whether it has alpha (in
its pixel format or, for VP8/VP9, in the `alpha_mode` side channel), and whether the file has
audio.

Key structures:
- `Probe`: the parsed properties.

Key functions:
- `probe()`: runs ffprobe (`-of flat`) and parses its output.
- `parse_flat()`: the pure parser (unit-tested on captured ffprobe output).

Notes:
Alpha and RGB coding come from ffprobe's own pixel-format table (`-show_pixel_formats`,
`flags.alpha` / `flags.rgb`), not from the format's name. The stream is later decoded by its
absolute index (`-map 0:<index>`). Frame rate: `avg_frame_rate`, else `r_frame_rate` when it is
plausible (at most 240 fps: above that it is usually a timebase, e.g. `90000/1` of a still image);
`0/0` means unknown.
*/

use std::collections::HashMap;
use std::ffi::OsString;
use std::path::Path;

use super::super::error::VideoError;
use super::super::stream::FrameRate;
use super::paths;
use super::process::run_to_string;

/// Side-data entries of a stream searched for a rotation.
const MAX_SIDE_DATA: usize = 16;
/// Highest `r_frame_rate` accepted as a fallback frame rate, frames per second.
const MAX_FALLBACK_RATE: u32 = 240;

/// Properties of the first real video stream of a file.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Probe {
    /// Absolute index of the stream in the file (`-map 0:<index>`).
    pub stream_index: usize,
    /// The stream's codec name (`h264`, `vp9`, ...).
    pub codec_name: String,
    /// Width of decoded frames (rotation applied).
    pub width: u32,
    /// Height of decoded frames (rotation applied).
    pub height: u32,
    /// Average frame rate, when ffprobe knows it.
    pub frame_rate: Option<FrameRate>,
    /// Why no frame rate was taken, when one was reported but rejected.
    pub frame_rate_note: Option<String>,
    /// Sample aspect ratio, when known and not square.
    pub sample_aspect_ratio: Option<(u32, u32)>,
    /// Length in seconds (container, else stream), when known.
    pub duration: Option<f64>,
    /// The stream's pixel format name.
    pub pixel_format: String,
    /// Where the stream's alpha is, if it has any.
    pub alpha: StreamAlpha,
    /// Whether the pixel format is RGB-coded (no YCbCr matrix).
    pub is_rgb: bool,
    /// `color_space` tag (`bt709`, `unknown`, ...).
    pub color_space: String,
    /// `color_range` tag (`tv`, `pc`, `unknown`).
    pub color_range: String,
    /// `color_transfer` tag.
    pub color_transfer: String,
    /// `color_primaries` tag.
    pub color_primaries: String,
    /// Whether the file has an audio stream.
    pub has_audio: bool,
}

/// Where a video stream stores alpha.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum StreamAlpha {
    /// No alpha.
    #[default]
    None,
    /// In the pixel format (`yuva420p`, `rgba`, ...).
    PixelFormat,
    /// In a VP8/VP9 side channel (`alpha_mode=1`), which only the libvpx decoders read.
    SideChannel,
}

impl Probe {
    /// Whether the stream has alpha.
    pub(crate) fn has_alpha(&self) -> bool {
        self.alpha != StreamAlpha::None
    }

    /// The decoder to force for this stream (before `-i`): the libvpx decoders for VP8/VP9 with
    /// an alpha side channel, since ffmpeg's native ones drop it.
    pub(crate) fn alpha_decoder(&self) -> Option<&'static str> {
        match (self.alpha, self.codec_name.as_str()) {
            (StreamAlpha::SideChannel, "vp9") => Some("libvpx-vp9"),
            (StreamAlpha::SideChannel, "vp8") => Some("libvpx"),
            _ => None,
        }
    }
}

/// Runs `ffprobe` on `input`.
///
/// # Errors
/// [`VideoError::ToolNotFound`] / [`VideoError::ProcessFailed`] (e.g. an unreadable file),
/// [`VideoError::NoVideoStream`], [`VideoError::MalformedProbe`].
pub(crate) fn probe(ffprobe: &Path, input: &Path) -> Result<Probe, VideoError> {
    let args: Vec<OsString> = [
        "-v",
        "error",
        "-show_entries",
        "stream=index,codec_type,codec_name,width,height,pix_fmt,sample_aspect_ratio,avg_frame_rate,r_frame_rate,duration,color_space,color_range,color_transfer,color_primaries:stream_disposition=attached_pic:stream_tags=alpha_mode:stream_side_data=rotation:format=duration",
        "-show_pixel_formats",
        "-of",
        "flat",
        "-i",
    ]
    .into_iter()
    .map(OsString::from)
    .chain(std::iter::once(paths::url(input)))
    .collect();
    let text = run_to_string(ffprobe, &args)?;
    parse_flat(&text).map_err(|failure| match failure {
        ProbeFailure::NoVideo => VideoError::NoVideoStream { input: input.to_owned() },
        ProbeFailure::Malformed(reason) => VideoError::MalformedProbe { input: input.to_owned(), reason },
    })
}

/// Why ffprobe's output could not be used.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum ProbeFailure {
    /// No stream has `codec_type="video"` (attached pictures do not count).
    NoVideo,
    /// A required value is missing or invalid.
    Malformed(String),
}

/// Parses ffprobe's `-of flat` output (`section.index.key=value` lines, strings quoted).
///
/// # Errors
/// [`ProbeFailure::NoVideo`] without a video stream; [`ProbeFailure::Malformed`] when the video
/// stream's size is missing or invalid.
pub(crate) fn parse_flat(text: &str) -> Result<Probe, ProbeFailure> {
    let values: HashMap<&str, String> = text.lines().filter_map(|line| line.split_once('=')).map(|(key, value)| (key.trim(), unquote(value.trim()))).collect();
    let get = |key: &str| values.get(key).map(String::as_str);

    let stream_count = (0..usize::MAX).take_while(|index| get(&format!("streams.stream.{index}.codec_type")).is_some()).count();
    let codec_type = |index: usize| get(&format!("streams.stream.{index}.codec_type"));
    let attached_picture = |index: usize| get(&format!("streams.stream.{index}.disposition.attached_pic")) == Some("1");
    let video = (0..stream_count).find(|&index| codec_type(index) == Some("video") && !attached_picture(index)).ok_or(ProbeFailure::NoVideo)?;
    let has_audio = (0..stream_count).any(|index| codec_type(index) == Some("audio"));
    let field = |name: &str| get(&format!("streams.stream.{video}.{name}"));
    let stream_index = match field("index") {
        Some(value) => value.parse::<usize>().map_err(|_| ProbeFailure::Malformed(format!("the video stream has an invalid index {value:?}")))?,
        None => video,
    };
    let text_field = |name: &str| field(name).unwrap_or("").to_owned();

    let dimension = |name: &str| {
        field(name).and_then(|value| value.parse::<u32>().ok()).filter(|&value| value > 0).ok_or_else(|| ProbeFailure::Malformed(format!("the video stream has no valid {name} (got {:?})", field(name))))
    };
    let (mut width, mut height) = (dimension("width")?, dimension("height")?);
    // ffmpeg rotates frames by the display matrix when decoding (autorotate): a quarter turn
    // swaps the decoded frame's sides.
    // Side data entries of other types come first in some files; a stream has only a handful.
    let rotation = (0..MAX_SIDE_DATA).find_map(|index| field(&format!("side_data_list.side_data.{index}.rotation"))?.parse::<f64>().ok());
    if rotation.is_some_and(|degrees| (degrees.round().rem_euclid(180.0) - 90.0).abs() < 0.5) {
        (width, height) = (height, width);
    }

    let parse_rate = |name: &str| field(name).and_then(|value| value.parse::<FrameRate>().ok());
    let (frame_rate, frame_rate_note) = match (parse_rate("avg_frame_rate"), parse_rate("r_frame_rate")) {
        (Some(rate), _) => (Some(rate), None),
        // `r_frame_rate` is a guess, often the timebase for streams without a regular rate.
        (None, Some(rate)) if rate.numerator() > rate.denominator().saturating_mul(MAX_FALLBACK_RATE) => (None, Some(format!("ffprobe only guesses {rate} fps, which is implausible"))),
        (None, fallback) => (fallback, None),
    };
    let sample_aspect_ratio = field("sample_aspect_ratio")
        .and_then(|value| value.split_once(':'))
        .and_then(|(numerator, denominator)| Some((numerator.parse::<u32>().ok()?, denominator.parse::<u32>().ok()?)))
        .filter(|&(numerator, denominator)| numerator != 0 && denominator != 0 && numerator != denominator);
    let seconds = |value: Option<&str>| value.and_then(|value| value.parse::<f64>().ok()).filter(|seconds| seconds.is_finite() && *seconds > 0.0);
    let duration = seconds(get("format.duration")).or_else(|| seconds(field("duration")));

    let pixel_format = field("pix_fmt").unwrap_or("").to_owned();
    let format_index = (0..usize::MAX).map_while(|index| get(&format!("pixel_formats.pixel_format.{index}.name")).map(|name| (index, name))).find(|(_, name)| *name == pixel_format).map(|(index, _)| index);
    let format_flag = |flag: &str| format_index.is_some_and(|index| get(&format!("pixel_formats.pixel_format.{index}.flags.{flag}")) == Some("1"));
    let alpha = if format_flag("alpha") {
        StreamAlpha::PixelFormat
    } else if field("tags.alpha_mode") == Some("1") {
        StreamAlpha::SideChannel
    } else {
        StreamAlpha::None
    };

    Ok(Probe {
        stream_index,
        codec_name: text_field("codec_name"),
        width,
        height,
        frame_rate,
        frame_rate_note,
        sample_aspect_ratio,
        duration,
        pixel_format,
        alpha,
        is_rgb: format_flag("rgb"),
        color_space: text_field("color_space"),
        color_range: text_field("color_range"),
        color_transfer: text_field("color_transfer"),
        color_primaries: text_field("color_primaries"),
        has_audio,
    })
}

/// Removes the quotes and backslash escapes of a `flat` string value; other values unchanged.
fn unquote(value: &str) -> String {
    let Some(inner) = value.strip_prefix('"').and_then(|rest| rest.strip_suffix('"')) else {
        return value.to_owned();
    };
    let mut out = String::with_capacity(inner.len());
    let mut chars = inner.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            if let Some(escaped) = chars.next() {
                out.push(escaped);
            }
        } else {
            out.push(c);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ffprobe 8.0 output for a rotated H.264 + AAC file (pixel-format table shortened).
    const ROTATED: &str = r#"streams.stream.0.index=0
streams.stream.0.codec_type="video"
streams.stream.0.width=64
streams.stream.0.height=48
streams.stream.0.pix_fmt="yuv420p"
streams.stream.0.r_frame_rate="25/1"
streams.stream.0.avg_frame_rate="25/1"
streams.stream.0.duration="1.000000"
streams.stream.0.side_data_list.side_data.0.rotation=90
streams.stream.1.index=1
streams.stream.1.codec_type="audio"
streams.stream.1.r_frame_rate="0/0"
streams.stream.1.avg_frame_rate="0/0"
streams.stream.1.duration="1.000000"
format.duration="1.023000"
pixel_formats.pixel_format.0.name="yuv420p"
pixel_formats.pixel_format.0.nb_components=3
pixel_formats.pixel_format.0.flags.alpha=0
pixel_formats.pixel_format.1.name="yuva420p"
pixel_formats.pixel_format.1.flags.alpha=1
"#;

    #[test]
    fn a_rotated_stream_swaps_its_sides() {
        let probe = parse_flat(ROTATED).unwrap_or_else(|error| panic!("{error:?}"));
        assert_eq!((probe.width, probe.height), (48, 64));
        assert_eq!(probe.frame_rate.map(|rate| rate.to_string()).as_deref(), Some("25/1"));
        assert_eq!(probe.duration, Some(1.023), "the container's duration wins");
        assert_eq!(probe.pixel_format, "yuv420p");
        assert!(!probe.has_alpha());
        assert!(probe.has_audio);
    }

    #[test]
    fn alpha_comes_from_the_pixel_format_table() {
        let text = ROTATED.replace("streams.stream.0.pix_fmt=\"yuv420p\"", "streams.stream.0.pix_fmt=\"yuva420p\"").replace("rotation=90", "rotation=-180");
        let probe = parse_flat(&text).unwrap_or_else(|error| panic!("{error:?}"));
        assert!(probe.has_alpha());
        assert_eq!((probe.width, probe.height), (64, 48), "a half turn keeps the sides");
    }

    #[test]
    fn frame_rate_falls_back_and_may_be_unknown() {
        let text = ROTATED.replace("streams.stream.0.avg_frame_rate=\"25/1\"", "streams.stream.0.avg_frame_rate=\"0/0\"").replace("streams.stream.0.r_frame_rate=\"25/1\"", "streams.stream.0.r_frame_rate=\"30000/1001\"");
        assert_eq!(parse_flat(&text).ok().and_then(|probe| probe.frame_rate).map(|rate| rate.to_string()).as_deref(), Some("30000/1001"));
        let unknown = text.replace("\"30000/1001\"", "\"0/0\"").replace("format.duration=\"1.023000\"", "format.duration=\"N/A\"");
        let probe = parse_flat(&unknown).unwrap_or_else(|error| panic!("{error:?}"));
        assert_eq!(probe.frame_rate, None);
        assert_eq!(probe.duration, Some(1.0), "falls back to the stream's duration");
    }

    #[test]
    fn missing_video_or_size_is_reported() {
        let audio_only = "streams.stream.0.codec_type=\"audio\"\nformat.duration=\"3.0\"\n";
        assert_eq!(parse_flat(audio_only), Err(ProbeFailure::NoVideo));
        assert_eq!(parse_flat(""), Err(ProbeFailure::NoVideo));
        let no_width = ROTATED.replace("streams.stream.0.width=64\n", "");
        assert!(matches!(parse_flat(&no_width), Err(ProbeFailure::Malformed(reason)) if reason.contains("width")));
        let zero = ROTATED.replace("height=48", "height=0");
        assert!(matches!(parse_flat(&zero), Err(ProbeFailure::Malformed(_))));
    }

    #[test]
    fn the_first_video_stream_is_chosen() {
        let text = "streams.stream.0.codec_type=\"audio\"\nstreams.stream.1.codec_type=\"video\"\nstreams.stream.1.width=10\nstreams.stream.1.height=20\nstreams.stream.1.avg_frame_rate=\"24/1\"\nstreams.stream.2.codec_type=\"video\"\nstreams.stream.2.width=99\nstreams.stream.2.height=99\n";
        let probe = parse_flat(text).unwrap_or_else(|error| panic!("{error:?}"));
        assert_eq!((probe.width, probe.height), (10, 20));
        assert!(probe.has_audio);
        assert_eq!(probe.duration, None);
        assert_eq!(probe.stream_index, 1, "without an `index` field the position is the index");
    }

    /// ffprobe 8.0 output for an audio file whose cover art (an attached picture) precedes a
    /// real, anamorphic, BT.709-tagged video stream.
    const COVER_FIRST: &str = r#"streams.stream.0.index=0
streams.stream.0.codec_name="aac"
streams.stream.0.codec_type="audio"
streams.stream.0.disposition.attached_pic=0
streams.stream.1.index=1
streams.stream.1.codec_name="mjpeg"
streams.stream.1.codec_type="video"
streams.stream.1.width=32
streams.stream.1.height=32
streams.stream.1.pix_fmt="yuvj420p"
streams.stream.1.r_frame_rate="90000/1"
streams.stream.1.avg_frame_rate="0/0"
streams.stream.1.disposition.attached_pic=1
streams.stream.2.index=2
streams.stream.2.codec_name="h264"
streams.stream.2.codec_type="video"
streams.stream.2.width=32
streams.stream.2.height=24
streams.stream.2.sample_aspect_ratio="4:3"
streams.stream.2.pix_fmt="yuv420p"
streams.stream.2.color_range="tv"
streams.stream.2.color_space="bt709"
streams.stream.2.color_transfer="bt709"
streams.stream.2.color_primaries="bt709"
streams.stream.2.r_frame_rate="25/1"
streams.stream.2.avg_frame_rate="25/1"
streams.stream.2.disposition.attached_pic=0
format.duration="0.200000"
"#;

    #[test]
    fn attached_pictures_are_skipped_and_the_real_stream_is_described() {
        let probe = parse_flat(COVER_FIRST).unwrap_or_else(|error| panic!("{error:?}"));
        assert_eq!((probe.stream_index, probe.codec_name.as_str(), probe.width, probe.height), (2, "h264", 32, 24));
        assert_eq!(probe.sample_aspect_ratio, Some((4, 3)));
        assert_eq!((probe.color_space.as_str(), probe.color_range.as_str(), probe.color_transfer.as_str(), probe.color_primaries.as_str()), ("bt709", "tv", "bt709", "bt709"));
        assert!(!probe.is_rgb && !probe.has_alpha() && probe.alpha_decoder().is_none());
        // Only the cover: no video.
        let cover_only = COVER_FIRST.split("streams.stream.2.").next().unwrap_or_default();
        assert_eq!(parse_flat(cover_only), Err(ProbeFailure::NoVideo));
        let square = COVER_FIRST.replace("\"4:3\"", "\"1:1\"");
        assert_eq!(parse_flat(&square).ok().and_then(|probe| probe.sample_aspect_ratio), None);
    }

    #[test]
    fn implausible_fallback_rates_are_rejected() {
        let text = COVER_FIRST.replace("streams.stream.1.disposition.attached_pic=1", "streams.stream.1.disposition.attached_pic=0");
        let probe = parse_flat(&text).unwrap_or_else(|error| panic!("{error:?}"));
        assert_eq!(probe.stream_index, 1);
        assert_eq!(probe.frame_rate, None);
        assert!(probe.frame_rate_note.as_deref().is_some_and(|note| note.contains("90000/1")), "{probe:?}");
    }

    #[test]
    fn vp9_alpha_side_channels_need_libvpx() {
        let text = "streams.stream.0.index=0\nstreams.stream.0.codec_name=\"vp9\"\nstreams.stream.0.codec_type=\"video\"\nstreams.stream.0.width=32\nstreams.stream.0.height=32\nstreams.stream.0.pix_fmt=\"yuv420p\"\nstreams.stream.0.avg_frame_rate=\"25/1\"\nstreams.stream.0.tags.alpha_mode=\"1\"\npixel_formats.pixel_format.0.name=\"yuv420p\"\npixel_formats.pixel_format.0.flags.alpha=0\npixel_formats.pixel_format.0.flags.rgb=0\npixel_formats.pixel_format.1.name=\"gbrp\"\npixel_formats.pixel_format.1.flags.rgb=1\n";
        let probe = parse_flat(text).unwrap_or_else(|error| panic!("{error:?}"));
        assert_eq!(probe.alpha, StreamAlpha::SideChannel);
        assert_eq!(probe.alpha_decoder(), Some("libvpx-vp9"));
        let vp8 = parse_flat(&text.replace("\"vp9\"", "\"vp8\"")).ok().and_then(|probe| probe.alpha_decoder());
        assert_eq!(vp8, Some("libvpx"));
        let rgb = parse_flat(&text.replace("pix_fmt=\"yuv420p\"", "pix_fmt=\"gbrp\"").replace("alpha_mode=\"1\"", "alpha_mode=\"0\"")).unwrap_or_else(|error| panic!("{error:?}"));
        assert!(rgb.is_rgb && !rgb.has_alpha() && rgb.alpha_decoder().is_none());
    }

    #[test]
    fn quoted_values_are_unescaped() {
        assert_eq!(unquote(r#""a \"b\" \\c""#), r#"a "b" \c"#);
        assert_eq!(unquote("42"), "42");
        assert_eq!(unquote("\""), "\"");
    }
}
