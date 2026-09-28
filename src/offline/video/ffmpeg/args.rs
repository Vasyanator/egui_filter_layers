/*
File: offline/video/ffmpeg/args.rs

Purpose:
Command lines of the ffmpeg adapters: output codec presets (`VideoCodec`), audio handling
(`AudioMode`), and the pure argument builders for decoding to raw RGBA on stdout and encoding raw
RGBA from stdin (unit-tested; no process is started here). Each output gets exactly one video
filter chain: sample aspect ratio, then the preset's or the custom arguments' own filter, then
the colour conversion and tags of `color.rs`.

Key structures:
- `VideoCodec` (public): H.264, VP9, AV1, ProRes 4444, FFV1, GIF presets, or custom arguments.
- `AudioMode` (public): copy the input's audio, re-encode it, or drop it.

Key functions:
- `decode_args()`: forced alpha decoder, `-map 0:<index>`, constant frame rate
  (`-fps_mode cfr -r N/D`), colour conversion filter, `rawvideo rgba`.
- `encode_args()`: rawvideo input at the stream's size and rate, optional audio input
  (`-map 1:a?`), the merged filter chain, colour tags, codec arguments, output path.
- `VideoCodec::output_args()`: the output arguments of a preset for a sample aspect ratio.

Notes:
`-fps_mode` needs ffmpeg 5.1 or newer (checked by `version.rs`). Every path is passed as a
`file:` URL (`paths.rs`), so no file name reads as an option, a protocol or stdin/stdout.
*/

use std::ffi::OsString;
use std::path::Path;

use super::super::stream::FrameRate;
use super::color::OutputColor;
use super::paths;

/// Output options that set the video filter chain; custom arguments' ones are merged.
const VIDEO_FILTER_OPTIONS: [&str; 3] = ["-vf", "-filter:v", "-filter:v:0"];

/// Output video codec and its encoder settings.
///
/// Every preset names an ffmpeg encoder; whether it is available (and its licence) depends on the
/// ffmpeg build installed. Codecs without alpha need an opaque result: frames that are not
/// opaque are rejected with [`VideoError::AlphaNotSupported`](super::super::VideoError::AlphaNotSupported)
/// unless a background colour is set ([`OfflineOptions::with_background`](crate::OfflineOptions::with_background)).
///
/// Colour: YCbCr outputs (H.264, VP9, AV1, `ProRes`, custom with `yuv`) are converted with the
/// BT.709 matrix to limited range and tagged BT.709 (matrix, primaries, transfer) and `tv`
/// range, so players show the colours the filters produced; RGB outputs (FFV1, custom without
/// `yuv`) are tagged sRGB, full range; GIF has no tags.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum VideoCodec {
    /// H.264 (`libx264`, 4:2:0, `-preset medium`), quality `crf` (0 = lossless, 18 ≈ visually
    /// lossless, 23 = x264's default). No alpha; even sizes only. `.mp4`, `.mkv`, `.mov`.
    H264 {
        /// Constant rate factor, 0..=51.
        crf: u8,
    },
    /// VP9 (`libvpx-vp9`, `yuva420p`, constant quality `crf` 0..=63) with alpha. `.webm`
    /// (audio must be Opus or Vorbis there: re-encode it with [`AudioMode::Encode`]) or `.mkv`.
    Vp9 {
        /// Constant quality, 0..=63.
        crf: u8,
    },
    /// AV1 (`libsvtav1`, 4:2:0, `crf` 0..=63). No alpha; even sizes only (and at least 64x64 for
    /// SVT-AV1). `.mp4`, `.mkv`, `.webm`.
    Av1 {
        /// Constant rate factor, 0..=63.
        crf: u8,
    },
    /// Apple `ProRes` 4444 (`prores_ks`, 10-bit 4:4:4 with alpha). `.mov`.
    ProRes4444,
    /// FFV1, lossless RGB with alpha (`bgra`). `.mkv`.
    Ffv1,
    /// Animated GIF (one palette generated from the whole clip). No alpha, no audio. `.gif`.
    ///
    /// Memory: the palette is known only at the end, so ffmpeg keeps **every frame** (RGBA) in
    /// memory until then — about `width * height * 4 * frames` bytes (1080p at 30 fps for 60 s
    /// is roughly 15 GB). Keep GIFs short and small, or use `Custom` with a per-frame palette
    /// (`-vf split[a][b];[a]palettegen=stats_mode=single[p];[b][p]paletteuse=new=1`).
    Gif,
    /// Any encoder: `args` are inserted between the inputs and the output path (e.g.
    /// `["-c:v", "libx265", "-crf", "20", "-pix_fmt", "yuv420p"]`).
    ///
    /// A video filter in `args` (`-vf`, `-filter:v`, `-filter:v:0`; the last one, as ffmpeg
    /// does) is merged into the one chain the sink builds, before the colour conversion; its
    /// last chain must produce the output. `-filter_complex` cannot be combined with it.
    Custom {
        /// Output arguments for the video stream.
        args: Vec<String>,
        /// Whether the encoded format keeps transparency.
        alpha: bool,
        /// Whether the encoder stores YCbCr (then the frames are converted with the BT.709
        /// matrix to limited range and tagged so) rather than RGB (tagged sRGB, full range).
        yuv: bool,
    },
}

impl Default for VideoCodec {
    /// H.264 at `crf` 18.
    fn default() -> Self {
        Self::H264 { crf: 18 }
    }
}

impl VideoCodec {
    /// The output arguments for square pixels: the merged video filter chain (colour
    /// conversion included), the colour tags and the encoder's options.
    #[must_use]
    pub fn args(&self) -> Vec<String> {
        self.output_args((1, 1))
    }

    /// [`VideoCodec::args`] for frames of the sample aspect ratio `(numerator, denominator)`.
    #[must_use]
    pub fn output_args(&self, sample_aspect_ratio: (u32, u32)) -> Vec<String> {
        let (own_filter, encoder) = self.encoder_args();
        let mut chain: Vec<String> = Vec::new();
        if sample_aspect_ratio != (1, 1) && sample_aspect_ratio.0 != 0 && sample_aspect_ratio.1 != 0 {
            chain.push(format!("setsar={}/{}", sample_aspect_ratio.0, sample_aspect_ratio.1));
        }
        chain.extend(own_filter);
        let color = self.output_color();
        chain.extend(color.filter().map(str::to_owned));
        let mut args = Vec::new();
        if !chain.is_empty() {
            args.extend(["-filter:v".to_owned(), chain.join(",")]);
        }
        args.extend(color.tags().iter().map(|&tag| tag.to_owned()));
        args.extend(encoder);
        args
    }

    /// The preset's (or custom arguments') own video filter, and the other output arguments.
    fn encoder_args(&self) -> (Option<String>, Vec<String>) {
        let strings = |args: &[&str]| args.iter().map(|&arg| arg.to_owned()).collect::<Vec<_>>();
        let encoder = match self {
            Self::H264 { crf } => [strings(&["-c:v", "libx264", "-preset", "medium", "-crf"]), vec![crf.to_string()], strings(&["-pix_fmt", "yuv420p"])].concat(),
            Self::Vp9 { crf } => [strings(&["-c:v", "libvpx-vp9", "-crf"]), vec![crf.to_string()], strings(&["-b:v", "0", "-row-mt", "1", "-pix_fmt", "yuva420p"])].concat(),
            Self::Av1 { crf } => [strings(&["-c:v", "libsvtav1", "-crf"]), vec![crf.to_string()], strings(&["-pix_fmt", "yuv420p"])].concat(),
            Self::ProRes4444 => strings(&["-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le"]),
            Self::Ffv1 => strings(&["-c:v", "ffv1", "-level", "3", "-pix_fmt", "bgra"]),
            Self::Gif => return (Some("split[a][b];[a]palettegen[p];[b][p]paletteuse".to_owned()), strings(&["-c:v", "gif"])),
            Self::Custom { args, .. } => {
                // Pull the video filter options out; the last one wins, as in ffmpeg.
                let (mut filter, mut rest) = (None, Vec::with_capacity(args.len()));
                let mut iter = args.iter();
                while let Some(arg) = iter.next() {
                    if VIDEO_FILTER_OPTIONS.contains(&arg.as_str())
                        && let Some(value) = iter.next()
                    {
                        filter = Some(value.clone());
                        continue;
                    }
                    rest.push(arg.clone());
                }
                return (filter, rest);
            }
        };
        (None, encoder)
    }

    /// How the output stores colour.
    fn output_color(&self) -> OutputColor {
        match self {
            Self::H264 { .. } | Self::Vp9 { .. } | Self::Av1 { .. } | Self::ProRes4444 | Self::Custom { yuv: true, .. } => OutputColor::Yuv,
            Self::Ffv1 | Self::Custom { yuv: false, .. } => OutputColor::Rgb,
            Self::Gif => OutputColor::Palette,
        }
    }

    /// Whether the output keeps transparency.
    #[must_use]
    pub fn supports_alpha(&self) -> bool {
        match self {
            Self::Vp9 { .. } | Self::ProRes4444 | Self::Ffv1 => true,
            Self::H264 { .. } | Self::Av1 { .. } | Self::Gif => false,
            Self::Custom { alpha, .. } => *alpha,
        }
    }

    /// Whether the format can carry audio.
    #[must_use]
    pub fn supports_audio(&self) -> bool {
        !matches!(self, Self::Gif)
    }

    /// Whether frame sides must be even (4:2:0 chroma subsampling).
    #[must_use]
    pub fn needs_even_size(&self) -> bool {
        matches!(self, Self::H264 { .. } | Self::Vp9 { .. } | Self::Av1 { .. })
    }

    /// A short name for messages.
    #[must_use]
    pub fn name(&self) -> String {
        match self {
            Self::H264 { .. } => "H.264 (libx264)".to_owned(),
            Self::Vp9 { .. } => "VP9 (libvpx-vp9)".to_owned(),
            Self::Av1 { .. } => "AV1 (libsvtav1)".to_owned(),
            Self::ProRes4444 => "ProRes 4444 (prores_ks)".to_owned(),
            Self::Ffv1 => "FFV1".to_owned(),
            Self::Gif => "GIF".to_owned(),
            Self::Custom { args, .. } => format!("custom encoder ({})", args.join(" ")),
        }
    }
}

/// What happens to the input's audio.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum AudioMode {
    /// Copy the audio streams unchanged (`-map 1:a? -c:a copy`); a file without audio just gets
    /// none. The container must accept the input's audio codec (`WebM` does not take AAC: use
    /// `Encode` there). Formats without audio (GIF) get none.
    #[default]
    Copy,
    /// Re-encode the audio with these output arguments, e.g. `["-c:a", "libopus", "-b:a", "128k"]`.
    Encode(Vec<String>),
    /// No audio in the output.
    Drop,
}

/// Everything a decoder command line depends on.
#[derive(Debug)]
pub(crate) struct DecodeCommand<'a> {
    /// The file read.
    pub input: &'a Path,
    /// Absolute index of the video stream.
    pub stream_index: usize,
    /// Constant output frame rate.
    pub rate: FrameRate,
    /// Decoder forced before `-i` (VP8/VP9 alpha), if any.
    pub decoder: Option<&'a str>,
    /// Filter chain converting the decoded frames to RGBA (`DecodeColor::filter`).
    pub filter: &'a str,
}

/// Arguments decoding stream `command.stream_index` of the input to straight-alpha RGBA8 frames
/// on stdout at the constant rate (VFR input resampled; see the module docs).
pub(crate) fn decode_args(command: &DecodeCommand<'_>) -> Vec<OsString> {
    let mut args: Vec<OsString> = ["-nostdin", "-hide_banner", "-loglevel", "error"].into_iter().map(OsString::from).collect();
    if let Some(decoder) = command.decoder {
        args.extend(["-c:v", decoder].into_iter().map(OsString::from));
    }
    args.push(OsString::from("-i"));
    args.push(paths::url(command.input));
    let (map, rate) = (format!("0:{}", command.stream_index), command.rate.to_string());
    args.extend(["-map", map.as_str(), "-an", "-sn", "-dn", "-fps_mode", "cfr", "-r", rate.as_str(), "-filter:v", command.filter, "-f", "rawvideo", "-pix_fmt", "rgba", "pipe:1"].into_iter().map(OsString::from));
    args
}

/// Everything an encoder command line depends on.
#[derive(Debug)]
pub(crate) struct EncodeCommand<'a> {
    /// Frame width.
    pub width: u32,
    /// Frame height.
    pub height: u32,
    /// Constant frame rate of the output.
    pub rate: FrameRate,
    /// Sample aspect ratio of the frames.
    pub sample_aspect_ratio: (u32, u32),
    /// Codec preset.
    pub codec: &'a VideoCodec,
    /// Where the audio comes from, and how it is treated.
    pub audio: Option<(&'a Path, &'a AudioMode)>,
    /// Output file.
    pub output: &'a Path,
}

/// Arguments encoding raw RGBA8 frames from stdin into `command.output` (overwritten).
pub(crate) fn encode_args(command: &EncodeCommand<'_>) -> Vec<OsString> {
    let mut args: Vec<OsString> = Vec::new();
    let mut push = |values: &[&str]| args.extend(values.iter().map(OsString::from));
    let size = format!("{}x{}", command.width, command.height);
    let rate = command.rate.to_string();
    push(&["-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", size.as_str(), "-framerate", rate.as_str(), "-i", "pipe:0"]);
    // The audio input and its codec arguments, unless dropped or unsupported by the format.
    let audio = match command.audio {
        Some((input, mode)) if command.codec.supports_audio() => match mode {
            AudioMode::Copy => Some((input, vec!["-c:a".to_owned(), "copy".to_owned()])),
            AudioMode::Encode(encode) => Some((input, encode.clone())),
            AudioMode::Drop => None,
        },
        Some(_) | None => None,
    };
    if let Some((input, _)) = &audio {
        args.extend([OsString::from("-i"), paths::url(input)]);
    }
    args.extend(["-map", "0:v:0"].into_iter().map(OsString::from));
    args.extend(command.codec.output_args(command.sample_aspect_ratio).into_iter().map(OsString::from));
    match audio {
        Some((_, codec)) => {
            args.extend(["-map", "1:a?"].into_iter().map(OsString::from));
            args.extend(codec.into_iter().map(OsString::from));
            // The audio of the whole input would outlast a video that stopped early.
            args.push(OsString::from("-shortest"));
        }
        None => args.push(OsString::from("-an")),
    }
    args.push(paths::url(command.output));
    args
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(args: &[OsString]) -> Vec<String> {
        args.iter().map(|arg| arg.to_string_lossy().into_owned()).collect()
    }

    fn rate(numerator: u32, denominator: u32) -> FrameRate {
        FrameRate::new(numerator, denominator).unwrap_or_else(|error| panic!("{error}"))
    }

    #[test]
    fn decoding_maps_the_probed_stream_at_a_constant_rate() {
        let command = DecodeCommand { input: Path::new("in put.mov"), stream_index: 2, rate: rate(30000, 1001), decoder: None, filter: "scale=in_color_matrix=bt709:in_range=tv,format=rgba" };
        let args = strings(&decode_args(&command));
        assert_eq!(
            args.join(" "),
            "-nostdin -hide_banner -loglevel error -i file:in put.mov -map 0:2 -an -sn -dn -fps_mode cfr -r 30000/1001 \
             -filter:v scale=in_color_matrix=bt709:in_range=tv,format=rgba -f rawvideo -pix_fmt rgba pipe:1"
        );
        let alpha = DecodeCommand { decoder: Some("libvpx-vp9"), ..command };
        assert!(strings(&decode_args(&alpha)).join(" ").starts_with("-nostdin -hide_banner -loglevel error -c:v libvpx-vp9 -i file:in put.mov"));
    }

    #[test]
    fn encoding_copies_audio_from_the_input() {
        let codec = VideoCodec::default();
        let command = EncodeCommand { width: 640, height: 360, rate: rate(25, 1), sample_aspect_ratio: (1, 1), codec: &codec, audio: Some((Path::new("in.mp4"), &AudioMode::Copy)), output: Path::new("out.mp4") };
        let args = strings(&encode_args(&command)).join(" ");
        assert_eq!(
            args,
            "-nostdin -hide_banner -loglevel error -y -f rawvideo -pix_fmt rgba -s 640x360 -framerate 25/1 -i pipe:0 -i file:in.mp4 -map 0:v:0 \
             -filter:v scale=out_color_matrix=bt709:out_range=tv,setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv \
             -colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv \
             -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p -map 1:a? -c:a copy -shortest file:out.mp4"
        );
    }

    #[test]
    fn audio_can_be_reencoded_or_dropped() {
        let codec = VideoCodec::Vp9 { crf: 30 };
        let encode = AudioMode::Encode(vec!["-c:a".to_owned(), "libopus".to_owned()]);
        let command = EncodeCommand { width: 2, height: 2, rate: rate(24, 1), sample_aspect_ratio: (1, 1), codec: &codec, audio: Some((Path::new("a.mkv"), &encode)), output: Path::new("-o.webm") };
        let args = strings(&encode_args(&command)).join(" ");
        assert!(args.ends_with("-map 1:a? -c:a libopus -shortest file:-o.webm"), "{args}");
        assert!(args.contains("-c:v libvpx-vp9 -crf 30 -b:v 0 -row-mt 1 -pix_fmt yuva420p"), "{args}");
        let dropped = EncodeCommand { audio: Some((Path::new("a.mkv"), &AudioMode::Drop)), ..command };
        let args = strings(&encode_args(&dropped)).join(" ");
        assert!(!args.contains("a.mkv") && args.ends_with("-an file:-o.webm"), "{args}");
        let none = EncodeCommand { audio: None, ..dropped };
        assert!(strings(&encode_args(&none)).join(" ").ends_with("-an file:-o.webm"));
    }

    #[test]
    fn gif_output_has_no_audio_input() {
        let codec = VideoCodec::Gif;
        let command = EncodeCommand { width: 4, height: 4, rate: rate(10, 1), sample_aspect_ratio: (1, 1), codec: &codec, audio: Some((Path::new("in.mp4"), &AudioMode::Copy)), output: Path::new("out.gif") };
        let args = strings(&encode_args(&command)).join(" ");
        assert!(!args.contains("in.mp4") && args.ends_with("-an file:out.gif"), "{args}");
        assert!(args.contains("-map 0:v:0 -filter:v split[a][b];[a]palettegen[p];[b][p]paletteuse -c:v gif"), "no colour tags for GIF: {args}");
    }

    #[test]
    fn one_filter_chain_merges_aspect_custom_filter_and_colour() {
        let custom = VideoCodec::Custom { args: ["-vf", "hflip", "-c:v", "libx265", "-filter:v", "vflip,negate", "-pix_fmt", "yuv420p"].map(str::to_owned).to_vec(), alpha: false, yuv: true };
        let args = custom.output_args((4, 3));
        let filters: Vec<&String> = args.iter().zip(args.iter().skip(1)).filter(|(option, _)| VIDEO_FILTER_OPTIONS.contains(&option.as_str())).map(|(_, value)| value).collect();
        assert_eq!(filters.len(), 1, "{args:?}");
        assert!(filters[0].starts_with("setsar=4/3,vflip,negate,scale=out_color_matrix=bt709"), "{}", filters[0]);
        assert!(args.join(" ").ends_with("-color_range tv -c:v libx265 -pix_fmt yuv420p"), "{args:?}");
        let rgb = VideoCodec::Custom { args: vec!["-c:v".to_owned(), "png".to_owned()], alpha: true, yuv: false };
        assert_eq!(rgb.args().join(" "), "-filter:v setparams=color_primaries=bt709:color_trc=iec61966-2-1:colorspace=gbr:range=pc -colorspace rgb -color_primaries bt709 -color_trc iec61966-2-1 -color_range pc -c:v png");
        assert!(VideoCodec::Gif.output_args((2, 1)).join(" ").contains("-filter:v setsar=2/1,split[a][b]"));
    }

    #[test]
    fn presets_declare_alpha_audio_and_size_constraints() {
        assert!(!VideoCodec::default().supports_alpha() && VideoCodec::default().needs_even_size());
        for codec in [VideoCodec::Vp9 { crf: 31 }, VideoCodec::ProRes4444, VideoCodec::Ffv1] {
            assert!(codec.supports_alpha(), "{}", codec.name());
        }
        assert!(!VideoCodec::Gif.supports_alpha() && !VideoCodec::Gif.supports_audio());
        assert!(!VideoCodec::Ffv1.needs_even_size() && !VideoCodec::ProRes4444.needs_even_size());
        let custom = VideoCodec::Custom { args: vec!["-c:v".to_owned(), "libx265".to_owned()], alpha: false, yuv: true };
        assert!(custom.args().ends_with(&["-c:v".to_owned(), "libx265".to_owned()]));
        assert!(!custom.supports_alpha() && custom.supports_audio() && !custom.needs_even_size());
        assert!(VideoCodec::Av1 { crf: 30 }.args().join(" ").ends_with("-color_range tv -c:v libsvtav1 -crf 30 -pix_fmt yuv420p"));
        assert!(VideoCodec::ProRes4444.args().join(" ").ends_with("-color_range tv -c:v prores_ks -profile:v 4444 -pix_fmt yuva444p10le"));
        assert!(VideoCodec::Ffv1.args().join(" ").contains("-colorspace rgb"));
    }
}
