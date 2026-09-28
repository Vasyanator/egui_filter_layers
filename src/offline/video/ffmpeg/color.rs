/*
File: offline/video/ffmpeg/color.rs

Purpose:
Colour handling of the ffmpeg adapters, so that a decoded, filtered and re-encoded video looks
like the frames the filters saw. Decoding: which YCbCr matrix and range convert the input to the
sRGB frames the filters work on (from the probed tags, with a documented default for untagged
input), and which inputs cannot be converted without tone or gamut mapping. Encoding: the filter
and the tags that store frames as BT.709 limited-range YCbCr, or as tagged sRGB.

Key structures:
- `DecodeColor`: the `scale` filter options of a decode (`in_color_matrix`, `in_range`).
- `OutputColor`: how an output format stores colour (YCbCr, RGB, palette).

Key functions:
- `decode_color()`: probe tags -> `DecodeColor` or the reason the input is unsupported.
- `OutputColor::filter()` / `OutputColor::tags()`: the conversion filter and output options.

Notes:
The explicit filter matters: ffmpeg's default conversion is BT.601 for untagged frames, and with
ffmpeg 8 the frames' own properties override `-colorspace` / `-color_*` output options, so tags
are set on the frames with `setparams` as well as on the output.
*/

use super::probe::Probe;

/// Width or height from which untagged YCbCr is treated as HD (BT.709), like common players.
const HD_WIDTH: u32 = 1280;
/// Height above which untagged YCbCr is treated as HD (BT.709).
const SD_MAX_HEIGHT: u32 = 576;

/// Options of the `scale` filter converting a decoded stream to RGBA.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DecodeColor {
    /// `in_color_matrix` value (`bt709`, `bt601`, ...); `None` for RGB-coded input.
    pub matrix: Option<&'static str>,
    /// `in_range` value (`tv`, `pc` or `auto`).
    pub range: &'static str,
    /// Whether the matrix is the default for an untagged stream (worth a warning).
    pub guessed: bool,
}

impl DecodeColor {
    /// The decode filter chain ending in straight RGBA.
    pub(crate) fn filter(&self) -> String {
        match self.matrix {
            Some(matrix) => format!("scale=in_color_matrix={matrix}:in_range={},format=rgba", self.range),
            None => "format=rgba".to_owned(),
        }
    }
}

/// Chooses the conversion of `probe`'s stream to sRGB frames.
///
/// Tagged matrices and ranges are honoured. An untagged YCbCr stream is BT.709 when it is HD
/// (width at least 1280 or height above 576) and BT.601 otherwise — what common players assume —
/// and its range is left to ffmpeg (limited, full for JPEG-style `yuvj` formats).
///
/// # Errors
/// The reason, when the stream needs a conversion this crate does not do: HDR transfers (PQ,
/// HLG), wide-gamut primaries (BT.2020, DCI-P3) or matrices `scale` cannot invert.
pub(crate) fn decode_color(probe: &Probe) -> Result<DecodeColor, String> {
    match probe.color_transfer.as_str() {
        "smpte2084" | "arib-std-b67" | "smpte428" => {
            return Err(format!("the transfer `{}` is HDR; tone-map it to SDR BT.709 first (e.g. with ffmpeg's `zscale` and `tonemap` filters)", probe.color_transfer));
        }
        _ => {}
    }
    match probe.color_primaries.as_str() {
        "bt2020" | "smpte428" | "smpte431" | "smpte432" => {
            return Err(format!("the primaries `{}` are wide-gamut; convert the video to BT.709 first", probe.color_primaries));
        }
        _ => {}
    }
    let range = match probe.color_range.as_str() {
        "tv" => "tv",
        "pc" => "pc",
        _ => "auto",
    };
    if probe.is_rgb {
        return Ok(DecodeColor { matrix: None, range, guessed: false });
    }
    let tagged = match probe.color_space.as_str() {
        "bt709" => Some("bt709"),
        "fcc" => Some("fcc"),
        "bt470bg" | "smpte170m" => Some("bt601"),
        "smpte240m" => Some("smpte240m"),
        "bt2020nc" => Some("bt2020"),
        "gbr" | "rgb" => return Ok(DecodeColor { matrix: None, range, guessed: false }),
        "unknown" | "unspecified" | "reserved" | "" => None,
        other => return Err(format!("the YCbCr matrix `{other}` is not supported; convert the video to BT.709 first")),
    };
    if let Some(matrix) = tagged {
        return Ok(DecodeColor { matrix: Some(matrix), range, guessed: false });
    }
    let hd = probe.width.max(probe.height) >= HD_WIDTH || probe.width.min(probe.height) > SD_MAX_HEIGHT;
    Ok(DecodeColor { matrix: Some(if hd { "bt709" } else { "bt601" }), range, guessed: true })
}

/// How an output format stores colour.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OutputColor {
    /// YCbCr: converted with the BT.709 matrix to limited range, tagged BT.709.
    Yuv,
    /// RGB: stored as is, tagged sRGB (BT.709 primaries, sRGB transfer, full range).
    Rgb,
    /// Palette (GIF): stored as is, no tags (the format has none).
    Palette,
}

impl OutputColor {
    /// The filter converting the RGBA input and setting the frames' colour properties.
    pub(crate) fn filter(self) -> Option<&'static str> {
        match self {
            Self::Yuv => Some("scale=out_color_matrix=bt709:out_range=tv,setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv"),
            Self::Rgb => Some("setparams=color_primaries=bt709:color_trc=iec61966-2-1:colorspace=gbr:range=pc"),
            Self::Palette => None,
        }
    }

    /// Output options tagging the stream (also set on the frames by [`OutputColor::filter`]).
    pub(crate) fn tags(self) -> &'static [&'static str] {
        match self {
            Self::Yuv => &["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"],
            Self::Rgb => &["-colorspace", "rgb", "-color_primaries", "bt709", "-color_trc", "iec61966-2-1", "-color_range", "pc"],
            Self::Palette => &[],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn probe(width: u32, height: u32, space: &str, range: &str) -> Probe {
        Probe { width, height, color_space: space.to_owned(), color_range: range.to_owned(), ..Probe::default() }
    }

    #[test]
    fn tagged_matrices_and_ranges_are_honoured() {
        let color = decode_color(&probe(640, 480, "bt709", "tv")).ok();
        assert_eq!(color.map(|color| color.filter()).as_deref(), Some("scale=in_color_matrix=bt709:in_range=tv,format=rgba"));
        assert_eq!(decode_color(&probe(1920, 1080, "smpte170m", "pc")).ok(), Some(DecodeColor { matrix: Some("bt601"), range: "pc", guessed: false }));
        assert_eq!(decode_color(&probe(1920, 1080, "bt2020nc", "tv")).ok().and_then(|color| color.matrix), Some("bt2020"));
    }

    #[test]
    fn untagged_input_defaults_by_size() {
        assert_eq!(decode_color(&probe(1280, 720, "unknown", "unknown")).ok(), Some(DecodeColor { matrix: Some("bt709"), range: "auto", guessed: true }));
        assert_eq!(decode_color(&probe(720, 576, "unknown", "unknown")).ok(), Some(DecodeColor { matrix: Some("bt601"), range: "auto", guessed: true }));
        assert_eq!(decode_color(&probe(720, 1280, "", "")).ok().and_then(|color| color.matrix), Some("bt709"), "portrait HD");
    }

    #[test]
    fn rgb_input_needs_no_matrix() {
        let rgb = Probe { is_rgb: true, ..probe(64, 64, "unknown", "unknown") };
        assert_eq!(decode_color(&rgb).ok().map(|color| color.filter()).as_deref(), Some("format=rgba"));
        assert_eq!(decode_color(&probe(64, 64, "gbr", "pc")).ok().and_then(|color| color.matrix), None);
    }

    #[test]
    fn hdr_and_wide_gamut_are_rejected() {
        let hlg = Probe { color_transfer: "arib-std-b67".to_owned(), ..probe(1920, 1080, "bt2020nc", "tv") };
        assert!(decode_color(&hlg).is_err_and(|reason| reason.contains("HDR")));
        let p3 = Probe { color_primaries: "smpte432".to_owned(), ..probe(1920, 1080, "bt709", "tv") };
        assert!(decode_color(&p3).is_err_and(|reason| reason.contains("wide-gamut")));
        assert!(decode_color(&probe(64, 64, "ycgco", "tv")).is_err());
    }

    #[test]
    fn outputs_are_tagged_consistently() {
        assert!(OutputColor::Yuv.filter().is_some_and(|filter| filter.contains("out_color_matrix=bt709") && filter.contains("range=tv")));
        assert_eq!(OutputColor::Yuv.tags().join(" "), "-colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv");
        assert!(OutputColor::Rgb.filter().is_some_and(|filter| filter.contains("colorspace=gbr")));
        assert!(OutputColor::Palette.filter().is_none() && OutputColor::Palette.tags().is_empty());
    }
}
