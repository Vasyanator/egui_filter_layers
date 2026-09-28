/*
File: offline/frame.rs

Purpose:
CPU side of an offline frame: the borrowed input (`FrameRef`), the owned output (`RgbaFrame`),
their alpha convention (`AlphaMode`), and the exact colour conversions that make the GPU input
equal to what egui uploads: premultiplication with egui's own rounding, compositing over an
optional background, and unpremultiplication of the result.

Key structures:
- `FrameRef` (public): RGBA8 pixels with a row stride and an alpha mode.
- `RgbaFrame` (public): tightly packed RGBA8 result.
- `AlphaMode` (public): straight or premultiplied.

Key functions:
- `FrameRef::premultiplied_rows_into()`: validated, tightly packed, premultiplied copy of the
  frame into a reused buffer.
- `over_background()`, `convert_output()`: per-pixel colour maths (unit-tested).

Notes:
Premultiplication goes through `egui::Color32::from_rgba_unmultiplied` and unpremultiplication
through `Color32::to_srgba_unmultiplied`, so both round exactly like egui does for a
`ColorImage` shown in the UI. Every size computation is checked.
*/

use egui::Color32;

use super::error::OfflineError;

/// Largest supported image side: pixel coordinates are handled as `f32`, exact below 2^24.
pub(crate) const MAX_SIDE: u32 = (1 << 24) - 1;

/// How colour channels relate to alpha in a buffer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum AlphaMode {
    /// Colour is independent of alpha (PNG, most decoders, `ffmpeg -pix_fmt rgba`).
    #[default]
    Straight,
    /// Colour is already multiplied by alpha (what egui and the filter chain work on).
    Premultiplied,
}

/// A borrowed RGBA8 frame: `height` rows of `width` pixels, row `y` starting at byte
/// `y * stride`, colours in sRGB gamma space (as decoded, not linearised).
#[derive(Clone, Copy, Debug)]
pub struct FrameRef<'a> {
    /// The pixel bytes.
    pixels: &'a [u8],
    /// Width in pixels.
    width: u32,
    /// Height in pixels.
    height: u32,
    /// Bytes from one row to the next (`>= width * 4`).
    stride: usize,
    /// Alpha convention of `pixels`.
    alpha: AlphaMode,
}

impl<'a> FrameRef<'a> {
    /// A tightly packed frame (`stride = width * 4`). Sizes are validated when the frame is
    /// processed ([`OfflineError::BufferTooShort`], [`OfflineError::EmptyFrame`], ...).
    #[must_use]
    pub fn new(pixels: &'a [u8], width: u32, height: u32, alpha: AlphaMode) -> Self {
        let stride = usize::try_from(width).ok().and_then(|width| width.checked_mul(4)).unwrap_or(usize::MAX);
        Self { pixels, width, height, stride, alpha }
    }

    /// The same frame with rows `stride` bytes apart (e.g. a decoder's padded rows).
    #[must_use]
    pub fn with_stride(mut self, stride: usize) -> Self {
        self.stride = stride;
        self
    }

    /// Width in pixels.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Height in pixels.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Alpha convention of the pixels.
    #[must_use]
    pub fn alpha(&self) -> AlphaMode {
        self.alpha
    }

    /// The pixel bytes as given (row `y` starts at `y * stride()`; not validated until the
    /// frame is processed).
    #[must_use]
    pub fn pixels(&self) -> &'a [u8] {
        self.pixels
    }

    /// Bytes from one row to the next.
    #[must_use]
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Checks the size, stride and buffer length.
    ///
    /// # Errors
    /// [`OfflineError::EmptyFrame`], [`OfflineError::TooLarge`], [`OfflineError::StrideTooSmall`],
    /// [`OfflineError::BufferTooShort`] or [`OfflineError::SizeOverflow`].
    pub(crate) fn validate(&self) -> Result<(), OfflineError> {
        let (width, height) = (self.width, self.height);
        if width == 0 || height == 0 {
            return Err(OfflineError::EmptyFrame { width, height });
        }
        if width > MAX_SIDE || height > MAX_SIDE {
            return Err(OfflineError::TooLarge { width, height, max: MAX_SIDE });
        }
        let row = row_bytes(width)?;
        if self.stride < row {
            return Err(OfflineError::StrideTooSmall { stride: self.stride, width, min: row });
        }
        let rows_before_last = usize::try_from(height - 1).map_err(|_| OfflineError::SizeOverflow { what: "the input frame" })?;
        let expected = self.stride.checked_mul(rows_before_last).and_then(|bytes| bytes.checked_add(row)).ok_or(OfflineError::SizeOverflow { what: "the input frame" })?;
        if self.pixels.len() < expected {
            return Err(OfflineError::BufferTooShort { width, height, stride: self.stride, expected, actual: self.pixels.len() });
        }
        Ok(())
    }

    /// The validated pixels as one tightly packed buffer (`width * 4` bytes per row), borrowed
    /// when the frame is already packed, copied otherwise. The alpha convention is unchanged.
    ///
    /// # Errors
    /// As [`FrameRef::validate`].
    #[cfg(all(feature = "video", not(target_arch = "wasm32")))]
    pub(crate) fn packed(&self) -> Result<std::borrow::Cow<'a, [u8]>, OfflineError> {
        self.validate()?;
        let row = row_bytes(self.width)?;
        let total = frame_bytes(self.width, self.height)?;
        if self.stride == row {
            // `validate` proved `total` bytes are present.
            return Ok(std::borrow::Cow::Borrowed(self.pixels.get(..total).unwrap_or(self.pixels)));
        }
        let rows = self.pixels.chunks(self.stride).take(usize::try_from(self.height).unwrap_or(0));
        Ok(std::borrow::Cow::Owned(rows.flat_map(|line| line.get(..row).unwrap_or(line).iter().copied()).collect()))
    }

    /// A tightly packed, premultiplied copy of the frame, composited over `background` when
    /// given — the texel values egui would upload for this image (drawn over that colour).
    ///
    /// # Errors
    /// As [`FrameRef::validate`].
    #[cfg(test)]
    pub(crate) fn premultiplied_rows(&self, background: Option<Color32>) -> Result<Vec<u8>, OfflineError> {
        let mut out = Vec::new();
        self.premultiplied_rows_into(background, &mut out)?;
        Ok(out)
    }

    /// [`FrameRef::premultiplied_rows`] into `out` (cleared first; its capacity is reused, so a
    /// frame stream does not allocate per frame).
    ///
    /// # Errors
    /// As [`FrameRef::validate`].
    pub(crate) fn premultiplied_rows_into(&self, background: Option<Color32>, out: &mut Vec<u8>) -> Result<(), OfflineError> {
        self.validate()?;
        let row = row_bytes(self.width)?;
        let total = frame_bytes(self.width, self.height)?;
        out.clear();
        out.reserve(total);
        // `validate` proved every row lies inside `pixels`.
        for line in self.pixels.chunks(self.stride).take(usize::try_from(self.height).unwrap_or(0)) {
            let line = line.get(..row).unwrap_or(line);
            for pixel in line.chunks_exact(4) {
                let [r, g, b, a] = [pixel[0], pixel[1], pixel[2], pixel[3]];
                let color = match self.alpha {
                    AlphaMode::Straight => Color32::from_rgba_unmultiplied(r, g, b, a),
                    AlphaMode::Premultiplied => Color32::from_rgba_premultiplied(r, g, b, a),
                };
                let color = background.map_or(color, |background| over_background(color, background));
                out.extend_from_slice(&color.to_array());
            }
        }
        Ok(())
    }
}

/// An owned, tightly packed RGBA8 frame (`width * 4` bytes per row).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RgbaFrame {
    /// Width in pixels.
    width: u32,
    /// Height in pixels.
    height: u32,
    /// Alpha convention of `pixels`.
    alpha: AlphaMode,
    /// `width * height * 4` bytes, row-major, top row first.
    pixels: Vec<u8>,
}

impl RgbaFrame {
    /// Wraps a result buffer (crate-internal: the length is guaranteed by the producer).
    pub(crate) fn from_parts(width: u32, height: u32, alpha: AlphaMode, pixels: Vec<u8>) -> Self {
        Self { width, height, alpha, pixels }
    }

    /// Width in pixels.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Height in pixels.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Alpha convention of the pixels (`OfflineOptions::with_output_alpha`).
    #[must_use]
    pub fn alpha(&self) -> AlphaMode {
        self.alpha
    }

    /// The pixels: `width * height * 4` bytes, top row first.
    #[must_use]
    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }

    /// Consumes the frame into its pixel buffer.
    #[must_use]
    pub fn into_pixels(self) -> Vec<u8> {
        self.pixels
    }

    /// A borrowed view of this frame, e.g. to feed it into another run.
    #[must_use]
    pub fn as_frame_ref(&self) -> FrameRef<'_> {
        FrameRef::new(&self.pixels, self.width, self.height, self.alpha)
    }
}

/// `width * 4`, checked.
///
/// # Errors
/// [`OfflineError::SizeOverflow`].
pub(crate) fn row_bytes(width: u32) -> Result<usize, OfflineError> {
    usize::try_from(width).ok().and_then(|width| width.checked_mul(4)).ok_or(OfflineError::SizeOverflow { what: "a row of pixels" })
}

/// `width * height * 4`, checked.
///
/// # Errors
/// [`OfflineError::SizeOverflow`].
pub(crate) fn frame_bytes(width: u32, height: u32) -> Result<usize, OfflineError> {
    let height = usize::try_from(height).map_err(|_| OfflineError::SizeOverflow { what: "the frame" })?;
    row_bytes(width)?.checked_mul(height).ok_or(OfflineError::SizeOverflow { what: "the frame" })
}

/// Premultiplied `color` drawn over the premultiplied `background` with egui's blend
/// (`rgb = src + dst * (1 - src.a)`, `a = src.a * (1 - dst.a) + dst.a`), rounded to 8 bits.
/// The UI blends on the GPU, whose rounding may differ by one step for translucent pixels.
#[must_use]
pub(crate) fn over_background(color: Color32, background: Color32) -> Color32 {
    let src = color.to_array().map(|channel| f32::from(channel) / 255.0);
    let dst = background.to_array().map(|channel| f32::from(channel) / 255.0);
    let keep = 1.0 - src[3];
    let blend = |s: f32, d: f32| unorm8(s + d * keep);
    Color32::from_rgba_premultiplied(blend(src[0], dst[0]), blend(src[1], dst[1]), blend(src[2], dst[2]), unorm8(src[3] * (1.0 - dst[3]) + dst[3]))
}

/// Converts a `0..=1` value to 8 bits like a unorm render target: clamp, scale, round to nearest.
fn unorm8(value: f32) -> u8 {
    let scaled = (value.clamp(0.0, 1.0) * 255.0).round();
    // Clamped to 0..=255 above, so the saturating cast is exact.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let byte = scaled as u8;
    byte
}

/// Converts the premultiplied `pixels` in place to `alpha` (unpremultiplying with egui's
/// rounding for [`AlphaMode::Straight`]; low alpha loses colour precision, as in any 8-bit
/// unpremultiply).
pub(crate) fn convert_output(pixels: &mut [u8], alpha: AlphaMode) {
    match alpha {
        AlphaMode::Premultiplied => {}
        AlphaMode::Straight => {
            for pixel in pixels.chunks_exact_mut(4) {
                let straight = Color32::from_rgba_premultiplied(pixel[0], pixel[1], pixel[2], pixel[3]).to_srgba_unmultiplied();
                pixel.copy_from_slice(&straight);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn premultiplication_matches_egui_for_every_pair() {
        let mut pixels = Vec::with_capacity(256 * 256 * 4);
        for a in 0..=255_u8 {
            for c in 0..=255_u8 {
                pixels.extend_from_slice(&[c, 255 - c, c / 2, a]);
            }
        }
        let frame = FrameRef::new(&pixels, 256, 256, AlphaMode::Straight);
        let Ok(out) = frame.premultiplied_rows(None) else { panic!("valid frame") };
        for (input, output) in pixels.chunks_exact(4).zip(out.chunks_exact(4)) {
            assert_eq!(output, Color32::from_rgba_unmultiplied(input[0], input[1], input[2], input[3]).to_array());
        }
    }

    #[test]
    fn strides_are_honoured_and_validated() {
        // 2x2 frame with 3 bytes of row padding.
        let pixels = [1, 2, 3, 255, 4, 5, 6, 255, 9, 9, 9, 7, 8, 9, 255, 10, 11, 12, 255];
        let frame = FrameRef::new(&pixels, 2, 2, AlphaMode::Premultiplied).with_stride(11);
        assert_eq!(frame.premultiplied_rows(None).ok(), Some(vec![1, 2, 3, 255, 4, 5, 6, 255, 7, 8, 9, 255, 10, 11, 12, 255]));
        assert!(matches!(FrameRef::new(&pixels, 2, 2, AlphaMode::Straight).with_stride(7).validate(), Err(OfflineError::StrideTooSmall { min: 8, .. })));
        assert!(matches!(FrameRef::new(&pixels[..18], 2, 2, AlphaMode::Straight).with_stride(11).validate(), Err(OfflineError::BufferTooShort { expected: 19, actual: 18, .. })));
        assert!(matches!(FrameRef::new(&pixels, 0, 2, AlphaMode::Straight).validate(), Err(OfflineError::EmptyFrame { .. })));
        assert!(matches!(FrameRef::new(&pixels, MAX_SIDE + 1, 1, AlphaMode::Straight).validate(), Err(OfflineError::TooLarge { .. })));
        assert!(matches!(FrameRef::new(&pixels, 2, 2, AlphaMode::Straight).with_stride(usize::MAX).validate(), Err(OfflineError::SizeOverflow { .. })));
    }

    #[cfg(all(feature = "video", not(target_arch = "wasm32")))]
    #[test]
    fn packed_frames_borrow_or_drop_the_row_padding() {
        let pixels = [1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 9, 10, 11, 12, 13, 14, 15, 16];
        let padded = FrameRef::new(&pixels, 2, 2, AlphaMode::Straight).with_stride(10);
        let Ok(packed) = padded.packed() else { panic!("valid") };
        assert_eq!(&*packed, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        let tight = FrameRef::new(&pixels[..8], 1, 2, AlphaMode::Straight);
        assert!(matches!(tight.packed(), Ok(std::borrow::Cow::Borrowed(bytes)) if bytes.len() == 8));
        assert!(FrameRef::new(&pixels[..7], 1, 2, AlphaMode::Straight).packed().is_err());
    }

    #[test]
    fn background_compositing_is_egui_over() {
        let background = Color32::from_rgb(200, 100, 0);
        // Opaque pixels are unchanged, transparent ones become the background.
        assert_eq!(over_background(Color32::from_rgb(1, 2, 3), background), Color32::from_rgb(1, 2, 3));
        assert_eq!(over_background(Color32::TRANSPARENT, background), background);
        // Half-covered white over the background: 128 + 200 * (1 - 128/255) = 227.6 -> 228.
        let half = Color32::from_rgba_premultiplied(128, 128, 128, 128);
        assert_eq!(over_background(half, background).to_array(), [228, 178, 128, 255]);
    }

    #[test]
    fn straight_output_unpremultiplies_like_egui() {
        let mut pixels = [64, 32, 0, 128, 10, 20, 30, 255, 0, 0, 0, 0];
        convert_output(&mut pixels, AlphaMode::Straight);
        assert_eq!(pixels[..4], Color32::from_rgba_premultiplied(64, 32, 0, 128).to_srgba_unmultiplied());
        assert_eq!(pixels[4..], [10, 20, 30, 255, 0, 0, 0, 0]);
        let mut untouched = [64, 32, 0, 128];
        convert_output(&mut untouched, AlphaMode::Premultiplied);
        assert_eq!(untouched, [64, 32, 0, 128]);
    }

    #[test]
    fn frame_sizes_are_checked() {
        assert_eq!(frame_bytes(3, 5).ok(), Some(60));
        assert!(frame_bytes(u32::MAX, u32::MAX).is_err(), "2^66 bytes overflow usize");
    }
}
