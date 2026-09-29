/*
File: offline/image.rs

Purpose:
`image` crate adapters (feature `image`): filter a `DynamicImage` or an image file with an
`OfflineRenderer`.

Key functions:
- `OfflineRenderer::process_image()` (native, blocking) / `process_image_async()`.
- `OfflineRenderer::process_file()` (native): decode by content, encode by the output extension.

Notes:
Every input is converted to 8-bit straight-alpha RGBA first (`DynamicImage::to_rgba8`): the
pipeline's capture is 8-bit, as in the UI, so 16-bit and float images lose precision. Colour
profiles (ICC, gAMA) are ignored: pixels are treated as sRGB. Only PNG and JPEG are compiled in;
enable more formats with the `image` crate's own features in your `Cargo.toml` (e.g.
`image = { version = "0.25", features = ["webp", "gif"] }`).
*/

use image::{DynamicImage, RgbaImage};

use super::{AlphaMode, FrameRef, OfflineError, OfflineOptions, OfflineRenderer, RgbaFrame};
use crate::pass::FilterChain;

impl OfflineRenderer {
    /// Filters `image` (converted to 8-bit RGBA) and returns an RGBA image (blocking; native).
    /// The result's alpha follows `options` ([`AlphaMode::Straight`] by default, what the
    /// `image` crate's encoders expect).
    ///
    /// # Errors
    /// As [`OfflineRenderer::process`].
    #[cfg(not(target_arch = "wasm32"))]
    pub fn process_image(&mut self, image: &DynamicImage, chain: &FilterChain, options: &OfflineOptions) -> Result<RgbaImage, OfflineError> {
        let input = image.to_rgba8();
        let result = self.process(&FrameRef::new(input.as_raw(), input.width(), input.height(), AlphaMode::Straight), chain, options)?;
        into_rgba_image(result)
    }

    /// Async [`OfflineRenderer::process_image`] (see [`OfflineRenderer::process_async`] for
    /// who drives the readback on native).
    ///
    /// # Errors
    /// As [`OfflineRenderer::process`].
    pub async fn process_image_async(&mut self, image: &DynamicImage, chain: &FilterChain, options: &OfflineOptions) -> Result<RgbaImage, OfflineError> {
        let input = image.to_rgba8();
        let result = self.process_async(&FrameRef::new(input.as_raw(), input.width(), input.height(), AlphaMode::Straight), chain, options).await?;
        into_rgba_image(result)
    }

    /// Reads `input` (format detected from its content), filters it and writes `output` (format
    /// from its extension). Blocking and file I/O: native only, never on a GUI thread.
    ///
    /// # Errors
    /// [`OfflineError::Image`] for I/O, decoding, encoding or an unknown output extension;
    /// [`OfflineError::AlphaNotSupported`] when the output format has no alpha channel and the
    /// result is not opaque (give `options` a background colour); otherwise as
    /// [`OfflineRenderer::process`].
    #[cfg(not(target_arch = "wasm32"))]
    pub fn process_file(&mut self, input: &std::path::Path, output: &std::path::Path, chain: &FilterChain, options: &OfflineOptions) -> Result<(), OfflineError> {
        let decoded = image::ImageReader::open(input).map_err(image::ImageError::IoError)?.with_guessed_format().map_err(image::ImageError::IoError)?.decode()?;
        log::info!("egui_shader_layers (offline): processing {} ({}x{})", input.display(), decoded.width(), decoded.height());
        let format = image::ImageFormat::from_path(output)?;
        // Encoders need straight alpha, whatever the caller asked the renderer for.
        let result = self.process_image(&decoded, chain, &options.clone().with_output_alpha(AlphaMode::Straight))?;
        let encoded = if format_has_alpha(format) {
            DynamicImage::ImageRgba8(result)
        } else if result.pixels().all(|pixel| pixel.0[3] == u8::MAX) {
            DynamicImage::ImageRgb8(DynamicImage::ImageRgba8(result).to_rgb8())
        } else {
            return Err(OfflineError::AlphaNotSupported { format: format!("{format:?}") });
        };
        encoded.save_with_format(output, format)?;
        log::info!("egui_shader_layers (offline): wrote {}", output.display());
        Ok(())
    }
}

/// Whether `format` stores an alpha channel. The `image` crate's encoders for JPEG, Radiance HDR
/// and PNM silently drop alpha (`make_compatible_img`), so those are handled explicitly.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn format_has_alpha(format: image::ImageFormat) -> bool {
    !matches!(format, image::ImageFormat::Jpeg | image::ImageFormat::Hdr | image::ImageFormat::Pnm)
}

/// Wraps a result frame as an `RgbaImage`.
///
/// # Errors
/// [`OfflineError::SizeOverflow`] if the buffer does not match its size (cannot happen for a
/// frame produced by the renderer).
fn into_rgba_image(frame: RgbaFrame) -> Result<RgbaImage, OfflineError> {
    let (width, height) = (frame.width(), frame.height());
    RgbaImage::from_raw(width, height, frame.into_pixels()).ok_or(OfflineError::SizeOverflow { what: "the result image" })
}
