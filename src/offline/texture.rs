/*
File: offline/texture.rs

Purpose:
`OfflineTexture`, the offline counterpart of an egui texture a `ShaderLayer` attaches (F8): an
owned, premultiplied RGBA8 image plus the egui `TextureOptions` it is sampled with, shared cheaply
between clones (a video processor passes the same one with every frame).

Key structures:
- `OfflineTexture` (public).

Key functions:
- `OfflineTexture::new()`: from a `FrameRef` (straight alpha premultiplied with egui's rounding,
  like the frame itself).
- `OfflineTexture::from_color_image()`: from an `egui::ColorImage`, byte for byte what egui
  uploads for `Context::load_texture` (the UI / offline equivalence path).

Notes:
Identity, not content, decides whether the renderer re-uploads: `OfflineRenderer` keeps the GPU
copy of the texture it last saw at each slot and reuses it while the same `OfflineTexture` (a clone
of it) is passed again (`OfflineTexture::same`). `PartialEq` is that identity too.
*/

use std::fmt;
use std::sync::Arc;

use egui::{ColorImage, TextureOptions};

use super::frame::{FrameRef, MAX_SIDE, frame_bytes};
use super::OfflineError;

/// An image an offline run's effects read as a user texture (`tex(i, uv)`, attached with
/// [`OfflineOptions::with_texture`](super::OfflineOptions::with_texture)): premultiplied RGBA8
/// and the egui [`TextureOptions`] it is sampled with (filters and wrap mode, as egui-wgpu maps
/// them; the mipmap mode is ignored, effects read level 0).
///
/// Cloning is cheap (shared pixels). The renderer uploads a texture once and reuses the upload
/// while the same texture (or a clone of it) stays attached at that slot, so build it once, not
/// per frame.
#[derive(Clone)]
pub struct OfflineTexture {
    /// The shared image.
    inner: Arc<TextureData>,
}

/// The image behind an [`OfflineTexture`].
struct TextureData {
    /// `[width, height]` in texels, both in `1..=MAX_SIDE`.
    size: [u32; 2],
    /// Tightly packed premultiplied RGBA8 rows, top row first.
    pixels: Vec<u8>,
    /// Sampling options.
    options: TextureOptions,
}

impl OfflineTexture {
    /// A texture from `frame` (straight or premultiplied RGBA8, any stride), premultiplied with
    /// egui's rounding like the frames an `OfflineRenderer` processes.
    ///
    /// # Errors
    /// The frame validation errors of [`OfflineError`] (empty, too large, stride, buffer length).
    pub fn new(frame: &FrameRef<'_>, options: TextureOptions) -> Result<Self, OfflineError> {
        let mut pixels = Vec::new();
        frame.premultiplied_rows_into(None, &mut pixels)?;
        Ok(Self { inner: Arc::new(TextureData { size: [frame.width(), frame.height()], pixels, options }) })
    }

    /// A texture holding exactly the bytes egui uploads for `image` (its `Color32`s are already
    /// premultiplied): what a layer attaching `ctx.load_texture(name, image, options)` samples.
    ///
    /// # Errors
    /// [`OfflineError::EmptyFrame`] / [`OfflineError::TooLarge`] for an image the pipeline cannot
    /// address.
    pub fn from_color_image(image: &ColorImage, options: TextureOptions) -> Result<Self, OfflineError> {
        let [width, height] = image.size.map(|side| u32::try_from(side).unwrap_or(u32::MAX));
        if width == 0 || height == 0 {
            return Err(OfflineError::EmptyFrame { width, height });
        }
        if width > MAX_SIDE || height > MAX_SIDE {
            return Err(OfflineError::TooLarge { width, height, max: MAX_SIDE });
        }
        let expected = frame_bytes(width, height)?;
        let pixels: Vec<u8> = image.pixels.iter().flat_map(egui::Color32::to_array).collect();
        if pixels.len() != expected {
            // A `ColorImage` whose pixel count disagrees with its size (constructible through its
            // public fields) cannot be uploaded as that size.
            return Err(OfflineError::BufferTooShort { width, height, stride: usize::try_from(width).unwrap_or(usize::MAX).saturating_mul(4), expected, actual: pixels.len() });
        }
        Ok(Self { inner: Arc::new(TextureData { size: [width, height], pixels, options }) })
    }

    /// `[width, height]` in texels.
    #[must_use]
    pub fn size(&self) -> [u32; 2] {
        self.inner.size
    }

    /// The sampling options.
    #[must_use]
    pub fn options(&self) -> TextureOptions {
        self.inner.options
    }

    /// The premultiplied, tightly packed RGBA8 rows.
    pub(super) fn pixels(&self) -> &[u8] {
        &self.inner.pixels
    }

    /// Whether `other` is this texture or a clone of it (the renderer's re-upload key).
    pub(super) fn same(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl PartialEq for OfflineTexture {
    /// Identity: `true` for the same texture or a clone of it (not a pixel comparison).
    fn eq(&self, other: &Self) -> bool {
        self.same(other)
    }
}

impl fmt::Debug for OfflineTexture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OfflineTexture").field("size", &self.inner.size).field("options", &self.inner.options).finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::offline::AlphaMode;
    use egui::Color32;

    #[test]
    fn textures_hold_premultiplied_rows_and_compare_by_identity() {
        let straight = [255_u8, 0, 0, 128, 0, 0, 255, 255];
        let Ok(texture) = OfflineTexture::new(&FrameRef::new(&straight, 2, 1, AlphaMode::Straight), TextureOptions::NEAREST) else { panic!("valid") };
        let red = Color32::from_rgba_unmultiplied(255, 0, 0, 128).to_array();
        assert_eq!(texture.pixels(), [red, [0, 0, 255, 255]].concat());
        assert_eq!(texture.size(), [2, 1]);
        assert_eq!(texture.options(), TextureOptions::NEAREST);
        assert_eq!(texture.clone(), texture, "a clone is the same texture");
        let Ok(copy) = OfflineTexture::new(&FrameRef::new(&straight, 2, 1, AlphaMode::Straight), TextureOptions::NEAREST) else { panic!("valid") };
        assert_ne!(copy, texture, "equal pixels are still another upload");

        let image = ColorImage::new([2, 1], vec![Color32::from_rgba_unmultiplied(255, 0, 0, 128), Color32::BLUE]);
        let Ok(from_image) = OfflineTexture::from_color_image(&image, TextureOptions::LINEAR) else { panic!("valid") };
        assert_eq!(from_image.pixels(), texture.pixels(), "egui's Color32 bytes, as uploaded");
        assert!(matches!(OfflineTexture::from_color_image(&ColorImage::new([0, 3], Vec::new()), TextureOptions::LINEAR), Err(OfflineError::EmptyFrame { .. })));
        assert!(matches!(OfflineTexture::new(&FrameRef::new(&straight, 3, 1, AlphaMode::Straight), TextureOptions::LINEAR), Err(OfflineError::BufferTooShort { .. })));
    }
}
