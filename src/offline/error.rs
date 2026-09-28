/*
File: offline/error.rs

Purpose:
The one error type of offline processing, `OfflineError`: invalid input frames and options,
sizes beyond what the pipeline can address, GPU / effect failures, readback failures and (with
the `image` feature) image I/O.

Notes:
Messages are user-facing sentences; the variants carry the numbers needed to diagnose the
problem (sizes, strides, limits) without any pixel data.
*/

/// Why an offline run (or creating an [`OfflineRenderer`](super::OfflineRenderer)) failed.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum OfflineError {
    /// No GPU adapter could be obtained for a headless renderer.
    #[error("no GPU adapter is available for offline processing: {0}")]
    NoAdapter(String),
    /// The adapter refused to create a device.
    #[error("could not create a GPU device for offline processing: {0}")]
    RequestDevice(String),
    /// The frame has a zero width or height.
    #[error("the frame is empty ({width}x{height} pixels)")]
    EmptyFrame {
        /// Width in pixels.
        width: u32,
        /// Height in pixels.
        height: u32,
    },
    /// The row stride is smaller than one row of RGBA8 pixels.
    #[error("the row stride of {stride} bytes is smaller than one row of {width} RGBA8 pixels ({min} bytes)")]
    StrideTooSmall {
        /// The stride given, in bytes.
        stride: usize,
        /// Width in pixels.
        width: u32,
        /// `width * 4`.
        min: usize,
    },
    /// The pixel buffer is shorter than its size and stride require.
    #[error("the frame buffer holds {actual} bytes, but {width}x{height} pixels with a stride of {stride} bytes need {expected}")]
    BufferTooShort {
        /// Width in pixels.
        width: u32,
        /// Height in pixels.
        height: u32,
        /// Row stride in bytes.
        stride: usize,
        /// Bytes required.
        expected: usize,
        /// Bytes given.
        actual: usize,
    },
    /// A side exceeds what the pipeline addresses exactly (pixel coordinates are `f32`).
    #[error("{width}x{height} pixels is too large: each side may be at most {max} pixels")]
    TooLarge {
        /// Width in pixels.
        width: u32,
        /// Height in pixels.
        height: u32,
        /// The largest supported side.
        max: u32,
    },
    /// A size computation overflowed (a buffer larger than the address space).
    #[error("the size of {what} overflows")]
    SizeOverflow {
        /// What was being sized.
        what: &'static str,
    },
    /// An option has an unusable value.
    #[error("invalid offline option `{option}`: {reason}")]
    InvalidOption {
        /// The option's name.
        option: &'static str,
        /// Why it is rejected.
        reason: String,
    },
    /// The chain reaches so far that tiles of the allowed size have no room left for content.
    #[error(
        "the chain reaches {overlap_px} pixels around each pixel, which leaves no room for tiles of at most {max_tile_px} pixels; \
         reduce the sample radii or allow larger tiles"
    )]
    PaddingTooLarge {
        /// Overlap every tile needs on each side, in pixels.
        overlap_px: u32,
        /// The largest tile (texture) side allowed.
        max_tile_px: u32,
    },
    /// The frame needs tiling (it exceeds the tile size), but the chain cannot be split into
    /// tiles exactly: e.g. a pass with [`Pass::with_scale`](crate::Pass::with_scale) below 1,
    /// whose reduced-resolution pixels would not line up across tile edges. Process the frame
    /// whole: `OfflineOptions::with_max_tile_size(None)` (it must then fit the device's limits).
    #[error("the frame must be tiled, but the chain cannot be tiled exactly: {reason}; process it whole with `OfflineOptions::with_max_tile_size(None)`")]
    NotTileable {
        /// Why the chain cannot be tiled.
        reason: String,
    },
    /// The attached user textures cannot serve the chain: an invalid slot, or an effect reading a
    /// slot with no texture (see [`OfflineOptions::with_texture`](super::OfflineOptions::with_texture)).
    #[error("the chain's textures are unusable: {0}")]
    Texture(#[from] crate::TextureError),
    /// The chain's persistent state cannot run on this device (its format is not renderable /
    /// filterable, [`StateError::UnsupportedFormat`](crate::StateError::UnsupportedFormat)).
    #[error("the chain's state is unusable: {0}")]
    State(#[from] crate::StateError),
    /// A user texture is larger than the device can hold.
    #[error("the texture at slot {slot} is {width}x{height} pixels, but this device allows at most {max} per side")]
    TextureTooLarge {
        /// The slot.
        slot: usize,
        /// Width in pixels.
        width: u32,
        /// Height in pixels.
        height: u32,
        /// The device's `max_texture_dimension_2d`.
        max: u32,
    },
    /// The device rejected an effect's shader or pipeline.
    #[error("effect `{label}` was rejected by the GPU: {message}")]
    Effect {
        /// The effect's label.
        label: String,
        /// The device's message.
        message: String,
    },
    /// The device ran out of memory for the run's textures or buffers. Retry with smaller tiles
    /// ([`OfflineOptions::with_max_tile_size`](super::OfflineOptions::with_max_tile_size)).
    #[error("the GPU ran out of memory during offline processing ({0}); use smaller tiles (`OfflineOptions::with_max_tile_size`)")]
    OutOfMemory(String),
    /// The device rejected the crate's own GPU work.
    #[error("GPU error during offline processing: {0}")]
    Gpu(String),
    /// Mapping the result for reading failed (device lost, out of memory, ...).
    #[error("reading the result back from the GPU failed: {0}")]
    Readback(String),
    /// The output format cannot store the result's transparency.
    #[cfg(feature = "image")]
    #[error("the {format} format has no alpha channel, but the result is not opaque; set a background colour (`OfflineOptions::with_background`) or save to PNG")]
    AlphaNotSupported {
        /// The output format's name.
        format: String,
    },
    /// Reading, decoding, encoding or writing an image failed.
    #[cfg(feature = "image")]
    #[error("image I/O failed: {0}")]
    Image(#[from] image::ImageError),
}
