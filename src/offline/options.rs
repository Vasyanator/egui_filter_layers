/*
File: offline/options.rs

Purpose:
`OfflineOptions`, the settings of one offline run, and `PointerState`, the pointer an offline run
shows its effects.

Key structures:
- `OfflineOptions` (public): pixels per point, time, per-frame input (`delta_time`, `frame`,
  pointer), background, mask (corner radius or a shape, F9), output alpha, tile size limit, user
  textures (F8), persistent state options (F7).
- `PointerState` (public): pointer position in frame uv + pressed.

Key functions:
- `OfflineOptions::validate()`: numeric checks, run before any GPU work.
- `OfflineOptions::input_uniforms()`: `fl.delta_time` / `fl.frame` / `fl.pointer` of a run, the
  pointer through the UI's `geometry::pointer_uniform`.

Notes:
`delta_time` and `frame` remember whether they were set: a video processor fills unset ones from
its own frame times / index, and an explicit value wins.
*/

use egui::{Color32, CornerRadius, pos2};

use crate::mask::Mask;

use super::{AlphaMode, DEFAULT_MAX_TILE_SIZE, OfflineError, OfflineTexture};
use crate::backend::TextureError;
use crate::effect::TEXTURE_SLOTS;
use crate::geometry::{LayerGeometry, pointer_uniform, px_f32};
use crate::pass::InputUniforms;
use crate::state::StateOptions;

/// Settings of one offline run. Build with [`OfflineOptions::default`] and the `with_*` methods.
#[derive(Clone, Debug, PartialEq)]
pub struct OfflineOptions {
    /// Image pixels per layer point (see [`OfflineOptions::with_pixels_per_point`]).
    pub(super) pixels_per_point: f32,
    /// `fl.time`, seconds.
    pub(super) time: f32,
    /// Colour the frame is drawn over before filtering; `None` = transparent.
    pub(super) background: Option<Color32>,
    /// The layer's mask (rounded rect by default; a shape in layer points, F9).
    pub(super) mask: Mask,
    /// Alpha convention of the result.
    pub(super) output_alpha: AlphaMode,
    /// Largest texture side used; `None` = the device limit (see [`DEFAULT_MAX_TILE_SIZE`]).
    pub(super) max_tile_size: Option<u32>,
    /// `fl.delta_time`, seconds; `None` = not set (0, or the video processor's own value).
    delta_time: Option<f32>,
    /// `fl.frame`; `None` = not set (0, or the video processor's frame index).
    frame: Option<u32>,
    /// `fl.pointer`; `None` = no pointer (`fl.pointer.w = 0`).
    pointer: Option<PointerState>,
    /// The user texture attached at each slot (`tex(i, uv)`).
    textures: [Option<OfflineTexture>; TEXTURE_SLOTS],
    /// The first out-of-range slot given to `with_texture`, reported by `validate`.
    invalid_texture_slot: Option<usize>,
    /// How a stateful chain's state is stored (resolution, format).
    pub(super) state: StateOptions,
}

/// The pointer an offline run shows its effects (`fl.pointer`), in the frame's uv: `0..1` across
/// the frame, top-left origin, y down — the effects' `uv` space (the frame is the layer and the
/// capture). Values outside `0..1` are a pointer off the frame.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct PointerState {
    /// Position in frame uv.
    pub uv: [f32; 2],
    /// Whether the primary button is down (`fl.pointer.z`).
    pub pressed: bool,
}

impl PointerState {
    /// The pointer at `uv` (frame uv), with the primary button `pressed` or not. Validated by
    /// `OfflineOptions::validate` (a non-finite position is refused).
    #[must_use]
    pub fn new(uv: [f32; 2], pressed: bool) -> Self {
        Self { uv, pressed }
    }
}

impl Default for OfflineOptions {
    fn default() -> Self {
        Self {
            pixels_per_point: 1.0,
            time: 0.0,
            background: None,
            mask: Mask::default(),
            output_alpha: AlphaMode::Straight,
            max_tile_size: Some(DEFAULT_MAX_TILE_SIZE),
            delta_time: None,
            frame: None,
            pointer: None,
            textures: [const { None }; TEXTURE_SLOTS],
            invalid_texture_slot: None,
            state: StateOptions::default(),
        }
    }
}

impl OfflineOptions {
    /// Image pixels per layer point (default 1): how many frame pixels one layer point spans, which
    /// converts effect distances, sample radii and corner radii (all in points) to pixels. The
    /// value that makes the export look like the editor depends on how the editor zooms:
    ///
    /// * The image is painted into a larger or smaller rect (the layer covers that rect, no layer
    ///   transform): one image pixel spans `N` screen pixels at egui scale factor `ppp`, and a
    ///   point spans `ppp` screen pixels, so the match is `ppp / N` (bit-exact only for `N = 1`).
    /// * The zoom is a layer transform (e.g. `egui::Scene`), with the image painted 1:1 in layer
    ///   points: distances scale with the transform together with the image, so the match is the
    ///   UI's `ppp` itself, whatever the zoom.
    #[must_use]
    pub fn with_pixels_per_point(mut self, pixels_per_point: f32) -> Self {
        self.pixels_per_point = pixels_per_point;
        self
    }

    /// `fl.time` in seconds (default 0): the frame's presentation time for a video.
    #[must_use]
    pub fn with_time(mut self, seconds: f32) -> Self {
        self.time = seconds;
        self
    }

    /// `fl.delta_time` in seconds (default 0): the time since the previous frame, finite and
    /// `>= 0`. A `VideoProcessor` (feature `video`) derives it from the
    /// frame times itself; setting it here overrides that for every frame.
    #[must_use]
    pub fn with_delta_time(mut self, seconds: f32) -> Self {
        self.delta_time = Some(seconds);
        self
    }

    /// `fl.frame` (default 0). A `VideoProcessor` (feature `video`) counts
    /// its frames itself; setting it here overrides that for every frame.
    #[must_use]
    pub fn with_frame(mut self, frame: u32) -> Self {
        self.frame = Some(frame);
        self
    }

    /// `fl.pointer` (default `None`: no pointer, `fl.pointer.w = 0`). A video processor never
    /// sets it; set it per frame (`VideoProcessor::set_options`) to replay pointer input.
    #[must_use]
    pub fn with_pointer(mut self, pointer: Option<PointerState>) -> Self {
        self.pointer = pointer;
        self
    }

    /// Attaches `texture` at user texture `slot` (`0..TEXTURE_SLOTS`), which effects read with
    /// `tex(slot, uv)`, `tex_size(slot)` and `tex_load(slot, px)` — the offline counterpart of
    /// [`FilterLayer::texture`](crate::FilterLayer::texture). The renderer uploads it once and
    /// reuses the upload while the same texture (a clone) stays attached, so a video passes the
    /// same one with every frame at no cost. A `slot` out of range makes the run fail with
    /// [`OfflineError::Texture`].
    #[must_use]
    pub fn with_texture(mut self, slot: usize, texture: OfflineTexture) -> Self {
        match self.textures.get_mut(slot) {
            Some(entry) => *entry = Some(texture),
            None => {
                self.invalid_texture_slot.get_or_insert(slot);
            }
        }
        self
    }

    /// How the persistent state of a chain with a state pass is stored (default: full
    /// resolution, RGBA16F) — the offline counterpart of
    /// [`FilterLayer::state`](crate::FilterLayer::state). The renderer carries the state from
    /// one run to the next; a changed format starts it over.
    #[must_use]
    pub fn with_state(mut self, state: StateOptions) -> Self {
        self.state = state;
        self
    }

    /// The state options.
    #[must_use]
    pub fn state(&self) -> StateOptions {
        self.state
    }

    /// The texture attached at `slot`, if any.
    #[must_use]
    pub fn texture(&self, slot: usize) -> Option<&OfflineTexture> {
        self.textures.get(slot).and_then(Option::as_ref)
    }

    /// Which user texture slots have a texture.
    pub(super) fn attached_textures(&self) -> [bool; TEXTURE_SLOTS] {
        self.textures.each_ref().map(Option::is_some)
    }

    /// Draws the frame over `color` before filtering (default: transparent). Match the UI's
    /// backdrop (canvas colour) for translucent images, or give an opaque colour to get an
    /// opaque result for formats without alpha.
    #[must_use]
    pub fn with_background(mut self, color: Option<Color32>) -> Self {
        self.background = color;
        self
    }

    /// Rounds the layer's corners (layer points), like [`FilterLayer::corner_radius`](crate::FilterLayer::corner_radius):
    /// outside the mask the unfiltered frame is kept. Shorthand for
    /// `with_mask(Mask::RoundedRect(radius))`, replacing any earlier mask.
    #[must_use]
    pub fn with_corner_radius(mut self, radius: impl Into<CornerRadius>) -> Self {
        self.mask = Mask::RoundedRect(radius.into());
        self
    }

    /// The layer's mask, like [`FilterLayer::mask`](crate::FilterLayer::mask): a shape is in
    /// layer points (the frame spans `0..width / pixels_per_point` points), tessellated per run at
    /// the options' pixels per point with egui's default tessellation options. Offline masks must
    /// pass [`Mask::check_offline`]: no text (no font atlas offline) and only untextured meshes;
    /// otherwise the run fails with [`OfflineError::Mask`] before any GPU work.
    #[must_use]
    pub fn with_mask(mut self, mask: Mask) -> Self {
        self.mask = mask;
        self
    }

    /// Alpha convention of the result (default [`AlphaMode::Straight`], what image and video
    /// encoders expect; the chain itself produces premultiplied colour).
    #[must_use]
    pub fn with_output_alpha(mut self, alpha: AlphaMode) -> Self {
        self.output_alpha = alpha;
        self
    }

    /// Caps the texture side of a run (default [`DEFAULT_MAX_TILE_SIZE`]; `None` = the device's
    /// `max_texture_dimension_2d`). The side is further limited by the device (texture limit, and
    /// `max_buffer_size` for the readback). Larger frames are processed in overlapping tiles:
    /// identical to one whole-frame run for effects sampling at texel centres (every preset
    /// except `ripple`, `lens` and `gaussian_blur` above `sigma * pixels_per_point = 32 / 3`);
    /// effects sampling between texel centres may differ by at most two 8-bit steps on a few
    /// values, because hardware bilinear weights depend on the texture size. Pass `None` for a
    /// frame the UI shows untiled (up to the UI device's texture limit) when that matters more
    /// than bounded GPU memory.
    #[must_use]
    pub fn with_max_tile_size(mut self, pixels: Option<u32>) -> Self {
        self.max_tile_size = pixels;
        self
    }

    /// Image pixels per layer point.
    #[must_use]
    pub fn pixels_per_point(&self) -> f32 {
        self.pixels_per_point
    }

    /// `fl.time` in seconds.
    #[must_use]
    pub fn time(&self) -> f32 {
        self.time
    }

    /// `fl.delta_time` in seconds (0 unless set).
    #[must_use]
    pub fn delta_time(&self) -> f32 {
        self.delta_time.unwrap_or(0.0)
    }

    /// `fl.frame` (0 unless set).
    #[must_use]
    pub fn frame(&self) -> u32 {
        self.frame.unwrap_or(0)
    }

    /// `fl.pointer`, if any.
    #[must_use]
    pub fn pointer(&self) -> Option<PointerState> {
        self.pointer
    }

    /// Whether `fl.delta_time` / `fl.frame` were set explicitly (they then win over a video
    /// processor's own values).
    pub(super) fn explicit_counters(&self) -> (bool, bool) {
        (self.delta_time.is_some(), self.frame.is_some())
    }

    /// The input members of the uniform block of a run over `geometry` (the frame's own: the
    /// capture is the frame). The pointer goes through the UI's conversion
    /// (`geometry::pointer_uniform`) on a screen of one "point" per frame pixel.
    pub(super) fn input_uniforms(&self, geometry: &LayerGeometry) -> InputUniforms {
        let [w, h] = [px_f32(geometry.capture.w), px_f32(geometry.capture.h)];
        let pointer = self.pointer.map(|pointer| (pos2(pointer.uv[0] * w, pointer.uv[1] * h), pointer.pressed));
        InputUniforms { delta_time: self.delta_time(), frame: self.frame(), pointer: pointer_uniform(pointer, 1.0, geometry.capture) }
    }

    /// Background colour, if any.
    #[must_use]
    pub fn background(&self) -> Option<Color32> {
        self.background
    }

    /// Checks the numeric options and the texture slots.
    pub(super) fn validate(&self) -> Result<(), OfflineError> {
        if let Some(slot) = self.invalid_texture_slot {
            return Err(OfflineError::Texture(TextureError::InvalidSlot { slot }));
        }
        self.mask.check_offline()?;
        if !(self.pixels_per_point.is_finite() && self.pixels_per_point > 0.0) {
            return Err(OfflineError::InvalidOption { option: "pixels_per_point", reason: format!("must be finite and positive, got {}", self.pixels_per_point) });
        }
        if !self.time.is_finite() {
            return Err(OfflineError::InvalidOption { option: "time", reason: format!("must be finite, got {}", self.time) });
        }
        if let Some(delta_time) = self.delta_time
            && !(delta_time.is_finite() && delta_time >= 0.0)
        {
            return Err(OfflineError::InvalidOption { option: "delta_time", reason: format!("must be finite and non-negative, got {delta_time}") });
        }
        if let Some(pointer) = self.pointer
            && !pointer.uv.iter().all(|value| value.is_finite())
        {
            return Err(OfflineError::InvalidOption { option: "pointer", reason: format!("the pointer uv must be finite, got {:?}", pointer.uv) });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{PxRect, layer_geometry};
    use egui::{Rect, vec2};

    /// F9: masks offline are checked with `Mask::check_offline` before any GPU work.
    #[test]
    fn offline_masks_are_validated() {
        let circle = egui::Shape::circle_filled(pos2(5.0, 5.0), 3.0, Color32::WHITE);
        assert!(OfflineOptions::default().with_mask(Mask::Shape(circle.clone())).validate().is_ok());
        let callback = egui::Shape::Callback(egui::PaintCallback { rect: Rect::EVERYTHING, callback: std::sync::Arc::new(()) });
        let options = OfflineOptions::default().with_mask(Mask::Shape(egui::Shape::Vec(vec![circle, callback])));
        assert!(matches!(options.validate(), Err(OfflineError::Mask(crate::MaskError::Callback))));
        assert_eq!(options.with_corner_radius(3).mask, Mask::RoundedRect(CornerRadius::same(3)), "the corner radius replaces the mask");
    }

    #[test]
    fn frame_input_options_are_validated_and_resolved() {
        assert!(OfflineOptions::default().with_delta_time(0.0).with_frame(3).validate().is_ok());
        assert!(matches!(OfflineOptions::default().with_delta_time(-0.1).validate(), Err(OfflineError::InvalidOption { option: "delta_time", .. })));
        assert!(matches!(OfflineOptions::default().with_delta_time(f32::NAN).validate(), Err(OfflineError::InvalidOption { option: "delta_time", .. })));
        let bad_pointer = Some(PointerState::new([f32::INFINITY, 0.0], false));
        assert!(matches!(OfflineOptions::default().with_pointer(bad_pointer).validate(), Err(OfflineError::InvalidOption { option: "pointer", .. })));

        // A 200x100 frame at 2 pixels per point: the capture is the frame, whatever the scale.
        let Some(geometry) = layer_geometry(Rect::from_min_size(pos2(0.0, 0.0), vec2(100.0, 50.0)), Rect::EVERYTHING, 2.0, [200, 100], 0.0, CornerRadius::ZERO, 1.0) else { panic!("visible") };
        assert_eq!(geometry.capture, PxRect { x: 0, y: 0, w: 200, h: 100 });
        let neutral = OfflineOptions::default().input_uniforms(&geometry);
        assert_eq!(neutral, InputUniforms::default());
        let options = OfflineOptions::default().with_delta_time(0.5).with_frame(4).with_pointer(Some(PointerState::new([0.25, 0.75], true)));
        let input = options.input_uniforms(&geometry);
        assert_eq!((input.frame, input.delta_time.to_bits()), (4, 0.5_f32.to_bits()));
        assert_eq!(input.pointer.map(f32::to_bits), [0.25_f32, 0.75, 1.0, 1.0].map(f32::to_bits), "frame uv is capture uv");
        assert_eq!(options.explicit_counters(), (true, true));
        assert_eq!(OfflineOptions::default().explicit_counters(), (false, false));
    }
}
