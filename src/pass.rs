/*
File: pass.rs

Purpose:
The per-frame description of a filter: a `Pass` is an `Effect` plus its 16 `vec4` parameters
(and an optional sample-radius override); a `FilterChain` is an ordered list of passes run
ping-pong. Also owns the byte layout of the `fl` uniform block shared by all backends.

Key structures:
- `Pass`, `FilterChain`, `ParamIndexError` (public).
- `FrameUniforms` (crate-internal): the non-parameter half of the uniform block.

Key functions:
- `FrameUniforms::for_layer()`: the one place the per-layer uniforms are derived from geometry.
- `pack_uniforms()`: serializes one pass's uniforms into the exact std140 / WGSL-uniform layout
  of `FlUniforms` in `shaders/prelude.wgsl` (offsets pinned by a test against naga's layout).
*/

use crate::effect::{Effect, sanitize_radius};
use crate::geometry::{LayerGeometry, px_f32};

/// Number of `vec4<f32>` parameter slots a pass has (`params(0)` .. `params(15)`).
pub const PARAM_SLOTS: usize = 16;

/// The parameter block of one pass: `PARAM_SLOTS` `vec4<f32>` values.
pub type Params = [[f32; 4]; PARAM_SLOTS];

/// A parameter index outside `0..PARAM_SLOTS`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("parameter slot {index} is out of range (a pass has {PARAM_SLOTS} slots, 0..{PARAM_SLOTS})")]
pub struct ParamIndexError {
    /// The rejected index.
    pub index: usize,
}

/// One filter pass: an [`Effect`] and the parameter values it sees this frame.
#[derive(Clone, Debug)]
pub struct Pass {
    /// The effect run by this pass.
    effect: Effect,
    /// Values returned by `params(i)` in WGSL.
    params: Params,
    /// Overrides the effect's default sample radius when set.
    sample_radius_points: Option<f32>,
}

impl Pass {
    /// A pass running `effect` with all parameters zero.
    #[must_use]
    pub fn new(effect: &Effect) -> Self {
        Self { effect: effect.clone(), params: [[0.0; 4]; PARAM_SLOTS], sample_radius_points: None }
    }

    /// Sets parameter slot `index` (read in WGSL as `params(index)`).
    ///
    /// # Errors
    /// [`ParamIndexError`] when `index >= PARAM_SLOTS`.
    pub fn with_param(mut self, index: usize, value: [f32; 4]) -> Result<Self, ParamIndexError> {
        self.set_param(index, value)?;
        Ok(self)
    }

    /// Replaces the whole parameter block.
    #[must_use]
    pub fn with_params(mut self, params: Params) -> Self {
        self.params = params;
        self
    }

    /// Sets parameter slot `index` in place.
    ///
    /// # Errors
    /// [`ParamIndexError`] when `index >= PARAM_SLOTS`.
    pub fn set_param(&mut self, index: usize, value: [f32; 4]) -> Result<(), ParamIndexError> {
        let slot = self.params.get_mut(index).ok_or(ParamIndexError { index })?;
        *slot = value;
        Ok(())
    }

    /// The value of slot `index`, or `None` when out of range.
    #[must_use]
    pub fn param(&self, index: usize) -> Option<[f32; 4]> {
        self.params.get(index).copied()
    }

    /// All parameter slots.
    #[must_use]
    pub fn params(&self) -> &Params {
        &self.params
    }

    /// All parameter slots, mutable (e.g. for a UI editing them in place).
    pub fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }

    /// Overrides the effect's sample radius for this pass (points; negative / non-finite -> 0).
    #[must_use]
    pub fn with_sample_radius(mut self, points: f32) -> Self {
        self.sample_radius_points = Some(sanitize_radius(points));
        self
    }

    /// The sample radius this pass needs, in points: the override if set, else the effect's.
    #[must_use]
    pub fn sample_radius(&self) -> f32 {
        self.sample_radius_points.unwrap_or_else(|| self.effect.sample_radius())
    }

    /// The effect this pass runs.
    #[must_use]
    pub fn effect(&self) -> &Effect {
        &self.effect
    }
}

/// An ordered list of passes. Pass `i` reads the output of pass `i - 1`; the first reads the
/// captured backdrop.
#[derive(Clone, Debug, Default)]
pub struct FilterChain {
    /// The passes, in execution order.
    passes: Vec<Pass>,
}

impl FilterChain {
    /// An empty chain (painting it does nothing).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a pass or a whole chain.
    #[must_use]
    pub fn then(mut self, next: impl Into<FilterChain>) -> Self {
        self.passes.extend(next.into().passes);
        self
    }

    /// Appends one pass in place.
    pub fn push(&mut self, pass: Pass) {
        self.passes.push(pass);
    }

    /// The passes in execution order.
    #[must_use]
    pub fn passes(&self) -> &[Pass] {
        &self.passes
    }

    /// The passes, mutable (to edit parameters between frames).
    pub fn passes_mut(&mut self) -> &mut [Pass] {
        &mut self.passes
    }

    /// Whether the chain has no passes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.passes.is_empty()
    }

    /// Number of passes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.passes.len()
    }

    /// Capture padding the chain needs, in points: the SUM of the passes' sample radii, because
    /// every neighbourhood pass widens the region the next one depends on.
    #[must_use]
    pub fn padding_points(&self) -> f32 {
        self.passes.iter().map(Pass::sample_radius).sum()
    }

    /// Whether any pass runs an animated effect.
    #[must_use]
    pub fn is_animated(&self) -> bool {
        self.passes.iter().any(|pass| pass.effect().is_animated())
    }

    /// Consumes the chain into its passes.
    pub(crate) fn into_passes(self) -> Vec<Pass> {
        self.passes
    }
}

impl From<Pass> for FilterChain {
    fn from(pass: Pass) -> Self {
        Self { passes: vec![pass] }
    }
}

impl From<Vec<Pass>> for FilterChain {
    fn from(passes: Vec<Pass>) -> Self {
        Self { passes }
    }
}

impl FromIterator<Pass> for FilterChain {
    fn from_iter<I: IntoIterator<Item = Pass>>(iter: I) -> Self {
        Self { passes: iter.into_iter().collect() }
    }
}

impl Extend<Pass> for FilterChain {
    fn extend<I: IntoIterator<Item = Pass>>(&mut self, iter: I) {
        self.passes.extend(iter);
    }
}

/// Size in bytes of the `FlUniforms` block (see `shaders/prelude.wgsl`).
pub(crate) const UNIFORM_BLOCK_SIZE: usize = 352;

/// Byte offsets of the `FlUniforms` members; pinned against naga's computed layout by a test.
pub(crate) mod offsets {
    /// `rect_uv: vec4<f32>`.
    pub const RECT_UV: usize = 0;
    /// `texel: vec2<f32>`.
    pub const TEXEL: usize = 16;
    /// `size_px: vec2<f32>`.
    pub const SIZE_PX: usize = 24;
    /// `time: f32`.
    pub const TIME: usize = 32;
    /// `pixels_per_point: f32`.
    pub const PIXELS_PER_POINT: usize = 36;
    /// `internal_flip_y: f32`.
    pub const FLIP_Y: usize = 40;
    /// `internal_frag_mode: f32`.
    pub const FRAG_MODE: usize = 44;
    /// `internal_src_scale: vec2<f32>`.
    pub const SRC_SCALE: usize = 48;
    /// `internal_aux_scale: vec2<f32>`.
    pub const AUX_SCALE: usize = 56;
    /// `internal_frag_offset: vec2<f32>`.
    pub const FRAG_OFFSET: usize = 64;
    /// `internal_src_offset: vec2<f32>`.
    pub const SRC_OFFSET: usize = 72;
    /// `internal_aux_offset: vec2<f32>`.
    pub const AUX_OFFSET: usize = 80;
    /// `params: array<vec4<f32>, 16>`.
    pub const PARAMS: usize = 96;
}

/// The non-parameter uniforms of one pass.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct FrameUniforms {
    /// The layer's full rect inside the capture, in capture UV (`x, y, w, h`; may exceed 0..1).
    pub rect_uv: [f32; 4],
    /// Size of one physical pixel in capture UV.
    pub texel: [f32; 2],
    /// Capture size in physical pixels.
    pub size_px: [f32; 2],
    /// Seconds (egui input time).
    pub time: f32,
    /// Physical pixels per layer point (egui's pixels-per-point times the layer transform scale).
    pub pixels_per_point: f32,
    /// Whether texture storage is bottom-up (OpenGL) and `src()` must flip y.
    pub flip_y: bool,
    /// Capture size / allocated size of the `src()` texture (grow-only intermediates; with a
    /// window, the WHOLE capture's size over the window texture's allocation).
    pub src_scale: [f32; 2],
    /// Same for the auxiliary (original capture) texture of the composite pass.
    pub aux_scale: [f32; 2],
    /// Whether the fragment's capture uv comes from its pixel position (wgpu) instead of the
    /// viewport-interpolated uv (OpenGL); see `shaders/prelude.wgsl`.
    pub frag_from_position: bool,
    /// Position mode: added to the fragment position (target pixels) to get capture pixels.
    pub frag_offset: [f32; 2],
    /// The `src()` texture's window origin in its storage uv (subtracted after `src_scale`).
    pub src_offset: [f32; 2],
    /// Same for the auxiliary texture.
    pub aux_offset: [f32; 2],
}

impl FrameUniforms {
    /// The uniforms every pass of one layer shares, derived from its `geometry`: `rect_uv`,
    /// `texel` / `size_px` from the capture size, and `fl.pixels_per_point` =
    /// `geometry.pixels_per_layer_point` (physical pixels per LAYER point, so effect distances in
    /// points follow a layer transform's zoom exactly like the padding does). `src_scale` /
    /// `aux_scale` start at `1` and are set per pass by the backend (grow-only targets); the
    /// window fields (`frag_*`, `*_offset`) start in the interpolated, whole-capture state.
    ///
    /// `time` is seconds; `flip_y` whether the backend's texture storage is bottom-up (OpenGL).
    #[must_use]
    pub(crate) fn for_layer(geometry: &LayerGeometry, time: f32, flip_y: bool) -> Self {
        let [w, h] = [px_f32(geometry.capture.w), px_f32(geometry.capture.h)];
        Self {
            rect_uv: geometry.rect_uv,
            texel: [1.0 / w, 1.0 / h],
            size_px: [w, h],
            time,
            pixels_per_point: geometry.pixels_per_layer_point,
            flip_y,
            src_scale: [1.0, 1.0],
            aux_scale: [1.0, 1.0],
            frag_from_position: false,
            frag_offset: [0.0, 0.0],
            src_offset: [0.0, 0.0],
            aux_offset: [0.0, 0.0],
        }
    }
}

/// Writes `values` as native-endian `f32`s at `offset`.
fn put_f32s(bytes: &mut [u8; UNIFORM_BLOCK_SIZE], offset: usize, values: &[f32]) {
    for (i, value) in values.iter().enumerate() {
        let start = offset + i * 4;
        if let Some(chunk) = bytes.get_mut(start..start + 4) {
            chunk.copy_from_slice(&value.to_ne_bytes());
        }
    }
}

/// Serializes one pass's uniforms into the `FlUniforms` byte layout (native endianness, as the
/// GPU APIs expect for host-written buffers).
#[must_use]
pub(crate) fn pack_uniforms(frame: &FrameUniforms, params: &Params) -> [u8; UNIFORM_BLOCK_SIZE] {
    let mut bytes = [0_u8; UNIFORM_BLOCK_SIZE];
    put_f32s(&mut bytes, offsets::RECT_UV, &frame.rect_uv);
    put_f32s(&mut bytes, offsets::TEXEL, &frame.texel);
    put_f32s(&mut bytes, offsets::SIZE_PX, &frame.size_px);
    put_f32s(&mut bytes, offsets::TIME, &[frame.time]);
    put_f32s(&mut bytes, offsets::PIXELS_PER_POINT, &[frame.pixels_per_point]);
    put_f32s(&mut bytes, offsets::FLIP_Y, &[if frame.flip_y { 1.0 } else { 0.0 }]);
    put_f32s(&mut bytes, offsets::SRC_SCALE, &frame.src_scale);
    put_f32s(&mut bytes, offsets::AUX_SCALE, &frame.aux_scale);
    put_f32s(&mut bytes, offsets::FRAG_MODE, &[if frame.frag_from_position { 1.0 } else { 0.0 }]);
    put_f32s(&mut bytes, offsets::FRAG_OFFSET, &frame.frag_offset);
    put_f32s(&mut bytes, offsets::SRC_OFFSET, &frame.src_offset);
    put_f32s(&mut bytes, offsets::AUX_OFFSET, &frame.aux_offset);
    for (i, slot) in params.iter().enumerate() {
        put_f32s(&mut bytes, offsets::PARAMS + i * 16, slot);
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::effect::PRELUDE_WGSL;

    fn read_f32(bytes: &[u8], offset: usize) -> f32 {
        let mut raw = [0_u8; 4];
        raw.copy_from_slice(&bytes[offset..offset + 4]);
        f32::from_ne_bytes(raw)
    }

    /// The Rust-side offsets must equal what naga computes for `FlUniforms`; a drift would feed
    /// every shader garbage on every backend.
    #[test]
    fn offsets_match_the_wgsl_struct_layout() {
        let source = format!("{PRELUDE_WGSL}fn effect(uv: vec2<f32>) -> vec4<f32> {{ return src(uv); }}\n");
        let module = naga::front::wgsl::parse_str(&source).map_err(|e| e.emit_to_string(&source));
        let Ok(module) = module else { panic!("{module:?}") };
        let (_, ty) = module.types.iter().find(|(_, ty)| ty.name.as_deref() == Some("FlUniforms")).unwrap_or_else(|| panic!("FlUniforms missing"));
        let naga::TypeInner::Struct { members, span } = &ty.inner else { panic!("FlUniforms is not a struct") };
        let offset_of = |name: &str| {
            let member = members.iter().find(|m| m.name.as_deref() == Some(name)).unwrap_or_else(|| panic!("member {name} missing"));
            usize::try_from(member.offset).unwrap_or(usize::MAX)
        };
        assert_eq!(offset_of("rect_uv"), offsets::RECT_UV);
        assert_eq!(offset_of("texel"), offsets::TEXEL);
        assert_eq!(offset_of("size_px"), offsets::SIZE_PX);
        assert_eq!(offset_of("time"), offsets::TIME);
        assert_eq!(offset_of("pixels_per_point"), offsets::PIXELS_PER_POINT);
        assert_eq!(offset_of("internal_flip_y"), offsets::FLIP_Y);
        assert_eq!(offset_of("internal_src_scale"), offsets::SRC_SCALE);
        assert_eq!(offset_of("internal_aux_scale"), offsets::AUX_SCALE);
        assert_eq!(offset_of("internal_frag_mode"), offsets::FRAG_MODE);
        assert_eq!(offset_of("internal_frag_offset"), offsets::FRAG_OFFSET);
        assert_eq!(offset_of("internal_src_offset"), offsets::SRC_OFFSET);
        assert_eq!(offset_of("internal_aux_offset"), offsets::AUX_OFFSET);
        assert_eq!(offset_of("params"), offsets::PARAMS);
        assert_eq!(usize::try_from(*span).unwrap_or(0), UNIFORM_BLOCK_SIZE);
    }

    #[test]
    fn packing_places_every_field() {
        let frame = FrameUniforms {
            rect_uv: [0.1, 0.2, 0.3, 0.4],
            texel: [0.5, 0.25],
            size_px: [2.0, 4.0],
            time: 7.5,
            pixels_per_point: 1.5,
            flip_y: true,
            src_scale: [0.75, 0.5],
            aux_scale: [0.125, 1.0],
            frag_from_position: true,
            frag_offset: [-3.0, 5.0],
            src_offset: [0.25, 0.5],
            aux_offset: [0.0625, 2.0],
        };
        let mut params = [[0.0; 4]; PARAM_SLOTS];
        params[0] = [1.0, 2.0, 3.0, 4.0];
        params[15] = [-1.0, -2.0, -3.0, -4.0];
        let bytes = pack_uniforms(&frame, &params);
        assert!((read_f32(&bytes, offsets::RECT_UV + 12) - 0.4).abs() < f32::EPSILON);
        assert!((read_f32(&bytes, offsets::TEXEL + 4) - 0.25).abs() < f32::EPSILON);
        assert!((read_f32(&bytes, offsets::SIZE_PX) - 2.0).abs() < f32::EPSILON);
        assert!((read_f32(&bytes, offsets::TIME) - 7.5).abs() < f32::EPSILON);
        assert!((read_f32(&bytes, offsets::PIXELS_PER_POINT) - 1.5).abs() < f32::EPSILON);
        assert!((read_f32(&bytes, offsets::FLIP_Y) - 1.0).abs() < f32::EPSILON);
        assert!((read_f32(&bytes, offsets::SRC_SCALE + 4) - 0.5).abs() < f32::EPSILON);
        assert!((read_f32(&bytes, offsets::AUX_SCALE) - 0.125).abs() < f32::EPSILON);
        assert!((read_f32(&bytes, offsets::PARAMS + 8) - 3.0).abs() < f32::EPSILON);
        assert!((read_f32(&bytes, offsets::PARAMS + 15 * 16 + 12) + 4.0).abs() < f32::EPSILON);
        assert!((read_f32(&bytes, offsets::FRAG_MODE) - 1.0).abs() < f32::EPSILON);
        assert!((read_f32(&bytes, offsets::FRAG_OFFSET) + 3.0).abs() < f32::EPSILON);
        assert!((read_f32(&bytes, offsets::SRC_OFFSET + 4) - 0.5).abs() < f32::EPSILON);
        assert!((read_f32(&bytes, offsets::AUX_OFFSET + 4) - 2.0).abs() < f32::EPSILON);
        assert!(read_f32(&bytes, offsets::AUX_OFFSET + 8).abs() < f32::EPSILON, "reserved slot stays zero");
    }

    #[test]
    fn layer_uniforms_expose_pixels_per_layer_point() {
        use egui::{CornerRadius, Rect, pos2, vec2};
        let rect = Rect::from_min_size(pos2(100.0, 100.0), vec2(100.0, 50.0));
        for (scale, expected) in [(1.0, 2.0), (0.5, 1.0), (2.0, 4.0)] {
            let Some(geometry) = crate::geometry::layer_geometry(rect, Rect::EVERYTHING, 2.0, [800, 600], 0.0, CornerRadius::ZERO, scale) else { panic!("visible") };
            let frame = FrameUniforms::for_layer(&geometry, 3.0, false);
            assert!((frame.pixels_per_point - expected).abs() < f32::EPSILON, "scale {scale}: {}", frame.pixels_per_point);
            assert!((frame.size_px[0] - 200.0).abs() < f32::EPSILON && (frame.texel[1] - 0.01).abs() < 1e-6, "{frame:?}");
            assert!((frame.time - 3.0).abs() < f32::EPSILON && !frame.flip_y);
        }
    }

    #[test]
    fn param_indices_are_checked() {
        let Ok(effect) = Effect::new("id", "fn effect(uv: vec2<f32>) -> vec4<f32> { return src(uv); }") else { panic!("identity must compile") };
        let pass = Pass::new(&effect);
        assert!(pass.clone().with_param(15, [1.0; 4]).is_ok());
        assert_eq!(pass.clone().with_param(16, [1.0; 4]).err(), Some(ParamIndexError { index: 16 }));
        assert_eq!(pass.param(16), None);
    }

    #[test]
    fn chain_padding_sums_the_pass_radii_and_overrides_win() {
        let Ok(effect) = Effect::new("id", "fn effect(uv: vec2<f32>) -> vec4<f32> { return src(uv); }") else { panic!("identity must compile") };
        let effect = effect.with_sample_radius(2.0);
        let chain = FilterChain::new().then(Pass::new(&effect)).then(Pass::new(&effect).with_sample_radius(5.0));
        assert_eq!(chain.len(), 2);
        assert!((chain.padding_points() - 7.0).abs() < f32::EPSILON);
        assert!(!chain.is_animated());
        assert!(chain.clone().then(Pass::new(&effect.animated())).is_animated());
    }
}
