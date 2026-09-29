/*
File: pass.rs

Purpose:
The per-frame description of a filter: a `Pass` is an `Effect` plus its 16 anonymous `vec4`
parameters, its named-parameter block (`struct Params`, typed access by member name), an
optional sample-radius override and its output scale (`with_scale`, reduced-resolution passes); a `FilterChain` is an ordered list of passes run ping-pong. Also
owns the byte layout of the `fl` uniform block shared by all backends.

Key structures:
- `Pass`, `FilterChain`, `ParamIndexError` (public).
- `FrameUniforms` (crate-internal): the non-parameter half of the uniform block.
- `InputUniforms` (crate-internal): the per-frame input members (`fl.delta_time`, `fl.frame`,
  `fl.pointer`) of one run, already resolved for its capture.

Key functions:
- `Pass::set()` / `set_slice()` / `get()` / `get_slice()`: typed named parameters, checked
  against the effect's `ParamLayout` (unknown name, type or length -> `ParamError`); the bytes
  are stored in the effect's layout, ready for binding 4 (`plan::params_block`).
- `FrameUniforms::for_layer()`: the one place the per-layer uniforms are derived from geometry;
  `FrameUniforms::with_input()` adds the per-frame input members.
- `pack_uniforms()`: serializes one draw's uniforms into the exact std140 / WGSL-uniform layout
  of `FlUniforms` in `shaders/prelude.wgsl` (offsets pinned by a test against naga's layout).
  Which texture mapping and fragment offset each draw gets is decided by `plan.rs`, not here.
*/

use crate::effect::{Effect, ParamError, ParamInfo, ParamValue, element_range, sanitize_radius};
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
///
/// Two kinds of parameters, usable side by side: the untyped block of [`PARAM_SLOTS`] `vec4`s
/// (`params(i)` in WGSL; [`Pass::with_param`]) and — preferred — the typed members of the
/// effect's `struct Params` (`p.<name>` in WGSL; [`Pass::set`], [`Pass::with`]).
#[derive(Clone, Debug)]
pub struct Pass {
    /// The effect run by this pass.
    effect: Effect,
    /// Values returned by `params(i)` in WGSL.
    params: Params,
    /// The effect's `struct Params` block (`p` in WGSL), laid out by `ParamLayout`; its length is
    /// the layout's padded `block_size` (empty when the effect declares no named parameters).
    named: Vec<u8>,
    /// Overrides the effect's default sample radius when set.
    sample_radius_points: Option<f32>,
    /// Output resolution relative to the capture, in `[MIN_PASS_SCALE, 1]` (`1` = full).
    scale: f32,
}

/// The smallest output scale a pass accepts ([`Pass::with_scale`] clamps to it): 1/1024 of the
/// capture per axis (a scaled output is never smaller than 1 pixel anyway).
pub const MIN_PASS_SCALE: f32 = 1.0 / 1024.0;

impl Pass {
    /// A pass running `effect` with all parameters (anonymous and named) zero.
    #[must_use]
    pub fn new(effect: &Effect) -> Self {
        Self { effect: effect.clone(), params: [[0.0; 4]; PARAM_SLOTS], named: vec![0; effect.param_layout().block_size], sample_radius_points: None, scale: 1.0 }
    }

    /// Runs this pass at a reduced output resolution: it renders `ceil(capture * scale)` pixels
    /// per axis (e.g. `0.5` = a quarter of the pixels), and the next pass (or the final composite,
    /// which upsamples bilinearly) reads that smaller image. `uv`, `src()`, `original()`,
    /// `fl.texel`, `fl.size_px` and `fl.pixels_per_point` keep their capture meaning, so distance
    /// maths is unchanged; `src_px()` / `src_texel()` step one pixel of the pass's INPUT and
    /// `out_texel()` is one pixel of its own output (`fl.pass_scale` = this value).
    ///
    /// `scale` is clamped to `[MIN_PASS_SCALE, 1]`; a non-finite value means `1`. A scaled pass
    /// resamples: when this matters at the layer edge, its sample radius should include about
    /// `1 / scale` capture pixels (the kawase / bloom presets declare theirs). Offline, a chain
    /// with a scaled pass refuses tiling (`OfflineError::NotTileable`).
    #[must_use]
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.set_scale(scale);
        self
    }

    /// [`Pass::with_scale`] in place.
    pub fn set_scale(&mut self, scale: f32) {
        self.scale = if scale.is_finite() { scale.clamp(MIN_PASS_SCALE, 1.0) } else { 1.0 };
    }

    /// The pass's output scale (`1` = full resolution), see [`Pass::with_scale`].
    #[must_use]
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Whether the pass renders below full resolution.
    #[must_use]
    pub(crate) fn is_scaled(&self) -> bool {
        self.scale < 1.0
    }

    /// The named parameter `name` of this pass's effect.
    fn named_field(&self, name: &str) -> Result<&ParamInfo, ParamError> {
        self.effect.param(name).ok_or_else(|| ParamError::Unknown { effect: self.effect.label().to_owned(), name: name.to_owned() })
    }

    /// Sets the named parameter `name` (a member of the effect's `struct Params`, read in WGSL as
    /// `p.name`). The value's type must be the member's exactly: `1.0` for `f32`, `1` for `i32`,
    /// `1u32` for `u32`, `[f32; N]` for `vecN<f32>`, `[[f32; 4]; 4]` (columns) for `mat4x4<f32>`.
    ///
    /// # Errors
    /// [`ParamError::Unknown`] (no such member), [`ParamError::Type`] (other type),
    /// [`ParamError::Array`] (an array member: use [`Pass::set_slice`]). The pass is unchanged.
    pub fn set(&mut self, name: &str, value: impl Into<ParamValue>) -> Result<(), ParamError> {
        let value = value.into();
        let field = self.named_field(name)?;
        if field.is_array {
            return Err(ParamError::Array { name: name.to_owned(), count: field.count });
        }
        if field.ty != value.ty() {
            return Err(ParamError::Type { name: name.to_owned(), expected: field.ty, given: value.ty() });
        }
        if let Some(range) = element_range(field, 0, self.named.len()) {
            value.write(&mut self.named[range]);
        }
        Ok(())
    }

    /// [`Pass::set`] by value, for building passes in one expression.
    ///
    /// # Errors
    /// As [`Pass::set`].
    pub fn with(mut self, name: &str, value: impl Into<ParamValue>) -> Result<Self, ParamError> {
        self.set(name, value)?;
        Ok(self)
    }

    /// Sets the first `values.len()` elements of the named parameter `name` (an array member,
    /// or a plain member as a one-element array) and ZEROES the remaining elements, so a shorter
    /// list never leaves stale values behind.
    ///
    /// # Errors
    /// [`ParamError::Unknown`], [`ParamError::Type`] (any value of another type),
    /// [`ParamError::Length`] (more values than elements). The pass is unchanged.
    pub fn set_slice<V: Into<ParamValue> + Copy>(&mut self, name: &str, values: &[V]) -> Result<(), ParamError> {
        let field = self.named_field(name)?;
        if values.len() > usize::try_from(field.count).unwrap_or(usize::MAX) {
            return Err(ParamError::Length { name: name.to_owned(), expected: field.count, given: values.len() });
        }
        if let Some(wrong) = values.iter().map(|&value| value.into().ty()).find(|&ty| ty != field.ty) {
            return Err(ParamError::Type { name: name.to_owned(), expected: field.ty, given: wrong });
        }
        let field = field.clone();
        for index in 0..field.count {
            let value = usize::try_from(index).ok().and_then(|index| values.get(index)).map_or_else(|| ParamValue::zero(field.ty), |&value| value.into());
            if let Some(range) = element_range(&field, index, self.named.len()) {
                value.write(&mut self.named[range]);
            }
        }
        Ok(())
    }

    /// The value of the plain (non-array) named parameter `name`, or `None` when the effect has
    /// no such member or it is an array (see [`Pass::get_slice`]).
    #[must_use]
    pub fn get(&self, name: &str) -> Option<ParamValue> {
        let field = self.effect.param(name).filter(|field| !field.is_array)?;
        ParamValue::read(field.ty, self.named.get(element_range(field, 0, self.named.len())?)?)
    }

    /// Every element of the named parameter `name` (one for a plain member), or `None` when the
    /// effect has no such member.
    #[must_use]
    pub fn get_slice(&self, name: &str) -> Option<Vec<ParamValue>> {
        let field = self.effect.param(name)?;
        (0..field.count).map(|index| ParamValue::read(field.ty, self.named.get(element_range(field, index, self.named.len())?)?)).collect()
    }

    /// The named-parameter block as uploaded at binding 4 (empty without `struct Params`).
    pub(crate) fn named_params(&self) -> &[u8] {
        &self.named
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
    pub const fn new() -> Self {
        Self { passes: Vec::new() }
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
pub(crate) const UNIFORM_BLOCK_SIZE: usize = 464;

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
    /// `delta_time: f32`.
    pub const DELTA_TIME: usize = 88;
    /// `frame: u32`.
    pub const FRAME: usize = 92;
    /// `pointer: vec4<f32>`.
    pub const POINTER: usize = 96;
    /// `internal_state_scale: vec2<f32>`.
    pub const STATE_SCALE: usize = 112;
    /// `internal_state_offset: vec2<f32>`.
    pub const STATE_OFFSET: usize = 120;
    /// `internal_cover_scale: vec2<f32>`.
    pub const COVER_SCALE: usize = 128;
    /// `internal_cover_offset: vec2<f32>`.
    pub const COVER_OFFSET: usize = 136;
    /// `pass_scale: f32`.
    pub const PASS_SCALE: usize = 144;
    /// `pass_index: u32`.
    pub const PASS_INDEX: usize = 148;
    /// `internal_frag_texel: vec2<f32>`.
    pub const FRAG_TEXEL: usize = 152;
    /// `internal_src_texel: vec2<f32>`.
    pub const SRC_TEXEL: usize = 160;
    /// `internal_state_texel: vec2<f32>`.
    pub const STATE_TEXEL: usize = 168;
    /// `internal_state_size: vec2<f32>`.
    pub const STATE_SIZE: usize = 176;
    /// `internal_region_scale: vec2<f32>`.
    pub const REGION_SCALE: usize = 184;
    /// `internal_region_offset: vec2<f32>`.
    pub const REGION_OFFSET: usize = 192;
    /// `params: array<vec4<f32>, 16>`.
    pub const PARAMS: usize = 208;
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
    /// Seconds since the previous frame (`fl.delta_time`); `0` until `with_input`.
    pub delta_time: f32,
    /// Frame counter (`fl.frame`); `0` until `with_input`.
    pub frame: u32,
    /// Pointer in capture uv, `z` pressed, `w` present (`fl.pointer`); `w = 0`: no pointer
    /// (the value until `with_input`).
    pub pointer: [f32; 4],
    /// Capture uv -> storage uv of the state texture (`internal_state_scale` / `_offset`);
    /// identity for a draw that binds no state (`plan.rs` sets it for the others).
    pub state: [[f32; 2]; 2],
    /// One texel of the state content the draw reads, in capture uv (`internal_state_texel`,
    /// `state_texel()`, the `state()` edge clamp); `[1, 1]` without state (the 1x1 dummy).
    pub state_texel: [f32; 2],
    /// The state content's size in texels (`internal_state_size`, `state_size()`, the
    /// `state_load` clamp); `[1, 1]` without state (the 1x1 dummy).
    pub state_size: [f32; 2],
    /// Same for the coverage texture (`internal_cover_*`); identity while no mask needs one.
    pub cover: [[f32; 2]; 2],
    /// Same for the region-id texture (`internal_region_*`, R1); identity for a draw that binds
    /// none (the 1x1 dummy).
    pub region: [[f32; 2]; 2],
    /// Output resolution of the draw relative to the capture (`fl.pass_scale`); `1` = full.
    pub pass_scale: f32,
    /// Index of the draw among the chain's passes of its kind, display or state (`fl.pass_index`;
    /// consecutive state passes are the substeps of one frame). Set per draw by `plan.rs`.
    pub pass_index: u32,
    /// One pixel of the draw's OUTPUT in capture uv (`internal_frag_texel`: fragment-mode-1 uv,
    /// `out_texel()`); `texel` at full resolution.
    pub frag_texel: [f32; 2],
    /// One pixel of the draw's `src()` INPUT in capture uv (`internal_src_texel`: `src_px()`,
    /// `src_texel()`, the `src()` edge clamp); `texel` for a full-resolution input.
    pub src_texel: [f32; 2],
}

impl FrameUniforms {
    /// The uniforms every pass of one layer shares, derived from its `geometry`: `rect_uv`,
    /// `texel` / `size_px` from the capture size, and `fl.pixels_per_point` =
    /// `geometry.pixels_per_layer_point` (physical pixels per LAYER point, so effect distances in
    /// points follow a layer transform's zoom exactly like the padding does). `src_scale` /
    /// `aux_scale` start at `1` and are set per draw by `plan.rs` (grow-only targets); the
    /// window fields (`frag_*`, `*_offset`) start in the interpolated, whole-capture state. The
    /// remaining members hold their neutral values (see each field; `frag_texel` / `src_texel`
    /// start at `texel`, full resolution, and `plan.rs` sets them for scaled draws); the per-frame input members
    /// are set by [`FrameUniforms::with_input`].
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
            delta_time: 0.0,
            frame: 0,
            pointer: [0.0; 4],
            state: [[1.0, 1.0], [0.0, 0.0]],
            state_texel: [1.0, 1.0],
            state_size: [1.0, 1.0],
            cover: [[1.0, 1.0], [0.0, 0.0]],
            region: [[1.0, 1.0], [0.0, 0.0]],
            pass_scale: 1.0,
            pass_index: 0,
            frag_texel: [1.0 / w, 1.0 / h],
            src_texel: [1.0 / w, 1.0 / h],
        }
    }
}

/// The per-frame input members of one run's uniform block: `fl.delta_time`, `fl.frame` and
/// `fl.pointer`, already resolved for the run's capture (the pointer through
/// `geometry::pointer_uniform`). `Default` is the neutral input (0 s, frame 0, no pointer).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct InputUniforms {
    /// Seconds since the previous frame, finite and `>= 0`.
    pub delta_time: f32,
    /// Frame counter (wraps at `u32::MAX`).
    pub frame: u32,
    /// Pointer in capture uv (`xy`), pressed (`z`, 0/1), present (`w`, 0/1).
    pub pointer: [f32; 4],
}

impl FrameUniforms {
    /// `self` with the per-frame input members of `input` (the rest unchanged).
    #[must_use]
    pub(crate) fn with_input(mut self, input: &InputUniforms) -> Self {
        self.delta_time = input.delta_time;
        self.frame = input.frame;
        self.pointer = input.pointer;
        self
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

/// Writes `value` as a native-endian `u32` at `offset`.
fn put_u32(bytes: &mut [u8; UNIFORM_BLOCK_SIZE], offset: usize, value: u32) {
    if let Some(chunk) = bytes.get_mut(offset..offset + 4) {
        chunk.copy_from_slice(&value.to_ne_bytes());
    }
}

/// Serializes one draw's uniforms into the `FlUniforms` byte layout (native endianness, as the
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
    put_f32s(&mut bytes, offsets::DELTA_TIME, &[frame.delta_time]);
    put_u32(&mut bytes, offsets::FRAME, frame.frame);
    put_f32s(&mut bytes, offsets::POINTER, &frame.pointer);
    put_f32s(&mut bytes, offsets::STATE_SCALE, &frame.state[0]);
    put_f32s(&mut bytes, offsets::STATE_OFFSET, &frame.state[1]);
    put_f32s(&mut bytes, offsets::COVER_SCALE, &frame.cover[0]);
    put_f32s(&mut bytes, offsets::COVER_OFFSET, &frame.cover[1]);
    put_f32s(&mut bytes, offsets::PASS_SCALE, &[frame.pass_scale]);
    put_u32(&mut bytes, offsets::PASS_INDEX, frame.pass_index);
    put_f32s(&mut bytes, offsets::FRAG_TEXEL, &frame.frag_texel);
    put_f32s(&mut bytes, offsets::SRC_TEXEL, &frame.src_texel);
    put_f32s(&mut bytes, offsets::STATE_TEXEL, &frame.state_texel);
    put_f32s(&mut bytes, offsets::STATE_SIZE, &frame.state_size);
    put_f32s(&mut bytes, offsets::REGION_SCALE, &frame.region[0]);
    put_f32s(&mut bytes, offsets::REGION_OFFSET, &frame.region[1]);
    for (i, slot) in params.iter().enumerate() {
        put_f32s(&mut bytes, offsets::PARAMS + i * 16, slot);
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::effect::{PRELUDE_WGSL, ParamType};

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
        assert_eq!(offset_of("delta_time"), offsets::DELTA_TIME);
        assert_eq!(offset_of("frame"), offsets::FRAME);
        assert_eq!(offset_of("pointer"), offsets::POINTER);
        assert_eq!(offset_of("internal_state_scale"), offsets::STATE_SCALE);
        assert_eq!(offset_of("internal_state_offset"), offsets::STATE_OFFSET);
        assert_eq!(offset_of("internal_cover_scale"), offsets::COVER_SCALE);
        assert_eq!(offset_of("internal_cover_offset"), offsets::COVER_OFFSET);
        assert_eq!(offset_of("pass_scale"), offsets::PASS_SCALE);
        assert_eq!(offset_of("pass_index"), offsets::PASS_INDEX);
        assert_eq!(offset_of("internal_frag_texel"), offsets::FRAG_TEXEL);
        assert_eq!(offset_of("internal_src_texel"), offsets::SRC_TEXEL);
        assert_eq!(offset_of("internal_state_texel"), offsets::STATE_TEXEL);
        assert_eq!(offset_of("internal_state_size"), offsets::STATE_SIZE);
        assert_eq!(offset_of("internal_region_scale"), offsets::REGION_SCALE);
        assert_eq!(offset_of("internal_region_offset"), offsets::REGION_OFFSET);
        assert_eq!(offset_of("params"), offsets::PARAMS);
        assert_eq!(members.len(), 28, "every member is pinned above");
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
            delta_time: 0.0625,
            frame: 77,
            pointer: [0.5, 0.25, 1.0, 1.0],
            state: [[0.5, 0.75], [0.125, 0.25]],
            state_texel: [0.25, 0.375],
            state_size: [4.0, 8.0],
            cover: [[0.375, 1.0], [0.0, 0.5]],
            region: [[0.625, 0.25], [0.75, 0.875]],
            pass_scale: 0.5,
            pass_index: 3,
            frag_texel: [1.0, 0.5],
            src_texel: [0.0625, 0.125],
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
        let read_u32 = |offset: usize| u32::from_ne_bytes([bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]]);
        assert!((read_f32(&bytes, offsets::DELTA_TIME) - 0.0625).abs() < f32::EPSILON);
        assert_eq!(read_u32(offsets::FRAME), 77);
        assert!((read_f32(&bytes, offsets::POINTER + 4) - 0.25).abs() < f32::EPSILON);
        assert!((read_f32(&bytes, offsets::STATE_SCALE + 4) - 0.75).abs() < f32::EPSILON);
        assert!((read_f32(&bytes, offsets::STATE_OFFSET) - 0.125).abs() < f32::EPSILON);
        assert!((read_f32(&bytes, offsets::COVER_SCALE) - 0.375).abs() < f32::EPSILON);
        assert!((read_f32(&bytes, offsets::COVER_OFFSET + 4) - 0.5).abs() < f32::EPSILON);
        assert!((read_f32(&bytes, offsets::PASS_SCALE) - 0.5).abs() < f32::EPSILON);
        assert_eq!(read_u32(offsets::PASS_INDEX), 3);
        assert!((read_f32(&bytes, offsets::FRAG_TEXEL) - 1.0).abs() < f32::EPSILON && (read_f32(&bytes, offsets::FRAG_TEXEL + 4) - 0.5).abs() < f32::EPSILON);
        assert!((read_f32(&bytes, offsets::SRC_TEXEL) - 0.0625).abs() < f32::EPSILON && (read_f32(&bytes, offsets::SRC_TEXEL + 4) - 0.125).abs() < f32::EPSILON);
        assert!((read_f32(&bytes, offsets::STATE_TEXEL + 4) - 0.375).abs() < f32::EPSILON && (read_f32(&bytes, offsets::STATE_SIZE) - 4.0).abs() < f32::EPSILON);
        assert!((read_f32(&bytes, offsets::REGION_SCALE) - 0.625).abs() < f32::EPSILON && (read_f32(&bytes, offsets::REGION_OFFSET + 4) - 0.875).abs() < f32::EPSILON);
    }

    #[test]
    fn layer_uniforms_start_neutral() {
        use egui::{CornerRadius, Rect, pos2, vec2};
        let rect = Rect::from_min_size(pos2(10.0, 10.0), vec2(40.0, 20.0));
        let Some(geometry) = crate::geometry::layer_geometry(rect, Rect::EVERYTHING, 1.0, [100, 100], 0.0, CornerRadius::ZERO, 1.0) else { panic!("visible") };
        let bytes = pack_uniforms(&FrameUniforms::for_layer(&geometry, 1.0, false), &[[0.0; 4]; PARAM_SLOTS]);
        assert!((read_f32(&bytes, offsets::PASS_SCALE) - 1.0).abs() < f32::EPSILON);
        assert!((read_f32(&bytes, offsets::FRAG_TEXEL) - read_f32(&bytes, offsets::TEXEL)).abs() < f32::EPSILON, "full resolution: the draw's texel is the capture's");
        assert_eq!(bytes[offsets::SRC_TEXEL..offsets::SRC_TEXEL + 8], bytes[offsets::TEXEL..offsets::TEXEL + 8], "full-resolution input: src texel = capture texel, bit for bit");
        assert!((read_f32(&bytes, offsets::STATE_SCALE) - 1.0).abs() < f32::EPSILON && (read_f32(&bytes, offsets::COVER_SCALE + 4) - 1.0).abs() < f32::EPSILON);
        assert!(read_f32(&bytes, offsets::POINTER + 12).abs() < f32::EPSILON, "no pointer present");
    }

    #[test]
    fn input_members_are_packed_where_the_prelude_reads_them() {
        use egui::{CornerRadius, Rect, pos2, vec2};
        let rect = Rect::from_min_size(pos2(10.0, 10.0), vec2(40.0, 20.0));
        let Some(geometry) = crate::geometry::layer_geometry(rect, Rect::EVERYTHING, 1.0, [100, 100], 0.0, CornerRadius::ZERO, 1.0) else { panic!("visible") };
        let input = InputUniforms { delta_time: 0.015_625, frame: u32::MAX, pointer: [0.25, -0.5, 1.0, 1.0] };
        let frame = FrameUniforms::for_layer(&geometry, 2.0, false).with_input(&input);
        assert!((frame.time - 2.0).abs() < f32::EPSILON, "with_input keeps the rest");
        let bytes = pack_uniforms(&frame, &[[0.0; 4]; PARAM_SLOTS]);
        let read_u32 = |offset: usize| u32::from_ne_bytes([bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]]);
        assert!((read_f32(&bytes, offsets::DELTA_TIME) - 0.015_625).abs() < f32::EPSILON);
        assert_eq!(read_u32(offsets::FRAME), u32::MAX);
        for (i, expected) in [0.25, -0.5, 1.0, 1.0].into_iter().enumerate() {
            assert!((read_f32(&bytes, offsets::POINTER + i * 4) - expected).abs() < f32::EPSILON, "pointer[{i}]");
        }
        // The neutral input packs the neutral values.
        let neutral = pack_uniforms(&FrameUniforms::for_layer(&geometry, 2.0, false).with_input(&InputUniforms::default()), &[[0.0; 4]; PARAM_SLOTS]);
        assert_eq!(neutral, pack_uniforms(&FrameUniforms::for_layer(&geometry, 2.0, false), &[[0.0; 4]; PARAM_SLOTS]));
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
    fn pass_scales_are_clamped() {
        let Ok(effect) = Effect::new("id", "fn effect(uv: vec2<f32>) -> vec4<f32> { return src(uv); }") else { panic!("identity must compile") };
        let pass = Pass::new(&effect);
        assert!((pass.scale() - 1.0).abs() < f32::EPSILON && !pass.is_scaled(), "full resolution by default");
        assert!((pass.clone().with_scale(0.5).scale() - 0.5).abs() < f32::EPSILON && pass.clone().with_scale(0.5).is_scaled());
        assert!((pass.clone().with_scale(2.0).scale() - 1.0).abs() < f32::EPSILON, "never above full resolution");
        for low in [0.0, -1.0, -5.0, 1e-9] {
            assert!((pass.clone().with_scale(low).scale() - MIN_PASS_SCALE).abs() < f32::EPSILON, "{low}");
        }
        assert!((pass.clone().with_scale(f32::NAN).scale() - 1.0).abs() < f32::EPSILON && (pass.with_scale(f32::INFINITY).scale() - 1.0).abs() < f32::EPSILON, "non-finite = full");
    }

    #[test]
    fn param_indices_are_checked() {
        let Ok(effect) = Effect::new("id", "fn effect(uv: vec2<f32>) -> vec4<f32> { return src(uv); }") else { panic!("identity must compile") };
        let pass = Pass::new(&effect);
        assert!(pass.clone().with_param(15, [1.0; 4]).is_ok());
        assert_eq!(pass.clone().with_param(16, [1.0; 4]).err(), Some(ParamIndexError { index: 16 }));
        assert_eq!(pass.param(16), None);
    }

    fn named_effect() -> Effect {
        let snippet = "struct Params { gain: f32, count: u32, offset: i32, dir: vec2<f32>, tint: vec3<f32>, color: vec4<f32>, xf: mat4x4<f32>, lights: array<vec4<f32>, 3> }
fn effect(uv: vec2<f32>) -> vec4<f32> { return src(uv) * p.color * p.gain; }";
        match Effect::new("named", snippet) {
            Ok(effect) => effect,
            Err(error) => panic!("{error}"),
        }
    }

    #[test]
    fn named_params_are_typed_and_stored_at_their_offsets() {
        let effect = named_effect();
        let mut pass = Pass::new(&effect);
        assert_eq!(pass.named_params().len(), effect.param_layout().block_size);
        assert!(pass.named_params().iter().all(|&byte| byte == 0), "starts zeroed");
        let matrix = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]];
        let set = pass.set("gain", 1.5).and_then(|()| pass.set("count", 7_u32)).and_then(|()| pass.set("offset", -3)).and_then(|()| pass.set("dir", [0.5, -0.5])).and_then(|()| pass.set("tint", [0.25, 0.5, 0.75]));
        let set = set.and_then(|()| pass.set("color", [1.0, 0.0, 0.5, 1.0])).and_then(|()| pass.set("xf", matrix));
        assert_eq!(set, Ok(()));
        assert_eq!(pass.get("gain"), Some(ParamValue::F32(1.5)));
        assert_eq!(pass.get("count"), Some(ParamValue::U32(7)));
        assert_eq!(pass.get("offset"), Some(ParamValue::I32(-3)));
        assert_eq!(pass.get("tint"), Some(ParamValue::Vec3([0.25, 0.5, 0.75])));
        assert_eq!(pass.get("xf"), Some(ParamValue::Mat4(matrix)));
        // The bytes land where the effect's layout (naga's) says.
        let bytes = pass.named_params();
        for info in effect.params().iter().filter(|info| !info.is_array) {
            let Some(value) = pass.get(&info.name) else { panic!("{}", info.name) };
            let start = usize::try_from(info.offset).unwrap_or(0);
            assert_eq!(ParamValue::read(info.ty, &bytes[start..]), Some(value), "{}", info.name);
        }
        assert!((read_f32(bytes, usize::try_from(effect.param("xf").map_or(0, |info| info.offset)).unwrap_or(0) + 4 * 4) - 5.0).abs() < f32::EPSILON, "matrix columns are consecutive vec4s");
    }

    #[test]
    fn named_param_errors_leave_the_pass_unchanged() {
        let effect = named_effect();
        let mut pass = Pass::new(&effect).with("gain", 2.0).unwrap_or_else(|error| panic!("{error}"));
        let before = pass.named_params().to_vec();
        assert_eq!(pass.set("nope", 1.0), Err(ParamError::Unknown { effect: "named".to_owned(), name: "nope".to_owned() }));
        assert_eq!(pass.set("gain", 1), Err(ParamError::Type { name: "gain".to_owned(), expected: ParamType::F32, given: ParamType::I32 }));
        assert_eq!(pass.set("count", 1), Err(ParamError::Type { name: "count".to_owned(), expected: ParamType::U32, given: ParamType::I32 }));
        assert_eq!(pass.set("lights", [1.0; 4]), Err(ParamError::Array { name: "lights".to_owned(), count: 3 }));
        assert_eq!(pass.set_slice("lights", &[[1.0_f32; 4]; 4]), Err(ParamError::Length { name: "lights".to_owned(), expected: 3, given: 4 }));
        assert_eq!(pass.set_slice("lights", &[ParamValue::Vec4([1.0; 4]), ParamValue::F32(1.0)]), Err(ParamError::Type { name: "lights".to_owned(), expected: ParamType::Vec4, given: ParamType::F32 }));
        assert_eq!(pass.named_params(), &before[..]);
        let Ok(plain) = Effect::new("plain", "fn effect(uv: vec2<f32>) -> vec4<f32> { return src(uv); }") else { panic!("identity must compile") };
        assert!(matches!(Pass::new(&plain).set("gain", 1.0), Err(ParamError::Unknown { .. })), "no struct Params: every name is unknown");
        assert!(Pass::new(&plain).named_params().is_empty());
    }

    #[test]
    fn set_slice_writes_a_prefix_and_zeroes_the_rest() {
        let effect = named_effect();
        let mut pass = Pass::new(&effect);
        assert_eq!(pass.set_slice("lights", &[[1.0_f32; 4], [2.0; 4], [3.0; 4]]), Ok(()));
        assert_eq!(pass.set_slice("lights", &[[9.0_f32; 4]]), Ok(()));
        assert_eq!(pass.get_slice("lights"), Some(vec![ParamValue::Vec4([9.0; 4]), ParamValue::Vec4([0.0; 4]), ParamValue::Vec4([0.0; 4])]));
        assert_eq!(pass.get("lights"), None, "an array is read with get_slice");
        // A plain member is a one-element array for set_slice / get_slice.
        assert_eq!(pass.set_slice("gain", &[4.0_f32]), Ok(()));
        assert_eq!(pass.get_slice("gain"), Some(vec![ParamValue::F32(4.0)]));
        assert!(matches!(pass.set_slice("gain", &[1.0_f32, 2.0]), Err(ParamError::Length { expected: 1, given: 2, .. })));
        let stride = effect.param("lights").map_or(0, |info| info.stride);
        assert_eq!(stride, 16);
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
