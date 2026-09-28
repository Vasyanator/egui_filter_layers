/*
File: plan.rs

Purpose:
The single, backend-neutral owner of HOW a chain is wired: which texture every draw of a run
reads at each binding, which one it writes, its viewport and output resolution (per-pass scale),
the size the intermediates must have, and the complete uniform blocks each draw uploads
(binding 0: storage-uv mappings, texel sizes, fragment offsets, anonymous parameters; binding 4:
the named-parameter block). Backends (wgpu `chain.rs`, glow `render.rs`) only create objects,
bind the texture a `Slot` names, upload the bytes produced here and draw; they make no wiring
decision of their own.

Key structures:
- `Slot`: a texture of one run (the capture, a ping-pong intermediate or a state texture).
- `StepKind`: display step (writes the display chain) or state step (writes the state, F7).
- `StatePlanInput` / `StatePlan`: what the caller knows of a layer's state (which texture holds
  the latest, its size, whether to step it this run) and what the run leaves (clear, sizes,
  where the latest state is afterwards).
- `Content`: what a slot holds at one point of the run (the whole capture at some resolution, or
  a window of it at the texture origin).
- `Bindings`: the slots one draw binds at `fl_src_texture` / `fl_original_texture` /
  `fl_state_texture` (`None` = the backend's neutral dummy), and which user texture slots get the
  layer's texture (else a dummy).
- `PassStep`, `CompositeStep`, `ChainPlan`: the draws of one run, in order, and the pool size.
- `Allocations`: the allocated size of every slot's texture (backends report them).
- `TextureMapping`: capture uv -> storage uv of one texture content (`clamp(uv) * scale -
  offset`, the prelude's `internal_*_scale` / `internal_*_offset`).

Key functions:
- `plan()`: the draws of a run over a window of the capture.
- `check_textures()`: every user texture slot an effect of the chain reads has a texture
  attached (the one check the UI layer and the offline renderer run before drawing).
- `scaled_px()`: the output size of a scaled pass (`ceil(side * scale)`, at least 1).
- `TextureMapping::of()`: the ONE formula for a texture's storage-uv mapping.
- `step_uniforms()` / `composite_uniforms()`: the full `FlUniforms` block of one draw.
- `params_block()` / `composite_params_block()`: the bytes bound at binding 4 (`p: Params`) for a
  step / the composite; `EMPTY_PARAMS_BLOCK` (16 zero bytes) when a module declares none.

Notes:
Pass `i` reads the capture (i = 0) or pass `i - 1`'s output and writes `Ping` (even i) / `Pong`
(odd i); the composite reads the last output as `src`. EVERY draw (each pass and the composite)
binds the capture as the original (`original(uv)`, prelude), mapped with the capture's own
storage mapping (`internal_aux_*`), which may differ from the ping-pong `src` mapping (grow-only
targets of other sizes). Every other texture binding of the layout (state, coverage) is the
backend's dummy. User textures (F8): every pass step binds the layer's attached texture of each
slot (wgpu `@group(1 + i)`, glow unit `4 + i`) and the backend's dummy where none is attached;
the composite binds dummies (its module reads none). Their uv is the texture's own, so they need
no mapping and are independent of windows and scales. Backend-specific uniform flags (`flip_y`, fragment mode) stay in the
`FrameUniforms` each backend passes in.
State (F7): state passes (`Effect::new_state`) are STATE steps: they read the display chain's
current input as `src` (the display input does not advance past them), the latest state at
binding 5, and write the OTHER state texture (`StateA` / `StateB` ping-pong, swapped after every
state step), `ceil(capture * state scale)` texels over the whole capture. Display steps bind the
latest state (the last state step before them, or last run's). A state with no content yet (new,
reset) is cleared first (`StatePlan::clear`). A repeated run in the same frame (`step = false`)
skips the state steps: its display steps read the state as it is. The read mapping of the latest
content uses ITS size, the writes the new size, so a resized layer reads the previous state
stretched. Stateful runs cover the whole capture only (never tiled).
Scale (F6): a pass with `scale < 1` renders `scaled_px(capture, scale)` pixels holding the WHOLE
capture at that resolution (at the texture origin); readers map it with `TextureMapping::of(
scaled, allocated, 0)`, so `uv` keeps its capture meaning and bilinear sampling resamples (the
composite upsamples a scaled last output). A scaled chain runs only over the whole capture
(window edges would not round consistently); `plan` refuses other windows. Every scaled size is
at most the capture, so the two ping-pong targets sized `ChainPlan::target_px` (the window) hold
every step.
*/

use crate::backend::TextureError;
use crate::effect::TEXTURE_SLOTS;
use crate::geometry::{LayerGeometry, PxRect, px_f32};
use crate::pass::{FrameUniforms, Params, Pass, UNIFORM_BLOCK_SIZE, pack_uniforms};

/// One texture of a chain run. Backends map every variant to their own texture object.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Slot {
    /// The captured backdrop (the run's window of it, at the texture origin).
    Source,
    /// The first ping-pong intermediate.
    Ping,
    /// The second ping-pong intermediate.
    Pong,
    /// State texture 0 of the layer (F7).
    StateA,
    /// State texture 1 of the layer (F7).
    StateB,
}

impl Slot {
    /// The state slot of index `index` (0 = [`Slot::StateA`], anything else = [`Slot::StateB`]).
    #[must_use]
    pub(crate) fn state(index: u8) -> Self {
        if index == 0 { Self::StateA } else { Self::StateB }
    }
}

/// What a step writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StepKind {
    /// A display pass: writes the display chain's next input (a ping-pong slot).
    Display,
    /// A state pass: writes the next state (a state slot).
    State,
}

/// What the caller knows of a stateful layer's state before a run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct StatePlanInput {
    /// `StateOptions::scale`: the state holds `scaled_px(capture, scale)` texels.
    pub scale: f32,
    /// Which state slot (0 / 1) holds the latest state.
    pub current: u8,
    /// The size of the latest state content; `None` = no content (new or reset): cleared first.
    pub previous_px: Option<[u32; 2]>,
    /// Whether this run steps the state (false: repeated paint in the same frame; state steps
    /// are skipped).
    pub step: bool,
}

/// What a stateful run does with the state textures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct StatePlan {
    /// Clear this slot (the whole texture) to transparent before any draw: it has no content.
    pub clear: Option<Slot>,
    /// Both state textures must hold at least this many texels (grown keeping their content).
    pub target_px: [u32; 2],
    /// Which state slot (0 / 1) holds the latest state after the run.
    pub final_index: u8,
    /// The size of that content.
    pub final_px: [u32; 2],
}

/// The slots one draw binds; `None` means the backend's neutral dummy texture for that binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Bindings {
    /// Bound at `fl_src_texture` (`src()`).
    pub src: Slot,
    /// Bound at `fl_original_texture` (`original()`): the plan binds [`Slot::Source`] for every
    /// draw. `None` = a transparent dummy (its `internal_aux_*` mapping then keeps the frame's
    /// identity value); no current plan produces it.
    pub original: Option<Slot>,
    /// Bound at `fl_state_texture` (`state()`): the latest state for every pass step of a
    /// stateful run, `None` (a transparent 1x1 dummy, i.e. zero state) otherwise and for the
    /// composite.
    pub state: Option<Slot>,
    /// Per user texture slot: `true` = bind the texture the layer attached there, `false` = the
    /// backend's dummy (nothing attached, or the composite).
    pub user: [bool; TEXTURE_SLOTS],
}

/// What a slot's texture holds at one point of a run: the whole capture rendered at
/// `logical_px` pixels, of which the window starting at `origin_px` (pixels of that resolution)
/// lies at the texture origin. Full resolution: `logical_px` = the capture size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Content {
    /// The whole capture's size at this content's resolution, in pixels.
    pub logical_px: [u32; 2],
    /// The window origin held at the texture origin, in pixels of this resolution.
    pub origin_px: [u32; 2],
    /// Whether this is the capture's own resolution (its texel is then exactly `fl.texel`).
    pub full: bool,
}

impl Content {
    /// The storage mapping of this content in a texture of `allocated` pixels.
    #[must_use]
    pub(crate) fn mapping(self, allocated: [u32; 2]) -> TextureMapping {
        TextureMapping::of(self.logical_px, allocated, self.origin_px)
    }

    /// One pixel of this content in capture uv: `frame.texel` at full resolution (bit for bit
    /// what every full-resolution draw uploaded before scales existed), else `1 / logical_px`.
    #[must_use]
    fn texel(self, frame: &FrameUniforms) -> [f32; 2] {
        if self.full { frame.texel } else { self.logical_px.map(|side| 1.0 / to_f32_exact(side)) }
    }
}

/// One effect pass of a run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PassStep {
    /// Index into the chain's passes (whose effect and parameters this draw runs).
    pub pass: usize,
    /// Display or state step (which chain its output continues).
    pub kind: StepKind,
    /// Its index among the run's steps of its kind (`fl.pass_index`).
    pub pass_index: u32,
    /// What its state slot holds (`bindings.state`), when it binds one.
    pub state: Option<Content>,
    /// The textures it reads.
    pub bindings: Bindings,
    /// What its `src` slot holds when it runs.
    pub input: Content,
    /// The intermediate it renders into (at the target's origin).
    pub output: Slot,
    /// What it writes into `output`.
    pub written: Content,
    /// Its output scale (`fl.pass_scale`; `1` = full resolution).
    pub scale: f32,
    /// Viewport size in pixels, from the target's origin: the window of the capture it computes
    /// (at its own resolution).
    pub used_px: [i32; 2],
    /// Added to the fragment position (target pixels) to get pixels of the step's resolution
    /// (`internal_frag_offset`, read only in fragment mode 1).
    pub frag_offset: [f32; 2],
}

/// The composite draw of a run (the viewport is the caller's region in its own render target).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct CompositeStep {
    /// The textures it reads: the last pass output as `src`, the capture as the original.
    pub bindings: Bindings,
    /// What its `src` slot holds (a scaled last pass is upsampled here, bilinearly).
    pub input: Content,
    /// Added to the fragment position (target pixels) to get capture pixels.
    pub frag_offset: [f32; 2],
}

/// Every draw of one chain run, in execution order.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ChainPlan {
    /// The effect passes (empty only for a repeated stateful run of a chain of state passes).
    pub steps: Vec<PassStep>,
    /// The composite that writes the result.
    pub composite: CompositeStep,
    /// What [`Slot::Source`] holds: the run's window of the capture at full resolution. It is
    /// every draw's original.
    pub source: Content,
    /// The smallest allocation BOTH ping-pong targets need: the window, which holds every
    /// step's viewport (scaled images are never larger). Also for a chain of scaled passes only,
    /// so every caller that allocates a window's targets gets the same sizes (and so the same
    /// bilinear weights) whatever the scales; larger (grow-only) targets are fine, the mappings
    /// use the actual allocation.
    pub target_px: [u32; 2],
    /// The state textures' part of the run (stateful runs only).
    pub state: Option<StatePlan>,
}

/// The allocated size of every slot's texture in one run, in pixels (reported by the backend
/// after it made the targets at least [`ChainPlan::target_px`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Allocations {
    /// Of [`Slot::Source`].
    pub source: [u32; 2],
    /// Of [`Slot::Ping`] and [`Slot::Pong`].
    pub ping_pong: [[u32; 2]; 2],
    /// Of [`Slot::StateA`] and [`Slot::StateB`] (zero without state).
    pub state: [[u32; 2]; 2],
}

impl Allocations {
    /// The allocated size of `slot`.
    #[must_use]
    pub(crate) fn of(&self, slot: Slot) -> [u32; 2] {
        match slot {
            Slot::Source => self.source,
            Slot::Ping => self.ping_pong[0],
            Slot::Pong => self.ping_pong[1],
            Slot::StateA => self.state[0],
            Slot::StateB => self.state[1],
        }
    }
}

/// The output size of a pass at `scale` over a capture side of `side` pixels:
/// `ceil(side * scale)`, at least 1 and at most `side`. `scale` is a [`Pass::scale`] value
/// (finite, in `(0, 1]`). The product is exact in `f64`; the `f32` scale's own rounding error
/// (at most `2^-24` relative) is taken off first, so `0.1` over 10 pixels is 1 pixel, not 2.
#[must_use]
pub(crate) fn scaled_px(side: u32, scale: f32) -> u32 {
    let side = side.min(1 << 24);
    let tolerance = f64::from(side) * f64::from(f32::EPSILON) * 0.5;
    let product = (f64::from(side) * f64::from(scale) - tolerance).ceil();
    // `product` lies in `[0, 2^24]` for the scales `Pass` stores (a NaN / negative saturates to
    // 0 and is clamped up below), so the conversion is exact.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // In-range by the bound above; `as` saturates otherwise.
    let pixels = product as u32;
    pixels.clamp(1, side.max(1))
}

/// Plans a run of `passes` over `window` (capture pixels; the whole capture when not tiled) of
/// a capture of `capture` pixels, whose composite puts capture pixel `(0, 0)` at
/// `composite_origin` of its render target (target pixels, top-left origin). `attached[i]`: the
/// layer attached a user texture at slot `i` (every pass step binds it; see
/// [`check_textures`] for the slots the effects need). `state`: the layer's state, required when
/// a pass is a state pass (`Effect::new_state`), see the file notes.
///
/// Returns `None` when there is nothing to draw (no passes, an empty / negative window) or the
/// run cannot be planned: a chain with a scaled pass ([`Pass::is_scaled`]) or a stateful run over
/// anything but the whole capture (callers that tile refuse such chains first,
/// `OfflineError::NotTileable`), or a chain with a state pass and no `state`.
#[must_use]
pub(crate) fn plan(passes: &[Pass], capture: [i32; 2], window: PxRect, composite_origin: [i32; 2], attached: [bool; TEXTURE_SLOTS], state: Option<StatePlanInput>) -> Option<ChainPlan> {
    if passes.is_empty() || window.w <= 0 || window.h <= 0 || window.x < 0 || window.y < 0 {
        return None;
    }
    let has_state_steps = passes.iter().any(|pass| pass.effect().is_state());
    if has_state_steps && state.is_none() {
        return None;
    }
    let whole = window.x == 0 && window.y == 0 && window.size() == capture;
    if !whole && (state.is_some() || passes.iter().any(Pass::is_scaled)) {
        return None;
    }
    let capture_px = capture.map(|side| u32::try_from(side).unwrap_or(0));
    let origin = [u32::try_from(window.x).ok()?, u32::try_from(window.y).ok()?];
    let source = Content { logical_px: capture_px, origin_px: origin, full: true };
    // The whole capture at `px` texels, at the texture origin (scaled images and states).
    let whole_at = |px: [u32; 2]| Content { logical_px: px, origin_px: [0, 0], full: px == capture_px };
    let state_px = state.map(|state| capture_px.map(|side| scaled_px(side, state.scale)));
    // The latest state: its slot index and content (a cleared state reads as the new size).
    let mut latest = state.zip(state_px).map(|(state, px)| (state.current, whole_at(state.previous_px.unwrap_or(px))));
    let mut steps = Vec::with_capacity(passes.len());
    let (mut input, mut input_slot) = (source, Slot::Source);
    let (mut display_count, mut state_count) = (0_u32, 0_u32);
    for (index, pass) in passes.iter().enumerate() {
        let state_binding = latest.map(|(slot, _)| Slot::state(slot));
        let state_content = latest.map(|(_, content)| content);
        let bindings = Bindings { src: input_slot, original: Some(Slot::Source), state: state_binding, user: attached };
        if pass.effect().is_state() {
            // `state` and `state_px` are `Some` here (checked above); a repeated run skips it.
            let (Some(state), Some(px), Some((read, _))) = (state, state_px, latest) else {
                return None;
            };
            if !state.step {
                continue;
            }
            let write = u8::from(read == 0);
            let written = whole_at(px);
            let used = px.map(|side| i32::try_from(side).unwrap_or(i32::MAX));
            steps.push(PassStep { pass: index, kind: StepKind::State, pass_index: state_count, state: state_content, bindings, input, output: Slot::state(write), written, scale: state.scale, used_px: used, frag_offset: [0.0, 0.0] });
            latest = Some((write, written));
            state_count += 1;
            continue;
        }
        let output = if display_count.is_multiple_of(2) { Slot::Ping } else { Slot::Pong };
        let (written, used_px, frag_offset) = if pass.is_scaled() {
            // Whole capture only (checked above): the smaller image starts at capture uv 0.
            let logical = capture_px.map(|side| scaled_px(side, pass.scale()));
            (whole_at(logical), logical.map(|side| i32::try_from(side).unwrap_or(i32::MAX)), [0.0, 0.0])
        } else {
            (source, window.size(), [px_f32(window.x), px_f32(window.y)])
        };
        // Every pass may read the untouched capture through `original()` (F1).
        steps.push(PassStep { pass: index, kind: StepKind::Display, pass_index: display_count, state: state_content, bindings, input, output, written, scale: pass.scale(), used_px, frag_offset });
        (input, input_slot) = (written, output);
        display_count += 1;
    }
    let composite = CompositeStep {
        // The composite module reads no user texture and no state: dummies.
        bindings: Bindings { src: input_slot, original: Some(Slot::Source), state: None, user: [false; TEXTURE_SLOTS] },
        input,
        frag_offset: [-px_f32(composite_origin[0]), -px_f32(composite_origin[1])],
    };
    let target_px = window.size().map(|side| u32::try_from(side).unwrap_or(0));
    let state = state.zip(state_px).zip(latest).map(|((state, px), (final_index, content))| StatePlan {
        clear: state.previous_px.is_none().then_some(Slot::state(state.current)),
        target_px: px,
        final_index,
        final_px: content.logical_px,
    });
    Some(ChainPlan { steps, composite, source, target_px, state })
}

/// Checks that every user texture slot an effect of `passes` may read
/// (`Effect::reads_texture`) has a texture attached (`attached[i]`).
///
/// # Errors
/// [`TextureError::NotAttached`] naming the first pass (in chain order) and slot that lacks one.
pub(crate) fn check_textures(passes: &[Pass], attached: [bool; TEXTURE_SLOTS]) -> Result<(), TextureError> {
    for (pass_index, pass) in passes.iter().enumerate() {
        if let Some(slot) = (0..TEXTURE_SLOTS).find(|&slot| pass.effect().reads_texture(slot) && !attached[slot]) {
            return Err(TextureError::NotAttached { effect: pass.effect().label().to_owned(), pass: pass_index, slot });
        }
    }
    Ok(())
}

/// How capture uv maps into one texture's storage uv: `clamp(uv) * scale - offset` (the
/// prelude's `internal_*_scale` / `internal_*_offset`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct TextureMapping {
    /// Capture size / allocated size.
    pub scale: [f32; 2],
    /// The window origin / allocated size (0 for a whole-capture window).
    pub offset: [f32; 2],
}

impl TextureMapping {
    /// The mapping of a texture of `allocated` pixels holding, at its origin, the window of a
    /// capture of `capture` pixels that starts at `window_origin` (capture pixels). Every ratio
    /// is one correctly rounded `f32` division of exact sizes (exact below 2^24), so backends
    /// computing from `i32` or `u32` sizes get identical bits. An unallocated axis (0) maps with
    /// scale 1 and offset 0.
    #[must_use]
    pub(crate) fn of(capture: [u32; 2], allocated: [u32; 2], window_origin: [u32; 2]) -> Self {
        let offset = |origin: u32, allocated: u32| if allocated == 0 { 0.0 } else { px_ratio(origin, allocated) };
        Self { scale: [px_ratio(capture[0], allocated[0]), px_ratio(capture[1], allocated[1])], offset: [offset(window_origin[0], allocated[0]), offset(window_origin[1], allocated[1])] }
    }
}

/// `used / allocated` as one correctly rounded `f32` division of the exact sizes (`1` when
/// `allocated` is 0). Sizes are exact below 2^24 (a whole image may exceed any texture limit when
/// the offline tiler splits it); larger values saturate there. With a power-of-two `allocated`
/// the result is exact.
#[must_use]
fn px_ratio(used: u32, allocated: u32) -> f32 {
    if allocated == 0 {
        return 1.0;
    }
    to_f32_exact(used) / to_f32_exact(allocated)
}

/// `value` as `f32`, exact below 2^24 (saturating at 2^24): the high and low 16 bits are both
/// exact in `f32` and so is their recombination for values below 2^24.
fn to_f32_exact(value: u32) -> f32 {
    let value = value.min(1 << 24);
    let high = u16::try_from(value >> 16).unwrap_or(u16::MAX);
    let low = u16::try_from(value & 0xffff).unwrap_or(u16::MAX);
    f32::from(high) * 65536.0 + f32::from(low)
}

/// Everything that varies between the draws of one run besides the parameters.
#[derive(Clone, Copy, Debug)]
struct Draw {
    /// The slots it binds.
    bindings: Bindings,
    /// What its `src` slot holds.
    input: Content,
    /// What its original slot holds (the capture window).
    original: Content,
    /// `internal_frag_offset`.
    frag_offset: [f32; 2],
    /// Its output resolution: `None` = the capture's (the composite), else what it writes.
    written: Option<Content>,
    /// `fl.pass_scale`.
    scale: f32,
    /// What its state slot holds (`bindings.state`), when it binds one.
    state: Option<Content>,
    /// `fl.pass_index`.
    pass_index: u32,
}

/// The uniform block of one draw: `frame` (the layer's values and the backend's flags) with the
/// `src` / original mappings and texels of `draw` in the textures `allocations` describes, its
/// fragment offset, output texel and scale, plus `params`.
fn draw_uniforms(frame: &FrameUniforms, draw: &Draw, allocations: &Allocations, params: &Params) -> [u8; UNIFORM_BLOCK_SIZE] {
    let mut out = *frame;
    let src = draw.input.mapping(allocations.of(draw.bindings.src));
    out.src_scale = src.scale;
    out.src_offset = src.offset;
    out.src_texel = draw.input.texel(frame);
    // A dummy original is never sampled; its mapping keeps the frame's identity value.
    if let Some(original) = draw.bindings.original {
        let aux = draw.original.mapping(allocations.of(original));
        out.aux_scale = aux.scale;
        out.aux_offset = aux.offset;
    }
    // A dummy state keeps the frame's neutral values (identity mapping, one 1x1 texel).
    if let (Some(slot), Some(content)) = (draw.bindings.state, draw.state) {
        let mapping = content.mapping(allocations.of(slot));
        out.state = [mapping.scale, mapping.offset];
        out.state_texel = content.texel(frame);
        out.state_size = content.logical_px.map(to_f32_exact);
    }
    out.pass_index = draw.pass_index;
    out.frag_offset = draw.frag_offset;
    out.frag_texel = draw.written.map_or(frame.texel, |written| written.texel(frame));
    out.pass_scale = draw.scale;
    pack_uniforms(&out, params)
}

/// The complete `FlUniforms` block of `step` of `plan`, which runs `pass`. `frame` is the
/// backend's `FrameUniforms::for_layer` value (with its `flip_y` / fragment-mode flags).
#[must_use]
pub(crate) fn step_uniforms(frame: &FrameUniforms, plan: &ChainPlan, step: &PassStep, allocations: &Allocations, pass: &Pass) -> [u8; UNIFORM_BLOCK_SIZE] {
    let draw = Draw { bindings: step.bindings, input: step.input, original: plan.source, frag_offset: step.frag_offset, written: Some(step.written), scale: step.scale, state: step.state, pass_index: step.pass_index };
    draw_uniforms(frame, &draw, allocations, pass.params())
}

/// The complete `FlUniforms` block of `plan`'s composite: its bindings' mappings and the mask /
/// region parameters of `geometry` (`LayerGeometry::composite_params`).
#[must_use]
pub(crate) fn composite_uniforms(frame: &FrameUniforms, plan: &ChainPlan, allocations: &Allocations, geometry: &LayerGeometry) -> [u8; UNIFORM_BLOCK_SIZE] {
    let composite = &plan.composite;
    let draw = Draw { bindings: composite.bindings, input: composite.input, original: plan.source, frag_offset: composite.frag_offset, written: None, scale: 1.0, state: None, pass_index: 0 };
    draw_uniforms(frame, &draw, allocations, &geometry.composite_params())
}

/// The binding-4 bytes of a draw whose module declares no `struct Params`: 16 zero bytes, the
/// smallest valid uniform binding (a GL std140 block and a wgpu binding are 16-byte multiples).
pub(crate) const EMPTY_PARAMS_BLOCK: [u8; 16] = [0; 16];

/// The named-parameter block (binding 4, `p: Params`) of a step running `pass` (the step's
/// `pass` index selects it): the pass's block in its effect's layout, or [`EMPTY_PARAMS_BLOCK`]
/// when the effect declares no `struct Params`. Its length is a multiple of 16.
#[must_use]
pub(crate) fn params_block(pass: &Pass) -> &[u8] {
    let named = pass.named_params();
    if named.is_empty() { &EMPTY_PARAMS_BLOCK } else { named }
}

/// The named-parameter block of the composite (whose module declares none).
#[must_use]
pub(crate) fn composite_params_block() -> &'static [u8] {
    &EMPTY_PARAMS_BLOCK
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::effect::Effect;
    use crate::pass::offsets;

    /// No user texture attached.
    const NO_TEXTURES: [bool; TEXTURE_SLOTS] = [false; TEXTURE_SLOTS];

    /// Whether the two `f32`s at `offset` equal `expected` (all test values are exact in f32).
    fn holds(bytes: &[u8], offset: usize, expected: [f32; 2]) -> bool {
        let read = |at: usize| f32::from_ne_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]);
        (read(offset) - expected[0]).abs() < f32::EPSILON && (read(offset + 4) - expected[1]).abs() < f32::EPSILON
    }

    fn identity() -> Effect {
        match Effect::new("id", "fn effect(uv: vec2<f32>) -> vec4<f32> { return src(uv); }") {
            Ok(effect) => effect,
            Err(error) => panic!("identity must compile: {error}"),
        }
    }

    /// `count` identity passes at full resolution.
    fn passes(count: usize) -> Vec<Pass> {
        vec![Pass::new(&identity()); count]
    }

    /// Reads the `f32` at `offset`.
    fn read(bytes: &[u8], offset: usize) -> f32 {
        f32::from_ne_bytes([bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]])
    }

    #[test]
    fn steps_follow_the_ping_pong_order() {
        let window = PxRect { x: 16, y: 32, w: 100, h: 50 };
        let Some(plan) = plan(&passes(3), [200, 100], window, [4, 8], NO_TEXTURES, None) else { panic!("three passes over a non-empty window") };
        let reads: Vec<Slot> = plan.steps.iter().map(|step| step.bindings.src).collect();
        let writes: Vec<Slot> = plan.steps.iter().map(|step| step.output).collect();
        assert_eq!(reads, [Slot::Source, Slot::Ping, Slot::Pong]);
        assert_eq!(writes, [Slot::Ping, Slot::Pong, Slot::Ping]);
        assert!(plan.steps.iter().enumerate().all(|(index, step)| step.pass == index && step.used_px == [100, 50]));
        assert_eq!(plan.composite.bindings, Bindings { src: Slot::Ping, original: Some(Slot::Source), state: None, user: NO_TEXTURES });
        assert_eq!(plan.composite.frag_offset.map(f32::to_bits), [-4.0_f32, -8.0].map(f32::to_bits));
        assert_eq!(plan.steps[0].frag_offset.map(f32::to_bits), [16.0_f32, 32.0].map(f32::to_bits));
        assert_eq!(plan.target_px, [100, 50], "the pool holds the window");
        let window_content = Content { logical_px: [200, 100], origin_px: [16, 32], full: true };
        assert_eq!(plan.source, window_content);
        assert!(plan.steps.iter().all(|step| step.input == window_content && step.written == window_content && (step.scale - 1.0).abs() < f32::EPSILON));

        let Some(single) = super::plan(&passes(1), [200, 100], window, [0, 0], NO_TEXTURES, None) else { panic!("one pass") };
        assert_eq!(single.composite.bindings.src, Slot::Ping);
        assert!(super::plan(&[], [200, 100], window, [0, 0], NO_TEXTURES, None).is_none(), "no passes, nothing to draw");
        assert!(super::plan(&passes(1), [200, 100], PxRect { x: 0, y: 0, w: 0, h: 5 }, [0, 0], NO_TEXTURES, None).is_none(), "empty window");
        assert!(super::plan(&passes(1), [200, 100], PxRect { x: -1, y: 0, w: 5, h: 5 }, [0, 0], NO_TEXTURES, None).is_none(), "negative origin");
    }

    #[test]
    fn uniform_blocks_carry_the_mappings_of_their_bindings() {
        let effect = identity();
        let passes: Vec<Pass> = (0..3).map(|index| Pass::new(&effect).with_param(0, [f32::from(u8::try_from(index).unwrap_or(0)); 4]).unwrap_or_else(|error| panic!("{error}"))).collect();
        // Capture 64x32; window (16, 8) 16x16; textures: source 32x32, ping 64x16, pong 32x64.
        let allocations = Allocations { source: [32, 32], ping_pong: [[64, 16], [32, 64]], state: [[0, 0]; 2] };
        let Some(plan) = plan(&passes, [64, 32], PxRect { x: 16, y: 8, w: 16, h: 16 }, [4, 8], NO_TEXTURES, None) else { panic!("planned") };
        let frame = FrameUniforms { aux_scale: [3.0, 3.0], texel: [0.25, 0.5], ..FrameUniforms::default() };
        let blocks: Vec<[u8; UNIFORM_BLOCK_SIZE]> = plan.steps.iter().map(|step| step_uniforms(&frame, &plan, step, &allocations, &passes[step.pass])).collect();
        // Pass 0 reads the source, pass 1 ping, pass 2 pong: capture / allocated, origin / allocated.
        assert!(holds(&blocks[0], offsets::SRC_SCALE, [2.0, 1.0]) && holds(&blocks[0], offsets::SRC_OFFSET, [0.5, 0.25]));
        assert!(holds(&blocks[1], offsets::SRC_SCALE, [1.0, 2.0]) && holds(&blocks[1], offsets::SRC_OFFSET, [0.25, 0.5]));
        assert!(holds(&blocks[2], offsets::SRC_SCALE, [2.0, 0.5]) && holds(&blocks[2], offsets::SRC_OFFSET, [0.5, 0.125]));
        // Every pass maps its original through the SOURCE's mapping, not its `src` slot's.
        assert!(blocks.iter().all(|block| holds(block, offsets::AUX_SCALE, [2.0, 1.0]) && holds(block, offsets::AUX_OFFSET, [0.5, 0.25])));
        // Full resolution: both texels are the frame's, the pass scale 1.
        assert!(blocks.iter().all(|block| holds(block, offsets::SRC_TEXEL, [0.25, 0.5]) && holds(block, offsets::FRAG_TEXEL, [0.25, 0.5]) && (read(block, offsets::PASS_SCALE) - 1.0).abs() < f32::EPSILON));
        let dummy_draw = Draw { bindings: Bindings { src: Slot::Ping, original: None, state: None, user: NO_TEXTURES }, input: plan.source, original: plan.source, frag_offset: [0.0; 2], written: None, scale: 1.0, state: None, pass_index: 0 };
        let dummy = draw_uniforms(&frame, &dummy_draw, &allocations, passes[0].params());
        assert!(holds(&dummy, offsets::AUX_SCALE, [3.0, 3.0]), "a dummy original keeps the frame's aux mapping");
        assert!(holds(&blocks[2], offsets::PARAMS, [2.0, 2.0]));
        assert!(holds(&blocks[1], offsets::FRAG_OFFSET, [16.0, 8.0]));

        let geometry = LayerGeometry {
            region: PxRect { x: 0, y: 0, w: 10, h: 10 },
            capture: PxRect { x: 0, y: 0, w: 10, h: 10 },
            rect_uv: [0.0; 4],
            region_uv: [9.0; 4],
            mask_rect_px: [0.0; 4],
            mask_radii_px: [0.0; 4],
            pixels_per_layer_point: 1.0,
        };
        let composite = composite_uniforms(&frame, &plan, &allocations, &geometry);
        // The composite reads pass 2's output (ping) and the source as the original.
        assert!(holds(&composite, offsets::SRC_SCALE, [1.0, 2.0]) && holds(&composite, offsets::SRC_OFFSET, [0.25, 0.5]));
        assert!(holds(&composite, offsets::AUX_SCALE, [2.0, 1.0]) && holds(&composite, offsets::AUX_OFFSET, [0.5, 0.25]));
        assert!(holds(&composite, offsets::PARAMS + 2 * 16, [9.0, 9.0]), "composite params slot 2 = region uv");
        assert!(holds(&composite, offsets::FRAG_OFFSET, [-4.0, -8.0]));
    }

    /// F1: every draw of every plan binds the untouched capture as its original, so `original()`
    /// reads the backdrop in any pass, whatever `src` is at that point of the chain.
    #[test]
    fn every_step_binds_the_capture_as_its_original() {
        for pass_count in 1..=5 {
            for window in [PxRect { x: 0, y: 0, w: 64, h: 32 }, PxRect { x: 16, y: 8, w: 7, h: 9 }] {
                let Some(plan) = plan(&passes(pass_count), [64, 32], window, [3, 5], NO_TEXTURES, None) else { panic!("{pass_count} passes over a non-empty window") };
                assert_eq!(plan.steps.len(), pass_count);
                assert!(plan.steps.iter().all(|step| step.bindings.original == Some(Slot::Source)), "{pass_count} passes: every step maps the original");
                assert_eq!(plan.composite.bindings.original, Some(Slot::Source));
            }
        }
    }

    /// F6: scaled sizes round UP (a partial pixel still covers capture), never reach 0 and never
    /// exceed the capture.
    #[test]
    fn scaled_sizes_round_up_at_odd_sizes() {
        assert_eq!(scaled_px(100, 0.5), 50);
        assert_eq!(scaled_px(101, 0.5), 51);
        assert_eq!(scaled_px(1, 0.5), 1);
        assert_eq!(scaled_px(7, 0.25), 2);
        assert_eq!(scaled_px(1000, crate::pass::MIN_PASS_SCALE), 1);
        assert_eq!(scaled_px(3000, crate::pass::MIN_PASS_SCALE), 3);
        assert_eq!(scaled_px(333, 1.0), 333);
        // 0.1_f32 is slightly above 1/10; its rounding error must not add a pixel.
        assert_eq!(scaled_px(10, 0.1), 1);
        assert_eq!(scaled_px(30, 0.1), 3);
        assert_eq!(scaled_px(31, 0.1), 4);
    }

    /// F6: per-step output sizes, what each step reads and writes, the mappings and texels its
    /// block carries, and the pool size, for a down / down / up / full chain over an odd capture.
    #[test]
    fn scaled_steps_map_their_inputs_and_outputs() {
        let effect = identity();
        let scales = [0.5, 0.25, 0.5, 1.0];
        let chain: Vec<Pass> = scales.iter().map(|&scale| Pass::new(&effect).with_scale(scale)).collect();
        let capture = [101, 51];
        let whole = PxRect { x: 0, y: 0, w: 101, h: 51 };
        let Some(plan) = plan(&chain, capture, whole, [7, 9], NO_TEXTURES, None) else { panic!("a scaled chain over the whole capture") };
        let used: Vec<[i32; 2]> = plan.steps.iter().map(|step| step.used_px).collect();
        assert_eq!(used, [[51, 26], [26, 13], [51, 26], [101, 51]]);
        assert_eq!(plan.target_px, [101, 51]);
        assert!(plan.steps.iter().take(3).all(|step| step.frag_offset == [0.0, 0.0] && !step.written.full && step.written.origin_px == [0, 0]));
        assert_eq!(plan.steps[1].input, plan.steps[0].written, "each step reads what the previous one wrote");
        assert_eq!(plan.steps[3].written, plan.source, "a full step after scaled ones writes the capture resolution");
        assert_eq!(plan.composite.input, plan.source);

        let allocations = Allocations { source: [101, 51], ping_pong: [[128, 64], [101, 51]], state: [[0, 0]; 2] };
        let frame = FrameUniforms { texel: [1.0 / 101.0, 1.0 / 51.0], ..FrameUniforms::default() };
        let blocks: Vec<[u8; UNIFORM_BLOCK_SIZE]> = plan.steps.iter().map(|step| step_uniforms(&frame, &plan, step, &allocations, &chain[step.pass])).collect();
        // Step 0: full-resolution input (the source), 51x26 output.
        assert!(holds(&blocks[0], offsets::SRC_TEXEL, frame.texel) && holds(&blocks[0], offsets::FRAG_TEXEL, [1.0 / 51.0, 1.0 / 26.0]));
        assert!((read(&blocks[0], offsets::PASS_SCALE) - 0.5).abs() < f32::EPSILON);
        // Step 1 reads step 0's 51x26 image from ping (128x64): src_px = one pixel of THAT image.
        assert!(holds(&blocks[1], offsets::SRC_SCALE, [51.0 / 128.0, 26.0 / 64.0]) && holds(&blocks[1], offsets::SRC_OFFSET, [0.0, 0.0]));
        assert!(holds(&blocks[1], offsets::SRC_TEXEL, [1.0 / 51.0, 1.0 / 26.0]) && holds(&blocks[1], offsets::FRAG_TEXEL, [1.0 / 26.0, 1.0 / 13.0]));
        // Step 3 (full) reads step 2's 51x26 image from ping (128x64) and writes full resolution.
        assert_eq!(plan.steps[3].bindings.src, Slot::Ping);
        assert!(holds(&blocks[3], offsets::SRC_SCALE, [51.0 / 128.0, 26.0 / 64.0]) && holds(&blocks[3], offsets::FRAG_TEXEL, frame.texel));
        // Step 2 reads step 1's 26x13 image from pong (101x51).
        assert!(holds(&blocks[2], offsets::SRC_SCALE, [26.0 / 101.0, 13.0 / 51.0]) && holds(&blocks[2], offsets::SRC_TEXEL, [1.0 / 26.0, 1.0 / 13.0]));
        // The original is always the capture at full resolution.
        assert!(blocks.iter().all(|block| holds(block, offsets::AUX_SCALE, [1.0, 1.0])));

        // Ending on a scaled pass: the composite upsamples it (maps the small image, full texel).
        let Some(ending) = super::plan(&chain[..2], capture, whole, [0, 0], NO_TEXTURES, None) else { panic!("planned") };
        assert_eq!(ending.composite.input, ending.steps[1].written);
        assert_eq!(ending.target_px, [101, 51], "the pool is the window, also without a full-resolution step");
        assert!(ending.steps.iter().all(|step| step.used_px[0] <= 101 && step.used_px[1] <= 51));
        let geometry = LayerGeometry { region: whole, capture: whole, rect_uv: [0.0; 4], region_uv: [0.0; 4], mask_rect_px: [0.0; 4], mask_radii_px: [0.0; 4], pixels_per_layer_point: 1.0 };
        let composite = composite_uniforms(&frame, &ending, &allocations, &geometry);
        assert_eq!(ending.composite.bindings.src, Slot::Pong);
        assert!(holds(&composite, offsets::SRC_SCALE, [26.0 / 101.0, 13.0 / 51.0]) && holds(&composite, offsets::SRC_TEXEL, [1.0 / 26.0, 1.0 / 13.0]));
        assert!(holds(&composite, offsets::FRAG_TEXEL, frame.texel) && (read(&composite, offsets::PASS_SCALE) - 1.0).abs() < f32::EPSILON);

        // A scaled chain over a window of the capture cannot be planned (tiling refuses it).
        assert!(super::plan(&chain, capture, PxRect { x: 0, y: 0, w: 50, h: 51 }, [0, 0], NO_TEXTURES, None).is_none());
        assert!(super::plan(&chain, capture, PxRect { x: 1, y: 0, w: 100, h: 51 }, [0, 0], NO_TEXTURES, None).is_none());
        // Unscaled chains keep planning windows.
        assert!(super::plan(&passes(2), capture, PxRect { x: 1, y: 0, w: 100, h: 51 }, [0, 0], NO_TEXTURES, None).is_some());
    }

    /// F4: a step binds its pass's named block (layout-sized), a module without `struct Params`
    /// and the composite bind the 16-byte zero block.
    #[test]
    fn params_blocks_are_the_pass_block_or_the_empty_one() {
        let Ok(plain) = Effect::new("plain", "fn effect(uv: vec2<f32>) -> vec4<f32> { return src(uv); }") else { panic!("identity must compile") };
        let named = Effect::new("named", "struct Params { tint: vec4<f32>, gain: f32 }\nfn effect(uv: vec2<f32>) -> vec4<f32> { return src(uv) * p.tint * p.gain; }");
        let Ok(named) = named else { panic!("{named:?}") };
        assert_eq!(params_block(&Pass::new(&plain)), &EMPTY_PARAMS_BLOCK);
        let pass = Pass::new(&named).with("gain", 2.0).unwrap_or_else(|error| panic!("{error}"));
        let block = params_block(&pass);
        assert_eq!(block.len(), 32, "span 20 padded to 32");
        assert!(holds(block, 16, [2.0, 0.0]));
        assert_eq!(composite_params_block(), &EMPTY_PARAMS_BLOCK);
    }

    /// F8: every pass step binds the attached user textures, the composite dummies; a chain whose
    /// effect reads a slot with nothing attached is refused, naming the pass and the slot.
    #[test]
    fn user_textures_bind_where_attached_and_are_required_where_read() {
        let reads_one = Effect::new("one", "fn effect(uv: vec2<f32>) -> vec4<f32> { return src(uv) * tex(1u, uv); }");
        let Ok(reads_one) = reads_one else { panic!("{reads_one:?}") };
        let chain = vec![Pass::new(&identity()), Pass::new(&reads_one)];
        let window = PxRect { x: 0, y: 0, w: 8, h: 8 };
        let attached = [false, true, false];
        let Some(plan) = plan(&chain, [8, 8], window, [0, 0], attached, None) else { panic!("planned") };
        assert!(plan.steps.iter().all(|step| step.bindings.user == attached));
        assert_eq!(plan.composite.bindings.user, NO_TEXTURES);
        assert_eq!(check_textures(&chain, attached), Ok(()));
        assert_eq!(check_textures(&chain, [true, false, true]), Err(TextureError::NotAttached { effect: "one".to_owned(), pass: 1, slot: 1 }));
        assert_eq!(check_textures(&passes(2), NO_TEXTURES), Ok(()), "effects reading no texture need none");
    }

    /// A state pass effect writing `state + 0.25`.
    fn stepper() -> Effect {
        match Effect::new_state("step", "fn effect(uv: vec2<f32>) -> vec4<f32> { return state(uv) + vec4<f32>(0.25); }") {
            Ok(effect) => effect,
            Err(error) => panic!("the state stepper must compile: {error}"),
        }
    }

    /// F7: state steps read the latest state and write the other state slot (swapping after each
    /// step: substeps read what the previous one wrote), keep the display input for the pass after
    /// them, number themselves apart from display steps, and display steps read the latest state.
    #[test]
    fn state_steps_ping_pong_the_state_and_keep_the_display_input() {
        let (display, state) = (identity(), stepper());
        let chain = vec![Pass::new(&display), Pass::new(&state), Pass::new(&state), Pass::new(&display)];
        let capture = [101, 51];
        let whole = PxRect { x: 0, y: 0, w: 101, h: 51 };
        let input = StatePlanInput { scale: 0.5, current: 1, previous_px: Some([30, 20]), step: true };
        let Some(plan) = plan(&chain, capture, whole, [0, 0], NO_TEXTURES, Some(input)) else { panic!("a stateful chain over the whole capture") };
        let kinds: Vec<(StepKind, u32)> = plan.steps.iter().map(|step| (step.kind, step.pass_index)).collect();
        assert_eq!(kinds, [(StepKind::Display, 0), (StepKind::State, 0), (StepKind::State, 1), (StepKind::Display, 1)]);
        let wiring: Vec<(Slot, Option<Slot>, Slot)> = plan.steps.iter().map(|step| (step.bindings.src, step.bindings.state, step.output)).collect();
        assert_eq!(
            wiring,
            [(Slot::Source, Some(Slot::StateB), Slot::Ping), (Slot::Ping, Some(Slot::StateB), Slot::StateA), (Slot::Ping, Some(Slot::StateA), Slot::StateB), (Slot::Ping, Some(Slot::StateB), Slot::Pong)],
            "the display input does not advance past state steps; each state step reads what the previous wrote"
        );
        let previous = Content { logical_px: [30, 20], origin_px: [0, 0], full: false };
        let stepped = Content { logical_px: [51, 26], origin_px: [0, 0], full: false };
        assert_eq!(plan.steps.iter().map(|step| step.state).collect::<Vec<_>>(), [Some(previous), Some(previous), Some(stepped), Some(stepped)]);
        assert!(plan.steps[1..3].iter().all(|step| step.written == stepped && step.used_px == [51, 26] && step.frag_offset == [0.0, 0.0] && (step.scale - 0.5).abs() < f32::EPSILON));
        assert_eq!(plan.steps[1].input, plan.steps[0].written, "a state pass's src is the display chain's input");
        assert_eq!(plan.composite.bindings, Bindings { src: Slot::Pong, original: Some(Slot::Source), state: None, user: NO_TEXTURES });
        assert_eq!(plan.state, Some(StatePlan { clear: None, target_px: [51, 26], final_index: 1, final_px: [51, 26] }));

        // Uniforms: a state read maps the content it reads (previous size first, new after).
        let allocations = Allocations { source: [101, 51], ping_pong: [[101, 51]; 2], state: [[64, 32], [32, 32]] };
        let frame = FrameUniforms { texel: [1.0 / 101.0, 1.0 / 51.0], ..FrameUniforms::default() };
        let blocks: Vec<[u8; UNIFORM_BLOCK_SIZE]> = plan.steps.iter().map(|step| step_uniforms(&frame, &plan, step, &allocations, &chain[step.pass])).collect();
        assert!(holds(&blocks[1], offsets::STATE_SCALE, [30.0 / 32.0, 20.0 / 32.0]) && holds(&blocks[1], offsets::STATE_OFFSET, [0.0, 0.0]));
        assert!(holds(&blocks[1], offsets::STATE_SIZE, [30.0, 20.0]) && holds(&blocks[1], offsets::STATE_TEXEL, [1.0 / 30.0, 1.0 / 20.0]));
        assert!(holds(&blocks[1], offsets::FRAG_TEXEL, [1.0 / 51.0, 1.0 / 26.0]) && (read(&blocks[1], offsets::PASS_SCALE) - 0.5).abs() < f32::EPSILON);
        assert!(holds(&blocks[2], offsets::STATE_SCALE, [51.0 / 64.0, 26.0 / 32.0]) && holds(&blocks[2], offsets::STATE_SIZE, [51.0, 26.0]));
        let pass_index = |block: &[u8; UNIFORM_BLOCK_SIZE]| u32::from_ne_bytes([block[offsets::PASS_INDEX], block[offsets::PASS_INDEX + 1], block[offsets::PASS_INDEX + 2], block[offsets::PASS_INDEX + 3]]);
        assert_eq!(blocks.iter().map(pass_index).collect::<Vec<_>>(), [0, 0, 1, 1]);
        assert!(holds(&blocks[3], offsets::STATE_SCALE, [51.0 / 32.0, 26.0 / 32.0]), "the last display step reads state B at the new size");
        let geometry = LayerGeometry { region: whole, capture: whole, rect_uv: [0.0; 4], region_uv: [0.0; 4], mask_rect_px: [0.0; 4], mask_radii_px: [0.0; 4], pixels_per_layer_point: 1.0 };
        let composite = composite_uniforms(&frame, &plan, &allocations, &geometry);
        assert!(holds(&composite, offsets::STATE_SIZE, frame.state_size) && holds(&composite, offsets::STATE_SCALE, frame.state[0]), "the composite binds the dummy state: the frame's neutral values");
    }

    /// F7: a new or reset state is cleared first (its readers see the new size); a repeated run
    /// in the same frame skips the state steps; a chain of state passes only composites its input.
    #[test]
    fn state_plans_clear_skip_and_refuse() {
        let (display, state) = (identity(), stepper());
        let chain = vec![Pass::new(&display), Pass::new(&state)];
        let whole = PxRect { x: 0, y: 0, w: 40, h: 20 };
        let fresh = StatePlanInput { scale: 1.0, current: 0, previous_px: None, step: true };
        let Some(cleared) = plan(&chain, [40, 20], whole, [0, 0], NO_TEXTURES, Some(fresh)) else { panic!("planned") };
        let full = Content { logical_px: [40, 20], origin_px: [0, 0], full: true };
        assert_eq!(cleared.state, Some(StatePlan { clear: Some(Slot::StateA), target_px: [40, 20], final_index: 1, final_px: [40, 20] }));
        assert_eq!(cleared.steps[0].state, Some(full), "readers before the first write see the cleared state at the new size");

        let repeat = StatePlanInput { scale: 1.0, current: 1, previous_px: Some([40, 20]), step: false };
        let Some(skipped) = plan(&chain, [40, 20], whole, [0, 0], NO_TEXTURES, Some(repeat)) else { panic!("planned") };
        assert_eq!(skipped.steps.len(), 1, "the state step is skipped");
        assert_eq!(skipped.steps[0].bindings.state, Some(Slot::StateB));
        assert_eq!(skipped.state, Some(StatePlan { clear: None, target_px: [40, 20], final_index: 1, final_px: [40, 20] }), "the state is left as it is");

        let Some(only_state) = plan(&[Pass::new(&state)], [40, 20], whole, [0, 0], NO_TEXTURES, Some(repeat)) else { panic!("planned") };
        assert!(only_state.steps.is_empty() && only_state.composite.bindings.src == Slot::Source);

        assert!(plan(&chain, [40, 20], whole, [0, 0], NO_TEXTURES, None).is_none(), "a state pass needs the state");
        assert!(plan(&chain, [40, 20], PxRect { x: 0, y: 0, w: 20, h: 20 }, [0, 0], NO_TEXTURES, Some(fresh)).is_none(), "stateful runs are never windowed");
        let Some(stateless) = plan(&chain[..1], [40, 20], whole, [0, 0], NO_TEXTURES, None) else { panic!("planned") };
        assert_eq!((stateless.steps[0].bindings.state, stateless.state), (None, None));
    }

    #[test]
    fn original_is_a_prelude_helper() {
        let snippet = "fn effect(uv: vec2<f32>) -> vec4<f32> { return mix(src(uv), original(uv), 0.5); }";
        assert!(Effect::new("original", snippet).is_ok(), "the interface check accepts `original()`");
    }

    #[test]
    fn texture_mappings_are_exact_ratios() {
        let whole = TextureMapping::of([300, 200], [600, 256], [0, 0]);
        assert_eq!(whole, TextureMapping { scale: [0.5, 0.781_25], offset: [0.0, 0.0] });
        let window = TextureMapping::of([300, 200], [128, 64], [64, 32]);
        assert_eq!(window.offset.map(f32::to_bits), [0.5_f32, 0.5].map(f32::to_bits));
        assert_eq!(TextureMapping::of([5, 5], [0, 0], [3, 3]), TextureMapping { scale: [1.0, 1.0], offset: [0.0, 0.0] }, "unallocated axes are neutral");
    }

    #[test]
    fn px_ratio_is_exact_and_safe() {
        assert!((px_ratio(300, 600) - 0.5).abs() < f32::EPSILON);
        assert!((px_ratio(7, 7) - 1.0).abs() < f32::EPSILON);
        assert!((px_ratio(5, 0) - 1.0).abs() < f32::EPSILON);
        // Beyond 2^16 (a whole image split into tiles) the sizes stay exact.
        assert!((px_ratio(100_000, 131_072) - 0.762_939_45).abs() < 1e-7);
        assert!((px_ratio(70_000, 65_536) - 70_000.0 / 65_536.0).abs() < f32::EPSILON);
        assert!((to_f32_exact((1 << 24) - 1) - 16_777_215.0).abs() < f32::EPSILON);
        assert!((to_f32_exact(u32::MAX) - 16_777_216.0).abs() < f32::EPSILON, "saturates at 2^24");
    }
}
