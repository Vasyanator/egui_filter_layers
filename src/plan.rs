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
- `CoverageStep`: a shape mask's coverage rasterization (F9), `ChainPlan::with_coverage`; with
  `regions > 0` also the region-id passes (R1), `ChainPlan::with_regions`.
- `TextureMapping`: capture uv -> storage uv of one texture content (`clamp(uv) * scale -
  offset`, the prelude's `internal_*_scale` / `internal_*_offset`).
- `CaptureSource`, `Op`: the steps of a multi-entry paint (R3) and what a backend's captures see.

Key functions:
- `plan()`: the draws of a run over a window of the capture.
- `check_textures()`: every user texture slot an effect of the chain reads has a texture
  attached (the one check the UI layer and the offline renderer run before drawing).
- `scaled_px()`: the output size of a scaled pass (`ceil(side * scale)`, at least 1).
- `TextureMapping::of()`: the ONE formula for a texture's storage-uv mapping.
- `step_uniforms()` / `composite_uniforms()` / `composite_step_uniforms()`: the full `FlUniforms`
  block of one draw (the last for a composite at another origin: `composite_for_origin()`).
- `schedule()`: the ONE order of captures, runs, `Stacked` hand-offs and composites of a
  multi-entry layer (R3) for every backend; `entry_state_key()`: one state per entry.
- `params_block()` / `composite_params_block()`: the bytes bound at binding 4 (`p: Params`) for a
  step / the composite; `EMPTY_PARAMS_BLOCK` (16 zero bytes) when a module declares none.
- `mesh_locals()` / `coverage_locals()` / `region_locals()`: the `MeshLocals` block of
  `shaders/mesh.wgsl` (the wgpu backdrop re-render; the coverage step and the region-id passes on
  every backend).

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
Shape mask (F9): `ChainPlan::with_coverage` adds a `CoverageStep` that rasterizes the mask's
meshes into `Slot::Coverage` before the passes; the texture holds the run's WINDOW at its origin
(like the source), so the composite maps it with the original's content in the coverage
allocation (`internal_cover_*`) and sets `params[3].x = 1` (`COVERAGE_MASK_FLAG`): the composite
reads the texture's alpha instead of the rounded-rect SDF. Only the composite binds it; every
other draw binds the backend's white dummy. Windows (offline tiles) rasterize their own window.
Regions (R1): `ChainPlan::with_regions(count)` is `with_coverage` (the union of the regions is the
coverage) plus two id passes over all regions in list order, run after the coverage draw into
`Slot::RegionId` (cleared to id 255 = `REGION_NONE`, no blend): pass A (`alpha_threshold` 0)
gives every pixel a region touches that region's id, pass B (threshold `0.5 - 1/128`) every pixel
a region covers by about half or more; the later region wins either way (the margin makes a
mid-pixel shared edge, where the later region's coverage is 0.5 up to float noise, go to the later
region deterministically, independent of the window origin). So no pixel of the mask's antialiased
edge keeps the sentinel, and interior boundaries sit at the later region's 50 % contour. Every
PASS step binds the region texture (mapped like the coverage: the window in its own allocation,
`internal_region_*`); the composite does not read it (the white dummy = `REGION_NONE`).
Entries (R3): a layer's entries are separate runs, each with its own plan over its own capture.
`schedule` orders them; a `Stacked` hand-off on a pristine source is the replace composite of an
earlier entry drawn into a later entry's capture (`composite_for_origin`, viewport = the earlier
region ∩ the later capture), so the later entry filters `mix(capture, earlier, coverage)`. Outside
the earlier mask it writes the earlier capture, which holds the same backdrop there.
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
use crate::entry::Overlap;
use crate::geometry::{LayerGeometry, MaskTransform, PxRect, px_f32};
use crate::pass::{FrameUniforms, Params, Pass, UNIFORM_BLOCK_SIZE, pack_uniforms};
use crate::state::StateKey;

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
    /// The mask coverage texture of a shape mask (F9): the run's window at the texture origin,
    /// rasterized by the plan's [`CoverageStep`] before the composite reads it.
    Coverage,
    /// The region-id texture of a region mask (R1): the run's window at the texture origin,
    /// `.r * 255` = the id of the region owning the pixel (255 = none), written by the coverage
    /// step's id passes; read by every pass step.
    RegionId,
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
    /// Bound at `fl_coverage_texture`: [`Slot::Coverage`] for the composite of a shape-masked run,
    /// `None` (the backend's 1x1 WHITE dummy) otherwise.
    pub coverage: Option<Slot>,
    /// Bound at `fl_region_texture`: [`Slot::RegionId`] for every pass step of a region run (R1),
    /// `None` (the backend's 1x1 WHITE dummy, i.e. `REGION_NONE`) otherwise and for the composite.
    pub region: Option<Slot>,
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

/// The coverage rasterization of a shape-masked run (F9), drawn before the passes: the mask's
/// meshes (`mask::PackedCoverage`) into [`Slot::Coverage`], cleared to transparent first, over
/// `(0, 0, used_px)` with the ADDITIVE, saturating blend (coverage = the summed alpha, so abutting
/// pieces leave no seam). The texture holds the run's WINDOW at its origin (like the source), at
/// full resolution. With `regions > 0` (R1) the id passes A and B follow into [`Slot::RegionId`]
/// (same window, same viewport; see the file notes and [`region_locals`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct CoverageStep {
    /// Viewport size in pixels from the target's origin: the window.
    pub used_px: [i32; 2],
    /// The window's origin in capture pixels (the mesh transform subtracts it).
    pub window_origin: [i32; 2],
    /// How many regions the id passes draw (their ids are `0..regions`); 0 = a plain shape mask,
    /// no id passes.
    pub regions: u32,
}

/// The coverage threshold of region-id pass A: every pixel a region touches at all.
pub(crate) const REGION_PASS_A_THRESHOLD: f32 = 0.0;
/// The coverage threshold of region-id pass B: every pixel a region covers by about half or more.
/// Just BELOW 0.5: on a shared edge through pixel centres both regions cover those pixels by 0.5
/// up to interpolation noise, which differs with the window origin (offline tiles, a grown UI
/// capture); with a threshold of exactly 0.5 the owner of such a pixel was decided by that noise.
/// The 1/128 margin (a contour shift of < 0.01 px) exceeds the noise, so the later region owns
/// the whole shared column / row, whatever the window.
pub(crate) const REGION_PASS_B_THRESHOLD: f32 = 0.5 - 1.0 / 128.0;
/// The id passes in execution order (their `alpha_threshold`s).
pub(crate) const REGION_PASS_THRESHOLDS: [f32; 2] = [REGION_PASS_A_THRESHOLD, REGION_PASS_B_THRESHOLD];

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
    /// The mask coverage step (shape-masked runs only, [`ChainPlan::with_coverage`]).
    pub coverage: Option<CoverageStep>,
}

impl ChainPlan {
    /// The plan of a SHAPE-masked run (F9): adds the coverage step (the window, like
    /// [`ChainPlan::source`]; it needs a [`Slot::Coverage`] texture of at least
    /// [`ChainPlan::target_px`]) and binds it as the composite's coverage, which switches the
    /// composite from the rounded-rect SDF to the texture's alpha (`composite_uniforms`).
    #[must_use]
    pub(crate) fn with_coverage(mut self) -> Self {
        let window_origin = self.source.origin_px.map(|side| i32::try_from(side).unwrap_or(i32::MAX));
        let used_px = self.target_px.map(|side| i32::try_from(side).unwrap_or(i32::MAX));
        self.coverage = Some(CoverageStep { used_px, window_origin, regions: 0 });
        self.composite.bindings.coverage = Some(Slot::Coverage);
        self
    }

    /// The plan of a REGION-masked run (R1) with `count` regions (ids `0..count`, at most
    /// `mask::MAX_REGIONS`): [`ChainPlan::with_coverage`] (the regions' union is the coverage)
    /// whose coverage step also runs the id passes into [`Slot::RegionId`] (a texture of at least
    /// [`ChainPlan::target_px`], like the coverage), bound by every pass step (not the composite).
    /// `count == 0` is a plain shape mask (no id passes, no region binding).
    #[must_use]
    pub(crate) fn with_regions(self, count: u32) -> Self {
        let mut plan = self.with_coverage();
        if count == 0 {
            return plan;
        }
        if let Some(step) = plan.coverage.as_mut() {
            step.regions = count;
        }
        for step in &mut plan.steps {
            step.bindings.region = Some(Slot::RegionId);
        }
        plan
    }
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
    /// Of [`Slot::Coverage`] (zero without a shape mask).
    pub coverage: [u32; 2],
    /// Of [`Slot::RegionId`] (zero without regions).
    pub region: [u32; 2],
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
            Slot::Coverage => self.coverage,
            Slot::RegionId => self.region,
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
        let bindings = Bindings { src: input_slot, original: Some(Slot::Source), state: state_binding, user: attached, coverage: None, region: None };
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
        bindings: Bindings { src: input_slot, original: Some(Slot::Source), state: None, user: [false; TEXTURE_SLOTS], coverage: None, region: None },
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
    Some(ChainPlan { steps, composite, source, target_px, state, coverage: None })
}

/// The composite of `plan`'s run drawn into a render target where capture pixel `(0, 0)` lies at
/// `composite_origin` (target pixels): `plan.composite` with that fragment offset. The run's own
/// composite uses the origin it was planned with; a `Stacked` hand-off ([`Op::HandOff`]) draws the
/// composite of an earlier entry into a later entry's capture, whose origin is
/// `capture_from.xy - capture_into.xy`.
#[must_use]
#[cfg_attr(not(feature = "wgpu"), allow(dead_code))] // TEMPORARY: read by the wgpu UI and offline; glow executes the schedule from roadmap phase R3b (this release), which removes this allow.
pub(crate) fn composite_for_origin(plan: &ChainPlan, composite_origin: [i32; 2]) -> CompositeStep {
    CompositeStep { frag_offset: [-px_f32(composite_origin[0]), -px_f32(composite_origin[1])], ..plan.composite }
}

/// Whether a backend's entry captures see the results of earlier composites of the same layer
/// (`Framebuffer`: glow copies the framebuffer inside the paint callback) or not (`Pristine`: the
/// wgpu re-render and the offline copy of the unfiltered frame).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(not(feature = "wgpu"), allow(dead_code))] // TEMPORARY: read by the wgpu UI and offline; glow executes the schedule from roadmap phase R3b (this release), which removes this allow.
pub(crate) enum CaptureSource {
    /// Captures copy the framebuffer: an entry captured after an earlier entry's composite sees it.
    Framebuffer,
    /// Captures hold the pristine backdrop: `Stacked` needs explicit hand-offs.
    Pristine,
}

/// One step of a multi-entry paint ([`schedule`]); indices are entry indices of the layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(not(feature = "wgpu"), allow(dead_code))] // TEMPORARY: read by the wgpu UI and offline; glow executes the schedule from roadmap phase R3b (this release), which removes this allow.
pub(crate) enum Op {
    /// Fill entry `k`'s capture (a framebuffer copy, a re-render or a copy of the frame).
    Capture(usize),
    /// Run entry `k`'s chain over its capture.
    Run(usize),
    /// `Stacked` on a `Pristine` source: draw the REPLACE composite of entry `from` (gamma, the
    /// capture format) into entry `into`'s capture, so `into` filters `from`'s result.
    HandOff {
        /// The earlier entry, already run.
        from: usize,
        /// The later entry, not run yet.
        into: usize,
        /// The draw's viewport in `into`'s capture pixels: `from`'s region ∩ `into`'s capture.
        viewport: PxRect,
        /// Where `from`'s capture pixel `(0, 0)` lies in `into`'s capture:
        /// `capture_from.xy - capture_into.xy` (the argument of [`composite_for_origin`]).
        origin: [i32; 2],
    },
    /// Composite entry `k` over the screen / output (its region, blended by its coverage).
    Composite(usize),
}

/// The ops of one multi-entry paint, in execution order. `geometry[k]` is entry `k`'s
/// `LayerGeometry` in screen pixels (`None`: the entry is invisible and gets no op). This is the
/// ONE place that decides the order of captures, runs, hand-offs and composites (roadmap §19.4.3):
///
/// * `Framebuffer` + `Stacked`: `[Capture0 Run0 Composite0, Capture1 Run1 Composite1, …]` — every
///   capture copies the framebuffer after the earlier composites.
/// * `Framebuffer` + `Independent`: `[Capture0..CaptureN, Run0 Composite0, Run1 Composite1, …]` —
///   every capture before any composite.
/// * `Pristine` (either mode): `[Capture0..CaptureN, Run0 HandOff(0→k)* Composite0, Run1 …]`, the
///   hand-offs only for `Stacked`, into every later visible entry `k` whose capture intersects
///   entry 0's region (in list order of `k`).
///
/// Composites are always in list order. A backend that defers composites (the wgpu UI draws them
/// in its paint callback) keeps that order.
#[must_use]
#[cfg_attr(not(feature = "wgpu"), allow(dead_code))] // TEMPORARY: read by the wgpu UI and offline; glow executes the schedule from roadmap phase R3b (this release), which removes this allow.
pub(crate) fn schedule(geometry: &[Option<LayerGeometry>], overlap: Overlap, source: CaptureSource) -> Vec<Op> {
    let visible: Vec<usize> = geometry.iter().enumerate().filter_map(|(index, geometry)| geometry.map(|_| index)).collect();
    let mut ops = Vec::with_capacity(visible.len() * 3);
    let capture_first = !(source == CaptureSource::Framebuffer && overlap == Overlap::Stacked);
    if capture_first {
        ops.extend(visible.iter().map(|&index| Op::Capture(index)));
    }
    for (position, &from) in visible.iter().enumerate() {
        if !capture_first {
            ops.push(Op::Capture(from));
        }
        ops.push(Op::Run(from));
        if source == CaptureSource::Pristine && overlap == Overlap::Stacked
            && let Some(from_geometry) = geometry[from]
        {
            for &into in &visible[position + 1..] {
                let Some(into_geometry) = geometry[into] else {
                    continue;
                };
                if let Some(viewport) = intersect(from_geometry.region, into_geometry.capture) {
                    let (capture_from, capture_into) = (from_geometry.capture, into_geometry.capture);
                    ops.push(Op::HandOff {
                        from,
                        into,
                        viewport: PxRect { x: viewport.x - capture_into.x, y: viewport.y - capture_into.y, w: viewport.w, h: viewport.h },
                        origin: [capture_from.x - capture_into.x, capture_from.y - capture_into.y],
                    });
                }
            }
        }
        ops.push(Op::Composite(from));
    }
    ops
}

/// The intersection of two pixel rects, `None` when it is empty (touching edges do not overlap).
#[must_use]
#[cfg_attr(not(feature = "wgpu"), allow(dead_code))] // TEMPORARY: read by the wgpu UI and offline; glow executes the schedule from roadmap phase R3b (this release), which removes this allow.
fn intersect(a: PxRect, b: PxRect) -> Option<PxRect> {
    let left = a.x.max(b.x);
    let top = a.y.max(b.y);
    let right = a.x.saturating_add(a.w).min(b.x.saturating_add(b.w));
    let bottom = a.y.saturating_add(a.h).min(b.y.saturating_add(b.h));
    (right > left && bottom > top).then(|| PxRect { x: left, y: top, w: right - left, h: bottom - top })
}

/// The state key of entry `index` of a layer whose key is `layer` (D5): `layer.id.with(index)`,
/// same viewport. Every entry keeps its own state; `reset_state(ctx, layer id)` reaches them all
/// through the layer's one reset ticket.
#[must_use]
pub(crate) fn entry_state_key(layer: StateKey, index: usize) -> StateKey {
    StateKey { viewport: layer.viewport, id: layer.id.with(index) }
}

/// Checks that every user texture slot an effect of `passes` (the chain of layer entry `entry`)
/// may read (`Effect::reads_texture`) has a texture attached (`attached[i]`).
///
/// # Errors
/// [`TextureError::NotAttached`] naming `entry`, the first pass (in chain order) and the slot
/// that lacks one.
pub(crate) fn check_textures(passes: &[Pass], attached: [bool; TEXTURE_SLOTS], entry: usize) -> Result<(), TextureError> {
    for (pass_index, pass) in passes.iter().enumerate() {
        if let Some(slot) = (0..TEXTURE_SLOTS).find(|&slot| pass.effect().reads_texture(slot) && !attached[slot]) {
            return Err(TextureError::NotAttached { entry, effect: pass.effect().label().to_owned(), pass: pass_index, slot });
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
    // The coverage holds the window like the source does (same content, its own allocation).
    if let Some(slot) = draw.bindings.coverage {
        let mapping = draw.original.mapping(allocations.of(slot));
        out.cover = [mapping.scale, mapping.offset];
    }
    // The region ids hold the window too, in their own allocation.
    if let Some(slot) = draw.bindings.region {
        let mapping = draw.original.mapping(allocations.of(slot));
        out.region = [mapping.scale, mapping.offset];
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
/// region parameters of `geometry` (`LayerGeometry::composite_params`). A composite that binds a
/// coverage texture (a shape mask) also sets `params[3].x = 1` ([`COVERAGE_MASK_FLAG`]): the
/// composite then takes the coverage from that texture's alpha instead of the rounded-rect SDF.
#[must_use]
pub(crate) fn composite_uniforms(frame: &FrameUniforms, plan: &ChainPlan, allocations: &Allocations, geometry: &LayerGeometry) -> [u8; UNIFORM_BLOCK_SIZE] {
    composite_step_uniforms(frame, plan, &plan.composite, allocations, geometry)
}

/// [`composite_uniforms`] for another composite draw of `plan`'s run (`composite`, e.g. a
/// hand-off's [`composite_for_origin`]): the same bindings and parameters, its own fragment offset.
#[must_use]
pub(crate) fn composite_step_uniforms(frame: &FrameUniforms, plan: &ChainPlan, composite: &CompositeStep, allocations: &Allocations, geometry: &LayerGeometry) -> [u8; UNIFORM_BLOCK_SIZE] {
    let draw = Draw { bindings: composite.bindings, input: composite.input, original: plan.source, frag_offset: composite.frag_offset, written: None, scale: 1.0, state: None, pass_index: 0 };
    let mut params = geometry.composite_params();
    if composite.bindings.coverage.is_some() {
        params[COVERAGE_MASK_FLAG][0] = 1.0;
    }
    draw_uniforms(frame, &draw, allocations, &params)
}

/// The composite parameter slot whose `x` is 1 when the mask is the coverage texture
/// (`composite.wgsl`'s `fl_coverage`), 0 for the rounded-rect SDF.
pub(crate) const COVERAGE_MASK_FLAG: usize = 3;

/// Size of the mesh shader's `MeshLocals` block (`shaders/mesh.wgsl`; std140 = WGSL layout).
pub(crate) const MESH_LOCALS_SIZE: usize = 32;

/// The `MeshLocals` block of a mesh draw (`shaders/mesh.wgsl`, the backdrop re-render on wgpu and
/// the coverage rasterizer and region-id passes everywhere): a vertex at `pos` lands at pixel
/// `pos * scale - origin_px` of a viewport of `size_px` pixels; `dithering` as egui-wgpu (never
/// for coverage); `region_id` / `alpha_threshold` are read by `fs_region` only (0 elsewhere).
#[must_use]
pub(crate) fn mesh_locals(scale: f32, dithering: bool, origin_px: [f32; 2], size_px: [f32; 2], region_id: f32, alpha_threshold: f32) -> [u8; MESH_LOCALS_SIZE] {
    let mut bytes = [0_u8; MESH_LOCALS_SIZE];
    let fields: [[u8; 4]; 8] = [
        scale.to_ne_bytes(),
        u32::from(dithering).to_ne_bytes(),
        origin_px[0].to_ne_bytes(),
        origin_px[1].to_ne_bytes(),
        size_px[0].to_ne_bytes(),
        size_px[1].to_ne_bytes(),
        region_id.to_ne_bytes(),
        alpha_threshold.to_ne_bytes(),
    ];
    for (chunk, field) in bytes.chunks_exact_mut(4).zip(fields) {
        chunk.copy_from_slice(&field);
    }
    bytes
}

/// The `MeshLocals` block of `step` (the coverage rasterization of a run) for a mask whose layer
/// points map to capture pixels by `transform` (`geometry::MaskTransform`): the window's origin is
/// subtracted, so the texture holds the window at its origin like the source. No dithering.
#[must_use]
pub(crate) fn coverage_locals(transform: &MaskTransform, step: &CoverageStep) -> [u8; MESH_LOCALS_SIZE] {
    let origin = [transform.origin_px[0] + px_f32(step.window_origin[0]), transform.origin_px[1] + px_f32(step.window_origin[1])];
    mesh_locals(transform.scale, false, origin, [px_f32(step.used_px[0]), px_f32(step.used_px[1])], 0.0, 0.0)
}

/// The `MeshLocals` block of region `region`'s draws in the id pass with `threshold`
/// ([`REGION_PASS_THRESHOLDS`]): the coverage step's placement ([`coverage_locals`]) plus the id
/// (`region`, exact in `f32`; `fs_region` writes `region / 255`) and the threshold, at bytes 24 / 28.
#[must_use]
pub(crate) fn region_locals(transform: &MaskTransform, step: &CoverageStep, region: u32, threshold: f32) -> [u8; MESH_LOCALS_SIZE] {
    let mut bytes = coverage_locals(transform, step);
    // Ids are below `MAX_REGIONS` (255); anything larger saturates at the sentinel.
    let id = f32::from(u8::try_from(region).unwrap_or(u8::MAX));
    bytes[24..28].copy_from_slice(&id.to_ne_bytes());
    bytes[28..32].copy_from_slice(&threshold.to_ne_bytes());
    bytes
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
        assert_eq!(plan.composite.bindings, Bindings { src: Slot::Ping, original: Some(Slot::Source), state: None, user: NO_TEXTURES, coverage: None, region: None });
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
        let allocations = Allocations { source: [32, 32], ping_pong: [[64, 16], [32, 64]], state: [[0, 0]; 2], coverage: [0, 0], region: [0, 0] };
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
        let dummy_draw = Draw { bindings: Bindings { src: Slot::Ping, original: None, state: None, user: NO_TEXTURES, coverage: None, region: None }, input: plan.source, original: plan.source, frag_offset: [0.0; 2], written: None, scale: 1.0, state: None, pass_index: 0 };
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

        let allocations = Allocations { source: [101, 51], ping_pong: [[128, 64], [101, 51]], state: [[0, 0]; 2], coverage: [0, 0], region: [0, 0] };
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
        assert_eq!(check_textures(&chain, attached, 0), Ok(()));
        assert_eq!(check_textures(&chain, [true, false, true], 4), Err(TextureError::NotAttached { entry: 4, effect: "one".to_owned(), pass: 1, slot: 1 }));
        assert_eq!(check_textures(&passes(2), NO_TEXTURES, 0), Ok(()), "effects reading no texture need none");
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
        assert_eq!(plan.composite.bindings, Bindings { src: Slot::Pong, original: Some(Slot::Source), state: None, user: NO_TEXTURES, coverage: None, region: None });
        assert_eq!(plan.state, Some(StatePlan { clear: None, target_px: [51, 26], final_index: 1, final_px: [51, 26] }));

        // Uniforms: a state read maps the content it reads (previous size first, new after).
        let allocations = Allocations { source: [101, 51], ping_pong: [[101, 51]; 2], state: [[64, 32], [32, 32]], coverage: [0, 0], region: [0, 0] };
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

    /// F9: a shape-masked plan rasterizes the window into `Slot::Coverage` before the passes and
    /// binds it only for the composite, whose block carries the coverage mapping (the window like
    /// the original, in the coverage texture's own allocation) and the mask-kind flag; the passes'
    /// blocks are unchanged, and a rounded-rect plan has no coverage at all.
    #[test]
    fn a_shape_mask_adds_the_coverage_step_for_the_composite_only() {
        let window = PxRect { x: 16, y: 8, w: 48, h: 24 };
        let Some(rounded) = plan(&passes(2), [64, 32], window, [4, 8], NO_TEXTURES, None) else { panic!("planned") };
        assert_eq!((rounded.coverage, rounded.composite.bindings.coverage), (None, None));
        let masked = rounded.clone().with_coverage();
        assert_eq!(masked.coverage, Some(CoverageStep { used_px: [48, 24], window_origin: [16, 8], regions: 0 }));
        assert_eq!(masked.composite.bindings.coverage, Some(Slot::Coverage));
        assert_eq!(masked.steps, rounded.steps, "the passes do not change");
        assert!(masked.steps.iter().all(|step| step.bindings.coverage.is_none()));

        let geometry = LayerGeometry { region: window, capture: PxRect { x: 0, y: 0, w: 64, h: 32 }, rect_uv: [0.0, 0.0, 1.0, 1.0], region_uv: [0.0, 0.0, 1.0, 1.0], mask_rect_px: [0.0, 0.0, 64.0, 32.0], mask_radii_px: [0.0; 4], pixels_per_layer_point: 1.0 };
        let frame = FrameUniforms::for_layer(&geometry, 0.0, false);
        let allocations = Allocations { source: [64, 32], ping_pong: [[64, 32]; 2], state: [[0, 0]; 2], coverage: [128, 64], region: [0, 0] };
        let flag = offsets::PARAMS + COVERAGE_MASK_FLAG * 16;
        let rounded_block = composite_uniforms(&frame, &rounded, &allocations, &geometry);
        assert!(read(&rounded_block, flag).abs() < f32::EPSILON && holds(&rounded_block, offsets::COVER_SCALE, [1.0, 1.0]), "rounded rect: SDF, identity cover mapping");
        let block = composite_uniforms(&frame, &masked, &allocations, &geometry);
        assert!((read(&block, flag) - 1.0).abs() < f32::EPSILON);
        // The window (48x24 at 16, 8) of a 64x32 capture in a 128x64 texture.
        assert!(holds(&block, offsets::COVER_SCALE, [0.5, 0.5]) && holds(&block, offsets::COVER_OFFSET, [0.125, 0.125]));
        let differing: Vec<usize> = (0..UNIFORM_BLOCK_SIZE / 4).map(|word| word * 4).filter(|&at| block[at..at + 4] != rounded_block[at..at + 4]).collect();
        assert_eq!(differing, [offsets::COVER_SCALE, offsets::COVER_SCALE + 4, offsets::COVER_OFFSET, offsets::COVER_OFFSET + 4, flag], "only the coverage mapping and the flag differ");
        for step in &masked.steps {
            assert_eq!(step_uniforms(&frame, &masked, step, &allocations, &passes(1)[0]), step_uniforms(&frame, &rounded, step, &allocations, &passes(1)[0]));
        }
    }

    /// The mesh locals: a layer point lands at `point * scale - origin`, the window origin
    /// subtracted so the coverage texture holds the window at its origin; no dithering.
    #[test]
    fn coverage_locals_follow_the_mesh_shader_layout() {
        let transform = MaskTransform { scale: 2.5, origin_px: [10.0, -4.0] };
        let bytes = coverage_locals(&transform, &CoverageStep { used_px: [48, 24], window_origin: [16, 8], regions: 0 });
        assert!((read(&bytes, 0) - 2.5).abs() < f32::EPSILON);
        assert_eq!(u32::from_ne_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]), 0, "coverage is never dithered");
        assert!(holds(&bytes, 8, [26.0, 4.0]) && holds(&bytes, 16, [48.0, 24.0]));
        assert!(bytes[24..].iter().all(|&byte| byte == 0));
    }

    /// R1: a region plan is the coverage plan plus the id passes; every pass step (state steps
    /// too) binds the region texture, the composite does not; zero regions = a plain shape mask.
    #[test]
    fn with_regions_binds_the_region_ids_on_every_pass_step_only() {
        let (display, state) = (identity(), stepper());
        let chain = vec![Pass::new(&display), Pass::new(&state), Pass::new(&display)];
        let whole = PxRect { x: 0, y: 0, w: 40, h: 20 };
        let input = StatePlanInput { scale: 1.0, current: 0, previous_px: Some([40, 20]), step: true };
        let Some(base) = plan(&chain, [40, 20], whole, [3, 4], NO_TEXTURES, Some(input)) else { panic!("planned") };
        let regions = base.clone().with_regions(3);
        assert_eq!(regions.coverage, Some(CoverageStep { used_px: [40, 20], window_origin: [0, 0], regions: 3 }));
        assert_eq!(regions.composite.bindings.coverage, Some(Slot::Coverage));
        assert_eq!(regions.composite.bindings.region, None, "the composite reads no region");
        assert_eq!(regions.steps.len(), 3);
        assert!(regions.steps.iter().all(|step| step.bindings.region == Some(Slot::RegionId)));
        assert!(regions.steps.iter().zip(&base.steps).all(|(with, without)| Bindings { region: None, ..with.bindings } == without.bindings), "nothing else changes");
        assert_eq!(base.clone().with_regions(0), base.clone().with_coverage(), "no regions: a plain shape mask");
    }

    /// R1: the region mapping is the window in the region texture's own allocation (equal to the
    /// coverage mapping when both allocations are equal, differing otherwise); steps without a
    /// region binding keep the identity.
    #[test]
    fn region_mappings_follow_their_own_allocation() {
        let window = PxRect { x: 16, y: 8, w: 48, h: 24 };
        let Some(base) = plan(&passes(2), [64, 32], window, [0, 0], NO_TEXTURES, None) else { panic!("planned") };
        let regions = base.clone().with_regions(2);
        let frame = FrameUniforms { texel: [1.0 / 64.0, 1.0 / 32.0], ..FrameUniforms::default() };
        let pass = &passes(1)[0];
        let equal = Allocations { source: [64, 32], ping_pong: [[64, 32]; 2], state: [[0, 0]; 2], coverage: [128, 64], region: [128, 64] };
        let block = step_uniforms(&frame, &regions, &regions.steps[0], &equal, pass);
        assert!(holds(&block, offsets::REGION_SCALE, [0.5, 0.5]) && holds(&block, offsets::REGION_OFFSET, [0.125, 0.125]));
        let geometry = LayerGeometry { region: window, capture: PxRect { x: 0, y: 0, w: 64, h: 32 }, rect_uv: [0.0, 0.0, 1.0, 1.0], region_uv: [0.0, 0.0, 1.0, 1.0], mask_rect_px: [0.0, 0.0, 64.0, 32.0], mask_radii_px: [0.0; 4], pixels_per_layer_point: 1.0 };
        let composite = composite_uniforms(&frame, &regions, &equal, &geometry);
        assert_eq!(block[offsets::REGION_SCALE..offsets::REGION_SCALE + 16], composite[offsets::COVER_SCALE..offsets::COVER_SCALE + 16], "equal allocations: region mapping = coverage mapping");
        let other = Allocations { region: [64, 32], ..equal };
        let block = step_uniforms(&frame, &regions, &regions.steps[0], &other, pass);
        assert!(holds(&block, offsets::REGION_SCALE, [1.0, 1.0]) && holds(&block, offsets::REGION_OFFSET, [0.25, 0.25]), "its own allocation");
        let plain = step_uniforms(&frame, &base, &base.steps[0], &equal, pass);
        assert!(holds(&plain, offsets::REGION_SCALE, frame.region[0]) && holds(&plain, offsets::REGION_OFFSET, frame.region[1]), "no region binding: the frame's identity");
        assert_eq!(Allocations { region: [7, 9], ..equal }.of(Slot::RegionId), [7, 9]);
    }

    /// R1: the id pass locals = the coverage placement plus the id and the threshold at bytes
    /// 24 / 28 (`MeshLocals.region_id`, `alpha_threshold`).
    #[test]
    fn region_locals_pack_the_id_and_the_threshold() {
        let transform = MaskTransform { scale: 2.0, origin_px: [1.0, 2.0] };
        let step = CoverageStep { used_px: [48, 24], window_origin: [16, 8], regions: 3 };
        let bytes = region_locals(&transform, &step, 2, REGION_PASS_B_THRESHOLD);
        assert_eq!(bytes[..24], coverage_locals(&transform, &step)[..24]);
        assert!((read(&bytes, 24) - 2.0).abs() < f32::EPSILON && (read(&bytes, 28) - REGION_PASS_B_THRESHOLD).abs() < f32::EPSILON);
        assert!(read(&region_locals(&transform, &step, 0, REGION_PASS_A_THRESHOLD), 28).abs() < f32::EPSILON);
        assert!((read(&region_locals(&transform, &step, 900, 0.0), 24) - 255.0).abs() < f32::EPSILON, "out-of-range ids saturate at the sentinel");
        assert_eq!(REGION_PASS_THRESHOLDS.map(f32::to_bits), [0.0_f32, 0.5 - 1.0 / 128.0].map(f32::to_bits));
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

    /// A geometry whose region is `region` and whose capture is `region` padded by `pad` pixels
    /// (only `region` / `capture` matter to the schedule).
    fn entry_geometry(region: PxRect, pad: i32) -> LayerGeometry {
        let capture = PxRect { x: region.x - pad, y: region.y - pad, w: region.w + 2 * pad, h: region.h + 2 * pad };
        LayerGeometry { region, capture, rect_uv: [0.0; 4], region_uv: [0.0; 4], mask_rect_px: [0.0; 4], mask_radii_px: [0.0; 4], pixels_per_layer_point: 1.0 }
    }

    /// R3 (§19.4.3): the three emitted orders for two overlapping entries.
    #[test]
    fn schedule_orders_follow_the_capture_source_and_mode() {
        use Op::{Capture, Composite, HandOff, Run};
        let geometry = [Some(entry_geometry(PxRect { x: 10, y: 10, w: 40, h: 20 }, 4)), Some(entry_geometry(PxRect { x: 30, y: 15, w: 40, h: 20 }, 2))];
        assert_eq!(schedule(&geometry, Overlap::Stacked, CaptureSource::Framebuffer), [Capture(0), Run(0), Composite(0), Capture(1), Run(1), Composite(1)]);
        assert_eq!(schedule(&geometry, Overlap::Independent, CaptureSource::Framebuffer), [Capture(0), Capture(1), Run(0), Composite(0), Run(1), Composite(1)]);
        assert_eq!(schedule(&geometry, Overlap::Independent, CaptureSource::Pristine), [Capture(0), Capture(1), Run(0), Composite(0), Run(1), Composite(1)]);
        // Region 0 (10..50, 10..30) ∩ capture 1 (28..72, 13..37) = (28..50, 13..30), in capture-1
        // pixels (0..22, 0..17); capture 0 starts at (6, 6): origin (6 - 28, 6 - 13).
        let handoff = HandOff { from: 0, into: 1, viewport: PxRect { x: 0, y: 0, w: 22, h: 17 }, origin: [-22, -7] };
        assert_eq!(schedule(&geometry, Overlap::Stacked, CaptureSource::Pristine), [Capture(0), Capture(1), Run(0), handoff, Composite(0), Run(1), Composite(1)]);
        assert!(schedule(&[], Overlap::Stacked, CaptureSource::Pristine).is_empty());
        // An invisible entry gets no op; indices stay the layer's.
        assert_eq!(schedule(&[None, geometry[1]], Overlap::Stacked, CaptureSource::Framebuffer), [Capture(1), Run(1), Composite(1)]);
    }

    /// Hand-offs only where the earlier REGION meets the later CAPTURE: touching rects and
    /// disjoint rects get none, a region inside a later capture gets one covering the region.
    #[test]
    fn hand_offs_need_the_earlier_region_to_meet_the_later_capture() {
        let stacked = |geometry: &[Option<LayerGeometry>]| schedule(geometry, Overlap::Stacked, CaptureSource::Pristine).into_iter().filter(|op| matches!(op, Op::HandOff { .. })).collect::<Vec<_>>();
        // Capture 1 starts exactly where region 0 ends (50): touching, no overlap.
        let touching = [Some(entry_geometry(PxRect { x: 10, y: 10, w: 40, h: 20 }, 0)), Some(entry_geometry(PxRect { x: 52, y: 10, w: 20, h: 20 }, 2))];
        assert!(stacked(&touching).is_empty());
        // The later entry's padding reaches into the earlier region: a hand-off of that strip.
        let padded = [Some(entry_geometry(PxRect { x: 10, y: 10, w: 40, h: 20 }, 0)), Some(entry_geometry(PxRect { x: 51, y: 10, w: 20, h: 20 }, 2))];
        assert_eq!(stacked(&padded), [Op::HandOff { from: 0, into: 1, viewport: PxRect { x: 0, y: 2, w: 1, h: 20 }, origin: [-39, 2] }]);
        let disjoint = [Some(entry_geometry(PxRect { x: 0, y: 0, w: 10, h: 10 }, 1)), Some(entry_geometry(PxRect { x: 100, y: 100, w: 10, h: 10 }, 1))];
        assert!(stacked(&disjoint).is_empty());
        // Region 0 fully inside capture 1: the whole region, at its offset in capture 1.
        let inside = [Some(entry_geometry(PxRect { x: 20, y: 20, w: 10, h: 10 }, 3)), Some(entry_geometry(PxRect { x: 10, y: 10, w: 40, h: 40 }, 0))];
        assert_eq!(stacked(&inside), [Op::HandOff { from: 0, into: 1, viewport: PxRect { x: 10, y: 10, w: 10, h: 10 }, origin: [7, 7] }]);
        // A LATER region over an EARLIER capture is no hand-off (only later entries read earlier ones).
        let reversed = [inside[1], inside[0]];
        assert!(stacked(&reversed).iter().all(|op| matches!(op, Op::HandOff { from: 0, into: 1, .. })));
    }

    /// Risk 1 of §19.10: three entries where entry 0 overlaps entry 2 but not entry 1, and entry 1
    /// overlaps entry 2: hand-offs 0→2 then 1→2, each right after its source's run.
    #[test]
    fn three_entries_hand_off_only_where_they_overlap() {
        use Op::{Capture, Composite, HandOff, Run};
        let geometry = [
            Some(entry_geometry(PxRect { x: 0, y: 0, w: 30, h: 30 }, 0)),
            Some(entry_geometry(PxRect { x: 60, y: 0, w: 30, h: 30 }, 0)),
            Some(entry_geometry(PxRect { x: 20, y: 10, w: 50, h: 10 }, 0)),
        ];
        let ops = schedule(&geometry, Overlap::Stacked, CaptureSource::Pristine);
        let zero_to_two = HandOff { from: 0, into: 2, viewport: PxRect { x: 0, y: 0, w: 10, h: 10 }, origin: [-20, -10] };
        let one_to_two = HandOff { from: 1, into: 2, viewport: PxRect { x: 40, y: 0, w: 10, h: 10 }, origin: [40, -10] };
        assert_eq!(ops, [Capture(0), Capture(1), Capture(2), Run(0), zero_to_two, Composite(0), Run(1), one_to_two, Composite(1), Run(2), Composite(2)]);
    }

    /// The hand-off composite is the run's composite at another origin; every entry keeps its own
    /// state key in the layer's viewport.
    #[test]
    fn hand_off_composites_and_entry_state_keys() {
        let window = PxRect { x: 0, y: 0, w: 30, h: 20 };
        let Some(plan) = plan(&passes(2), window.size(), window, [100, 50], NO_TEXTURES, None) else { panic!("planned") };
        let moved = composite_for_origin(&plan, [-7, 3]);
        assert_eq!((moved.bindings, moved.input), (plan.composite.bindings, plan.composite.input));
        assert_eq!(moved.frag_offset.map(f32::to_bits), [7.0_f32, -3.0].map(f32::to_bits));
        assert_eq!(composite_for_origin(&plan, [100, 50]), plan.composite, "its own origin gives the run's composite");
        let layer = StateKey { viewport: egui::ViewportId::ROOT, id: egui::Id::new("layer") };
        let keys = [entry_state_key(layer, 0), entry_state_key(layer, 1)];
        assert_ne!(keys[0], keys[1]);
        assert!(keys.iter().all(|key| key.viewport == layer.viewport && key.id != layer.id));
        assert_eq!(keys[1], entry_state_key(layer, 1), "stable across frames");
    }
}
