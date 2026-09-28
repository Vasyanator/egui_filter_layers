/*
File: plan.rs

Purpose:
The single, backend-neutral owner of HOW a chain is wired: which texture every draw of a run
reads at each binding, which one it writes, its viewport, and the complete uniform block it
uploads (storage-uv mappings, fragment offsets, parameters). Backends (wgpu `chain.rs`, glow
`render.rs`) only create objects, bind the texture a `Slot` names, upload the bytes produced here
and draw; they make no wiring decision of their own.

Key structures:
- `Slot`: a texture of one run (the capture or a ping-pong intermediate).
- `Bindings`: the slots one draw binds at `fl_src_texture` / `fl_original_texture` (`None` = the
  backend's neutral dummy).
- `PassStep`, `CompositeStep`, `ChainPlan`: the draws of one run, in order.
- `TextureMapping`, `SlotMappings`: capture uv -> storage uv per slot (`clamp(uv) * scale -
  offset`, the prelude's `internal_*_scale` / `internal_*_offset`).

Key functions:
- `plan()`: the draws of a run over a window of the capture.
- `TextureMapping::of()`: the ONE formula for a texture's storage-uv mapping.
- `step_uniforms()` / `composite_uniforms()`: the full `FlUniforms` block of one draw.

Notes:
Pass `i` reads the capture (i = 0) or pass `i - 1`'s output and writes `Ping` (even i) / `Pong`
(odd i); the composite reads the last output as `src`. EVERY draw (each pass and the composite)
binds the capture as the original (`original(uv)`, prelude), mapped with the capture's own
storage mapping (`internal_aux_*`), which may differ from the ping-pong `src` mapping (grow-only
targets of other sizes). Every other texture binding of the layout (state, coverage) is the
backend's dummy. Backend-specific uniform flags (`flip_y`, fragment mode) stay in the
`FrameUniforms` each backend passes in.
*/

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
}

/// One effect pass of a run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PassStep {
    /// Index into the chain's passes (whose effect and parameters this draw runs).
    pub pass: usize,
    /// The textures it reads.
    pub bindings: Bindings,
    /// The intermediate it renders into (at the target's origin).
    pub output: Slot,
    /// Viewport size in pixels, from the target's origin: the window of the capture it computes.
    pub used_px: [i32; 2],
    /// Added to the fragment position (target pixels) to get capture pixels (`internal_frag_offset`,
    /// read only in fragment mode 1).
    pub frag_offset: [f32; 2],
}

/// The composite draw of a run (the viewport is the caller's region in its own render target).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct CompositeStep {
    /// The textures it reads: the last pass output as `src`, the capture as the original.
    pub bindings: Bindings,
    /// Added to the fragment position (target pixels) to get capture pixels.
    pub frag_offset: [f32; 2],
}

/// Every draw of one chain run, in execution order.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ChainPlan {
    /// The effect passes (never empty).
    pub steps: Vec<PassStep>,
    /// The composite that writes the result.
    pub composite: CompositeStep,
}

/// Plans a run of `pass_count` passes over `window` (capture pixels; the whole capture when not
/// tiled), whose composite puts capture pixel `(0, 0)` at `composite_origin` of its render target
/// (target pixels, top-left origin).
///
/// Returns `None` when there is nothing to draw: no passes, or an empty / negative window.
#[must_use]
pub(crate) fn plan(pass_count: usize, window: PxRect, composite_origin: [i32; 2]) -> Option<ChainPlan> {
    if pass_count == 0 || window.w <= 0 || window.h <= 0 || window.x < 0 || window.y < 0 {
        return None;
    }
    let output_of = |index: usize| if index.is_multiple_of(2) { Slot::Ping } else { Slot::Pong };
    let pass_frag_offset = [px_f32(window.x), px_f32(window.y)];
    let steps = (0..pass_count)
        .map(|index| PassStep {
            pass: index,
            // Every pass may read the untouched capture through `original()` (F1).
            bindings: Bindings { src: if index == 0 { Slot::Source } else { output_of(index - 1) }, original: Some(Slot::Source) },
            output: output_of(index),
            used_px: window.size(),
            frag_offset: pass_frag_offset,
        })
        .collect();
    let composite = CompositeStep {
        bindings: Bindings { src: output_of(pass_count - 1), original: Some(Slot::Source) },
        frag_offset: [-px_f32(composite_origin[0]), -px_f32(composite_origin[1])],
    };
    Some(ChainPlan { steps, composite })
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

/// The storage mapping of every slot of one run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SlotMappings {
    /// Of [`Slot::Source`].
    pub source: TextureMapping,
    /// Of [`Slot::Ping`] and [`Slot::Pong`].
    pub ping_pong: [TextureMapping; 2],
}

impl SlotMappings {
    /// The mapping of `slot`.
    #[must_use]
    pub(crate) fn of(&self, slot: Slot) -> TextureMapping {
        match slot {
            Slot::Source => self.source,
            Slot::Ping => self.ping_pong[0],
            Slot::Pong => self.ping_pong[1],
        }
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

/// The uniform block of one draw: `frame` (the layer's values and the backend's flags) with the
/// `src` / original mappings of `bindings` and the draw's fragment offset, plus `params`.
fn draw_uniforms(frame: &FrameUniforms, bindings: Bindings, frag_offset: [f32; 2], maps: &SlotMappings, params: &Params) -> [u8; UNIFORM_BLOCK_SIZE] {
    let mut frame = *frame;
    let src = maps.of(bindings.src);
    frame.src_scale = src.scale;
    frame.src_offset = src.offset;
    // A dummy original is never sampled; its mapping keeps the frame's identity value.
    if let Some(original) = bindings.original {
        let aux = maps.of(original);
        frame.aux_scale = aux.scale;
        frame.aux_offset = aux.offset;
    }
    frame.frag_offset = frag_offset;
    pack_uniforms(&frame, params)
}

/// The complete `FlUniforms` block of `step`, which runs `pass`. `frame` is the backend's
/// `FrameUniforms::for_layer` value (with its `flip_y` / fragment-mode flags).
#[must_use]
pub(crate) fn step_uniforms(frame: &FrameUniforms, step: &PassStep, maps: &SlotMappings, pass: &Pass) -> [u8; UNIFORM_BLOCK_SIZE] {
    draw_uniforms(frame, step.bindings, step.frag_offset, maps, pass.params())
}

/// The complete `FlUniforms` block of `plan`'s composite: its bindings' mappings and the mask /
/// region parameters of `geometry` (`LayerGeometry::composite_params`).
#[must_use]
pub(crate) fn composite_uniforms(frame: &FrameUniforms, plan: &ChainPlan, maps: &SlotMappings, geometry: &LayerGeometry) -> [u8; UNIFORM_BLOCK_SIZE] {
    draw_uniforms(frame, plan.composite.bindings, plan.composite.frag_offset, maps, &geometry.composite_params())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::effect::Effect;
    use crate::pass::offsets;

    /// Whether the two `f32`s at `offset` equal `expected` (all test values are exact in f32).
    fn holds(bytes: &[u8], offset: usize, expected: [f32; 2]) -> bool {
        let read = |at: usize| f32::from_ne_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]);
        (read(offset) - expected[0]).abs() < f32::EPSILON && (read(offset + 4) - expected[1]).abs() < f32::EPSILON
    }

    #[test]
    fn steps_follow_the_ping_pong_order() {
        let window = PxRect { x: 16, y: 32, w: 100, h: 50 };
        let Some(plan) = plan(3, window, [4, 8]) else { panic!("three passes over a non-empty window") };
        let reads: Vec<Slot> = plan.steps.iter().map(|step| step.bindings.src).collect();
        let writes: Vec<Slot> = plan.steps.iter().map(|step| step.output).collect();
        assert_eq!(reads, [Slot::Source, Slot::Ping, Slot::Pong]);
        assert_eq!(writes, [Slot::Ping, Slot::Pong, Slot::Ping]);
        assert!(plan.steps.iter().enumerate().all(|(index, step)| step.pass == index && step.used_px == [100, 50]));
        assert_eq!(plan.composite.bindings, Bindings { src: Slot::Ping, original: Some(Slot::Source) });
        assert_eq!(plan.composite.frag_offset.map(f32::to_bits), [-4.0_f32, -8.0].map(f32::to_bits));
        assert_eq!(plan.steps[0].frag_offset.map(f32::to_bits), [16.0_f32, 32.0].map(f32::to_bits));

        let Some(single) = super::plan(1, window, [0, 0]) else { panic!("one pass") };
        assert_eq!(single.composite.bindings.src, Slot::Ping);
        assert!(super::plan(0, window, [0, 0]).is_none(), "no passes, nothing to draw");
        assert!(super::plan(1, PxRect { x: 0, y: 0, w: 0, h: 5 }, [0, 0]).is_none(), "empty window");
        assert!(super::plan(1, PxRect { x: -1, y: 0, w: 5, h: 5 }, [0, 0]).is_none(), "negative origin");
    }

    #[test]
    fn uniform_blocks_carry_the_mappings_of_their_bindings() {
        let Ok(effect) = Effect::new("id", "fn effect(uv: vec2<f32>) -> vec4<f32> { return src(uv); }") else { panic!("identity must compile") };
        let passes: Vec<Pass> = (0..3).map(|index| Pass::new(&effect).with_param(0, [f32::from(u8::try_from(index).unwrap_or(0)); 4]).unwrap_or_else(|error| panic!("{error}"))).collect();
        let map = |scale: [f32; 2], offset: [f32; 2]| TextureMapping { scale, offset };
        let maps = SlotMappings { source: map([0.5, 0.25], [0.125, 0.0]), ping_pong: [map([0.75, 0.75], [0.25, 0.5]), map([1.0, 0.5], [0.0, 0.25])] };
        let Some(plan) = plan(3, PxRect { x: 16, y: 32, w: 10, h: 10 }, [4, 8]) else { panic!("planned") };
        let frame = FrameUniforms { aux_scale: [3.0, 3.0], ..FrameUniforms::default() };
        let blocks: Vec<[u8; UNIFORM_BLOCK_SIZE]> = plan.steps.iter().map(|step| step_uniforms(&frame, step, &maps, &passes[step.pass])).collect();
        // Pass 0 reads the source, pass 1 ping, pass 2 pong.
        assert!(holds(&blocks[0], offsets::SRC_SCALE, [0.5, 0.25]) && holds(&blocks[0], offsets::SRC_OFFSET, [0.125, 0.0]));
        assert!(holds(&blocks[1], offsets::SRC_SCALE, [0.75, 0.75]) && holds(&blocks[1], offsets::SRC_OFFSET, [0.25, 0.5]));
        assert!(holds(&blocks[2], offsets::SRC_SCALE, [1.0, 0.5]) && holds(&blocks[2], offsets::SRC_OFFSET, [0.0, 0.25]));
        // Every pass maps its original through the SOURCE's mapping, not its `src` slot's.
        assert!(blocks.iter().all(|block| holds(block, offsets::AUX_SCALE, [0.5, 0.25]) && holds(block, offsets::AUX_OFFSET, [0.125, 0.0])));
        let dummy = draw_uniforms(&frame, Bindings { src: Slot::Ping, original: None }, [0.0; 2], &maps, passes[0].params());
        assert!(holds(&dummy, offsets::AUX_SCALE, [3.0, 3.0]), "a dummy original keeps the frame's aux mapping");
        assert!(holds(&blocks[2], offsets::PARAMS, [2.0, 2.0]));
        assert!(holds(&blocks[1], offsets::FRAG_OFFSET, [16.0, 32.0]));

        let geometry = LayerGeometry {
            region: PxRect { x: 0, y: 0, w: 10, h: 10 },
            capture: PxRect { x: 0, y: 0, w: 10, h: 10 },
            rect_uv: [0.0; 4],
            region_uv: [9.0; 4],
            mask_rect_px: [0.0; 4],
            mask_radii_px: [0.0; 4],
            pixels_per_layer_point: 1.0,
        };
        let composite = composite_uniforms(&frame, &plan, &maps, &geometry);
        // The composite reads pass 2's output (ping) and the source as the original.
        assert!(holds(&composite, offsets::SRC_SCALE, [0.75, 0.75]) && holds(&composite, offsets::SRC_OFFSET, [0.25, 0.5]));
        assert!(holds(&composite, offsets::AUX_SCALE, [0.5, 0.25]) && holds(&composite, offsets::AUX_OFFSET, [0.125, 0.0]));
        assert!(holds(&composite, offsets::PARAMS + 2 * 16, [9.0, 9.0]), "composite params slot 2 = region uv");
        assert!(holds(&composite, offsets::FRAG_OFFSET, [-4.0, -8.0]));
    }

    /// F1: every draw of every plan binds the untouched capture as its original, so `original()`
    /// reads the backdrop in any pass, whatever `src` is at that point of the chain.
    #[test]
    fn every_step_binds_the_capture_as_its_original() {
        for pass_count in 1..=5 {
            for window in [PxRect { x: 0, y: 0, w: 64, h: 32 }, PxRect { x: 16, y: 8, w: 7, h: 9 }] {
                let Some(plan) = plan(pass_count, window, [3, 5]) else { panic!("{pass_count} passes over a non-empty window") };
                assert_eq!(plan.steps.len(), pass_count);
                assert!(plan.steps.iter().all(|step| step.bindings.original == Some(Slot::Source)), "{pass_count} passes: every step maps the original");
                assert_eq!(plan.composite.bindings.original, Some(Slot::Source));
            }
        }
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
