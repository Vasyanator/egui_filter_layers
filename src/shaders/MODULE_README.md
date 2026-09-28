# Module: src/shaders

## Purpose
WGSL sources compiled into the crate with `include_str!` (by `effect.rs`).

## Files
- `prelude.wgsl`: prepended to every effect. Declares `FlUniforms` / `fl`, the input texture and
  sampler, the public helpers (`src`, `src_px`, `params`, colour helpers), and both entry points
  (`fl_vertex`: attribute-less full-viewport triangle; `fl_fragment`: computes the fragment's
  capture uv with `fl_capture_uv` and calls the user's `effect`).
  It must stay BEFORE user code so snippet line numbers are a plain offset.
- `composite.wgsl`: body of the internal write-back pass, `fl_composite(frag_uv)` (binding 3 =
  original capture; params 0..2 = mask rect px, corner radii px, processed region in capture UV).
  In frag mode 0 the viewport is that region and the local uv maps through params[2]; in frag
  mode 1 `frag_uv` already is the capture uv. Helpers: `fl_composite_uv`, `fl_coverage` (snaps
  within 1/1024 of 0 / 1 to exactly 0 / 1), `fl_coverage_or_discard`. `effect.rs` appends one
  `effect` wrapper: the REPLACE composite `fl_composite` (gamma output, or `to_linear` for `*Srgb`
  targets; glow, offline), or one of the two draws of the BLENDED composite the wgpu UI uses
  (erase: `(0, 0, 0, coverage)`; add: `coverage * filtered`, linearised on `*Srgb`; both discard
  uncovered fragments; the blend states live in `wgpu_backend/chain.rs`).

## Contracts and invariants
- `FlUniforms` must match `pass::offsets` (test `offsets_match_the_wgsl_struct_layout`).
- Two fragment-uv modes (`internal_frag_mode`): 0 = interpolated over the viewport (glow), 1 =
  `(position + internal_frag_offset) * texel` (wgpu). Mode 1 makes a pixel's uv independent of
  which window of the capture a pass renders (offline tiles). Storage uv is always
  `clamp(uv) * internal_*_scale - internal_*_offset`; offsets are 0 for a whole-capture window, so
  glow and the UI see exactly the pre-window arithmetic.
- Only `textureSampleLevel` is used (valid in non-uniform control flow, translates to `textureLod`).
- Public names listed in the crate docs are API; `fl_*` / `internal_*` names are private.
