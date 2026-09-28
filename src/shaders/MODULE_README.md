# Module: src/shaders

## Purpose
WGSL sources compiled into the crate with `include_str!` (by `effect.rs`).

## Files
- `prelude.wgsl`: prepended to every effect. Declares `FlUniforms` / `fl`, the WHOLE `@group(0)`
  resource layout (bindings 0-3 and 5-6; 4 is the named-parameter block `p: Params`, which
  `effect/` appends AFTER a snippet declaring `struct Params`), the user texture groups 1..=3
  (`fl_userN` at 0, `fl_userN_sampler` at 1: egui-wgpu's texture layout), the public helpers
  (`src`, `src_px`, `src_texel`, `out_texel`, `original`, `params`, `tex`, `tex_size`,
  `tex_load`, colour helpers), and both entry points
  (`fl_vertex`: attribute-less full-viewport triangle; `fl_fragment`: computes the fragment's
  capture uv with `fl_capture_uv` and calls the user's `effect`).
  It must stay BEFORE user code so snippet line numbers are a plain offset.
- `composite.wgsl`: body of the internal write-back pass, `fl_composite(frag_uv)` (reads the
  original capture through the prelude's `original()`, binding 3; params 0..2 = mask rect px, corner radii px, processed region in capture UV).
  In frag mode 0 the viewport is that region and the local uv maps through params[2]; in frag
  mode 1 `frag_uv` already is the capture uv. Helpers: `fl_composite_uv`, `fl_coverage` (snaps
  within 1/1024 of 0 / 1 to exactly 0 / 1), `fl_coverage_or_discard`. `effect.rs` appends one
  `effect` wrapper: the REPLACE composite `fl_composite` (gamma output, or `to_linear` for `*Srgb`
  targets; glow, offline), or one of the two draws of the BLENDED composite the wgpu UI uses
  (erase: `(0, 0, 0, coverage)`; add: `coverage * filtered`, linearised on `*Srgb`; both discard
  uncovered fragments; the blend states live in `wgpu_backend/chain.rs`).

## Contracts and invariants
- `FlUniforms` must match `pass::offsets` (test `offsets_match_the_wgsl_struct_layout`).
- Every resource binding is declared here, never in `composite.wgsl` or a snippet: the backends
  bind one fixed layout for every module (`effect::PRELUDE_RESOURCES`, `src/MODULE_README.md`).
- Unused declarations cost nothing: naga's GLSL writer drops globals (and functions using them)
  an entry point does not reach, and wgpu's explicit layout covers every binding anyway. On glow a
  texture must be read through ONE sampler per module (naga's GLSL writer rejects a texture used
  with two samplers).
- Two fragment-uv modes (`internal_frag_mode`): 0 = interpolated over the viewport (glow), 1 =
  `(position + internal_frag_offset) * internal_frag_texel` (wgpu; the draw's OUTPUT pixel,
  `fl.texel` at full resolution, larger for a scaled pass). Mode 1 makes a pixel's uv independent of
  which window of the capture a pass renders (offline tiles). Storage uv is always
  `clamp(uv) * internal_*_scale - internal_*_offset`; offsets are 0 for a whole-capture window, so
  glow and the UI see exactly the pre-window arithmetic. The edge clamp uses the SAMPLED image's
  pixel: `internal_src_texel` for `src` (a scaled input), `fl.texel` for `original`.
- Only `textureSampleLevel` (valid in non-uniform control flow, translates to `textureLod`) and
  `textureLoad` (`texelFetch`) read textures. The slot helpers `tex` / `tex_size` / `tex_load` take
  the slot as their FIRST argument and `switch` on it (a slot `>= 3` yields zero); `effect/`'s
  `texture_slots` relies on that signature and on these three being the only prelude functions
  that name `fl_user*` globals.
- Public names listed in the crate docs are API; `fl_*` / `internal_*` names are private.
