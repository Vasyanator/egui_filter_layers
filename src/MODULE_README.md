# Module: src

## Purpose
The whole `egui_filter_layers` library: a `FilterLayer` shape that redraws everything painted
beneath a rect through a chain of WGSL filter passes, on pluggable GPU backends.

## Architecture
```text
user code ──► FilterLayer::paint (layer.rs)
                 │  resolves FilterChain -> LayerJob (rect, mask, passes, padding, time, layer_id)
                 ▼
          backend registry plugin (backend.rs)       ──►  Arc<dyn Backend>
                 │                                         ├─ glow_backend/  (feature "glow")
                 ▼                                         └─ wgpu_backend/  (feature "wgpu")
          Backend::paint_layer adds its own Shape::Callback; at paint time it uses
          geometry.rs (pixel math) + pass.rs (uniform packing) + effect.rs (validated naga module)
```
- **Shader model**: every effect is WGSL = `shaders/prelude.wgsl` + the user's `fn effect(uv)`.
  naga parses and validates it once in `Effect::new` on every backend. Backends translate the
  validated `naga::Module` (glow: GLSL via naga `glsl-out`; wgpu: WGSL directly).
- **Pipeline per layer** (backend-neutral contract): capture the backdrop region (padded by
  `FilterChain::padding_points`, clamped to the screen) into an 8-bit texture; run passes
  ping-pong into intermediates; run the internal composite pass (`shaders/composite.wgsl`) that
  writes the result over the visible region by the rounded-rect mask. glow copies the backdrop
  out of the framebuffer and REPLACES the region (mixing with that exact copy); wgpu re-renders
  the beneath content offscreen (it cannot read its surface), so on wgpu that content is drawn
  twice, and the UI composite BLENDS over egui's pass by mask coverage (the re-render lacks paint
  callbacks, so outside the mask the screen must stay as it is). Offline replaces (its input is
  the whole backdrop).
- **Coordinates**: effect `uv` is top-left origin, y down on every backend. A backend whose
  texture storage is bottom-up sets `FrameUniforms::flip_y`; the prelude's `src()` applies it.
  Intermediates are grow-only; `src_scale` / `aux_scale` map the used region to storage.
  wgpu derives the fragment uv from the pixel position (`frag_from_position`) and can run a pass
  over a WINDOW of the capture (`*_offset`, offline tiles); glow interpolates, whole capture only.

## Files and submodules
- `lib.rs`: crate docs (semantics, prelude table, backend matrix), re-exports, lints.
- `effect.rs`: `Effect`, `EffectError` (+ accessors), prelude assembly, naga validation,
  snippet-relative diagnostics (`codespan-reporting`), interface check (bindings, entry points,
  `override`), conservative `fl.time` detection (animated effects), internal composite modules
  (gamma / linear output), binding constants.
- `pass.rs`: `Pass`, `FilterChain`, `PARAM_SLOTS`, uniform block layout (`offsets`, `pack_uniforms`),
  `FrameUniforms::for_layer` (the per-layer uniform values every backend uploads).
- `geometry.rs`: `layer_geometry` — region / capture / UV / mask math (pure, unit-tested);
  `LayerGeometry::composite_params` (the composite pass's parameter block, every backend).
- `layer.rs`: `FilterLayer` and its `paint` (skips a missing or failed backend; requests a
  repaint for animated chains only when the backend is usable).
- `backend.rs`: crate-internal `Backend` trait, `LayerJob` (backend-neutral: rect, mask, passes,
  padding, time), registry (`register`, `installed`, `uninstall`; a hook-less egui plugin, so it
  lives in the `Context` and survives `Memory` resets), public `status`, `failed_effects`,
  `BackendStatus`, `EffectFailure`, `InstallError`.
- `presets.rs`: built-in effects (compiled once in `OnceLock`s) and their constructors.
- `shaders/`: the WGSL prelude and composite pass (see its readme).
- `glow_backend/`: the `egui_glow` backend (see its readme).
- `wgpu_backend/`: the `egui-wgpu` backend (see its readme); its `chain.rs` is the single, egui-free
  owner of executing a chain on wgpu.
- `offline/` (features `image` / `video`): `OfflineRenderer` — chains over ordinary frames,
  tiling, readback; executes chains ONLY through `wgpu_backend/chain.rs`. Its `video/` submodule
  (feature `video`) adds frame streams, source / sink traits and ffmpeg / image-sequence adapters
  on top of the renderer (see the readmes).

## Contracts and invariants
- Public API never panics on user input; the only panics are in `presets::builtin` for invalid
  built-in WGSL, which tests exclude.
- `FlUniforms` byte layout: `pass::offsets` must match the WGSL struct (pinned by a naga-layout test).
- Bindings in `@group(0)`: 0 = `fl` uniforms, 1 = pass input, 2 = sampler, 3 = original capture
  (composite only). User effects may not declare other resources or entry points.
- A backend never draws a fake: when it is missing or failed, layers draw nothing extra and
  `status()` says why. Failures are latched and logged once (`log` crate), never retried per frame.
  Per-effect failures disable only that effect (`failed_effects()`), not the backend.
- No process-wide mutable state: log-once latches live in the install state (wgpu `Shared`,
  so a re-install may log again) or in the `Context`'s temp memory (`layer.rs`, no backend).
  Allowed statics: `OnceLock`s of immutable compiled data (`presets.rs`, `effect.rs` composites)
  and monotonic id counters that must be unique process-wide (`effect::NEXT_MODULE_ID`,
  `wgpu_backend::NEXT_GENERATION`; the reason is on each declaration).
- Distances are in LAYER points: one factor, `LayerGeometry::pixels_per_layer_point` (egui ppp ×
  `geometry::transform_scale`, the callback's rendered / painted width), converts padding, corner
  radii and `fl.pixels_per_point` (set only by `FrameUniforms::for_layer`). Untransformed layers
  get a scale of exactly 1.
- Chain padding is the SUM of pass sample radii; a preset whose passes reach along different axes
  (separable blur) declares its reach on one pass only.
- Backend-specific public API lives in its module behind its feature; core never names backend types.
- Offline output equals the UI's only through shared code: `layer_geometry` with the frame as the
  screen, `FrameUniforms::for_layer`, and `wgpu_backend/chain.rs`. Never add a second pass /
  composite recorder.

## Editing map
- New preset: add the WGSL constant + entry in `BUILTIN_EFFECTS` + constructor in `presets.rs`.
- New prelude helper or uniform: `shaders/prelude.wgsl`, then `pass.rs` offsets/packing and tests.
- New backend: implement `backend::Backend` in a feature-gated module, add its `install_*` API,
  register with `backend::register`.
- Capture/padding/mask math: `geometry.rs`.
