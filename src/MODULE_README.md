# Module: src

## Purpose
The whole `egui_filter_layers` library: a `FilterLayer` shape that redraws everything painted
beneath a rect through a chain of WGSL filter passes, on pluggable GPU backends.

## Architecture
```text
user code ──► FilterLayer::paint (layer.rs)
                 │  resolves FilterChain -> LayerJob (rect, mask, passes, padding, time, frame_input)
                 ▼
          backend registry plugin (backend.rs)       ──►  Arc<dyn Backend>
                 │                                         ├─ glow_backend/  (feature "glow")
                 ▼                                         └─ wgpu_backend/  (feature "wgpu")
          Backend::paint_layer adds its own Shape::Callback; at paint time it uses
          geometry.rs (pixel math) + plan.rs (the draws of a run: slots, viewports, uniform
          blocks) + pass.rs (uniform layout) + effect.rs (validated naga module)
```
- **Shader model**: every effect is WGSL = `shaders/prelude.wgsl` + the user's `fn effect(uv)`.
  naga parses and validates it once in `Effect::new` on every backend. Backends translate the
  validated `naga::Module` (glow: GLSL via naga `glsl-out`; wgpu: WGSL directly).
- **Pipeline per layer** (backend-neutral contract): capture the backdrop region (padded by
  `FilterChain::padding_points`, clamped to the screen) into an 8-bit texture; run passes
  ping-pong into intermediates (the wiring — which slot each draw binds and writes, its viewport
  and complete uniform block — is `plan.rs`'s alone; backends map slots to objects and draw); run
  the internal composite pass (`shaders/composite.wgsl`) that
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
- `plan.rs`: backend-neutral wiring of one chain run — `plan()` (steps: bindings by `Slot`,
  output, viewport, fragment offset; the composite), `TextureMapping::of` (the one storage-uv
  formula), `step_uniforms` / `composite_uniforms` (every block a backend uploads).
- `geometry.rs`: `layer_geometry` — region / capture / UV / mask math (pure, unit-tested);
  `LayerGeometry::composite_params` (the composite pass's parameter block, every backend);
  `pointer_uniform` (the one screen-points -> capture-uv conversion of `fl.pointer`).
- `layer.rs`: `FilterLayer` and its `paint` (skips a missing or failed backend; requests a
  repaint for animated chains only when the backend is usable; reads the per-frame input).
- `backend.rs`: crate-internal `Backend` trait, `LayerJob` (backend-neutral: rect, mask, passes,
  padding, time, `FrameInput`), registry (`register`, `installed`, `uninstall`; a hook-less egui plugin, so it
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
  Members without a producer yet hold neutral values set in `FrameUniforms::for_layer`.
- Per-frame input (`fl.delta_time`, `fl.frame`, `fl.pointer`): read on the GUI thread in
  `FilterLayer::paint` into `LayerJob::frame_input` (per context, not per layer: every layer of a
  frame sees the same values; `stable_dt` clamped to `0..=0.1`, `cumulative_frame_nr` wrapping to
  `u32`, pointer in screen points). Backends resolve it at paint time with the callback's pixels
  per point (`FrameInput::resolve` -> `pass::InputUniforms`, pointer via
  `geometry::pointer_uniform` into capture uv — no layer transform needed, the capture is in
  screen pixels) and add it with `FrameUniforms::with_input`. Offline: `OfflineOptions`
  (`with_delta_time` / `with_frame` / `with_pointer`), the video processor's frame clock fills
  unset counters. Only `fl.time` makes an effect animated.
- Bindings in `@group(0)`, all declared by the prelude and bound for EVERY draw (effects and the
  composite share one layout): 0 = `fl`, 1 = pass input, 2 = linear sampler, 3 = original
  capture, 4 = a crate-appended uniform block, 5 = state, 6 = mask coverage, 7 = nearest sampler.
  A binding a draw does not use gets a neutral dummy (transparent / white texture, zero buffer);
  every pass and the composite bind the capture as the original (`original()`, mapped by the
  capture's own `internal_aux_*`). The interface check accepts only these
  prelude declarations (`effect::PRELUDE_RESOURCES`): a snippet declares no binding at all.
- Wiring has one owner: backends never decide which slot a draw reads, which mapping or
  fragment offset it uploads; they call `plan.rs` (glow and wgpu blocks are byte-identical but for
  `flip_y` / fragment mode, pinned by `glow_and_wgpu_upload_identical_blocks`).
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
- A new texture a draw reads, a new kind of step, per-step sizes: `plan.rs` first, then each
  backend's slot -> object mapping (wgpu `chain.rs`, glow `render.rs` + `glsl.rs` units).
- New backend: implement `backend::Backend` in a feature-gated module, add its `install_*` API,
  register with `backend::register`.
- Capture/padding/mask math: `geometry.rs`.
