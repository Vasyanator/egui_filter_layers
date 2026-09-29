# Module: src

## Purpose
The whole `egui_shader_layers` library: a `ShaderLayer` shape that redraws everything painted
beneath a rect through a chain of WGSL filter passes, on pluggable GPU backends.

## Architecture
```text
user code ──► ShaderLayer::paint (layer.rs)
                 │  checks user textures (plan::check_textures), the state id (check_state) and
                 │  the mask (check_mask); tessellates a shape mask (mask.rs, GUI thread);
                 │  resolves FilterChain -> LayerJob (rect, mask, passes, padding, time,
                 │  frame_input, textures, id / viewport / state options / reset ticket)
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
  writes the result over the visible region by the mask (the rounded-rect SDF, or the alpha of a
  shape mask's coverage texture, rasterized by the plan's coverage step before the passes). glow copies the backdrop
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
- `effect/`: `Effect`, `EffectError` (+ accessors), prelude assembly, naga validation,
  snippet-relative diagnostics (`codespan-reporting`), interface check (bindings, entry points,
  `override`), conservative `fl.time` detection (animated effects), named parameters
  (`struct Params` detection and layout, `ParamType` / `ParamInfo` / `ParamValue` /
  `ParamError`), internal composite modules, binding constants (see its readme).
- `pass.rs`: `Pass` (anonymous `params(i)` block and typed named parameters: `set` / `with` /
  `set_slice` / `get`), `FilterChain`, `PARAM_SLOTS`, uniform block layout (`offsets`, `pack_uniforms`),
  `FrameUniforms::for_layer` (the per-layer uniform values every backend uploads).
- `state.rs`: persistent state (F7), the backend-neutral half: public `StateOptions`,
  `StateFormat`, `StateError`, `StateFailure`, `reset_state` / `state_count` / `failed_states`;
  per-context reset requests (`pending_reset` hands a `ResetTicket` to the job; the backend claims it
  where the registry's `begin` runs, so an unrun paint keeps it); `StateRegistry<T>`
  (per `(viewport, id)` entry: the backend's targets `T`, which one holds the latest state and its
  size, one step per frame, eviction), used by both UI backends.
- `plan.rs`: backend-neutral wiring of one chain run — `plan()` (steps: bindings by `Slot`,
  what each slot holds (`Content`: resolution, window), output, viewport, fragment offset; the
  composite; the ping-pong pool size `target_px`; which user texture slots each draw binds),
  `check_textures` (every slot an effect reads is attached), `scaled_px` (per-pass scale sizes),
  `TextureMapping::of` (the one storage-uv formula), `step_uniforms` / `composite_uniforms` / `params_block` / `composite_params_block`
  (every block a backend uploads, at bindings 0 and 4).
- `geometry.rs`: `layer_geometry` — region / capture / UV / mask math (pure, unit-tested);
  `LayerGeometry::composite_params` (the composite pass's parameter block, every backend);
  `pointer_uniform` (the one screen-points -> capture-uv conversion of `fl.pointer`).
- `mask.rs`: `Mask` (public: rounded rect / any egui `Shape` in layer points / `Regions`),
  `Region`, `MAX_REGIONS`, `REGION_NONE`, `MaskError`, `RegionMeshes` / `PackedRegions` (every
  region packed in id order with its draw range),
  `JobMask`, `tessellate` (the one tessellation, UI and offline: no pre-rasterized discs),
  `CoverageMeshes::pack` -> `PackedCoverage` (egui vertex bytes, absolute `u32` indices, one draw
  per texture run; every backend uploads exactly this), `MESH_WGSL`.
- `layer.rs`: `ShaderLayer` and its `paint` (skips a missing or failed backend, and a layer whose
  user textures fail `check_textures` (logged once per context and message); requests a repaint
  for animated chains only when the backend is usable; reads the per-frame input);
  `ShaderLayer::texture` attaches user textures (F8); `ShaderLayer::id` / `state` /
  `check_state` (F7: a stateful chain without an id is skipped, logged once per message).
- `backend.rs`: crate-internal `Backend` trait, `LayerJob` (backend-neutral: rect, mask, passes,
  padding, time, `FrameInput`, attached texture ids), registry (`register`, `installed`, `uninstall`; a hook-less egui plugin, so it
  lives in the `Context` and survives `Memory` resets), public `status`, `failed_effects`,
  `failed_textures`, `BackendStatus`, `EffectFailure`, `TextureError`, `TextureFailure`, `InstallError`.
- `presets.rs`: built-in effects (compiled once in `OnceLock`s) and their constructors.
- `impulses.rs`: `Impulses` (F3) — user-owned, bounded (16, oldest dropped first) timed impulses
  in layer-rect uv; packs them into the named params `impulses` / `impulse_count` (`apply`), and
  is the only thing that requests repaints for impulse-driven effects (`request_repaint_if_alive`,
  only while one is alive). No egui memory, no global state; usable offline with any clock.
- `hot.rs` (feature `hot-reload`, not compiled on wasm32): `HotEffect` — a std thread polls a
  WGSL file's `(mtime, length)`, reads and compiles it off the GUI thread and publishes the
  effect / the distinct errors (logged once each) over a channel; `poll` never blocks; drop wakes
  the thread and joins it within 100 ms or detaches it. `rebind` rebuilds a `Pass` on a new effect
  keeping matching named params (public `Pass` API only).
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
  built-in WGSL and `presets::builtin_param` for a built-in rejecting its own named parameters,
  both crate bugs that tests exclude.
- `FlUniforms` byte layout: `pass::offsets` must match the WGSL struct (pinned by a naga-layout test).
  Members without a producer yet hold neutral values set in `FrameUniforms::for_layer`.
- Per-frame input (`fl.delta_time`, `fl.frame`, `fl.pointer`): read on the GUI thread in
  `ShaderLayer::paint` into `LayerJob::frame_input` (per context, not per layer: every layer of a
  frame sees the same values; `stable_dt` clamped to `0..=0.1`, `cumulative_frame_nr` wrapping to
  `u32`, pointer in screen points). Backends resolve it at paint time with the callback's pixels
  per point (`FrameInput::resolve` -> `pass::InputUniforms`, pointer via
  `geometry::pointer_uniform` into capture uv — no layer transform needed, the capture is in
  screen pixels) and add it with `FrameUniforms::with_input`. Offline: `OfflineOptions`
  (`with_delta_time` / `with_frame` / `with_pointer`), the video processor's frame clock fills
  unset counters. Only `fl.time` makes an effect animated.
- Bindings in `@group(0)`, all declared by the prelude and bound for EVERY draw (effects and the
  composite share one layout): 0 = `fl`, 1 = pass input, 2 = linear sampler, 3 = original
  capture, 4 = the named-parameter block `p: Params` (appended by the crate after a snippet that
  declares `struct Params`; every draw binds `plan::params_block`, 16 zero bytes without one),
  5 = state (the layer's latest state for every pass step of a stateful run; a 1x1 zero dummy
  otherwise and for the composite), 6 = mask coverage, 7 = region ids (R1; every pass step of a
  region run, a 1x1 white dummy = `REGION_NONE` otherwise and for the composite). There is no nearest sampler: every texture is read through ONE
  sampler (naga's GLSL writer rejects two per texture); exact reads use `textureLoad`.
  A binding a draw does not use gets a neutral dummy (transparent / white texture, zero buffer);
  every pass and the composite bind the capture as the original (`original()`, mapped by the
  capture's own `internal_aux_*`). The interface check accepts only these
  prelude declarations (`effect::PRELUDE_RESOURCES`): a snippet declares no binding at all.
- User textures (F8): slot `i` of `TEXTURE_SLOTS` (3, every backend: WebGL2 has 4 bind groups)
  is `@group(1 + i)` = egui-wgpu's texture bind-group layout (texture 0, its own sampler 1); wgpu
  binds egui's own bind group of the texture there, glow the GL texture on unit `4 + i` (its own
  sampling parameters), offline the renderer's upload of an `OfflineTexture`. Sampled in the
  texture's own uv (no mapping, independent of windows and scales). Which slots an effect reads
  is found in its IR (`Effect::reads_texture`, conservative); a layer reading an unattached slot is
  refused before any backend work (`plan::check_textures` -> `TextureError`, UI and offline), never
  given a dummy; an attached id the backend cannot resolve skips the layer and is reported by
  `failed_textures` (latched per id, logged once). Unattached slots are bound to dummies only so the
  layout is complete; no effect that reads them runs.
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
- Per-pass scale (F6, `Pass::with_scale`): a scaled step renders `plan::scaled_px(capture,
  scale)` pixels of the WHOLE capture into its ping-pong slot; `uv`, `fl.texel`, `fl.size_px`,
  `fl.pixels_per_point` stay the capture's, `internal_src_texel` / `internal_frag_texel` carry
  the input / output pixel size (`src_px`, `src_texel()`, `out_texel()`). Scaled chains run
  only over the whole capture (offline tiling refuses them: `OfflineError::NotTileable`); the
  composite upsamples a scaled last output bilinearly.
- Persistent state (F7): a pass whose effect is `Effect::new_state` is a STATE step (`plan.rs`):
  it reads the display chain's current input as `src`, the latest state at binding 5, and writes
  the other of the layer's two state textures (`Slot::StateA` / `StateB`), `ceil(capture * scale)`
  texels over the whole capture; display steps bind the latest state. Stateful chains need
  `ShaderLayer::id`; states are kept per `(ViewportId, Id)` (egui counts frames per viewport) in
  each backend's `StateRegistry` (wgpu: `WgpuResources::states`, outside the per-frame pool, dropped
  on re-install; glow: `GlowState::states`, freed by eviction / `destroy_glow`; offline: the
  renderer's one state). The first paint of a key in a frame steps it; a repeat skips the state
  steps. A new / reset state is cleared before use; growth keeps the content (backends copy); the
  read mapping uses the stored content's size, so a resized layer reads it stretched. Unpainted
  for > 120 frames of the entry's OWN viewport (the registry keeps each viewport's latest frame)
  AND > 2 s: evicted. RGBA16F state requires renderable + filterable RGBA16F
  (else refused, `failed_states`; never downgraded); RGBA8 is opt-in. Stateful runs are never
  windowed (offline: `NotTileable`).
- Shape masks (F9): a `Mask::Shape` is in LAYER points, tessellated where the job is built
  (`ShaderLayer::paint` on the GUI thread with the context's ppp / tessellation options / font
  atlas size; offline per run with default options), without egui's pre-rasterized discs (only
  text samples the font atlas). Coverage = the meshes' alpha (an ADDITIVE, saturating union into a
  cleared RGBA8 texture at capture resolution, the run's window at its origin; abutting pieces'
  feathers sum to 1, so they leave no seam), drawn by
  `shaders/mesh.wgsl` on every backend (glow: naga-translated); the layer transform maps it at
  paint time (`geometry::mask_transform`), so the feather scales with a zoom. The composite alone
  binds it (`plan::ChainPlan::with_coverage`, `params[3].x = 1`); uncovered pixels keep the screen
  on every backend (wgpu: discard). Paint callbacks and malformed meshes (an index past the
  vertices, a partial triangle) in a mask are refused (`MaskError`, logged once per message; the
  packer skips such meshes too, so no out-of-range index reaches a draw); offline also text and textured meshes (`OfflineError::Mask`). A mask texture
  the renderer cannot resolve skips the layer (`failed_textures`; wgpu: a stale font atlas skips
  only the frame).
- Regions (R1, `Mask::Regions` / `ShaderLayer::regions`, at most `MAX_REGIONS` = 255): the union
  of the regions is the coverage (F9 path); `plan::ChainPlan::with_regions` adds two id passes
  after the coverage draw into `Slot::RegionId` (window at the texture origin, cleared to 255,
  no blend: pass A threshold 0 = every touched pixel, pass B threshold `0.5 - 1/128` (a shared mid-pixel edge goes to the later region whatever the window); later regions win),
  bound at binding 7 of every PASS step and read by the prelude's `region()` / `region_at(uv)`.
  Per-region values are F4 arrays indexed by the id. Runs on wgpu (UI), glow and offline (the same `wgpu_backend/coverage.rs` rasterizer per tile window).
- Chain padding is the SUM of pass sample radii; a preset whose passes reach along different axes
  (separable blur) declares its reach on one pass only.
- Backend-specific public API lives in its module behind its feature; core never names backend types.
- Offline output equals the UI's only through shared code: `layer_geometry` with the frame as the
  screen, `FrameUniforms::for_layer`, and `wgpu_backend/chain.rs`. Never add a second pass /
  composite recorder.

## Editing map
- New preset: add the WGSL constant + entry in `BUILTIN_EFFECTS` + constructor in `presets.rs`.
- New prelude helper or uniform: `shaders/prelude.wgsl`, then `pass.rs` offsets/packing and tests.
- Named parameters (accepted types, layout, errors): `effect/params.rs`; storage and typed
  setters: `pass.rs`; which bytes each draw binds: `plan.rs`.
- A new texture a draw reads, a new kind of step, per-step sizes: `plan.rs` first, then each
  backend's slot -> object mapping (wgpu `chain.rs`, glow `render.rs` + `glsl.rs` units).
- State lifetime (reset, eviction, repeat paints, keys): `state.rs`; which step reads / writes
  which state texture: `plan.rs`; state objects: wgpu `chain.rs` (`ChainState`) + `render.rs`,
  glow `render.rs` + `resources.rs` (`ensure_kept`), offline `offline/mod.rs`.
- New backend: implement `backend::Backend` in a feature-gated module, add its `install_*` API,
  register with `backend::register`.
- Capture/padding/mask math: `geometry.rs`.
- Shape masks: what is accepted / tessellation / packing `mask.rs`; the coverage step and its
  uniform bytes `plan.rs`; rasterization wgpu `wgpu_backend/coverage.rs` (UI and offline), glow
  `glow_backend/coverage.rs`; the composite's read `shaders/composite.wgsl` (`fl_coverage`).
- Regions: `Region` / packing `mask.rs`; the id passes and the region binding `plan.rs`
  (`with_regions`, `region_locals`); rasterization `wgpu_backend/coverage.rs` (`fs_region` in
  `shaders/mesh.wgsl`); the reads `shaders/prelude.wgsl` (`region_at`).
