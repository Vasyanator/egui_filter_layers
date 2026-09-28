# Module: src/wgpu_backend

## Purpose
The `egui-wgpu` backend (feature `wgpu`), native and web. egui-wgpu's surface cannot be read
mid-frame, so the backdrop is **re-rendered**: the content under each filter is drawn twice.

## Architecture
One frame, per filter layer:
1. `WgpuBackend::paint_layer` (GUI thread, `mod.rs`) adds an `egui_wgpu::Callback` and pushes a
   `Pending { Weak<callback object>, LayerSlot, rect, padding }` into `Shared::pending`.
2. `FilterLayersPlugin::output_hook` (`plugin.rs`, egui `Context::end_pass` -> `on_output`) finds
   the callback in `FullOutput::shapes` by `Arc` address — egui's exact final paint order with layer
   transforms applied, so there is no stale layer order. Everything before it is "beneath":
   `beneath.rs` clips it to the padded capture rect and skips callbacks; `ctx.tessellate`
   tessellates; `textures.rs` resolves texture bind groups. The `BeneathPlan` goes into the slot.
   The hook records EVERY pass's texture delta first (`TextureCopies::observe`), then skips
   planning when egui will discard the pass (`request_discard`, `run_ui` repeats it).
3. `FilterCallback::prepare` (`render.rs`, egui's encoder, before egui's pass) re-renders the
   meshes into an `Rgba8Unorm` capture (own mesh pipeline mirroring egui-wgpu's shader; on
   `*Srgb` targets through an `Rgba8UnormSrgb` view of it with the linear-output entry, so
   meshes blend in linear like egui's), then hands the capture's unorm view to the chain executor
   (`ChainResources::record`, `chain.rs`): it runs the draws of `plan::plan` — effect passes
   ping-pong (RGBA16F when the adapter can) — and returns the composite bind group.
4. `FilterCallback::paint` draws the BLENDED composite (`chain::draw_composite_over`: an erase
   draw then an add draw, together `mix(dst, filtered, coverage)`, uncovered pixels discarded)
   into egui's pass. Outside the rounded mask the screen keeps what egui drew, including paint
   callbacks and lower filter layers the re-render lacks.

**Chain executor boundary** (`chain.rs` + `gpu.rs`): everything wgpu-specific about running a
`FilterChain` — intermediate format, uniform buffer stride, the one `@group(0)` layout with its
samplers and dummies, effect pipeline cache, composite pipelines for any output format — lives
there, while the wiring (ping-pong order, which slot each binding gets, storage mappings, uniform
blocks, viewports) comes from the backend-neutral `src/plan.rs`, shared with glow. It takes
only `&wgpu::Device`, `&wgpu::Queue`, a `CommandEncoder`, a source texture view (capture at its
origin), `LayerGeometry`, time and pixels-per-point. It knows nothing of egui (no
`CallbackResources`, no egui `Renderer`), so a runner on its own device reuses it unchanged; the UI
callback adds only the backdrop re-render, the egui-pass composite pipeline and the frame pool;
the offline renderer (`src/offline/`) adds only uploads, tiles and readback.

**Windows**: a run computes a `ChainWindow` of the capture — the UI always the whole capture
(composite origin = the capture's screen position), an offline tile a sub-rect. Every texture
holds its window at its origin (`src_offset` / `aux_offset` in storage uv) and fragments take
their capture uv from their pixel position (prelude frag mode 1), so uniforms and per-pixel maths
are those of the whole layer. Hardware bilinear weights between texel centres can still differ
(rarely, <= 2 steps) with the texture size / window origin; texel-centre sampling is exact.

## Files
- `mod.rs`: `install_wgpu`, `WgpuOptions` (builder mirroring eframe's option names,
  `From<&egui_wgpu::RendererOptions>`), `WgpuBackend`, `Shared` (health and log-once latches, pending list), slots.
- `plugin.rs`: the egui plugin; matching pending layers, planning.
- `beneath.rs`: pure selection / clipping / callback skipping (unit-tested, no GPU).
- `textures.rs`: egui-wgpu bind groups; own copies of textures uploaded in full that egui-wgpu
  has not applied yet (kept across the passes of a frame, partial deltas followed).
- `chain.rs`: the chain executor (`ChainResources`, `ChainTargets`, `ChainWindow`, `record`,
  `frame_uniforms`, `chain_uniforms` = the plan's blocks at the uniform stride; composites: replace `composite_pipeline` / `draw_composite` for offline,
  blended `CompositeOver` / `draw_composite_over` for the UI); egui-free; shared with `src/offline/`.
- `gpu.rs`: egui-free wgpu helpers (`Target` with an optional reinterpreting render view, `GrowBuffer`, layout entries, `pass_pipeline`,
  `begin_pass`, `ErrorScopes` (out-of-memory + validation + internal, awaitable) and the
  synchronous `with_error_scope` on top of it, format capability check).
- `resources.rs`: the UI's objects: `ChainResources` + egui-pass blended composite pipelines +
  mesh pipeline (and whether the capture renders through its sRGB view), per-layer target pool
  (`LayerTargets`).
- `render.rs`: the `CallbackTrait` implementation (backdrop re-render, then the chain executor).
- `mesh.wgsl`: mesh re-render shader (egui-wgpu maths + dithering; `fs_main` = gamma framebuffer
  path, `fs_main_linear` = linear framebuffer path).

## Contracts and invariants
- Every effect and composite pipeline uses ONE pipeline layout (`ChainResources::layout`, all
  eight prelude bindings); every draw's bind group fills all eight, dummies where the plan binds
  nothing. The params binding (4) is unsized so modules may declare any block size there.
- The mesh pipeline's group 1 layout is the SAME descriptor as egui-wgpu's texture layout, so
  egui's per-texture bind groups bind unchanged (wgpu-core dedups layouts; WebGPU group-equivalence).
- Every layer of a frame gets its own `LayerTargets` (pool index = prepare order, reset in
  `finish_prepare`): all prepares are recorded before the single egui pass. Likewise any caller of
  the executor needs one `ChainTargets` per run whose composite is still pending.
- `chain.rs` / `gpu.rs` must not depend on egui, egui-wgpu or this module's UI types; they are
  `pub(crate)` because `src/offline/` runs chains through them.
- Pipelines must match egui's pass: `RenderState::target_format` (`*Srgb` -> linear composite),
  `WgpuOptions` MSAA and depth/stencil format (egui-wgpu's depth state is mirrored). egui-wgpu
  exposes neither, so the user supplies them; a mismatch is NOT detectable by the crate: pipeline
  creation succeeds, and wgpu rejects the pipeline when egui's encoder is finished (outside the
  crate) — uncaptured, so wgpu-core's default handler panics (native and WebGL2); browser WebGPU
  logs it and drops the frame. Never claim it latches.
- The UI composite never writes pixels its mask does not cover (discard), and inside the mask it
  equals the filtered value exactly; only the 1-px antialiased edge is a blend with what the pass
  holds (on `*Srgb` targets in linear, so it differs from the offline / glow gamma mix there).
- The capture texture is always `Rgba8Unorm` and effects always sample its unorm view (the gamma
  bytes the screen stores). On `*Srgb` targets the meshes render through an `Rgba8UnormSrgb` view
  when the device has `DownlevelFlags::VIEW_FORMATS` (`WgpuResources::srgb_capture`). Known gap:
  without it (GL / WebGL2) the re-render blends in gamma, so semi-transparent edges under a filter
  differ from the screen (e.g. 50% white over grey: about 25 steps); logged at warn once per install (`WgpuResources::create`). A copy
  from an sRGB texture would fix it but GL's sRGB copy semantics are unverified, so none is made;
  egui-wgpu prefers non-sRGB surfaces, which keeps this rare.
- egui-wgpu's `predictable_texture_filtering` is not mirrored by `mesh.wgsl` (hardware filtering).
- Paint callbacks beneath a filter (incl. lower filter layers) are skipped, debug-logged once per install
  (`Shared::logged`; a re-install may log again, as for every log-once message here).
- egui-wgpu's `Renderer` is read-locked only in `output_hook` (eframe locks it only while
  painting, never across the pass: no deadlock).
- A texture uploaded in full is served from the crate's copy until egui-wgpu holds a different
  `wgpu::Texture` for that id (egui-wgpu creates a new texture per full upload) or it is freed;
  frame/viewport boundaries are not observable in the hook (`end_pass` pops the viewport stack).
- The capture is cleared like eframe clears the layer's viewport (immediate: transparent, else
  `WgpuOptions::clear_color()`): through the sRGB view wgpu encodes it like egui's own clear; on
  the no-view-formats fallback of a `*Srgb` target it is sRGB-encoded by the crate.
- One-time object creation (`ChainResources::create` / `effect_pipeline` / `composite_pipeline` /
  `composite_over_pipelines`, `WgpuResources::create`) runs in `with_error_scope` (out-of-memory, validation and internal
  scopes; resolves immediately on native, reports nothing on the web): failures latch the backend or the effect instead
  of panicking; nothing is retried per frame. Per-frame work (grow-only targets and buffers,
  texture copies, bind groups, queue writes) is NOT scoped: its sizes are bounded by the screen
  and egui's own textures, and a validation error there takes wgpu's uncaptured-error path
  (eframe installs no handler, so it panics).
- Performance: beneath shapes are cloned (meshes and galleys are `Arc`s, so shallowly) and
  tessellated once per layer per frame; an effect's first use compiles its pipeline
  synchronously inside `prepare`.
- Only integrations that end passes with `Context::end_pass` / `run_ui` are supported (eframe).

## Editing map
- What counts as "beneath" / clipping: `beneath.rs`. Texture staleness: `textures.rs`.
- Which texture / mapping a draw gets: `src/plan.rs` (shared with glow). Recording, bind groups,
  uniform stride, intermediate format: `chain.rs` (shared by every wgpu caller). Backdrop re-render, egui-pass pipelines: `render.rs` / `resources.rs`.
- Per-layer uniform values and composite params: core `FrameUniforms::for_layer` (`pass.rs`) and
  `LayerGeometry::composite_params` (`geometry.rs`), shared with glow.
- Tests: unit tests in each file; GPU end-to-end in `tests/wgpu_headless.rs` (`--ignored`).
