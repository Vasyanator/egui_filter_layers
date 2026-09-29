# Module: src/glow_backend

## Purpose
The `egui_glow` backend (feature `glow`): OpenGL 3.3+, OpenGL ES 3.0+, WebGL2.

## Architecture
`GlowBackend` (registered via `install_glow`) owns `Arc<Mutex<GlowState>>`. `paint_layer` adds an
`egui_glow::CallbackFn`; inside it `render::paint_layer` runs one layer:
1. save the caller's DRAW/READ framebuffer bindings and `SAMPLES` FIRST (resource creation binds
   framebuffer 0); every exit path restores them;
2. lazily create `Resources` (empty VAO, UBO, capture pool / ping-pong targets, composite program;
   the blended composite programs with the first multi-entry layer);
3. compute every entry's geometry (`geometry::entry_rect` + `layer_geometry`) and the op list
   (`plan::schedule`, framebuffer source); fetch or build each visible entry's pass programs
   (`GlowState::program_for`, cached by module id, evicted when the effect is dropped). Steps
   4-6 then run per op: `Capture(k)` = 4 into capture `k`, `Run(k)` = 5, `Composite(k)` = 6;
4. disable scissor + blend; query the read buffer's `ReadFormat` (cached for framebuffer 0); reject
   the layer for non-normalized formats; multisampled -> resolve with identical rects into a
   renderbuffer of the SAME internal format, then blit the padded capture rect into the RGBA8
   capture texture;
5. plan the run (`plan::plan`, whole capture, `with_regions(count)` for a mask — `count` = 0 for a
   shape mask, i.e. `with_coverage` — whose meshes are then rasterized into the coverage target
   and, for a region mask, into the region-id target by `coverage.rs` before the passes; a stateful layer's `GlowState::states` entry, looked
   up after evicting unpainted ones, where the job's reset ticket is claimed (`LayerJob::claim_reset`),
   feeds its state input), grow its state targets keeping their
   content (`RenderTarget::ensure_kept`: new texture + blit) and clear the one the plan names,
   grow the ping-pong targets to the plan's
   `target_px`, and run its pass steps into them at `(0, 0, used_px)` (smaller for scaled
   passes; mode-0 uv interpolation over that viewport needs nothing else), binding every texture
   unit per draw (the plan's `@group(0)` slots and attached user textures, dummies elsewhere) and
   uploading `plan::step_uniforms` (binding point 0) and `plan::params_block` (binding point 1,
   a grow-only buffer written with `glBufferSubData`, re-allocated only when a block outgrows it);
6. composite back into the original framebuffer over the entry's visible region, scissor
   re-enabled (`plan::composite_uniforms`; replace program for one entry, blended erase + add for
   several), scissor + blend off again;
7. `apply_outcome`: latch / per-effect failure / suspects / once-per-format rejection log.

## Files
- `mod.rs`: `install_glow`, `destroy_glow`, `GlowBackend` (`Backend` impl), `GlowState`, latches;
  the paint callback resolves the layer's user textures through `egui_glow::Painter::texture`
  (an unknown id skips the layer: `GlowState::fail_texture`, `failed_textures()`).
- `glsl.rs`: pure WGSL->GLSL translation (naga `glsl-out`, empty writer flags; also
  `translate_mesh(target, MeshEntry)`, the mask rasterizer's programs — `fs_main` coverage,
  `fs_region` region ids — whose `MeshLocals` block naga names per stage, so both names are
  bound), dialect choice,
  the reflected `fl` and named-parameter block names (by `@binding` 0 / 4), and the texture-unit table of the prelude's layout (`GROUP0_TEXTURE_BINDINGS`,
  `USER_UNIT_BASE`).
- `resources.rs`: `GlError`, `RenderTarget` (incl. `ensure_kept` / `clear` for state targets),
  `ParamsBuffer` (grow-only params UBO), `BlendedComposite` (erase + add programs), `Resources`
  (incl. the per-entry capture pool, `ensure_captures`)
  (incl. dummy textures), `build_program` (all GL object code).
- `render.rs`: the per-layer GL sequence and state restoration.
- `coverage.rs`: the mask rasterizer: shape-mask coverage (F9) and region ids (R1). Two mesh
  programs (`shaders/mesh.wgsl` via `glsl::translate_mesh`), their shared VAO (float pos / uv,
  INTEGER colour attribute), VBO / EBO / locals UBO, the RGBA8 coverage target and the RGBA8
  region-id target; created with the first masked layer, a failure latches.

## Contracts and invariants
- All GL calls happen with the context current (inside the callback or in `on_exit`).
- egui_glow re-runs `prepare_painting` after each callback; this module restores only what that
  does not: framebuffer bindings, textures on units 0-7, uniform-buffer bindings (points 0 and
  1), own VAO/program. No sampler object is ever bound.
- Texture units: `GROUP0_TEXTURE_BINDINGS[i]` is unit `i` (0 src, 1 original, 2 state,
  3 coverage, 4 region ids; the texture's own `LINEAR` parameters); user texture slot `i` is unit
  `USER_UNIT_BASE + i` (5-7), egui_glow's own texture
  object with the parameters egui_glow set from its `TextureOptions`. Max unit 7 < the 16 GLES 3 /
  WebGL2 guarantee. naga cannot translate a texture read through two samplers in one module, so
  the prelude has one sampler per texture and exact reads via `textureLoad` (`texelFetch`).
- Mask textures (the font atlas for text, a mask mesh's texture) are egui_glow's own, resolved by
  the paint callback (`Painter::texture`; unknown: `fail_texture`) and bound on unit 0 of the
  coverage draw only. The coverage target is bottom-up like every target (clip-space y up), so
  the prelude's `flip_y` reads it like the capture.
- User textures are only read, never modified or kept: egui_glow applies a frame's texture deltas
  before painting and frees after it, so an id resolved in the callback is current and alive.
- Every wiring decision (which slot a draw binds and writes, mappings, uniform bytes) comes from
  `src/plan.rs`; this module maps `plan::Slot` to `RenderTarget`s only.
- State (F7): per `(viewport, id)` two `RenderTarget`s in `GlowState::states` (RGBA16F only when the
  float-intermediate probe passed — half-float filtering is core in GL 3.3 / GLES 3.0 — else the
  layer is refused, `GlowState::fail_state`; RGBA8 opt-in), sampled on unit 2 (`state()`,
  `texelFetch` for `state_load`); an entry is updated only by a clean run (`LayerOutcome::Done`),
  any other outcome forgets its content. Evicted / replaced targets are freed in the paint
  callback; `destroy_glow` frees all.
- Region masks (R1): after the additive coverage draw, the region-id target (the window at its
  origin, like the coverage) is cleared to id 255 (`REGION_NONE`) and every region's draw range is
  drawn with `fs_region`, blending disabled, twice (`plan::REGION_PASS_THRESHOLDS`, pass A then B,
  regions in id order, `plan::region_locals` re-uploaded per region) — the same sequence as
  wgpu's `coverage.rs`, so the later region owns a pixel. The plan binds it on unit 4 for every
  PASS step (`target_of(Slot::RegionId)`, `allocations().region`); the composite and layers
  without regions get the white dummy (`REGION_NONE`). A packed mask that does not match its plan
  (`CoverageResources::fits`) latches (crate bug).
- Entries (R3): one paint callback runs every entry. `render.rs` executes `plan::schedule(..,
  CaptureSource::Framebuffer)` op by op and decides no order itself (`Stacked`: capture, run,
  composite per entry, so later captures see earlier composites; `Independent`: every capture
  first). Entry k captures into `Resources::captures[k]` (a grow-only pool); ping-pong, coverage
  and region-id targets are shared (each run is composited before the next; a schedule that
  breaks this, or a hand-off op, latches as a crate bug). A layer of several VISIBLE entries
  composites each with the BLENDED composite (`BlendedComposite`: erase `ZERO, ONE_MINUS_SRC_ALPHA`
  then add `ONE, ONE`, D4), so a later entry lands over an earlier one; a single visible entry
  keeps the replace composite. Programs, the blended programs, the rasterizer, state formats
  and every entry's textures are settled before the first capture (one failure skips the whole
  layer); a failing op later stops the layer, earlier composites stay. States per entry
  (`LayerJob::state_key(k)`: keyed by the entry's user index `EntryJob::index`, not the job
  position k; reset ticket claimed once per paint), started at `Run(k)` and
  finished after `Composite(k)` (`StateRegistry::get_mut`). The blended composite rounds twice,
  so AA edges of overlapping entries may differ from wgpu by one 8-bit step.
- Backend-wide failures (resource creation, unexplained capture / composite GL error) latch
  `GlowState::failure`; per-effect failures (translation, compile/link, a GL error of that
  effect's draw — attributed directly for single-effect chains, else by per-pass checks on later
  frames via `GlowState::suspects`) latch only that effect. An uncapturable framebuffer format
  skips only its layers. All are logged once at `error` level.
- Targets grow in `GROW_STEP` (64 px) steps, clamped to `MAX_TEXTURE_SIZE`/`MAX_RENDERBUFFER_SIZE`.
- `install_glow` on a context that already has a glow backend destroys the old one's objects when
  given the same `glow::Context` object (pointer identity), otherwise leaves them and warns.
- Uniform locations are never stored (`!Send` on wasm32); sampler units and the block bindings
  (`fl` -> `UNIFORM_BINDING_POINT`, `p` -> `PARAMS_BINDING_POINT`) are set once after linking.
- WebGL1 / GLES2 / GL < 3.3 are rejected at install (`glsl_target`), before any GL object exists.

## Tests
- Unit tests: translation and dialect choice in `glsl.rs`.
- Runtime: `tests/glow_headless.rs` (Linux, `#[ignore]`d; `cargo test --all-features --test
  glow_headless -- --ignored`) paints real egui frames through `egui_glow::Painter` on a
  windowless EGL device context into a plain / 4x MSAA FBO and checks read-back pixels
  (invert, lossless identity, rounded mask, multi-pass chains, zoom, alpha, per-effect failure,
  named parameters per pass, `original()` in later passes, pointer / frame counters, scaled
  passes, user textures: exact / nearest / linear, updates, three slots, unknown / freed ids,
  shape masks, regions: grid without seam / fringe incl. 4x MSAA, `REGION_NONE`, later-wins overlap,
  entries: far-apart windows, `Independent` / `Stacked` overlap incl. AA edge and 4x MSAA, three
  stacked entries, `Stacked` == `Independent` outside the earlier mask, per-entry states / textures).
  Extend it with every change to this backend; the demo example needs a display and is no test.

## Editing map
- GLSL dialect or translation issues: `glsl.rs`.
- Mask coverage / region-id rasterization: `coverage.rs` (what the step does: `plan::CoverageStep`).
- New GL object or format, resolve format mapping (`ReadFormat::resolve_format`): `resources.rs`.
- Capture / pass / composite order or state handling: `render.rs` (what each draw binds:
  `src/plan.rs`).
