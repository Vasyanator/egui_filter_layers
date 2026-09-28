# Module: src/glow_backend

## Purpose
The `egui_glow` backend (feature `glow`): OpenGL 3.3+, OpenGL ES 3.0+, WebGL2.

## Architecture
`GlowBackend` (registered via `install_glow`) owns `Arc<Mutex<GlowState>>`. `paint_layer` adds an
`egui_glow::CallbackFn`; inside it `render::paint_layer` runs one layer:
1. save the caller's DRAW/READ framebuffer bindings and `SAMPLES` FIRST (resource creation binds
   framebuffer 0); every exit path restores them;
2. lazily create `Resources` (empty VAO, UBO, capture/ping-pong targets, composite program);
3. fetch or build each pass program (`GlowState::program_for`, cached by module id, evicted when
   the effect is dropped);
4. disable scissor + blend; query the read buffer's `ReadFormat` (cached for framebuffer 0); reject
   the layer for non-normalized formats; multisampled -> resolve with identical rects into a
   renderbuffer of the SAME internal format, then blit the padded capture rect into the RGBA8
   capture texture;
5. plan the run (`plan::plan`, whole capture), grow the ping-pong targets to the plan's
   `target_px`, and run its pass steps into them at `(0, 0, used_px)` (smaller for scaled
   passes; mode-0 uv interpolation over that viewport needs nothing else), binding every `@group(0)` texture unit per draw (the plan's slots, dummies
   elsewhere) and uploading `plan::step_uniforms` (binding point 0) and `plan::params_block`
   (binding point 1, `glBufferData` per draw: the size varies by effect);
6. composite back into the original framebuffer over the visible region, scissor re-enabled
   (`plan::composite_uniforms`);
7. `apply_outcome`: latch / per-effect failure / suspects / once-per-format rejection log.

## Files
- `mod.rs`: `install_glow`, `destroy_glow`, `GlowBackend` (`Backend` impl), `GlowState`, latches.
- `glsl.rs`: pure WGSL->GLSL translation (naga `glsl-out`, empty writer flags), dialect choice,
  the reflected `fl` and named-parameter block names (by `@binding` 0 / 4), and the texture-unit table of the prelude's layout (`GROUP0_TEXTURE_BINDINGS`,
  `NEAREST_UNIT_BASE`).
- `resources.rs`: `GlError`, `RenderTarget`, `Resources` (incl. dummy textures and the nearest
  sampler object), `build_program` (all GL object code).
- `render.rs`: the per-layer GL sequence and state restoration.

## Contracts and invariants
- All GL calls happen with the context current (inside the callback or in `on_exit`).
- egui_glow re-runs `prepare_painting` after each callback; this module restores only what that
  does not: framebuffer bindings, textures and sampler objects on units 0-3 / 7-10,
  uniform-buffer bindings (points 0 and 1), own VAO/program.
- Texture units: `GROUP0_TEXTURE_BINDINGS[i]` is unit `i` through the linear sampler (the
  texture's own `LINEAR` parameters) and unit `NEAREST_UNIT_BASE + i` (7-10) through the nearest
  sampler object; 4-6 are free for user textures. Max unit 10 < the 16 GLES 3 / WebGL2 guarantee.
  naga cannot translate a texture read through both samplers in one module.
- Every wiring decision (which slot a draw binds and writes, mappings, uniform bytes) comes from
  `src/plan.rs`; this module maps `plan::Slot` to `RenderTarget`s only.
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
  named parameters per pass).
  Extend it with every change to this backend; the demo example needs a display and is no test.

## Editing map
- GLSL dialect or translation issues: `glsl.rs`.
- New GL object or format, resolve format mapping (`ReadFormat::resolve_format`): `resources.rs`.
- Capture / pass / composite order or state handling: `render.rs` (what each draw binds:
  `src/plan.rs`).
