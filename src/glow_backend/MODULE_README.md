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
5. run passes into ping-pong targets at `(0, 0, w, h)`;
6. composite back into the original framebuffer over the visible region, scissor re-enabled;
7. `apply_outcome`: latch / per-effect failure / suspects / once-per-format rejection log.

## Files
- `mod.rs`: `install_glow`, `destroy_glow`, `GlowBackend` (`Backend` impl), `GlowState`, latches.
- `glsl.rs`: pure WGSL->GLSL translation (naga `glsl-out`, empty writer flags) and dialect choice.
- `resources.rs`: `GlError`, `RenderTarget`, `Resources`, `build_program` (all GL object code).
- `render.rs`: the per-layer GL sequence and state restoration.

## Contracts and invariants
- All GL calls happen with the context current (inside the callback or in `on_exit`).
- egui_glow re-runs `prepare_painting` after each callback; this module restores only what that
  does not: framebuffer bindings, texture unit 1, uniform-buffer bindings, own VAO/program.
- Backend-wide failures (resource creation, unexplained capture / composite GL error) latch
  `GlowState::failure`; per-effect failures (translation, compile/link, a GL error of that
  effect's draw — attributed directly for single-effect chains, else by per-pass checks on later
  frames via `GlowState::suspects`) latch only that effect. An uncapturable framebuffer format
  skips only its layers. All are logged once at `error` level.
- Targets grow in `GROW_STEP` (64 px) steps, clamped to `MAX_TEXTURE_SIZE`/`MAX_RENDERBUFFER_SIZE`.
- `install_glow` on a context that already has a glow backend destroys the old one's objects when
  given the same `glow::Context` object (pointer identity), otherwise leaves them and warns.
- Uniform locations are never stored (`!Send` on wasm32); sampler units and the block binding
  are set once after linking.
- WebGL1 / GLES2 / GL < 3.3 are rejected at install (`glsl_target`), before any GL object exists.

## Editing map
- GLSL dialect or translation issues: `glsl.rs`.
- New GL object or format, resolve format mapping (`ReadFormat::resolve_format`): `resources.rs`.
- Capture / pass / composite order or state handling: `render.rs`.
