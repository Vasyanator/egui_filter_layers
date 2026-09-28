# Module: examples

## Purpose
Runnable examples of the public API: the interactive demo (a directory example) and two offline
processors. They are also the manual test bench for both backends.

## Files and submodules
- `demo/`: interactive showcase (`--renderer glow|wgpu`, `--msaa N`, `--scene NAME`,
  `--stack a,b`, `--rect x,y,w,h`, `--screenshot out.ppm` for scripted checks,
  `--record DIR [--frames N] [--fps F]` for README GIF frames). Requires both backend features and `hot-reload`
  (`[[example]] required-features`; the "hot" scene uses `HotEffect`).
  - `main.rs`: command line, stderr logger, eframe start-up on either renderer, the app shell
    (status block, scene selector, lazily built scenes, dispatch).
  - `record.rs`: `--record` / `--screenshot` machinery and the PPM writer thread (`FrameWriter`,
    bounded queue: the GUI thread does no file I/O; a full queue blocks the hand-over on purpose so
    each step is painted a fixed number of times); scene-independent.
    On native Wayland a script-launched window stalls after two frames (winit waits for a frame
    callback the compositor withholds; a bare eframe app does the same), so scripted captures
    run under XWayland: `WAYLAND_DISPLAY= cargo run --example demo ...`.
  - `backdrop.rs`: the shared busy content scenes paint beneath their layers (`Backdrop::paint`,
    `CircleMotion`, `scroll_widgets` of real widgets).
  - `stack.rs`: the preset stack editor (`Kind`, `Entry`, `StackEditor`, `parse_stack`) incl. the
    live-compiled custom WGSL effect; reusable by any scene.
  - `params_ui.rs`: generic editor over `Effect::params()` (`pass_params_ui`): a `DragValue` per
    scalar / vector component, a colour picker for a `vec4` named `*color*` / `*colour*`, a grid
    for `mat4x4`, arrays collapsed.
  - `scenes/`: `mod.rs` (the `Scene` trait, `SceneEntry` registry `SCENES`, `SceneInit`,
    `SceneFrame`, shared `DragRect` and `record_rect`) and one file per scene: `stack.rs`
    (scene "stack": the preset stack on a draggable rect); `hot.rs` + `hot.wgsl` (scene "hot": a
    `HotEffect` watching `hot.wgsl`, whose path is fixed at compile time from
    `CARGO_MANIFEST_DIR`; an in-app editor saving the file from a worker thread, compile errors
    as an overlay on the layer and in the panel, named values kept across reloads by `rebind`);
    `bloom.rs` + `bloom_{bright,combine}.wgsl` (scene "bloom": own dark content; a custom bloom
    with a scaled bright pass, `presets::bloom`, or Gaussian vs Kawase blur alone, with an averaged
    frame-time readout); `textures.rs` + `tex_{displace,refract,lut}.wgsl` (scene "textures":
    noise, normal map and a 256x16 flattened 16^3 LUT generated at build and attached to slots
    0-2, one custom effect per slot; shows `check_textures` / `failed_textures`); `mask.rs` (scene
    "mask": star from overlapping convex kites, circle and a "GLASS" galley as `Mask::Shape`
    layers, draggable, frosted / lens / pixelate); `ripples.rs` (scene "ripples": full-area
    `click_ripples` driven by an `Impulses` from presses / drags, lifetime and capacity controls,
    live count; `scripted_impulses` = recording clicks as a pure function of the phase, shared
    with water); `water.rs` (scene "water": full-area `wave_sim` substeps + `wave_display` under
    an `egui::Window` menu whose rect is a `WaveObstacle` every frame, a draggable rock obstacle
    painted after the layer, splashes, pointer wake, rain, `reset_state`); `trails.rs` (scene
    "trails": `presets::trails` on a `DragRect` over the moving circles, an orbiting comet and a
    pointer brush).
- `process_image.rs` (feature `image`): offline processing of an image file on a headless wgpu
  device (`OfflineRenderer::headless` + `process_file`) with a fixed stack incl. a custom effect;
  `--ppp`, `--time`, `--max-tile` map to `OfflineOptions`. JPEG output of translucent input relies
  on the example's white background.
- `process_video.rs` (feature `video`): offline processing of a video file through the ffmpeg
  programs (`FfmpegTranscode`) on a headless wgpu device with an animated stack (vignette + ripple,
  so `fl.time` follows the video); `--codec`, `--crf`, `--audio copy|drop|opus`, `--background`,
  `--ppp`; prints progress.

## Demo architecture
- The app owns one `Option<Box<dyn Scene>>` per registry entry, built on first show (from the
  side panel, or directly while recording) with `SceneInit` (`--rect`, `--stack`, the context).
  Scene state (incl. persistent layer state) therefore survives switching scenes.
- Per frame: the side panel shows the shared status (renderer, `status()`, install error,
  `failed_effects()`), the selector, then `Scene::controls`; the central panel calls `Scene::ui`
  with `SceneFrame { backdrop, time, phase }`. The app requests a repaint every frame.
- Named parameters in the stack editor: kawase blur, bloom and a custom effect declaring
  `struct Params` keep their built passes; sliders are constructor arguments (a change rebuilds,
  resetting named edits) and `params_ui` edits each built pass in place.

## Adding a scene
1. Create `demo/scenes/<name>.rs` with a `Scene` implementation and
   `pub const ENTRY: SceneEntry = SceneEntry { key: "<name>", title: "...", build };`.
2. In `demo/scenes/mod.rs`: add `mod <name>;` and `<name>::ENTRY,` to `SCENES`. Nothing else
   changes (`--scene <name>`, the selector and `--record` pick it up).

## Contracts
- Uses only the public API of the crate. Shows `status()` so a missing/failed backend is visible.
- `--record`: fixed 800x500 window at 1 px/pt, no side panel, pointer input dropped,
  `RawInput::time` pinned to `step / fps` in `App::raw_input_hook`; a step advances only when its
  tagged screenshot arrives, so each `frame_NNNN.ppm` is exactly one step. A scene must make
  everything it animates itself periodic in `SceneFrame::phase` (`Some` only while recording);
  effect speeds passed via `--stack name:v1:...` must be periodic with the loop too.
- Scene constructors and `Scene::ui` run on the GUI thread: no file I/O or heavy work there
  (use worker threads, e.g. the hot-reload scene's writer).
- Scene-owned layers that keep state need a stable `FilterLayer::id` distinct across scenes.
- Stateful layers step once per PAINT. The recorder paints every step a fixed number of times
  (its screenshot handshake: currently twice), so recordings of stateful scenes are deterministic
  but step faster than one step per frame and are not seamless loops (state crosses the seam);
  `fl.delta_time` is 0 on the repeated paint, so per-second effects (trails) are unaffected.
