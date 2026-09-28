# Module: examples

## Purpose
Runnable examples of the public API: the interactive demo (a directory example) and two offline
processors. They are also the manual test bench for both backends.

## Files and submodules
- `demo/`: interactive showcase (`--renderer glow|wgpu`, `--msaa N`, `--scene NAME`,
  `--stack a,b`, `--rect x,y,w,h`, `--screenshot out.ppm` for scripted checks,
  `--record DIR [--frames N] [--fps F]` for README GIF frames). Requires both backend features.
  - `main.rs`: command line, stderr logger, eframe start-up on either renderer, the app shell
    (status block, scene selector, lazily built scenes, dispatch).
  - `record.rs`: `--record` / `--screenshot` machinery and the PPM writer; scene-independent.
  - `backdrop.rs`: the shared busy content scenes paint beneath their layers (`Backdrop::paint`,
    `CircleMotion`, `scroll_widgets` of real widgets).
  - `stack.rs`: the preset stack editor (`Kind`, `Entry`, `StackEditor`, `parse_stack`) incl. the
    live-compiled custom WGSL effect; reusable by any scene.
  - `params_ui.rs`: generic editor over `Effect::params()` (`pass_params_ui`): a `DragValue` per
    scalar / vector component, a colour picker for a `vec4` named `*color*` / `*colour*`, a grid
    for `mat4x4`, arrays collapsed.
  - `scenes/`: `mod.rs` (the `Scene` trait, `SceneEntry` registry `SCENES`, `SceneInit`,
    `SceneFrame`, shared `DragRect` and `record_rect`) and one file per scene: `stack.rs`
    (scene "stack": the preset stack on a draggable rect).
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
