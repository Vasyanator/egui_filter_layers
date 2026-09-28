# Module: examples

## Purpose
`demo.rs`: interactive showcase and manual test bench (`--renderer glow|wgpu`, `--msaa N`,
`--stack a,b`, `--rect x,y,w,h`, `--screenshot out.ppm` for scripted checks). Uses plain egui
widgets. The scene is animated, so it requests a repaint every frame. On wgpu the side panel
shows the "rendered twice" warning.

`--record DIR [--frames N] [--fps F]` renders the README GIF frames: fixed 800x500 window at
1 px/pt, no side panel, pointer input dropped, `RawInput::time` pinned to `step / fps` in
`App::raw_input_hook`. A step advances only when its tagged screenshot (`UserData` = step) arrives,
so each `frame_NNNN.ppm` is exactly one step. Everything animated in this mode (rect path, circle
swing) is periodic with the loop length; effect speeds passed via `--stack name:v1:...` must be too.

`process_image.rs` (feature `image`): offline processing of an image file on a headless wgpu
device (`OfflineRenderer::headless` + `process_file`) with a fixed stack incl. a custom effect;
`--ppp`, `--time`, `--max-tile` map to `OfflineOptions`. JPEG output of translucent input relies on
the example's white background.

`process_video.rs` (feature `video`): offline processing of a video file through the ffmpeg
programs (`FfmpegTranscode`) on a headless wgpu device with an animated stack (vignette + ripple,
so `fl.time` follows the video); `--codec`, `--crf`, `--audio copy|drop|opus`, `--background`,
`--ppp`; prints progress.

## Contracts
- Uses only the public API of the crate. Shows `status()` so a missing/failed backend is visible.
