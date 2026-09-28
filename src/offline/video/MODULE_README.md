# Module: src/offline/video

## Purpose
Video processing (feature `video`): filter streams of frames with the offline renderer, move them
between sources and sinks, and adapt the ffmpeg programs and image sequences. No GPU work and no
Rust dependency of its own: every frame goes through `OfflineRenderer::run`.

## Architecture
```text
VideoSource::read_frame ──► VideoProcessor (renderer + chain + options + output buffer)
   (straight RGBA8, time)        │ run_frame: options.with_time(time), OfflineRenderer::run
                                 ▼
                         VideoSink::write_frame (straight RGBA8, time)
process_video / process_video_async = drive(): begin → loop { read, filter, write, progress }
                                              → finish | abort (error, cancellation)
```
Four usage levels, lowest first: `VideoProcessor::process_frame*` (own decode loop); the
`VideoSource` / `VideoSink` traits + `process_video`; `ffmpeg/` (native); `sequence.rs` (native,
`image` feature too).

## Files and submodules
- `mod.rs`: `VideoProcessor`, `VideoProgress`, `process_video` (native, blocking),
  `process_video_async`, the shared `drive` / `frame_loop`; module docs (four levels, timing).
- `stream.rs`: `FrameRate` (reduced rational; exact frame times), `VideoInfo`, the traits (their
  docs sketch ffmpeg-next and GStreamer implementations).
- `error.rs`: `VideoError` (wraps `OfflineError`; adapter, stream-contract, cancellation and
  user-implementation variants).
- `ffmpeg/`: `ffmpeg` / `ffprobe` command-line adapters (see its readme).
- `sequence.rs`: `ImageSequenceSource` / `ImageSequenceSink` and the `%0Nd` pattern parser.

## Contracts and invariants
- Frames crossing the traits are straight-alpha RGBA8 of `VideoInfo`'s size; `drive` rejects a
  source frame of the wrong length (`FrameBytesMismatch`) and always asks the processor for
  straight output, whatever the options' output alpha.
- `fl.time` of a frame is the time the source gives (seconds from the first frame), as `f32`.
- `fl.frame` / `fl.delta_time` come from the processor's `FrameClock` (frame index from 0, time
  minus the previous processed frame's, 0 first, never negative), advanced only after a
  successful frame and reset by `drive` per stream (`reset_frame_clock` for custom loops); values
  set explicitly in `OfflineOptions` win.
- A stateful chain's state (F7) is the renderer's, so it continues from frame to frame; `drive`
  resets it together with the frame clock per stream (`VideoProcessor::reset_state` for custom
  loops).
- The driver calls `begin` once, then `write_frame` per frame, then exactly one of `finish`
  (success) or `abort` (a frame-loop error or cancellation, after a successful `begin`); a
  failing `begin` or `finish` is not followed by `abort`. Cancellation is `ControlFlow::Break` from the
  progress callback → `VideoError::Cancelled`.
- Blocking entry points are native-only and never belong on a GUI thread; the async ones do not
  poll the device (same rule as the offline core).
- Sinks never flatten transparency silently: outputs without alpha reject non-opaque frames
  (`AlphaNotSupported`); `OfflineOptions::with_background` is the explicit way to flatten.

## Editing map
- Per-frame behaviour / time and counter mapping: `VideoProcessor::run_frame`, `frame_options`,
  `FrameClock` (`mod.rs`).
- Driver order, progress, cancellation: `drive` / `frame_loop` (`mod.rs`).
- Stream metadata or trait contract: `stream.rs` (a trait change is a breaking API change).
- New adapter: a new file implementing the traits, gated like `sequence.rs`.
- Tests: unit tests per file; GPU + ffmpeg end-to-end tests in `tests/video.rs` (ignored).
