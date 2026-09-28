# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-09-28

### Added
- `FilterLayer`: filters everything painted beneath a rect, with an optional rounded mask.
- `Effect`: custom filters written as one WGSL function, validated by naga at construction with
  typed errors located in the user's snippet. `EffectError::label()` / `location()` /
  `diagnostic()`; diagnostics are rendered against the user's snippet. `override` declarations
  are rejected by `Effect::new` on every backend.
- Effects reading `fl.time` are detected as animated (continuous repaint);
  `Effect::with_animated` overrides it.
- `Pass` / `FilterChain`: 16 `vec4` parameters per pass, ping-pong execution, per-pass sample
  radius for capture padding.
- Presets: brightness/contrast, saturation, grayscale, invert, tint, Gaussian blur, sharpen,
  pixelate, ripple (animated), lens, vignette, frosted glass.
- Distances are in layer points: `fl.pixels_per_point` is egui's pixels-per-point times the
  layer transform's scale (an `egui::Scene` zoom), the same factor the capture padding and corner
  radii use, so effects and their padding scale together inside transformed layers.
- Mask coverage within 1/1024 of 0 or 1 is treated as exactly 0 or 1, so pixel-aligned layer
  edges are exact on every backend.
- `status()` / `failed_effects()` for diagnosing a missing or failed backend.
- `egui_glow` backend (OpenGL 3.3+, OpenGL ES 3.0+, WebGL2): framebuffer capture in the
  framebuffer's own colour format (multisample resolve through a same-format renderbuffer),
  RGBA16F intermediates where renderable, explicit `destroy_glow`. Unsupported formats skip only
  the affected layers; a GL error from one effect's draw disables only that effect; re-installing
  on the same GL context frees the previous backend's objects; offscreen targets grow in
  64-pixel steps.
- `egui-wgpu` backend (`install_wgpu`, `WgpuOptions`; native and web): the content beneath each
  filter layer is re-rendered offscreen (it is drawn twice per frame), selected from egui's final
  paint order by an egui plugin; MSAA, depth/stencil and `*Srgb` targets supported. The
  composite blends over egui's pass by mask coverage, so the corners outside a rounded mask keep
  what is on screen (a lower filter layer or a paint callback stays visible there). On `*Srgb`
  targets the backdrop re-render blends in linear like egui itself (through an sRGB view of the
  capture), so semi-transparent content under a filter matches the unfiltered screen; GL /
  WebGL2 lack texture view formats and keep gamma blending (logged once). Fragments derive their
  `uv` from their pixel position, and a chain run can cover a window of the capture, so offline
  tiles go through the UI's own chain executor.
- `WgpuOptions` is a builder with private fields whose setters mirror eframe's option names
  (`with_multisampling`, `with_depth_buffer`, `with_stencil_buffer`, `with_dithering`,
  `with_clear_color`, plus `with_depth_stencil_format`), getters, a `Default` equal to eframe's
  defaults, and an exact `From<&egui_wgpu::RendererOptions>` for custom integrations. The options
  must match egui's render pass: a mismatch cannot be detected by the crate and makes wgpu reject
  egui's frame (a panic natively and on WebGL2, a lost frame on browser WebGPU).
- Offline processing (feature `image`): `OfflineRenderer` (`new`, `from_render_state`,
  `headless` / `headless_async`) runs chains over RGBA8 frames (`FrameRef`, straight or
  premultiplied alpha, any stride) with `OfflineOptions` (pixels per point, time, background,
  corner radius, output alpha, tile size); `process` / `process_into` (blocking, native),
  `process_async`; typed `OfflineError`. Bit-identical to a filter layer over the same image in
  the UI on the same device (conditions in the README). Frames above the texture limit are
  tiled: above `offline::DEFAULT_MAX_TILE_SIZE` (4096) by default and always within the device's
  texture and buffer limits; GPU out-of-memory and validation errors of a run are typed
  (`OfflineError::OutOfMemory`, `OfflineError::Gpu`, also awaited on the web), and a failed or
  cancelled readback leaves the renderer usable. `image` crate adapters (`process_image`, `process_file`; PNG and JPEG built in) and the
  `process_image` example.
- Video processing (feature `video`, no extra Rust dependencies): `VideoProcessor` filters frame
  streams with its renderer, pipelines and buffers reused across frames (`process_frame`,
  `process_frame_into`, `process_frame_async`; the frame's time becomes `fl.time`); object-safe
  `VideoSource` / `VideoSink` traits with `VideoInfo` and an exact rational `FrameRate`;
  `process_video` / `process_video_async` with progress and cancellation; typed `VideoError`.
  Native adapters to the `ffmpeg` / `ffprobe` programs (5.1+): `FfmpegSource` (ffprobe metadata,
  rotation applied, constant-frame-rate decode over a pipe), `FfmpegSink` (codec presets H.264,
  VP9, AV1, ProRes 4444, FFV1, GIF or custom arguments; audio copied, re-encoded or dropped),
  `FfmpegTranscode` for one-call file processing; a missing program, a failing ffmpeg (with the
  tail of its stderr), malformed probe output, truncated frames and a closed pipe are distinct
  errors; child processes are killed and reaped on error or drop. Translucent frames are rejected
  for codecs without alpha unless a background is set. Colour: decoding honours the input's
  matrix and range tags (untagged: BT.709 for HD, BT.601 for SD; HDR / wide gamut rejected),
  YCbCr outputs are converted with BT.709 to limited range and tagged BT.709, FFV1 tagged sRGB.
  VP8/VP9 alpha inputs decode with libvpx; cover art is skipped; the sample aspect ratio is kept
  (`VideoInfo::with_sample_aspect_ratio`); paths go through the `file:` protocol; ffmpeg older
  than 5.1 is a typed error; outputs are written to a temporary file and renamed on success,
  and an output equal to the input is refused. With `image` as well:
  `ImageSequenceSource` / `ImageSequenceSink` (`frame_%04d.png`). `process_video` example.
- `FrameRef::pixels` / `FrameRef::stride`; offline runs reuse their prepared-input buffer.
- `demo` example (`--renderer glow|wgpu`, scripted `--stack name:values`, `--rect`,
  `--screenshot`, seamless-loop `--record` used for the README GIFs).

[0.1.0]: https://github.com/Vasyanator/egui_filter_layers/releases/tag/v0.1.0
