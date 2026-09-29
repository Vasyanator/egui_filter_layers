# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-09-29

### Renamed
- **The crate is now `egui-shader-layers` (library `egui_shader_layers`); 0.1.0 was published as
  `egui_filter_layers`. The main type `FilterLayer` is now `ShaderLayer`.** Other names
  (`FilterChain`, `Pass`, `Effect`, presets, ...) are unchanged. Migration: replace the
  `egui_filter_layers` dependency with `egui-shader-layers = "0.2"`, then change
  `egui_filter_layers::` paths to `egui_shader_layers::` and `FilterLayer` to `ShaderLayer`.

### Added
- `original(uv)` in the effect prelude: the untouched captured backdrop, readable in every pass
  of a chain (UI and offline, also tiled).
- Frame inputs in the `fl` block: `fl.delta_time` (egui's `stable_dt` clamped to `0..=0.1`),
  `fl.frame` (egui's `cumulative_frame_nr`, wrapping) and `fl.pointer` (`xy` in capture uv, `z`
  primary button down, `w` pointer present), the same for every layer of a frame. Reading them
  does not make an effect animated. Offline: `OfflineOptions::with_delta_time` / `with_frame` /
  `with_pointer` (`PointerState::new`); `VideoProcessor` derives delta time and frame index from its
  frame clock (`reset_frame_clock`), explicit options win.
- `Impulses` (`Impulse`, `ImpulsePack`, `IMPULSE_SLOTS`, `IMPULSES_PARAM`,
  `IMPULSE_COUNT_PARAM`): a user-owned, bounded collection of timed impulses (clicks, taps)
  positioned in layer-rect uv, aged against a caller-supplied clock, oldest dropped first;
  `apply` fills an effect's named parameters, `request_repaint_if_alive` repaints only while an
  impulse lives. Preset `presets::click_ripples`.
- Named, typed effect parameters: a snippet's `struct Params { ... }` is bound as `p`; values are
  set per pass by name and type-checked (`Pass::set` / `with` / `set_slice` / `get` /
  `get_slice`, `ParamError`, `ParamValue` incl. `From<egui::Vec2>` / `From<egui::Color32>` and
  `ParamValue::linear_rgba`); `Effect::params` / `param` reflect the members (`ParamInfo`,
  `ParamType`) so a UI can build its controls. Members start at zero; structs up to 16 KiB.
- Reduced-resolution passes: `Pass::with_scale` / `set_scale` / `scale` (clamped to
  `[MIN_PASS_SCALE, 1]`), prelude `src_texel()` / `out_texel()` and `fl.pass_scale`; the final
  composite upsamples a scaled last pass. Presets `presets::kawase_blur` (dual-Kawase pyramid)
  and `presets::bloom`.
- User textures: `ShaderLayer::texture(slot, TextureId)` / `texture_at` / `check_textures`,
  `TEXTURE_SLOTS` (3); prelude `tex(i, uv)` (the texture's own egui sampler), `tex_size(i)`,
  `tex_load(i, px)`; `Effect::reads_texture`; an effect reading an unattached slot is refused
  (`TextureError`), an unknown texture id skips its layers (`failed_textures`, `TextureFailure`).
  Offline: `OfflineTexture` (`new`, `from_color_image`) with `OfflineOptions::with_texture`,
  bit-identical to the UI; `OfflineError::Texture` / `TextureTooLarge`.
- Persistent per-layer state: state passes (`Effect::new_state`, `Effect::is_state`) write two
  ping-pong textures that survive between frames; prelude `state(uv)`, `state_load(px)`,
  `state_px(offset)`, `state_size()`, `state_texel()`, `fl.pass_index` (substeps).
  `ShaderLayer::id` (required for stateful chains: `StateError::MissingId`, `check_state`,
  `layer_id`), `ShaderLayer::state(StateOptions)` (`with_scale`, `with_format`: `StateFormat`
  RGBA16F by default, RGBA8 opt-in), `reset_state`, `state_count`, `failed_states`
  (`StateFailure`); states unpainted for 120 frames of their own viewport and 2 s are freed (an
  idle viewport keeps its states while another animates); a `reset_state` requested while the
  layer draws nothing (off screen) stays pending until a paint actually runs. Offline the renderer
  carries the state across calls (`OfflineOptions::with_state`, `OfflineRenderer::reset_state`,
  `VideoProcessor::reset_state`; `OfflineError::State`). Presets `presets::trails`,
  `presets::wave_sim` + `presets::wave_display` with `WaveObstacle` (`new`, `from_rect`) /
  `WAVE_OBSTACLES`.
- Shape masks: `Mask` (`RoundedRect`, `Shape`: any egui shape incl. meshes and text galleys, in
  layer points), `MaskError` (callbacks, and malformed meshes: `InvalidMesh`),
  `ShaderLayer::mask` / `layer_mask` / `check_mask`,
  `OfflineOptions::with_mask`, `OfflineError::Mask`. The shape's alpha is the coverage; the
  coverage of several pieces is the sum of their alphas, capped at 1 (pieces meeting edge to edge
  leave no seam), and a shape or region coverage from 254/255 up counts as full (two pieces
  sharing an edge through pixel centres round their half feathers separately in the 8-bit
  coverage; on `*Srgb` targets that one-step shortfall would show as a seam). The rounded-rect
  mask keeps 0.1's snap within 1/1024 of 0 or 1.
- Regions: `Mask::Regions` / `ShaderLayer::regions` with `Region` (`rounded_rect`, `shape`,
  `shape_ref`), up to `MAX_REGIONS` (255, `MaskError::TooManyRegions`): ONE capture and one
  chain run serve many areas. The union of the regions is the mask and every pass reads the
  owning region's id with the prelude's `region()` / `region_at(uv)` (`REGION_NONE` = 255
  outside the regions; also `REGION_NONE` in Rust), so per-region values are named-parameter
  arrays indexed by it; later regions win where they overlap, and the boundary between two
  regions is pixel-hard; `Effect::reads_regions`. Runs on the wgpu and glow backends and
  offline (`OfflineOptions::with_mask`, tiled runs included).
- Entries: several independent areas in one layer — `Entry` (`new`, `mask`, `corner_radius`,
  `texture`, `state`, accessors) with `ShaderLayer::from_entries` / `with_entry` / `entries`,
  and `Overlap` (`Independent`: every entry filters the original backdrop; `Stacked`: a later
  entry filters the earlier entries' result, also on wgpu) via `ShaderLayer::overlap` /
  `overlap_mode`. Each entry captures only its own padded rect and runs its own chain; entries
  share one paint callback (on wgpu one selection and tessellation of the content beneath) and
  are composited in list order. Layer-level builders apply to every entry, accessors return the
  first entry's, `ShaderLayer::rect` is the union; a stateful entry keeps its own state under the
  layer id (`reset_state` resets all). Offline: `OfflineEntry`, `OfflineLayer`,
  `OfflineRenderer::process_layer` / `process_layer_into` / `process_layer_async` (bit-identical
  to the UI in both modes, never tiled: `NotTileable` above the tile limit). Runs on the wgpu and
  glow backends and offline. `TextureError::InvalidSlot` / `NotAttached` and
  `StateError::MissingId` name the failing `entry`; `FilterChain::new` is `const`.
- README "Choosing layers, regions or entries" guide and the `bench_many_elements` benchmark
  (`#[ignore]`d, in the wgpu and glow headless suites): N layers vs one layer of N regions vs one
  layer of N entries, per backend, with the measured numbers.
- Feature `hot-reload` (native only, no extra dependencies): `HotEffect` (`watch`,
  `watch_with` + `HotOptions`, `poll` -> `HotReloadEvent`, `current`, `error`, `status_text`)
  recompiles a WGSL file off the GUI thread and keeps the last good effect while the file is
  broken; `rebind` (`Rebound`) moves a pass's named parameter values to a reloaded effect;
  `HotReloadError`, `DEFAULT_POLL_INTERVAL`, `MIN_POLL_INTERVAL`.
- `OfflineError::NotTileable`: chains with a scaled or state pass refuse tiling instead of
  producing a seam.
- `WaveObstacle` is also re-exported at the crate root.
- Headless GPU tests for the glow backend (EGL, Linux) next to the wgpu and offline suites.
- Demo: a scene-based directory example with `--scene NAME` (`stack`, `hot`, `ripples`,
  `water`, `trails`, `bloom`, `textures`, `mask`, `grid`: the many-elements modes live) and a
  generic named-parameter editor.

### Changed
- **Breaking:** `EffectError::Interface` gained a `location: Option<SnippetLocation>` field
  (the `struct Params` declaration for named-parameter errors); patterns that list its fields
  need `location` or `..`, and `EffectError::location()` now returns it.
- **Breaking (WGSL):** new prelude names are reserved — the functions `original`, `src_texel`,
  `out_texel`, `tex`, `tex_size`, `tex_load`, `state`, `state_load`, `state_px`, `state_size`,
  `state_texel`, `region`, `region_at` and the constant `REGION_NONE`. A snippet that defines a
  function, constant or global with one of these names no longer compiles (`EffectError`);
  rename it.
- **Breaking (WGSL):** a snippet's `struct Params` is now the named-parameter block, bound as the
  module-scope `p`. A snippet that used its own `struct Params` for something else must rename it
  (unsupported member types are an `EffectError::Interface`, and a module-scope `p` of its own
  then collides).
- The new public types are `#[non_exhaustive]` so they can grow in minor releases: `Mask`,
  `HotReloadEvent` (match with a wildcard arm), `WaveObstacle` and `PointerState` (build them
  with `new` / `WaveObstacle::from_rect`), and the output-only `Impulse`, `ImpulsePack`,
  `Rebound` (read their fields; destructure with `..`). `OfflineOptions::state()` stays the
  getter of `with_state` (every `OfflineOptions` getter is the bare name of its `with_*` builder),
  while `ShaderLayer::state(..)` is a builder whose getter is `state_options()`.
- `OfflineError` has new variants (`NotTileable`, `Texture`, `State`, `Mask`,
  `TextureTooLarge`); it is `#[non_exhaustive]`, so exhaustive matches were already impossible.
- `ShaderLayer::corner_radius` and `OfflineOptions::with_corner_radius` are shorthands for
  `Mask::RoundedRect`; a later `mask(...)` replaces them (and vice versa).
- `ShaderLayer::rect` is the union of the layer's entries and `ShaderLayer::chain` the first
  entry's chain; for a layer built with `ShaderLayer::new` (one entry) both return what 0.1 did.
- The crate-private `fl` uniform block grew from 352 to 464 bytes; its documented members keep
  their meaning (undocumented `internal_*` members changed). `presets::gaussian_blur` and
  `presets::ripple` step by `fl.texel` (bit-identical output at full resolution).
- Dependency ranges are unchanged (egui / eframe 0.36, wgpu / naga 30, glow 0.17); MSRV stays
  Rust 1.95.

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

[0.2.0]: https://github.com/Vasyanator/egui-shader-layers/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Vasyanator/egui-shader-layers/releases/tag/v0.1.0
