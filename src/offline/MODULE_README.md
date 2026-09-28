# Module: src/offline

## Purpose
Offline processing (features `image` and `video`, gated together in `lib.rs`): run a
`FilterChain` over an ordinary RGBA8 frame on wgpu and read the pixels back, producing what a
`FilterLayer` covering that frame shows in the UI. wgpu only (glow apps enable `image` / `video`
for export; headless GL is not supported).

## Architecture
```text
FrameRef (straight / premultiplied RGBA8, stride)
   │ frame.rs: validate, premultiply with egui's rounding, optional background (CPU)
   ▼
OfflineRenderer::run (mod.rs)
   │ frame_geometry: layer_geometry(frame rect, screen = frame)  ── same as the UI
   │ tiles.rs: one run, or overlapping power-of-two windows
   │ per tile: write_texture(window) → wgpu_backend::chain::record(ChainWindow)
   │           → draw_composite into the output texture → copy to readback buffer → submit
   ▼
readback.rs: map_async future (blocking helpers poll with Wait) → rows into the frame
   │ frame.rs: unpremultiply (egui rounding) unless premultiplied output was asked for
   ▼
RgbaFrame / image::RgbaImage (image.rs)
```
The renderer owns its device handles, a `ChainResources` (pipelines are per renderer, compiled
from the same WGSL as the UI's), exact-size textures (recreated when the allocation changes), the
readback buffer and the prepared-input buffer; everything is reused across calls, which is what a
frame stream needs (`video/` builds on exactly that).

## Files and submodules
- `mod.rs`: `OfflineRenderer` (constructors, `process` / `process_into` / `process_async`,
  `run`, per-tile recording), `frame_geometry`, upload / readback copies.
- `texture.rs`: `OfflineTexture` (owned premultiplied RGBA8 + egui `TextureOptions`, shared by
  clones; `from_color_image` = egui's upload bytes; identity decides re-uploads).
- `options.rs`: `OfflineOptions` (persistent state options, `with_state`; user textures per slot, `with_texture`; validated before GPU work; `delta_time` / `frame` remember
  whether they were set, so a video processor fills only unset ones), `PointerState` (frame uv),
  `input_uniforms` (the run's `fl.delta_time` / `fl.frame` / `fl.pointer`, the pointer through
  the UI's `geometry::pointer_uniform`).
- `frame.rs`: `FrameRef`, `RgbaFrame`, `AlphaMode`, colour conversions (unit-tested).
- `tiles.rs`: pure tile planning (unit-tested).
- `readback.rs`: row padding, `MapFuture`, native `block_on` (no async runtime dependency).
- `error.rs`: `OfflineError`.
- `image.rs` (`image` feature): `process_image`, `process_image_async`, `process_file`;
  `format_has_alpha` (shared with the image-sequence sink).
- `video/` (`video` feature): video on top of `OfflineRenderer::run` (see its readme).

## Contracts and invariants
- Chain execution ONLY through `wgpu_backend/chain.rs` (`record`, `draw_composite`,
  `composite_pipeline`: the REPLACE composite, since the input is the whole backdrop and the
  output is cleared first); no pass or composite logic here. Uniform values come from
  `FrameUniforms::for_layer` via `record`. The UI instead blends its composite over egui's pass,
  so a rounded mask's 1-px antialiased edge is not bit-identical (one step; on `*Srgb` it blends
  in linear, up to ~70 steps measured).
- Bit-identity with the UI rests on: same device; the same `layer_geometry` inputs (frame = screen,
  layer = frame, transform scale 1); an 8-bit `Rgba8Unorm` input equal to egui's texture bytes
  (`Color32::from_rgba_unmultiplied`); textures allocated exactly at the frame size when untiled
  (like a freshly allocated UI capture). Hardware bilinear sampling BETWEEN texel centres depends
  rarely (<= 2 steps) on texture size / window origin: tiles and a UI capture that grew larger
  than the layer are exact only for texel-centre sampling (tested in `tests/offline_equivalence.rs`).
- A chain with a scaled pass (`Pass::with_scale`) is never tiled: when the plan needs more than
  one tile, `run` returns `OfflineError::NotTileable` before any GPU work (scaled sizes are
  rounded over the WHOLE frame; tile windows cannot reproduce them). Whole-frame runs equal the
  UI bit for bit (same targets, same plan).
- User textures: checked against the chain with the UI's `plan::check_textures` before any GPU
  work (`OfflineError::Texture`); each slot's upload (`Rgba8Unorm`, egui-wgpu's sampler for its
  options, `ChainResources::user_layout`) is kept while the same `OfflineTexture` stays attached
  and dropped when the slot empties; uploads run in awaited `ErrorScopes`. They are sampled in
  their own uv, so tiles need nothing for them (tiled == whole, tested).
- Persistent state (F7): a chain with a state pass runs over the renderer's one `OfflineState`
  (two state textures, the latest's index and size), created on first use or when the format
  changes, carried across runs (a `VideoProcessor` continues it frame to frame; `process_video`
  and `reset_state` start it over), dropped after a GPU error. Such a chain is never tiled
  (`NotTileable`); an RGBA16F state on a device that cannot render / filter it is
  `OfflineError::State`.
- Tiles overlap by the chain padding + 1 px; effects must read no further than their sample radius
  (the `lens` preset needs an explicit radius when tiled).
- Every size is checked (`MAX_SIDE` = 2^24 - 1 per side, `usize` / `u32` / `u64` arithmetic);
  `frame_geometry` rejects a scale whose geometry is not exactly the frame (f32 rounding).
- Memory: the tile side is `min(OfflineOptions cap (default DEFAULT_MAX_TILE_SIZE = 4096; None =
  device), max_texture_dimension_2d, readback side fitting max_buffer_size)`. Textures and the
  readback buffer are sized exactly per plan (recreated, never only grown). By default frames
  above 4096 px are tiled: exact for texel-centre effects, <= 2 steps on a few values otherwise.
- Per-frame GPU work (allocations in `ensure_allocations`, each tile's record + submit) runs in
  `gpu::ErrorScopes` (out-of-memory + validation + internal) whose result is AWAITED (so also on
  the web): typed `OfflineError::OutOfMemory` / `Gpu`, no panic; after an error the textures and
  buffer are dropped. The readback buffer is taken out of the renderer while mapped, so a failed
  or cancelled (`process_async` dropped) readback drops it instead of leaving it mapped.
- Blocking entry points are native-only and must not run on a GUI thread; `process_async` does
  not poll the device (on native someone else must: eframe's submissions, or `device().poll`).

## Editing map
- New input / output pixel format: `frame.rs` (convert to the 8-bit premultiplied rows).
- Tile strategy: `tiles.rs`. Readback / async behaviour: `readback.rs`, `run` in `mod.rs`.
- Codec adapters: `image.rs`; video: `video/` (drives `run` per frame, time per frame).
- Tests: unit tests in each file; GPU equivalence and tiling in `tests/offline_equivalence.rs`.
