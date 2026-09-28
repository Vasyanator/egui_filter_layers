/*
File: lib.rs

Purpose:
Crate root of `egui_filter_layers`: public re-exports, backend installers, and the crate-level
documentation (semantics, shader model, backend matrix).

Key items:
- `FilterLayer`, `FilterChain`, `Pass`, `Effect` (re-exported from their modules)
- `presets` module (its `WaveObstacle` also re-exported at the root: the one preset input type)
- `Impulses` (+ `Impulse`, `ImpulsePack`, param-name constants): input-driven transient effects
- `install_glow()` / `destroy_glow()` (feature `glow`), `install_wgpu()` / `WgpuOptions`
  (feature `wgpu`), `status()`, `failed_effects()`, `failed_textures()`
- user textures: `TEXTURE_SLOTS`, `TextureError`, `TextureFailure` (`FilterLayer::texture`)
- persistent state (F7): `StateOptions`, `StateFormat`, `StateError`, `StateFailure`,
  `reset_state()`, `state_count()`, `failed_states()` (`Effect::new_state`, `FilterLayer::id`)
- `offline` module and its re-exports (`OfflineRenderer`, ...; features `image` / `video`)
- `HotEffect`, `rebind` (+ options, events, errors; feature `hot-reload`, native only)
- shape masks (F9): `Mask`, `MaskError` (`FilterLayer::mask`, `OfflineOptions::with_mask`)

Notes:
Backends plug in behind the crate-internal `backend::Backend` trait; nothing outside a backend
module names a backend type.
*/

//! GPU **filter layers** for [egui](https://github.com/emilk/egui): a shape that takes
//! everything already painted beneath a rect and redraws that region through a chain of
//! WGSL filter passes — blur, colour adjustments, refraction, or your own shader.
//!
//! ```no_run
//! # #[cfg(feature = "glow")]
//! # fn app_setup(cc_gl: &glow::Context, ctx: &egui::Context) {
//! // Once, at startup (eframe: `cc.gl` in your app creator):
//! egui_filter_layers::install_glow(ctx, cc_gl).ok();
//! # }
//! # fn frame(ui: &mut egui::Ui) {
//! use egui_filter_layers::{FilterLayer, presets};
//! // Every frame, after painting whatever should end up blurred:
//! let rect = egui::Rect::from_min_size(egui::pos2(40.0, 40.0), egui::vec2(240.0, 160.0));
//! FilterLayer::new(rect, presets::gaussian_blur(8.0).then(presets::saturation(1.3)))
//!     .corner_radius(16)
//!     .paint(ui.painter());
//! # }
//! ```
//!
//! What the crate offers, each detailed below:
//!
//! * **Presets** ([`presets`]): blurs (Gaussian, dual-Kawase [`presets::kawase_blur`]), colour
//!   adjustments, sharpen, pixelate, lens, vignette, frosted glass, [`presets::bloom`], animated
//!   ripples, input-driven [`presets::click_ripples`], and the stateful [`presets::trails`] and
//!   water ([`presets::wave_sim`] + [`presets::wave_display`] around [`WaveObstacle`]s).
//! * **Custom effects** ([`Effect`]): one WGSL function with a prelude (the pass input, the
//!   untouched backdrop `original()`, time, frame, pointer, user textures, persistent state),
//!   validated by naga with errors located in your snippet; typed **named parameters** from a
//!   `struct Params` ([`Pass::set`]).
//! * **Chains** ([`FilterChain`]) of passes, each optionally at reduced resolution
//!   ([`Pass::with_scale`]).
//! * **Input**: [`Impulses`] turns clicks into transient animations that repaint only while alive.
//! * **User textures** ([`FilterLayer::texture`]), **persistent per-layer state**
//!   ([`Effect::new_state`], [`FilterLayer::id`]) and **shape masks** ([`FilterLayer::mask`]).
//! * **Backends**: `egui_glow` and `egui-wgpu`; **offline** processing of images and video
//!   (features `image` / `video`); **hot reload** of WGSL files (feature `hot-reload`).
//!
//! # Semantics
//!
//! * **"Beneath" means earlier in egui's paint order**: lower layers and the shapes painted
//!   earlier into the same layer. Content painted later — later shapes of the same layer, higher
//!   `egui::Order` areas, popups, tooltips — is drawn on top and is not filtered.
//! * The filter is **view-only**: it never touches egui textures or your data.
//! * **Neighbourhood effects need padding.** The capture is the layer rect expanded by the chain's
//!   total sample radius ([`FilterChain::padding_points`]), clamped to the screen. At the screen
//!   edge, sampling clamps to the last available pixel.
//! * **Precision**: the backdrop is captured as 8-bit, gamma-encoded, premultiplied RGBA — exactly
//!   what egui wrote. Intermediate pass textures are RGBA16F where the backend can render to it
//!   (wgpu where the adapter supports it, desktop GL 3.3, WebGL2 with `EXT_color_buffer_float`),
//!   otherwise RGBA8;
//!   see [`BackendStatus::Active`].
//! * With no backend installed, or after a backend failure, layers draw nothing extra (the
//!   backdrop stays unfiltered). Nothing panics; check [`status`].
//!
//! # Writing an effect
//!
//! An [`Effect`] is one WGSL function, validated by naga when you build it:
//!
//! ```wgsl
//! fn effect(uv: vec2<f32>) -> vec4<f32> {
//!     let wave = sin(uv.y * 40.0 + fl.time * 3.0) * params(0u).x;
//!     return src_px(vec2<f32>(wave, 0.0));
//! }
//! ```
//!
//! `uv` is the position of the pixel being written in the captured region: `(0, 0)` top-left,
//! `(1, 1)` bottom-right, y down. The prelude provides:
//!
//! | name | meaning |
//! |---|---|
//! | `src(uv)` | the pass input (bilinear, clamped to the captured region) |
//! | `src_px(offset)` | the input at `offset` pixels OF THE INPUT from the current pixel (physical pixels unless the previous pass is scaled) |
//! | `src_texel()`, `out_texel()` | one pixel of the pass's input / of its own output, in `uv` units (both `fl.texel` at full resolution) |
//! | `original(uv)` | the untouched captured backdrop (before any pass), in every pass of the chain |
//! | `p.<name>` | a named parameter: member `<name>` of the snippet's own `struct Params` (see below) |
//! | `params(i)` | anonymous parameter slot `i` (`0..16`) of the [`Pass`], a `vec4<f32>` (the untyped block; prefer named parameters) |
//! | `fl.texel` | size of one physical pixel in `uv` units (of the capture, also in scaled passes) |
//! | `fl.size_px` | captured region size in physical pixels |
//! | `fl.rect_uv` | the layer rect inside the capture (`xy` = min, `zw` = size), in `uv` units |
//! | `fl.time` | seconds (egui input time); reading it makes the effect animated (see below) |
//! | `fl.delta_time` | seconds since the previous frame: egui's `stable_dt` clamped to `0..=0.1` (offline: `OfflineOptions::with_delta_time`, or the video frame step) |
//! | `fl.frame` | frame counter, `u32` (egui's `cumulative_frame_nr`, wrapping; offline: the video frame index) |
//! | `fl.pointer` | the pointer: `xy` in `uv` units (outside `0..1` off the capture), `z` = 1 while the primary button is down, `w` = 1 when a pointer exists (`w = 0`: none, all zero) |
//! | `fl.pixels_per_point` | physical pixels per layer point (egui's scale factor × the layer transform's zoom), to convert point distances to pixels |
//! | `fl.pass_scale` | this pass's output resolution relative to the capture ([`Pass::with_scale`]; `1` = full) |
//! | `to_linear(c)`, `to_gamma(c)` | sRGB transfer functions on `rgb` (alpha untouched) |
//! | `premultiply(c)`, `unpremultiply(c)`, `luma(rgb)` | colour helpers |
//! | `tex(i, uv)` | user texture `i` (`0..`[`TEXTURE_SLOTS`], attached with [`FilterLayer::texture`]) at `uv` in the TEXTURE's own `0..1` coordinates (top-left origin), through its own sampler (its egui `TextureOptions`: filters, wrap mode; mip level 0) |
//! | `tex_size(i)` | size of user texture `i` in texels (`vec2<f32>`) |
//! | `tex_load(i, px)` | the exact texel `px` (`vec2<i32>`, clamped to the texture) of user texture `i`, no filtering |
//! | `state(uv)` | the layer's persistent state at `uv` (bilinear; see [Persistent state](#persistent-state)); zero without a state pass |
//! | `state_load(px)`, `state_px(offset)` | the exact state texel `px` (`vec2<i32>`, (0, 0) top-left, clamped) / the one under the current pixel offset by `offset` texels, no filtering |
//! | `state_size()`, `state_texel()` | the state's size in texels / one state texel in `uv` units |
//! | `fl.pass_index` | index of this pass among the chain's passes of its kind (state or display), `u32`: the substep number of consecutive state passes |
//!
//! An effect whose code reads `fl.time` is **animated**: painting a layer that uses it requests
//! an egui repaint every frame, so it keeps moving in an otherwise idle app. The detection walks
//! naga's IR (a real `fl.time` access, not a text match); override it with
//! [`Effect::with_animated`]. Reading `fl.delta_time`, `fl.frame` or `fl.pointer` does **not**:
//! pointer changes arrive with input, which repaints anyway, and an effect that must advance
//! every frame on its own reads `fl.time` or is marked with [`Effect::with_animated`]. The three
//! are per frame, not per layer: every layer painted in one frame sees the same values.
//!
//! ## Named parameters
//!
//! A snippet may declare `struct Params { ... }`; the crate then binds it as `p` and the effect
//! reads `p.<member>`. Each [`Pass`] stores its own values, set by name and type-checked against
//! the struct ([`Pass::set`], [`Pass::with`], [`Pass::set_slice`] for arrays); a mismatch is a
//! [`ParamError`], never a panic. [`Effect::params`] reflects the members ([`ParamInfo`]: name,
//! [`ParamType`], offset, array length) so a UI can build its controls from the effect alone.
//! Members start at zero (WGSL has no default values).
//!
//! ```
//! use egui_filter_layers::{Effect, ParamError, ParamType, Pass};
//!
//! let effect = Effect::new("tint", "
//! struct Params {
//!     color: vec4<f32>,
//!     amount: f32,
//!     stops: array<vec4<f32>, 4>,
//! }
//! fn effect(uv: vec2<f32>) -> vec4<f32> {
//!     return mix(src(uv), p.color, p.amount) + p.stops[0] * 0.0;
//! }")?;
//! assert_eq!(effect.params().len(), 3);
//! assert_eq!(effect.param("amount").map(|info| info.ty), Some(ParamType::F32));
//!
//! let mut pass = Pass::new(&effect).with("color", [1.0, 0.5, 0.0, 1.0])?.with("amount", 0.25)?;
//! pass.set_slice("stops", &[[0.0_f32, 0.0, 0.0, 1.0]])?; // the remaining elements become zero
//! assert!(matches!(pass.set("amount", 1), Err(ParamError::Type { .. }))); // `1` is an i32
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! Accepted member types: `f32`, `i32`, `u32`, `vec2<f32>`, `vec3<f32>`, `vec4<f32>`,
//! `mat4x4<f32>` and fixed-size arrays of `vec3<f32>` / `vec4<f32>` / `mat4x4<f32>` (uniform
//! arrays need a 16-byte stride). Anything else (`bool`, nested structs, `array<f32, N>`) and a
//! struct over 16 KiB (the WebGL2 uniform block limit) is an [`EffectError::Interface`] pointing
//! at the declaration. The snippet must not declare its own module-scope `p`. Named and
//! anonymous (`params(i)`) parameters can be used together.
//!
//! ## Textures
//!
//! A layer attaches up to [`TEXTURE_SLOTS`] (3, on every backend) egui textures with
//! [`FilterLayer::texture`] — noise, normal maps, lookup tables, images — which its effects read
//! with `tex(i, uv)`, `tex_size(i)` and `tex_load(i, px)`. Values are premultiplied gamma RGBA,
//! like `src()`. Each texture is read through its own sampler only (its `TextureOptions`); exact
//! reads use `tex_load`, which needs none.
//!
//! ```
//! use egui_filter_layers::{Effect, FilterLayer, Pass};
//! # fn frame(ui: &mut egui::Ui, noise: &egui::TextureHandle, rect: egui::Rect) -> Result<(), egui_filter_layers::EffectError> {
//! // `noise` from `ctx.load_texture(.., egui::TextureOptions::LINEAR_REPEAT)`, kept alive.
//! let displace = Effect::new("displace", "
//! fn effect(uv: vec2<f32>) -> vec4<f32> {
//!     let n = tex(0u, uv * 4.0).rg - vec2<f32>(0.5);
//!     return src(uv + n * 8.0 * fl.texel);
//! }")?.with_sample_radius(4.0);
//! assert!(displace.reads_texture(0) && !displace.reads_texture(1));
//! FilterLayer::new(rect, Pass::new(&displace)).texture(0, noise.id()).paint(ui.painter());
//! # Ok(()) }
//! ```
//!
//! Which slots an effect reads is found in naga's IR ([`Effect::reads_texture`]; a slot index
//! that is not a constant counts as every slot). A layer whose effect reads a slot with nothing
//! attached is refused before drawing — [`FilterLayer::check_textures`] returns the
//! [`TextureError`], `paint` draws nothing and logs it once — never silently fed a dummy. A
//! texture id the renderer does not know (freed: keep the `TextureHandle` alive) skips the
//! layers attaching it and is listed by [`failed_textures`]. Offline runs attach an
//! `OfflineTexture` with `OfflineOptions::with_texture` (features `image` / `video`).
//!
//! ## Impulses: transient effects from input
//!
//! [`Impulses`] is a small collection you keep in your app state: push an impulse where the
//! user clicked ([`Impulses::push_point`], positions relative to the layer rect), and every
//! frame hand it to [`presets::click_ripples`] (rings expanding from each click) or to your own
//! effect through [`Impulses::apply`], which fills the named parameters [`IMPULSES_PARAM`]
//! (`impulses: array<vec4<f32>, 16>`, each `(uv.x, uv.y, age_seconds, strength)`) and
//! [`IMPULSE_COUNT_PARAM`] (`impulse_count: u32`). Impulses age against the clock you pass
//! (egui's `input.time` in the UI, frame time offline), expire after their life, and a full
//! collection drops the oldest first ([`IMPULSE_SLOTS`] at most). Such effects are not animated by
//! `fl.time`: call [`Impulses::request_repaint_if_alive`] each frame, so the app repaints while a
//! ring is alive and stays idle afterwards.
//!
//! Colours are **premultiplied-alpha, gamma-space** RGBA in and out. Names starting with `fl_`
//! and the prelude names above are reserved. An effect cannot declare its own bindings or entry
//! points. If it reads further than the current pixel, declare how far with
//! [`Effect::with_sample_radius`] (or per pass with [`Pass::with_sample_radius`]).
//!
//! ## Persistent state
//!
//! A **state pass** ([`Effect::new_state`]) writes the layer's persistent state instead of the
//! next image: two RGBA textures per layer (ping-pong) that survive from frame to frame —
//! simulations (water, [`presets::wave_sim`]), trails ([`presets::trails`]), accumulation.
//!
//! * **Reading.** Every pass reads the latest state with `state(uv)` (bilinear), `state_load(px)` /
//!   `state_px(offset)` (exact texels) — a display pass sees what the last state pass BEFORE it
//!   wrote (or last frame's state), a state pass the state it steps. Inside a state pass `src()` is
//!   the display chain's input at that point (a state pass does not advance it), `original()` the
//!   backdrop. N consecutive state passes are N substeps (`fl.pass_index` = 0..N).
//! * **Identity.** A layer whose chain has a state pass needs [`FilterLayer::id`]: the state is
//!   kept per id and viewport. Without an id it is refused ([`FilterLayer::check_state`]:
//!   [`StateError::MissingId`]; it draws nothing, logged once). Painting one id twice in a frame
//!   steps it once (the second paint shows the state as it is).
//! * **Size.** The state spans the whole capture (the padded rect, `uv` space) at
//!   `ceil(capture * scale)` texels ([`StateOptions::with_scale`]); it lives in layer uv, so a
//!   moved layer keeps it, and a resized layer reads the previous state stretched (no reset).
//! * **Format.** [`StateFormat::Rgba16Float`] by default (signed, unbounded): it must be
//!   renderable and filterable on the device, else stateful layers are refused and listed by
//!   [`failed_states`] (WebGL2 / GLES without `EXT_color_buffer_float`); never silently downgraded.
//!   [`StateFormat::Rgba8Unorm`] is an explicit opt-in for effects that pack their state in 8 bits.
//! * **Lifetime.** A state starts at zero; [`reset_state`] zeroes it at the id's next paint; a state
//!   not painted for 120 frames and 2 seconds is freed ([`state_count`] for debugging); a
//!   re-install (or `destroy_glow`) drops all. It advances only on painted frames: the effect is
//!   not animated by itself — repaint while it should move.
//! * **Offline**, the renderer carries the state from one `process` call to the next
//!   (`OfflineOptions::with_state`, `OfflineRenderer::reset_state`, a video processor's frames);
//!   stateful chains are never tiled (`OfflineError::NotTileable`).
//!
//! ```
//! use egui_filter_layers::{Effect, FilterChain, FilterLayer, Pass};
//! # fn frame(ui: &mut egui::Ui, rect: egui::Rect) -> Result<(), egui_filter_layers::EffectError> {
//! // Accumulates the backdrop's brightness over frames and shows it.
//! let accumulate = Effect::new_state("glow", "
//! fn effect(uv: vec2<f32>) -> vec4<f32> {
//!     return state_px(vec2<i32>(0)) * 0.9 + src(uv) * 0.1;
//! }")?;
//! let show = Effect::new("show", "fn effect(uv: vec2<f32>) -> vec4<f32> { return state(uv); }")?;
//! let chain = FilterChain::new().then(Pass::new(&accumulate)).then(Pass::new(&show));
//! FilterLayer::new(rect, chain).id(ui.id().with("glow")).paint(ui.painter());
//! # Ok(()) }
//! ```
//!
//! **Reduced-resolution passes.** [`Pass::with_scale`] runs a pass over `ceil(capture * scale)`
//! pixels; the next pass (or the final composite, bilinearly) reads that smaller image. `uv`,
//! `src()`, `original()`, `fl.texel` and `fl.size_px` keep their capture meaning, so an
//! effect's distance maths is the same at any scale; kernels that step in pixels use `src_px()`
//! / `src_texel()` (input pixels) or `out_texel()`. [`presets::kawase_blur`] and
//! [`presets::bloom`] are built from such passes. Offline, a scaled chain refuses tiling.
//!
//! **Masks.** A layer filters its whole rect, or a rounded rect ([`FilterLayer::corner_radius`]),
//! or any egui shape: [`FilterLayer::mask`] with [`Mask::Shape`] — paths, circles, rects,
//! meshes, text galleys (UI only), in the layer's points (so it follows layer transforms). The
//! shape's alpha is the coverage (paint it opaque for a full-strength filter; egui's feathered edge
//! anti-aliases it); outside it the backdrop stays untouched. It is tessellated when the layer is
//! painted and rasterized into a coverage texture at capture resolution. egui fills convex paths
//! only: build a concave outline from convex pieces that **overlap** (pieces meeting edge to edge
//! leave a visible seam, where both feathered edges fade out) or from a mesh.
//!
//! ```no_run
//! # fn frame(ui: &mut egui::Ui, rect: egui::Rect) {
//! use egui_filter_layers::{FilterLayer, Mask, presets};
//! let lens = egui::Shape::circle_filled(rect.center(), 60.0, egui::Color32::WHITE);
//! FilterLayer::new(rect, presets::gaussian_blur(8.0)).mask(Mask::Shape(lens)).paint(ui.painter());
//! # }
//! ```
//!
//! **Distances are in layer points.** Sample radii, corner radii and every effect distance
//! converted with `fl.pixels_per_point` are measured in the points of the layer the filter is
//! painted in. Inside a transformed layer (an `egui::Scene`, `Context::set_transform_layer`) they
//! scale with its zoom exactly like the painted content: a 4-point blur covers 8 pixels at zoom 2
//! and 2 pixels at zoom 0.5 (at 1 pixel per point), and the capture padding follows.
//!
//! # Backends
//!
//! | backend | how the backdrop is obtained | status |
//! |---|---|---|
//! | `egui_glow` (GL 3.3+, GLES 3.0+, WebGL2) | copied out of the framebuffer inside the paint callback (MSAA resolved) | feature `glow` |
//! | `egui_glow` on `WebGL1` / GL < 3.3 | — | unsupported: [`InstallError::Unsupported`] |
//! | `egui-wgpu` (native and web) | the content beneath is **rendered a second time** offscreen | feature `wgpu`, `install_wgpu` |
//!
//! **Warning — wgpu draws the content under a filter twice.** egui-wgpu renders into a surface
//! that cannot be read back mid-pass, so the wgpu backend re-renders everything beneath each
//! filter layer (clipped to its padded rect) into an offscreen texture. The content under a
//! filter is therefore **drawn twice** per frame on wgpu. Paint callbacks beneath a filter —
//! including a lower filter layer — cannot be re-rendered and are left out of its backdrop.
// Backend items exist only with their feature: sentences linking to them are feature-gated so
// the docs build with `-D warnings` under any feature set (docs.rs builds with all features).
#![cfg_attr(feature = "wgpu", doc = "See [`install_wgpu`].")]
//!
#![cfg_attr(feature = "glow", doc = "OpenGL objects must be freed explicitly: call [`destroy_glow`] from `eframe::App::on_exit`.")]
#![cfg_attr(not(feature = "glow"), doc = "OpenGL objects must be freed explicitly: call `destroy_glow` (feature `glow`) from `eframe::App::on_exit`.")]
//!
//! The installed backend is stored in the `egui::Context` itself (as a hook-less egui plugin),
//! not in egui `Memory`, so resetting memory does not uninstall it.
//!
//! # Offline processing
//!
//! With the `image` or `video` feature, an `OfflineRenderer` (wgpu only, also for glow apps) runs
//! the same chains over ordinary images and video frames and returns the pixels — identical to a
//! layer over that frame in the UI under the conditions listed in the `offline` module docs.
//! `video` adds a frame-stream processor, source / sink traits and ffmpeg adapters.
#![cfg_attr(any(feature = "image", feature = "video"), doc = "See [`offline`].")]
//!
//! # Hot reload
//!
//! With the `hot-reload` feature (native only; on wasm32 it compiles to nothing), a `HotEffect`
//! follows a WGSL file on disk: a background thread polls its modification time, recompiles it
//! off the GUI thread, and `HotEffect::poll` (once per frame, never blocking) swaps in the new
//! effect. While the file is broken the last good effect keeps rendering and the error, with
//! naga's diagnostic, is available for an overlay. `rebind` rebuilds a `Pass` on the reloaded
//! effect, keeping named parameter values whose name and type still match.

// docs.rs builds with `--cfg docsrs` on nightly: badge feature-gated items with their feature.
#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(clippy::pedantic)]
#![warn(clippy::undocumented_unsafe_blocks)]
#![deny(missing_docs)]
#![allow(clippy::module_name_repetitions)] // `FilterLayer` in `layer`, `EffectError` in `effect` read best as-is.
// Without any backend feature the backend-facing internals (LayerJob fields, registry writers,
// binding constants) have no reader by design; the crate still builds for validation-only use.
#![cfg_attr(not(any(feature = "glow", feature = "wgpu")), allow(dead_code))]

mod backend;
mod effect;
mod geometry;
mod impulses;
mod layer;
mod mask;
mod pass;
mod plan;
pub mod presets;
mod state;

#[cfg(feature = "glow")]
mod glow_backend;
#[cfg(feature = "wgpu")]
mod wgpu_backend;

// Offline processing: the core is shared by the `image` and `video` features; this gate is the
// one place to widen when another feature needs it.
#[cfg(any(feature = "image", feature = "video"))]
pub mod offline;

// Hot reload needs a file system and threads: native only. On wasm32 the feature is accepted
// (so `--all-features` builds everywhere) but the module is absent.
#[cfg(all(feature = "hot-reload", not(target_arch = "wasm32")))]
mod hot;

pub use backend::{BackendKind, BackendStatus, EffectFailure, InstallError, TextureError, TextureFailure, failed_effects, failed_textures, status};
pub use effect::{Effect, EffectError, ParamError, ParamInfo, ParamType, ParamValue, SnippetLocation, TEXTURE_SLOTS};
pub use impulses::{IMPULSE_COUNT_PARAM, IMPULSE_SLOTS, IMPULSES_PARAM, Impulse, ImpulsePack, Impulses};
pub use layer::FilterLayer;
pub use mask::{Mask, MaskError};
pub use pass::{FilterChain, MIN_PASS_SCALE, PARAM_SLOTS, ParamIndexError, Params, Pass};
pub use presets::WaveObstacle;
pub use state::{StateError, StateFailure, StateFormat, StateOptions, failed_states, reset_state, state_count};

#[cfg(feature = "glow")]
pub use glow_backend::{destroy_glow, install_glow};
#[cfg(feature = "wgpu")]
pub use wgpu_backend::{WgpuOptions, install_wgpu};
#[cfg(any(feature = "image", feature = "video"))]
pub use offline::{AlphaMode, FrameRef, OfflineError, OfflineOptions, OfflineRenderer, OfflineTexture, PointerState, RgbaFrame};
#[cfg(all(feature = "hot-reload", not(target_arch = "wasm32")))]
pub use hot::{DEFAULT_POLL_INTERVAL, HotEffect, HotOptions, HotReloadError, HotReloadEvent, MIN_POLL_INTERVAL, Rebound, rebind};
