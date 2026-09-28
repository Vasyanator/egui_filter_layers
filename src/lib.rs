/*
File: lib.rs

Purpose:
Crate root of `egui_filter_layers`: public re-exports, backend installers, and the crate-level
documentation (semantics, shader model, backend matrix).

Key items:
- `FilterLayer`, `FilterChain`, `Pass`, `Effect` (re-exported from their modules)
- `presets` module
- `install_glow()` / `destroy_glow()` (feature `glow`), `install_wgpu()` / `WgpuOptions`
  (feature `wgpu`), `status()`, `failed_effects()`
- `offline` module and its re-exports (`OfflineRenderer`, ...; features `image` / `video`)

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
//! | `src_px(offset)` | the input at `offset` physical pixels from the current pixel |
//! | `original(uv)` | the untouched captured backdrop (before any pass), in every pass of the chain |
//! | `params(i)` | parameter slot `i` (`0..16`) of the [`Pass`], a `vec4<f32>` |
//! | `fl.texel` | size of one physical pixel in `uv` units |
//! | `fl.size_px` | captured region size in physical pixels |
//! | `fl.rect_uv` | the layer rect inside the capture (`xy` = min, `zw` = size), in `uv` units |
//! | `fl.time` | seconds (egui input time); reading it makes the effect animated (see below) |
//! | `fl.delta_time` | seconds since the previous frame: egui's `stable_dt` clamped to `0..=0.1` (offline: `OfflineOptions::with_delta_time`, or the video frame step) |
//! | `fl.frame` | frame counter, `u32` (egui's `cumulative_frame_nr`, wrapping; offline: the video frame index) |
//! | `fl.pointer` | the pointer: `xy` in `uv` units (outside `0..1` off the capture), `z` = 1 while the primary button is down, `w` = 1 when a pointer exists (`w = 0`: none, all zero) |
//! | `fl.pixels_per_point` | physical pixels per layer point (egui's scale factor × the layer transform's zoom), to convert point distances to pixels |
//! | `to_linear(c)`, `to_gamma(c)` | sRGB transfer functions on `rgb` (alpha untouched) |
//! | `premultiply(c)`, `unpremultiply(c)`, `luma(rgb)` | colour helpers |
//!
//! An effect whose code reads `fl.time` is **animated**: painting a layer that uses it requests
//! an egui repaint every frame, so it keeps moving in an otherwise idle app. The detection walks
//! naga's IR (a real `fl.time` access, not a text match); override it with
//! [`Effect::with_animated`]. Reading `fl.delta_time`, `fl.frame` or `fl.pointer` does **not**:
//! pointer changes arrive with input, which repaints anyway, and an effect that must advance
//! every frame on its own reads `fl.time` or is marked with [`Effect::with_animated`]. The three
//! are per frame, not per layer: every layer painted in one frame sees the same values.
//!
//! Colours are **premultiplied-alpha, gamma-space** RGBA in and out. Names starting with `fl_`
//! and the prelude names above are reserved. An effect cannot declare its own bindings or entry
//! points. If it reads further than the current pixel, declare how far with
//! [`Effect::with_sample_radius`] (or per pass with [`Pass::with_sample_radius`]).
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
mod layer;
mod pass;
mod plan;
pub mod presets;

#[cfg(feature = "glow")]
mod glow_backend;
#[cfg(feature = "wgpu")]
mod wgpu_backend;

// Offline processing: the core is shared by the `image` and `video` features; this gate is the
// one place to widen when another feature needs it.
#[cfg(any(feature = "image", feature = "video"))]
pub mod offline;

pub use backend::{BackendKind, BackendStatus, EffectFailure, InstallError, failed_effects, status};
pub use effect::{Effect, EffectError, SnippetLocation};
pub use layer::FilterLayer;
pub use pass::{FilterChain, PARAM_SLOTS, ParamIndexError, Params, Pass};

#[cfg(feature = "glow")]
pub use glow_backend::{destroy_glow, install_glow};
#[cfg(feature = "wgpu")]
pub use wgpu_backend::{WgpuOptions, install_wgpu};
#[cfg(any(feature = "image", feature = "video"))]
pub use offline::{AlphaMode, FrameRef, OfflineError, OfflineOptions, OfflineRenderer, PointerState, RgbaFrame};
