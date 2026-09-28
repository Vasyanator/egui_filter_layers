//! Headless end-to-end check of the glow backend on a real OpenGL context without a window: an
//! EGL context on the device platform (`EGL_EXT_platform_device`, made current surfaceless)
//! paints egui frames through the REAL public path — `install_glow`, `egui::Context::run_ui` with
//! shapes and `FilterLayer`s, `egui_glow::Painter::paint_and_update_textures` — into an offscreen
//! framebuffer object of known size, which is read back with `glReadPixels` and checked pixel by
//! pixel.
//!
//! Scenes: invert over known colours (solids, a user texture, a mid-tone, orientation, a shape
//! painted after the filter, glow's documented nesting where an upper layer sees the lower
//! filter's output); an invert layer with rounded corners over a lower filter; multi-pass chains
//! (ping-pong beyond two targets, separable blur); effect distances inside a zoomed layer; the
//! composite's alpha over a transparent frame; the same scenes into a 4x multisampled
//! renderbuffer (the backend resolves before capturing); and an effect whose GLSL translation
//! fails, which must disable only that effect while the backend stays `Active`. Reduced-resolution
//! passes (F6): the same scaled-identity, kawase and bloom scenes as `wgpu_headless.rs`, and like
//! it: `original()` in later passes (F1), `fl.pointer` / `fl.delta_time` / `fl.frame` (F2), and
//! user textures (F8: exact / nearest / linear reads, full and partial updates, three slots,
//! unknown / freed / missing textures), and persistent state (F7: accumulation over frames in
//! RGBA16F and RGBA8, reset, one step per frame for an id painted twice, growth keeping the state,
//! a missing id refused).
//!
//! Linux only (EGL device platform, Mesa or a vendor EGL). Needs a GPU or Mesa's software
//! device, so it is ignored by default: `cargo test --all-features --test glow_headless -- --ignored`.
//! `cargo run --example demo` is not a test: it needs a display.

#![cfg(all(feature = "glow", target_os = "linux"))]

use std::sync::Arc;

use egui::emath::TSTransform;
use egui::{Color32, ColorImage, Pos2, Rect, TextureOptions, pos2, vec2};
use egui_filter_layers::{BackendKind, BackendStatus, Effect, FilterChain, FilterLayer, Pass, StateFormat, StateOptions, destroy_glow, install_glow, presets};
use glow::HasContext as _;
use glutin::api::egl::context::PossiblyCurrentContext;
use glutin::api::egl::device::Device;
use glutin::api::egl::display::Display;
use glutin::config::{ConfigSurfaceTypes, ConfigTemplateBuilder};
use glutin::context::{ContextApi, ContextAttributesBuilder, GlProfile, Version};
use glutin::display::GlDisplay as _;

const WIDTH: u32 = 256;
const HEIGHT: u32 = 128;
/// `WIDTH` / `HEIGHT` as GL sizes (both are small constants, so the conversion is exact).
const WIDTH_I32: i32 = 256;
const HEIGHT_I32: i32 = 128;

/// A current, windowless OpenGL context. The context must outlive every GL object of the test,
/// so it is dropped last (field order).
struct GlContext {
    /// Shared with `egui_glow::Painter` and the filter backend.
    gl: Arc<glow::Context>,
    /// Kept alive (and current on this thread) for the whole test.
    _context: PossiblyCurrentContext,
    /// Kept alive with the context; glutin reference-counts the display.
    _display: Display,
}

/// Creates a GL 3.3 core (or, failing that, GLES 3.0) context on the first EGL device that
/// accepts one, made current without a surface. Panics with every attempt's reason when none does.
fn gl_context() -> GlContext {
    let devices = match Device::query_devices() {
        Ok(devices) => devices.collect::<Vec<_>>(),
        Err(error) => panic!("no EGL device enumeration (needs libEGL with EGL_EXT_device_enumeration, e.g. Mesa): {error}"),
    };
    let mut attempts = Vec::new();
    for device in &devices {
        let label = format!("{:?} / {:?}", device.vendor(), device.name());
        // SAFETY: `None` is the documented "no DRM handle" argument; nothing else is required.
        let display = match unsafe { Display::with_device(device, None) } {
            Ok(display) => display,
            Err(error) => {
                attempts.push(format!("{label}: display: {error}"));
                continue;
            }
        };
        // Empty surface type: any config, the context is used surfaceless.
        let template = ConfigTemplateBuilder::new().with_surface_type(ConfigSurfaceTypes::empty()).with_alpha_size(8).build();
        // SAFETY: the template holds no native window handle.
        let config = match unsafe { display.find_configs(template) } {
            Ok(mut configs) => configs.next(),
            Err(error) => {
                attempts.push(format!("{label}: configs: {error}"));
                continue;
            }
        };
        let Some(config) = config else {
            attempts.push(format!("{label}: no EGL config"));
            continue;
        };
        let apis = [(ContextApi::OpenGl(Some(Version::new(3, 3))), GlProfile::Core), (ContextApi::Gles(Some(Version::new(3, 0))), GlProfile::Core)];
        for (api, profile) in apis {
            let attributes = ContextAttributesBuilder::new().with_context_api(api).with_profile(profile).build(None);
            // SAFETY: `config` belongs to `display`; no window handle is involved.
            let context = match unsafe { display.create_context(&config, &attributes) } {
                Ok(context) => context,
                Err(error) => {
                    attempts.push(format!("{label}: {api:?}: {error}"));
                    continue;
                }
            };
            let context = match context.make_current_surfaceless() {
                Ok(context) => context,
                Err(error) => {
                    attempts.push(format!("{label}: {api:?}: make current surfaceless: {error}"));
                    continue;
                }
            };
            // SAFETY: the context is current on this thread and the loader is its display's.
            let gl = unsafe { glow::Context::from_loader_function_cstr(|name| display.get_proc_address(name)) };
            // SAFETY: the context is current; string queries only.
            let (version, renderer) = unsafe { (gl.get_parameter_string(glow::VERSION), gl.get_parameter_string(glow::RENDERER)) };
            eprintln!("EGL device {label}: {api:?} -> {version} on {renderer}");
            return GlContext { gl: Arc::new(gl), _context: context, _display: display };
        }
    }
    panic!("could not create a windowless GL 3.3 / GLES 3.0 context on any of {} EGL device(s):\n{}", devices.len(), attempts.join("\n"));
}

/// The framebuffer egui paints into: an RGBA8 renderbuffer, multisampled when `samples > 1`,
/// plus a single-sample resolve framebuffer that is read back in that case.
struct Target {
    /// Bound as `GL_FRAMEBUFFER` while egui paints (egui_glow paints into the bound framebuffer).
    draw: glow::Framebuffer,
    /// Its colour attachment.
    draw_color: glow::Renderbuffer,
    /// Resolve framebuffer and its colour attachment, when multisampled.
    resolve: Option<(glow::Framebuffer, glow::Renderbuffer)>,
}

/// A framebuffer with one RGBA8 renderbuffer of `WIDTH` x `HEIGHT` and `samples` samples (0 = none).
///
/// # Safety
/// `gl` must be current on this thread.
unsafe fn color_framebuffer(gl: &glow::Context, samples: i32) -> (glow::Framebuffer, glow::Renderbuffer) {
    // SAFETY: forwarded contract; the objects are created and bound on this context only.
    unsafe {
        let (Ok(framebuffer), Ok(renderbuffer)) = (gl.create_framebuffer(), gl.create_renderbuffer()) else {
            panic!("could not create a framebuffer / renderbuffer");
        };
        gl.bind_renderbuffer(glow::RENDERBUFFER, Some(renderbuffer));
        if samples > 0 {
            gl.renderbuffer_storage_multisample(glow::RENDERBUFFER, samples, glow::RGBA8, WIDTH_I32, HEIGHT_I32);
        } else {
            gl.renderbuffer_storage(glow::RENDERBUFFER, glow::RGBA8, WIDTH_I32, HEIGHT_I32);
        }
        gl.bind_renderbuffer(glow::RENDERBUFFER, None);
        gl.bind_framebuffer(glow::FRAMEBUFFER, Some(framebuffer));
        gl.framebuffer_renderbuffer(glow::FRAMEBUFFER, glow::COLOR_ATTACHMENT0, glow::RENDERBUFFER, Some(renderbuffer));
        let status = gl.check_framebuffer_status(glow::FRAMEBUFFER);
        assert_eq!(status, glow::FRAMEBUFFER_COMPLETE, "test framebuffer ({samples} samples) incomplete: 0x{status:04x}");
        gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        (framebuffer, renderbuffer)
    }
}

/// Everything one test renders with.
struct Harness {
    /// An egui context with the glow backend installed.
    ctx: egui::Context,
    /// egui's own glow painter, the one the backend's callbacks run inside.
    painter: egui_glow::Painter,
    /// Where frames are painted.
    target: Target,
    /// The colour each frame is cleared to (premultiplied RGBA, like `eframe::App::clear_color`).
    clear: [f32; 4],
    /// A 4x4 green user texture.
    texture: egui::TextureHandle,
    /// Dropped last.
    context: GlContext,
}

impl Harness {
    /// A context, an egui_glow painter (no dithering, so colours are exact), the backend installed
    /// and an RGBA8 target with `samples` samples (0 = single-sample) cleared to `clear`.
    fn new(samples: i32, clear: [f32; 4]) -> Self {
        let context = gl_context();
        let painter = match egui_glow::Painter::new(Arc::clone(&context.gl), "", None, false) {
            Ok(painter) => painter,
            Err(error) => panic!("egui_glow painter: {error}"),
        };
        let ctx = egui::Context::default();
        let installed = install_glow(&ctx, &context.gl);
        assert!(installed.is_ok(), "{installed:?}");
        assert!(matches!(egui_filter_layers::status(&ctx), BackendStatus::Installed { backend: BackendKind::Glow }), "{}", egui_filter_layers::status(&ctx));
        // SAFETY: the context is current on this thread.
        let (draw, draw_color) = unsafe { color_framebuffer(&context.gl, samples) };
        // SAFETY: as above.
        let resolve = (samples > 0).then(|| unsafe { color_framebuffer(&context.gl, 0) });
        if samples > 0 {
            // SAFETY: as above; queries the draw framebuffer's actual sample count.
            let actual = unsafe {
                context.gl.bind_framebuffer(glow::FRAMEBUFFER, Some(draw));
                let actual = context.gl.get_parameter_i32(glow::SAMPLES);
                context.gl.bind_framebuffer(glow::FRAMEBUFFER, None);
                actual
            };
            eprintln!("multisampled target: {actual} samples");
            assert!(actual > 1, "the driver gave the multisampled renderbuffer {actual} samples");
        }
        let texture = ctx.load_texture("green", ColorImage::new([4, 4], vec![Color32::GREEN; 16]), TextureOptions::NEAREST);
        Self { ctx, painter, target: Target { draw, draw_color, resolve }, clear, texture, context }
    }

    /// Renders one egui frame through egui_glow into the target and returns its RGBA8 pixels, top
    /// row first. Asserts that the frame left no GL error behind.
    fn frame(&mut self, paint: &mut dyn FnMut(&mut egui::Ui)) -> Vec<u8> {
        self.frame_with(Vec::new(), paint)
    }

    /// [`Harness::frame`] with input `events` (the pointer persists into later frames).
    fn frame_with(&mut self, events: Vec<egui::Event>, paint: &mut dyn FnMut(&mut egui::Ui)) -> Vec<u8> {
        let input = egui::RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(256.0, 128.0))), events, ..Default::default() };
        let mut output = self.ctx.run_ui(input, |ui| paint(ui));
        let mut textures_delta = std::mem::take(&mut output.textures_delta);
        let primitives = self.ctx.tessellate(output.shapes, output.pixels_per_point);
        let gl = Arc::clone(&self.context.gl);
        // SAFETY: the context is current on this thread; the framebuffer is this harness's own.
        unsafe { gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.target.draw)) };
        self.painter.clear([WIDTH, HEIGHT], self.clear);
        self.painter.paint_and_update_textures([WIDTH, HEIGHT], output.pixels_per_point, &primitives, &mut textures_delta);
        let mut pixels = vec![0_u8; usize::try_from(WIDTH * HEIGHT * 4).unwrap_or(0)];
        // SAFETY: the context is current; all objects are this harness's own; `pixels` holds
        // exactly WIDTH x HEIGHT RGBA8 texels (pack alignment 4 divides the 1024-byte rows).
        let error = unsafe {
            // The callback must have handed egui's framebuffer back.
            let bound = gl.get_parameter_framebuffer(glow::DRAW_FRAMEBUFFER_BINDING);
            assert_eq!(bound, Some(self.target.draw), "the filter callback did not restore the draw framebuffer");
            if let Some((resolve, _)) = self.target.resolve {
                gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(self.target.draw));
                gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(resolve));
                gl.blit_framebuffer(0, 0, WIDTH_I32, HEIGHT_I32, 0, 0, WIDTH_I32, HEIGHT_I32, glow::COLOR_BUFFER_BIT, glow::NEAREST);
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(resolve));
            }
            gl.read_pixels(0, 0, WIDTH_I32, HEIGHT_I32, glow::RGBA, glow::UNSIGNED_BYTE, glow::PixelPackData::Slice(Some(&mut pixels)));
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            gl.get_error()
        };
        assert_eq!(error, glow::NO_ERROR, "GL error 0x{error:04x} after the frame");
        // GL rows are bottom-up; the assertions below use egui's top-down pixel coordinates.
        pixels.chunks_exact(usize::try_from(WIDTH * 4).unwrap_or(0)).rev().flatten().copied().collect()
    }
}

impl Drop for Harness {
    fn drop(&mut self) {
        let gl = &self.context.gl;
        destroy_glow(&self.ctx, gl);
        self.painter.destroy();
        // SAFETY: the context is still current (it is dropped after this); the objects are ours.
        unsafe {
            gl.delete_framebuffer(self.target.draw);
            gl.delete_renderbuffer(self.target.draw_color);
            if let Some((framebuffer, renderbuffer)) = self.target.resolve {
                gl.delete_framebuffer(framebuffer);
                gl.delete_renderbuffer(renderbuffer);
            }
        }
    }
}

/// The RGBA of pixel `(x, y)` (top-left origin).
fn rgba(pixels: &[u8], x: u32, y: u32) -> [u8; 4] {
    let i = usize::try_from((y * WIDTH + x) * 4).unwrap_or(0);
    [pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]]
}

/// The RGB of pixel `(x, y)` (top-left origin).
fn rgb(pixels: &[u8], x: u32, y: u32) -> [u8; 3] {
    let [r, g, b, _] = rgba(pixels, x, y);
    [r, g, b]
}

fn assert_near(pixels: &[u8], x: u32, y: u32, expected: [u8; 3], what: &str) {
    let actual = rgb(pixels, x, y);
    let close = actual.iter().zip(expected).all(|(a, e)| a.abs_diff(e) <= 3);
    assert!(close, "{what}: pixel ({x}, {y}) is {actual:?}, expected {expected:?}");
}

fn assert_active(ctx: &egui::Context) {
    let status = egui_filter_layers::status(ctx);
    assert!(matches!(status, BackendStatus::Active { backend: BackendKind::Glow, .. }), "{status}");
}

/// A pass of an effect from `snippet`, panicking on a WGSL error (the snippets here are fixed).
fn pass(label: &str, snippet: &str) -> Pass {
    match Effect::new(label, snippet) {
        Ok(effect) => Pass::new(&effect),
        Err(error) => panic!("{error}"),
    }
}

/// An effect reading `params(0).x` LAYER points to the right; `points` is also its sample radius.
fn shift(points: f32) -> Pass {
    let shifted = pass("shift", "fn effect(uv: vec2<f32>) -> vec4<f32> { return src_px(vec2<f32>(params(0u).x * fl.pixels_per_point, 0.0)); }").with_sample_radius(points);
    let Ok(shifted) = shifted.with_param(0, [points, 0.0, 0.0, 0.0]) else { panic!("slot 0 exists") };
    shifted
}

/// Red | blue halves split at x = 128.
fn halves(painter: &egui::Painter) {
    painter.rect_filled(Rect::from_min_max(pos2(0.0, 0.0), pos2(128.0, 128.0)), 0.0, Color32::RED);
    painter.rect_filled(Rect::from_min_max(pos2(128.0, 0.0), pos2(256.0, 128.0)), 0.0, Color32::BLUE);
}

/// The wgpu test's backdrop scene: solids, a mid-tone, a user texture under an invert layer, a
/// shape painted after it, and a second, overlapping invert layer.
fn inverted_scene(ui: &egui::Ui, texture: &egui::TextureHandle) {
    let painter = ui.painter();
    halves(painter);
    painter.rect_filled(Rect::from_min_max(pos2(60.0, 60.0), pos2(80.0, 80.0)), 0.0, Color32::from_rgb(128, 64, 32));
    painter.image(texture.id(), Rect::from_min_max(pos2(8.0, 8.0), pos2(40.0, 40.0)), Rect::from_min_max(Pos2::ZERO, pos2(1.0, 1.0)), Color32::WHITE);
    FilterLayer::new(Rect::from_min_max(pos2(4.0, 4.0), pos2(200.0, 100.0)), presets::invert(1.0)).paint(painter);
    // Painted after the filter: must stay untouched.
    painter.rect_filled(Rect::from_min_max(pos2(160.0, 60.0), pos2(180.0, 80.0)), 0.0, Color32::WHITE);
    // Overlapping the first layer: on glow it captures the framebuffer, so it DOES see the first
    // filter's output (unlike wgpu, which re-renders the unfiltered backdrop).
    FilterLayer::new(Rect::from_min_max(pos2(150.0, 90.0), pos2(250.0, 125.0)), presets::invert(1.0)).paint(painter);
}

/// Checks a frame of [`inverted_scene`].
fn check_inverted_scene(pixels: &[u8], what: &str) {
    assert_near(pixels, 100, 50, [0, 255, 255], &format!("{what}: red under the filter"));
    assert_near(pixels, 150, 50, [255, 255, 0], &format!("{what}: blue under the filter"));
    assert_near(pixels, 20, 20, [255, 0, 255], &format!("{what}: user texture under the filter"));
    assert_near(pixels, 70, 70, [127, 191, 223], &format!("{what}: mid-tone under the filter (gamma handling)"));
    assert_near(pixels, 5, 5, [0, 255, 255], &format!("{what}: top-left corner inside the filter (orientation)"));
    assert_near(pixels, 2, 2, [255, 0, 0], &format!("{what}: outside the filter, left"));
    assert_near(pixels, 220, 50, [0, 0, 255], &format!("{what}: outside the filter, right"));
    assert_near(pixels, 100, 110, [255, 0, 0], &format!("{what}: outside the filter, below"));
    assert_near(pixels, 170, 70, [255, 255, 255], &format!("{what}: painted after the filter"));
    assert_near(pixels, 220, 110, [255, 255, 0], &format!("{what}: second layer over blue"));
    assert_near(pixels, 170, 95, [0, 0, 255], &format!("{what}: second layer over the first: inverted twice"));
}

#[test]
#[ignore = "needs an EGL device (GPU or Mesa); run with --ignored"]
fn a_known_backdrop_is_inverted_under_the_layer_only() {
    let mut harness = Harness::new(0, [0.0, 0.0, 0.0, 1.0]);
    let texture = harness.texture.clone();
    // Frame 1 creates the backend's resources and uploads the user texture before painting.
    let first = harness.frame(&mut |ui| inverted_scene(ui, &texture));
    check_inverted_scene(&first, "frame 1");
    assert_active(&harness.ctx);
    let second = harness.frame(&mut |ui| inverted_scene(ui, &texture));
    check_inverted_scene(&second, "frame 2");
    assert!(egui_filter_layers::failed_effects(&harness.ctx).is_empty(), "{:?}", egui_filter_layers::failed_effects(&harness.ctx));
}

/// An identity effect leaves the backdrop bit-exact (capture, pass and composite are lossless
/// in 8 bits).
#[test]
#[ignore = "needs an EGL device (GPU or Mesa); run with --ignored"]
fn an_identity_layer_is_lossless() {
    let mut harness = Harness::new(0, [0.0, 0.0, 0.0, 1.0]);
    let identity = pass("identity", "fn effect(uv: vec2<f32>) -> vec4<f32> { return src(uv); }");
    let mut paint = |ui: &mut egui::Ui| {
        let painter = ui.painter();
        halves(painter);
        painter.rect_filled(Rect::from_min_max(pos2(60.0, 60.0), pos2(80.0, 80.0)), 0.0, Color32::from_rgb(128, 64, 32));
        FilterLayer::new(Rect::from_min_max(pos2(4.0, 4.0), pos2(200.0, 100.0)), identity.clone()).paint(painter);
    };
    harness.frame(&mut paint);
    let pixels = harness.frame(&mut paint);
    assert_eq!(rgb(&pixels, 100, 50), [255, 0, 0], "red under the identity filter");
    assert_eq!(rgb(&pixels, 150, 50), [0, 0, 255], "blue under the identity filter");
    assert_eq!(rgb(&pixels, 70, 70), [128, 64, 32], "mid-tone under the identity filter");
    assert_active(&harness.ctx);
}

/// An upper invert layer with rounded corners over a lower invert layer: outside its mask (the
/// corners) the screen keeps what is there, inside it the backdrop — which on glow includes the
/// lower filter — is inverted.
fn run_rounded_corner(harness: &mut Harness) {
    let mut paint = |ui: &mut egui::Ui| {
        let painter = ui.painter();
        halves(painter);
        FilterLayer::new(Rect::from_min_max(pos2(4.0, 4.0), pos2(200.0, 100.0)), presets::invert(1.0)).paint(painter);
        FilterLayer::new(Rect::from_min_max(pos2(150.0, 60.0), pos2(250.0, 125.0)), presets::invert(1.0)).corner_radius(16.0).paint(painter);
    };
    harness.frame(&mut paint);
    let pixels = harness.frame(&mut paint);
    assert_near(&pixels, 151, 61, [255, 255, 0], "upper layer's cut-off corner over the lower filter keeps the lower filter's output");
    assert_near(&pixels, 248, 123, [0, 0, 255], "upper layer's cut-off corner outside the lower filter");
    assert_near(&pixels, 248, 62, [0, 0, 255], "upper layer's top-right cut-off corner");
    assert_near(&pixels, 170, 80, [0, 0, 255], "inside both: inverted twice");
    assert_near(&pixels, 220, 110, [255, 255, 0], "inside the upper layer's mask only");
    assert_near(&pixels, 100, 50, [0, 255, 255], "lower filter away from the upper layer");
    assert_active(&harness.ctx);
}

#[test]
#[ignore = "needs an EGL device (GPU or Mesa); run with --ignored"]
fn a_rounded_corner_keeps_the_screen() {
    run_rounded_corner(&mut Harness::new(0, [0.0, 0.0, 0.0, 1.0]));
}

/// Chains of several passes: three passes ping-pong through the intermediates (A -> B -> A), each
/// reading its predecessor's output, and a separable blur's two passes.
#[test]
#[ignore = "needs an EGL device (GPU or Mesa); run with --ignored"]
fn multi_pass_chains_run_every_pass_in_order() {
    let mut harness = Harness::new(0, [0.0, 0.0, 0.0, 1.0]);
    let chain = FilterChain::new().then(shift(8.0)).then(shift(8.0)).then(presets::invert(1.0));
    assert_eq!(chain.len(), 3);
    let blur = presets::gaussian_blur(4.0);
    assert!(blur.len() >= 2, "the blur preset is expected to be multi-pass");
    let mut paint = |ui: &mut egui::Ui| {
        let painter = ui.painter();
        halves(painter);
        FilterLayer::new(Rect::from_min_max(pos2(64.0, 8.0), pos2(192.0, 56.0)), chain.clone()).paint(painter);
        FilterLayer::new(Rect::from_min_max(pos2(64.0, 72.0), pos2(192.0, 120.0)), blur.clone()).paint(painter);
    };
    harness.frame(&mut paint);
    let pixels = harness.frame(&mut paint);
    // Two 8 px shifts, then invert: the red | blue edge (x = 128) appears at x = 112, inverted.
    assert_near(&pixels, 108, 32, [0, 255, 255], "shifted twice then inverted, left of the moved edge");
    assert_near(&pixels, 116, 32, [255, 255, 0], "shifted twice then inverted, right of the moved edge");
    assert_near(&pixels, 180, 32, [255, 255, 0], "right end: reads the padded capture beyond the layer");
    assert_near(&pixels, 60, 32, [255, 0, 0], "outside the chain layer");
    // Blur: flat areas unchanged, the edge mixed.
    assert_near(&pixels, 80, 96, [255, 0, 0], "blur over flat red");
    assert_near(&pixels, 176, 96, [0, 0, 255], "blur over flat blue");
    let [r, g, b] = rgb(&pixels, 128, 96);
    assert!(r > 40 && b > 40 && g < 8, "blur across the edge must mix red and blue, got {:?}", [r, g, b]);
    assert_active(&harness.ctx);
}

/// A filter inside a zoomed layer (`Context::set_transform_layer`): effect distances in points
/// scale with the zoom. The shift reads 8 LAYER points right, so the edge beneath (x = 128) shows
/// up `8 * zoom` pixels to the left.
#[test]
#[ignore = "needs an EGL device (GPU or Mesa); run with --ignored"]
fn effect_distances_follow_the_layer_zoom() {
    let mut harness = Harness::new(0, [0.0, 0.0, 0.0, 1.0]);
    let shifted = shift(8.0);
    let layer = egui::LayerId::new(egui::Order::Foreground, egui::Id::new("zoomed filter layer"));
    for zoom in [2.0_f32, 0.5] {
        let mut paint = |ui: &mut egui::Ui| {
            ui.ctx().set_transform_layer(layer, TSTransform::from_scaling(zoom));
            let painter = egui::Painter::new(ui.ctx().clone(), layer, Rect::EVERYTHING);
            let at = |x: f32, y: f32| pos2(x / zoom, y / zoom);
            painter.rect_filled(Rect::from_min_max(at(0.0, 0.0), at(128.0, 128.0)), 0.0, Color32::RED);
            painter.rect_filled(Rect::from_min_max(at(128.0, 0.0), at(256.0, 128.0)), 0.0, Color32::BLUE);
            FilterLayer::new(Rect::from_min_max(at(16.0, 16.0), at(240.0, 112.0)), shifted.clone()).paint(&painter);
        };
        harness.frame(&mut paint);
        let pixels = harness.frame(&mut paint);
        let edge = if zoom > 1.0 { 112 } else { 124 };
        assert_near(&pixels, edge - 2, 64, [255, 0, 0], &format!("zoom {zoom}: left of the shifted edge"));
        assert_near(&pixels, edge + 2, 64, [0, 0, 255], &format!("zoom {zoom}: right of the shifted edge"));
        assert_near(&pixels, 8, 64, [255, 0, 0], &format!("zoom {zoom}: outside the filter"));
    }
}

/// The composite's alpha over a TRANSPARENT frame: a half-transparent white backdrop (alpha 128)
/// under an effect halving the premultiplied colour. Inside the mask alpha is 64, outside 128, on
/// the partly covered left edge (x = 100.25 -> coverage 0.75 of pixel 100) the gamma mix of both
/// (glow's composite mixes with the exact capture of the 8-bit framebuffer).
#[test]
#[ignore = "needs an EGL device (GPU or Mesa); run with --ignored"]
fn the_composite_mixes_alpha_like_colour() {
    let mut harness = Harness::new(0, [0.0; 4]);
    let half = pass("half", "fn effect(uv: vec2<f32>) -> vec4<f32> { return 0.5 * src(uv); }");
    let mut paint = |ui: &mut egui::Ui| {
        let painter = ui.painter();
        painter.rect_filled(Rect::from_min_max(pos2(0.0, 0.0), pos2(256.0, 128.0)), 0.0, Color32::from_rgba_unmultiplied(255, 255, 255, 128));
        FilterLayer::new(Rect::from_min_max(pos2(100.25, 20.0), pos2(200.0, 100.0)), half.clone()).paint(painter);
    };
    harness.frame(&mut paint);
    let pixels = harness.frame(&mut paint);
    let (outside, inside, edge) = (rgba(&pixels, 50, 50), rgba(&pixels, 150, 50), rgba(&pixels, 100, 50));
    eprintln!("outside {outside:?}, inside {inside:?}, edge {edge:?}");
    assert!(outside[3].abs_diff(128) <= 1, "alpha outside the layer {outside:?}");
    assert!(inside[3].abs_diff(64) <= 1, "alpha inside the layer {inside:?}");
    assert!(edge[3].abs_diff(80) <= 1, "alpha on the partly covered edge {edge:?}");
    let expected = 0.25 * f32::from(outside[0]) + 0.75 * f32::from(inside[0]);
    assert!((f32::from(edge[0]) - expected).abs() <= 1.5, "edge colour {edge:?}, expected about {expected}");
}

/// The same scenes painted into a 4x multisampled renderbuffer: the backend must resolve it
/// (same-format renderbuffer, identical rects) before capturing, and composite back into it.
#[test]
#[ignore = "needs an EGL device (GPU or Mesa); run with --ignored"]
fn a_multisampled_framebuffer_is_resolved_and_filtered() {
    let mut harness = Harness::new(4, [0.0, 0.0, 0.0, 1.0]);
    let texture = harness.texture.clone();
    harness.frame(&mut |ui| inverted_scene(ui, &texture));
    let pixels = harness.frame(&mut |ui| inverted_scene(ui, &texture));
    check_inverted_scene(&pixels, "MSAA");
    run_rounded_corner(&mut harness);
}

/// An effect naga validates but cannot translate to GLSL (one texture read through two
/// samplers) fails on glow: only that effect is disabled — its layer draws nothing extra, other
/// layers still filter, the backend stays `Active` — and the failure is reported, not retried.
#[test]
#[ignore = "needs an EGL device (GPU or Mesa); run with --ignored"]
fn a_failing_effect_is_disabled_alone() {
    let mut harness = Harness::new(0, [0.0, 0.0, 0.0, 1.0]);
    // Pinned as untranslatable by `glow_backend::glsl`'s `one_texture_through_two_samplers_does_not_translate`.
    // Reads the pass input through user texture slot 0's sampler, so the layers attach slot 0.
    let broken = pass("broken on glow", "fn effect(uv: vec2<f32>) -> vec4<f32> { return src(uv) + textureSampleLevel(fl_src_texture, fl_user0_sampler, uv, 0.0); }");
    let texture = harness.texture.id();
    let mut paint = |ui: &mut egui::Ui| {
        let painter = ui.painter();
        halves(painter);
        FilterLayer::new(Rect::from_min_max(pos2(8.0, 8.0), pos2(120.0, 120.0)), broken.clone()).texture(0, texture).paint(painter);
        // A chain containing the broken effect is skipped as a whole.
        FilterLayer::new(Rect::from_min_max(pos2(136.0, 8.0), pos2(192.0, 120.0)), FilterChain::new().then(presets::invert(1.0)).then(broken.clone())).texture(0, texture).paint(painter);
        FilterLayer::new(Rect::from_min_max(pos2(200.0, 8.0), pos2(248.0, 120.0)), presets::invert(1.0)).paint(painter);
    };
    for frame in 1..=3 {
        let pixels = harness.frame(&mut paint);
        assert_near(&pixels, 60, 60, [255, 0, 0], &format!("frame {frame}: broken effect's layer leaves the backdrop"));
        assert_near(&pixels, 160, 60, [0, 0, 255], &format!("frame {frame}: chain with the broken effect leaves the backdrop"));
        assert_near(&pixels, 220, 60, [255, 255, 0], &format!("frame {frame}: a healthy layer still filters"));
        assert_active(&harness.ctx);
        let failed = egui_filter_layers::failed_effects(&harness.ctx);
        assert_eq!(failed.len(), 1, "frame {frame}: {failed:?}");
        assert_eq!(failed[0].label, "broken on glow");
        assert!(failed[0].error.contains("multiple samplers"), "{}", failed[0].error);
    }
}

/// A named-parameter effect (F4); identical to the one in `wgpu_headless.rs`:
/// `mix(src, color, amount) * tint * gain + lights[which] * (offset + 1)`.
const NAMED_PARAMS_WGSL: &str = "struct Params {
    color: vec4<f32>,
    amount: f32,
    which: u32,
    offset: i32,
    lights: array<vec4<f32>, 3>,
    tint: vec3<f32>,
    gain: f32,
}
fn effect(uv: vec2<f32>) -> vec4<f32> {
    let base = mix(src(uv), p.color, p.amount);
    return vec4<f32>(base.rgb * p.tint * p.gain + p.lights[p.which].rgb * f32(p.offset + 1), 1.0);
}";

/// F4 on glow: the second uniform block (binding point 1) carries each pass's own values; the
/// chain `a -> invert -> b` gives `(64, 96, 64)` over red and `(128, 96, 0)` over blue.
#[test]
#[ignore = "needs an EGL device (GPU or Mesa); run with --ignored"]
fn named_params_reach_each_pass() {
    let mut harness = Harness::new(0, [0.0, 0.0, 0.0, 1.0]);
    let effect = Effect::new("named", NAMED_PARAMS_WGSL).unwrap_or_else(|error| panic!("{error}"));
    let a = Pass::new(&effect).with("color", [0.0, 1.0, 0.0, 1.0]).and_then(|pass| pass.with("amount", 0.5)).and_then(|pass| pass.with("which", 2_u32)).and_then(|pass| pass.with("offset", 0)).and_then(|pass| pass.with("tint", [1.0, 0.5, 1.0])).and_then(|pass| pass.with("gain", 1.0));
    let mut a = a.unwrap_or_else(|error| panic!("{error}"));
    a.set_slice("lights", &[[9.0_f32; 4], [9.0; 4], [0.0, 0.0, 0.5, 0.0]]).unwrap_or_else(|error| panic!("{error}"));
    let b = Pass::new(&effect).with("tint", [1.0, 1.0, 1.0]).and_then(|pass| pass.with("gain", 0.5)).unwrap_or_else(|error| panic!("{error}"));
    let chain = FilterChain::from(a).then(presets::invert(1.0)).then(b);
    let mut paint = |ui: &mut egui::Ui| {
        let painter = ui.painter();
        halves(painter);
        FilterLayer::new(Rect::from_min_max(pos2(20.0, 20.0), pos2(236.0, 108.0)), chain.clone()).paint(painter);
    };
    harness.frame(&mut paint);
    let pixels = harness.frame(&mut paint);
    assert_active(&harness.ctx);
    assert!(egui_filter_layers::failed_effects(&harness.ctx).is_empty(), "{:?}", egui_filter_layers::failed_effects(&harness.ctx));
    assert_near(&pixels, 60, 60, [64, 96, 64], "named params over red");
    assert_near(&pixels, 200, 60, [128, 96, 0], "named params over blue");
    assert_near(&pixels, 10, 10, [255, 0, 0], "outside the layer");
}

/// Vertical stripes, 4 points wide, alternating red / blue over the whole frame: any horizontal
/// displacement of a read changes the pixel.
fn ripple_stripes(painter: &egui::Painter) {
    for i in 0..64_u8 {
        let x = f32::from(i) * 4.0;
        let colour = if i % 2 == 0 { Color32::RED } else { Color32::BLUE };
        painter.rect_filled(Rect::from_min_max(pos2(x, 0.0), pos2(x + 4.0, 128.0)), 0.0, colour);
    }
}

/// F3 on glow; identical scene to `wgpu_headless.rs`: one impulse at the layer centre (128, 64),
/// 1 s old, travelling 40 points per second with an 8-point wavelength. Pixels in the ring band
/// must move against the same layer without impulses; pixels farther than two wavelengths from
/// the front (the window's compact support) must not.
#[test]
#[ignore = "needs an EGL device (GPU or Mesa); run with --ignored"]
fn click_ripples_displace_the_ring_only() {
    let mut harness = Harness::new(0, [0.0, 0.0, 0.0, 1.0]);
    let rect = Rect::from_min_max(pos2(40.0, 4.0), pos2(216.0, 124.0));
    let mut impulses = egui_filter_layers::Impulses::new(4.0);
    assert!(impulses.push_point(pos2(128.0, 64.0), rect, 1.0, 0.0));
    let rippled = presets::click_ripples(&impulses, 1.0, 6.0, 8.0, 40.0);
    let still = presets::click_ripples(&egui_filter_layers::Impulses::new(4.0), 1.0, 6.0, 8.0, 40.0);
    let mut frame = |pass: &Pass| {
        let mut paint = |ui: &mut egui::Ui| {
            ripple_stripes(ui.painter());
            FilterLayer::new(rect, pass.clone()).paint(ui.painter());
        };
        harness.frame(&mut paint);
        harness.frame(&mut paint)
    };
    let (rippled, still) = (frame(&rippled), frame(&still));
    assert_active(&harness.ctx);
    assert!(egui_filter_layers::failed_effects(&harness.ctx).is_empty(), "{:?}", egui_filter_layers::failed_effects(&harness.ctx));
    assert_near(&still, 130, 64, [255, 0, 0], "no impulses: the stripes pass through");
    let (mut moved_in_band, mut moved_elsewhere) = (0_usize, Vec::new());
    // Inside the layer rect, keeping 2 px off its edges.
    for y in 6..122_u32 {
        for x in 42..214_u32 {
            let dist = (f32::from(u16::try_from(x).unwrap_or(0)) + 0.5 - 128.0).hypot(f32::from(u16::try_from(y).unwrap_or(0)) + 0.5 - 64.0);
            let [a, b] = [rgb(&rippled, x, y), rgb(&still, x, y)];
            let moved = a.iter().zip(b).any(|(a, b)| a.abs_diff(b) > 1);
            // Front at 40 px, wavelength 8 px: non-zero displacement only for 24 < dist < 56.
            if (26.0..54.0).contains(&dist) {
                moved_in_band += usize::from(moved);
            } else if !(22.0..58.0).contains(&dist) && moved {
                moved_elsewhere.push((x, y, a, b));
            }
        }
    }
    assert!(moved_in_band > 100, "the ring must displace pixels near its front: {moved_in_band} moved");
    assert!(moved_elsewhere.is_empty(), "pixels away from the ring must stay exactly: {:?}", &moved_elsewhere[..moved_elsewhere.len().min(8)]);
}

// ---- F6: scaled passes, kawase blur, bloom (same scenes as `wgpu_headless.rs`) ----

/// A smooth backdrop over the whole frame: a mesh whose vertex colours ramp red left -> right
/// and green top -> bottom (blue constant).
fn smooth_ramp(painter: &egui::Painter) {
    let mut mesh = egui::Mesh::default();
    let corners = [(pos2(0.0, 0.0), Color32::from_rgb(0, 0, 64)), (pos2(256.0, 0.0), Color32::from_rgb(255, 0, 64)), (pos2(0.0, 128.0), Color32::from_rgb(0, 255, 64)), (pos2(256.0, 128.0), Color32::from_rgb(255, 255, 64))];
    for (pos, color) in corners {
        mesh.colored_vertex(pos, color);
    }
    mesh.add_triangle(0, 1, 2);
    mesh.add_triangle(1, 3, 2);
    painter.add(egui::Shape::mesh(mesh));
}

/// The largest channel difference between `a` and `b` inside `rect` (pixels, exclusive max).
fn max_diff_in(a: &[u8], b: &[u8], rect: [u32; 4]) -> (u8, (u32, u32)) {
    let mut worst = (0, (0, 0));
    for y in rect[1]..rect[3] {
        for x in rect[0]..rect[2] {
            let diff = rgb(a, x, y).iter().zip(rgb(b, x, y)).map(|(a, b)| a.abs_diff(b)).max().unwrap_or(0);
            if diff > worst.0 {
                worst = (diff, (x, y));
            }
        }
    }
    worst
}

/// The red channel along a row / column through `(x, y)`, `reach` pixels each way.
fn profiles(pixels: &[u8], x: u32, y: u32, reach: u32) -> (Vec<u8>, Vec<u8>) {
    ((x - reach..x + reach).map(|x| rgb(pixels, x, y)[0]).collect(), (y - reach..y + reach).map(|y| rgb(pixels, x, y)[0]).collect())
}

/// Checks a blurred 8x8 white dot centred on the pixel corner (128, 64) over black: symmetric in
/// both axes, spread beyond the dot, energy kept (normalized kernel), nothing beyond `reach`.
fn check_blurred_dot(pixels: &[u8], reach: u32, what: &str) {
    let (row, column) = profiles(pixels, 128, 64, 30);
    for k in 0..30 {
        // Pixel 128 + k mirrors pixel 127 - k (the dot is centred on the corner between them).
        let (right, left) = (row[30 + k], row[29 - k]);
        let (down, up) = (column[30 + k], column[29 - k]);
        assert!(right.abs_diff(left) <= 3 && down.abs_diff(up) <= 3, "{what}: asymmetric at {k}: row {row:?} column {column:?}");
    }
    let centre = rgb(pixels, 128, 64)[0];
    assert!((40..250).contains(&centre), "{what}: the dot must be spread out, centre {centre}");
    assert!(rgb(pixels, 128 + 8, 64)[0] > 8 && rgb(pixels, 128, 64 + 8)[0] > 8, "{what}: light 4 px beyond the dot's edge: {row:?}");
    let far = 132 + reach + 2;
    assert!(rgb(pixels, far, 64)[0] <= 1 && rgb(pixels, 128, 68 + reach.min(50) + 2)[0] <= 1, "{what}: nothing beyond the reach ({reach} px): {row:?}");
    // 64 white pixels of energy (8-bit rounding of many small values: within 12%).
    let energy: u32 = (64 - 50..64 + 50).flat_map(|y| (128 - 60..128 + 60).map(move |x| (x, y))).map(|(x, y)| u32::from(rgb(pixels, x, y)[0])).sum();
    assert!((64 * 255 * 88 / 100..64 * 255 * 112 / 100).contains(&energy), "{what}: energy {energy}, expected about {}", 64 * 255);
}

/// Paints `background`, an 8x8 white dot at (124, 60) .. (132, 68), and `chain` over
/// (32, 4) .. (224, 124).
fn dot_scene(painter: &egui::Painter, chain: &FilterChain, background: Color32) {
    painter.rect_filled(Rect::from_min_max(pos2(0.0, 0.0), pos2(256.0, 128.0)), 0.0, background);
    painter.rect_filled(Rect::from_min_max(pos2(124.0, 60.0), pos2(132.0, 68.0)), 0.0, Color32::WHITE);
    FilterLayer::new(Rect::from_min_max(pos2(32.0, 4.0), pos2(224.0, 124.0)), chain.clone()).paint(painter);
}

/// F6 on glow (viewports from the plan, interpolated uv over the smaller viewport, bottom-up
/// storage of the scaled images): a half-resolution identity (upsampled by the composite) and a
/// half / quarter / full chain match the full-resolution identity on a smooth backdrop over an
/// odd-sized layer.
#[test]
#[ignore = "needs an EGL device (GPU or Mesa); run with --ignored"]
fn a_scaled_identity_chain_matches_the_unscaled_one() {
    let mut harness = Harness::new(0, [0.0, 0.0, 0.0, 1.0]);
    let identity = pass("id", "fn effect(uv: vec2<f32>) -> vec4<f32> { return src(uv); }");
    let rect = Rect::from_min_max(pos2(17.0, 9.0), pos2(234.0, 118.0));
    let mut frame = |chain: &FilterChain| {
        let mut paint = |ui: &mut egui::Ui| {
            smooth_ramp(ui.painter());
            FilterLayer::new(rect, chain.clone()).paint(ui.painter());
        };
        harness.frame(&mut paint);
        harness.frame(&mut paint)
    };
    let full = frame(&FilterChain::from(identity.clone()));
    let half = frame(&FilterChain::from(identity.clone().with_scale(0.5)));
    let pyramid = frame(&FilterChain::from(vec![identity.clone().with_scale(0.5), identity.clone().with_scale(0.25), identity.clone()]));
    assert_active(&harness.ctx);
    assert!(egui_filter_layers::failed_effects(&harness.ctx).is_empty(), "{:?}", egui_filter_layers::failed_effects(&harness.ctx));
    let inside = [19, 11, 232, 116];
    let (worst, at) = max_diff_in(&full, &half, inside);
    assert!(worst <= 3, "half resolution: max diff {worst} at {at:?}");
    let (worst, at) = max_diff_in(&full, &pyramid, inside);
    assert!(worst <= 4, "half / quarter / full: max diff {worst} at {at:?}");
    assert_near(&half, 5, 5, rgb(&full, 5, 5), "outside the layer");
}

/// F6 on glow: `presets::kawase_blur` spreads a bright dot symmetrically, normalized, within its reach.
#[test]
#[ignore = "needs an EGL device (GPU or Mesa); run with --ignored"]
fn a_kawase_blur_spreads_a_dot_symmetrically() {
    let mut harness = Harness::new(0, [0.0, 0.0, 0.0, 1.0]);
    let chain = presets::kawase_blur(16.0);
    let mut paint = |ui: &mut egui::Ui| dot_scene(ui.painter(), &chain, Color32::BLACK);
    harness.frame(&mut paint);
    let pixels = harness.frame(&mut paint);
    assert_active(&harness.ctx);
    assert!(egui_filter_layers::failed_effects(&harness.ctx).is_empty(), "{:?}", egui_filter_layers::failed_effects(&harness.ctx));
    check_blurred_dot(&pixels, 16, "kawase_blur(16)");
}

/// F6 on glow: `presets::bloom` adds light around a bright spot, keeps the spot, and leaves dark
/// areas (below the threshold, beyond the reach) and the outside of the layer untouched.
#[test]
#[ignore = "needs an EGL device (GPU or Mesa); run with --ignored"]
fn bloom_brightens_around_a_bright_spot_only() {
    let mut harness = Harness::new(0, [0.0, 0.0, 0.0, 1.0]);
    let chain = presets::bloom(0.5, 1.0, 16.0);
    let mut paint = |ui: &mut egui::Ui| dot_scene(ui.painter(), &chain, Color32::from_gray(40));
    harness.frame(&mut paint);
    let pixels = harness.frame(&mut paint);
    assert_active(&harness.ctx);
    assert!(egui_filter_layers::failed_effects(&harness.ctx).is_empty(), "{:?}", egui_filter_layers::failed_effects(&harness.ctx));
    assert_near(&pixels, 128, 64, [255, 255, 255], "the spot stays white");
    let (row, _) = profiles(&pixels, 128, 64, 30);
    eprintln!("bloom row through the spot: {row:?}");
    // Light falls off with the distance from the spot's edge (x = 132) and is grey (a white spot).
    let glow = |x: u32| rgb(&pixels, x, 64);
    assert!(glow(134).iter().all(|&value| value >= 52) && glow(134)[0] == glow(134)[2], "2 px from the spot must glow: {row:?}");
    assert!(glow(134)[0] > glow(138)[0] && glow(138)[0] >= glow(142)[0] && glow(138)[0] > 40, "the glow falls off: {row:?}");
    assert!(glow(136)[0].abs_diff(glow(119)[0]) <= 3, "symmetric glow: {row:?}");
    assert_near(&pixels, 60, 30, [40, 40, 40], "dark area beyond the reach");
    assert_near(&pixels, 200, 110, [40, 40, 40], "dark area beyond the reach");
    assert_near(&pixels, 10, 64, [40, 40, 40], "outside the layer");
}

// ---- F1 / F2 on glow (same scenes as `wgpu_headless.rs`) ----

/// F1 on glow: `original(uv)` reads the unfiltered backdrop in EVERY pass, whatever earlier
/// passes did (the capture is bound on unit 1 for every pass, not only the composite):
/// `invert -> original` and `invert -> invert -> original` give the backdrop back, and a pass
/// reading both `src` (inverted) and `original` sees `src + original = 1`. A large layer runs
/// first, so the smaller one after it runs in grown targets (a storage mapping other than 1).
#[test]
#[ignore = "needs an EGL device (GPU or Mesa); run with --ignored"]
fn original_reads_the_backdrop_in_every_pass() {
    let mut harness = Harness::new(0, [0.0, 0.0, 0.0, 1.0]);
    let texture = harness.texture.clone();
    let original = pass("original", "fn effect(uv: vec2<f32>) -> vec4<f32> { return original(uv); }");
    let sum = pass("sum", "fn effect(uv: vec2<f32>) -> vec4<f32> { return vec4<f32>(abs(src(uv).rgb + original(uv).rgb - vec3<f32>(1.0)), 1.0); }");
    let invert = || FilterChain::from(presets::invert(1.0));
    let chains = [("invert, original", invert().then(original.clone()), false), ("invert, invert, original", invert().then(invert()).then(original.clone()), false), ("invert, sum", invert().then(sum), true)];
    let large = Rect::from_min_max(pos2(2.0, 2.0), pos2(254.0, 126.0));
    let small = Rect::from_min_max(pos2(20.0, 20.0), pos2(150.0, 100.0));
    for (name, chain, black) in &chains {
        for rect in [large, small] {
            let mut paint = |ui: &mut egui::Ui| {
                let painter = ui.painter();
                halves(painter);
                painter.rect_filled(Rect::from_min_max(pos2(60.0, 60.0), pos2(80.0, 80.0)), 0.0, Color32::from_rgb(128, 64, 32));
                painter.image(texture.id(), Rect::from_min_max(pos2(24.0, 24.0), pos2(40.0, 40.0)), Rect::from_min_max(Pos2::ZERO, pos2(1.0, 1.0)), Color32::WHITE);
                FilterLayer::new(rect, chain.clone()).paint(painter);
            };
            harness.frame(&mut paint);
            let pixels = harness.frame(&mut paint);
            assert_active(&harness.ctx);
            let what = |point: &str| format!("{name}, layer {rect:?}: {point}");
            let expect = |colour: [u8; 3]| if *black { [0, 0, 0] } else { colour };
            assert_near(&pixels, 100, 50, expect([255, 0, 0]), &what("red"));
            assert_near(&pixels, 140, 50, expect([0, 0, 255]), &what("blue"));
            assert_near(&pixels, 70, 70, expect([128, 64, 32]), &what("mid-tone"));
            assert_near(&pixels, 30, 30, expect([0, 255, 0]), &what("user texture"));
            assert_near(&pixels, 21, 21, expect([255, 0, 0]), &what("top-left corner (orientation)"));
            assert_near(&pixels, 148, 98, expect([0, 0, 255]), &what("bottom-right corner (orientation)"));
        }
    }
}

/// F2 on glow: an effect writing `fl.pointer` sees the pointer in capture uv with the pressed /
/// present flags (converted from screen points with the callback's pixels per point), and one
/// writing `fl.delta_time` / `fl.frame` sees egui's `stable_dt` (1/60 s for frames without a
/// time) and a counter that advances per frame.
#[test]
#[ignore = "needs an EGL device (GPU or Mesa); run with --ignored"]
fn effects_see_the_pointer_and_the_frame_counters() {
    let mut harness = Harness::new(0, [0.0, 0.0, 0.0, 1.0]);
    let pointer = pass("pointer", "fn effect(uv: vec2<f32>) -> vec4<f32> { return vec4<f32>(fl.pointer.x, fl.pointer.y, fl.pointer.z, 1.0) * fl.pointer.w + vec4<f32>(0.0, 0.0, 0.0, 1.0 - fl.pointer.w); }");
    // Capture = region = (32, 16) .. (160, 80) pixels at 1 pixel per point (no padding).
    let rect = Rect::from_min_max(pos2(32.0, 16.0), pos2(160.0, 80.0));
    let mut paint_pointer = |ui: &mut egui::Ui| FilterLayer::new(rect, pointer.clone()).paint(ui.painter());
    harness.frame(&mut paint_pointer);
    let pixels = harness.frame(&mut paint_pointer);
    assert_active(&harness.ctx);
    assert_near(&pixels, 64, 40, [0, 0, 0], "no pointer yet: w = 0");

    // Screen point (64, 32) = capture uv ((64 - 32) / 128, (32 - 16) / 64) = (0.25, 0.25).
    harness.frame_with(vec![egui::Event::PointerMoved(pos2(64.0, 32.0))], &mut |_| {});
    let pixels = harness.frame(&mut paint_pointer);
    assert_near(&pixels, 64, 40, [64, 64, 0], "hovering pointer at uv (0.25, 0.25)");
    harness.frame_with(vec![egui::Event::PointerButton { pos: pos2(64.0, 32.0), button: egui::PointerButton::Primary, pressed: true, modifiers: egui::Modifiers::NONE }], &mut |_| {});
    let pixels = harness.frame(&mut paint_pointer);
    assert_near(&pixels, 100, 70, [64, 64, 255], "pressed pointer: z = 1");

    // `fl.delta_time * 60` = 1 for egui's default 1/60 s step; `fl.frame` mod 2 alternates.
    let counters = pass("counters", "fn effect(uv: vec2<f32>) -> vec4<f32> { return vec4<f32>(fl.delta_time * 60.0, f32(fl.frame % 2u), 0.0, 1.0); }");
    let mut paint_counters = |ui: &mut egui::Ui| FilterLayer::new(rect, counters.clone()).paint(ui.painter());
    let first = harness.frame(&mut paint_counters);
    let second = harness.frame(&mut paint_counters);
    let [first, second] = [rgb(&first, 64, 40), rgb(&second, 64, 40)];
    assert!(first[0] >= 252 && second[0] >= 252, "delta_time = 1/60 s: {first:?} {second:?}");
    assert!(first[1].abs_diff(second[1]) >= 252, "fl.frame advances by one per frame: {first:?} {second:?}");
}

// ---- F8: user textures (same scenes as `wgpu_headless.rs`) ----

/// A 2x2 checker: red | green over blue | white (top row first).
fn checker() -> ColorImage {
    ColorImage::new([2, 2], vec![Color32::RED, Color32::GREEN, Color32::BLUE, Color32::WHITE])
}

/// Slot 0 through its own (egui `TextureOptions`) sampler.
const TEX_SAMPLED: &str = "fn effect(uv: vec2<f32>) -> vec4<f32> { return tex(0u, uv); }";
/// Slot 0 texel by texel (`textureLoad`, no sampler).
const TEX_EXACT: &str = "fn effect(uv: vec2<f32>) -> vec4<f32> { return tex_load(0u, vec2<i32>(floor(uv * tex_size(0u)))); }";
/// Slot 1 through its own sampler.
const TEX_SLOT1: &str = "fn effect(uv: vec2<f32>) -> vec4<f32> { return tex(1u, uv); }";

/// Checks a whole-frame layer showing the 2x2 checker texel by texel: each quadrant one texel,
/// exactly, up to the pixels next to the centre lines.
fn check_checker_quadrants(pixels: &[u8], what: &str) {
    for (x, y, colour) in [(64, 32, [255, 0, 0]), (192, 32, [0, 255, 0]), (64, 96, [0, 0, 255]), (192, 96, [255, 255, 255]), (127, 63, [255, 0, 0]), (128, 63, [0, 255, 0]), (127, 64, [0, 0, 255]), (128, 64, [255, 255, 255])] {
        assert_eq!(rgb(pixels, x, y), colour, "{what}: pixel ({x}, {y})");
    }
}

/// Checks a whole-frame layer showing the 2x2 checker bilinearly: texel colour at a texel centre,
/// the four-texel average at the centre, a quarter step between two texels.
fn check_checker_linear(pixels: &[u8], what: &str) {
    // uv (0.252, 0.254): the red texel's centre.
    assert_near(pixels, 64, 32, [255, 0, 0], &format!("{what}: at the red texel's centre"));
    // uv (0.502, 0.504): the average of all four texels.
    assert_near(pixels, 128, 64, [128, 128, 129], &format!("{what}: between all four texels"));
    // uv (0.377, 0.254): a quarter of the way from red to green.
    assert_near(pixels, 96, 32, [189, 65, 2], &format!("{what}: a quarter from red to green"));
}

/// F8 on glow: a 2x2 checker attached twice, NEAREST at slot 0 and LINEAR at slot 1 (each
/// texture's egui `TextureOptions` is its sampler, i.e. its GL texture parameters), read exactly
/// (`tex_load`), through the nearest sampler (`tex`) and bilinearly; then replaced in full and
/// patched in part between frames, which the next frame must show (no stale texture).
#[test]
#[ignore = "needs an EGL device (GPU or Mesa); run with --ignored"]
fn user_textures_are_sampled_exactly_and_linearly() {
    let mut harness = Harness::new(0, [0.0, 0.0, 0.0, 1.0]);
    let mut nearest = harness.ctx.load_texture("checker nearest", checker(), TextureOptions::NEAREST);
    let linear = harness.ctx.load_texture("checker linear", checker(), TextureOptions::LINEAR);
    let screen = Rect::from_min_max(pos2(0.0, 0.0), pos2(256.0, 128.0));
    let (nearest_id, linear_id) = (nearest.id(), linear.id());
    let frames = |harness: &mut Harness, snippet: &str, count: usize| {
        let effect = pass("textured", snippet);
        let (nearest, linear) = (nearest_id, linear_id);
        let mut paint = |ui: &mut egui::Ui| {
            halves(ui.painter());
            FilterLayer::new(screen, effect.clone()).texture(0, nearest).texture(1, linear).paint(ui.painter());
        };
        (1..count).for_each(|_| drop(harness.frame(&mut paint)));
        harness.frame(&mut paint)
    };
    check_checker_quadrants(&frames(&mut harness, TEX_EXACT, 2), "exact");
    check_checker_quadrants(&frames(&mut harness, TEX_SAMPLED, 2), "nearest sampler");
    check_checker_linear(&frames(&mut harness, TEX_SLOT1, 2), "linear sampler");
    assert_active(&harness.ctx);
    assert!(egui_filter_layers::failed_textures(&harness.ctx).is_empty(), "{:?}", egui_filter_layers::failed_textures(&harness.ctx));

    // A full re-upload, then a partial one, each visible in the very next frame.
    nearest.set(ColorImage::new([2, 2], vec![Color32::YELLOW; 4]), TextureOptions::NEAREST);
    let pixels = frames(&mut harness, TEX_EXACT, 1);
    assert_eq!(rgb(&pixels, 192, 96), [255, 255, 0], "after a full update");
    nearest.set_partial([1, 1], ColorImage::new([1, 1], vec![Color32::BLACK]), TextureOptions::NEAREST);
    let pixels = frames(&mut harness, TEX_EXACT, 1);
    assert_eq!((rgb(&pixels, 192, 96), rgb(&pixels, 64, 32)), ([0, 0, 0], [255, 255, 0]), "after a partial update");
}

/// F8 on glow: three slots at once, each its own texture and size: `(tex(0).r, tex(1).g,
/// tex(2).b)` on the left half, `tex_size` of the three (2x2, 4x4, 3x1) times 32 on the right.
#[test]
#[ignore = "needs an EGL device (GPU or Mesa); run with --ignored"]
fn three_texture_slots_at_once() {
    let mut harness = Harness::new(0, [0.0, 0.0, 0.0, 1.0]);
    let checker = harness.ctx.load_texture("checker", checker(), TextureOptions::NEAREST);
    let blue = harness.ctx.load_texture("blue 3x1", ColorImage::new([3, 1], vec![Color32::from_rgb(0, 0, 100); 3]), TextureOptions::LINEAR);
    let ids = [checker.id(), harness.texture.id(), blue.id()];
    let channels = pass("channels", "fn effect(uv: vec2<f32>) -> vec4<f32> { return vec4<f32>(tex(0u, vec2<f32>(0.25)).r, tex(1u, uv).g, tex(2u, uv).b, 1.0); }");
    let sizes = pass("sizes", "fn effect(uv: vec2<f32>) -> vec4<f32> { return vec4<f32>(vec3<f32>(tex_size(0u).x, tex_size(1u).y, tex_size(2u).x) * 32.0 / 255.0, 1.0); }");
    let mut paint = |ui: &mut egui::Ui| {
        halves(ui.painter());
        let attach = |layer: FilterLayer| ids.iter().enumerate().fold(layer, |layer, (slot, id)| layer.texture(slot, *id));
        attach(FilterLayer::new(Rect::from_min_max(pos2(0.0, 0.0), pos2(128.0, 128.0)), channels.clone())).paint(ui.painter());
        attach(FilterLayer::new(Rect::from_min_max(pos2(128.0, 0.0), pos2(256.0, 128.0)), sizes.clone())).paint(ui.painter());
    };
    harness.frame(&mut paint);
    let pixels = harness.frame(&mut paint);
    assert_active(&harness.ctx);
    assert_near(&pixels, 64, 64, [255, 255, 100], "one channel from each slot");
    assert_near(&pixels, 192, 64, [64, 128, 96], "each slot's own size");
}

/// F8 on glow: a layer attaching an id the painter does not know draws nothing extra and the id
/// is reported (`failed_textures`) without disabling the backend or any effect; a texture freed
/// by egui is reported the same way once the free was applied; a layer whose effect reads a slot
/// with nothing attached is refused before the backend (not a backend failure).
#[test]
#[ignore = "needs an EGL device (GPU or Mesa); run with --ignored"]
fn unknown_freed_and_missing_textures_skip_their_layer_only() {
    let mut harness = Harness::new(0, [0.0, 0.0, 0.0, 1.0]);
    let unknown = egui::TextureId::User(424_242);
    let temporary = harness.ctx.load_texture("temporary", ColorImage::new([1, 1], vec![Color32::WHITE]), TextureOptions::NEAREST);
    let freed = temporary.id();
    let mut temporary = Some(temporary);
    let sampled = pass("sampled", TEX_SAMPLED);
    let mut paint = |ui: &mut egui::Ui| {
        let painter = ui.painter();
        halves(painter);
        FilterLayer::new(Rect::from_min_max(pos2(8.0, 8.0), pos2(56.0, 120.0)), sampled.clone()).texture(0, unknown).paint(painter);
        FilterLayer::new(Rect::from_min_max(pos2(72.0, 8.0), pos2(120.0, 120.0)), sampled.clone()).texture(0, freed).paint(painter);
        // Nothing attached although the effect reads slot 0.
        FilterLayer::new(Rect::from_min_max(pos2(136.0, 8.0), pos2(184.0, 120.0)), sampled.clone()).paint(painter);
        FilterLayer::new(Rect::from_min_max(pos2(200.0, 8.0), pos2(248.0, 120.0)), presets::invert(1.0)).paint(painter);
    };
    harness.frame(&mut paint);
    let pixels = harness.frame(&mut paint);
    assert_near(&pixels, 96, 60, [255, 255, 255], "the live texture is drawn");
    // Dropping the handle frees the texture with the next frame's delta (after its painting).
    temporary.take();
    harness.frame(&mut paint);
    let pixels = harness.frame(&mut paint);
    assert_near(&pixels, 30, 60, [255, 0, 0], "unknown id: the backdrop stays");
    assert_near(&pixels, 96, 60, [255, 0, 0], "freed id: the backdrop stays");
    assert_near(&pixels, 160, 60, [0, 0, 255], "missing attachment: the backdrop stays");
    assert_near(&pixels, 220, 60, [255, 255, 0], "a healthy layer still filters");
    assert_active(&harness.ctx);
    assert!(egui_filter_layers::failed_effects(&harness.ctx).is_empty(), "{:?}", egui_filter_layers::failed_effects(&harness.ctx));
    let mut failed: Vec<egui::TextureId> = egui_filter_layers::failed_textures(&harness.ctx).into_iter().map(|failure| failure.id).collect();
    failed.sort_by_key(|id| format!("{id:?}"));
    let mut expected = vec![unknown, freed];
    expected.sort_by_key(|id| format!("{id:?}"));
    assert_eq!(failed, expected);
}

/// F7: a state pass adding 0.25 per frame (exact texel reads) and a display pass showing it.
fn accumulator() -> FilterChain {
    let step = match Effect::new_state("accumulate", "fn effect(uv: vec2<f32>) -> vec4<f32> { return state_px(vec2<i32>(0)) + vec4<f32>(0.25); }") {
        Ok(effect) => Pass::new(&effect),
        Err(error) => panic!("{error}"),
    };
    FilterChain::new().then(step).then(pass("show", "fn effect(uv: vec2<f32>) -> vec4<f32> { return min(state(uv), vec4<f32>(1.0)); }"))
}

/// The value (0..=255) an [`accumulator`] layer shows after `frames` steps.
fn accumulated(frames: u32) -> u8 {
    [0, 64, 128, 191, 255].get(usize::try_from(frames).unwrap_or(usize::MAX)).copied().unwrap_or(255)
}

/// Checks the grey of pixel `(x, y)` within one step of `expected`.
fn assert_grey(pixels: &[u8], x: u32, y: u32, expected: u8, what: &str) {
    let actual = rgb(pixels, x, y);
    assert!(actual.iter().all(|value| value.abs_diff(expected) <= 1), "{what}: pixel ({x}, {y}) is {actual:?}, expected grey {expected}");
}

/// F7 on glow: accumulation over frames (RGBA16F), `reset_state`, one step per frame for an id
/// painted twice, a missing id refused, an explicit RGBA8 state, and a layer growing past its
/// allocation keeping its state (the blit of `ensure_kept`, rows bottom-up).
#[test]
#[ignore = "needs an EGL device (GPU or Mesa); run with --ignored"]
fn state_accumulates_resets_grows_and_steps_once_per_frame() {
    let mut harness = Harness::new(0, [0.0, 0.0, 0.0, 1.0]);
    let chain = accumulator();
    let id = egui::Id::new("accumulator");
    let left = Rect::from_min_max(pos2(8.0, 8.0), pos2(120.0, 120.0));
    let right = Rect::from_min_max(pos2(136.0, 8.0), pos2(248.0, 60.0));
    let unnamed = Rect::from_min_max(pos2(136.0, 70.0), pos2(248.0, 120.0));
    let mut paint = |ui: &mut egui::Ui| {
        let painter = ui.painter();
        halves(painter);
        FilterLayer::new(left, chain.clone()).id(id).paint(painter);
        FilterLayer::new(right, chain.clone()).id(id).paint(painter);
        FilterLayer::new(unnamed, chain.clone()).paint(painter);
    };
    for frame in 1..=4 {
        let pixels = harness.frame(&mut paint);
        assert_grey(&pixels, 60, 60, accumulated(frame), &format!("frame {frame}, first paint"));
        assert_grey(&pixels, 190, 30, accumulated(frame), &format!("frame {frame}, second paint of the id"));
        assert_near(&pixels, 190, 100, [0, 0, 255], &format!("frame {frame}: no id, refused"));
    }
    assert_active(&harness.ctx);
    assert_eq!(egui_filter_layers::state_count(&harness.ctx), 1);
    egui_filter_layers::reset_state(&harness.ctx, id);
    let pixels = harness.frame(&mut paint);
    assert_grey(&pixels, 60, 60, accumulated(1), "after reset_state");

    for options in [StateOptions::default().with_format(StateFormat::Rgba8Unorm), StateOptions::default().with_scale(0.5)] {
        let id = egui::Id::new(("growing", options.scale().to_bits(), options.format() == StateFormat::Rgba8Unorm));
        let mut rect = Rect::from_min_max(pos2(8.0, 8.0), pos2(40.0, 24.0));
        for frame in 1..=3 {
            if frame == 3 {
                // Far beyond the first allocation (64-px growth steps).
                rect = Rect::from_min_max(pos2(4.0, 4.0), pos2(250.0, 124.0));
            }
            let pixels = harness.frame(&mut |ui| {
                halves(ui.painter());
                FilterLayer::new(rect, chain.clone()).id(id).state(options).paint(ui.painter());
            });
            let what = format!("{options:?}, frame {frame}");
            assert_grey(&pixels, 20, 15, accumulated(frame), &what);
            if frame == 3 {
                assert_grey(&pixels, 200, 100, accumulated(frame), &format!("{what}: the grown part reads the stretched previous state"));
            }
        }
    }
    assert!(egui_filter_layers::failed_states(&harness.ctx).is_empty(), "{:?}", egui_filter_layers::failed_states(&harness.ctx));
    assert!(egui_filter_layers::failed_effects(&harness.ctx).is_empty(), "{:?}", egui_filter_layers::failed_effects(&harness.ctx));
}
