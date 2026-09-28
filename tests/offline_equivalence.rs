//! Offline processing vs the UI, on a real GPU.
//!
//! * Equivalence: an image drawn 1:1 by egui-wgpu under a `FilterLayer` covering the whole screen
//!   and the same image run through `OfflineRenderer::from_render_state` (same device) must give
//!   byte-identical pixels for every preset and a custom effect (dithering off, opaque image,
//!   `Rgba8Unorm` surface), also at 2 pixels per point — except the 1-px antialiased edge of a
//!   rounded mask, which the UI blends over the screen by coverage (one step on a unorm surface,
//!   linear blending on an sRGB one, where it is not compared). An sRGB surface is allowed one 8-bit step
//!   (the hardware's sRGB encode vs gamma values), and so is a translucent image over a background
//!   colour (the UI blends on the GPU, the offline path on the CPU). A dithered UI backdrop
//!   differs from the image by one step, which an effect amplifies by its gain: the bound there is
//!   `ceil(gain) + 1` (one more step for rounding of the RGBA16F intermediates and the output).
//! * Tiling: a frame processed in small tiles must equal the same frame processed whole — exactly
//!   for effects that sample at texel centres. Effects sampling BETWEEN texel centres (ripple,
//!   lens, the custom wave, a Gaussian blur above `sigma * ppp = 32 / 3`) go through hardware bilinear filtering, whose sub-texel weights depend
//!   (rarely) on the texture size and the window origin; they are bounded instead (2 steps,
//!   under 0.1% of the values).
//! * Reduced-resolution chains (F6: `with_scale`, `kawase_blur`, `bloom`): bit for bit with the
//!   UI when processed whole; a frame that would need tiles is refused (`NotTileable`).
//! * User textures (F8): an effect sampling three textures (nearest, linear repeat, exact loads)
//!   attached as egui textures in the UI and as `OfflineTexture::from_color_image` offline is bit
//!   for bit the same, tiled or whole; textures are uploaded once per `OfflineTexture`; missing and
//!   invalid slots are typed errors.
//! * Persistent state (F7): a state pass accumulates across `process` calls of one renderer,
//!   `reset_state` starts over, a new frame size continues the state, a stateful chain refuses
//!   tiling (`NotTileable`); the `wave_sim` preset spreads a centred impulse symmetrically and its
//!   energy stays bounded over many frames.
//! * Shape masks (F9): a star mask matches the UI (exact where coverage is 0 or 1, within one step
//!   on the AA edge), tiled equals whole; text masks and textured meshes are refused.
//! * The GPU tests of this file share one device (`GPU`, a test-only static; see there) and run
//!   in parallel on it.
//!
//! Needs a GPU adapter, so the tests are ignored by default:
//! `cargo test --all-features --test offline_equivalence -- --ignored --nocapture`.

#![cfg(feature = "image")]

use std::sync::Arc;

use egui::{Color32, ColorImage, Pos2, Rect, TextureOptions, ViewportId, ViewportInfo, pos2, vec2};
use egui_filter_layers::{AlphaMode, BackendStatus, Effect, FilterChain, FilterLayer, FrameRef, Impulses, OfflineOptions, OfflineRenderer, OfflineTexture, Pass, StateOptions, TEXTURE_SLOTS, WgpuOptions, install_wgpu, presets};

const WIDTH: u32 = 192;
const HEIGHT: u32 = 128;
/// `fl.time` of every case (exact in `f32` and `f64`).
const TIME: f32 = 0.375;

/// Everything the tests of this file render with, created once on first use.
struct SharedGpu {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// A second device of `adapter` with `max_buffer_size` = [`SMALL_BUFFER_LIMIT`], or why it
    /// could not be created (only the test using it fails then).
    limited: Result<(wgpu::Device, wgpu::Queue), String>,
}

/// `max_buffer_size` of [`SharedGpu::limited`].
const SMALL_BUFFER_LIMIT: u64 = 1 << 20;

/// The one GPU shared by every test of this file (test-only static). With wgpu's Vulkan
/// validation layer (debug builds) on Mesa radv, tests creating and destroying their own
/// instances and devices in parallel crashed with SIGSEGV inside `libvulkan` called from
/// `libVkLayer_khronos_validation` (e.g. in `vkSetDebugUtilsObjectNameEXT` of one test's
/// resource creation while another created or destroyed a device). All devices are therefore
/// created here, together and once, and never destroyed; the tests keep full validation and run
/// in parallel on them (wgpu's handles are `Send + Sync`, its error scopes are per thread).
static GPU: std::sync::OnceLock<SharedGpu> = std::sync::OnceLock::new();

/// The shared GPU of [`GPU`]; panics with the reason when there is no adapter or device.
fn shared_gpu() -> &'static SharedGpu {
    GPU.get_or_init(|| {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).unwrap_or_else(|error| panic!("no wgpu adapter: {error}"));
        eprintln!("adapter: {:?}", adapter.get_info());
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap_or_else(|error| panic!("no wgpu device: {error}"));
        let limits = wgpu::Limits { max_buffer_size: SMALL_BUFFER_LIMIT, ..wgpu::Limits::default() };
        let limited = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor { required_limits: limits, ..Default::default() })).map_err(|error| error.to_string());
        SharedGpu { instance, adapter, device, queue, limited }
    })
}

/// Handles to the shared device, queue and adapter of [`GPU`]; panics with the reason otherwise.
fn gpu() -> (wgpu::Instance, wgpu::Adapter, wgpu::Device, wgpu::Queue) {
    let shared = shared_gpu();
    (shared.instance.clone(), shared.adapter.clone(), shared.device.clone(), shared.queue.clone())
}

/// A deterministic test image: gradients, hash noise and hard edges; `alpha(x, y)` per pixel.
fn test_image(width: u32, height: u32, alpha: impl Fn(u32, u32) -> u8) -> Vec<u8> {
    let mut pixels = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let hash = (x.wrapping_mul(73_856_093) ^ y.wrapping_mul(19_349_663)).wrapping_mul(2_654_435_761) >> 24;
            let noise = u8::try_from(hash & 0xff).unwrap_or(0);
            let r = u8::try_from((x * 255) / width.max(1)).unwrap_or(0);
            let g = if (x / 16 + y / 16) % 2 == 0 { noise } else { 255 - noise / 2 };
            let b = u8::try_from((y * 255) / height.max(1)).unwrap_or(0) ^ (noise & 0x0f);
            pixels.extend_from_slice(&[r, g, b, alpha(x, y)]);
        }
    }
    pixels
}

/// One chain to compare.
struct Case {
    /// Label in the report.
    name: &'static str,
    /// The chain.
    chain: FilterChain,
    /// Corner radius of the layer, points.
    radius: u8,
    /// Largest factor by which the chain amplifies a one-step change of its input.
    gain: f32,
    /// Whether it samples between texel centres (hardware bilinear weights; see the file docs).
    continuous: bool,
}

/// Every preset, a multi-pass combination and a custom effect reading `rect_uv`, `size_px`,
/// `time` and `pixels_per_point`.
fn cases() -> Vec<Case> {
    let custom = Effect::new(
        "custom",
        "fn effect(uv: vec2<f32>) -> vec4<f32> {
    let local = (uv - fl.rect_uv.xy) / fl.rect_uv.zw;
    let wave = sin(local.y * 25.0 + fl.time * 3.0) * params(0u).x * fl.pixels_per_point;
    let c = src_px(vec2<f32>(wave, 0.5 * fl.size_px.y * fl.texel.y));
    return vec4<f32>(c.rgb * (0.6 + 0.4 * local.x), c.a);
}",
    )
    .unwrap_or_else(|error| panic!("{error}"))
    .with_sample_radius(4.0);
    let custom = Pass::new(&custom).with_param(0, [3.0, 0.0, 0.0, 0.0]).unwrap_or_else(|error| panic!("{error}"));
    // F1: `original()` after a pass that changed the image gives the backdrop back.
    let original = Pass::new(&Effect::new("original", "fn effect(uv: vec2<f32>) -> vec4<f32> { return original(uv); }").unwrap_or_else(|error| panic!("{error}")));
    // F4: named parameters per pass (binding 4), with a plain pass between two named ones.
    let named = Effect::new(
        "named",
        "struct Params {
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
}",
    )
    .unwrap_or_else(|error| panic!("{error}"));
    let mut named_a = Pass::new(&named).with("color", [0.0, 1.0, 0.0, 1.0]).and_then(|pass| pass.with("amount", 0.5)).and_then(|pass| pass.with("which", 2_u32)).and_then(|pass| pass.with("tint", [1.0, 0.5, 1.0])).and_then(|pass| pass.with("gain", 1.0)).unwrap_or_else(|error| panic!("{error}"));
    named_a.set_slice("lights", &[[9.0_f32; 4], [9.0; 4], [0.0, 0.0, 0.5, 0.0]]).unwrap_or_else(|error| panic!("{error}"));
    let named_b = Pass::new(&named).with("tint", [1.0, 1.0, 1.0]).and_then(|pass| pass.with("gain", 0.5)).unwrap_or_else(|error| panic!("{error}"));
    let case = |name, chain, radius, gain, continuous| Case { name, chain, radius, gain, continuous };
    vec![
        case("brightness_contrast", presets::brightness_contrast(0.1, 1.3).into(), 0, 1.3, false),
        // grey + 1.5 * (c - grey): a one-step change moves the output by up to 2 steps.
        case("saturation", presets::saturation(1.5).into(), 0, 2.0, false),
        case("grayscale", presets::grayscale(0.7).into(), 0, 1.0, false),
        case("invert", presets::invert(1.0).into(), 0, 1.0, false),
        case("tint", presets::tint(Color32::from_rgb(40, 120, 200), 0.4).into(), 0, 1.0, false),
        case("gaussian_blur", presets::gaussian_blur(3.0), 0, 1.0, false),
        // sigma * ppp above 32 / 3: 32 taps per side at fractional pixel steps (bilinear).
        case("gaussian_blur, large", presets::gaussian_blur(12.0), 0, 1.0, true),
        // c * (1 + 4a) - a * (4 neighbours): 1 + 8 * 0.8.
        case("sharpen", presets::sharpen(0.8).into(), 0, 7.4, false),
        case("pixelate", presets::pixelate(6.0).into(), 0, 1.0, false),
        case("ripple", presets::ripple(3.0, 20.0, 0.5).into(), 0, 1.0, true),
        case("lens", presets::lens(2.0).into(), 0, 1.0, true),
        case("vignette", presets::vignette(0.7, 0.3, 0.4).into(), 0, 1.0, false),
        case("frosted_glass", presets::frosted_glass(2.0, Color32::from_rgb(230, 240, 255), 0.2), 0, 1.0, false),
        case("custom", custom.into(), 0, 1.0, true),
        case("blur+invert, rounded", presets::gaussian_blur(2.0).then(presets::invert(1.0)), 12, 1.0, false),
        case("original_passthrough", FilterChain::from(presets::invert(1.0)).then(original), 0, 1.0, false),
        case("named_params", FilterChain::from(named_a).then(presets::invert(1.0)).then(named_b), 0, 1.0, false),
    ]
}

/// An egui context with the wgpu backend installed on a real GPU, rendering into a `format`
/// target of `WIDTH x HEIGHT` pixels at `ppp` pixels per point.
struct Ui {
    /// The context.
    ctx: egui::Context,
    /// egui-wgpu state (its device is the offline renderer's too).
    state: egui_wgpu::RenderState,
    /// Render target.
    target: wgpu::Texture,
    /// The test image as an egui texture (NEAREST).
    texture: egui::TextureHandle,
    /// Pixels per point.
    ppp: f32,
}

impl Ui {
    fn new(format: wgpu::TextureFormat, dithering: bool, ppp: f32, image: &[u8]) -> Self {
        let (instance, adapter, device, queue) = gpu();
        // One value feeds both, so egui's pass and the filter's pipelines cannot drift apart.
        let renderer_options = egui_wgpu::RendererOptions { dithering, ..Default::default() };
        let renderer = egui_wgpu::Renderer::new(&device, format, renderer_options);
        let state = egui_wgpu::RenderState {
            available_adapters: vec![adapter.clone()],
            adapter,
            instance,
            device,
            queue,
            target_format: format,
            renderer: Arc::new(egui::mutex::RwLock::new(renderer)),
            surface_config: egui_wgpu::SurfaceConfig::LOW_LATENCY,
        };
        let ctx = egui::Context::default();
        let installed = install_wgpu(&ctx, &state, WgpuOptions::from(&renderer_options).with_clear_color([0.0, 0.0, 0.0, 1.0]));
        assert!(installed.is_ok(), "{installed:?}");
        let size = [usize::try_from(WIDTH).unwrap_or(0), usize::try_from(HEIGHT).unwrap_or(0)];
        let texture = ctx.load_texture("test image", ColorImage::from_rgba_unmultiplied(size, image), TextureOptions::NEAREST);
        let target = state.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("test target"),
            size: wgpu::Extent3d { width: WIDTH, height: HEIGHT, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        Self { ctx, state, target, texture, ppp }
    }

    /// The image (1:1, the whole screen) under a layer covering it, as egui-wgpu draws it;
    /// the second frame's pixels (the first uploads the textures).
    fn render(&self, chain: &FilterChain, radius: u8) -> Vec<u8> {
        self.render_textured(chain, radius, [None; TEXTURE_SLOTS])
    }

    /// [`Ui::render`] with `textures` attached to the layer's user texture slots.
    fn render_textured(&self, chain: &FilterChain, radius: u8, textures: [Option<egui::TextureId>; TEXTURE_SLOTS]) -> Vec<u8> {
        let points = vec2(px(WIDTH) / self.ppp, px(HEIGHT) / self.ppp);
        let rect = Rect::from_min_size(Pos2::ZERO, points);
        let mut pixels = Vec::new();
        for _ in 0..2 {
            let viewports = std::iter::once((ViewportId::ROOT, ViewportInfo { native_pixels_per_point: Some(self.ppp), ..Default::default() })).collect();
            let input = egui::RawInput { screen_rect: Some(rect), time: Some(f64::from(TIME)), viewports, ..Default::default() };
            let mut output = self.ctx.run_ui(input, |ui| {
                let painter = ui.painter();
                painter.image(self.texture.id(), rect, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
                let layer = FilterLayer::new(rect, chain.clone()).corner_radius(radius);
                textures.iter().enumerate().fold(layer, |layer, (slot, id)| match id {
                    Some(id) => layer.texture(slot, *id),
                    None => layer,
                }).paint(painter);
            });
            assert!((output.pixels_per_point - self.ppp).abs() < f32::EPSILON, "ppp {}", output.pixels_per_point);
            let mut textures_delta = std::mem::take(&mut output.textures_delta);
            let primitives = self.ctx.tessellate(output.shapes, output.pixels_per_point);
            pixels = self.draw(&primitives, &mut textures_delta);
            textures_delta.clear();
        }
        assert!(matches!(egui_filter_layers::status(&self.ctx), BackendStatus::Active { .. }), "{}", egui_filter_layers::status(&self.ctx));
        pixels
    }

    /// Renders `primitives` with egui-wgpu into the target and reads it back.
    fn draw(&self, primitives: &[egui::ClippedPrimitive], textures_delta: &mut egui::TexturesDelta) -> Vec<u8> {
        let screen = egui_wgpu::ScreenDescriptor { size_in_pixels: [WIDTH, HEIGHT], pixels_per_point: self.ppp };
        let (device, queue) = (&self.state.device, &self.state.queue);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let user_buffers = {
            let mut renderer = self.state.renderer.write();
            for (id, deltas) in &textures_delta.set {
                for delta in deltas {
                    renderer.update_texture(device, queue, *id, delta);
                }
            }
            renderer.update_buffers(device, queue, &mut encoder, primitives, &screen)
        };
        {
            let renderer = self.state.renderer.read();
            let view = self.target.create_view(&wgpu::TextureViewDescriptor::default());
            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("test egui pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            renderer.render(&mut pass.forget_lifetime(), primitives, &screen);
        }
        // WIDTH * 4 = 768 is a multiple of the 256-byte copy alignment.
        let readback = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: u64::from(WIDTH * HEIGHT * 4), usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: &self.target, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo { buffer: &readback, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(WIDTH * 4), rows_per_image: Some(HEIGHT) } },
            wgpu::Extent3d { width: WIDTH, height: HEIGHT, depth_or_array_layers: 1 },
        );
        queue.submit(user_buffers.into_iter().chain(std::iter::once(encoder.finish())));
        readback.slice(..).map_async(wgpu::MapMode::Read, |result| assert!(result.is_ok(), "{result:?}"));
        let polled = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        assert!(polled.is_ok(), "{polled:?}");
        let pixels = readback.slice(..).get_mapped_range().unwrap_or_else(|error| panic!("readback not mapped: {error:?}")).to_vec();
        readback.unmap();
        pixels
    }
}

/// A pixel count as `f32` (test sizes are tiny).
fn px(value: u32) -> f32 {
    f32::from(u16::try_from(value).unwrap_or(u16::MAX))
}

/// Number of differing bytes (RGB only when `rgb_only`), the largest difference, and the first
/// differing pixel.
fn compare(a: &[u8], b: &[u8], width: u32, rgb_only: bool) -> (usize, u8, Option<(usize, usize)>) {
    assert_eq!(a.len(), b.len());
    let mut count = 0;
    let mut max = 0;
    let mut first = None;
    let row = usize::try_from(width).unwrap_or(1);
    for (index, (x, y)) in a.iter().zip(b).enumerate() {
        if rgb_only && index % 4 == 3 {
            continue;
        }
        let diff = x.abs_diff(*y);
        if diff > 0 {
            count += 1;
            max = max.max(diff);
            first.get_or_insert((index / 4 % row, index / 4 / row));
        }
    }
    (count, max, first)
}

/// Whether pixel `(x, y)` lies in the 1-px antialiased edge of the rounded mask of a layer
/// covering the whole `WIDTH` x `HEIGHT` frame with corner radius `radius_px` (the composite's
/// signed distance at the pixel centre within 0.5 px, plus a margin for float error).
fn on_mask_edge(x: u32, y: u32, radius_px: f32) -> bool {
    let half = [px(WIDTH) / 2.0, px(HEIGHT) / 2.0];
    let q = [(px(x) + 0.5 - half[0]).abs(), (px(y) + 0.5 - half[1]).abs()];
    let r = radius_px.min(half[0]).min(half[1]);
    let d = [q[0] - half[0] + r, q[1] - half[1] + r];
    let distance = d[0].max(0.0).hypot(d[1].max(0.0)) + d[0].max(d[1]).min(0.0) - r;
    distance.abs() < 0.5 + 1e-3
}

/// Runs every case through the UI and offline; asserts every colour value within
/// `tolerance(case)` steps. Pixels on a rounded mask's antialiased edge are compared against
/// `edge_bound` instead (`None`: not compared): the UI blends the composite over the screen by
/// coverage (two 8-bit roundings; in linear on an sRGB surface), offline mixes in the shader.
fn assert_equivalent(label: &str, format: wgpu::TextureFormat, dithering: bool, ppp: f32, tolerance: &dyn Fn(&Case) -> u8, edge_bound: Option<u8>) {
    let image = test_image(WIDTH, HEIGHT, |_, _| 255);
    let ui = Ui::new(format, dithering, ppp, &image);
    let mut offline = OfflineRenderer::from_render_state(&ui.state).unwrap_or_else(|error| panic!("{error}"));
    let mut failures = Vec::new();
    for case in cases() {
        let (name, bound) = (case.name, tolerance(&case));
        let expected = ui.render(&case.chain, case.radius);
        let options = OfflineOptions::default().with_pixels_per_point(ppp).with_time(TIME).with_corner_radius(case.radius).with_output_alpha(AlphaMode::Premultiplied);
        let result = offline.process(&FrameRef::new(&image, WIDTH, HEIGHT, AlphaMode::Straight), &case.chain, &options).unwrap_or_else(|error| panic!("{name}: {error}"));
        // Split off the mask edge: interior and exterior pixels keep the case's bound.
        let mut actual = result.pixels().to_vec();
        let (mut edge_expected, mut edge_actual) = (Vec::new(), Vec::new());
        if case.radius > 0 {
            for (index, (ui_pixel, offline_pixel)) in expected.chunks_exact(4).zip(actual.chunks_exact_mut(4)).enumerate() {
                let index = u32::try_from(index).unwrap_or(u32::MAX);
                if on_mask_edge(index % WIDTH, index / WIDTH, f32::from(case.radius) * ppp) {
                    edge_expected.extend_from_slice(ui_pixel);
                    edge_actual.extend_from_slice(offline_pixel);
                    offline_pixel.copy_from_slice(ui_pixel);
                }
            }
        }
        // The surface's alpha is not meaningful on an opaque image; compare colour.
        let (count, max, first) = compare(&expected, &actual, WIDTH, true);
        eprintln!("{label} | {name}: {count} differing channel values, max diff {max} (bound {bound}), first at {first:?}");
        if max > bound {
            failures.push(format!("{name}: max diff {max} > {bound} ({count} values, first at {first:?})"));
        }
        if !edge_expected.is_empty() {
            let (edge_count, edge_max, _) = compare(&edge_expected, &edge_actual, 1, true);
            eprintln!("{label} | {name}, mask edge: {edge_count} differing channel values, max diff {edge_max} (bound {edge_bound:?})");
            if let Some(edge_bound) = edge_bound
                && edge_max > bound.max(edge_bound)
            {
                failures.push(format!("{name}, mask edge: max diff {edge_max} > {}", bound.max(edge_bound)));
            }
        }
    }
    assert!(failures.is_empty(), "{label}: {failures:#?}");
}

#[test]
#[ignore = "needs a GPU adapter (Vulkan / Metal / DX12 / GL); run with --ignored"]
fn offline_equals_the_ui_bit_for_bit() {
    assert_equivalent("exact, ppp 1", wgpu::TextureFormat::Rgba8Unorm, false, 1.0, &|_| 0, Some(1));
}

#[test]
#[ignore = "needs a GPU adapter (Vulkan / Metal / DX12 / GL); run with --ignored"]
fn offline_equals_the_ui_bit_for_bit_at_two_pixels_per_point() {
    assert_equivalent("exact, ppp 2", wgpu::TextureFormat::Rgba8Unorm, false, 2.0, &|_| 0, Some(1));
}

#[test]
#[ignore = "needs a GPU adapter (Vulkan / Metal / DX12 / GL); run with --ignored"]
fn offline_is_within_the_amplified_dither_of_a_dithered_ui() {
    // The dithered backdrop is the image +-1 step; the chain amplifies that by its gain, and the
    // RGBA16F intermediates plus the final rounding add at most one more step.
    assert_equivalent("dithering", wgpu::TextureFormat::Rgba8Unorm, true, 1.0, &|case| dither_bound(case.gain), Some(1));
}

/// `ceil(gain) + 1` steps.
fn dither_bound(gain: f32) -> u8 {
    (1..=u8::MAX).find(|&steps| f32::from(steps) >= gain).map_or(u8::MAX, |steps| steps.saturating_add(1))
}

#[test]
#[ignore = "needs a GPU adapter (Vulkan / Metal / DX12 / GL); run with --ignored"]
fn offline_is_within_one_step_of_an_srgb_ui() {
    // The UI blends a rounded mask's edge in linear on an sRGB surface (like egui's own
    // antialiasing there), offline in gamma: the edge is not comparable.
    assert_equivalent("sRGB surface", wgpu::TextureFormat::Rgba8UnormSrgb, false, 1.0, &|_| 1, None);
}

/// A translucent image: the UI draws it over its clear colour (opaque black) on the GPU, the
/// offline run over `with_background(BLACK)` on the CPU.
#[test]
#[ignore = "needs a GPU adapter (Vulkan / Metal / DX12 / GL); run with --ignored"]
fn a_translucent_image_over_a_background_is_within_one_step() {
    let image = test_image(WIDTH, HEIGHT, |x, y| u8::try_from((x * 7 + y * 3) % 256).unwrap_or(0));
    let ui = Ui::new(wgpu::TextureFormat::Rgba8Unorm, false, 1.0, &image);
    let mut offline = OfflineRenderer::from_render_state(&ui.state).unwrap_or_else(|error| panic!("{error}"));
    let mut worst = 0;
    for Case { name, chain, radius, .. } in cases() {
        let expected = ui.render(&chain, radius);
        let options = OfflineOptions::default().with_time(TIME).with_corner_radius(radius).with_background(Some(Color32::BLACK)).with_output_alpha(AlphaMode::Premultiplied);
        let result = offline.process(&FrameRef::new(&image, WIDTH, HEIGHT, AlphaMode::Straight), &chain, &options).unwrap_or_else(|error| panic!("{name}: {error}"));
        let (count, max, first) = compare(&expected, result.pixels(), WIDTH, true);
        eprintln!("translucent over black | {name}: {count} differing channel values, max diff {max}, first at {first:?}");
        worst = worst.max(max);
    }
    assert!(worst <= 1, "translucent over black: max diff {worst}");
}

/// Tiled processing (forced small tiles) equals whole-frame processing, at 1 and 1.5 pixels per
/// point: exactly for texel-centre sampling, within the hardware-filtering bound otherwise.
#[test]
#[ignore = "needs a GPU adapter (Vulkan / Metal / DX12 / GL); run with --ignored"]
fn tiled_frames_equal_whole_frames() {
    let (_instance, adapter, device, queue) = gpu();
    let mut offline = OfflineRenderer::new(&adapter, &device, &queue).unwrap_or_else(|error| panic!("{error}"));
    let (width, height) = (300, 200);
    let image = test_image(width, height, |x, y| if (x + y) % 5 == 0 { 128 } else { 255 });
    let frame = FrameRef::new(&image, width, height, AlphaMode::Straight);
    let mut failures = Vec::new();
    for ppp in [1.0, 1.5] {
        for Case { name, chain, radius, continuous, .. } in cases() {
            // The lens reads anywhere inside the layer: its reach (half the smaller side times
            // 1 - 1/magnification) must be declared for tiles to overlap enough.
            let (chain, tile) = match name {
                "lens" => (FilterChain::from(presets::lens(2.0).with_sample_radius(px(height) / ppp * 0.25 + 1.0)), 256),
                // Reaches 36+ pixels: 64-pixel tiles would leave no room for content.
                "gaussian_blur, large" => (chain, 128),
                _ => (chain, 64),
            };
            let options = OfflineOptions::default().with_pixels_per_point(ppp).with_time(TIME).with_corner_radius(radius);
            let whole = offline.process(&frame, &chain, &options).unwrap_or_else(|error| panic!("{name}: {error}"));
            let tiled = offline.process(&frame, &chain, &options.clone().with_max_tile_size(Some(tile))).unwrap_or_else(|error| panic!("{name} (tiled): {error}"));
            let (count, max, first) = compare(whole.pixels(), tiled.pixels(), width, false);
            eprintln!("tiles ppp {ppp} | {name}: {count} differing values, max diff {max}, first at {first:?}");
            // Continuous sampling: at most 2 steps on under 0.1% of the values (see the file docs).
            let allowed = if continuous { (2, whole.pixels().len() / 1000) } else { (0, 0) };
            if max > allowed.0 || count > allowed.1 {
                failures.push(format!("ppp {ppp} {name}: {count} values differ (allowed {}), max {max} (allowed {}), first at {first:?}", allowed.1, allowed.0));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// An empty chain returns the frame unchanged; a straight-alpha round trip keeps opaque pixels.
#[test]
#[ignore = "needs a GPU adapter (Vulkan / Metal / DX12 / GL); run with --ignored"]
fn an_empty_chain_and_identity_effects_keep_the_frame() {
    let (_instance, adapter, device, queue) = gpu();
    let mut offline = OfflineRenderer::new(&adapter, &device, &queue).unwrap_or_else(|error| panic!("{error}"));
    let image = test_image(70, 33, |_, _| 255);
    let frame = FrameRef::new(&image, 70, 33, AlphaMode::Straight);
    let empty = offline.process(&frame, &FilterChain::new(), &OfflineOptions::default()).unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(empty.pixels(), image.as_slice());
    let identity = Effect::new("identity", "fn effect(uv: vec2<f32>) -> vec4<f32> { return src(uv); }").unwrap_or_else(|error| panic!("{error}"));
    let same = offline.process(&frame, &Pass::new(&identity).into(), &OfflineOptions::default()).unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(same.pixels(), image.as_slice(), "an identity pass over an opaque frame is exact");
    assert_eq!((same.width(), same.height()), (70, 33));
}

/// Frames wider than `offline::DEFAULT_MAX_TILE_SIZE` are tiled by default (bounded GPU memory),
/// and for texel-centre effects the result equals one untiled run (`with_max_tile_size(None)`).
#[test]
#[ignore = "needs a GPU adapter (Vulkan / Metal / DX12 / GL); run with --ignored"]
fn default_tiling_equals_an_untiled_run() {
    let (_instance, adapter, device, queue) = gpu();
    let mut offline = OfflineRenderer::new(&adapter, &device, &queue).unwrap_or_else(|error| panic!("{error}"));
    let (width, height) = (egui_filter_layers::offline::DEFAULT_MAX_TILE_SIZE + 200, 96);
    assert!(device.limits().max_texture_dimension_2d >= width, "the device cannot run the frame untiled");
    let image = test_image(width, height, |_, _| 255);
    let frame = FrameRef::new(&image, width, height, AlphaMode::Straight);
    let chain = presets::gaussian_blur(3.0).then(presets::sharpen(0.5));
    let options = OfflineOptions::default();
    let tiled = offline.process(&frame, &chain, &options).unwrap_or_else(|error| panic!("{error}"));
    let whole = offline.process(&frame, &chain, &options.clone().with_max_tile_size(None)).unwrap_or_else(|error| panic!("{error}"));
    let (count, max, first) = compare(whole.pixels(), tiled.pixels(), width, false);
    assert_eq!(count, 0, "max diff {max}, first at {first:?}");
}

/// A device whose `max_buffer_size` cannot hold the whole readback: the renderer tiles to fit it
/// instead of failing, with the same pixels as an unconstrained device.
#[test]
#[ignore = "needs a GPU adapter (Vulkan / Metal / DX12 / GL); run with --ignored"]
fn a_small_buffer_limit_is_respected_by_tiling() {
    let (_instance, adapter, device, queue) = gpu();
    let (small_device, small_queue) = shared_gpu().limited.as_ref().unwrap_or_else(|error| panic!("no limited device: {error}"));
    assert_eq!(small_device.limits().max_buffer_size, SMALL_BUFFER_LIMIT);
    let mut reference = OfflineRenderer::new(&adapter, &device, &queue).unwrap_or_else(|error| panic!("{error}"));
    let mut limited = OfflineRenderer::new(&adapter, small_device, small_queue).unwrap_or_else(|error| panic!("{error}"));
    let (width, height) = (1000, 700);
    let image = test_image(width, height, |_, _| 255);
    let frame = FrameRef::new(&image, width, height, AlphaMode::Straight);
    let chain = presets::gaussian_blur(3.0);
    let expected = reference.process(&frame, &chain, &OfflineOptions::default()).unwrap_or_else(|error| panic!("{error}"));
    let result = limited.process(&frame, &chain, &OfflineOptions::default()).unwrap_or_else(|error| panic!("limited device: {error}"));
    let (count, max, first) = compare(expected.pixels(), result.pixels(), width, false);
    assert_eq!(count, 0, "max diff {max}, first at {first:?}");
}

/// Dropping `process_async` while its readback is pending (a cancelled task) must not leave the
/// renderer with a mapped or map-pending buffer: the next run succeeds with the right pixels.
#[test]
#[ignore = "needs a GPU adapter (Vulkan / Metal / DX12 / GL); run with --ignored"]
fn a_cancelled_readback_does_not_brick_the_renderer() {
    let (_instance, adapter, device, queue) = gpu();
    let mut offline = OfflineRenderer::new(&adapter, &device, &queue).unwrap_or_else(|error| panic!("{error}"));
    let image = test_image(WIDTH, HEIGHT, |_, _| 255);
    let frame = FrameRef::new(&image, WIDTH, HEIGHT, AlphaMode::Straight);
    let chain = presets::gaussian_blur(2.0);
    let options = OfflineOptions::default();
    let expected = offline.process(&frame, &chain, &options).unwrap_or_else(|error| panic!("{error}"));
    {
        // Nothing polls the device, so the map cannot complete: the future stops at the readback.
        let mut future = std::pin::pin!(offline.process_async(&frame, &chain, &options));
        let polled = future.as_mut().poll(&mut std::task::Context::from_waker(std::task::Waker::noop()));
        assert!(polled.is_pending(), "the readback completed without a device poll");
    }
    let again = offline.process(&frame, &chain, &options).unwrap_or_else(|error| panic!("after a cancelled run: {error}"));
    assert_eq!(again.pixels(), expected.pixels());
}

/// F1 offline: `original(uv)` after passes that changed the image returns the input exactly, whole
/// and in small tiles (every window's capture mapping has a non-zero offset), and a pass reading
/// both `src` (inverted) and `original` sees exactly their per-pixel pair.
#[test]
#[ignore = "needs a GPU adapter (Vulkan / Metal / DX12 / GL); run with --ignored"]
fn original_returns_the_input_in_every_pass_and_tile() {
    let (_instance, adapter, device, queue) = gpu();
    let mut offline = OfflineRenderer::new(&adapter, &device, &queue).unwrap_or_else(|error| panic!("{error}"));
    let (width, height) = (300, 200);
    let image = test_image(width, height, |_, _| 255);
    let frame = FrameRef::new(&image, width, height, AlphaMode::Straight);
    let pass = |label: &str, source: &str| Pass::new(&Effect::new(label, source).unwrap_or_else(|error| panic!("{label}: {error}")));
    let original = pass("original", "fn effect(uv: vec2<f32>) -> vec4<f32> { return original(uv); }");
    // `1 - src` undoes the invert: equal to `original` only if both read the same pixel.
    let pair = pass("pair", "fn effect(uv: vec2<f32>) -> vec4<f32> { let s = src(uv); let o = original(uv); return select(vec4<f32>(1.0, 0.0, 1.0, 1.0), o, all(abs(vec3<f32>(1.0) - s.rgb - o.rgb) < vec3<f32>(0.5 / 255.0))); }");
    let chains = [("invert, original", FilterChain::from(presets::invert(1.0)).then(original.clone())), ("blur, invert, original", presets::gaussian_blur(3.0).then(presets::invert(1.0)).then(original)), ("invert, pair", FilterChain::from(presets::invert(1.0)).then(pair))];
    for (name, chain) in chains {
        for tile in [None, Some(64)] {
            let options = OfflineOptions::default().with_max_tile_size(tile);
            let result = offline.process(&frame, &chain, &options).unwrap_or_else(|error| panic!("{name}, tile {tile:?}: {error}"));
            let (count, max, first) = compare(&image, result.pixels(), width, false);
            assert_eq!(count, 0, "{name}, tile {tile:?}: max diff {max}, first at {first:?}");
        }
    }
}

/// F2 offline: `fl.delta_time` / `fl.frame` follow a video processor's frame times and index
/// (0 for the first frame), explicit options win, and `fl.pointer` is the option's frame uv with
/// its flags (`w = 0` without a pointer), also when tiled.
#[cfg(feature = "video")]
#[test]
#[ignore = "needs a GPU adapter (Vulkan / Metal / DX12 / GL); run with --ignored"]
fn offline_frame_input_follows_the_options_and_the_video_clock() {
    use egui_filter_layers::PointerState;
    use egui_filter_layers::offline::video::VideoProcessor;

    let (_instance, adapter, device, queue) = gpu();
    let (width, height) = (96, 64);
    let image = test_image(width, height, |_, _| 255);
    let frame = FrameRef::new(&image, width, height, AlphaMode::Straight);
    let pass = |label: &str, source: &str| Pass::new(&Effect::new(label, source).unwrap_or_else(|error| panic!("{label}: {error}")));
    let pixel = |pixels: &[u8], x: u32, y: u32| {
        let at = usize::try_from((y * width + x) * 4).unwrap_or(0);
        [pixels[at], pixels[at + 1], pixels[at + 2]]
    };
    let near = |actual: [u8; 3], expected: [u8; 3], what: &str| assert!(actual.iter().zip(expected).all(|(a, e)| a.abs_diff(e) <= 1), "{what}: {actual:?}, expected {expected:?}");

    // Frame times 0, 0.2, 0.6 s: steps of 51 / 255 and 102 / 255.
    let counters = pass("counters", "fn effect(uv: vec2<f32>) -> vec4<f32> { return vec4<f32>(fl.delta_time, f32(fl.frame) / 255.0, 0.0, 1.0); }");
    let renderer = OfflineRenderer::new(&adapter, &device, &queue).unwrap_or_else(|error| panic!("{error}"));
    let mut processor = VideoProcessor::new(renderer, counters.into(), OfflineOptions::default());
    for (index, (time, delta_byte)) in [(0.0, 0), (0.2, 51), (0.6, 102)].into_iter().enumerate() {
        let result = processor.process_frame(&frame, time).unwrap_or_else(|error| panic!("{error}"));
        let frame_byte = u8::try_from(index).unwrap_or(0);
        near(pixel(result.pixels(), 10, 10), [delta_byte, frame_byte, 0], &format!("video frame {index} at {time} s"));
    }
    processor.set_options(OfflineOptions::default().with_delta_time(0.4).with_frame(9));
    let result = processor.process_frame(&frame, 5.0).unwrap_or_else(|error| panic!("{error}"));
    near(pixel(result.pixels(), 10, 10), [102, 9, 0], "explicit delta_time / frame win");
    processor.set_options(OfflineOptions::default());
    processor.reset_frame_clock();
    let result = processor.process_frame(&frame, 7.0).unwrap_or_else(|error| panic!("{error}"));
    near(pixel(result.pixels(), 10, 10), [0, 0, 0], "a reset clock starts at frame 0, delta 0");

    let pointer = pass("pointer", "fn effect(uv: vec2<f32>) -> vec4<f32> { return vec4<f32>(fl.pointer.x, fl.pointer.y, fl.pointer.z, 1.0) * fl.pointer.w + vec4<f32>(0.0, 0.0, 0.0, 1.0 - fl.pointer.w); }");
    let chain = FilterChain::from(pointer);
    let mut offline = processor.into_renderer();
    for tile in [None, Some(32)] {
        let base = OfflineOptions::default().with_max_tile_size(tile);
        let none = offline.process(&frame, &chain, &base).unwrap_or_else(|error| panic!("{error}"));
        near(pixel(none.pixels(), 50, 40), [0, 0, 0], &format!("tile {tile:?}: no pointer"));
        let options = base.with_pointer(Some(PointerState { uv: [0.25, 0.75], pressed: true }));
        let pressed = offline.process(&frame, &chain, &options).unwrap_or_else(|error| panic!("{error}"));
        for (x, y) in [(1, 1), (50, 40), (95, 63)] {
            near(pixel(pressed.pixels(), x, y), [64, 191, 255], &format!("tile {tile:?}: pointer at ({x}, {y})"));
        }
    }
    let invalid = OfflineOptions::default().with_delta_time(-1.0);
    assert!(offline.process(&frame, &chain, &invalid).is_err(), "a negative delta_time is rejected");
}

/// F6 chains: a kawase blur, a bloom, scaled identity passes ending at full resolution, and one
/// ending scaled (the composite upsamples it). The UI allocates its intermediates at the capture
/// size (the plan's pool), like the offline run, so both sample identical textures.
fn scaled_cases() -> Vec<(&'static str, FilterChain)> {
    let identity = Pass::new(&Effect::new("id", "fn effect(uv: vec2<f32>) -> vec4<f32> { return src(uv); }").unwrap_or_else(|error| panic!("{error}")));
    vec![
        ("kawase_blur", presets::kawase_blur(12.0)),
        ("bloom", presets::bloom(0.4, 1.3, 10.0)),
        ("half / quarter / invert", FilterChain::from(vec![identity.clone().with_scale(0.5), identity.clone().with_scale(0.25)]).then(presets::invert(1.0))),
        ("invert at a third, upsampled by the composite", FilterChain::from(presets::invert(1.0).with_scale(1.0 / 3.0))),
    ]
}

/// F6: offline equals the UI bit for bit for scaled chains (whole frame), at 1 and 2 pixels per
/// point (the kawase tap spacing follows the pixels per point).
#[test]
#[ignore = "needs a GPU adapter (Vulkan / Metal / DX12 / GL); run with --ignored"]
fn scaled_chains_offline_equal_the_ui() {
    let image = test_image(WIDTH, HEIGHT, |_, _| 255);
    let mut failures = Vec::new();
    for ppp in [1.0, 2.0] {
        let ui = Ui::new(wgpu::TextureFormat::Rgba8Unorm, false, ppp, &image);
        let mut offline = OfflineRenderer::from_render_state(&ui.state).unwrap_or_else(|error| panic!("{error}"));
        for (name, chain) in scaled_cases() {
            let expected = ui.render(&chain, 0);
            let options = OfflineOptions::default().with_pixels_per_point(ppp).with_time(TIME).with_output_alpha(AlphaMode::Premultiplied);
            let result = offline.process(&FrameRef::new(&image, WIDTH, HEIGHT, AlphaMode::Straight), &chain, &options).unwrap_or_else(|error| panic!("{name}: {error}"));
            let (count, max, first) = compare(&expected, result.pixels(), WIDTH, true);
            eprintln!("scaled, ppp {ppp} | {name}: {count} differing channel values, max diff {max}, first at {first:?}");
            if max > 0 {
                failures.push(format!("ppp {ppp} {name}: max diff {max} ({count} values, first at {first:?})"));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// F6: a scaled chain on a frame that needs tiles is refused with `NotTileable` (no
/// approximation); the same frame processed whole works and equals an untiled default run; an
/// unscaled chain still tiles.
#[test]
#[ignore = "needs a GPU adapter (Vulkan / Metal / DX12 / GL); run with --ignored"]
fn scaled_chains_refuse_tiling_and_run_whole() {
    let (_instance, adapter, device, queue) = gpu();
    let mut offline = OfflineRenderer::new(&adapter, &device, &queue).unwrap_or_else(|error| panic!("{error}"));
    let (width, height) = (300, 200);
    let image = test_image(width, height, |_, _| 255);
    let frame = FrameRef::new(&image, width, height, AlphaMode::Straight);
    for (name, chain) in scaled_cases() {
        let tiled = offline.process(&frame, &chain, &OfflineOptions::default().with_max_tile_size(Some(128)));
        match tiled {
            Err(egui_filter_layers::OfflineError::NotTileable { reason }) => eprintln!("{name}: refused as expected: {reason}"),
            other => panic!("{name}: a scaled chain must refuse tiling, got {:?}", other.map(|frame| frame.width())),
        }
        let whole = offline.process(&frame, &chain, &OfflineOptions::default().with_max_tile_size(None)).unwrap_or_else(|error| panic!("{name}: {error}"));
        let default = offline.process(&frame, &chain, &OfflineOptions::default()).unwrap_or_else(|error| panic!("{name}: {error}"));
        assert_eq!(whole.pixels(), default.pixels(), "{name}: a frame under the default tile size runs whole");
    }
    // Refusal is about scaled passes only.
    assert!(offline.process(&frame, &presets::gaussian_blur(2.0), &OfflineOptions::default().with_max_tile_size(Some(128))).is_ok());
}

// ---- F8: user textures ----

/// The user-texture effect: a nearest read (clamped) of slot 0, a linear read (repeating,
/// between texel centres) of slot 1 and an exact load of slot 2, mixed with the backdrop.
const TEXTURED_WGSL: &str = "fn effect(uv: vec2<f32>) -> vec4<f32> {
    let a = tex(0u, uv * 1.7);
    let b = tex(1u, uv * 2.3 - vec2<f32>(0.4));
    let c = tex_load(2u, vec2<i32>(floor(uv * tex_size(2u))));
    return src(uv) * 0.25 + a * 0.25 + b * 0.25 + c * 0.25;
}";

/// The three textures of [`TEXTURED_WGSL`]: images of odd sizes (straight colours with alpha,
/// premultiplied by `ColorImage`) and their options.
fn texture_images() -> [(ColorImage, TextureOptions); TEXTURE_SLOTS] {
    let image = |width: u32, height: u32, seed: u32| {
        let pixels = test_image(width, height, |x, y| u8::try_from((x * 37 + y * 91 + seed) % 256).unwrap_or(255));
        ColorImage::from_rgba_unmultiplied([usize::try_from(width).unwrap_or(0), usize::try_from(height).unwrap_or(0)], &pixels)
    };
    [(image(5, 3, 1), TextureOptions::NEAREST), (image(7, 4, 2), TextureOptions::LINEAR_REPEAT), (image(9, 6, 3), TextureOptions::LINEAR)]
}

/// F8: the same textures attached in the UI (egui textures) and offline (`OfflineTexture::
/// from_color_image`, the bytes egui uploads) give the same pixels bit for bit, whole and tiled
/// (user textures are sampled in their own uv, independent of windows).
#[test]
#[ignore = "needs a GPU adapter (Vulkan / Metal / DX12 / GL); run with --ignored"]
fn user_textures_offline_equal_the_ui() {
    let image = test_image(WIDTH, HEIGHT, |_, _| 255);
    let ui = Ui::new(wgpu::TextureFormat::Rgba8Unorm, false, 1.0, &image);
    let chain = FilterChain::from(Pass::new(&Effect::new("textured", TEXTURED_WGSL).unwrap_or_else(|error| panic!("{error}"))));
    let images = texture_images();
    let handles: Vec<egui::TextureHandle> = images.iter().enumerate().map(|(slot, (image, options))| ui.ctx.load_texture(format!("slot {slot}"), image.clone(), *options)).collect();
    let ids = [Some(handles[0].id()), Some(handles[1].id()), Some(handles[2].id())];
    let expected = ui.render_textured(&chain, 0, ids);

    let mut offline = OfflineRenderer::from_render_state(&ui.state).unwrap_or_else(|error| panic!("{error}"));
    let mut options = OfflineOptions::default().with_time(TIME).with_output_alpha(AlphaMode::Premultiplied);
    for (slot, (image, texture_options)) in images.iter().enumerate() {
        options = options.with_texture(slot, OfflineTexture::from_color_image(image, *texture_options).unwrap_or_else(|error| panic!("{error}")));
    }
    let frame = FrameRef::new(&image, WIDTH, HEIGHT, AlphaMode::Straight);
    let whole = offline.process(&frame, &chain, &options).unwrap_or_else(|error| panic!("{error}"));
    let (count, max, first) = compare(&expected, whole.pixels(), WIDTH, true);
    eprintln!("textured, whole: {count} differing channel values, max diff {max}, first at {first:?}");
    assert_eq!(max, 0, "offline must equal the UI ({count} values, first at {first:?})");
    let tiled = offline.process(&frame, &chain, &options.clone().with_max_tile_size(Some(64))).unwrap_or_else(|error| panic!("{error}"));
    let (count, max, first) = compare(whole.pixels(), tiled.pixels(), WIDTH, false);
    assert_eq!(max, 0, "tiles must equal the whole frame ({count} values, first at {first:?})");
    // The texture actually contributes: without it (other contents at slot 0) the result changes.
    let other = OfflineTexture::from_color_image(&ColorImage::new([1, 1], vec![Color32::WHITE]), TextureOptions::NEAREST).unwrap_or_else(|error| panic!("{error}"));
    let changed = offline.process(&frame, &chain, &options.clone().with_texture(0, other)).unwrap_or_else(|error| panic!("{error}"));
    assert_ne!(changed.pixels(), whole.pixels(), "a different texture at slot 0 must change the result");
    let again = offline.process(&frame, &chain, &options).unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(again.pixels(), whole.pixels(), "re-attaching the first texture restores the first result");
}

/// F8 offline errors: an effect reading a slot with nothing attached, and an attachment at a
/// slot that does not exist, are typed errors before any GPU work; a texture larger than the
/// device allows is refused with its size.
#[test]
#[ignore = "needs a GPU adapter (Vulkan / Metal / DX12 / GL); run with --ignored"]
fn missing_and_invalid_textures_are_typed_errors() {
    let (_instance, adapter, device, queue) = gpu();
    let mut offline = OfflineRenderer::new(&adapter, &device, &queue).unwrap_or_else(|error| panic!("{error}"));
    let image = test_image(16, 8, |_, _| 255);
    let frame = FrameRef::new(&image, 16, 8, AlphaMode::Straight);
    let chain = FilterChain::from(Pass::new(&Effect::new("textured", TEXTURED_WGSL).unwrap_or_else(|error| panic!("{error}"))));
    let texture = OfflineTexture::from_color_image(&ColorImage::new([1, 1], vec![Color32::WHITE]), TextureOptions::NEAREST).unwrap_or_else(|error| panic!("{error}"));
    let two = OfflineOptions::default().with_texture(0, texture.clone()).with_texture(1, texture.clone());
    match offline.process(&frame, &chain, &two) {
        Err(egui_filter_layers::OfflineError::Texture(egui_filter_layers::TextureError::NotAttached { slot: 2, pass: 0, .. })) => {}
        other => panic!("slot 2 is read but empty: {:?}", other.map(|frame| frame.width())),
    }
    match offline.process(&frame, &chain, &two.clone().with_texture(3, texture.clone())) {
        Err(egui_filter_layers::OfflineError::Texture(egui_filter_layers::TextureError::InvalidSlot { slot: 3 })) => {}
        other => panic!("slot 3 does not exist: {:?}", other.map(|frame| frame.width())),
    }
    assert!(offline.process(&frame, &chain, &two.clone().with_texture(2, texture)).is_ok());
    let side = device.limits().max_texture_dimension_2d + 1;
    let wide = vec![255_u8; usize::try_from(side).unwrap_or(0) * 4];
    let too_large = OfflineTexture::new(&FrameRef::new(&wide, side, 1, AlphaMode::Straight), TextureOptions::NEAREST).unwrap_or_else(|error| panic!("{error}"));
    match offline.process(&frame, &chain, &two.with_texture(2, too_large)) {
        Err(egui_filter_layers::OfflineError::TextureTooLarge { slot: 2, width, .. }) => assert_eq!(width, side),
        other => panic!("a texture beyond the device limit: {:?}", other.map(|frame| frame.width())),
    }
}

// ---- F7: persistent state ----

/// A state pass adding 0.25 per run (exact texel reads) and a display pass showing the state.
fn accumulator() -> FilterChain {
    let step = Effect::new_state("accumulate", "fn effect(uv: vec2<f32>) -> vec4<f32> { return state_px(vec2<i32>(0)) + vec4<f32>(0.25); }").unwrap_or_else(|error| panic!("{error}"));
    let show = Effect::new("show", "fn effect(uv: vec2<f32>) -> vec4<f32> { return vec4<f32>(min(state(uv).rgb, vec3<f32>(1.0)), 1.0); }").unwrap_or_else(|error| panic!("{error}"));
    FilterChain::new().then(Pass::new(&step)).then(Pass::new(&show))
}

/// The grey an [`accumulator`] run shows after `runs` steps.
fn accumulated(runs: usize) -> u8 {
    [0, 64, 128, 191, 255].get(runs).copied().unwrap_or(255)
}

/// F7 offline: the renderer carries the state across `process` calls (4 calls reach 1.0),
/// `reset_state` starts over, a larger frame continues the state (stretched), and a frame that
/// would need tiles is refused.
#[test]
#[ignore = "needs a GPU adapter (Vulkan / Metal / DX12 / GL); run with --ignored"]
fn offline_state_accumulates_across_runs_and_refuses_tiling() {
    let (_instance, adapter, device, queue) = gpu();
    let mut offline = OfflineRenderer::new(&adapter, &device, &queue).unwrap_or_else(|error| panic!("{error}"));
    let chain = accumulator();
    let small = test_image(40, 30, |_, _| 255);
    let large = test_image(120, 90, |_, _| 255);
    let run = |offline: &mut OfflineRenderer, pixels: &[u8], width: u32, height: u32, options: &OfflineOptions| offline.process(&FrameRef::new(pixels, width, height, AlphaMode::Straight), &chain, options).unwrap_or_else(|error| panic!("{error}"));
    for options in [OfflineOptions::default(), OfflineOptions::default().with_state(StateOptions::default().with_scale(0.5))] {
        offline.reset_state();
        for runs in 1..=4 {
            let out = run(&mut offline, &small, 40, 30, &options);
            assert!(out.pixels().chunks_exact(4).all(|px| px[..3].iter().all(|v| v.abs_diff(accumulated(runs)) <= 1)), "{options:?}: run {runs}");
        }
        offline.reset_state();
        let out = run(&mut offline, &small, 40, 30, &options);
        assert!(out.pixels()[..3].iter().all(|v| v.abs_diff(accumulated(1)) <= 1), "{options:?}: after reset_state");
        let out = run(&mut offline, &large, 120, 90, &options);
        assert!(out.pixels().chunks_exact(4).all(|px| px[..3].iter().all(|v| v.abs_diff(accumulated(2)) <= 1)), "{options:?}: a larger frame continues the state");
    }
    let tiled = offline.process(&FrameRef::new(&large, 120, 90, AlphaMode::Straight), &chain, &OfflineOptions::default().with_max_tile_size(Some(64)));
    assert!(matches!(tiled, Err(egui_filter_layers::OfflineError::NotTileable { .. })), "{:?}", tiled.map(|frame| frame.width()));
}

/// F7: the water preset. One impulse at the centre of an odd-sized frame spreads as a ring that
/// is symmetric under the square's reflections (texel for texel within one 8-bit step of the
/// displayed height), reaches well away from the centre, and stays bounded: after the impulse
/// stopped pushing, the discrete wave energy (`sum v^2 + speed * |grad h|^2 + restore * h^2`,
/// estimated from the displayed height and velocity) never exceeds its peak by more than the
/// 8-bit quantization and has decayed after 120 frames, and nothing saturates.
#[test]
#[ignore = "needs a GPU adapter (Vulkan / Metal / DX12 / GL); run with --ignored"]
fn wave_sim_spreads_an_impulse_symmetrically_and_stays_bounded() {
    let (_instance, adapter, device, queue) = gpu();
    let mut offline = OfflineRenderer::new(&adapter, &device, &queue).unwrap_or_else(|error| panic!("{error}"));
    const SIDE: u32 = 65;
    const GAIN: f64 = 0.25 * 255.0;
    let centre = SIDE / 2;
    let image = test_image(SIDE, SIDE, |_, _| 255);
    let frame = FrameRef::new(&image, SIDE, SIDE, AlphaMode::Straight);
    // Height (red) and velocity (green) shown around mid-grey: 0.5 + 0.25 * value, exact reads.
    let show = Effect::new("height", "fn effect(uv: vec2<f32>) -> vec4<f32> { let s = state_px(vec2<i32>(0)); return vec4<f32>(clamp(vec2<f32>(0.5) + 0.25 * s.rg, vec2<f32>(0.0), vec2<f32>(1.0)), 0.5, 1.0); }").unwrap_or_else(|error| panic!("{error}"));
    let mut impulses = Impulses::new(0.1);
    assert!(impulses.push([0.5, 0.5], 20.0, 0.0));
    let dt = 1.0 / 60.0;
    let (mut energies, mut extremes) = (Vec::new(), Vec::new());
    let mut reached = false;
    for index in 0..120_u32 {
        let now = f64::from(index) * f64::from(dt);
        let sim = presets::wave_sim(&impulses, now, &[]).with("impulse_radius", 2.0).unwrap_or_else(|error| panic!("{error}"));
        let chain = FilterChain::new().then(sim).then(Pass::new(&show));
        let out = offline.process(&frame, &chain, &OfflineOptions::default().with_delta_time(dt)).unwrap_or_else(|error| panic!("{error}"));
        let channel = |x: u32, y: u32, c: u32| out.pixels()[usize::try_from((y * SIDE + x) * 4 + c).unwrap_or(0)];
        let grey = |x: u32, y: u32| channel(x, y, 0);
        for d in 1..centre {
            let axis = [grey(centre + d, centre), grey(centre - d, centre), grey(centre, centre + d), grey(centre, centre - d)];
            let diagonal = [grey(centre + d, centre + d), grey(centre - d, centre - d), grey(centre + d, centre - d), grey(centre - d, centre + d)];
            let knight = [grey(centre + d, centre + d / 2), grey(centre + d / 2, centre + d), grey(centre - d, centre - d / 2), grey(centre - d / 2, centre - d)];
            for group in [axis, diagonal, knight] {
                let (lo, hi) = (group.iter().min().copied().unwrap_or(0), group.iter().max().copied().unwrap_or(0));
                assert!(hi - lo <= 1, "frame {index}, distance {d}: {group:?} is not symmetric");
            }
        }
        let value = |x: u32, y: u32, c: u32| (f64::from(channel(x, y, c)) - 127.5) / GAIN;
        reached |= (value(centre + 16, centre, 0)).abs() >= 0.05;
        let mut energy = 0.0;
        let mut extreme = 0_u8;
        for y in 0..SIDE {
            for x in 0..SIDE {
                let (h, v) = (value(x, y, 0), value(x, y, 1));
                let dx = if x + 1 < SIDE { value(x + 1, y, 0) - h } else { 0.0 };
                let dy = if y + 1 < SIDE { value(x, y + 1, 0) - h } else { 0.0 };
                energy += v * v + 0.35 * (dx * dx + dy * dy) + 0.002 * h * h;
                extreme = extreme.max(channel(x, y, 0).abs_diff(128)).max(channel(x, y, 1).abs_diff(128));
            }
        }
        energies.push(energy);
        extremes.push(extreme);
    }
    assert!(reached, "the ring must travel 16 texels away: energies {energies:?}");
    assert!(extremes.iter().all(|&extreme| extreme < 120), "a value saturated (unstable): {extremes:?}");
    // The impulse pushes during its 0.1 s life (6 frames); afterwards the damped energy may not
    // grow (one quantization step of slack per texel: 1 / GAIN^2 each).
    let slack = f64::from(SIDE * SIDE) * 2.0 / (GAIN * GAIN);
    let peak = energies[7..].iter().copied().fold(0.0, f64::max);
    assert!(peak > slack, "the impulse must have made waves: {energies:?}");
    for (index, window) in energies[7..].windows(2).enumerate() {
        assert!(window[1] <= window[0] * 1.02 + slack, "frame {}: energy grew {} -> {}: {energies:?}", index + 8, window[0], window[1]);
    }
    assert!(energies[119] < 0.8 * peak, "damping must drain energy: {energies:?}");
}

/// F7: `trails`: a white square moving over black leaves a trace that halves every second
/// (`decay = 0.5`, one frame per second), while the square itself stays white.
#[test]
#[ignore = "needs a GPU adapter (Vulkan / Metal / DX12 / GL); run with --ignored"]
fn trails_fade_by_their_decay_per_second() {
    let (_instance, adapter, device, queue) = gpu();
    let mut offline = OfflineRenderer::new(&adapter, &device, &queue).unwrap_or_else(|error| panic!("{error}"));
    const SIDE: u32 = 48;
    let square_at = |left: u32| (0..SIDE).flat_map(|y| (0..SIDE).map(move |x| (x, y))).flat_map(|(x, y)| if (left..left + 8).contains(&x) && (20..28).contains(&y) { [255_u8; 4] } else { [0, 0, 0, 255] }).collect::<Vec<u8>>();
    let chain = presets::trails(0.5);
    let options = OfflineOptions::default().with_delta_time(1.0);
    let mut red_at = Vec::new();
    for left in [4, 16, 28] {
        let pixels = square_at(left);
        let out = offline.process(&FrameRef::new(&pixels, SIDE, SIDE, AlphaMode::Straight), &chain, &options).unwrap_or_else(|error| panic!("{error}"));
        let red = |x: u32| out.pixels()[usize::try_from((24 * SIDE + x) * 4).unwrap_or(0)];
        red_at.push([red(6), red(18), red(30)]);
    }
    // Rows: frames; columns: where the square was on frames 1, 2, 3.
    let expected = [[255, 0, 0], [128, 255, 0], [64, 128, 255]];
    for (row, want) in red_at.iter().zip(expected) {
        assert!(row.iter().zip(want).all(|(value, want)| value.abs_diff(want) <= 1), "{red_at:?}");
    }
}

// ---- F9: shape masks ----

/// A five-pointed star centred at `center` (points), tips at `outer`, notches at `inner`, as
/// convex pieces (egui fills convex paths only): the inner pentagon and one triangle per tip.
fn star(center: Pos2, outer: f32, inner: f32) -> egui::Shape {
    let at = |radius: f32, degrees: f32| center + radius * vec2(degrees.to_radians().cos(), degrees.to_radians().sin());
    let notch = |k: u8| at(inner, -54.0 + 72.0 * f32::from(k));
    let mut pieces = vec![egui::Shape::convex_polygon((0..5).map(notch).collect(), Color32::WHITE, egui::Stroke::NONE)];
    for k in 0..5_u8 {
        let tip = at(outer, -90.0 + 72.0 * f32::from(k));
        pieces.push(egui::Shape::convex_polygon(vec![notch((k + 4) % 5), tip, notch(k)], Color32::WHITE, egui::Stroke::NONE));
    }
    egui::Shape::Vec(pieces)
}

/// The star mask of the equivalence test in frame points at `ppp`: centred, 56 / 22 PIXELS.
fn frame_star(ppp: f32) -> egui::Shape {
    star(pos2(px(WIDTH) / 2.0 / ppp, px(HEIGHT) / 2.0 / ppp), 56.0 / ppp, 22.0 / ppp)
}

impl Ui {
    /// The image under a layer covering it masked by `mask` (F9), as egui-wgpu draws it; the
    /// second frame's pixels.
    fn render_masked(&self, chain: &FilterChain, mask: &egui_filter_layers::Mask) -> Vec<u8> {
        let rect = Rect::from_min_size(Pos2::ZERO, vec2(px(WIDTH) / self.ppp, px(HEIGHT) / self.ppp));
        let mut pixels = Vec::new();
        for _ in 0..2 {
            let viewports = std::iter::once((ViewportId::ROOT, ViewportInfo { native_pixels_per_point: Some(self.ppp), ..Default::default() })).collect();
            let input = egui::RawInput { screen_rect: Some(rect), time: Some(f64::from(TIME)), viewports, ..Default::default() };
            let mut output = self.ctx.run_ui(input, |ui| {
                let painter = ui.painter();
                painter.image(self.texture.id(), rect, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
                FilterLayer::new(rect, chain.clone()).mask(mask.clone()).paint(painter);
            });
            let mut textures_delta = std::mem::take(&mut output.textures_delta);
            let primitives = self.ctx.tessellate(output.shapes, output.pixels_per_point);
            pixels = self.draw(&primitives, &mut textures_delta);
            textures_delta.clear();
        }
        assert!(matches!(egui_filter_layers::status(&self.ctx), BackendStatus::Active { .. }), "{}", egui_filter_layers::status(&self.ctx));
        pixels
    }
}

/// F9: a star path mask offline equals the UI — exactly wherever the coverage is 0 or 1 (the
/// same tessellation and the same rasterizer: `wgpu_backend/coverage.rs`), within one step on the
/// anti-aliased edge (the UI blends the composite over the screen, offline mixes in the shader);
/// outside the star the frame is untouched. The same run tiled equals the whole run within one
/// step on the edge (each tile rasterizes its own window). Text masks (no font atlas offline),
/// textured meshes and paint callbacks are refused with typed errors before any GPU work.
#[test]
#[ignore = "needs a GPU adapter (Vulkan / Metal / DX12 / GL); run with --ignored"]
fn a_path_mask_offline_equals_the_ui_and_text_masks_are_refused() {
    let image = test_image(WIDTH, HEIGHT, |_, _| 255);
    let chain = FilterChain::from(presets::invert(1.0)).then(presets::gaussian_blur(1.5));
    for ppp in [1.0, 2.0] {
        let ui = Ui::new(wgpu::TextureFormat::Rgba8Unorm, false, ppp, &image);
        let mut offline = OfflineRenderer::from_render_state(&ui.state).unwrap_or_else(|error| panic!("{error}"));
        let mask = egui_filter_layers::Mask::Shape(frame_star(ppp));
        let expected = ui.render_masked(&chain, &mask);
        let options = OfflineOptions::default().with_pixels_per_point(ppp).with_time(TIME).with_mask(mask.clone()).with_output_alpha(AlphaMode::Premultiplied);
        let result = offline.process(&FrameRef::new(&image, WIDTH, HEIGHT, AlphaMode::Straight), &chain, &options).unwrap_or_else(|error| panic!("{error}"));
        let (count, max, first) = compare(&expected, result.pixels(), WIDTH, true);
        eprintln!("star mask, ppp {ppp}: {count} differing values, max diff {max}, first at {first:?}");
        assert!(max <= 1, "ppp {ppp}: max diff {max} ({count} values, first at {first:?})");
        // The anti-aliased outline is a few hundred pixels at most; everything else is exact.
        assert!(count < 600, "ppp {ppp}: {count} values differ");
        let at = |pixels: &[u8], x: u32, y: u32| {
            let i = usize::try_from((y * WIDTH + x) * 4).unwrap_or(0);
            [pixels[i], pixels[i + 1], pixels[i + 2]]
        };
        // Centre: filtered (inverted); a corner: the image itself.
        let (centre, corner) = ((WIDTH / 2, HEIGHT / 2), (3, 3));
        assert_ne!(at(result.pixels(), centre.0, centre.1), at(&image, centre.0, centre.1), "ppp {ppp}: the star's centre is filtered");
        assert_eq!(at(result.pixels(), corner.0, corner.1), at(&image, corner.0, corner.1), "ppp {ppp}: outside the star the frame is kept");

        let tiled = offline.process(&FrameRef::new(&image, WIDTH, HEIGHT, AlphaMode::Straight), &chain, &options.clone().with_max_tile_size(Some(64))).unwrap_or_else(|error| panic!("tiled: {error}"));
        let (count, max, first) = compare(result.pixels(), tiled.pixels(), WIDTH, false);
        eprintln!("star mask tiled vs whole, ppp {ppp}: {count} differing values, max diff {max}, first at {first:?}");
        assert!(max <= 1 && count < 200, "ppp {ppp}: tiled vs whole: {count} values, max {max}, first at {first:?}");
    }

    // Refusals: typed, before any GPU work.
    let (_instance, adapter, device, queue) = gpu();
    let mut offline = OfflineRenderer::new(&adapter, &device, &queue).unwrap_or_else(|error| panic!("{error}"));
    let frame = FrameRef::new(&image, WIDTH, HEIGHT, AlphaMode::Straight);
    let layout = egui::Context::default();
    let mut galley = None;
    let mut output = layout.run_ui(egui::RawInput::default(), |ui| galley = Some(ui.painter().layout_no_wrap("MASK".to_owned(), egui::FontId::monospace(32.0), Color32::WHITE)));
    // The font atlas upload is irrelevant here; epaint asserts deltas are handled.
    output.textures_delta.clear();
    let Some(galley) = galley else { panic!("the layout pass ran") };
    let text = egui_filter_layers::Mask::Shape(egui::Shape::galley(pos2(10.0, 10.0), galley, Color32::WHITE));
    let refused = offline.process(&frame, &FilterChain::from(presets::invert(1.0)), &OfflineOptions::default().with_mask(text));
    assert!(matches!(refused, Err(egui_filter_layers::OfflineError::Mask(egui_filter_layers::MaskError::Text))), "{:?}", refused.map(|frame| frame.width()));
    let mut textured = egui::Mesh::with_texture(egui::TextureId::User(1));
    textured.add_rect_with_uv(Rect::from_min_max(pos2(0.0, 0.0), pos2(20.0, 20.0)), Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
    let refused = offline.process(&frame, &FilterChain::from(presets::invert(1.0)), &OfflineOptions::default().with_mask(egui_filter_layers::Mask::Shape(egui::Shape::mesh(textured))));
    assert!(matches!(refused, Err(egui_filter_layers::OfflineError::Mask(egui_filter_layers::MaskError::TexturedMesh { .. }))), "{:?}", refused.map(|frame| frame.width()));
}
