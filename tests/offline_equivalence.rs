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
//!
//! Needs a GPU adapter, so the tests are ignored by default:
//! `cargo test --all-features --test offline_equivalence -- --ignored --nocapture`.

#![cfg(feature = "image")]

use std::sync::Arc;

use egui::{Color32, ColorImage, Pos2, Rect, TextureOptions, ViewportId, ViewportInfo, pos2, vec2};
use egui_filter_layers::{AlphaMode, BackendStatus, Effect, FilterChain, FilterLayer, FrameRef, OfflineOptions, OfflineRenderer, Pass, WgpuOptions, install_wgpu, presets};

const WIDTH: u32 = 192;
const HEIGHT: u32 = 128;
/// `fl.time` of every case (exact in `f32` and `f64`).
const TIME: f32 = 0.375;

/// Device, queue and adapter of any available GPU; panics with the reason otherwise.
fn gpu() -> (wgpu::Instance, wgpu::Adapter, wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).unwrap_or_else(|error| panic!("no wgpu adapter: {error}"));
    eprintln!("adapter: {:?}", adapter.get_info());
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap_or_else(|error| panic!("no wgpu device: {error}"));
    (instance, adapter, device, queue)
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
        let points = vec2(px(WIDTH) / self.ppp, px(HEIGHT) / self.ppp);
        let rect = Rect::from_min_size(Pos2::ZERO, points);
        let mut pixels = Vec::new();
        for _ in 0..2 {
            let viewports = std::iter::once((ViewportId::ROOT, ViewportInfo { native_pixels_per_point: Some(self.ppp), ..Default::default() })).collect();
            let input = egui::RawInput { screen_rect: Some(rect), time: Some(f64::from(TIME)), viewports, ..Default::default() };
            let mut output = self.ctx.run_ui(input, |ui| {
                let painter = ui.painter();
                painter.image(self.texture.id(), rect, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
                FilterLayer::new(rect, chain.clone()).corner_radius(radius).paint(painter);
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
    let limits = wgpu::Limits { max_buffer_size: 1 << 20, ..wgpu::Limits::default() };
    let (small_device, small_queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor { required_limits: limits, ..Default::default() })).unwrap_or_else(|error| panic!("no limited device: {error}"));
    let mut reference = OfflineRenderer::new(&adapter, &device, &queue).unwrap_or_else(|error| panic!("{error}"));
    let mut limited = OfflineRenderer::new(&adapter, &small_device, &small_queue).unwrap_or_else(|error| panic!("{error}"));
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
