//! Headless end-to-end check of the wgpu backend: a real `egui_wgpu::Renderer` renders a known
//! backdrop (two solid halves and a user texture) under an invert filter into an offscreen
//! texture, which is read back and checked pixel by pixel; a second, overlapping layer checks the
//! documented nesting behaviour (a lower filter is not seen by an upper one on wgpu). A
//! multi-pass first frame (`request_discard`) checks textures uploaded in an earlier pass. A
//! filter inside a zoomed layer checks that effect distances in points follow the layer's zoom.
//! A rounded layer over a lower filter checks that its cut-off corners keep the screen (the
//! blended composite), and a semi-transparent shape half under an identity filter checks that the
//! re-render blends like egui (in linear on `*Srgb` targets; not asserted on adapters without
//! texture view formats, the documented GL / WebGL2 gap). The composite's alpha channel and its
//! partly covered edge are checked over a transparent frame against CPU references, and the
//! rounded-corner scene runs in MSAA / depth / stencil passes (pipeline-pass compatibility).
//! `original(uv)` after passes that changed the image must give the backdrop back in every pass.
//!
//! Needs a GPU adapter (Vulkan, Metal, DX12 or GL), so it is ignored by default:
//! `cargo test --all-features -- --ignored`.

#![cfg(feature = "wgpu")]

use std::sync::Arc;

use egui::{Align2, Color32, ColorImage, FontId, Pos2, Rect, TextureOptions, pos2, vec2};
use egui::emath::TSTransform;
use egui_filter_layers::{BackendStatus, Effect, FilterChain, FilterLayer, Pass, WgpuOptions, install_wgpu, presets};

const WIDTH: u32 = 256;
const HEIGHT: u32 = 128;

/// Device, queue and adapter of any available GPU, or `None` with the reason printed.
fn gpu() -> Option<(wgpu::Instance, wgpu::Adapter, wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())) {
        Ok(adapter) => adapter,
        Err(error) => {
            eprintln!("no wgpu adapter: {error}");
            return None;
        }
    };
    eprintln!("adapter: {:?}", adapter.get_info());
    match pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())) {
        Ok((device, queue)) => Some((instance, adapter, device, queue)),
        Err(error) => {
            eprintln!("no wgpu device: {error}");
            None
        }
    }
}

/// Paints the scene of one frame into `ui`.
fn scene(ui: &egui::Ui, texture: &egui::TextureHandle) {
    let painter = ui.painter();
    painter.rect_filled(Rect::from_min_max(pos2(0.0, 0.0), pos2(128.0, 128.0)), 0.0, Color32::RED);
    painter.rect_filled(Rect::from_min_max(pos2(128.0, 0.0), pos2(256.0, 128.0)), 0.0, Color32::BLUE);
    painter.rect_filled(Rect::from_min_max(pos2(60.0, 60.0), pos2(80.0, 80.0)), 0.0, Color32::from_rgb(128, 64, 32));
    painter.image(texture.id(), Rect::from_min_max(pos2(8.0, 8.0), pos2(40.0, 40.0)), Rect::from_min_max(Pos2::ZERO, pos2(1.0, 1.0)), Color32::WHITE);
    FilterLayer::new(Rect::from_min_max(pos2(4.0, 4.0), pos2(200.0, 100.0)), presets::invert(1.0)).paint(painter);
    // Painted after the filter: must stay untouched.
    painter.rect_filled(Rect::from_min_max(pos2(160.0, 60.0), pos2(180.0, 80.0)), 0.0, Color32::WHITE);
    // A second layer overlapping the first: on wgpu it sees the first one's UNFILTERED backdrop.
    FilterLayer::new(Rect::from_min_max(pos2(150.0, 90.0), pos2(250.0, 125.0)), presets::invert(1.0)).paint(painter);
}

/// The attachments of the test's egui render pass: what egui-wgpu was configured for.
struct Targets {
    /// The single-sample target that is read back (the resolve target under MSAA).
    resolved: wgpu::Texture,
    /// The multisampled colour attachment, when MSAA is on.
    multisampled: Option<wgpu::Texture>,
    /// The depth/stencil attachment and its format, when egui-wgpu has one.
    depth_stencil: Option<(wgpu::Texture, wgpu::TextureFormat)>,
    /// What the pass clears the colour attachment to (eframe's clear colour).
    clear: wgpu::Color,
}

/// Renders one egui frame (`paint` runs once per egui pass) through egui-wgpu into `targets`,
/// returning the RGBA8 pixels of the resolved target and the number of egui passes the frame took.
fn render_frame(ctx: &egui::Context, state: &egui_wgpu::RenderState, targets: &Targets, paint: &mut dyn FnMut(&mut egui::Ui)) -> (Vec<u8>, usize) {
    let input = egui::RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(256.0, 128.0))), ..Default::default() };
    let mut output = ctx.run_ui(input, |ui| paint(ui));
    let passes = output.platform_output.num_completed_passes;
    let mut textures_delta = std::mem::take(&mut output.textures_delta);
    let primitives = ctx.tessellate(output.shapes, output.pixels_per_point);
    let screen = egui_wgpu::ScreenDescriptor { size_in_pixels: [WIDTH, HEIGHT], pixels_per_point: output.pixels_per_point };
    let (device, queue) = (&state.device, &state.queue);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    let user_buffers = {
        let mut renderer = state.renderer.write();
        for (id, deltas) in &textures_delta.set {
            for delta in deltas {
                renderer.update_texture(device, queue, *id, delta);
            }
        }
        renderer.update_buffers(device, queue, &mut encoder, &primitives, &screen)
    };
    // Applied above (nothing is freed in this test); egui asserts deltas are handled.
    textures_delta.clear();
    {
        let renderer = state.renderer.read();
        let resolved = targets.resolved.create_view(&wgpu::TextureViewDescriptor::default());
        let multisampled = targets.multisampled.as_ref().map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()));
        let (view, resolve_target) = match &multisampled {
            Some(multisampled) => (multisampled, Some(&resolved)),
            None => (&resolved, None),
        };
        let depth_stencil = targets.depth_stencil.as_ref().map(|(texture, format)| (texture.create_view(&wgpu::TextureViewDescriptor::default()), *format));
        let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("test egui pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment { view, depth_slice: None, resolve_target, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(targets.clear), store: wgpu::StoreOp::Store } })],
            depth_stencil_attachment: depth_stencil.as_ref().map(|(view, format)| wgpu::RenderPassDepthStencilAttachment {
                view,
                depth_ops: format.has_depth_aspect().then_some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Discard }),
                stencil_ops: format.has_stencil_aspect().then_some(wgpu::Operations { load: wgpu::LoadOp::Clear(0), store: wgpu::StoreOp::Discard }),
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        renderer.render(&mut pass.forget_lifetime(), &primitives, &screen);
    }
    let readback = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: u64::from(WIDTH * HEIGHT * 4), usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo { texture: &targets.resolved, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        wgpu::TexelCopyBufferInfo { buffer: &readback, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(WIDTH * 4), rows_per_image: Some(HEIGHT) } },
        wgpu::Extent3d { width: WIDTH, height: HEIGHT, depth_or_array_layers: 1 },
    );
    queue.submit(user_buffers.into_iter().chain(std::iter::once(encoder.finish())));
    readback.slice(..).map_async(wgpu::MapMode::Read, |result| assert!(result.is_ok(), "{result:?}"));
    let polled = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
    assert!(polled.is_ok(), "{polled:?}");
    let pixels = match readback.slice(..).get_mapped_range() {
        Ok(view) => view.to_vec(),
        Err(error) => panic!("readback not mapped: {error:?}"),
    };
    readback.unmap();
    (pixels, passes)
}

/// The RGB of pixel `(x, y)`.
fn rgb(pixels: &[u8], x: u32, y: u32) -> [u8; 3] {
    let i = usize::try_from((y * WIDTH + x) * 4).unwrap_or(0);
    [pixels[i], pixels[i + 1], pixels[i + 2]]
}

fn assert_near(pixels: &[u8], x: u32, y: u32, expected: [u8; 3], what: &str) {
    let actual = rgb(pixels, x, y);
    let close = actual.iter().zip(expected).all(|(a, e)| a.abs_diff(e) <= 3);
    assert!(close, "{what}: pixel ({x}, {y}) is {actual:?}, expected {expected:?}");
}

#[test]
#[ignore = "needs a GPU adapter (Vulkan / Metal / DX12 / GL); run with --ignored"]
fn a_known_backdrop_is_inverted_under_the_layer_only() {
    run_scene(wgpu::TextureFormat::Rgba8Unorm);
}

/// Same scene on an sRGB render target: the composite must output linear values.
#[test]
#[ignore = "needs a GPU adapter (Vulkan / Metal / DX12 / GL); run with --ignored"]
fn an_srgb_target_gets_the_same_colours() {
    run_scene(wgpu::TextureFormat::Rgba8UnormSrgb);
}

/// Everything one test renders with: an egui context with the backend installed on a real GPU,
/// the render state, a 4x4 green user texture and a `format` render target (no MSAA, no depth,
/// no dithering, cleared to opaque black).
fn setup(format: wgpu::TextureFormat) -> (egui::Context, egui_wgpu::RenderState, egui::TextureHandle, Targets) {
    setup_with(format, egui_wgpu::RendererOptions { dithering: false, ..Default::default() }, [0.0, 0.0, 0.0, 1.0])
}

/// As [`setup`], with egui-wgpu configured by `renderer_options` (MSAA, depth/stencil) and the
/// frame cleared to `clear` (premultiplied gamma RGBA, eframe's `App::clear_color`).
fn setup_with(format: wgpu::TextureFormat, renderer_options: egui_wgpu::RendererOptions, clear: [f32; 4]) -> (egui::Context, egui_wgpu::RenderState, egui::TextureHandle, Targets) {
    let Some((instance, adapter, device, queue)) = gpu() else {
        panic!("this test needs a wgpu adapter; see stderr");
    };
    // One value feeds both, so egui's pass and the filter's pipelines cannot drift apart.
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
    let installed = install_wgpu(&ctx, &state, WgpuOptions::from(&renderer_options).with_clear_color(clear));
    assert!(installed.is_ok(), "{installed:?}");
    let texture = ctx.load_texture("green", ColorImage::new([4, 4], vec![Color32::GREEN; 16]), TextureOptions::NEAREST);
    let attachment = |sample_count: u32, format: wgpu::TextureFormat, usage: wgpu::TextureUsages| {
        state.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("test attachment"),
            size: wgpu::Extent3d { width: WIDTH, height: HEIGHT, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        })
    };
    let samples = renderer_options.msaa_samples.max(1);
    let targets = Targets {
        resolved: attachment(1, format, wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC),
        multisampled: (samples > 1).then(|| attachment(samples, format, wgpu::TextureUsages::RENDER_ATTACHMENT)),
        depth_stencil: renderer_options.depth_stencil_format.map(|depth| (attachment(samples, depth, wgpu::TextureUsages::RENDER_ATTACHMENT), depth)),
        // eframe hands its clear colour to wgpu unchanged.
        clear: wgpu::Color { r: f64::from(clear[0]), g: f64::from(clear[1]), b: f64::from(clear[2]), a: f64::from(clear[3]) },
    };
    (ctx, state, texture, targets)
}

/// Renders the scene for two frames into a `format` target and checks both.
fn run_scene(format: wgpu::TextureFormat) {
    let (ctx, state, texture, target) = setup(format);

    // Frame 1 uploads the font atlas and the user texture in full: the backdrop uses the crate's
    // own copies, since egui-wgpu applies them only after the pass.
    let (first, _) = render_frame(&ctx, &state, &target, &mut |ui| scene(ui, &texture));
    assert_near(&first, 100, 50, [0, 255, 255], "frame 1, red under the filter");
    assert_near(&first, 20, 20, [255, 0, 255], "frame 1, user texture under the filter");
    let (pixels, _) = render_frame(&ctx, &state, &target, &mut |ui| scene(ui, &texture));
    assert!(matches!(egui_filter_layers::status(&ctx), BackendStatus::Active { .. }), "{}", egui_filter_layers::status(&ctx));

    assert_near(&pixels, 100, 50, [0, 255, 255], "red under the filter");
    assert_near(&pixels, 150, 50, [255, 255, 0], "blue under the filter");
    assert_near(&pixels, 20, 20, [255, 0, 255], "user texture under the filter");
    assert_near(&pixels, 70, 70, [127, 191, 223], "mid-tone under the filter (gamma handling)");
    assert_near(&pixels, 5, 5, [0, 255, 255], "top-left corner inside the filter (orientation)");
    assert_near(&pixels, 2, 2, [255, 0, 0], "outside the filter, left");
    assert_near(&pixels, 220, 50, [0, 0, 255], "outside the filter, right");
    assert_near(&pixels, 100, 110, [255, 0, 0], "outside the filter, below");
    assert_near(&pixels, 170, 70, [255, 255, 255], "painted after the filter");
    assert_near(&pixels, 220, 110, [255, 255, 0], "second layer over blue");
    assert_near(&pixels, 170, 95, [255, 255, 0], "second layer over the first: the first filter is not seen");
}

/// Frame 1 takes two passes (`request_discard` in pass 1). The font atlas and the user texture
/// are uploaded in full in pass 1 and are still missing from egui-wgpu's renderer in pass 2;
/// glyphs of a size never used before are rasterised only in pass 2 (a partial atlas update).
/// The filtered backdrop must still hold the solids (they sample the atlas' white texel), the
/// user texture and the text.
#[test]
#[ignore = "needs a GPU adapter (Vulkan / Metal / DX12 / GL); run with --ignored"]
fn a_multi_pass_first_frame_keeps_solids_textures_and_text() {
    let (ctx, state, texture, target) = setup(wgpu::TextureFormat::Rgba8Unorm);
    let text_rect = Rect::from_min_max(pos2(136.0, 8.0), pos2(200.0, 56.0));
    let (pixels, passes) = render_frame(&ctx, &state, &target, &mut |ui| {
        let second_pass = ui.ctx().output(|output| output.num_completed_passes) > 0;
        if !second_pass {
            ui.ctx().request_discard("test: multi-pass first frame");
        }
        let painter = ui.painter();
        painter.rect_filled(Rect::from_min_max(pos2(0.0, 0.0), pos2(128.0, 128.0)), 0.0, Color32::RED);
        painter.rect_filled(Rect::from_min_max(pos2(128.0, 0.0), pos2(256.0, 128.0)), 0.0, Color32::BLUE);
        painter.image(texture.id(), Rect::from_min_max(pos2(8.0, 8.0), pos2(40.0, 40.0)), Rect::from_min_max(Pos2::ZERO, pos2(1.0, 1.0)), Color32::WHITE);
        if second_pass {
            painter.text(text_rect.min, Align2::LEFT_TOP, "WWW", FontId::proportional(40.0), Color32::WHITE);
        }
        FilterLayer::new(Rect::from_min_max(pos2(4.0, 4.0), pos2(200.0, 100.0)), presets::invert(1.0)).paint(painter);
    });
    assert_eq!(passes, 2, "the frame must have been multi-pass");
    assert_near(&pixels, 100, 50, [0, 255, 255], "red solid under the filter");
    assert_near(&pixels, 150, 70, [255, 255, 0], "blue solid under the filter");
    assert_near(&pixels, 20, 20, [255, 0, 255], "user texture under the filter");
    // White glyphs over blue, inverted: dark pixels. Without the text in the backdrop the whole
    // rect would be inverted blue (yellow).
    let mut dark = 0;
    for y in 8..56 {
        for x in 136..200 {
            if rgb(&pixels, x, y).iter().all(|&channel| channel < 64) {
                dark += 1;
            }
        }
    }
    assert!(dark >= 100, "text under the filter: only {dark} dark pixels in {text_rect:?}");
}

/// A filter painted inside a zoomed layer (`Context::set_transform_layer`, what `egui::Scene`
/// does): effect distances in points (`fl.pixels_per_point`) must scale with the zoom, like the
/// capture padding. The shift effect reads 8 LAYER points to the right, so the red | blue edge
/// beneath (screen x = 128) shows up `8 * zoom` pixels to the left of where it is.
#[test]
#[ignore = "needs a GPU adapter (Vulkan / Metal / DX12 / GL); run with --ignored"]
fn effect_distances_follow_the_layer_zoom() {
    let (ctx, state, _texture, target) = setup(wgpu::TextureFormat::Rgba8Unorm);
    let shift = Effect::new("shift", "fn effect(uv: vec2<f32>) -> vec4<f32> { return src_px(vec2<f32>(params(0u).x * fl.pixels_per_point, 0.0)); }");
    let shift = match shift {
        Ok(effect) => effect.with_sample_radius(8.0),
        Err(error) => panic!("{error}"),
    };
    let Ok(pass) = Pass::new(&shift).with_param(0, [8.0, 0.0, 0.0, 0.0]) else { panic!("slot 0 exists") };
    let layer = egui::LayerId::new(egui::Order::Foreground, egui::Id::new("zoomed filter layer"));
    for zoom in [2.0_f32, 0.5] {
        let mut paint = |ui: &mut egui::Ui| {
            ui.ctx().set_transform_layer(layer, TSTransform::from_scaling(zoom));
            let painter = egui::Painter::new(ui.ctx().clone(), layer, Rect::EVERYTHING);
            // Screen points (1 pixel each here) -> this layer's own points.
            let at = |x: f32, y: f32| pos2(x / zoom, y / zoom);
            painter.rect_filled(Rect::from_min_max(at(0.0, 0.0), at(128.0, 128.0)), 0.0, Color32::RED);
            painter.rect_filled(Rect::from_min_max(at(128.0, 0.0), at(256.0, 128.0)), 0.0, Color32::BLUE);
            FilterLayer::new(Rect::from_min_max(at(16.0, 16.0), at(240.0, 112.0)), pass.clone()).paint(&painter);
        };
        render_frame(&ctx, &state, &target, &mut paint);
        let (pixels, _) = render_frame(&ctx, &state, &target, &mut paint);
        // Pixel x shows the backdrop at x + 8 * zoom: the edge moves to 128 - 8 * zoom (112 at zoom
        // 2, 124 at zoom 0.5). Ignoring the zoom (8 px) would put it at 120 in both cases.
        let edge = if zoom > 1.0 { 112 } else { 124 };
        assert_near(&pixels, edge - 2, 64, [255, 0, 0], &format!("zoom {zoom}: left of the shifted edge"));
        assert_near(&pixels, edge + 2, 64, [0, 0, 255], &format!("zoom {zoom}: right of the shifted edge"));
        assert_near(&pixels, 8, 64, [255, 0, 0], &format!("zoom {zoom}: outside the filter"));
    }
}

/// An effect that returns its input unchanged.
fn identity() -> Pass {
    match Effect::new("identity", "fn effect(uv: vec2<f32>) -> vec4<f32> { return src(uv); }") {
        Ok(effect) => Pass::new(&effect),
        Err(error) => panic!("{error}"),
    }
}

/// An upper rounded layer over a lower filter layer: outside its rounded mask (the corners of its
/// region) the screen must keep what is already there — the lower filter's output — and not the
/// upper layer's re-rendered backdrop, which lacks the lower filter (a skipped paint callback).
fn run_rounded_corner(format: wgpu::TextureFormat) {
    run_rounded_corner_with(format, egui_wgpu::RendererOptions { dithering: false, ..Default::default() });
}

/// [`run_rounded_corner`] with egui-wgpu configured by `renderer_options`: the filter's
/// pipelines must be compatible with that pass (wgpu would panic at `finish` otherwise).
fn run_rounded_corner_with(format: wgpu::TextureFormat, renderer_options: egui_wgpu::RendererOptions) {
    let (ctx, state, _texture, target) = setup_with(format, renderer_options, [0.0, 0.0, 0.0, 1.0]);
    let mut paint = |ui: &mut egui::Ui| {
        let painter = ui.painter();
        painter.rect_filled(Rect::from_min_max(pos2(0.0, 0.0), pos2(128.0, 128.0)), 0.0, Color32::RED);
        painter.rect_filled(Rect::from_min_max(pos2(128.0, 0.0), pos2(256.0, 128.0)), 0.0, Color32::BLUE);
        FilterLayer::new(Rect::from_min_max(pos2(4.0, 4.0), pos2(200.0, 100.0)), presets::invert(1.0)).paint(painter);
        FilterLayer::new(Rect::from_min_max(pos2(150.0, 60.0), pos2(250.0, 125.0)), identity()).corner_radius(16.0).paint(painter);
    };
    render_frame(&ctx, &state, &target, &mut paint);
    let (pixels, _) = render_frame(&ctx, &state, &target, &mut paint);
    assert_near(&pixels, 151, 61, [255, 255, 0], "upper layer's cut-off corner over the lower filter keeps the lower filter's output");
    assert_near(&pixels, 248, 123, [0, 0, 255], "upper layer's cut-off corner outside the lower filter");
    // Fully inside the mask the composite is exactly the filtered value (erase to 0, add 1 * f).
    assert_eq!(rgb(&pixels, 200, 110), [0, 0, 255], "inside the upper layer's mask");
    assert_near(&pixels, 100, 50, [0, 255, 255], "lower filter away from the upper layer");
    assert!(matches!(egui_filter_layers::status(&ctx), BackendStatus::Active { .. }), "{}", egui_filter_layers::status(&ctx));
}

#[test]
#[ignore = "needs a GPU adapter (Vulkan / Metal / DX12 / GL); run with --ignored"]
fn a_rounded_corner_keeps_what_is_not_re_rendered() {
    run_rounded_corner(wgpu::TextureFormat::Rgba8Unorm);
}

#[test]
#[ignore = "needs a GPU adapter (Vulkan / Metal / DX12 / GL); run with --ignored"]
fn a_rounded_corner_keeps_what_is_not_re_rendered_on_an_srgb_target() {
    run_rounded_corner(wgpu::TextureFormat::Rgba8UnormSrgb);
}

/// A semi-transparent shape over grey, half under an identity filter: the filtered half must
/// match egui's own blend on the unfiltered half. On a `*Srgb` target egui blends in linear, so
/// the backdrop re-render must blend in linear too.
fn run_semi_transparent(format: wgpu::TextureFormat) {
    let (ctx, state, _texture, target) = setup(format);
    let mut paint = |ui: &mut egui::Ui| {
        let painter = ui.painter();
        painter.rect_filled(Rect::from_min_max(pos2(0.0, 0.0), pos2(256.0, 128.0)), 0.0, Color32::from_gray(64));
        painter.rect_filled(Rect::from_min_max(pos2(0.0, 40.0), pos2(256.0, 90.0)), 0.0, Color32::from_rgba_unmultiplied(255, 255, 255, 128));
        FilterLayer::new(Rect::from_min_max(pos2(0.0, 0.0), pos2(128.0, 128.0)), identity()).paint(painter);
    };
    render_frame(&ctx, &state, &target, &mut paint);
    let (pixels, _) = render_frame(&ctx, &state, &target, &mut paint);
    let unfiltered = rgb(&pixels, 200, 64);
    let filtered = rgb(&pixels, 60, 64);
    if format.is_srgb() && !state.adapter.get_downlevel_capabilities().flags.contains(wgpu::DownlevelFlags::VIEW_FORMATS) {
        // Documented gap (GL / WebGL2): no view formats, the re-render blends in gamma.
        eprintln!("{format:?} without VIEW_FORMATS: filtered {filtered:?} vs egui's blend {unfiltered:?} (documented gap, not asserted)");
        return;
    }
    let close = filtered.iter().zip(unfiltered).all(|(f, u)| f.abs_diff(u) <= 2);
    assert!(close, "{format:?}: filtered {filtered:?} differs from egui's own blend {unfiltered:?}");
    let grey_unfiltered = rgb(&pixels, 200, 20);
    let grey_filtered = rgb(&pixels, 60, 20);
    assert!(grey_filtered.iter().zip(grey_unfiltered).all(|(f, u)| f.abs_diff(u) <= 1), "{format:?}: opaque grey {grey_filtered:?} vs {grey_unfiltered:?}");
}

#[test]
#[ignore = "needs a GPU adapter (Vulkan / Metal / DX12 / GL); run with --ignored"]
fn semi_transparent_edges_blend_like_the_screen() {
    run_semi_transparent(wgpu::TextureFormat::Rgba8Unorm);
}

#[test]
#[ignore = "needs a GPU adapter (Vulkan / Metal / DX12 / GL); run with --ignored"]
fn semi_transparent_edges_blend_like_the_screen_on_an_srgb_target() {
    run_semi_transparent(wgpu::TextureFormat::Rgba8UnormSrgb);
}

/// The blended composite in an MSAA pass with a depth/stencil attachment (what eframe builds for
/// `multisampling` / `depth_buffer` / `stencil_buffer`): the two composite pipelines must be
/// compatible with it, and the corner / interior results must not change.
#[test]
#[ignore = "needs a GPU adapter (Vulkan / Metal / DX12 / GL); run with --ignored"]
fn the_composite_matches_an_msaa_pass_with_depth_and_stencil() {
    let configs = [
        (wgpu::TextureFormat::Rgba8Unorm, 4, Some(wgpu::TextureFormat::Depth24PlusStencil8)),
        (wgpu::TextureFormat::Rgba8UnormSrgb, 4, Some(wgpu::TextureFormat::Depth32Float)),
        (wgpu::TextureFormat::Rgba8Unorm, 1, Some(wgpu::TextureFormat::Stencil8)),
    ];
    for (format, msaa_samples, depth_stencil_format) in configs {
        eprintln!("{format:?}, MSAA {msaa_samples}, {depth_stencil_format:?}");
        run_rounded_corner_with(format, egui_wgpu::RendererOptions { msaa_samples, depth_stencil_format, dithering: false, ..Default::default() });
    }
}

/// sRGB encoding of a linear value in 0..=1, as 0..=255.
fn encode_srgb(linear: f32) -> f32 {
    255.0 * egui::ecolor::gamma_from_linear(linear)
}

/// The blended composite on all four channels over a TRANSPARENT frame (a transparent eframe
/// window): a half-transparent grey backdrop (alpha 128) under a layer whose effect halves the
/// premultiplied colour (alpha 64). Inside the mask alpha must be the filtered 64 (a wrong erase
/// or add alpha factor gives 192 or 0), outside it the backdrop's 128, and on the partly covered
/// left edge the mix of both. The rect starts at x = 100.25: the processed region starts at the
/// rounded x = 100, whose pixel centre is 0.25 px inside, so its coverage is 0.75. The edge colour
/// is checked against a CPU reference: a gamma mix on a unorm target, a linear mix on an sRGB one.
fn run_alpha(format: wgpu::TextureFormat) {
    let (ctx, state, _texture, target) = setup_with(format, egui_wgpu::RendererOptions { dithering: false, ..Default::default() }, [0.0; 4]);
    let half = match Effect::new("half", "fn effect(uv: vec2<f32>) -> vec4<f32> { return 0.5 * src(uv); }") {
        Ok(effect) => Pass::new(&effect),
        Err(error) => panic!("{error}"),
    };
    let mut paint = |ui: &mut egui::Ui| {
        let painter = ui.painter();
        painter.rect_filled(Rect::from_min_max(pos2(0.0, 0.0), pos2(256.0, 128.0)), 0.0, Color32::from_rgba_unmultiplied(255, 255, 255, 128));
        FilterLayer::new(Rect::from_min_max(pos2(100.25, 20.0), pos2(200.0, 100.0)), half.clone()).paint(painter);
    };
    render_frame(&ctx, &state, &target, &mut paint);
    let (pixels, _) = render_frame(&ctx, &state, &target, &mut paint);
    let rgba = |x: u32, y: u32| {
        let i = usize::try_from((y * WIDTH + x) * 4).unwrap_or(0);
        [pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]]
    };
    let (outside, inside, edge) = (rgba(50, 50), rgba(150, 50), rgba(100, 50));
    eprintln!("{format:?}: outside {outside:?}, inside {inside:?}, edge {edge:?}");
    assert!(outside[3].abs_diff(128) <= 1, "{format:?}: alpha outside the layer {outside:?}");
    assert!(inside[3].abs_diff(64) <= 1, "{format:?}: alpha inside the layer {inside:?}");
    // 0.25 * 128 + 0.75 * 64.
    assert!(edge[3].abs_diff(80) <= 1, "{format:?}: alpha on the partly covered edge {edge:?}");
    // The edge colour against a CPU reference built from the read-back neighbours.
    let expected = if format.is_srgb() {
        let linear = |value: u8| egui::ecolor::linear_from_gamma(f32::from(value) / 255.0);
        encode_srgb(0.25 * linear(outside[0]) + 0.75 * linear(inside[0]))
    } else {
        0.25 * f32::from(outside[0]) + 0.75 * f32::from(inside[0])
    };
    // The composite does not depend on view formats, so this holds on GL too.
    assert!((f32::from(edge[0]) - expected).abs() <= 1.5, "{format:?}: edge colour {edge:?}, expected about {expected}");
}

#[test]
#[ignore = "needs a GPU adapter (Vulkan / Metal / DX12 / GL); run with --ignored"]
fn the_composite_blends_alpha_like_colour() {
    run_alpha(wgpu::TextureFormat::Rgba8Unorm);
}

#[test]
#[ignore = "needs a GPU adapter (Vulkan / Metal / DX12 / GL); run with --ignored"]
fn the_composite_blends_alpha_like_colour_on_an_srgb_target() {
    run_alpha(wgpu::TextureFormat::Rgba8UnormSrgb);
}

/// A pass built from `source` (panics on invalid WGSL: test effects are fixed).
fn pass_of(label: &str, source: &str) -> Pass {
    match Effect::new(label, source) {
        Ok(effect) => Pass::new(&effect),
        Err(error) => panic!("{label}: {error}"),
    }
}

/// F1: `original(uv)` reads the unfiltered backdrop in EVERY pass, whatever earlier passes did to
/// the image: `invert -> original` and `invert -> invert -> original` give the backdrop back, and
/// a pass reading both `src` (inverted) and `original` sees `src + original = 1`. A large layer
/// runs first, so the smaller one after it runs in grown (larger than its capture) textures,
/// where the capture's storage mapping differs from 1.
#[test]
#[ignore = "needs a GPU adapter (Vulkan / Metal / DX12 / GL); run with --ignored"]
fn original_reads_the_backdrop_in_every_pass() {
    let (ctx, state, texture, target) = setup(wgpu::TextureFormat::Rgba8Unorm);
    let original = pass_of("original", "fn effect(uv: vec2<f32>) -> vec4<f32> { return original(uv); }");
    let sum = pass_of("sum", "fn effect(uv: vec2<f32>) -> vec4<f32> { return vec4<f32>(abs(src(uv).rgb + original(uv).rgb - vec3<f32>(1.0)), 1.0); }");
    let invert = || FilterChain::from(presets::invert(1.0));
    let chains = [("invert, original", invert().then(original.clone()), false), ("invert, invert, original", invert().then(invert()).then(original.clone()), false), ("invert, sum", invert().then(sum), true)];
    let large = Rect::from_min_max(pos2(2.0, 2.0), pos2(254.0, 126.0));
    let small = Rect::from_min_max(pos2(20.0, 20.0), pos2(150.0, 100.0));
    for (name, chain, black) in &chains {
        for rect in [large, small] {
            let mut paint = |ui: &mut egui::Ui| {
                let painter = ui.painter();
                painter.rect_filled(Rect::from_min_max(pos2(0.0, 0.0), pos2(128.0, 128.0)), 0.0, Color32::RED);
                painter.rect_filled(Rect::from_min_max(pos2(128.0, 0.0), pos2(256.0, 128.0)), 0.0, Color32::BLUE);
                painter.rect_filled(Rect::from_min_max(pos2(60.0, 60.0), pos2(80.0, 80.0)), 0.0, Color32::from_rgb(128, 64, 32));
                painter.image(texture.id(), Rect::from_min_max(pos2(24.0, 24.0), pos2(40.0, 40.0)), Rect::from_min_max(Pos2::ZERO, pos2(1.0, 1.0)), Color32::WHITE);
                FilterLayer::new(rect, chain.clone()).paint(painter);
            };
            render_frame(&ctx, &state, &target, &mut paint);
            let (pixels, _) = render_frame(&ctx, &state, &target, &mut paint);
            assert!(matches!(egui_filter_layers::status(&ctx), BackendStatus::Active { .. }), "{}", egui_filter_layers::status(&ctx));
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

/// Runs one egui frame with `events` and nothing painted, only to change egui's input state
/// (the pointer persists into later frames); any texture upload goes to the renderer so later
/// frames stay consistent.
fn feed_input(ctx: &egui::Context, state: &egui_wgpu::RenderState, events: Vec<egui::Event>) {
    let input = egui::RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(256.0, 128.0))), events, ..Default::default() };
    let mut output = ctx.run_ui(input, |_| {});
    {
        let mut renderer = state.renderer.write();
        for (id, deltas) in &output.textures_delta.set {
            for delta in deltas {
                renderer.update_texture(&state.device, &state.queue, *id, delta);
            }
        }
    }
    output.textures_delta.clear();
}

/// F2 in the UI: an effect writing `fl.pointer` sees the pointer in capture uv with the pressed /
/// present flags, and one writing `fl.delta_time` / `fl.frame` sees egui's `stable_dt` (1/60 s
/// for frames without a time, clamped to 0.1) and a counter that advances per frame.
#[test]
#[ignore = "needs a GPU adapter (Vulkan / Metal / DX12 / GL); run with --ignored"]
fn effects_see_the_pointer_and_the_frame_counters() {
    let (ctx, state, _texture, target) = setup(wgpu::TextureFormat::Rgba8Unorm);
    let pointer = pass_of("pointer", "fn effect(uv: vec2<f32>) -> vec4<f32> { return vec4<f32>(fl.pointer.x, fl.pointer.y, fl.pointer.z, 1.0) * fl.pointer.w + vec4<f32>(0.0, 0.0, 0.0, 1.0 - fl.pointer.w); }");
    // Capture = region = (32, 16) .. (160, 80) pixels at 1 pixel per point (no padding).
    let rect = Rect::from_min_max(pos2(32.0, 16.0), pos2(160.0, 80.0));
    let mut paint_pointer = |ui: &mut egui::Ui| FilterLayer::new(rect, pointer.clone()).paint(ui.painter());
    render_frame(&ctx, &state, &target, &mut paint_pointer);
    let (pixels, _) = render_frame(&ctx, &state, &target, &mut paint_pointer);
    assert!(matches!(egui_filter_layers::status(&ctx), BackendStatus::Active { .. }), "{}", egui_filter_layers::status(&ctx));
    assert_near(&pixels, 64, 40, [0, 0, 0], "no pointer yet: w = 0");

    // Screen point (64, 32) = capture uv ((64 - 32) / 128, (32 - 16) / 64) = (0.25, 0.25).
    feed_input(&ctx, &state, vec![egui::Event::PointerMoved(pos2(64.0, 32.0))]);
    let (pixels, _) = render_frame(&ctx, &state, &target, &mut paint_pointer);
    assert_near(&pixels, 64, 40, [64, 64, 0], "hovering pointer at uv (0.25, 0.25)");
    feed_input(&ctx, &state, vec![egui::Event::PointerButton { pos: pos2(64.0, 32.0), button: egui::PointerButton::Primary, pressed: true, modifiers: egui::Modifiers::NONE }]);
    let (pixels, _) = render_frame(&ctx, &state, &target, &mut paint_pointer);
    assert_near(&pixels, 100, 70, [64, 64, 255], "pressed pointer: z = 1");

    // `fl.delta_time * 60` = 1 for egui's default 1/60 s step; `fl.frame` mod 2 alternates.
    let counters = pass_of("counters", "fn effect(uv: vec2<f32>) -> vec4<f32> { return vec4<f32>(fl.delta_time * 60.0, f32(fl.frame % 2u), 0.0, 1.0); }");
    let mut paint_counters = |ui: &mut egui::Ui| FilterLayer::new(rect, counters.clone()).paint(ui.painter());
    let (first, _) = render_frame(&ctx, &state, &target, &mut paint_counters);
    let (second, _) = render_frame(&ctx, &state, &target, &mut paint_counters);
    let [first, second] = [rgb(&first, 64, 40), rgb(&second, 64, 40)];
    assert!(first[0] >= 252 && second[0] >= 252, "delta_time = 1/60 s: {first:?} {second:?}");
    assert!(first[1].abs_diff(second[1]) >= 252, "fl.frame advances by one per frame: {first:?} {second:?}");
}
