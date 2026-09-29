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
//! Reduced-resolution passes (F6): a scaled identity chain matches the unscaled one on a smooth
//! backdrop; `kawase_blur` spreads a dot symmetrically and normalized; `bloom` glows around a
//! bright spot only. User textures (F8): a 2x2 checker read exactly, through a nearest and a
//! linear sampler, updated in full and in part between frames; three slots at once; unknown and
//! freed ids reported (`failed_textures`) and a missing attachment refused, each skipping only its
//! layer. Persistent state (F7): a `state + 0.25` state pass accumulates over frames (RGBA16F and
//! opt-in RGBA8), survives the per-frame target pool, is reset by `reset_state`, stepped once per
//! frame when the id is painted twice, stretched (not reset) when the layer grows past its
//! allocation, dropped by a re-install; a stateful layer without an id is refused.
//! Shape masks (F9): a circle filters inside and leaves everything outside untouched, incl. a lower
//! shader layer; the AA edge is monotonic; star, transformed-layer and text-galley masks; abutting
//! mask pieces leave no seam (additive coverage). Regions (R1): a 2x2 grid of abutting regions
//! tinted per `region()` fills its cells, switches tint within a pixel of every shared edge with
//! full coverage there and no `REGION_NONE` fringe; overlapping regions: the later one wins;
//! `region()` is `REGION_NONE` outside the regions and on a layer without regions.
//! Entries (R3): far-apart entries capture only their own rects (`fl.size_px`); overlapping
//! entries in `Independent` (later filters the backdrop) and `Stacked` (later filters the earlier
//! result, also at its antialiased edge) modes; three stacked entries with padded captures (the
//! hand-off case of roadmap §19.10 risk 1); outside the earlier mask a hand-off keeps the
//! backdrop byte for byte (antialiased content included); per-entry states and textures.
//!
//! All tests share one wgpu device (`GPU`, a test-only static; see there) and run in parallel on it.
//!
//! Needs a GPU adapter (Vulkan, Metal, DX12 or GL), so it is ignored by default:
//! `cargo test --all-features -- --ignored`.

#![cfg(feature = "wgpu")]

use std::sync::Arc;

use egui::{Align2, Color32, ColorImage, FontId, Pos2, Rect, TextureOptions, pos2, vec2};
use egui::emath::TSTransform;
use egui_shader_layers::{BackendStatus, Effect, Entry, FilterChain, Overlap, ShaderLayer, Pass, Region, StateFormat, StateOptions, WgpuOptions, install_wgpu, presets};

const WIDTH: u32 = 256;
const HEIGHT: u32 = 128;

/// The one wgpu instance, adapter, device and queue shared by every test of this file, created
/// on first use (`None` when there is no adapter or device, with the reason printed). Test-only
/// static: with the Khronos validation layer (enabled by wgpu in debug builds) on Mesa radv, one
/// device per test crashed about one run in ten (SIGSEGV inside `libvulkan` called from
/// `libVkLayer_khronos_validation`, e.g. in `vkSetDebugUtilsObjectNameEXT` of `create_bind_group`,
/// or in `vkCreateInstance`) while other tests created or destroyed their own instances. One
/// device keeps full validation and lets the tests still render in parallel: wgpu's handles are
/// `Send + Sync` and its error scopes are per thread.
static GPU: std::sync::OnceLock<Option<(wgpu::Instance, wgpu::Adapter, wgpu::Device, wgpu::Queue)>> = std::sync::OnceLock::new();

/// Handles to the shared GPU of [`GPU`], or `None` with the reason printed (once).
fn gpu() -> Option<(wgpu::Instance, wgpu::Adapter, wgpu::Device, wgpu::Queue)> {
    GPU.get_or_init(|| {
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
    })
    .clone()
}

/// Paints the scene of one frame into `ui`.
fn scene(ui: &egui::Ui, texture: &egui::TextureHandle) {
    let painter = ui.painter();
    painter.rect_filled(Rect::from_min_max(pos2(0.0, 0.0), pos2(128.0, 128.0)), 0.0, Color32::RED);
    painter.rect_filled(Rect::from_min_max(pos2(128.0, 0.0), pos2(256.0, 128.0)), 0.0, Color32::BLUE);
    painter.rect_filled(Rect::from_min_max(pos2(60.0, 60.0), pos2(80.0, 80.0)), 0.0, Color32::from_rgb(128, 64, 32));
    painter.image(texture.id(), Rect::from_min_max(pos2(8.0, 8.0), pos2(40.0, 40.0)), Rect::from_min_max(Pos2::ZERO, pos2(1.0, 1.0)), Color32::WHITE);
    ShaderLayer::new(Rect::from_min_max(pos2(4.0, 4.0), pos2(200.0, 100.0)), presets::invert(1.0)).paint(painter);
    // Painted after the filter: must stay untouched.
    painter.rect_filled(Rect::from_min_max(pos2(160.0, 60.0), pos2(180.0, 80.0)), 0.0, Color32::WHITE);
    // A second layer overlapping the first: on wgpu it sees the first one's UNFILTERED backdrop.
    ShaderLayer::new(Rect::from_min_max(pos2(150.0, 90.0), pos2(250.0, 125.0)), presets::invert(1.0)).paint(painter);
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
    // Frees are applied after the frame is rendered, as eframe does; egui asserts deltas are handled.
    let freed = std::mem::take(&mut textures_delta.free);
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
    let mut renderer = state.renderer.write();
    for id in &freed {
        renderer.free_texture(id);
    }
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
    assert!(matches!(egui_shader_layers::status(&ctx), BackendStatus::Active { .. }), "{}", egui_shader_layers::status(&ctx));

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
        ShaderLayer::new(Rect::from_min_max(pos2(4.0, 4.0), pos2(200.0, 100.0)), presets::invert(1.0)).paint(painter);
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
    let layer = egui::LayerId::new(egui::Order::Foreground, egui::Id::new("zoomed shader layer"));
    for zoom in [2.0_f32, 0.5] {
        let mut paint = |ui: &mut egui::Ui| {
            ui.ctx().set_transform_layer(layer, TSTransform::from_scaling(zoom));
            let painter = egui::Painter::new(ui.ctx().clone(), layer, Rect::EVERYTHING);
            // Screen points (1 pixel each here) -> this layer's own points.
            let at = |x: f32, y: f32| pos2(x / zoom, y / zoom);
            painter.rect_filled(Rect::from_min_max(at(0.0, 0.0), at(128.0, 128.0)), 0.0, Color32::RED);
            painter.rect_filled(Rect::from_min_max(at(128.0, 0.0), at(256.0, 128.0)), 0.0, Color32::BLUE);
            ShaderLayer::new(Rect::from_min_max(at(16.0, 16.0), at(240.0, 112.0)), pass.clone()).paint(&painter);
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

/// An upper rounded layer over a lower shader layer: outside its rounded mask (the corners of its
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
        ShaderLayer::new(Rect::from_min_max(pos2(4.0, 4.0), pos2(200.0, 100.0)), presets::invert(1.0)).paint(painter);
        ShaderLayer::new(Rect::from_min_max(pos2(150.0, 60.0), pos2(250.0, 125.0)), identity()).corner_radius(16.0).paint(painter);
    };
    render_frame(&ctx, &state, &target, &mut paint);
    let (pixels, _) = render_frame(&ctx, &state, &target, &mut paint);
    assert_near(&pixels, 151, 61, [255, 255, 0], "upper layer's cut-off corner over the lower filter keeps the lower filter's output");
    assert_near(&pixels, 248, 123, [0, 0, 255], "upper layer's cut-off corner outside the lower filter");
    // Fully inside the mask the composite is exactly the filtered value (erase to 0, add 1 * f).
    assert_eq!(rgb(&pixels, 200, 110), [0, 0, 255], "inside the upper layer's mask");
    assert_near(&pixels, 100, 50, [0, 255, 255], "lower filter away from the upper layer");
    assert!(matches!(egui_shader_layers::status(&ctx), BackendStatus::Active { .. }), "{}", egui_shader_layers::status(&ctx));
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
        ShaderLayer::new(Rect::from_min_max(pos2(0.0, 0.0), pos2(128.0, 128.0)), identity()).paint(painter);
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
        ShaderLayer::new(Rect::from_min_max(pos2(100.25, 20.0), pos2(200.0, 100.0)), half.clone()).paint(painter);
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
                ShaderLayer::new(rect, chain.clone()).paint(painter);
            };
            render_frame(&ctx, &state, &target, &mut paint);
            let (pixels, _) = render_frame(&ctx, &state, &target, &mut paint);
            assert!(matches!(egui_shader_layers::status(&ctx), BackendStatus::Active { .. }), "{}", egui_shader_layers::status(&ctx));
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
    let mut paint_pointer = |ui: &mut egui::Ui| ShaderLayer::new(rect, pointer.clone()).paint(ui.painter());
    render_frame(&ctx, &state, &target, &mut paint_pointer);
    let (pixels, _) = render_frame(&ctx, &state, &target, &mut paint_pointer);
    assert!(matches!(egui_shader_layers::status(&ctx), BackendStatus::Active { .. }), "{}", egui_shader_layers::status(&ctx));
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
    let mut paint_counters = |ui: &mut egui::Ui| ShaderLayer::new(rect, counters.clone()).paint(ui.painter());
    let (first, _) = render_frame(&ctx, &state, &target, &mut paint_counters);
    let (second, _) = render_frame(&ctx, &state, &target, &mut paint_counters);
    let [first, second] = [rgb(&first, 64, 40), rgb(&second, 64, 40)];
    assert!(first[0] >= 252 && second[0] >= 252, "delta_time = 1/60 s: {first:?} {second:?}");
    assert!(first[1].abs_diff(second[1]) >= 252, "fl.frame advances by one per frame: {first:?} {second:?}");
}

/// A named-parameter effect (F4) exercising every scalar kind, a vector after an array, and an
/// indexed array: `mix(src, color, amount) * tint * gain + lights[which] * (offset + 1)`.
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

/// The chain `a -> invert -> b` of [`NAMED_PARAMS_WGSL`] passes with different values, so each
/// pass must get its own block (and the invert the empty one). Over red it gives
/// `(64, 96, 64)`, over blue `(128, 96, 0)`.
fn named_params_chain() -> FilterChain {
    let effect = Effect::new("named", NAMED_PARAMS_WGSL).unwrap_or_else(|error| panic!("{error}"));
    let a = Pass::new(&effect).with("color", [0.0, 1.0, 0.0, 1.0]).and_then(|pass| pass.with("amount", 0.5)).and_then(|pass| pass.with("which", 2_u32)).and_then(|pass| pass.with("offset", 0)).and_then(|pass| pass.with("tint", [1.0, 0.5, 1.0])).and_then(|pass| pass.with("gain", 1.0));
    let mut a = a.unwrap_or_else(|error| panic!("{error}"));
    a.set_slice("lights", &[[9.0_f32; 4], [9.0; 4], [0.0, 0.0, 0.5, 0.0]]).unwrap_or_else(|error| panic!("{error}"));
    let b = Pass::new(&effect).with("tint", [1.0, 1.0, 1.0]).and_then(|pass| pass.with("gain", 0.5)).unwrap_or_else(|error| panic!("{error}"));
    FilterChain::from(a).then(presets::invert(1.0)).then(b)
}

/// F4: named parameters reach the shader at naga's layout, per pass, through binding 4.
#[test]
#[ignore = "needs a GPU adapter (Vulkan / Metal / DX12 / GL); run with --ignored"]
fn named_params_reach_each_pass() {
    let (ctx, state, _texture, target) = setup(wgpu::TextureFormat::Rgba8Unorm);
    let chain = named_params_chain();
    let mut paint = |ui: &mut egui::Ui| {
        let painter = ui.painter();
        painter.rect_filled(Rect::from_min_max(pos2(0.0, 0.0), pos2(128.0, 128.0)), 0.0, Color32::RED);
        painter.rect_filled(Rect::from_min_max(pos2(128.0, 0.0), pos2(256.0, 128.0)), 0.0, Color32::BLUE);
        ShaderLayer::new(Rect::from_min_max(pos2(20.0, 20.0), pos2(236.0, 108.0)), chain.clone()).paint(painter);
    };
    render_frame(&ctx, &state, &target, &mut paint);
    let (pixels, _) = render_frame(&ctx, &state, &target, &mut paint);
    assert!(matches!(egui_shader_layers::status(&ctx), BackendStatus::Active { .. }), "{}", egui_shader_layers::status(&ctx));
    assert!(egui_shader_layers::failed_effects(&ctx).is_empty(), "{:?}", egui_shader_layers::failed_effects(&ctx));
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

/// Compares a `click_ripples` frame with the same layer without impulses (pixel-identical
/// backdrop pass-through): pixels in the ring band around (128, 64) must move, pixels farther
/// than two wavelengths from the front (the window's compact support) must not.
fn check_click_ripples(rippled: &[u8], still: &[u8]) {
    let (mut moved_in_band, mut moved_elsewhere) = (0_usize, Vec::new());
    // Inside the layer rect (40, 4) .. (216, 124), keeping 2 px off its edges.
    for y in 6..122_u32 {
        for x in 42..214_u32 {
            let dist = (f32::from(u16::try_from(x).unwrap_or(0)) + 0.5 - 128.0).hypot(f32::from(u16::try_from(y).unwrap_or(0)) + 0.5 - 64.0);
            let [a, b] = [rgb(rippled, x, y), rgb(still, x, y)];
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

/// F3: `presets::click_ripples` driven by `Impulses` — one impulse at the layer centre, 1 s old,
/// travelling 40 points per second — displaces the ring around its front and leaves the centre
/// and the far field untouched.
#[test]
#[ignore = "needs a GPU adapter (Vulkan / Metal / DX12 / GL); run with --ignored"]
fn click_ripples_displace_the_ring_only() {
    let (ctx, state, _texture, target) = setup(wgpu::TextureFormat::Rgba8Unorm);
    let rect = Rect::from_min_max(pos2(40.0, 4.0), pos2(216.0, 124.0));
    let mut impulses = egui_shader_layers::Impulses::new(4.0);
    assert!(impulses.push_point(pos2(128.0, 64.0), rect, 1.0, 0.0));
    let rippled = presets::click_ripples(&impulses, 1.0, 6.0, 8.0, 40.0);
    let still = presets::click_ripples(&egui_shader_layers::Impulses::new(4.0), 1.0, 6.0, 8.0, 40.0);
    let frame = |pass: &Pass| {
        let mut paint = |ui: &mut egui::Ui| {
            ripple_stripes(ui.painter());
            ShaderLayer::new(rect, pass.clone()).paint(ui.painter());
        };
        render_frame(&ctx, &state, &target, &mut paint);
        render_frame(&ctx, &state, &target, &mut paint).0
    };
    let (rippled, still) = (frame(&rippled), frame(&still));
    assert!(matches!(egui_shader_layers::status(&ctx), BackendStatus::Active { .. }), "{}", egui_shader_layers::status(&ctx));
    assert!(egui_shader_layers::failed_effects(&ctx).is_empty(), "{:?}", egui_shader_layers::failed_effects(&ctx));
    assert_near(&still, 130, 64, [255, 0, 0], "no impulses: the stripes pass through");
    check_click_ripples(&rippled, &still);
}

// ---- F6: scaled passes, kawase blur, bloom ----

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

/// Checks a blurred 8x8 white dot centred on the pixel corner (128, 64) over black: the blur is
/// mirror-symmetric about the dot's centre in both axes, spreads beyond the dot, keeps its
/// energy (the kernel is normalized), and reaches no further than `reach` pixels.
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

/// Paints black, an 8x8 white dot at (124, 60) .. (132, 68), and `chain` over (32, 4) .. (224, 124).
fn dot_scene(painter: &egui::Painter, chain: &FilterChain, background: Color32) {
    painter.rect_filled(Rect::from_min_max(pos2(0.0, 0.0), pos2(256.0, 128.0)), 0.0, background);
    painter.rect_filled(Rect::from_min_max(pos2(124.0, 60.0), pos2(132.0, 68.0)), 0.0, Color32::WHITE);
    ShaderLayer::new(Rect::from_min_max(pos2(32.0, 4.0), pos2(224.0, 124.0)), chain.clone()).paint(painter);
}

/// F6: an identity pass at half resolution (upsampled by the composite) over a smooth backdrop
/// equals the full-resolution identity within 3 steps, over an odd-sized layer (rounding of the
/// scaled size); a half / quarter / full chain too.
#[test]
#[ignore = "needs a GPU adapter (Vulkan / Metal / DX12 / GL); run with --ignored"]
fn a_scaled_identity_chain_matches_the_unscaled_one() {
    let (ctx, state, _texture, target) = setup(wgpu::TextureFormat::Rgba8Unorm);
    let identity = Pass::new(&Effect::new("id", "fn effect(uv: vec2<f32>) -> vec4<f32> { return src(uv); }").unwrap_or_else(|error| panic!("{error}")));
    // 217 x 109 pixels: odd sizes, scaled sizes rounded up.
    let rect = Rect::from_min_max(pos2(17.0, 9.0), pos2(234.0, 118.0));
    let frame = |chain: &FilterChain| {
        let mut paint = |ui: &mut egui::Ui| {
            smooth_ramp(ui.painter());
            ShaderLayer::new(rect, chain.clone()).paint(ui.painter());
        };
        render_frame(&ctx, &state, &target, &mut paint);
        render_frame(&ctx, &state, &target, &mut paint).0
    };
    let full = frame(&FilterChain::from(identity.clone()));
    let half = frame(&FilterChain::from(identity.clone().with_scale(0.5)));
    let pyramid = frame(&FilterChain::from(vec![identity.clone().with_scale(0.5), identity.clone().with_scale(0.25), identity.clone()]));
    assert!(matches!(egui_shader_layers::status(&ctx), BackendStatus::Active { .. }), "{}", egui_shader_layers::status(&ctx));
    assert!(egui_shader_layers::failed_effects(&ctx).is_empty(), "{:?}", egui_shader_layers::failed_effects(&ctx));
    let inside = [19, 11, 232, 116];
    let (worst, at) = max_diff_in(&full, &half, inside);
    assert!(worst <= 3, "half resolution: max diff {worst} at {at:?}");
    let (worst, at) = max_diff_in(&full, &pyramid, inside);
    assert!(worst <= 4, "half / quarter / full: max diff {worst} at {at:?}");
    assert_near(&half, 5, 5, rgb(&full, 5, 5), "outside the layer");
}

/// F6: `presets::kawase_blur` spreads a bright dot symmetrically, normalized, within its reach.
#[test]
#[ignore = "needs a GPU adapter (Vulkan / Metal / DX12 / GL); run with --ignored"]
fn a_kawase_blur_spreads_a_dot_symmetrically() {
    let (ctx, state, _texture, target) = setup(wgpu::TextureFormat::Rgba8Unorm);
    let chain = presets::kawase_blur(16.0);
    let mut paint = |ui: &mut egui::Ui| dot_scene(ui.painter(), &chain, Color32::BLACK);
    render_frame(&ctx, &state, &target, &mut paint);
    let (pixels, _) = render_frame(&ctx, &state, &target, &mut paint);
    assert!(matches!(egui_shader_layers::status(&ctx), BackendStatus::Active { .. }), "{}", egui_shader_layers::status(&ctx));
    assert!(egui_shader_layers::failed_effects(&ctx).is_empty(), "{:?}", egui_shader_layers::failed_effects(&ctx));
    check_blurred_dot(&pixels, 16, "kawase_blur(16)");
}

/// F6: `presets::bloom` adds light around a bright spot, keeps the spot, and leaves dark areas
/// (below the threshold, beyond the reach) and the outside of the layer untouched.
#[test]
#[ignore = "needs a GPU adapter (Vulkan / Metal / DX12 / GL); run with --ignored"]
fn bloom_brightens_around_a_bright_spot_only() {
    let (ctx, state, _texture, target) = setup(wgpu::TextureFormat::Rgba8Unorm);
    let dark = Color32::from_gray(40);
    let chain = presets::bloom(0.5, 1.0, 16.0);
    let mut paint = |ui: &mut egui::Ui| dot_scene(ui.painter(), &chain, dark);
    render_frame(&ctx, &state, &target, &mut paint);
    let (pixels, _) = render_frame(&ctx, &state, &target, &mut paint);
    assert!(matches!(egui_shader_layers::status(&ctx), BackendStatus::Active { .. }), "{}", egui_shader_layers::status(&ctx));
    assert!(egui_shader_layers::failed_effects(&ctx).is_empty(), "{:?}", egui_shader_layers::failed_effects(&ctx));
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

// ---- F8: user textures (same scenes as `glow_headless.rs`) ----

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

/// Red | blue halves split at x = 128.
fn halves(painter: &egui::Painter) {
    painter.rect_filled(Rect::from_min_max(pos2(0.0, 0.0), pos2(128.0, 128.0)), 0.0, Color32::RED);
    painter.rect_filled(Rect::from_min_max(pos2(128.0, 0.0), pos2(256.0, 128.0)), 0.0, Color32::BLUE);
}

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

/// F8: a 2x2 checker attached twice, NEAREST at slot 0 and LINEAR at slot 1 (each texture's egui
/// `TextureOptions` is its sampler: egui-wgpu's bind group is bound as the slot's group), read
/// exactly (`tex_load`), through the nearest sampler (`tex`) and bilinearly; then replaced in
/// full (egui-wgpu applies that after the pass: the crate's kept copy serves it) and patched in
/// part between frames, which the next frame must show.
#[test]
#[ignore = "needs a GPU adapter (Vulkan / Metal / DX12 / GL); run with --ignored"]
fn user_textures_are_sampled_exactly_and_linearly() {
    let (ctx, state, _texture, target) = setup(wgpu::TextureFormat::Rgba8Unorm);
    let mut nearest = ctx.load_texture("checker nearest", checker(), TextureOptions::NEAREST);
    let linear = ctx.load_texture("checker linear", checker(), TextureOptions::LINEAR);
    let (nearest_id, linear_id) = (nearest.id(), linear.id());
    let screen = Rect::from_min_max(pos2(0.0, 0.0), pos2(256.0, 128.0));
    let frames = |snippet: &str, count: usize| {
        let effect = pass_of("textured", snippet);
        let mut paint = |ui: &mut egui::Ui| {
            halves(ui.painter());
            ShaderLayer::new(screen, effect.clone()).texture(0, nearest_id).texture(1, linear_id).paint(ui.painter());
        };
        (1..count).for_each(|_| drop(render_frame(&ctx, &state, &target, &mut paint)));
        render_frame(&ctx, &state, &target, &mut paint).0
    };
    // The first frame uploads the textures in full: the kept copies serve them.
    check_checker_quadrants(&frames(TEX_EXACT, 1), "exact, first frame");
    check_checker_quadrants(&frames(TEX_EXACT, 2), "exact");
    check_checker_quadrants(&frames(TEX_SAMPLED, 2), "nearest sampler");
    check_checker_linear(&frames(TEX_SLOT1, 2), "linear sampler");
    assert!(matches!(egui_shader_layers::status(&ctx), BackendStatus::Active { .. }), "{}", egui_shader_layers::status(&ctx));
    assert!(egui_shader_layers::failed_textures(&ctx).is_empty(), "{:?}", egui_shader_layers::failed_textures(&ctx));

    nearest.set(ColorImage::new([2, 2], vec![Color32::YELLOW; 4]), TextureOptions::NEAREST);
    let pixels = frames(TEX_EXACT, 1);
    assert_eq!(rgb(&pixels, 192, 96), [255, 255, 0], "after a full update");
    nearest.set_partial([1, 1], ColorImage::new([1, 1], vec![Color32::BLACK]), TextureOptions::NEAREST);
    let pixels = frames(TEX_EXACT, 1);
    assert_eq!((rgb(&pixels, 192, 96), rgb(&pixels, 64, 32)), ([0, 0, 0], [255, 255, 0]), "after a partial update");
}

/// F8: three slots at once (groups 1..=3), each its own texture and size: `(tex(0).r, tex(1).g,
/// tex(2).b)` on the left half, `tex_size` of the three (2x2, 4x4, 3x1) times 32 on the right.
#[test]
#[ignore = "needs a GPU adapter (Vulkan / Metal / DX12 / GL); run with --ignored"]
fn three_texture_slots_at_once() {
    let (ctx, state, green, target) = setup(wgpu::TextureFormat::Rgba8Unorm);
    let checker = ctx.load_texture("checker", checker(), TextureOptions::NEAREST);
    let blue = ctx.load_texture("blue 3x1", ColorImage::new([3, 1], vec![Color32::from_rgb(0, 0, 100); 3]), TextureOptions::LINEAR);
    let ids = [checker.id(), green.id(), blue.id()];
    let channels = pass_of("channels", "fn effect(uv: vec2<f32>) -> vec4<f32> { return vec4<f32>(tex(0u, vec2<f32>(0.25)).r, tex(1u, uv).g, tex(2u, uv).b, 1.0); }");
    let sizes = pass_of("sizes", "fn effect(uv: vec2<f32>) -> vec4<f32> { return vec4<f32>(vec3<f32>(tex_size(0u).x, tex_size(1u).y, tex_size(2u).x) * 32.0 / 255.0, 1.0); }");
    let mut paint = |ui: &mut egui::Ui| {
        halves(ui.painter());
        let attach = |layer: ShaderLayer| ids.iter().enumerate().fold(layer, |layer, (slot, id)| layer.texture(slot, *id));
        attach(ShaderLayer::new(Rect::from_min_max(pos2(0.0, 0.0), pos2(128.0, 128.0)), channels.clone())).paint(ui.painter());
        attach(ShaderLayer::new(Rect::from_min_max(pos2(128.0, 0.0), pos2(256.0, 128.0)), sizes.clone())).paint(ui.painter());
    };
    render_frame(&ctx, &state, &target, &mut paint);
    let (pixels, _) = render_frame(&ctx, &state, &target, &mut paint);
    assert!(matches!(egui_shader_layers::status(&ctx), BackendStatus::Active { .. }), "{}", egui_shader_layers::status(&ctx));
    assert_near(&pixels, 64, 64, [255, 255, 100], "one channel from each slot");
    assert_near(&pixels, 192, 64, [64, 128, 96], "each slot's own size");
}

/// F8: a layer attaching an id egui-wgpu does not know draws nothing extra and the id is
/// reported (`failed_textures`) without disabling the backend or any effect; a texture freed by
/// egui is reported the same way once the free was applied; a layer whose effect reads a slot
/// with nothing attached is refused before the backend (not a backend failure).
#[test]
#[ignore = "needs a GPU adapter (Vulkan / Metal / DX12 / GL); run with --ignored"]
fn unknown_freed_and_missing_textures_skip_their_layer_only() {
    let (ctx, state, _texture, target) = setup(wgpu::TextureFormat::Rgba8Unorm);
    let unknown = egui::TextureId::User(424_242);
    let temporary = ctx.load_texture("temporary", ColorImage::new([1, 1], vec![Color32::WHITE]), TextureOptions::NEAREST);
    let freed = temporary.id();
    let mut temporary = Some(temporary);
    let sampled = pass_of("sampled", TEX_SAMPLED);
    let mut paint = |ui: &mut egui::Ui| {
        let painter = ui.painter();
        halves(painter);
        ShaderLayer::new(Rect::from_min_max(pos2(8.0, 8.0), pos2(56.0, 120.0)), sampled.clone()).texture(0, unknown).paint(painter);
        ShaderLayer::new(Rect::from_min_max(pos2(72.0, 8.0), pos2(120.0, 120.0)), sampled.clone()).texture(0, freed).paint(painter);
        // Nothing attached although the effect reads slot 0.
        ShaderLayer::new(Rect::from_min_max(pos2(136.0, 8.0), pos2(184.0, 120.0)), sampled.clone()).paint(painter);
        ShaderLayer::new(Rect::from_min_max(pos2(200.0, 8.0), pos2(248.0, 120.0)), presets::invert(1.0)).paint(painter);
    };
    render_frame(&ctx, &state, &target, &mut paint);
    let (pixels, _) = render_frame(&ctx, &state, &target, &mut paint);
    assert_near(&pixels, 96, 60, [255, 255, 255], "the live texture is drawn");
    // Dropping the handle frees the texture with the next frame's delta (after its rendering).
    temporary.take();
    render_frame(&ctx, &state, &target, &mut paint);
    let (pixels, _) = render_frame(&ctx, &state, &target, &mut paint);
    assert_near(&pixels, 30, 60, [255, 0, 0], "unknown id: the backdrop stays");
    assert_near(&pixels, 96, 60, [255, 0, 0], "freed id: the backdrop stays");
    assert_near(&pixels, 160, 60, [0, 0, 255], "missing attachment: the backdrop stays");
    assert_near(&pixels, 220, 60, [255, 255, 0], "a healthy layer still filters");
    assert!(matches!(egui_shader_layers::status(&ctx), BackendStatus::Active { .. }), "{}", egui_shader_layers::status(&ctx));
    assert!(egui_shader_layers::failed_effects(&ctx).is_empty(), "{:?}", egui_shader_layers::failed_effects(&ctx));
    let mut failed: Vec<egui::TextureId> = egui_shader_layers::failed_textures(&ctx).into_iter().map(|failure| failure.id).collect();
    failed.sort_by_key(|id| format!("{id:?}"));
    let mut expected = vec![unknown, freed];
    expected.sort_by_key(|id| format!("{id:?}"));
    assert_eq!(failed, expected);
}

/// F7: a state pass adding 0.25 to every state channel per frame (exact texel reads), and a display
/// pass showing the state (sampled at texel centres: exact).
fn accumulator() -> FilterChain {
    let step = match Effect::new_state("accumulate", "fn effect(uv: vec2<f32>) -> vec4<f32> { return state_px(vec2<i32>(0)) + vec4<f32>(0.25); }") {
        Ok(effect) => Pass::new(&effect),
        Err(error) => panic!("{error}"),
    };
    FilterChain::new().then(step).then(pass_of("show", "fn effect(uv: vec2<f32>) -> vec4<f32> { return min(state(uv), vec4<f32>(1.0)); }"))
}

/// The value (0..=255) a layer of [`accumulator`] shows after `frames` steps.
fn accumulated(frames: u32) -> u8 {
    [0, 64, 128, 191, 255].get(usize::try_from(frames).unwrap_or(usize::MAX)).copied().unwrap_or(255)
}

/// Checks the grey of pixel `(x, y)` within one step of `expected`.
fn assert_grey(pixels: &[u8], x: u32, y: u32, expected: u8, what: &str) {
    let actual = rgb(pixels, x, y);
    assert!(actual.iter().all(|value| value.abs_diff(expected) <= 1), "{what}: pixel ({x}, {y}) is {actual:?}, expected grey {expected}");
}

/// F7: over four frames the state accumulates 0.25 per frame (surviving the per-frame target pool
/// reset), then `reset_state` starts it over (even across an off-screen paint that does not run);
/// the same id painted twice in a frame is stepped once
/// (both paints show the same value); a layer without an id is refused (backdrop kept).
#[test]
#[ignore = "needs a GPU adapter (Vulkan / Metal / DX12 / GL); run with --ignored"]
fn state_accumulates_resets_and_steps_once_per_frame() {
    let (ctx, state, _texture, target) = setup(wgpu::TextureFormat::Rgba8Unorm);
    let chain = accumulator();
    let id = egui::Id::new("accumulator");
    let left = Rect::from_min_max(pos2(8.0, 8.0), pos2(120.0, 120.0));
    let right = Rect::from_min_max(pos2(136.0, 8.0), pos2(248.0, 60.0));
    let unnamed = Rect::from_min_max(pos2(136.0, 70.0), pos2(248.0, 120.0));
    let mut paint = |ui: &mut egui::Ui| {
        let painter = ui.painter();
        halves(painter);
        ShaderLayer::new(left, chain.clone()).id(id).paint(painter);
        // The same id again in the same frame: shows the state, does not step it.
        ShaderLayer::new(right, chain.clone()).id(id).paint(painter);
        ShaderLayer::new(unnamed, chain.clone()).paint(painter);
    };
    for frame in 1..=4 {
        let (pixels, _) = render_frame(&ctx, &state, &target, &mut paint);
        assert_grey(&pixels, 60, 60, accumulated(frame), &format!("frame {frame}, first paint"));
        assert_grey(&pixels, 190, 30, accumulated(frame), &format!("frame {frame}, second paint of the id"));
        assert_near(&pixels, 190, 100, [0, 0, 255], &format!("frame {frame}: no id, refused"));
    }
    assert!(matches!(egui_shader_layers::status(&ctx), BackendStatus::Active { .. }), "{}", egui_shader_layers::status(&ctx));
    assert_eq!(egui_shader_layers::state_count(&ctx), 1);
    egui_shader_layers::reset_state(&ctx, id);
    // A paint that never runs (the layer off screen, no plan) must leave the request pending.
    let offscreen = Rect::from_min_max(pos2(1000.0, 1000.0), pos2(1100.0, 1100.0));
    render_frame(&ctx, &state, &target, &mut |ui: &mut egui::Ui| ShaderLayer::new(offscreen, chain.clone()).id(id).paint(ui.painter()));
    let (pixels, _) = render_frame(&ctx, &state, &target, &mut paint);
    assert_grey(&pixels, 60, 60, accumulated(1), "after reset_state (requested before an off-screen paint)");
    assert!(egui_shader_layers::failed_states(&ctx).is_empty() && egui_shader_layers::failed_effects(&ctx).is_empty());
}

/// F7: an explicit RGBA8 state accumulates the same way (0.25 steps are exact enough in 8 bits).
#[test]
#[ignore = "needs a GPU adapter (Vulkan / Metal / DX12 / GL); run with --ignored"]
fn an_rgba8_state_accumulates_too() {
    let (ctx, state, _texture, target) = setup(wgpu::TextureFormat::Rgba8Unorm);
    let chain = accumulator();
    let mut paint = |ui: &mut egui::Ui| {
        halves(ui.painter());
        ShaderLayer::new(Rect::from_min_max(pos2(8.0, 8.0), pos2(120.0, 120.0)), chain.clone()).id(egui::Id::new("rgba8")).state(StateOptions::default().with_format(StateFormat::Rgba8Unorm)).paint(ui.painter());
    };
    for frame in 1..=4 {
        let (pixels, _) = render_frame(&ctx, &state, &target, &mut paint);
        assert_grey(&pixels, 60, 60, accumulated(frame), &format!("frame {frame}"));
    }
}

/// F7: a layer that grows past its state allocation continues its state (the previous state is
/// read stretched, the textures grow keeping their content), at a reduced state scale too; a
/// re-install drops every state.
#[test]
#[ignore = "needs a GPU adapter (Vulkan / Metal / DX12 / GL); run with --ignored"]
fn a_growing_layer_keeps_its_state_and_a_reinstall_drops_it() {
    let (ctx, state, _texture, target) = setup(wgpu::TextureFormat::Rgba8Unorm);
    let chain = accumulator();
    for options in [StateOptions::default(), StateOptions::default().with_scale(0.5)] {
        let id = egui::Id::new(("growing", options.scale().to_bits()));
        let mut rect = Rect::from_min_max(pos2(8.0, 8.0), pos2(40.0, 24.0));
        let paint = |ui: &mut egui::Ui, rect: Rect| {
            halves(ui.painter());
            ShaderLayer::new(rect, chain.clone()).id(id).state(options).paint(ui.painter());
        };
        for frame in 1..=3 {
            if frame == 3 {
                // Far beyond the first allocation.
                rect = Rect::from_min_max(pos2(4.0, 4.0), pos2(250.0, 124.0));
            }
            let (pixels, _) = render_frame(&ctx, &state, &target, &mut |ui| paint(ui, rect));
            assert_grey(&pixels, 20, 15, accumulated(frame), &format!("scale {}, frame {frame}", options.scale()));
            if frame == 3 {
                assert_grey(&pixels, 200, 100, accumulated(frame), "the grown part reads the stretched previous state");
            }
        }
    }
    assert_eq!(egui_shader_layers::state_count(&ctx), 2);
    let installed = install_wgpu(&ctx, &state, WgpuOptions::from(&egui_wgpu::RendererOptions { dithering: false, ..Default::default() }).with_clear_color([0.0, 0.0, 0.0, 1.0]));
    assert!(installed.is_ok(), "{installed:?}");
    let id = egui::Id::new(("growing", 1.0_f32.to_bits()));
    let (pixels, _) = render_frame(&ctx, &state, &target, &mut |ui| {
        halves(ui.painter());
        ShaderLayer::new(Rect::from_min_max(pos2(8.0, 8.0), pos2(40.0, 24.0)), chain.clone()).id(id).paint(ui.painter());
    });
    assert_grey(&pixels, 20, 15, accumulated(1), "a re-install starts every state over");
    assert_eq!(egui_shader_layers::state_count(&ctx), 1);
}

// ---- F9: shape masks ----

/// The F9 test effect: paints its region opaque green, so the composite's coverage is visible
/// as green over what the screen held.
const GREEN_WGSL: &str = "fn effect(uv: vec2<f32>) -> vec4<f32> { return vec4<f32>(0.0, 1.0, 0.0, 1.0); }";

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

/// The circle scene: red | blue halves, a lower invert layer, and an upper green layer masked by
/// a circle (centre (215.3, 100.5), radius 20) that overlaps the lower layer's corner region.
fn circle_mask_scene(painter: &egui::Painter, green: &Pass) {
    halves(painter);
    ShaderLayer::new(Rect::from_min_max(pos2(4.0, 4.0), pos2(200.0, 100.0)), presets::invert(1.0)).paint(painter);
    let circle = egui::Shape::circle_filled(pos2(215.3, 100.5), 20.0, Color32::WHITE);
    ShaderLayer::new(Rect::from_min_max(pos2(150.0, 60.0), pos2(250.0, 125.0)), green.clone()).mask(egui_shader_layers::Mask::Shape(circle)).paint(painter);
}

/// Inside the circle: green; outside it (also inside the layer rect): exactly what the screen
/// held, including the lower filter's output in the upper layer's corner; along the row through
/// the centre green falls and blue rises monotonically through at least one anti-aliased pixel.
fn check_circle_mask(pixels: &[u8], what: &str) {
    assert_near(pixels, 215, 100, [0, 255, 0], &format!("{what}: circle centre"));
    assert_near(pixels, 215, 84, [0, 255, 0], &format!("{what}: inside the circle, over the lower filter's region"));
    assert_near(pixels, 152, 62, [255, 255, 0], &format!("{what}: the upper layer's corner outside the circle keeps the lower filter"));
    assert_near(pixels, 245, 66, [0, 0, 255], &format!("{what}: outside the circle, outside the lower filter"));
    assert_near(pixels, 190, 100, [0, 0, 255], &format!("{what}: left of the circle, below the lower filter"));
    assert_near(pixels, 100, 50, [0, 255, 255], &format!("{what}: the lower filter away from the upper layer"));
    let row: Vec<[u8; 3]> = (215..=245).map(|x| rgb(pixels, x, 100)).collect();
    assert!(row.windows(2).all(|pair| pair[1][1] <= pair[0][1] && pair[1][2] >= pair[0][2]), "{what}: the edge must be monotonic: {row:?}");
    assert!(row.iter().any(|pixel| (8..248).contains(&pixel[1])), "{what}: an anti-aliased edge pixel: {row:?}");
    assert!(row.iter().all(|pixel| pixel[0] <= 3), "{what}: no red on the row: {row:?}");
}

/// The star scene: red | blue halves and a green layer over the left half masked by a star.
fn star_mask_scene(painter: &egui::Painter, green: &Pass) {
    halves(painter);
    ShaderLayer::new(Rect::from_min_max(pos2(4.0, 4.0), pos2(124.0, 124.0)), green.clone()).mask(egui_shader_layers::Mask::Shape(star(pos2(64.0, 64.0), 56.0, 22.0))).paint(painter);
}

/// Green at the centre and along a tip, red in a notch and outside the layer.
fn check_star_mask(pixels: &[u8], what: &str) {
    assert_near(pixels, 64, 64, [0, 255, 0], &format!("{what}: star centre"));
    assert_near(pixels, 63, 24, [0, 255, 0], &format!("{what}: along the top tip"));
    // The tip at -18 degrees (right-upper), 40 points out.
    assert_near(pixels, 101, 51, [0, 255, 0], &format!("{what}: along the right tip"));
    // The notch at -54 degrees, 40 points out (the star's concave part).
    assert_near(pixels, 87, 31, [255, 0, 0], &format!("{what}: in a notch"));
    assert_near(pixels, 10, 120, [255, 0, 0], &format!("{what}: a layer corner outside the star"));
    assert_near(pixels, 2, 2, [255, 0, 0], &format!("{what}: outside the layer"));
}

/// A shape mask in a transformed layer (translate (20, 10), scale 2): layer points follow the
/// layer transform like the layer rect, so the circle (55, 30) r 15 lands at screen (130, 70) r 30.
fn transformed_mask_scene(ui: &egui::Ui, green: &Pass) {
    halves(ui.painter());
    let layer = egui::LayerId::new(egui::Order::Foreground, egui::Id::new("masked zoomed layer"));
    ui.ctx().set_transform_layer(layer, TSTransform::new(vec2(20.0, 10.0), 2.0));
    let painter = egui::Painter::new(ui.ctx().clone(), layer, Rect::EVERYTHING);
    let circle = egui::Shape::circle_filled(pos2(55.0, 30.0), 15.0, Color32::WHITE);
    ShaderLayer::new(Rect::from_min_max(pos2(10.0, 10.0), pos2(100.0, 50.0)), green.clone()).mask(egui_shader_layers::Mask::Shape(circle)).paint(&painter);
}

fn check_transformed_mask(pixels: &[u8], what: &str) {
    assert_near(pixels, 130, 70, [0, 255, 0], &format!("{what}: mapped centre"));
    assert_near(pixels, 154, 70, [0, 255, 0], &format!("{what}: 24 px right (inside the scaled radius 30)"));
    assert_near(pixels, 108, 70, [0, 255, 0], &format!("{what}: 22 px left"));
    assert_near(pixels, 166, 70, [0, 0, 255], &format!("{what}: 36 px right: outside the circle, inside the layer rect"));
    assert_near(pixels, 45, 35, [255, 0, 0], &format!("{what}: the layer rect's corner"));
    assert_near(pixels, 130, 20, [0, 0, 255], &format!("{what}: above the layer (blue half)"));
}

/// A text mask ("MASK", 64-point monospace, UI only): green only where the glyphs are.
fn text_mask_scene(ui: &egui::Ui, green: &Pass) -> Rect {
    let painter = ui.painter();
    halves(painter);
    let galley = painter.layout_no_wrap("MASK".to_owned(), egui::FontId::monospace(64.0), Color32::WHITE);
    let origin = pos2(24.0, 24.0);
    let text_rect = galley.rect.translate(origin.to_vec2());
    ShaderLayer::new(Rect::from_min_max(pos2(8.0, 8.0), pos2(248.0, 120.0)), green.clone()).mask(egui_shader_layers::Mask::Shape(egui::Shape::galley(origin, galley, Color32::WHITE))).paint(painter);
    text_rect
}

/// Many green pixels inside the text's rect, none outside it.
fn check_text_mask(pixels: &[u8], text_rect: Rect, what: &str) {
    let greenish = |x: u32, y: u32| rgb(pixels, x, y)[1] > 16;
    let inside = |x: u32, y: u32| text_rect.expand(2.0).contains(pos2(f32::from(u16::try_from(x).unwrap_or(0)) + 0.5, f32::from(u16::try_from(y).unwrap_or(0)) + 0.5));
    let mut green_inside = 0;
    for y in 8..120 {
        for x in 8..248 {
            if !greenish(x, y) {
                continue;
            }
            assert!(inside(x, y), "{what}: green at ({x}, {y}) outside the text rect {text_rect:?}");
            green_inside += 1;
        }
    }
    assert!(green_inside > 300, "{what}: only {green_inside} glyph pixels filtered");
}

/// Two convex pieces meeting edge to edge at x = 60.5 (mid-pixel, so both feathers are 0.5
/// there): the coverage union is additive (roadmap §19 D1), so the shared column is fully covered.
fn abutting_pieces_scene(painter: &egui::Painter, green: &Pass) {
    halves(painter);
    let piece = |x0: f32, x1: f32| egui::Shape::convex_polygon(vec![pos2(x0, 20.0), pos2(x1, 20.0), pos2(x1, 100.0), pos2(x0, 100.0)], Color32::WHITE, egui::Stroke::NONE);
    let pieces = egui::Shape::Vec(vec![piece(20.0, 60.5), piece(60.5, 100.0)]);
    ShaderLayer::new(Rect::from_min_max(pos2(10.0, 10.0), pos2(110.0, 110.0)), green.clone()).mask(egui_shader_layers::Mask::Shape(pieces)).paint(painter);
}

/// No seam on the shared edge (egui's "over" union would leave 0.75 coverage: about
/// `[64, 191, 0]` over the red half); the outer edges stay anti-aliased / untouched.
fn check_abutting_pieces(pixels: &[u8], what: &str) {
    for y in [30, 60, 90] {
        for x in 58..=62 {
            assert_near(pixels, x, y, [0, 255, 0], &format!("{what}: shared edge column {x}, row {y}"));
        }
    }
    assert_near(pixels, 15, 60, [255, 0, 0], &format!("{what}: left of the pieces"));
    assert_near(pixels, 105, 60, [255, 0, 0], &format!("{what}: right of the pieces"));
}

/// F9 on wgpu: a circle mask filters inside only; outside it the BLENDED composite discards, so
/// the screen keeps what egui drew there — including a lower shader layer's output in the upper
/// layer's corner, which the re-rendered backdrop lacks; the anti-aliased edge is monotonic.
#[test]
#[ignore = "needs a GPU adapter (Vulkan / Metal / DX12 / GL); run with --ignored"]
fn a_circle_mask_filters_inside_and_keeps_the_screen_outside() {
    let green = pass_of("green", GREEN_WGSL);
    for format in [wgpu::TextureFormat::Rgba8Unorm, wgpu::TextureFormat::Rgba8UnormSrgb] {
        let (ctx, state, _texture, target) = setup(format);
        render_frame(&ctx, &state, &target, &mut |ui| circle_mask_scene(ui.painter(), &green));
        let (pixels, _) = render_frame(&ctx, &state, &target, &mut |ui| circle_mask_scene(ui.painter(), &green));
        check_circle_mask(&pixels, &format!("{format:?}"));
        assert!(matches!(egui_shader_layers::status(&ctx), BackendStatus::Active { .. }), "{}", egui_shader_layers::status(&ctx));
    }
}

/// F9 on wgpu: a star built from convex paths (concave outline), a mask in a transformed layer
/// (scale and translation), and a text galley mask (the font atlas is sampled by the rasterizer).
#[test]
#[ignore = "needs a GPU adapter (Vulkan / Metal / DX12 / GL); run with --ignored"]
fn star_transformed_and_text_masks() {
    let green = pass_of("green", GREEN_WGSL);
    let (ctx, state, _texture, target) = setup(wgpu::TextureFormat::Rgba8Unorm);
    render_frame(&ctx, &state, &target, &mut |ui| star_mask_scene(ui.painter(), &green));
    let (pixels, _) = render_frame(&ctx, &state, &target, &mut |ui| star_mask_scene(ui.painter(), &green));
    check_star_mask(&pixels, "star");
    render_frame(&ctx, &state, &target, &mut |ui| transformed_mask_scene(ui, &green));
    let (pixels, _) = render_frame(&ctx, &state, &target, &mut |ui| transformed_mask_scene(ui, &green));
    check_transformed_mask(&pixels, "transformed");
    let mut text_rect = Rect::NOTHING;
    for _ in 0..2 {
        let (pixels, _) = render_frame(&ctx, &state, &target, &mut |ui| text_rect = text_mask_scene(ui, &green));
        check_text_mask(&pixels, text_rect, "text");
    }
    assert!(egui_shader_layers::failed_textures(&ctx).is_empty(), "{:?}", egui_shader_layers::failed_textures(&ctx));
    assert!(matches!(egui_shader_layers::status(&ctx), BackendStatus::Active { .. }), "{}", egui_shader_layers::status(&ctx));
}

/// D1 on wgpu: abutting mask pieces leave no seam (additive coverage union).
#[test]
#[ignore = "needs a GPU adapter (Vulkan / Metal / DX12 / GL); run with --ignored"]
fn abutting_mask_pieces_leave_no_seam() {
    let green = pass_of("green", GREEN_WGSL);
    let (ctx, state, _texture, target) = setup(wgpu::TextureFormat::Rgba8Unorm);
    render_frame(&ctx, &state, &target, &mut |ui| abutting_pieces_scene(ui.painter(), &green));
    let (pixels, _) = render_frame(&ctx, &state, &target, &mut |ui| abutting_pieces_scene(ui.painter(), &green));
    check_abutting_pieces(&pixels, "wgpu");
}

// ---- R1: regions ----

/// Paints `p.tint[region()]` (opaque), white where `region()` is `REGION_NONE`: a white tint in
/// any covered pixel is a sentinel fringe.
const REGION_TINT_WGSL: &str = "struct Params { tint: array<vec4<f32>, 4> }
fn effect(uv: vec2<f32>) -> vec4<f32> {
    let r = region();
    if r == REGION_NONE { return vec4<f32>(1.0); }
    return p.tint[min(r, 3u)];
}";

/// The region tints by id: green, blue, cyan, magenta.
const REGION_TINTS: [[u8; 3]; 4] = [[0, 255, 0], [0, 0, 255], [0, 255, 255], [255, 0, 255]];

/// The tint pass of [`REGION_TINT_WGSL`] with [`REGION_TINTS`].
fn region_tint_pass() -> Pass {
    let mut pass = pass_of("region tint", REGION_TINT_WGSL);
    let tints: Vec<[f32; 4]> = REGION_TINTS.iter().map(|tint| [f32::from(tint[0]) / 255.0, f32::from(tint[1]) / 255.0, f32::from(tint[2]) / 255.0, 1.0]).collect();
    if let Err(error) = pass.set_slice("tint", &tints) {
        panic!("{error}");
    }
    pass
}

/// A convex quad region `[x0, x1] x [y0, y1]` (points; a path, so half-pixel edges stay where
/// they are instead of being rounded to pixels like a rect).
fn quad_region(x0: f32, y0: f32, x1: f32, y1: f32) -> Region {
    Region::shape(egui::Shape::convex_polygon(vec![pos2(x0, y0), pos2(x1, y0), pos2(x1, y1), pos2(x0, y1)], Color32::WHITE, egui::Stroke::NONE))
}

/// Red | blue halves and a layer over the red half split into a 2x2 grid of abutting regions,
/// all edges at half pixels (x, y in {20.5, 60.5, 100.5}): ids 0 top-left, 1 top-right,
/// 2 bottom-left, 3 bottom-right.
fn region_grid_scene(painter: &egui::Painter, pass: &Pass) {
    halves(painter);
    let (a, b, c) = (20.5, 60.5, 100.5);
    let cells = [quad_region(a, a, b, b), quad_region(b, a, c, b), quad_region(a, b, b, c), quad_region(b, b, c, c)];
    ShaderLayer::new(Rect::from_min_max(pos2(10.0, 10.0), pos2(110.0, 110.0)), pass.clone()).regions(cells).paint(painter);
}

/// [`check_region_cells`], and the outer antialiased edge mixes the backdrop with the cell's tint,
/// never with the sentinel white (checked in gamma: only on a non-sRGB target).
fn check_region_grid(pixels: &[u8], what: &str) {
    check_region_cells(pixels, what);
    // Outer edges (partly covered): the backdrop red mixed with the cell's tint by the coverage
    // `c`, i.e. `(255 (1 - c), 255 c, 0)` for green and `(255 (1 - c), 255 c, 255 c)` for cyan;
    // the white sentinel would give `(255, 255 c, 255 c)`. Row 60 is where the two left cells'
    // outer corners meet: each corner feather is 0.5 there, so the additive union covers it
    // fully (egui's own corner over-estimate, doubled).
    for y in 25..=95 {
        let [red, green, blue] = rgb(pixels, 20, y);
        let consistent = if y <= 59 { blue <= 3 } else { green.abs_diff(blue) <= 3 };
        assert!(consistent && (u16::from(red) + u16::from(green)).abs_diff(255) <= 6, "{what}: left outer edge (20, {y}) = {:?}: no sentinel fringe", [red, green, blue]);
        assert!(green >= 100, "{what}: left outer edge (20, {y}) = {:?}: at least half covered", [red, green, blue]);
        assert_near(pixels, 19, y, [255, 0, 0], &format!("{what}: left of the regions"));
    }
    for y in 25..=55 {
        assert!(rgb(pixels, 100, y)[1] <= 3, "{what}: right outer edge (100, {y}) = {:?}: no sentinel fringe", rgb(pixels, 100, y));
    }
    assert_near(pixels, 105, 60, [255, 0, 0], &format!("{what}: inside the layer rect, outside every region"));
}

/// Each cell holds its tint; on a shared edge (mid-pixel: both feathers 0.5, coverage 1 by the
/// additive union and the composite's snap of the 8-bit sum) the tint switches from one cell's to
/// the LATER cell's within one pixel with no backdrop showing, on gamma and sRGB targets alike;
/// the four-cell junction pixel belongs to the last cell.
fn check_region_cells(pixels: &[u8], what: &str) {
    for (x, y, id) in [(40, 40, 0), (80, 40, 1), (40, 80, 2), (80, 80, 3)] {
        assert_near(pixels, x, y, REGION_TINTS[id], &format!("{what}: centre of region {id}"));
    }
    // Vertical shared edge at x = 60.5: column 60 is shared, the later (right) region owns it.
    for (y, left, right) in [(40, 0, 1), (80, 2, 3)] {
        for x in 55..=65 {
            let expected = if x <= 59 { REGION_TINTS[left] } else { REGION_TINTS[right] };
            assert_near(pixels, x, y, expected, &format!("{what}: vertical edge, row {y}"));
        }
    }
    // Horizontal shared edge at y = 60.5: row 60 belongs to the later (lower) region.
    for (x, top, bottom) in [(40, 0, 2), (80, 1, 3)] {
        for y in 55..=65 {
            let expected = if y <= 59 { REGION_TINTS[top] } else { REGION_TINTS[bottom] };
            assert_near(pixels, x, y, expected, &format!("{what}: horizontal edge, column {x}"));
        }
    }
    assert_near(pixels, 60, 60, REGION_TINTS[3], &format!("{what}: the junction pixel belongs to the last region"));
}

/// R1 on wgpu: abutting regions tint their own cells and own their edges.
#[test]
#[ignore = "needs a GPU adapter (Vulkan / Metal / DX12 / GL); run with --ignored"]
fn regions_tint_each_cell_and_own_their_edges() {
    let pass = region_tint_pass();
    for format in [wgpu::TextureFormat::Rgba8Unorm, wgpu::TextureFormat::Rgba8UnormSrgb] {
        let (ctx, state, _texture, target) = setup(format);
        render_frame(&ctx, &state, &target, &mut |ui| region_grid_scene(ui.painter(), &pass));
        let (pixels, _) = render_frame(&ctx, &state, &target, &mut |ui| region_grid_scene(ui.painter(), &pass));
        // The sRGB target blends partly covered pixels in linear, so the outer edge is not
        // checked there in gamma; the shared edges must show no backdrop on both (the composite
        // snaps the 8-bit coverage sum of two half feathers, 254/255, to full coverage).
        if format.is_srgb() {
            check_region_cells(&pixels, &format!("{format:?}"));
        } else {
            check_region_grid(&pixels, &format!("{format:?}"));
        }
        assert!(matches!(egui_shader_layers::status(&ctx), BackendStatus::Active { .. }), "{}", egui_shader_layers::status(&ctx));
    }
}

/// Paints red where `region()` is `REGION_NONE`, green elsewhere.
const REGION_NONE_WGSL: &str = "fn effect(uv: vec2<f32>) -> vec4<f32> { return select(vec4<f32>(0.0, 1.0, 0.0, 1.0), vec4<f32>(1.0, 0.0, 0.0, 1.0), region() == REGION_NONE); }";

/// R1 on wgpu: a circle region over the blue half is green inside, the screen (blue) outside it,
/// and its antialiased edge mixes blue and green only (no pixel inside the mask reads
/// `REGION_NONE`); the same effect on a layer without regions is red everywhere (the dummy).
#[test]
#[ignore = "needs a GPU adapter (Vulkan / Metal / DX12 / GL); run with --ignored"]
fn region_none_outside_regions() {
    let pass = pass_of("region none", REGION_NONE_WGSL);
    let (ctx, state, _texture, target) = setup(wgpu::TextureFormat::Rgba8Unorm);
    let rect = Rect::from_min_max(pos2(140.0, 20.0), pos2(240.0, 100.0));
    let scene = |painter: &egui::Painter| {
        halves(painter);
        ShaderLayer::new(rect, pass.clone()).regions([Region::shape(egui::Shape::circle_filled(pos2(190.3, 60.3), 30.0, Color32::WHITE))]).paint(painter);
    };
    render_frame(&ctx, &state, &target, &mut |ui| scene(ui.painter()));
    let (pixels, _) = render_frame(&ctx, &state, &target, &mut |ui| scene(ui.painter()));
    assert_near(&pixels, 190, 60, [0, 255, 0], "circle centre");
    assert_near(&pixels, 145, 25, [0, 0, 255], "layer corner outside the circle: masked out");
    for y in 20..100 {
        for x in 140..240 {
            let [red, _, _] = rgb(&pixels, x, y);
            assert!(red <= 3, "({x}, {y}) = {:?}: a covered pixel read REGION_NONE", rgb(&pixels, x, y));
        }
    }
    let row: Vec<[u8; 3]> = (190..=225).map(|x| rgb(&pixels, x, 60)).collect();
    assert!(row.iter().any(|pixel| (8..248).contains(&pixel[1])), "an antialiased edge pixel: {row:?}");

    let plain = |painter: &egui::Painter| {
        halves(painter);
        ShaderLayer::new(rect, pass.clone()).paint(painter);
    };
    render_frame(&ctx, &state, &target, &mut |ui| plain(ui.painter()));
    let (pixels, _) = render_frame(&ctx, &state, &target, &mut |ui| plain(ui.painter()));
    for (x, y) in [(145, 25), (190, 60), (235, 95)] {
        assert_near(&pixels, x, y, [255, 0, 0], "a layer without regions reads REGION_NONE everywhere");
    }
    assert!(matches!(egui_shader_layers::status(&ctx), BackendStatus::Active { .. }), "{}", egui_shader_layers::status(&ctx));
}

/// R1 on wgpu: where regions overlap, the LATER one owns the pixels: a small region after a big
/// one keeps its own tint inside the big one; listed before it, it disappears into the big one.
#[test]
#[ignore = "needs a GPU adapter (Vulkan / Metal / DX12 / GL); run with --ignored"]
fn overlapping_regions_later_wins() {
    let pass = region_tint_pass();
    let (ctx, state, _texture, target) = setup(wgpu::TextureFormat::Rgba8Unorm);
    let big = Region::rounded_rect(Rect::from_min_max(pos2(20.0, 20.0), pos2(100.0, 100.0)), 0);
    let small = Region::rounded_rect(Rect::from_min_max(pos2(40.0, 40.0), pos2(80.0, 80.0)), 0);
    for (regions, centre, what) in [([big.clone(), small.clone()], REGION_TINTS[1], "small after big"), ([small, big], REGION_TINTS[1], "small before big")] {
        let scene = |painter: &egui::Painter| {
            halves(painter);
            ShaderLayer::new(Rect::from_min_max(pos2(10.0, 10.0), pos2(110.0, 110.0)), pass.clone()).regions(regions.clone()).paint(painter);
        };
        render_frame(&ctx, &state, &target, &mut |ui| scene(ui.painter()));
        let (pixels, _) = render_frame(&ctx, &state, &target, &mut |ui| scene(ui.painter()));
        assert_near(&pixels, 60, 60, centre, &format!("{what}: centre (id 1)"));
        let corner = if what == "small after big" { REGION_TINTS[0] } else { REGION_TINTS[1] };
        assert_near(&pixels, 30, 30, corner, &format!("{what}: the big region's corner"));
        // Hard boundary at the small region's edge (x = 40) when it is on top.
        if what == "small after big" {
            assert_near(&pixels, 39, 60, REGION_TINTS[0], &format!("{what}: left of the small region"));
            assert_near(&pixels, 40, 60, REGION_TINTS[1], &format!("{what}: first column of the small region"));
        }
    }
}

// ---- R3: several entries in one layer ----

/// The uniform backdrop of the entry scenes.
const ENTRY_BACKDROP: Color32 = Color32::from_rgb(100, 150, 200);
/// `ENTRY_BACKDROP` inverted.
const ENTRY_INVERTED: [u8; 3] = [155, 105, 55];
/// `ENTRY_BACKDROP` tinted halfway to red ([`entry_tint`]).
const ENTRY_TINTED: [u8; 3] = [178, 75, 100];
/// `ENTRY_INVERTED` tinted halfway to red.
const ENTRY_TINTED_INVERTED: [u8; 3] = [205, 53, 28];

/// Mixes the input halfway to opaque red (a pointwise effect: its result depends only on the
/// pixel it reads).
fn entry_tint() -> Pass {
    pass_of("half red", "fn effect(uv: vec2<f32>) -> vec4<f32> { let c = src(uv); return vec4<f32>(mix(c.rgb, vec3<f32>(1.0, 0.0, 0.0), 0.5), c.a); }")
}

/// Paints the uniform entry backdrop over the whole screen.
fn entry_backdrop(painter: &egui::Painter) {
    painter.rect_filled(Rect::from_min_max(pos2(0.0, 0.0), pos2(256.0, 128.0)), 0.0, ENTRY_BACKDROP);
}

/// Renders `layer` over the entry backdrop for two frames (the first uploads the font atlas) and
/// returns the second frame's pixels.
fn render_entries(layer: &ShaderLayer) -> Vec<u8> {
    let (ctx, state, _texture, target) = setup(wgpu::TextureFormat::Rgba8Unorm);
    let mut paint = |ui: &mut egui::Ui| {
        entry_backdrop(ui.painter());
        layer.clone().paint(ui.painter());
    };
    render_frame(&ctx, &state, &target, &mut paint);
    let (pixels, _) = render_frame(&ctx, &state, &target, &mut paint);
    assert!(matches!(egui_shader_layers::status(&ctx), BackendStatus::Active { .. }), "{}", egui_shader_layers::status(&ctx));
    assert!(egui_shader_layers::failed_effects(&ctx).is_empty(), "{:?}", egui_shader_layers::failed_effects(&ctx));
    pixels
}

/// R3: two far-apart entries with different chains filter only their own rects, the pixels
/// between them stay the backdrop, and each captures only ITS OWN padded rect: an effect writing
/// `fl.size_px` (the capture size) sees 32x32 and 48x24, not the 240-pixel-wide union.
#[test]
#[ignore = "needs a GPU adapter (Vulkan / Metal / DX12 / GL); run with --ignored"]
fn two_entries_far_apart_capture_only_their_windows() {
    let left = Rect::from_min_max(pos2(8.0, 8.0), pos2(40.0, 40.0));
    let right = Rect::from_min_max(pos2(200.0, 16.0), pos2(248.0, 40.0));
    let pixels = render_entries(&ShaderLayer::from_entries([Entry::new(left, presets::invert(1.0)), Entry::new(right, entry_tint())]));
    assert_near(&pixels, 20, 20, ENTRY_INVERTED, "left entry: invert");
    assert_near(&pixels, 220, 30, ENTRY_TINTED, "right entry: tint");
    for (x, y) in [(120, 30), (4, 4), (220, 60), (100, 100)] {
        assert_near(&pixels, x, y, [100, 150, 200], "outside every entry");
    }
    let size = |label: &str, blue: f32| pass_of(label, &format!("fn effect(uv: vec2<f32>) -> vec4<f32> {{ return vec4<f32>(fl.size_px / 255.0, {blue:.1}, 1.0); }}"));
    let chain_right = FilterChain::from(presets::invert(1.0)).then(size("size b", 1.0));
    let pixels = render_entries(&ShaderLayer::from_entries([Entry::new(left, size("size a", 0.0)), Entry::new(right, chain_right)]));
    assert_eq!(rgb(&pixels, 20, 20), [32, 32, 0], "the left entry's capture is its own 32x32 rect");
    assert_eq!(rgb(&pixels, 220, 30), [48, 24, 255], "the right entry's capture is its own 48x24 rect");
}

/// The overlap scene: entry 0 inverts `(20, 20)..(120, 100)`, entry 1 tints `(80.25, 40)..(200,
/// 110)` (its left edge a quarter pixel into column 80: coverage 0.75 there).
fn overlapping_entries(overlap: Overlap) -> ShaderLayer {
    let first = Rect::from_min_max(pos2(20.0, 20.0), pos2(120.0, 100.0));
    let second = Rect::from_min_max(pos2(80.25, 40.0), pos2(200.0, 110.0));
    ShaderLayer::from_entries([Entry::new(first, presets::invert(1.0)), Entry::new(second, entry_tint())]).overlap(overlap)
}

/// `a + (b - a) * t` per channel, rounded.
fn mix_rgb(a: [u8; 3], b: [u8; 3], t: f32) -> [u8; 3] {
    std::array::from_fn(|c| {
        let value = f32::from(a[c]) + (f32::from(b[c]) - f32::from(a[c])) * t;
        // In 0..=255 by construction.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let rounded = value.round() as u8;
        rounded
    })
}

/// R3 `Independent`: where the entries overlap the later one filters the ORIGINAL backdrop
/// (tint, not tint of invert); its antialiased edge blends over the earlier entry's result.
#[test]
#[ignore = "needs a GPU adapter (Vulkan / Metal / DX12 / GL); run with --ignored"]
fn overlapping_entries_independent_later_wins() {
    let pixels = render_entries(&overlapping_entries(Overlap::Independent));
    assert_near(&pixels, 40, 50, ENTRY_INVERTED, "entry 0 only");
    assert_near(&pixels, 150, 105, ENTRY_TINTED, "entry 1 only");
    assert_near(&pixels, 100, 70, ENTRY_TINTED, "overlap: the later entry filters the backdrop");
    assert_near(&pixels, 80, 70, mix_rgb(ENTRY_INVERTED, ENTRY_TINTED, 0.75), "entry 1's AA edge over entry 0's result");
    assert_near(&pixels, 230, 120, [100, 150, 200], "outside");
}

/// R3 `Stacked`: the later entry filters the earlier one's result (the hand-off), also at its
/// antialiased edge.
#[test]
#[ignore = "needs a GPU adapter (Vulkan / Metal / DX12 / GL); run with --ignored"]
fn overlapping_entries_stacked_compose() {
    let pixels = render_entries(&overlapping_entries(Overlap::Stacked));
    assert_near(&pixels, 40, 50, ENTRY_INVERTED, "entry 0 only");
    assert_near(&pixels, 150, 105, ENTRY_TINTED, "entry 1 only: the backdrop");
    assert_near(&pixels, 100, 70, ENTRY_TINTED_INVERTED, "overlap: tint of the inverted backdrop");
    assert_near(&pixels, 80, 70, mix_rgb(ENTRY_INVERTED, ENTRY_TINTED_INVERTED, 0.75), "entry 1's AA edge over entry 0's result");
    assert_near(&pixels, 230, 120, [100, 150, 200], "outside");
}

/// R3 risk 1 (roadmap §19.10): three `Stacked` entries where entry 0 overlaps entry 2 but not
/// entry 1, and entry 1 overlaps entry 2; entries 0 and 2 are padded (sample radii), so the
/// hand-off viewports are clipped to the later padded capture. Entry 2 sees both inverts; in
/// `Independent` it sees neither.
#[test]
#[ignore = "needs a GPU adapter (Vulkan / Metal / DX12 / GL); run with --ignored"]
fn three_stacked_entries_hand_off_where_they_overlap() {
    let layer = |overlap| {
        ShaderLayer::from_entries([
            Entry::new(Rect::from_min_max(pos2(10.0, 10.0), pos2(60.0, 60.0)), presets::invert(1.0).with_sample_radius(3.0)),
            Entry::new(Rect::from_min_max(pos2(150.0, 10.0), pos2(200.0, 60.0)), presets::invert(1.0)),
            Entry::new(Rect::from_min_max(pos2(40.0, 30.0), pos2(170.0, 50.0)), entry_tint().with_sample_radius(4.0)),
        ])
        .overlap(overlap)
    };
    let stacked = render_entries(&layer(Overlap::Stacked));
    assert_near(&stacked, 20, 20, ENTRY_INVERTED, "entry 0 only");
    assert_near(&stacked, 190, 20, ENTRY_INVERTED, "entry 1 only");
    assert_near(&stacked, 100, 40, ENTRY_TINTED, "entry 2 only");
    assert_near(&stacked, 50, 40, ENTRY_TINTED_INVERTED, "entries 0 and 2");
    assert_near(&stacked, 160, 40, ENTRY_TINTED_INVERTED, "entries 1 and 2");
    assert_near(&stacked, 100, 100, [100, 150, 200], "outside");
    // Just outside entry 0's region but inside entry 2's padded capture: the backdrop, handed
    // off as entry 0's original there.
    assert_near(&stacked, 62, 40, ENTRY_TINTED, "entry 2 beside entry 0");
    let independent = render_entries(&layer(Overlap::Independent));
    assert_near(&independent, 50, 40, ENTRY_TINTED, "independent: entry 2 over entry 0");
    assert_near(&independent, 160, 40, ENTRY_TINTED, "independent: entry 2 over entry 1");
}

/// R3 (§19.4.2): outside the earlier entry's MASK a `Stacked` hand-off writes the earlier
/// entry's own re-render of the backdrop into the later capture; it must equal the later entry's
/// re-render there, also on antialiased content (a circle and text straddling the overlap, drawn
/// at fractional positions). So `Stacked` equals `Independent` byte for byte wherever the earlier
/// mask covers nothing.
#[test]
#[ignore = "needs a GPU adapter (Vulkan / Metal / DX12 / GL); run with --ignored"]
fn a_hand_off_outside_the_earlier_mask_keeps_the_backdrop() {
    let layer = |overlap| {
        let small = egui::Shape::circle_filled(pos2(30.0, 30.0), 6.0, Color32::WHITE);
        ShaderLayer::from_entries([
            Entry::new(Rect::from_min_max(pos2(20.0, 20.0), pos2(140.0, 110.0)), presets::invert(1.0)).mask(egui_shader_layers::Mask::Shape(small)),
            Entry::new(Rect::from_min_max(pos2(60.0, 30.0), pos2(200.0, 120.0)), entry_tint().with_sample_radius(2.0)),
        ])
        .overlap(overlap)
    };
    let render = |overlap| {
        let (ctx, state, _texture, target) = setup(wgpu::TextureFormat::Rgba8Unorm);
        let mut paint = |ui: &mut egui::Ui| {
            let painter = ui.painter();
            entry_backdrop(painter);
            painter.circle_filled(pos2(100.3, 70.7), 23.4, Color32::from_rgb(250, 240, 20));
            painter.text(pos2(66.6, 88.2), Align2::LEFT_TOP, "Hand-off", FontId::proportional(13.3), Color32::BLACK);
            layer(overlap).paint(painter);
        };
        render_frame(&ctx, &state, &target, &mut paint);
        render_frame(&ctx, &state, &target, &mut paint).0
    };
    let (stacked, independent) = (render(Overlap::Stacked), render(Overlap::Independent));
    let (differing, max) = (60..140_u32).flat_map(|x| (30..110_u32).map(move |y| (x, y))).fold((0, 0), |(count, max), (x, y)| {
        let diff = rgb(&stacked, x, y).iter().zip(rgb(&independent, x, y)).map(|(a, b)| a.abs_diff(b)).max().unwrap_or(0);
        (count + usize::from(diff > 0), max.max(diff))
    });
    assert_eq!(differing, 0, "the overlap outside entry 0's mask: {differing} pixels differ, max {max}");
    assert_near(&stacked, 30, 30, ENTRY_INVERTED, "entry 0's mask is still inverted");
}

/// R3 (D5): stateful entries of one layer (one id) keep separate states — two accumulators at
/// different rates each show their own count; `reset_state` of the layer id restarts both.
#[test]
#[ignore = "needs a GPU adapter (Vulkan / Metal / DX12 / GL); run with --ignored"]
fn stateful_entries_keep_separate_states() {
    let (ctx, state, _texture, target) = setup(wgpu::TextureFormat::Rgba8Unorm);
    let slow = match Effect::new_state("accumulate slowly", "fn effect(uv: vec2<f32>) -> vec4<f32> { return state_px(vec2<i32>(0)) + vec4<f32>(0.125); }") {
        Ok(effect) => FilterChain::new().then(Pass::new(&effect)).then(pass_of("show", "fn effect(uv: vec2<f32>) -> vec4<f32> { return min(state(uv), vec4<f32>(1.0)); }")),
        Err(error) => panic!("{error}"),
    };
    let id = egui::Id::new("entries with state");
    let left = Rect::from_min_max(pos2(8.0, 8.0), pos2(120.0, 120.0));
    let right = Rect::from_min_max(pos2(136.0, 8.0), pos2(248.0, 120.0));
    let mut paint = |ui: &mut egui::Ui| {
        halves(ui.painter());
        ShaderLayer::from_entries([Entry::new(left, accumulator()), Entry::new(right, slow.clone())]).id(id).paint(ui.painter());
    };
    let slow_value = |frames: u32| [0, 32, 64, 96, 128].get(usize::try_from(frames).unwrap_or(0)).copied().unwrap_or(0);
    for frame in 1..=4 {
        let (pixels, _) = render_frame(&ctx, &state, &target, &mut paint);
        assert_grey(&pixels, 60, 60, accumulated(frame), &format!("frame {frame}, fast entry"));
        assert_grey(&pixels, 190, 60, slow_value(frame), &format!("frame {frame}, slow entry"));
    }
    assert_eq!(egui_shader_layers::state_count(&ctx), 2, "one state per entry");
    egui_shader_layers::reset_state(&ctx, id);
    let (pixels, _) = render_frame(&ctx, &state, &target, &mut paint);
    assert_grey(&pixels, 60, 60, accumulated(1), "fast entry after reset_state");
    assert_grey(&pixels, 190, 60, slow_value(1), "slow entry after reset_state");
    // Without an id the whole layer is refused (both entries keep the backdrop).
    let (pixels, _) = render_frame(&ctx, &state, &target, &mut |ui: &mut egui::Ui| {
        halves(ui.painter());
        ShaderLayer::from_entries([Entry::new(left, presets::invert(1.0)), Entry::new(right, slow.clone())]).paint(ui.painter());
    });
    assert_near(&pixels, 60, 60, [255, 0, 0], "an entry of a refused layer draws nothing");
    assert!(egui_shader_layers::failed_states(&ctx).is_empty());
}

/// R3: textures are per entry — the same effect reads the checker in one entry and a yellow
/// texture in the other; the layer-level `texture` attaches one to every entry.
#[test]
#[ignore = "needs a GPU adapter (Vulkan / Metal / DX12 / GL); run with --ignored"]
fn entry_textures_are_per_entry() {
    let (ctx, state, _texture, target) = setup(wgpu::TextureFormat::Rgba8Unorm);
    let checker = ctx.load_texture("checker", checker(), TextureOptions::NEAREST);
    let yellow = ctx.load_texture("yellow", ColorImage::new([2, 2], vec![Color32::YELLOW; 4]), TextureOptions::NEAREST);
    let exact = pass_of("exact", TEX_EXACT);
    let left = Rect::from_min_max(pos2(0.0, 0.0), pos2(128.0, 128.0));
    let right = Rect::from_min_max(pos2(128.0, 0.0), pos2(256.0, 128.0));
    let mut per_entry = |ui: &mut egui::Ui| {
        halves(ui.painter());
        ShaderLayer::from_entries([Entry::new(left, exact.clone()).texture(0, checker.id()), Entry::new(right, exact.clone()).texture(0, yellow.id())]).paint(ui.painter());
    };
    render_frame(&ctx, &state, &target, &mut per_entry);
    let (pixels, _) = render_frame(&ctx, &state, &target, &mut per_entry);
    // The left entry's own uv spans its capture: the checker's four texels in its quadrants.
    for (x, y, colour) in [(32, 32, [255, 0, 0]), (96, 32, [0, 255, 0]), (32, 96, [0, 0, 255]), (96, 96, [255, 255, 255])] {
        assert_eq!(rgb(&pixels, x, y), colour, "left entry: checker at ({x}, {y})");
    }
    assert_eq!(rgb(&pixels, 192, 64), [255, 255, 0], "right entry: its own texture");
    let (pixels, _) = render_frame(&ctx, &state, &target, &mut |ui: &mut egui::Ui| {
        halves(ui.painter());
        ShaderLayer::from_entries([Entry::new(left, exact.clone()), Entry::new(right, exact.clone())]).texture(0, yellow.id()).paint(ui.painter());
    });
    assert!(rgb(&pixels, 32, 32) == [255, 255, 0] && rgb(&pixels, 192, 64) == [255, 255, 0], "the layer-level texture reaches every entry");
    assert!(egui_shader_layers::failed_textures(&ctx).is_empty(), "{:?}", egui_shader_layers::failed_textures(&ctx));
}

// ---- R2: benchmark (`tests/common/mod.rs` holds the scene builder shared with glow) ----

mod common;

/// R2 benchmark on wgpu: N glass tiles as N layers vs one layer with N regions vs one layer with N
/// entries, each with one chain for every tile and mixed (odd tiles pixelate; `common::Mode`), mean wall time per
/// frame (egui + tessellation + GPU + readback; `render_frame` waits for the GPU). Run in release:
/// `cargo test --release --all-features --test wgpu_headless bench_many_elements -- --ignored --nocapture`.
/// Asserts only that every mode applied the glass (a timed frame that draws nothing is no data).
#[test]
#[ignore = "benchmark; needs a GPU adapter; run with --ignored --nocapture"]
fn bench_many_elements() {
    use common::{BenchRow, ELEMENT_COUNTS, GridScene, Mode, glass_applied, print_table, time_per_frame};
    let (ctx, state, _texture, target) = setup(wgpu::TextureFormat::Rgba8Unorm);
    let area = Rect::from_min_size(Pos2::ZERO, vec2(256.0, 128.0));
    let backdrop_scene = GridScene::new(1, area);
    let baseline = time_per_frame(|| {
        render_frame(&ctx, &state, &target, &mut |ui| backdrop_scene.paint_backdrop(ui.painter()));
    });
    let (backdrop, _) = render_frame(&ctx, &state, &target, &mut |ui| backdrop_scene.paint_backdrop(ui.painter()));
    let mut rows = vec![BenchRow { mode: None, count: 0, per_frame: baseline }];
    for count in ELEMENT_COUNTS {
        let scene = GridScene::new(count, area);
        for mode in Mode::ALL {
            let mut paint = |ui: &mut egui::Ui| {
                scene.paint_backdrop(ui.painter());
                scene.paint(ui.painter(), mode);
            };
            let (pixels, _) = render_frame(&ctx, &state, &target, &mut paint);
            assert!(glass_applied(&backdrop, &pixels, WIDTH, scene.first_tile_centre()), "{} x {count}: the glass was not applied", mode.name());
            let per_frame = time_per_frame(|| {
                render_frame(&ctx, &state, &target, &mut paint);
            });
            rows.push(BenchRow { mode: Some(mode), count, per_frame });
        }
    }
    print_table(&format!("wgpu, {:?}", state.adapter.get_info().backend), [WIDTH, HEIGHT], &rows);
    assert!(matches!(egui_shader_layers::status(&ctx), BackendStatus::Active { .. }), "{}", egui_shader_layers::status(&ctx));
}
