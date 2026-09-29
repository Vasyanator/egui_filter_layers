// egui_filter_layers: draws egui meshes (egui's 20-byte vertex: pos, uv, packed
// colour) into an offscreen texture. Two uses:
// 1. wgpu backdrop re-render: the maths mirror egui-wgpu 0.36's `egui.wgsl`
//    (vertex colour * texture sample, premultiplied gamma, optional dithering)
//    so the capture holds what egui itself writes to the screen. Two fragment
//    entries, like egui-wgpu: `fs_main` (gamma framebuffer path: the capture is
//    written as-is) and `fs_main_linear` (linear framebuffer path: rendered
//    through an sRGB view of the capture, so blending happens in linear exactly
//    as on a `*Srgb` screen).
// 2. Shape-mask coverage (F9, every backend): `vs_main` + `fs_main` without
//    dithering into an RGBA8 coverage texture; the alpha is the coverage.
//    Region ids (R1): `vs_main` + `fs_region` write `region_id / 255` into `.r`
//    of the region-id texture where the mesh covers more than `alpha_threshold`
//    (no blend: the last region drawn owns the pixel). glow
//    runs this same module translated by naga's GLSL writer (`glow_backend/
//    coverage.rs`), so it must stay translatable (one sampler per texture, a
//    `u32` colour attribute unpacked by hand).
// Differences from egui-wgpu: vertex positions are mapped into the target
// viewport (`pos * pixels_per_point - origin_px`, where `pixels_per_point` may
// include a layer transform's scale) instead of the whole screen, the dithering
// noise is evaluated at the fragment's SCREEN position, and egui-wgpu's
// `predictable_texture_filtering` is not mirrored (hardware filtering only).

struct MeshLocals {
    pixels_per_point: f32,
    // 1 = dither like egui-wgpu with `dithering: true`, 0 = off.
    dithering: u32,
    // Subtracted from `pos * pixels_per_point`: the target's top-left corner (the
    // capture's screen position for the re-render).
    origin_px: vec2<f32>,
    // Size of the target viewport, in pixels.
    size_px: vec2<f32>,
    // R1 (`fs_region` only): the id written, and the coverage it must exceed.
    region_id: f32,
    alpha_threshold: f32,
};

@group(0) @binding(0) var<uniform> locals: MeshLocals;
// Same layout as egui-wgpu's texture bind group: egui's own per-texture bind
// groups are bound here unchanged.
@group(1) @binding(0) var tex_color: texture_2d<f32>;
@group(1) @binding(1) var tex_sampler: sampler;

struct VertexOutput {
    @location(0) tex_coord: vec2<f32>,
    @location(1) color: vec4<f32>,
    @builtin(position) position: vec4<f32>,
};

fn unpack_color(color: u32) -> vec4<f32> {
    return vec4<f32>(
        f32(color & 255u),
        f32((color >> 8u) & 255u),
        f32((color >> 16u) & 255u),
        f32((color >> 24u) & 255u),
    ) / 255.0;
}

fn interleaved_gradient_noise(n: vec2<f32>) -> f32 {
    let f = 0.06711056 * n.x + 0.00583715 * n.y;
    return fract(52.9829189 * fract(f));
}

@vertex
fn vs_main(
    @location(0) a_pos: vec2<f32>,
    @location(1) a_tex_coord: vec2<f32>,
    @location(2) a_color: u32,
) -> VertexOutput {
    let px = a_pos * locals.pixels_per_point - locals.origin_px;
    var out: VertexOutput;
    out.tex_coord = a_tex_coord;
    out.color = unpack_color(a_color);
    out.position = vec4<f32>(2.0 * px.x / locals.size_px.x - 1.0, 1.0 - 2.0 * px.y / locals.size_px.y, 0.0, 1.0);
    return out;
}

// 0-1 linear from 0-1 sRGB gamma; the same cutoff (`<`) as egui-wgpu's `linear_from_gamma_rgb`.
fn linear_from_gamma_rgb(srgb: vec3<f32>) -> vec3<f32> {
    let cutoff = srgb < vec3<f32>(0.04045);
    let lower = srgb / vec3<f32>(12.92);
    let higher = pow((srgb + vec3<f32>(0.055)) / vec3<f32>(1.055), vec3<f32>(2.4));
    return select(higher, lower, cutoff);
}

// Premultiplied gamma colour of a fragment, dithered in gamma like egui-wgpu.
fn shade_gamma(in: VertexOutput) -> vec4<f32> {
    var color = in.color * textureSample(tex_color, tex_sampler, in.tex_coord);
    if locals.dithering == 1u {
        var noise = interleaved_gradient_noise(in.position.xy + locals.origin_px);
        noise = (noise - 0.5) * 0.95;
        color = vec4<f32>(color.rgb + noise / 255.0, color.a);
    }
    return color;
}

// Gamma framebuffer path (`fs_main_gamma_framebuffer`): the capture is a plain unorm target.
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return shade_gamma(in);
}

// Linear framebuffer path (`fs_main_linear_framebuffer`): rendered through an sRGB view, so the
// hardware blends in linear and stores the sRGB encoding. Alpha is never converted.
@fragment
fn fs_main_linear(in: VertexOutput) -> @location(0) vec4<f32> {
    let color = shade_gamma(in);
    return vec4<f32>(linear_from_gamma_rgb(color.rgb), color.a);
}

// R1: writes the region id where the mesh covers the pixel more than `alpha_threshold` (the
// coverage as `fs_main` computes it, without dithering). Drawn without blending, so the LAST
// region drawn owns the pixel.
@fragment
fn fs_region(in: VertexOutput) -> @location(0) vec4<f32> {
    let a = in.color.a * textureSample(tex_color, tex_sampler, in.tex_coord).a;
    if a <= locals.alpha_threshold {
        discard;
    }
    return vec4<f32>(locals.region_id / 255.0, 0.0, 0.0, 1.0);
}
