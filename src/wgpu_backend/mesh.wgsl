// egui_filter_layers (wgpu): re-renders egui meshes into the offscreen capture
// texture. The maths mirror egui-wgpu 0.36's `egui.wgsl` (vertex colour *
// texture sample, premultiplied gamma, optional dithering) so the capture holds
// what egui itself writes to the screen. Two fragment entries, like egui-wgpu:
// `fs_main` (gamma framebuffer path: the capture is written as-is) and
// `fs_main_linear` (linear framebuffer path: rendered through an sRGB view of
// the capture, so blending happens in linear exactly as on a `*Srgb` screen).
// Differences: vertex positions are mapped into the capture viewport instead of
// the whole screen, the dithering noise is evaluated at the fragment's SCREEN
// position, and egui-wgpu's `predictable_texture_filtering` is not mirrored
// (hardware filtering only).

struct MeshLocals {
    pixels_per_point: f32,
    // 1 = dither like egui-wgpu with `dithering: true`, 0 = off.
    dithering: u32,
    // Top-left of the capture viewport, in screen pixels.
    origin_px: vec2<f32>,
    // Size of the capture viewport, in pixels.
    size_px: vec2<f32>,
    reserved: vec2<f32>,
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
