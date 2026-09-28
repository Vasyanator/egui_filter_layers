// egui_filter_layers: prelude prepended to every effect.
//
// Public surface for effect authors (see the crate docs, "Writing an effect"):
//   fl.rect_uv, fl.texel, fl.size_px, fl.time, fl.pixels_per_point,
//   fl.delta_time, fl.frame, fl.pointer,
//   src(), src_px(), original(), params(), to_linear(), to_gamma(), premultiply(),
//   unpremultiply(), luma(), and `p` (the snippet's own `struct Params`, see below).
// Every other `fl_*` / `internal_*` name, and every `fl` member not listed in the crate docs
// (`pass_scale`, `pass_index` hold neutral values until they are documented there), is private
// to the crate and may change.
// Per-frame input (the same for every layer of a frame): `fl.delta_time` = seconds since the
// previous frame (UI: egui's stable_dt clamped to 0..=0.1; offline: options / video frame times),
// `fl.frame` = frame counter (u32, wrapping), `fl.pointer` = pointer in capture uv (`xy`, the
// space of `uv`), primary button down (`z`, 0/1), present (`w`, 0/1; 0 = no pointer, xyz = 0).
// Reading them does not make an effect animated (only `fl.time` does).
// Coordinates: `uv` is (0, 0) at the top-left of the captured region, y down.
// `fl.pixels_per_point`: physical pixels per LAYER point = egui's pixels_per_point times the
// layer transform's scale (Scene zoom), so point distances follow the zoom like the padding does.
// Windows (wgpu): a pass may render only a WINDOW of the capture (offline tiling). The fragment's
// `uv` is then derived from its pixel position (`internal_frag_mode` = 1: `(position +
// internal_frag_offset) * texel`, the same float operations for every window), and textures hold
// the window at their origin: storage uv = clamp(uv) * internal_*_scale - internal_*_offset. With
// a whole-capture window the offsets are 0 and the scale is used / allocated, as without windows.

struct FlUniforms {
    rect_uv: vec4<f32>,
    texel: vec2<f32>,
    size_px: vec2<f32>,
    time: f32,
    pixels_per_point: f32,
    internal_flip_y: f32,
    internal_frag_mode: f32,
    internal_src_scale: vec2<f32>,
    internal_aux_scale: vec2<f32>,
    internal_frag_offset: vec2<f32>,
    internal_src_offset: vec2<f32>,
    internal_aux_offset: vec2<f32>,
    delta_time: f32,
    frame: u32,
    pointer: vec4<f32>,
    internal_state_scale: vec2<f32>,
    internal_state_offset: vec2<f32>,
    internal_cover_scale: vec2<f32>,
    internal_cover_offset: vec2<f32>,
    pass_scale: f32,
    pass_index: u32,
    internal_frag_texel: vec2<f32>,
    params: array<vec4<f32>, 16>,
}

// The complete `@group(0)` layout: every pass and the composite are bound with ALL of these
// (unused ones get a neutral dummy), so one layout serves every module. Binding 4 is the
// named-parameter block: when the snippet declares `struct Params`, the crate appends
// `@group(0) @binding(4) var<uniform> p: Params;` AFTER the snippet (so snippet line numbers are
// unaffected); a snippet may declare no binding at all.
@group(0) @binding(0) var<uniform> fl: FlUniforms;
@group(0) @binding(1) var fl_src_texture: texture_2d<f32>;
@group(0) @binding(2) var fl_sampler: sampler;
@group(0) @binding(3) var fl_original_texture: texture_2d<f32>;
@group(0) @binding(5) var fl_state_texture: texture_2d<f32>;
@group(0) @binding(6) var fl_coverage_texture: texture_2d<f32>;
@group(0) @binding(7) var fl_nearest_sampler: sampler;

var<private> fl_frag_uv: vec2<f32>;

struct FlVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

fn fl_storage_uv(uv: vec2<f32>, scale: vec2<f32>, offset: vec2<f32>) -> vec2<f32> {
    let half_texel = 0.5 * fl.texel;
    let c = clamp(uv, half_texel, vec2<f32>(1.0) - half_texel);
    let y = select(c.y, 1.0 - c.y, fl.internal_flip_y > 0.5);
    return vec2<f32>(c.x, y) * scale - offset;
}

fn src(uv: vec2<f32>) -> vec4<f32> {
    return textureSampleLevel(fl_src_texture, fl_sampler, fl_storage_uv(uv, fl.internal_src_scale, fl.internal_src_offset), 0.0);
}

fn src_px(offset_px: vec2<f32>) -> vec4<f32> {
    return src(fl_frag_uv + offset_px * fl.texel);
}

// The untouched capture (the backdrop before any pass), in every pass and the composite: the
// plan binds the capture at `fl_original_texture` for every draw, mapped by `internal_aux_*`.
fn original(uv: vec2<f32>) -> vec4<f32> {
    return textureSampleLevel(fl_original_texture, fl_sampler, fl_storage_uv(uv, fl.internal_aux_scale, fl.internal_aux_offset), 0.0);
}

fn params(i: u32) -> vec4<f32> {
    return fl.params[min(i, 15u)];
}

fn to_linear(c: vec4<f32>) -> vec4<f32> {
    let rgb = max(c.rgb, vec3<f32>(0.0));
    let lo = rgb / 12.92;
    let hi = pow((rgb + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    return vec4<f32>(select(hi, lo, rgb <= vec3<f32>(0.04045)), c.a);
}

fn to_gamma(c: vec4<f32>) -> vec4<f32> {
    let rgb = max(c.rgb, vec3<f32>(0.0));
    let lo = rgb * 12.92;
    let hi = 1.055 * pow(rgb, vec3<f32>(1.0 / 2.4)) - vec3<f32>(0.055);
    return vec4<f32>(select(hi, lo, rgb <= vec3<f32>(0.0031308)), c.a);
}

fn unpremultiply(c: vec4<f32>) -> vec4<f32> {
    if c.a <= 0.0 {
        return vec4<f32>(0.0);
    }
    return vec4<f32>(c.rgb / c.a, c.a);
}

fn premultiply(c: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(c.rgb * c.a, c.a);
}

fn luma(rgb: vec3<f32>) -> f32 {
    return dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
}

@vertex
fn fl_vertex(@builtin(vertex_index) index: u32) -> FlVertexOutput {
    // One oversized triangle covering the viewport: uv (0,0), (2,0), (0,2).
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    var out: FlVertexOutput;
    out.position = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
    out.uv = uv;
    return out;
}

// The fragment's capture uv: interpolated over the viewport (mode 0, OpenGL), or from the pixel
// position (mode 1, wgpu), which is exact for any window of the capture.
fn fl_capture_uv(in: FlVertexOutput) -> vec2<f32> {
    let from_position = (in.position.xy + fl.internal_frag_offset) * fl.texel;
    return select(in.uv, from_position, fl.internal_frag_mode > 0.5);
}

@fragment
fn fl_fragment(in: FlVertexOutput) -> @location(0) vec4<f32> {
    fl_frag_uv = fl_capture_uv(in);
    return effect(fl_frag_uv);
}

// ---- user effect follows ----
