// egui_filter_layers: prelude prepended to every effect.
//
// Public surface for effect authors (see the crate docs, "Writing an effect"):
//   fl.rect_uv, fl.texel, fl.size_px, fl.time, fl.pixels_per_point,
//   fl.delta_time, fl.frame, fl.pointer, fl.pass_scale,
//   src(), src_px(), src_texel(), out_texel(), original(), params(), to_linear(), to_gamma(),
//   premultiply(), unpremultiply(), luma(), tex(), tex_size(), tex_load(), state(),
//   state_load(), state_px(), state_size(), state_texel(), fl.pass_index, region(),
//   region_at(), REGION_NONE, and `p` (the snippet's own `struct Params`, see below).
// Every other `fl_*` / `internal_*` name, and every `fl` member not listed in the crate docs, is
// private to the crate and may change.
// Persistent state (F7, `Effect::new_state`): a layer with a state pass owns two state textures
// (ping-pong) of `state_size()` pixels spanning the WHOLE capture (`uv` space). `state(uv)` samples
// the latest state bilinearly (the one sampler of `fl_state_texture`, like `src`); `state_load(px)`
// / `state_px(offset)` read exact texels (`textureLoad`) for simulations. A state pass's output
// becomes the new state; display passes read the state as the last state pass before them left
// it (or last frame's). Without a state pass the state is a 1x1 zero texture.
// Scaled passes (`Pass::with_scale`): a pass may render a smaller image that spans the whole
// capture (`fl.pass_scale` < 1). `uv`, `fl.texel`, `fl.size_px` stay the CAPTURE's; the pass's
// input and output resolutions are `internal_src_texel` (`src_texel()`, `src_px()`, the `src()`
// edge clamp) and `internal_frag_texel` (`out_texel()`, the mode-1 fragment uv).
// Per-frame input (the same for every layer of a frame): `fl.delta_time` = seconds since the
// previous frame (UI: egui's stable_dt clamped to 0..=0.1; offline: options / video frame times),
// `fl.frame` = frame counter (u32, wrapping), `fl.pointer` = pointer in capture uv (`xy`, the
// space of `uv`), primary button down (`z`, 0/1), present (`w`, 0/1; 0 = no pointer, xyz = 0).
// Reading them does not make an effect animated (only `fl.time` does).
// Regions (R1, `Mask::Regions`): a layer with regions binds a region-id texture (the run's window,
// full resolution, one id per pixel in `.r` as `id / 255`); `region_at(uv)` reads the exact texel
// (ids are never interpolated), `REGION_NONE` outside every region. Without regions the texture is
// a 1x1 white dummy, i.e. `REGION_NONE` everywhere.
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
    internal_src_texel: vec2<f32>,
    internal_state_texel: vec2<f32>,
    internal_state_size: vec2<f32>,
    internal_region_scale: vec2<f32>,
    internal_region_offset: vec2<f32>,
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
@group(0) @binding(7) var fl_region_texture: texture_2d<f32>;

// User textures (F8): slot `i` is `@group(1 + i)`, laid out like egui-wgpu's texture bind group
// (texture at 0, its sampler at 1), so wgpu binds egui's own bind group of the texture there and
// glow binds the texture on unit `4 + i`. Every texture is read through exactly ONE sampler, its
// own (the egui `TextureOptions` it was uploaded with): naga's GLSL writer rejects a texture read
// through two samplers. Exact texel reads use `textureLoad`, which needs no sampler. A slot the
// layer leaves empty is a 1x1 dummy, but an effect that may read it is refused before drawing
// (`Effect::reads_texture`), so a dummy is never what an effect sees.
@group(1) @binding(0) var fl_user0: texture_2d<f32>;
@group(1) @binding(1) var fl_user0_sampler: sampler;
@group(2) @binding(0) var fl_user1: texture_2d<f32>;
@group(2) @binding(1) var fl_user1_sampler: sampler;
@group(3) @binding(0) var fl_user2: texture_2d<f32>;
@group(3) @binding(1) var fl_user2_sampler: sampler;

var<private> fl_frag_uv: vec2<f32>;

struct FlVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// `texel` = one pixel of the sampled image in capture uv: the clamp keeps bilinear reads inside
// its used region (a grow-only texture holds garbage beyond it).
fn fl_storage_uv(uv: vec2<f32>, texel: vec2<f32>, scale: vec2<f32>, offset: vec2<f32>) -> vec2<f32> {
    let half_texel = 0.5 * texel;
    let c = clamp(uv, half_texel, vec2<f32>(1.0) - half_texel);
    let y = select(c.y, 1.0 - c.y, fl.internal_flip_y > 0.5);
    return vec2<f32>(c.x, y) * scale - offset;
}

fn src(uv: vec2<f32>) -> vec4<f32> {
    return textureSampleLevel(fl_src_texture, fl_sampler, fl_storage_uv(uv, fl.internal_src_texel, fl.internal_src_scale, fl.internal_src_offset), 0.0);
}

// `offset_px` in pixels of the pass's INPUT (capture pixels unless the previous pass is scaled).
fn src_px(offset_px: vec2<f32>) -> vec4<f32> {
    return src(fl_frag_uv + offset_px * fl.internal_src_texel);
}

// One pixel of the pass's input (`src`) in `uv` units.
fn src_texel() -> vec2<f32> {
    return fl.internal_src_texel;
}

// One pixel of the pass's own output in `uv` units (`fl.texel / fl.pass_scale`, up to rounding).
fn out_texel() -> vec2<f32> {
    return fl.internal_frag_texel;
}

// The untouched capture (the backdrop before any pass), in every pass and the composite: the
// plan binds the capture at `fl_original_texture` for every draw, mapped by `internal_aux_*`.
fn original(uv: vec2<f32>) -> vec4<f32> {
    return textureSampleLevel(fl_original_texture, fl_sampler, fl_storage_uv(uv, fl.texel, fl.internal_aux_scale, fl.internal_aux_offset), 0.0);
}

fn params(i: u32) -> vec4<f32> {
    return fl.params[min(i, 15u)];
}

// User texture `i` (0..3) at `uv` in ITS OWN coordinates ((0, 0) top-left, (1, 1) bottom-right
// of the texture, y down), through the texture's own sampler (filter and wrap mode from its egui
// `TextureOptions`; mip level 0). Premultiplied gamma-space RGBA, like `src()`. Slot >= 3: zero.
fn tex(i: u32, uv: vec2<f32>) -> vec4<f32> {
    var color = vec4<f32>(0.0);
    switch i {
        case 0u: { color = textureSampleLevel(fl_user0, fl_user0_sampler, uv, 0.0); }
        case 1u: { color = textureSampleLevel(fl_user1, fl_user1_sampler, uv, 0.0); }
        case 2u: { color = textureSampleLevel(fl_user2, fl_user2_sampler, uv, 0.0); }
        default: {}
    }
    return color;
}

// Size of user texture `i` in texels. Slot >= 3: zero.
fn tex_size(i: u32) -> vec2<f32> {
    var size = vec2<u32>(0u);
    switch i {
        case 0u: { size = textureDimensions(fl_user0); }
        case 1u: { size = textureDimensions(fl_user1); }
        case 2u: { size = textureDimensions(fl_user2); }
        default: {}
    }
    return vec2<f32>(size);
}

// The exact texel `px` (integer texel coordinates, (0, 0) top-left) of user texture `i`, no
// filtering; clamped to the texture. Slot >= 3: zero.
fn tex_load(i: u32, px: vec2<i32>) -> vec4<f32> {
    var color = vec4<f32>(0.0);
    switch i {
        case 0u: { color = textureLoad(fl_user0, clamp(px, vec2<i32>(0), vec2<i32>(textureDimensions(fl_user0)) - vec2<i32>(1)), 0); }
        case 1u: { color = textureLoad(fl_user1, clamp(px, vec2<i32>(0), vec2<i32>(textureDimensions(fl_user1)) - vec2<i32>(1)), 0); }
        case 2u: { color = textureLoad(fl_user2, clamp(px, vec2<i32>(0), vec2<i32>(textureDimensions(fl_user2)) - vec2<i32>(1)), 0); }
        default: {}
    }
    return color;
}

// The latest persistent state at `uv` (capture uv, like `src`), bilinear over the state texture
// (clamped to its used region). Zero before any state pass wrote it.
fn state(uv: vec2<f32>) -> vec4<f32> {
    return textureSampleLevel(fl_state_texture, fl_sampler, fl_storage_uv(uv, fl.internal_state_texel, fl.internal_state_scale, fl.internal_state_offset), 0.0);
}

// Size of the state in texels (`ceil(capture * StateOptions::scale)`; the stored content's size).
fn state_size() -> vec2<f32> {
    return fl.internal_state_size;
}

// One state texel in `uv` units.
fn state_texel() -> vec2<f32> {
    return fl.internal_state_texel;
}

// The exact state texel `px` (integer texel coordinates, (0, 0) top-left, y down), no
// filtering; clamped to the state (edges repeat: a reflecting border for simulations).
fn state_load(px: vec2<i32>) -> vec4<f32> {
    let size = vec2<i32>(fl.internal_state_size);
    let c = clamp(px, vec2<i32>(0), size - vec2<i32>(1));
    let y = select(c.y, size.y - 1 - c.y, fl.internal_flip_y > 0.5);
    return textureLoad(fl_state_texture, vec2<i32>(c.x, y), 0);
}

// The state texel under the current fragment, offset by `offset` texels (`state_load`).
fn state_px(offset: vec2<i32>) -> vec4<f32> {
    let here = vec2<i32>(floor(fl_frag_uv * fl.internal_state_size));
    return state_load(here + offset);
}

// The "no region" id of `region()` / `region_at(uv)`.
const REGION_NONE: u32 = 255u;

// Id of the region owning capture uv `uv` (exact texel: ids must never be interpolated), or
// REGION_NONE outside every region and on layers without regions. The texture holds the window
// at full capture resolution, so `fl.texel` is its texel.
fn region_at(uv: vec2<f32>) -> u32 {
    let storage = fl_storage_uv(uv, fl.texel, fl.internal_region_scale, fl.internal_region_offset);
    let px = vec2<i32>(floor(storage * vec2<f32>(textureDimensions(fl_region_texture))));
    return u32(round(textureLoad(fl_region_texture, px, 0).r * 255.0));
}

// Id of the region owning the current pixel (`region_at` at the fragment's uv).
fn region() -> u32 {
    return region_at(fl_frag_uv);
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
// position (mode 1, wgpu), which is exact for any window of the capture. A scaled pass's pixels
// are `internal_frag_texel` wide (the whole capture over its output).
fn fl_capture_uv(in: FlVertexOutput) -> vec2<f32> {
    let from_position = (in.position.xy + fl.internal_frag_offset) * fl.internal_frag_texel;
    return select(in.uv, from_position, fl.internal_frag_mode > 0.5);
}

@fragment
fn fl_fragment(in: FlVertexOutput) -> @location(0) vec4<f32> {
    fl_frag_uv = fl_capture_uv(in);
    return effect(fl_frag_uv);
}

// ---- user effect follows ----
