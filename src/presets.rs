/*
File: presets.rs

Purpose:
Ready-made filters. Each function returns a configured `Pass` (or a `FilterChain` for multi-pass
filters); the underlying `Effect`s are compiled once per process and shared.

Key functions:
- colour: `brightness_contrast`, `saturation`, `grayscale`, `invert`, `tint`, `vignette`
- neighbourhood: `gaussian_blur` (2 separable passes), `sharpen`, `pixelate`
- refraction: `ripple` (animated), `lens`
- impulse-driven: `click_ripples` (driven by an `Impulses`, not animated by itself)
- reduced resolution (F6, `Pass::with_scale`): `kawase_blur` (dual-Kawase pyramid), `bloom`
- persistent state (F7, state passes; the layer needs `FilterLayer::id`): `trails`, and the
  water pair `wave_sim` (height-field wave equation with obstacles and impulses) + `wave_display`
  (refraction and a highlight from the heights); `WaveObstacle`
- combos: `frosted_glass`

Notes:
All maths runs on premultiplied, gamma-space colour exactly as egui wrote it (see crate docs).
Distances are in layer points and are converted with `fl.pixels_per_point` (pixels per layer
point, Scene zoom included) in the shaders, so they scale with the layer like the padding.
`BUILTIN_EFFECTS` lists every built-in WGSL source so tests can validate and translate all of them.
*/

//! Ready-made filters. Each function returns a configured [`Pass`] or [`FilterChain`]; the
//! underlying effects are compiled once per process and shared by every call.

use std::sync::OnceLock;

use egui::{Color32, Rect};

use crate::effect::{Effect, EffectError, ParamError};
use crate::impulses::Impulses;
use crate::pass::{FilterChain, PARAM_SLOTS, Params, Pass};

/// Brightness / contrast. `params(0)`: x = brightness offset, y = contrast factor.
const BRIGHTNESS_CONTRAST: &str = "fn effect(uv: vec2<f32>) -> vec4<f32> {
    let c = unpremultiply(src(uv));
    let p = params(0u);
    let rgb = (c.rgb - vec3<f32>(0.5)) * p.y + vec3<f32>(0.5 + p.x);
    return premultiply(vec4<f32>(clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0)), c.a));
}
";

/// Saturation. `params(0).x` = factor (0 grey, 1 unchanged, >1 boosted).
const SATURATION: &str = "fn effect(uv: vec2<f32>) -> vec4<f32> {
    let c = unpremultiply(src(uv));
    let grey = vec3<f32>(luma(c.rgb));
    let rgb = mix(grey, c.rgb, params(0u).x);
    return premultiply(vec4<f32>(clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0)), c.a));
}
";

/// Grayscale. `params(0).x` = amount in 0..1.
const GRAYSCALE: &str = "fn effect(uv: vec2<f32>) -> vec4<f32> {
    let c = src(uv);
    return vec4<f32>(mix(c.rgb, vec3<f32>(luma(c.rgb)), clamp(params(0u).x, 0.0, 1.0)), c.a);
}
";

/// Invert. `params(0).x` = amount in 0..1.
const INVERT: &str = "fn effect(uv: vec2<f32>) -> vec4<f32> {
    let c = unpremultiply(src(uv));
    let rgb = mix(c.rgb, vec3<f32>(1.0) - c.rgb, clamp(params(0u).x, 0.0, 1.0));
    return premultiply(vec4<f32>(rgb, c.a));
}
";

/// Colour wash. `params(0)` = colour (unmultiplied gamma RGB), `params(1).x` = amount.
const TINT: &str = "fn effect(uv: vec2<f32>) -> vec4<f32> {
    let c = unpremultiply(src(uv));
    let rgb = mix(c.rgb, params(0u).rgb, clamp(params(1u).x, 0.0, 1.0));
    return premultiply(vec4<f32>(rgb, c.a));
}
";

/// One direction of a separable Gaussian. `params(0)`: xy = unit direction, z = sigma (points).
const GAUSSIAN_BLUR: &str = "fn effect(uv: vec2<f32>) -> vec4<f32> {
    let p = params(0u);
    let sigma = p.z * fl.pixels_per_point;
    if sigma < 0.01 {
        return src(uv);
    }
    let radius = ceil(3.0 * sigma);
    // At most 32 taps per side: large radii step over pixels, bilinear filtering fills the gaps.
    // `round`, not `ceil`: radius / step is 32 or the integer radius up to division rounding, and
    // one extra tap would read a step beyond the declared radius.
    let step = max(1.0, radius / 32.0);
    let taps = i32(round(radius / step));
    var sum = vec4<f32>(0.0);
    var total = 0.0;
    for (var i = -taps; i <= taps; i = i + 1) {
        let x = f32(i) * step;
        let w = exp(-0.5 * x * x / (sigma * sigma));
        // Capture pixels (not `src_px`, which steps INPUT pixels): the same blur at any scale.
        sum = sum + src(uv + p.xy * x * fl.texel) * w;
        total = total + w;
    }
    return sum / total;
}
";

/// Unsharp mask on the 4-neighbourhood. `params(0).x` = amount.
const SHARPEN: &str = "fn effect(uv: vec2<f32>) -> vec4<f32> {
    let a = params(0u).x;
    let c = src(uv);
    let n = src_px(vec2<f32>(0.0, -1.0)) + src_px(vec2<f32>(0.0, 1.0)) + src_px(vec2<f32>(-1.0, 0.0)) + src_px(vec2<f32>(1.0, 0.0));
    let s = c * (1.0 + 4.0 * a) - n * a;
    // Clamp alpha first and rgb to it, so the result stays valid premultiplied colour (rgb <= a).
    let alpha = clamp(s.a, 0.0, 1.0);
    return vec4<f32>(clamp(s.rgb, vec3<f32>(0.0), vec3<f32>(alpha)), alpha);
}
";

/// Pixelation with cells aligned to the layer rect. `params(0).x` = cell size (points).
const PIXELATE: &str = "fn effect(uv: vec2<f32>) -> vec4<f32> {
    let cell = max(params(0u).x * fl.pixels_per_point, 1.0) * fl.texel;
    let origin = fl.rect_uv.xy;
    return src((floor((uv - origin) / cell) + vec2<f32>(0.5)) * cell + origin);
}
";

/// Concentric animated ripples around the rect centre.
/// `params(0)`: x = amplitude (points), y = wavelength (points), z = speed (waves per second).
const RIPPLE: &str = "fn effect(uv: vec2<f32>) -> vec4<f32> {
    let p = params(0u);
    let center = fl.rect_uv.xy + 0.5 * fl.rect_uv.zw;
    let d = (uv - center) * fl.size_px;
    let dist = length(d);
    let wavelength = max(p.y * fl.pixels_per_point, 1.0);
    let phase = 6.2831853 * (dist / wavelength - fl.time * p.z);
    let dir = select(vec2<f32>(0.0), d / dist, dist > 0.0);
    return src(uv + dir * sin(phase) * p.x * fl.pixels_per_point * fl.texel);
}
";

/// Magnifier: a spherical lens inscribed in the rect. `params(0).x` = magnification (>= 1).
const LENS: &str = "fn effect(uv: vec2<f32>) -> vec4<f32> {
    let center = fl.rect_uv.xy + 0.5 * fl.rect_uv.zw;
    let half_px = 0.5 * fl.rect_uv.zw * fl.size_px;
    let radius = min(half_px.x, half_px.y);
    let r = length((uv - center) * fl.size_px) / max(radius, 1.0);
    if r >= 1.0 {
        return src(uv);
    }
    let k = mix(1.0 / max(params(0u).x, 1.0), 1.0, r * r);
    return src(center + (uv - center) * k);
}
";

/// Vignette over the layer rect. `params(0)`: x = strength 0..1, y = inner radius, z = softness
/// (radii as fractions of the half-diagonal).
const VIGNETTE: &str = "fn effect(uv: vec2<f32>) -> vec4<f32> {
    let p = params(0u);
    let local = (uv - fl.rect_uv.xy) / fl.rect_uv.zw - vec2<f32>(0.5);
    let d = length(local * 2.0) / sqrt(2.0);
    let v = smoothstep(p.y, p.y + max(p.z, 0.001), d);
    let c = src(uv);
    return vec4<f32>(c.rgb * (1.0 - clamp(p.x, 0.0, 1.0) * v), c.a);
}
";

// ---- click_ripples (F3, driven by `Impulses`) ----

/// Expanding, damped rings around each live impulse (`Impulses::apply` fills `impulses` /
/// `impulse_count`). Each ring is one wave packet around the front at `age * speed`: a sine of
/// the distance to the front in wavelengths, windowed to ZERO beyond two wavelengths (so pixels
/// away from every front are read exactly in place) and faded by `(1 - age / life)^2` so it has
/// vanished when the impulse expires. The summed displacement is clamped to `|amplitude|`, which
/// keeps the declared sample radius honest however many rings overlap.
const CLICK_RIPPLES: &str = "struct Params {
    impulses: array<vec4<f32>, 16>,
    impulse_count: u32,
    amplitude: f32,
    wavelength: f32,
    speed: f32,
    life: f32,
}
fn effect(uv: vec2<f32>) -> vec4<f32> {
    let amplitude_px = p.amplitude * fl.pixels_per_point;
    let reach_px = abs(amplitude_px);
    let wavelength_px = max(p.wavelength * fl.pixels_per_point, 1.0);
    let count = min(p.impulse_count, 16u);
    var offset = vec2<f32>(0.0);
    for (var i = 0u; i < count; i = i + 1u) {
        let impulse = p.impulses[i];
        let fade = 1.0 - clamp(impulse.z / max(p.life, 0.000001), 0.0, 1.0);
        let d = (uv - (fl.rect_uv.xy + impulse.xy * fl.rect_uv.zw)) * fl.size_px;
        let dist = length(d);
        // Distance to the ring's front, in wavelengths.
        let x = (dist - impulse.z * p.speed * fl.pixels_per_point) / wavelength_px;
        let window = max(1.0 - 0.25 * x * x, 0.0);
        let wave = sin(6.2831853 * x) * window * window * fade * fade * impulse.w;
        let dir = select(vec2<f32>(0.0), d / dist, dist > 0.0);
        offset = offset + dir * (wave * amplitude_px);
    }
    let len = length(offset);
    if len > reach_px {
        offset = offset * (reach_px / len);
    }
    return src_px(offset);
}
";

// ---- end click_ripples ----

// ---- kawase_blur / bloom (F6, reduced-resolution passes) ----

/// Dual-Kawase downsample (Bjørge, "Bandwidth-efficient rendering", 2015): the centre and four
/// diagonal taps `o` input texels away, run at half the input's resolution. Every level reaches
/// `(o + 1)` of its input texels (taps plus the bilinear footprint); `o = radius_px / span - 1`
/// makes the whole pyramid reach exactly `radius` points (`span` = the pyramid's reach at
/// `o + 1 = 1`, in capture pixels, set by the constructor). `o` is clamped to `0..4`: below the
/// pyramid's smallest reach, or taps so far apart that they would alias.
const KAWASE_DOWN: &str = "struct Params { radius: f32, span: f32 }
fn effect(uv: vec2<f32>) -> vec4<f32> {
    let o = clamp(p.radius * fl.pixels_per_point / max(p.span, 1.0), 1.0, 5.0) - 1.0;
    let t = src_texel() * o;
    let d = vec2<f32>(t.x, -t.y);
    return (src(uv) * 4.0 + src(uv - t) + src(uv + t) + src(uv + d) + src(uv - d)) * 0.125;
}
";

/// Dual-Kawase upsample: four axis taps `o` input texels away and four diagonal ones at half
/// that distance (weight 2), run at twice the input's resolution. Same `Params` as the
/// downsample.
const KAWASE_UP: &str = "struct Params { radius: f32, span: f32 }
fn effect(uv: vec2<f32>) -> vec4<f32> {
    let o = clamp(p.radius * fl.pixels_per_point / max(p.span, 1.0), 1.0, 5.0) - 1.0;
    let t = src_texel() * o;
    let h = 0.5 * t;
    let axes = src(uv + vec2<f32>(-t.x, 0.0)) + src(uv + vec2<f32>(t.x, 0.0)) + src(uv + vec2<f32>(0.0, -t.y)) + src(uv + vec2<f32>(0.0, t.y));
    let diagonals = src(uv + h) + src(uv - h) + src(uv + vec2<f32>(h.x, -h.y)) + src(uv + vec2<f32>(-h.x, h.y));
    return (axes + diagonals * 2.0) / 12.0;
}
";

/// Bloom bright pass, run at half resolution: the four input pixels under an output pixel, each
/// kept by how far its luma exceeds `threshold` (premultiplied colour scaled as a whole).
const BLOOM_BRIGHT: &str = "struct Params { threshold: f32 }
fn bloom_bright(c: vec4<f32>) -> vec4<f32> {
    let l = luma(c.rgb);
    return c * (max(l - p.threshold, 0.0) / max(l, 0.0001));
}
fn effect(uv: vec2<f32>) -> vec4<f32> {
    let h = 0.5 * src_texel();
    let d = vec2<f32>(h.x, -h.y);
    return (bloom_bright(src(uv + h)) + bloom_bright(src(uv - h)) + bloom_bright(src(uv + d)) + bloom_bright(src(uv - d))) * 0.25;
}
";

/// Bloom combine, at full resolution: the ORIGINAL backdrop (`original()`, F1) plus
/// `intensity` × the blurred bright pass (`src`, upsampled bilinearly), clamped to valid
/// premultiplied colour (alpha raised to cover the added light).
const BLOOM_COMBINE: &str = "struct Params { intensity: f32 }
fn effect(uv: vec2<f32>) -> vec4<f32> {
    let o = original(uv);
    let rgb = min(o.rgb + src(uv).rgb * max(p.intensity, 0.0), vec3<f32>(1.0));
    return vec4<f32>(rgb, max(o.a, max(rgb.r, max(rgb.g, rgb.b))));
}
";

// ---- end kawase_blur / bloom ----

// ---- trails / water (F7, persistent state) ----

/// Trails, state pass: the state keeps the brightest of its decayed self and the current input,
/// per premultiplied channel (`max` of two valid premultiplied colours is valid). `decay` = the
/// fraction kept after one second (`pow(decay, delta_time)` per frame: frame-rate independent).
const TRAILS_STATE: &str = "struct Params { decay: f32 }
fn effect(uv: vec2<f32>) -> vec4<f32> {
    let keep = pow(clamp(p.decay, 0.0, 1.0), fl.delta_time);
    return max(state_px(vec2<i32>(0)) * keep, src(uv));
}
";

/// Trails, display pass: the input with the trail behind it.
const TRAILS_DISPLAY: &str = "fn effect(uv: vec2<f32>) -> vec4<f32> {
    return max(src(uv), state(uv));
}
";

/// Water, state pass: one explicit step of the 2D wave equation on a height field
/// (`r` = height, `g` = vertical velocity, per state texel; symplectic Euler, stable for
/// `speed + restore / 8 <= 0.5`): `v' = (v + speed * laplacian(h) - restore * h) * damping`,
/// `h' = h + v'`. `restore` pulls the mean level back to 0 (impulses push the surface down, which
/// would otherwise make the whole surface drift without bound). The borders
/// reflect (`state_px` clamps). On the first substep of a frame (`fl.pass_index == 0`) every live
/// impulse pushes the surface down with a Gaussian of `impulse_radius` points, faded over its life,
/// and a pressed pointer does so continuously with `pointer_strength` (per second, times
/// `fl.delta_time`). Inside an obstacle (rounded rects in layer-rect uv, radii in points) the
/// surface is held flat and still.
const WAVE_SIM: &str = "struct Params {
    impulses: array<vec4<f32>, 16>,
    impulse_count: u32,
    life: f32,
    speed: f32,
    damping: f32,
    restore: f32,
    impulse_radius: f32,
    pointer_strength: f32,
    obstacle_count: u32,
    obstacles: array<vec4<f32>, 8>,
    obstacle_radii: array<vec4<f32>, 2>,
}
fn wave_rounded_rect_distance(q: vec2<f32>, lo: vec2<f32>, hi: vec2<f32>, radius: f32) -> f32 {
    let half_size = 0.5 * (hi - lo);
    let r = clamp(radius, 0.0, min(half_size.x, half_size.y));
    let d = abs(q - 0.5 * (lo + hi)) - half_size + vec2<f32>(r);
    return length(max(d, vec2<f32>(0.0))) + min(max(d.x, d.y), 0.0) - r;
}
fn wave_splash(uv: vec2<f32>, center: vec2<f32>, radius_px: f32) -> f32 {
    let d = (uv - center) * fl.size_px / radius_px;
    return exp(-dot(d, d));
}
fn effect(uv: vec2<f32>) -> vec4<f32> {
    let here = state_px(vec2<i32>(0));
    let around = state_px(vec2<i32>(1, 0)).r + state_px(vec2<i32>(-1, 0)).r + state_px(vec2<i32>(0, 1)).r + state_px(vec2<i32>(0, -1)).r;
    let restore = clamp(p.restore, 0.0, 0.1);
    var v = (here.g + clamp(p.speed, 0.0, 0.5 - restore / 8.0) * (around - 4.0 * here.r) - restore * here.r) * clamp(p.damping, 0.0, 1.0);
    if fl.pass_index == 0u {
        let radius_px = max(p.impulse_radius * fl.pixels_per_point, 1.0);
        let count = min(p.impulse_count, 16u);
        for (var i = 0u; i < count; i = i + 1u) {
            let impulse = p.impulses[i];
            let fade = 1.0 - clamp(impulse.z / max(p.life, 0.000001), 0.0, 1.0);
            v = v - impulse.w * fade * fl.delta_time * wave_splash(uv, fl.rect_uv.xy + impulse.xy * fl.rect_uv.zw, radius_px);
        }
        if fl.pointer.w > 0.5 && fl.pointer.z > 0.5 {
            v = v - p.pointer_strength * fl.delta_time * wave_splash(uv, fl.pointer.xy, radius_px);
        }
    }
    let local = (uv - fl.rect_uv.xy) / fl.rect_uv.zw;
    let layer_px = fl.rect_uv.zw * fl.size_px;
    let obstacles = min(p.obstacle_count, 8u);
    for (var i = 0u; i < obstacles; i = i + 1u) {
        let o = p.obstacles[i];
        let radius = p.obstacle_radii[i / 4u][i % 4u] * fl.pixels_per_point;
        if wave_rounded_rect_distance(local * layer_px, o.xy * layer_px, o.zw * layer_px, radius) <= 0.0 {
            return vec4<f32>(0.0, 0.0, 0.0, 1.0);
        }
    }
    return vec4<f32>(here.r + v, v, 0.0, 1.0);
}
";

/// Water, display pass: refracts the input by the height gradient (`refraction` points of
/// displacement per unit of slope, capped at `|refraction|`) and adds a white highlight on slopes
/// facing the light (`specular`, zero on a flat surface: a calm surface is the input unchanged).
const WAVE_DISPLAY: &str = "struct Params { refraction: f32, specular: f32 }
fn effect(uv: vec2<f32>) -> vec4<f32> {
    let t = state_texel();
    let slope = 0.5 * vec2<f32>(state(uv + vec2<f32>(t.x, 0.0)).r - state(uv - vec2<f32>(t.x, 0.0)).r, state(uv + vec2<f32>(0.0, t.y)).r - state(uv - vec2<f32>(0.0, t.y)).r);
    let reach = abs(p.refraction) * fl.pixels_per_point;
    var offset = slope * p.refraction * fl.pixels_per_point;
    let len = length(offset);
    if len > reach {
        offset = offset * (reach / len);
    }
    let c = src(uv + offset * fl.texel);
    let n = normalize(vec3<f32>(-slope, 1.0));
    let h = normalize(vec3<f32>(-0.4, -0.6, 1.0) + vec3<f32>(0.0, 0.0, 1.0));
    let highlight = max(pow(max(dot(n, h), 0.0), 64.0) - pow(h.z, 64.0), 0.0) * max(p.specular, 0.0);
    let rgb = min(c.rgb + vec3<f32>(highlight), vec3<f32>(1.0));
    return vec4<f32>(rgb, max(c.a, max(rgb.r, max(rgb.g, rgb.b))));
}
";

// ---- end trails / water ----

/// Every built-in effect: `(label, WGSL snippet)`. Tests validate and translate all of them.
#[cfg(test)]
pub(crate) const BUILTIN_EFFECTS: &[(&str, &str)] = &[
    ("brightness_contrast", BRIGHTNESS_CONTRAST),
    ("saturation", SATURATION),
    ("grayscale", GRAYSCALE),
    ("invert", INVERT),
    ("tint", TINT),
    ("gaussian_blur", GAUSSIAN_BLUR),
    ("sharpen", SHARPEN),
    ("pixelate", PIXELATE),
    ("ripple", RIPPLE),
    ("lens", LENS),
    ("vignette", VIGNETTE),
    ("click_ripples", CLICK_RIPPLES),
    ("kawase_down", KAWASE_DOWN),
    ("kawase_up", KAWASE_UP),
    ("bloom_bright", BLOOM_BRIGHT),
    ("bloom_combine", BLOOM_COMBINE),
    ("trails_state", TRAILS_STATE),
    ("trails_display", TRAILS_DISPLAY),
    ("wave_sim", WAVE_SIM),
    ("wave_display", WAVE_DISPLAY),
];

/// Returns the shared built-in effect, compiling it on first use.
///
/// Panics only if a built-in WGSL constant above fails naga validation — a crate bug, not a
/// runtime condition: the constants never change at runtime and the test
/// `every_builtin_effect_validates` compiles each of them.
fn builtin(cell: &'static OnceLock<Effect>, label: &str, source: &str, configure: fn(Effect) -> Effect) -> Effect {
    cell.get_or_init(|| match Effect::new(label, source) {
        Ok(effect) => configure(effect),
        Err(error) => panic!("egui_filter_layers: built-in effect `{label}` is invalid (crate bug): {error}"),
    })
    .clone()
}

/// [`builtin`] for a STATE pass effect ([`Effect::new_state`]); the same crate-bug panic contract.
fn builtin_state(cell: &'static OnceLock<Effect>, label: &str, source: &str) -> Effect {
    cell.get_or_init(|| match Effect::new_state(label, source) {
        Ok(effect) => effect,
        Err(error) => panic!("egui_filter_layers: built-in state effect `{label}` is invalid (crate bug): {}", EffectError::to_string(&error)),
    })
    .clone()
}

/// A parameter block with `slots` in the first positions and zeros elsewhere.
fn params_of(slots: &[[f32; 4]]) -> Params {
    let mut params = [[0.0; 4]; PARAM_SLOTS];
    for (target, value) in params.iter_mut().zip(slots) {
        *target = *value;
    }
    params
}

/// Brightness offset (`-1..1`, 0 = unchanged) and contrast factor (`0..`, 1 = unchanged).
#[must_use]
pub fn brightness_contrast(brightness: f32, contrast: f32) -> Pass {
    static EFFECT: OnceLock<Effect> = OnceLock::new();
    Pass::new(&builtin(&EFFECT, "brightness_contrast", BRIGHTNESS_CONTRAST, |e| e)).with_params(params_of(&[[brightness, contrast, 0.0, 0.0]]))
}

/// Saturation factor: 0 = grey, 1 = unchanged, >1 = boosted.
#[must_use]
pub fn saturation(factor: f32) -> Pass {
    static EFFECT: OnceLock<Effect> = OnceLock::new();
    Pass::new(&builtin(&EFFECT, "saturation", SATURATION, |e| e)).with_params(params_of(&[[factor, 0.0, 0.0, 0.0]]))
}

/// Desaturation towards Rec. 709 luma; `amount` in `0..1`.
#[must_use]
pub fn grayscale(amount: f32) -> Pass {
    static EFFECT: OnceLock<Effect> = OnceLock::new();
    Pass::new(&builtin(&EFFECT, "grayscale", GRAYSCALE, |e| e)).with_params(params_of(&[[amount, 0.0, 0.0, 0.0]]))
}

/// Colour inversion; `amount` in `0..1`.
#[must_use]
pub fn invert(amount: f32) -> Pass {
    static EFFECT: OnceLock<Effect> = OnceLock::new();
    Pass::new(&builtin(&EFFECT, "invert", INVERT, |e| e)).with_params(params_of(&[[amount, 0.0, 0.0, 0.0]]))
}

/// Washes the backdrop towards `color` (its alpha is ignored); `amount` in `0..1`.
#[must_use]
pub fn tint(color: Color32, amount: f32) -> Pass {
    static EFFECT: OnceLock<Effect> = OnceLock::new();
    let [r, g, b, _] = color.to_srgba_unmultiplied();
    let rgb = [f32::from(r) / 255.0, f32::from(g) / 255.0, f32::from(b) / 255.0, 1.0];
    Pass::new(&builtin(&EFFECT, "tint", TINT, |e| e)).with_params(params_of(&[rgb, [amount, 0.0, 0.0, 0.0]]))
}

/// Separable Gaussian blur with standard deviation `sigma` in points (two passes).
///
/// Pads the capture by `3 * sigma` (the kernel reaches exactly `ceil(3 * sigma)` pixels along one
/// axis per pass, its last tap; see the padding note in the body). Blurring happens in gamma space
/// on premultiplied colour, like most UI blurs.
///
/// Up to `sigma * pixels_per_point = 32 / 3` every tap is a whole pixel (texel-centre sampling).
/// Above that, at most 32 taps per side step over pixels at fractional positions and hardware
/// bilinear filtering fills the gaps: offline tiles may then differ from a whole-frame run by at
/// most two 8-bit steps on a few values (see the offline module docs).
#[must_use]
pub fn gaussian_blur(sigma: f32) -> FilterChain {
    static EFFECT: OnceLock<Effect> = OnceLock::new();
    let effect = builtin(&EFFECT, "gaussian_blur", GAUSSIAN_BLUR, |e| e);
    let sigma = if sigma.is_finite() { sigma.max(0.0) } else { 0.0 };
    let radius = 3.0 * sigma;
    // Padding is summed over passes, but the two passes reach along DIFFERENT axes: the vertical
    // pass only reads rows of the horizontal pass's output in the same column, so every output
    // pixel depends on backdrop at most `radius` away per axis. The first pass therefore carries
    // the whole `radius` and the second declares 0 (padding 6σ would only enlarge the capture).
    FilterChain::new()
        .then(Pass::new(&effect).with_params(params_of(&[[1.0, 0.0, sigma, 0.0]])).with_sample_radius(radius))
        .then(Pass::new(&effect).with_params(params_of(&[[0.0, 1.0, sigma, 0.0]])).with_sample_radius(0.0))
}

/// Unsharp-mask sharpening; `amount` around `0..2`.
#[must_use]
pub fn sharpen(amount: f32) -> Pass {
    static EFFECT: OnceLock<Effect> = OnceLock::new();
    // One physical pixel of reach; 2 points covers it at >= 0.5 pixels per layer point, and the
    // padding is rounded up to whole pixels, so any positive scale still pads at least 1 px.
    Pass::new(&builtin(&EFFECT, "sharpen", SHARPEN, |e| e.with_sample_radius(2.0))).with_params(params_of(&[[amount, 0.0, 0.0, 0.0]]))
}

/// Pixelation into square cells of `cell_size` points, aligned to the layer's top-left corner.
#[must_use]
pub fn pixelate(cell_size: f32) -> Pass {
    static EFFECT: OnceLock<Effect> = OnceLock::new();
    Pass::new(&builtin(&EFFECT, "pixelate", PIXELATE, |e| e)).with_params(params_of(&[[cell_size, 0.0, 0.0, 0.0]])).with_sample_radius(cell_size)
}

/// Animated concentric ripples (refraction) around the rect centre.
///
/// `amplitude` and `wavelength` in points, `speed` in waves per second.
#[must_use]
pub fn ripple(amplitude: f32, wavelength: f32, speed: f32) -> Pass {
    static EFFECT: OnceLock<Effect> = OnceLock::new();
    Pass::new(&builtin(&EFFECT, "ripple", RIPPLE, |e| e)).with_params(params_of(&[[amplitude, wavelength, speed, 0.0]])).with_sample_radius(amplitude.abs())
}

/// A magnifying lens inscribed in the rect; `magnification >= 1` (smaller values are clamped).
///
/// It reads anywhere inside the layer rect (up to `(1 - 1/magnification)` × half its smaller side
/// from a pixel) and declares no sample radius: the UI always captures the whole rect. Offline
/// tiling only overlaps tiles by the declared radius, so give a tiled lens its reach with
/// [`Pass::with_sample_radius`].
#[must_use]
pub fn lens(magnification: f32) -> Pass {
    static EFFECT: OnceLock<Effect> = OnceLock::new();
    Pass::new(&builtin(&EFFECT, "lens", LENS, |e| e)).with_params(params_of(&[[magnification, 0.0, 0.0, 0.0]]))
}

/// Darkens towards the rect's corners. `strength` in `0..1`; `inner` and `softness` are fractions
/// of the half-diagonal where the darkening starts and how wide its ramp is.
#[must_use]
pub fn vignette(strength: f32, inner: f32, softness: f32) -> Pass {
    static EFFECT: OnceLock<Effect> = OnceLock::new();
    Pass::new(&builtin(&EFFECT, "vignette", VIGNETTE, |e| e)).with_params(params_of(&[[strength, inner, softness, 0.0]]))
}

/// "Frosted glass": a Gaussian blur followed by a light colour wash.
#[must_use]
pub fn frosted_glass(sigma: f32, color: Color32, amount: f32) -> FilterChain {
    gaussian_blur(sigma).then(tint(color, amount))
}

// ---- click_ripples (F3) ----

/// Named parameters of the built-in `click_ripples` effect are fixed WGSL above; a mismatch
/// with what this module sets is a crate bug, never a runtime condition (pinned by the test
/// `click_ripples_is_configured_from_its_impulses`), like an invalid built-in in [`builtin`].
fn builtin_param<T>(label: &str, result: Result<T, ParamError>) -> T {
    match result {
        Ok(value) => value,
        Err(error) => panic!("egui_filter_layers: built-in effect `{label}` rejected its own parameters (crate bug): {error}"),
    }
}

/// Rings expanding from every live impulse of `impulses` at `now` (seconds on the same clock
/// the impulses were pushed with, e.g. egui's `input.time`): a refraction that displaces the
/// backdrop by at most `amplitude` points, with `wavelength` points between crests, travelling
/// `speed` points per second and fading out over the impulses' life. Pixels farther than two
/// wavelengths from every ring's front are left exactly as they are.
///
/// Not animated by itself (it reads no `fl.time`): rebuild it every frame with the current
/// `now` and keep frames coming with [`Impulses::request_repaint_if_alive`]. Declares
/// `|amplitude|` as its sample radius. Non-finite arguments count as 0. At most
/// [`IMPULSE_SLOTS`](crate::IMPULSE_SLOTS) impulses are drawn. See [`Impulses`] for a complete example.
#[must_use]
pub fn click_ripples(impulses: &Impulses, now: f64, amplitude: f32, wavelength: f32, speed: f32) -> Pass {
    static EFFECT: OnceLock<Effect> = OnceLock::new();
    const LABEL: &str = "click_ripples";
    let finite = |value: f32| if value.is_finite() { value } else { 0.0 };
    let (amplitude, wavelength, speed) = (finite(amplitude), finite(wavelength), finite(speed));
    let mut pass = Pass::new(&builtin(&EFFECT, LABEL, CLICK_RIPPLES, |e| e)).with_sample_radius(amplitude.abs());
    builtin_param(LABEL, impulses.apply(&mut pass, now));
    let values = [("amplitude", amplitude), ("wavelength", wavelength), ("speed", speed), ("life", impulses.life())];
    for (name, value) in values {
        builtin_param(LABEL, pass.set(name, value));
    }
    pass
}

// ---- end click_ripples ----

// ---- kawase_blur / bloom (F6) ----

/// The deepest dual-Kawase pyramid the presets build (levels below the base resolution).
const KAWASE_MAX_LEVELS: u32 = 8;

/// The reach, in capture pixels, of a dual-Kawase pyramid of `levels` levels whose base runs at
/// `2^-base` of the capture, at tap offset `o = 0` (each level reaching one of its input
/// texels): `2^base * (3 * 2^levels - 3)` — the downsamples reach `2^base * (2^levels - 1)`, the
/// upsamples (reading the levels below) `2^base * (2^(levels + 1) - 2)`.
fn kawase_span(levels: u32, base: u32) -> f32 {
    let pow2 = |exponent: u32| (0..exponent).fold(1.0_f32, |value, _| value * 2.0);
    pow2(base) * (3.0 * pow2(levels) - 3.0)
}

/// The number of pyramid levels for a reach of `radius` points: the most levels (1..=8) whose
/// span stays within `0.75 * radius`, so at 1 physical pixel per point the tap offset `o` is at
/// least `1/3` and the per-level taps stay dense up to about 3 pixels per point.
fn kawase_levels(radius: f32, base: u32) -> u32 {
    let mut levels = 1;
    while levels < KAWASE_MAX_LEVELS && kawase_span(levels + 1, base) <= 0.75 * radius {
        levels += 1;
    }
    levels
}

/// `0.5^exponent` exactly (a power of two).
fn half_pow(exponent: u32) -> f32 {
    (0..exponent).fold(1.0_f32, |value, _| value * 0.5)
}

/// The dual-Kawase passes of a pyramid of `levels` levels over a base at `2^-base` of the
/// capture: downsamples to `2^-(base + 1)` … `2^-(base + levels)`, then upsamples back to
/// `2^-base`. Every pass reads `radius` (points) and the pyramid's `span`; no pass declares a
/// sample radius (the caller declares the whole chain's reach on its first pass).
fn kawase_pyramid(radius: f32, levels: u32, base: u32) -> Vec<Pass> {
    static DOWN: OnceLock<Effect> = OnceLock::new();
    static UP: OnceLock<Effect> = OnceLock::new();
    let span = kawase_span(levels, base);
    let configure = |effect: &Effect, label: &str, scale: f32| {
        let pass = Pass::new(effect).with_scale(scale).with_sample_radius(0.0);
        builtin_param(label, pass.with("radius", radius).and_then(|pass| pass.with("span", span)))
    };
    let down = builtin(&DOWN, "kawase_down", KAWASE_DOWN, |e| e);
    let up = builtin(&UP, "kawase_up", KAWASE_UP, |e| e);
    let downs = (1..=levels).map(|level| configure(&down, "kawase_down", half_pow(base + level)));
    let ups = (0..levels).rev().map(|level| configure(&up, "kawase_up", half_pow(base + level)));
    downs.chain(ups).collect()
}

/// A fast, large blur: a dual-Kawase pyramid (Bjørge 2015) that halves the resolution up to 8
/// times and comes back up, `2 * levels` cheap passes whose cost barely grows with the radius —
/// unlike [`gaussian_blur`], whose taps grow with sigma. `radius` (points) is how far the blur
/// reaches; its falloff is smooth but not exactly Gaussian (compare `gaussian_blur(radius / 3)`).
///
/// The level count is fixed from `radius` when the chain is built; the tap spacing adapts to the
/// pixels per point at paint time so the reach stays `radius` points. The reach cannot shrink
/// below the pyramid's own (about 3 pixels for small radii, `0.75 * radius` pixels otherwise),
/// so below 1 pixel per point (a zoomed-out scene) it may exceed `radius` a little; the declared
/// sample radius (the padding) covers it down to 1 pixel per point. For small radii (a few
/// points) prefer [`gaussian_blur`]. A zero or non-finite radius gives an empty chain.
/// Offline, it refuses tiling ([`Pass::with_scale`]).
#[must_use]
pub fn kawase_blur(radius: f32) -> FilterChain {
    let radius = if radius.is_finite() { radius.max(0.0) } else { 0.0 };
    if radius <= 0.0 {
        return FilterChain::new();
    }
    let levels = kawase_levels(radius, 0);
    let mut passes = kawase_pyramid(radius, levels, 0);
    if let Some(first) = passes.first_mut() {
        // The whole pyramid's reach, declared once (padding sums the passes' radii).
        *first = first.clone().with_sample_radius(radius.max(kawase_span(levels, 0)));
    }
    FilterChain::from(passes)
}

/// Bloom: light from the parts of the backdrop brighter than `threshold` (luma, `0..1`), blurred
/// over `radius` points and added `intensity` times over the ORIGINAL backdrop. A bright pass at
/// half resolution, a dual-Kawase pyramid over it (see [`kawase_blur`]) and a full-resolution
/// combine that reads the untouched capture through `original()`: placed after other passes in a
/// chain, the combine still starts from the unfiltered backdrop (put `bloom` first to keep
/// earlier passes' results). Declares its reach (`radius` plus the half-resolution resampling)
/// on its first pass. Non-finite arguments count as 0; offline, it refuses tiling.
#[must_use]
pub fn bloom(threshold: f32, intensity: f32, radius: f32) -> FilterChain {
    static BRIGHT: OnceLock<Effect> = OnceLock::new();
    static COMBINE: OnceLock<Effect> = OnceLock::new();
    let finite = |value: f32| if value.is_finite() { value } else { 0.0 };
    let (threshold, intensity, radius) = (finite(threshold), finite(intensity), finite(radius).max(0.0));
    let levels = kawase_levels(radius, 1);
    // Reach: the pyramid (at least its own span), plus 1 pixel for the bright pass's taps and 2
    // for the combine's bilinear read of the half-resolution result (pixels = points at >= 1
    // pixel per point).
    let reach = radius.max(kawase_span(levels, 1)) + 3.0;
    let bright = Pass::new(&builtin(&BRIGHT, "bloom_bright", BLOOM_BRIGHT, |e| e)).with_scale(0.5).with_sample_radius(reach);
    let combine = Pass::new(&builtin(&COMBINE, "bloom_combine", BLOOM_COMBINE, |e| e)).with_sample_radius(0.0);
    FilterChain::new()
        .then(builtin_param("bloom_bright", bright.with("threshold", threshold)))
        .then(FilterChain::from(kawase_pyramid(radius, levels, 1)))
        .then(builtin_param("bloom_combine", combine.with("intensity", intensity)))
}

// ---- end kawase_blur / bloom ----

// ---- trails / water (F7) ----

/// Trails: everything that passes under the layer leaves a fading trace. A state pass keeps, per
/// premultiplied channel, the brightest of the input and the decayed state, and a display pass
/// shows the brightest of the input and that state. `decay` is the fraction of a trace left after
/// one second (`0..=1`, clamped; e.g. `0.1`), applied per frame through `fl.delta_time`, so the
/// fade does not depend on the frame rate (offline: set `OfflineOptions::with_delta_time`, or run
/// a video, whose frame clock sets it).
///
/// A state chain: the layer needs [`FilterLayer::id`](crate::FilterLayer::id). The trace fades only
/// on frames egui paints (repaint while it should move). Keep the default RGBA16F state: 8-bit
/// state rounds small decays away, so faint traces would never vanish.
#[must_use]
pub fn trails(decay: f32) -> FilterChain {
    static STATE: OnceLock<Effect> = OnceLock::new();
    static DISPLAY: OnceLock<Effect> = OnceLock::new();
    let decay = if decay.is_finite() { decay.clamp(0.0, 1.0) } else { 0.0 };
    let state = builtin_param("trails_state", Pass::new(&builtin_state(&STATE, "trails_state", TRAILS_STATE)).with("decay", decay));
    FilterChain::new().then(state).then(Pass::new(&builtin(&DISPLAY, "trails_display", TRAILS_DISPLAY, |e| e)))
}

/// How many obstacles [`wave_sim`] takes (more are ignored).
pub const WAVE_OBSTACLES: usize = 8;

/// A rounded rect the water of [`wave_sim`] flows around (the surface is held still inside it),
/// in the layer rect's uv (`(0, 0)` = its top-left corner, `(1, 1)` = its bottom-right), like
/// [`Impulses`]; its corner radius in points.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct WaveObstacle {
    /// Top-left corner in layer-rect uv.
    pub min: [f32; 2],
    /// Bottom-right corner in layer-rect uv.
    pub max: [f32; 2],
    /// Corner radius in points.
    pub corner_radius: f32,
}

impl WaveObstacle {
    /// The obstacle spanning `min..max` (top-left and bottom-right corners in layer-rect uv) with
    /// corners rounded by `corner_radius` points (a negative radius means 0). The corners are
    /// taken as given; [`WaveObstacle::from_rect`] converts a rect in points instead.
    #[must_use]
    pub fn new(min: [f32; 2], max: [f32; 2], corner_radius: f32) -> Self {
        Self { min, max, corner_radius: corner_radius.max(0.0) }
    }

    /// The obstacle covering `obstacle` (points, the space `layer_rect` is in) of a layer painted
    /// over `layer_rect`; `None` for a layer rect without area or non-finite input.
    #[must_use]
    pub fn from_rect(obstacle: Rect, layer_rect: Rect, corner_radius: f32) -> Option<Self> {
        if !layer_rect.is_positive() || !layer_rect.is_finite() || !obstacle.is_finite() || !corner_radius.is_finite() {
            return None;
        }
        let min = (obstacle.min - layer_rect.min) / layer_rect.size();
        let max = (obstacle.max - layer_rect.min) / layer_rect.size();
        Some(Self { min: [min.x, min.y], max: [max.x, max.y], corner_radius: corner_radius.max(0.0) })
    }
}

/// Water, the simulation half: a STATE pass running one step of a height-field wave equation
/// (state `r` = height, `g` = velocity) per paint. Pair it with [`wave_display`], which draws it;
/// chain several copies for substeps (`fl.pass_index` tells them apart, impulses are injected
/// on the first):
///
/// ```
/// use egui_filter_layers::{FilterChain, Impulses, presets};
/// let impulses = Impulses::new(0.15);
/// let sim = presets::wave_sim(&impulses, 0.0, &[]);
/// let water = FilterChain::new().then(sim.clone()).then(sim).then(presets::wave_display(8.0, 0.6));
/// assert_eq!(water.len(), 3);
/// ```
///
/// Every live impulse of `impulses` at `now` (push them where the user clicks, see [`Impulses`])
/// pushes the surface down; `obstacles` (at most [`WAVE_OBSTACLES`], the rest ignored) are held
/// still, so waves reflect off them. Named parameters, set on the returned pass with
/// [`Pass::set`]: `speed` (`0..=0.5`, the squared Courant number: how far a wave travels per
/// step; default `0.35`), `damping` (velocity kept per step, default `0.995`), `restore` (pull
/// of the mean level back to 0 per step, `0..=0.1`, default `0.002`), `impulse_radius`
/// (points, default `6`), `pointer_strength` (a pressed pointer's continuous push per second,
/// default `0`: off). Steps are per paint, not per second: at a fixed frame rate the speed is
/// constant, and a slower frame rate slows the water (substeps compensate).
///
/// The layer needs [`FilterLayer::id`](crate::FilterLayer::id). The surface moves only on frames
/// egui paints: repaint while it is not calm (e.g. `ctx.request_repaint()` for a few seconds after
/// the last impulse). The heights need the default RGBA16F state (signed values).
#[must_use]
pub fn wave_sim(impulses: &Impulses, now: f64, obstacles: &[WaveObstacle]) -> Pass {
    static EFFECT: OnceLock<Effect> = OnceLock::new();
    const LABEL: &str = "wave_sim";
    let mut pass = Pass::new(&builtin_state(&EFFECT, LABEL, WAVE_SIM));
    builtin_param(LABEL, impulses.apply(&mut pass, now));
    let used = &obstacles[..obstacles.len().min(WAVE_OBSTACLES)];
    let rects: Vec<[f32; 4]> = used.iter().map(|obstacle| [obstacle.min[0], obstacle.min[1], obstacle.max[0], obstacle.max[1]]).collect();
    let mut radii = [[0.0_f32; 4]; 2];
    for (index, obstacle) in used.iter().enumerate() {
        radii[index / 4][index % 4] = obstacle.corner_radius;
    }
    builtin_param(LABEL, pass.set_slice("obstacles", &rects));
    builtin_param(LABEL, pass.set_slice("obstacle_radii", &radii));
    // `used.len() <= WAVE_OBSTACLES` (8), so the conversion cannot fail.
    builtin_param(LABEL, pass.set("obstacle_count", u32::try_from(used.len()).unwrap_or(0)));
    let values = [("life", impulses.life()), ("speed", 0.35), ("damping", 0.995), ("restore", 0.002), ("impulse_radius", 6.0), ("pointer_strength", 0.0)];
    for (name, value) in values {
        builtin_param(LABEL, pass.set(name, value));
    }
    pass
}

/// Water, the drawing half (a display pass after [`wave_sim`]): refracts the layer's input by the
/// water's slope — `refraction` points of displacement per unit of slope, capped at
/// `|refraction|` (its declared sample radius) — and adds a white highlight of strength
/// `specular` on slopes facing the light. A calm surface shows the input unchanged.
/// Non-finite arguments count as 0.
#[must_use]
pub fn wave_display(refraction: f32, specular: f32) -> Pass {
    static EFFECT: OnceLock<Effect> = OnceLock::new();
    const LABEL: &str = "wave_display";
    let finite = |value: f32| if value.is_finite() { value } else { 0.0 };
    let (refraction, specular) = (finite(refraction), finite(specular));
    let pass = Pass::new(&builtin(&EFFECT, LABEL, WAVE_DISPLAY, |e| e)).with_sample_radius(refraction.abs());
    builtin_param(LABEL, pass.with("refraction", refraction).and_then(|pass| pass.with("specular", specular)))
}

// ---- end trails / water ----

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_builtin_effect_validates() {
        for (label, source) in BUILTIN_EFFECTS {
            let result = Effect::new(label, source);
            assert!(result.is_ok(), "{label}: {result:?}");
        }
    }

    #[test]
    fn every_preset_constructor_works_and_shares_its_effect() {
        let passes: Vec<Pass> = vec![
            brightness_contrast(0.1, 1.2),
            saturation(1.5),
            grayscale(1.0),
            invert(1.0),
            tint(Color32::from_rgb(10, 20, 30), 0.5),
            sharpen(0.5),
            pixelate(8.0),
            ripple(4.0, 30.0, 1.0),
            lens(2.0),
            vignette(0.6, 0.4, 0.5),
            click_ripples(&Impulses::new(1.0), 0.0, 4.0, 20.0, 60.0),
        ];
        // Chains: gaussian_blur (1 effect), kawase_blur / bloom (4 effects between them), trails (2),
        // wave_sim + wave_display (2).
        assert_eq!(passes.len() + 1 + 4 + 2 + 2, BUILTIN_EFFECTS.len(), "one constructor per built-in pass effect");
        // Detected from the WGSL reading `fl.time`, not configured by hand.
        assert!(ripple(1.0, 1.0, 1.0).effect().is_animated());
        assert!(!brightness_contrast(0.1, 1.2).effect().is_animated());
        assert!(!ripple(1.0, 1.0, 1.0).effect().clone().with_animated(false).is_animated(), "the override wins");
        assert!(passes.iter().filter(|pass| pass.effect().is_animated()).count() == 1, "only ripple reads fl.time");
        let blur = gaussian_blur(4.0);
        assert_eq!(blur.len(), 2);
        assert!((blur.padding_points() - 12.0).abs() < 1e-5, "3 sigma per axis, not per pass");
        assert_eq!(blur.passes()[0].effect().module().id, blur.passes()[1].effect().module().id);
        assert_eq!(grayscale(0.2).effect().module().id, grayscale(0.9).effect().module().id);
        let [r, g, b, a] = tint(Color32::from_rgb(255, 0, 51), 0.25).params()[0];
        assert!((r - 1.0).abs() < 1e-6 && g.abs() < 1e-6 && (b - 0.2).abs() < 1e-6 && (a - 1.0).abs() < 1e-6);
        assert!((gaussian_blur(f32::NAN).padding_points()).abs() < f32::EPSILON);
        assert_eq!(frosted_glass(3.0, Color32::WHITE, 0.1).len(), 3);
    }

    /// F6: the pyramid's levels, scales and reach.
    #[test]
    fn kawase_and_bloom_are_scaled_pyramids() {
        use crate::ParamValue;
        assert!(kawase_blur(0.0).is_empty() && kawase_blur(f32::NAN).is_empty());
        // radius 40: span(2) = 9 <= 30, span(3) = 21 <= 30, span(4) = 45 > 30 -> 3 levels.
        let blur = kawase_blur(40.0);
        let scales: Vec<f32> = blur.passes().iter().map(Pass::scale).collect();
        assert_eq!(scales, [0.5, 0.25, 0.125, 0.25, 0.5, 1.0]);
        assert!((blur.padding_points() - 40.0).abs() < 1e-5, "reach declared once");
        assert_eq!(blur.passes()[0].get("span"), Some(ParamValue::F32(21.0)));
        assert!(blur.passes().iter().all(|pass| pass.get("radius") == Some(ParamValue::F32(40.0))));
        // Small radii: one level, the pyramid's own reach (3 px) is declared.
        let small = kawase_blur(1.0);
        assert_eq!(small.len(), 2);
        assert!((small.padding_points() - 3.0).abs() < 1e-5);
        // Huge radii stop at 8 levels (scale 1/256).
        let huge = kawase_blur(1.0e6);
        assert_eq!(huge.len(), 16);
        assert!(huge.passes().iter().all(|pass| pass.scale() >= crate::MIN_PASS_SCALE));

        let glow = bloom(0.7, 1.5, 40.0);
        let scales: Vec<f32> = glow.passes().iter().map(Pass::scale).collect();
        // Bright pass at 1/2, a 2-level pyramid over it (span(2, base 1) = 18 <= 30, span(3) = 42 > 30), combine at 1.
        assert_eq!(scales, [0.5, 0.25, 0.125, 0.25, 0.5, 1.0]);
        assert!((glow.padding_points() - 43.0).abs() < 1e-5, "radius + 3 px of resampling");
        assert_eq!(glow.passes()[0].get("threshold"), Some(ParamValue::F32(0.7)));
        assert_eq!(glow.passes().last().and_then(|pass| pass.get("intensity")), Some(ParamValue::F32(1.5)));
        assert_eq!(bloom(f32::NAN, f32::INFINITY, f32::NAN).passes()[0].get("threshold"), Some(ParamValue::F32(0.0)));
    }

    /// F7: the state presets are state passes with their named parameters set.
    #[test]
    fn state_presets_are_configured() {
        use crate::ParamValue;
        let chain = trails(0.25);
        assert_eq!(chain.len(), 2);
        assert!(chain.passes()[0].effect().is_state() && !chain.passes()[1].effect().is_state());
        assert_eq!(chain.passes()[0].get("decay"), Some(ParamValue::F32(0.25)));
        assert_eq!(trails(f32::NAN).passes()[0].get("decay"), Some(ParamValue::F32(0.0)));
        let mut impulses = Impulses::new(0.5);
        assert!(impulses.push([0.25, 0.75], 3.0, 1.0));
        let layer = Rect::from_min_size(egui::pos2(100.0, 50.0), egui::vec2(200.0, 100.0));
        let menu = Rect::from_min_size(egui::pos2(150.0, 75.0), egui::vec2(100.0, 50.0));
        let Some(obstacle) = WaveObstacle::from_rect(menu, layer, 12.0) else { panic!("valid rects") };
        assert_eq!(obstacle, WaveObstacle::new([0.25, 0.25], [0.75, 0.75], 12.0));
        assert_eq!(WaveObstacle::from_rect(menu, Rect::NOTHING, 1.0), None);
        let sim = wave_sim(&impulses, 1.25, &[obstacle; 10]);
        assert!(sim.effect().is_state() && !sim.effect().is_animated());
        assert_eq!(sim.get("impulse_count"), Some(ParamValue::U32(1)));
        assert_eq!(sim.get("obstacle_count"), Some(ParamValue::U32(8)), "at most WAVE_OBSTACLES");
        assert_eq!(sim.get_slice("obstacle_radii").map(|radii| radii[1]), Some(ParamValue::Vec4([12.0; 4])));
        assert_eq!(sim.get_slice("obstacles").map(|rects| rects[7]), Some(ParamValue::Vec4([0.25, 0.25, 0.75, 0.75])));
        assert_eq!(sim.get("life"), Some(ParamValue::F32(0.5)));
        let display = wave_display(-8.0, f32::NAN);
        assert!(!display.effect().is_state());
        assert!((display.sample_radius() - 8.0).abs() < f32::EPSILON);
        assert_eq!(display.get("specular"), Some(ParamValue::F32(0.0)));
    }

    #[test]
    fn click_ripples_is_configured_from_its_impulses() {
        use crate::ParamValue;
        let mut impulses = Impulses::new(2.0);
        assert!(impulses.push([0.5, 0.25], 1.5, 1.0));
        let pass = click_ripples(&impulses, 1.5, -6.0, 24.0, f32::NAN);
        assert!(!pass.effect().is_animated(), "driven by Impulses, not by fl.time");
        assert!((pass.sample_radius() - 6.0).abs() < f32::EPSILON, "reach = |amplitude|");
        assert_eq!(pass.get("impulse_count"), Some(ParamValue::U32(1)));
        assert_eq!(pass.get_slice("impulses").map(|slots| slots[0]), Some(ParamValue::Vec4([0.5, 0.25, 0.5, 1.5])));
        assert_eq!(pass.get("amplitude"), Some(ParamValue::F32(-6.0)));
        assert_eq!(pass.get("wavelength"), Some(ParamValue::F32(24.0)));
        assert_eq!(pass.get("speed"), Some(ParamValue::F32(0.0)), "non-finite counts as 0");
        assert_eq!(pass.get("life"), Some(ParamValue::F32(2.0)));
        assert_eq!(pass.effect().param("impulses").map(|info| info.count), u32::try_from(crate::IMPULSE_SLOTS).ok());
    }
}
