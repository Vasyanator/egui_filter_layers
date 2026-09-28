/*
File: presets.rs

Purpose:
Ready-made filters. Each function returns a configured `Pass` (or a `FilterChain` for multi-pass
filters); the underlying `Effect`s are compiled once per process and shared.

Key functions:
- colour: `brightness_contrast`, `saturation`, `grayscale`, `invert`, `tint`, `vignette`
- neighbourhood: `gaussian_blur` (2 separable passes), `sharpen`, `pixelate`
- refraction: `ripple` (animated), `lens`
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

use egui::Color32;

use crate::effect::Effect;
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
        sum = sum + src_px(p.xy * x) * w;
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
    return src_px(dir * sin(phase) * p.x * fl.pixels_per_point);
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
        ];
        assert_eq!(passes.len() + 1, BUILTIN_EFFECTS.len(), "one constructor per built-in (blur is a chain)");
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
}
