// "textures" demo scene (examples/demo/scenes/textures.rs): colour grading through a 16x16x16 3D
// LUT flattened into the 256x16 texture in slot 2: texel (r + 16 * b, g) holds the graded colour
// of the input (r, g, b) / 15. Read with exact `tex_load` taps and interpolated trilinearly here
// (the hardware cannot filter across the flattened blue slices).
struct Params {
    // 0 = original, 1 = fully graded.
    amount: f32,
    // Layer-relative x (0..1) right of which the original is shown, for comparison.
    split: f32,
}

fn lut_at(c: vec3<i32>) -> vec3<f32> {
    return tex_load(2u, vec2<i32>(c.r + 16 * c.b, c.g)).rgb;
}

fn effect(uv: vec2<f32>) -> vec4<f32> {
    let c = src(uv);
    if (uv.x - fl.rect_uv.x) / fl.rect_uv.z > p.split {
        return c;
    }
    let straight = unpremultiply(c);
    let s = clamp(straight.rgb, vec3<f32>(0.0), vec3<f32>(1.0)) * 15.0;
    let i0 = vec3<i32>(floor(s));
    let i1 = min(i0 + vec3<i32>(1), vec3<i32>(15));
    let f = s - floor(s);
    // Trilinear: interpolate along r, then g, then b.
    let c00 = mix(lut_at(vec3<i32>(i0.r, i0.g, i0.b)), lut_at(vec3<i32>(i1.r, i0.g, i0.b)), f.r);
    let c10 = mix(lut_at(vec3<i32>(i0.r, i1.g, i0.b)), lut_at(vec3<i32>(i1.r, i1.g, i0.b)), f.r);
    let c01 = mix(lut_at(vec3<i32>(i0.r, i0.g, i1.b)), lut_at(vec3<i32>(i1.r, i0.g, i1.b)), f.r);
    let c11 = mix(lut_at(vec3<i32>(i0.r, i1.g, i1.b)), lut_at(vec3<i32>(i1.r, i1.g, i1.b)), f.r);
    let graded = mix(mix(c00, c10, f.g), mix(c01, c11, f.g), f.b);
    return premultiply(vec4<f32>(mix(straight.rgb, graded, clamp(p.amount, 0.0, 1.0)), straight.a));
}
