// Combine pass of the "bloom" demo scene's custom chain (examples/demo/scenes/bloom.rs), at full
// resolution: the ORIGINAL backdrop (`original()`, untouched by the bright pass and the blur)
// plus `intensity` x the blurred glow (`src`, upsampled bilinearly), tinted by `glow_color`.
struct Params {
    // Premultiplied gamma-space RGBA multiplying the glow (alpha unused).
    glow_color: vec4<f32>,
    // How strongly the glow is added.
    intensity: f32,
}

fn effect(uv: vec2<f32>) -> vec4<f32> {
    let o = original(uv);
    let rgb = min(o.rgb + src(uv).rgb * p.glow_color.rgb * max(p.intensity, 0.0), vec3<f32>(1.0));
    // Alpha raised to cover the added light, so the result stays valid premultiplied colour.
    return vec4<f32>(rgb, max(o.a, max(rgb.r, max(rgb.g, rgb.b))));
}
