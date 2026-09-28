// Bright pass of the "bloom" demo scene's custom chain (examples/demo/scenes/bloom.rs). The scene
// runs it at a reduced resolution (`Pass::with_scale`, the "pass scale" control); the blur and a
// full-resolution combine (bloom_combine.wgsl) follow.
struct Params {
    // Luma (0..1) above which the backdrop glows.
    threshold: f32,
    // Half-width of the soft transition around the threshold, in luma.
    knee: f32,
}

// The part of one premultiplied colour that glows: scaled as a whole, so it stays valid
// premultiplied colour.
fn glow_part(c: vec4<f32>) -> vec4<f32> {
    return c * smoothstep(p.threshold - p.knee, p.threshold + p.knee, luma(c.rgb));
}

fn effect(uv: vec2<f32>) -> vec4<f32> {
    // Four bilinear taps a quarter OUTPUT pixel off the centre: at a reduced scale they average
    // the input pixels under this output pixel instead of point-sampling one (less shimmer).
    let h = 0.25 * out_texel();
    let d = vec2<f32>(h.x, -h.y);
    return (glow_part(src(uv + h)) + glow_part(src(uv - h)) + glow_part(src(uv + d)) + glow_part(src(uv - d))) * 0.25;
}
