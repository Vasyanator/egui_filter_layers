// "textures" demo scene (examples/demo/scenes/textures.rs): displacement driven by a tiling noise
// texture in slot 0 (two channels = two independent noises, `LINEAR_REPEAT`).
struct Params {
    // Largest displacement, in points.
    strength: f32,
    // Size of one noise tile on screen, in points.
    tile: f32,
    // Offset of the noise in tile units (the scene pans it; periodic while recording).
    pan: vec2<f32>,
}

fn effect(uv: vec2<f32>) -> vec4<f32> {
    // Layer-relative position in points, so the pattern follows the layer when it is dragged.
    let points = (uv - fl.rect_uv.xy) * fl.size_px / fl.pixels_per_point;
    let t = points / p.tile + p.pan;
    // Two octaves; the texture repeats, so any `t` is valid.
    let n = (tex(0u, t).rg + tex(0u, t * 2.0 + vec2<f32>(0.37, 0.71)).rg * 0.5) / 1.5 - vec2<f32>(0.5);
    // n is in -0.5..0.5 per axis: 2 * n * strength points, converted to capture uv.
    return src(uv + 2.0 * n * p.strength * fl.pixels_per_point * fl.texel);
}
