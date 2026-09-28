// Hot-reload starter effect for the demo's "hot" scene.
//
// Edit it in the demo's editor (Save / Ctrl+S) or in any text editor: the file is recompiled in
// the background. A compile error shows on the layer while the last good version keeps running.
// Every member of `struct Params` gets a control in the side panel, and its value survives a
// reload as long as the member keeps its name and type. Members start at zero in a new pass.
struct Params {
    // Wave amplitude, in points.
    amplitude: f32,
    // Number of waves across the captured region.
    frequency: f32,
    // Wave speed, in radians per second.
    speed: f32,
    // Chromatic split between the red and blue channels, in points.
    split: f32,
    // Premultiplied tint laid over the result; alpha is its strength (zero = no tint).
    tint_color: vec4<f32>,
}

fn effect(uv: vec2<f32>) -> vec4<f32> {
    let ppp = fl.pixels_per_point;
    let phase = uv.y * p.frequency * 6.2831853 + fl.time * p.speed;
    let wave = sin(phase) * p.amplitude * ppp;
    let split = p.split * ppp;
    let r = src_px(vec2<f32>(wave + split, 0.0)).r;
    let g = src_px(vec2<f32>(wave, 0.0)).g;
    let b = src_px(vec2<f32>(wave - split, 0.0)).b;
    let a = src_px(vec2<f32>(wave, 0.0)).a;
    // Premultiplied "tint over colour", keeping the colour's coverage.
    let tinted = vec3<f32>(r, g, b) * (1.0 - p.tint_color.a) + p.tint_color.rgb * a;
    return vec4<f32>(tinted, a);
}
