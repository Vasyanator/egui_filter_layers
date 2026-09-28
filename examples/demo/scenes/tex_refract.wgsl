// "textures" demo scene (examples/demo/scenes/textures.rs): glass refraction from a normal map in
// slot 1 (rgb = normal * 0.5 + 0.5, y down, `LINEAR_REPEAT`) plus a Phong specular highlight.
struct Params {
    // Largest refraction offset, in points (for a normal lying flat).
    strength: f32,
    // Size of one normal-map tile on screen, in points.
    tile: f32,
    // Highlight strength (0 = none).
    specular: f32,
    // Highlight exponent (larger = tighter).
    shininess: f32,
    // Direction TOWARDS the light (xyz, y down, z out of the screen); normalized here.
    light: vec4<f32>,
}

fn effect(uv: vec2<f32>) -> vec4<f32> {
    let points = (uv - fl.rect_uv.xy) * fl.size_px / fl.pixels_per_point;
    let encoded = tex(1u, points / p.tile).rgb * 2.0 - vec3<f32>(1.0);
    let n = normalize(vec3<f32>(encoded.xy, max(encoded.z, 0.05)));
    // A slanted surface bends the view towards its downhill side.
    let base = src(uv - n.xy * p.strength * fl.pixels_per_point * fl.texel);
    let r = reflect(-normalize(p.light.xyz), n);
    // Viewer straight above: the highlight is the reflected light's z component.
    let spec = pow(max(r.z, 0.0), max(p.shininess, 1.0)) * p.specular;
    let a = max(base.a, min(spec, 1.0));
    // Premultiplied: rgb may not exceed alpha.
    return vec4<f32>(min(base.rgb + vec3<f32>(spec), vec3<f32>(a)), a);
}
