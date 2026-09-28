// Internal composite pass: writes the filtered region back over the processed
// region. In frag mode 0 (OpenGL) `frag_uv` is LOCAL to the viewport, which the
// backend sets to the processed region; params[2] = that region inside the
// capture, in capture UV (x, y, w, h). In frag mode 1 (wgpu) `frag_uv` already is
// the capture uv, derived from the pixel position (see the prelude). The original capture is
// the prelude's `fl_original_texture` (binding 3). params[0] = mask rect (min.xy, max.xy) in capture pixels,
// params[1] = corner radii (nw, ne, se, sw) in pixels.
// `effect.rs` appends one of the `effect` wrappers: the replace composite `fl_composite`
// (gamma or linear output; glow and offline), or the two draws of the blended composite the
// wgpu UI draws over egui's pass (erase: `(0, 0, 0, coverage)` under a `dst * (1 - src.a)`
// blend; add: `coverage * filtered` under an additive blend), which together give
// `mix(dst, filtered, coverage)` without touching pixels outside the mask.

fn fl_rounded_rect_distance(p: vec2<f32>, rect: vec4<f32>, radii: vec4<f32>) -> f32 {
    let center = 0.5 * (rect.xy + rect.zw);
    let half_size = 0.5 * (rect.zw - rect.xy);
    let q = p - center;
    let left_radius = select(radii.w, radii.x, q.y < 0.0);
    let right_radius = select(radii.z, radii.y, q.y < 0.0);
    let r = min(select(right_radius, left_radius, q.x < 0.0), min(half_size.x, half_size.y));
    let d = abs(q) - half_size + vec2<f32>(r);
    return length(max(d, vec2<f32>(0.0))) + min(max(d.x, d.y), 0.0) - r;
}

// The capture uv of a composite fragment (frag mode 0: through the processed region).
fn fl_composite_uv(frag_uv: vec2<f32>) -> vec2<f32> {
    let region = fl.params[2];
    return select(region.xy + frag_uv * region.zw, frag_uv, fl.internal_frag_mode > 0.5);
}

// Rounded-rect mask coverage at capture uv `uv`, in [0, 1] (1-px antialiased edge). Values
// within 1/1024 of 0 or 1 snap to exactly 0 or 1: a pixel-aligned edge puts pixel centres at
// distance -0.5 only up to float error (`uv * size_px`), and an almost-1 coverage would make
// the blended composite round differently from the filtered value (1/1024 is under a quarter
// of an 8-bit step).
fn fl_coverage(uv: vec2<f32>) -> f32 {
    let coverage = clamp(0.5 - fl_rounded_rect_distance(uv * fl.size_px, fl.params[0], fl.params[1]), 0.0, 1.0);
    let snap = 1.0 / 1024.0;
    return select(select(coverage, 1.0, coverage > 1.0 - snap), 0.0, coverage < snap);
}

// Coverage, discarding fragments the mask does not cover at all: the blended (wgpu UI)
// composite must leave those pixels exactly as they are.
fn fl_coverage_or_discard(uv: vec2<f32>) -> f32 {
    let coverage = fl_coverage(uv);
    if coverage <= 0.0 {
        discard;
    }
    return coverage;
}

// Replace composite (glow, offline): the whole region is written, the original capture
// outside the mask.
fn fl_composite(frag_uv: vec2<f32>) -> vec4<f32> {
    let uv = fl_composite_uv(frag_uv);
    let filtered = src(uv);
    let original = textureSampleLevel(fl_original_texture, fl_sampler, fl_storage_uv(uv, fl.internal_aux_scale, fl.internal_aux_offset), 0.0);
    let coverage = fl_coverage(uv);
    // Exact at full and zero coverage (`mix` may round `o + (f - o) * 1` away from `f`).
    return select(select(mix(original, filtered, coverage), filtered, coverage >= 1.0), original, coverage <= 0.0);
}
