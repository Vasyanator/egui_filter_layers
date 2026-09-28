/*
File: geometry.rs

Purpose:
Backend-independent pixel math of one filter layer at paint time: which physical-pixel region is
processed, how far the capture is padded, and the UV/pixel values the shaders receive.

Key structures:
- `PxRect`: an integer rect in physical pixels, top-left origin, y down.
- `LayerGeometry`: everything a backend needs to capture, run passes and composite.

Key functions:
- `transform_scale()`: the layer transform's scale, from the painted vs rendered callback rect.
- `layer_geometry()`: pure function from the callback's rect/clip/screen to `LayerGeometry`.
- `padding_px()`: the chain padding in whole pixels (the capture's and the offline tiles' overlap).
- `LayerGeometry::composite_params()`: the composite pass's parameter block (every backend).
- `pointer_uniform()`: the one screen-points -> capture-uv conversion of the pointer (`fl.pointer`),
  used by both backends and the offline renderer.
- `mask_transform()`: the one layer-points -> capture-pixels mapping of a shape mask (F9), with
  the layer transform, used by both backends and the offline renderer.
- `px_f32()`: exact pixel-count -> `f32` conversion shared by the backends.

Notes:
Rounding mirrors `epaint::ViewportInPixels::from_points` (round each edge, clamp to the screen)
so the processed region agrees with the viewport egui's own renderers compute.
*/

use egui::{CornerRadius, Pos2, Rect};

use crate::pass::{PARAM_SLOTS, Params};

/// An integer rect in physical pixels: `x, y` = top-left corner (y down), `w, h` > 0 when valid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PxRect {
    /// Left edge (inclusive).
    pub x: i32,
    /// Top edge (inclusive), counted from the top of the screen.
    pub y: i32,
    /// Width in pixels.
    pub w: i32,
    /// Height in pixels.
    pub h: i32,
}

impl PxRect {
    /// The bottom edge measured from the BOTTOM of a screen `screen_h` pixels high, which is what
    /// OpenGL's `glViewport` / `glBlitFramebuffer` expect for y.
    #[must_use]
    #[cfg_attr(not(feature = "glow"), allow(dead_code))] // Only OpenGL counts y from the bottom.
    pub fn gl_y(&self, screen_h: i32) -> i32 {
        screen_h - self.y - self.h
    }

    /// Size as `[w, h]`.
    #[must_use]
    pub fn size(&self) -> [i32; 2] {
        [self.w, self.h]
    }
}

/// The complete per-layer geometry handed to a backend.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct LayerGeometry {
    /// Region that is re-drawn (layer rect ∩ clip rect, rounded, clamped to the screen).
    pub region: PxRect,
    /// Region expanded by the chain's padding, clamped to the screen; what gets captured.
    pub capture: PxRect,
    /// The layer's full (unclipped) rect in capture UV: `x, y, w, h`. Exposed as `fl.rect_uv`.
    pub rect_uv: [f32; 4],
    /// `region` in capture UV: `x, y, w, h`. Used by the composite pass.
    pub region_uv: [f32; 4],
    /// The layer's full rect in capture pixels: `min.x, min.y, max.x, max.y` (the mask rect).
    pub mask_rect_px: [f32; 4],
    /// Corner radii in physical pixels, ordered `nw, ne, se, sw`.
    pub mask_radii_px: [f32; 4],
    /// Physical pixels per LAYER point: egui's pixels-per-point times the layer transform's
    /// scale. The one factor converting every distance given in layer points (padding, corner
    /// radii, and effect distances via `fl.pixels_per_point`) into pixels.
    pub pixels_per_layer_point: f32,
}

impl LayerGeometry {
    /// The parameter block of the internal composite pass (`shaders/composite.wgsl`): slot 0 =
    /// mask rect in capture pixels, 1 = mask corner radii in pixels, 2 = `region_uv`; the rest zero.
    #[must_use]
    pub(crate) fn composite_params(&self) -> Params {
        let mut params: Params = [[0.0; 4]; PARAM_SLOTS];
        params[0] = self.mask_rect_px;
        params[1] = self.mask_radii_px;
        params[2] = self.region_uv;
        params
    }
}

/// The layer transform's scale: `rendered` (the callback rect egui's renderer reports, layer
/// transform applied) over `painted` (the rect the layer was painted with, in layer points).
///
/// egui applies a layer's `TSTransform` (`Context::set_transform_layer`, a `Scene`, or
/// `Context::transform_layer_shapes`) to every shape of that layer at the end of the pass, the
/// callback rect included (`rect = transform * rect`), so the width ratio IS the transform's
/// `scaling`; untransformed layers are not touched and get exactly `1`. A zero-width painted rect
/// (nothing to draw) yields `1`.
#[must_use]
pub(crate) fn transform_scale(painted: Rect, rendered: Rect) -> f32 {
    if painted.width() > 0.0 { rendered.width() / painted.width() } else { 1.0 }
}

/// Rounds a finite pixel coordinate to `i32`.
///
/// `as` from `f32` saturates at the `i32` bounds and maps NaN to 0 (Rust reference, "Numeric
/// cast"), which is exactly the clamping wanted here; every result is clamped to the screen
/// afterwards anyway.
#[allow(clippy::cast_possible_truncation)] // Saturating float->int cast is the intended semantics (see above).
fn round_px(value: f32) -> i32 {
    value.round() as i32
}

/// Converts a pixel count to `f32`. Pixel coordinates are bounded by the screen size, far below
/// 2^24, so the conversion is exact.
#[allow(clippy::cast_precision_loss)] // Values are screen coordinates (< 2^24): exact in f32.
pub(crate) fn px_f32(value: i32) -> f32 {
    value as f32
}

/// The capture padding in whole pixels: `padding_points` layer points at
/// `pixels_per_layer_point`, rounded up; 0 for a non-finite or non-positive padding.
#[must_use]
pub(crate) fn padding_px(padding_points: f32, pixels_per_layer_point: f32) -> i32 {
    if padding_points.is_finite() && padding_points > 0.0 { round_px((padding_points * pixels_per_layer_point).ceil()) } else { 0 }
}

/// `fl.pointer` for a pointer at `pos` (screen points, i.e. egui's global input coordinates,
/// top-left origin, y down) on a screen of `pixels_per_point`, relative to `capture`: `xy` =
/// `(pos * pixels_per_point - capture.xy) / capture.wh` (capture uv, the same space as an
/// effect's `uv`; outside `0..1` when the pointer is off the capture), `z` = `1` while `pressed`,
/// `w` = `1` (present). `None`, a degenerate capture or a non-finite result gives `[0; 4]`
/// (`w = 0`: no pointer).
///
/// No layer transform is involved: the capture is in screen pixels and egui's pointer input is in
/// untransformed screen points, so a transformed (scaled / translated) layer maps the pointer
/// correctly through its capture alone.
#[must_use]
pub(crate) fn pointer_uniform(pointer: Option<(Pos2, bool)>, pixels_per_point: f32, capture: PxRect) -> [f32; 4] {
    let Some((pos, pressed)) = pointer else {
        return [0.0; 4];
    };
    if capture.w <= 0 || capture.h <= 0 {
        return [0.0; 4];
    }
    let u = (pos.x * pixels_per_point - px_f32(capture.x)) / px_f32(capture.w);
    let v = (pos.y * pixels_per_point - px_f32(capture.y)) / px_f32(capture.h);
    if !(u.is_finite() && v.is_finite()) {
        return [0.0; 4];
    }
    [u, v, if pressed { 1.0 } else { 0.0 }, 1.0]
}

/// How a shape mask's layer points map to capture pixels: `pixel = point * scale - origin_px`
/// (capture pixel `(0, 0)` = the capture's top-left corner).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct MaskTransform {
    /// Capture pixels per layer point (egui's pixels per point times the layer transform's scale).
    pub scale: f32,
    /// Subtracted after scaling.
    pub origin_px: [f32; 2],
}

/// The mapping of a shape mask painted in the space of `painted` (the layer rect as painted, layer
/// points) onto the capture `capture` of a layer rendered at `rendered` (the callback rect, layer
/// transform applied, screen points) on a screen of `pixels_per_point`.
///
/// egui's layer transforms are a uniform scale plus a translation (`TSTransform`): `rendered =
/// t + s * painted`, `s` = [`transform_scale`] and `t = rendered.min - s * painted.min`. A point
/// `p` lands at screen pixel `(t + s * p) * ppp`, i.e. capture pixel `p * (s * ppp) - (capture.xy -
/// t * ppp)`. Untransformed layers (`painted == rendered`) get `t = 0` exactly. Offline the frame is
/// the screen and the layer: `painted = rendered` = the frame in points, `capture` at the origin.
#[must_use]
pub(crate) fn mask_transform(painted: Rect, rendered: Rect, pixels_per_point: f32, capture: PxRect) -> MaskTransform {
    let s = transform_scale(painted, rendered);
    let translation = rendered.min - painted.min * s;
    let scale = pixels_per_point * s;
    MaskTransform { scale, origin_px: [px_f32(capture.x) - translation.x * pixels_per_point, px_f32(capture.y) - translation.y * pixels_per_point] }
}

/// Computes the geometry of one layer, or `None` when nothing of it is visible.
///
/// * `layer_rect` — the layer's rect in points as seen by the renderer (after layer transforms).
/// * `clip_rect` — the clip rect of the callback, in points.
/// * `pixels_per_point`, `screen_size_px` — from `egui::PaintCallbackInfo`.
/// * `padding_points` — the chain's total sample radius (see `FilterChain::padding_points`).
/// * `corner_radius` — mask radii in layer points.
/// * `transform_scale` — the layer transform's scale (see `transform_scale()`); with
///   `pixels_per_point` it forms `LayerGeometry::pixels_per_layer_point`, which converts the
///   padding, the radii and (as `fl.pixels_per_point`) effect distances.
#[must_use]
pub(crate) fn layer_geometry(
    layer_rect: Rect,
    clip_rect: Rect,
    pixels_per_point: f32,
    screen_size_px: [u32; 2],
    padding_points: f32,
    corner_radius: CornerRadius,
    transform_scale: f32,
) -> Option<LayerGeometry> {
    if !(pixels_per_point.is_finite() && pixels_per_point > 0.0) {
        return None;
    }
    let screen_w = i32::try_from(screen_size_px[0]).unwrap_or(i32::MAX);
    let screen_h = i32::try_from(screen_size_px[1]).unwrap_or(i32::MAX);

    let visible = layer_rect.intersect(clip_rect);
    if !visible.is_positive() {
        return None;
    }
    let left = round_px(visible.min.x * pixels_per_point).clamp(0, screen_w);
    let right = round_px(visible.max.x * pixels_per_point).clamp(left, screen_w);
    let top = round_px(visible.min.y * pixels_per_point).clamp(0, screen_h);
    let bottom = round_px(visible.max.y * pixels_per_point).clamp(top, screen_h);
    let region = PxRect { x: left, y: top, w: right - left, h: bottom - top };
    if region.w <= 0 || region.h <= 0 {
        return None;
    }

    // Layer points -> physical pixels. For an untransformed layer `transform_scale` is exactly 1,
    // so this equals `pixels_per_point` bit for bit.
    let pixels_per_layer_point = pixels_per_point * transform_scale.max(0.0);
    let padding = padding_px(padding_points, pixels_per_layer_point);
    let cap_left = left.saturating_sub(padding).max(0);
    let cap_top = top.saturating_sub(padding).max(0);
    let cap_right = right.saturating_add(padding).min(screen_w);
    let cap_bottom = bottom.saturating_add(padding).min(screen_h);
    let capture = PxRect { x: cap_left, y: cap_top, w: cap_right - cap_left, h: cap_bottom - cap_top };

    let cap_x = px_f32(capture.x);
    let cap_y = px_f32(capture.y);
    let cap_w = px_f32(capture.w);
    let cap_h = px_f32(capture.h);
    let full_min_x = layer_rect.min.x * pixels_per_point - cap_x;
    let full_min_y = layer_rect.min.y * pixels_per_point - cap_y;
    let full_max_x = layer_rect.max.x * pixels_per_point - cap_x;
    let full_max_y = layer_rect.max.y * pixels_per_point - cap_y;

    let radius_px = |r: u8| f32::from(r) * pixels_per_layer_point;
    Some(LayerGeometry {
        region,
        capture,
        rect_uv: [full_min_x / cap_w, full_min_y / cap_h, (full_max_x - full_min_x) / cap_w, (full_max_y - full_min_y) / cap_h],
        region_uv: [px_f32(region.x - capture.x) / cap_w, px_f32(region.y - capture.y) / cap_h, px_f32(region.w) / cap_w, px_f32(region.h) / cap_h],
        mask_rect_px: [full_min_x, full_min_y, full_max_x, full_max_y],
        mask_radii_px: [radius_px(corner_radius.nw), radius_px(corner_radius.ne), radius_px(corner_radius.se), radius_px(corner_radius.sw)],
        pixels_per_layer_point,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::{pos2, vec2};

    const SCREEN: [u32; 2] = [800, 600];

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    fn approx4(a: [f32; 4], b: [f32; 4]) -> bool {
        a.iter().zip(b).all(|(x, y)| approx(*x, y))
    }

    #[test]
    fn an_interior_rect_is_padded_on_every_side() {
        let rect = Rect::from_min_size(pos2(100.0, 50.0), vec2(200.0, 100.0));
        let Some(g) = layer_geometry(rect, Rect::EVERYTHING, 2.0, SCREEN, 5.0, CornerRadius::ZERO, 1.0) else { panic!("visible") };
        assert_eq!(g.region, PxRect { x: 200, y: 100, w: 400, h: 200 });
        assert_eq!(g.capture, PxRect { x: 190, y: 90, w: 420, h: 220 });
        assert!(approx(g.rect_uv[0], 10.0 / 420.0) && approx(g.rect_uv[2], 400.0 / 420.0));
        assert!(approx(g.region_uv[1], 10.0 / 220.0) && approx(g.region_uv[3], 200.0 / 220.0));
        assert!(approx4(g.mask_rect_px, [10.0, 10.0, 410.0, 210.0]), "{:?}", g.mask_rect_px);
    }

    #[test]
    fn padding_is_clamped_at_the_screen_edge() {
        let rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(100.0, 100.0));
        let Some(g) = layer_geometry(rect, Rect::EVERYTHING, 1.0, SCREEN, 20.0, CornerRadius::ZERO, 1.0) else { panic!("visible") };
        assert_eq!(g.capture, PxRect { x: 0, y: 0, w: 120, h: 120 });
        assert_eq!(g.region, PxRect { x: 0, y: 0, w: 100, h: 100 });
    }

    #[test]
    fn a_partly_offscreen_rect_is_clamped_and_its_uv_rect_extends_past_the_capture() {
        let rect = Rect::from_min_size(pos2(-50.0, 550.0), vec2(100.0, 100.0));
        let Some(g) = layer_geometry(rect, Rect::EVERYTHING, 1.0, SCREEN, 0.0, CornerRadius::ZERO, 1.0) else { panic!("visible") };
        assert_eq!(g.region, PxRect { x: 0, y: 550, w: 50, h: 50 });
        assert_eq!(g.capture, g.region);
        assert!(g.rect_uv[0] < 0.0, "the full rect starts left of the capture");
        assert!(approx(g.rect_uv[2], 2.0));
        assert_eq!(g.region.gl_y(600), 0);
    }

    #[test]
    fn the_clip_rect_limits_the_processed_region_but_not_the_mask() {
        let rect = Rect::from_min_size(pos2(100.0, 100.0), vec2(200.0, 200.0));
        let clip = Rect::from_min_max(pos2(0.0, 0.0), pos2(200.0, 600.0));
        let Some(g) = layer_geometry(rect, clip, 1.0, SCREEN, 0.0, CornerRadius::same(8), 1.0) else { panic!("visible") };
        assert_eq!(g.region, PxRect { x: 100, y: 100, w: 100, h: 200 });
        assert!(approx4(g.mask_rect_px, [0.0, 0.0, 200.0, 200.0]), "{:?}", g.mask_rect_px);
        assert!(approx4(g.mask_radii_px, [8.0; 4]), "{:?}", g.mask_radii_px);
    }

    #[test]
    fn invisible_or_degenerate_layers_produce_nothing() {
        let rect = Rect::from_min_size(pos2(900.0, 10.0), vec2(50.0, 50.0));
        assert!(layer_geometry(rect, Rect::EVERYTHING, 1.0, SCREEN, 0.0, CornerRadius::ZERO, 1.0).is_none());
        let rect = Rect::from_min_size(pos2(10.0, 10.0), vec2(50.0, 50.0));
        let clip = Rect::from_min_size(pos2(200.0, 200.0), vec2(10.0, 10.0));
        assert!(layer_geometry(rect, clip, 1.0, SCREEN, 0.0, CornerRadius::ZERO, 1.0).is_none());
        assert!(layer_geometry(rect, Rect::EVERYTHING, 0.0, SCREEN, 0.0, CornerRadius::ZERO, 1.0).is_none());
        assert!(layer_geometry(rect, Rect::EVERYTHING, f32::NAN, SCREEN, 0.0, CornerRadius::ZERO, 1.0).is_none());
        let sliver = Rect::from_min_size(pos2(10.0, 10.0), vec2(0.2, 50.0));
        assert!(layer_geometry(sliver, Rect::EVERYTHING, 1.0, SCREEN, 0.0, CornerRadius::ZERO, 1.0).is_none(), "rounds to zero width");
    }

    #[test]
    fn the_transform_scale_is_the_rendered_over_painted_width() {
        let painted = Rect::from_min_size(pos2(10.0, 10.0), vec2(100.0, 40.0));
        assert!(approx(transform_scale(painted, painted), 1.0));
        assert!(approx(transform_scale(painted, Rect::from_min_size(pos2(5.0, 5.0), vec2(50.0, 20.0))), 0.5));
        assert!(approx(transform_scale(painted, Rect::from_min_size(pos2(20.0, 20.0), vec2(200.0, 80.0))), 2.0));
        assert!(approx(transform_scale(Rect::from_min_size(pos2(0.0, 0.0), vec2(0.0, 10.0)), painted), 1.0), "zero width");
    }

    #[test]
    fn padding_radii_and_pixels_per_layer_point_follow_the_transform_scale() {
        let rect = Rect::from_min_size(pos2(100.0, 100.0), vec2(100.0, 100.0));
        let radius = CornerRadius::same(4);
        // ppp 2, 10 layer points of padding: 20 px untransformed, 10 px at scale 0.5, 40 px at scale 2.
        for (scale, per_layer_point) in [(1.0, 2.0), (0.5, 1.0), (2.0, 4.0)] {
            let Some(g) = layer_geometry(rect, Rect::EVERYTHING, 2.0, SCREEN, 10.0, radius, scale) else { panic!("visible") };
            assert!(approx(g.pixels_per_layer_point, per_layer_point), "scale {scale}: {}", g.pixels_per_layer_point);
            let padding = g.region.x - g.capture.x;
            assert!(approx(px_f32(padding), 10.0 * per_layer_point), "scale {scale}: padding {padding}");
            assert!(approx4(g.mask_radii_px, [4.0 * per_layer_point; 4]), "scale {scale}: {:?}", g.mask_radii_px);
        }
    }

    #[test]
    fn non_finite_padding_is_ignored_and_radii_scale_with_ppp_and_transform() {
        let rect = Rect::from_min_size(pos2(100.0, 100.0), vec2(100.0, 100.0));
        let Some(g) = layer_geometry(rect, Rect::EVERYTHING, 2.0, SCREEN, f32::INFINITY, CornerRadius { nw: 1, ne: 2, sw: 3, se: 4 }, 0.5) else { panic!("visible") };
        assert_eq!(g.capture, g.region);
        assert!(approx4(g.mask_radii_px, [1.0, 2.0, 4.0, 3.0]), "{:?}", g.mask_radii_px);
    }

    #[test]
    fn the_pointer_maps_to_capture_uv() {
        let capture = PxRect { x: 100, y: 50, w: 200, h: 100 };
        // ppp 2: screen point (100, 50) is pixel (200, 100) = capture uv (0.5, 0.5).
        assert!(approx4(pointer_uniform(Some((pos2(100.0, 50.0), true)), 2.0, capture), [0.5, 0.5, 1.0, 1.0]));
        assert!(approx4(pointer_uniform(Some((pos2(50.0, 25.0), false)), 2.0, capture), [0.0, 0.0, 0.0, 1.0]));
        // Off the capture: outside 0..1 but still present.
        assert!(approx4(pointer_uniform(Some((pos2(0.0, 0.0), false)), 1.0, capture), [-0.5, -0.5, 0.0, 1.0]));
        assert!(approx4(pointer_uniform(None, 2.0, capture), [0.0; 4]));
        assert!(approx4(pointer_uniform(Some((pos2(1.0, 1.0), true)), f32::NAN, capture), [0.0; 4]));
        assert!(approx4(pointer_uniform(Some((pos2(1.0, 1.0), true)), 1.0, PxRect { x: 0, y: 0, w: 0, h: 5 }), [0.0; 4]));
    }

    /// A shape mask point maps through the layer transform like the layer rect itself: the painted
    /// rect's corners land on the rendered rect's corners in capture pixels.
    #[test]
    fn mask_points_follow_the_layer_transform() {
        let painted = Rect::from_min_size(pos2(20.0, 10.0), vec2(100.0, 50.0));
        let capture = PxRect { x: 30, y: 40, w: 500, h: 300 };
        for transform in [egui::emath::TSTransform::IDENTITY, egui::emath::TSTransform::new(vec2(15.0, -5.0), 1.5)] {
            let rendered = transform * painted;
            let ppp = 2.0;
            let map = mask_transform(painted, rendered, ppp, capture);
            for (point, screen) in [(painted.min, rendered.min), (painted.max, rendered.max), (painted.center(), rendered.center())] {
                let expected = [screen.x * ppp - 30.0, screen.y * ppp - 40.0];
                let got = [point.x * map.scale - map.origin_px[0], point.y * map.scale - map.origin_px[1]];
                assert!(approx(got[0], expected[0]) && approx(got[1], expected[1]), "{transform:?}: {got:?} vs {expected:?}");
            }
        }
        let identity = mask_transform(painted, painted, 1.25, capture);
        assert_eq!(identity.origin_px.map(f32::to_bits), [30.0_f32, 40.0].map(f32::to_bits), "no translation: exactly the capture origin");
    }

    /// A transformed layer: egui reports the callback rect with the layer transform applied and
    /// the pointer in untransformed screen points, so the pointer over the RENDERED layer's centre
    /// lands on the centre of `rect_uv`.
    #[test]
    fn the_pointer_follows_a_transformed_layer() {
        let painted = Rect::from_min_size(pos2(0.0, 0.0), vec2(100.0, 50.0));
        // Scale 2, then translate by (10, 20): rendered (10, 20)..(210, 120).
        let transform = egui::emath::TSTransform::new(vec2(10.0, 20.0), 2.0);
        let rendered = transform * painted;
        let scale = transform_scale(painted, rendered);
        let ppp = 1.5;
        let Some(g) = layer_geometry(rendered, Rect::EVERYTHING, ppp, SCREEN, 4.0, CornerRadius::ZERO, scale) else { panic!("visible") };
        let centre = transform * painted.center();
        let pointer = pointer_uniform(Some((centre, true)), ppp, g.capture);
        let expected = [g.rect_uv[0] + g.rect_uv[2] * 0.5, g.rect_uv[1] + g.rect_uv[3] * 0.5];
        assert!(approx(pointer[0], expected[0]) && approx(pointer[1], expected[1]), "{pointer:?} vs {expected:?}");
        // The rendered rect's top-left corner is `rect_uv.xy`.
        let corner = pointer_uniform(Some((rendered.min, false)), ppp, g.capture);
        assert!(approx(corner[0], g.rect_uv[0]) && approx(corner[1], g.rect_uv[1]), "{corner:?} vs {:?}", g.rect_uv);
    }
}
