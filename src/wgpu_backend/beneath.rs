/*
File: wgpu_backend/beneath.rs

Purpose:
GPU-free selection of what lies "beneath" a filter layer in egui's final paint list, for the
wgpu backend's offscreen re-render.

Key functions:
- `find_callback()`: index of a given paint callback (by `Arc` identity) in the flattened list.
- `capture_points()`: the padded capture rect, in points, that the beneath content is clipped to.
- `select_beneath()`: clones and clips the shapes painted before the filter; skips callbacks.
- `strip_callbacks()`: drops callback primitives left after tessellation (nested in `Shape::Vec`).

Notes:
The input is `FullOutput::shapes`: egui's exact paint order (layers flattened by `Order` and area
order, layer transforms applied). Everything BEFORE the filter's own callback shape is beneath
it; nothing after it is. Paint callbacks cannot be replayed into another render pass, so they are
skipped — including filter layers painted lower down (the upper filter sees their unfiltered
backdrop on wgpu).
*/

use std::sync::Arc;

use egui::epaint::{ClippedPrimitive, ClippedShape, Primitive};
use egui::{Rect, Shape};

/// Shapes selected beneath one filter layer, ready for tessellation.
#[derive(Debug, Default)]
pub(super) struct Selection {
    /// Clones of the beneath shapes, clip rects intersected with the capture rect.
    pub shapes: Vec<ClippedShape>,
    /// Top-level paint callbacks that were beneath the filter and had to be skipped.
    pub skipped_callbacks: usize,
}

/// Index of the `Shape::Callback` whose callback object is `target` (compared by address).
#[must_use]
pub(super) fn find_callback(shapes: &[ClippedShape], target: *const ()) -> Option<usize> {
    shapes.iter().position(|clipped| match &clipped.shape {
        Shape::Callback(callback) => std::ptr::eq(Arc::as_ptr(&callback.callback).cast::<()>(), target),
        _ => false,
    })
}

/// The rect (points, global space) the beneath content must cover: the visible part of the
/// filter (`callback_rect ∩ clip_rect`) expanded by the padding plus a two-pixel margin that
/// absorbs the rounding `layer_geometry` applies when it converts to pixels.
///
/// `padding_points` is in the layer's own units; `transform_scale` converts it to global points.
/// Returns `None` when nothing of the filter is visible or `pixels_per_point` is unusable.
#[must_use]
pub(super) fn capture_points(callback_rect: Rect, clip_rect: Rect, padding_points: f32, transform_scale: f32, pixels_per_point: f32) -> Option<Rect> {
    if !(pixels_per_point.is_finite() && pixels_per_point > 0.0) {
        return None;
    }
    let visible = callback_rect.intersect(clip_rect);
    if !visible.is_positive() {
        return None;
    }
    let padding = if padding_points.is_finite() && padding_points > 0.0 { padding_points * transform_scale.max(0.0) } else { 0.0 };
    Some(visible.expand(padding + 2.0 / pixels_per_point))
}

/// Clones the shapes of `beneath` (everything painted before the filter) that can affect
/// `capture`, with their clip rects intersected with it. Top-level paint callbacks are skipped
/// and counted.
#[must_use]
pub(super) fn select_beneath(beneath: &[ClippedShape], capture: Rect) -> Selection {
    let mut selection = Selection::default();
    for ClippedShape { clip_rect, shape } in beneath {
        if matches!(shape, Shape::Callback(_)) {
            selection.skipped_callbacks += 1;
            continue;
        }
        let clip = clip_rect.intersect(capture);
        if !clip.is_positive() || !shape.visual_bounding_rect().intersects(capture) {
            continue;
        }
        selection.shapes.push(ClippedShape { clip_rect: clip, shape: shape.clone() });
    }
    selection
}

/// Removes `Primitive::Callback`s (from callbacks nested inside `Shape::Vec`) and returns how many
/// were dropped.
pub(super) fn strip_callbacks(primitives: &mut Vec<ClippedPrimitive>) -> usize {
    let before = primitives.len();
    primitives.retain(|primitive| matches!(primitive.primitive, Primitive::Mesh(_)));
    before - primitives.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::any::Any;
    use egui::epaint::{Mesh, PaintCallback};
    use egui::{Color32, pos2, vec2};

    /// Address of a paint callback object, the identity `find_callback` matches on.
    fn callback_address(callback: &Arc<dyn Any + Send + Sync>) -> *const () {
        Arc::as_ptr(callback).cast::<()>()
    }

    fn rect_shape(min: (f32, f32), size: (f32, f32)) -> Shape {
        Shape::rect_filled(Rect::from_min_size(pos2(min.0, min.1), vec2(size.0, size.1)), 0.0, Color32::RED)
    }

    fn callback(rect: Rect) -> (Shape, Arc<dyn Any + Send + Sync>) {
        let object: Arc<dyn Any + Send + Sync> = Arc::new(7_u32);
        (Shape::Callback(PaintCallback { rect, callback: Arc::clone(&object) }), object)
    }

    fn clipped(shape: Shape) -> ClippedShape {
        ClippedShape { clip_rect: Rect::EVERYTHING, shape }
    }

    #[test]
    fn the_filter_is_found_by_identity_not_by_equal_contents() {
        let rect = Rect::from_min_size(pos2(10.0, 10.0), vec2(50.0, 50.0));
        let (other, _other_object) = callback(rect);
        let (ours, our_object) = callback(rect);
        let shapes = vec![clipped(rect_shape((0.0, 0.0), (5.0, 5.0))), clipped(other), clipped(ours)];
        assert_eq!(find_callback(&shapes, callback_address(&our_object)), Some(2));
        let (_, unrelated) = callback(rect);
        assert_eq!(find_callback(&shapes, callback_address(&unrelated)), None);
    }

    #[test]
    fn only_shapes_before_the_filter_that_touch_the_capture_are_selected() {
        let capture = Rect::from_min_size(pos2(100.0, 100.0), vec2(100.0, 100.0));
        let (filter, filter_object) = callback(capture);
        let shapes = vec![
            clipped(rect_shape((0.0, 0.0), (400.0, 400.0))),  // covers the capture: kept
            clipped(rect_shape((300.0, 300.0), (10.0, 10.0))), // outside: culled
            clipped(Shape::Noop),                              // nothing to draw: culled
            clipped(filter),
            clipped(rect_shape((120.0, 120.0), (10.0, 10.0))), // painted AFTER the filter
        ];
        let index = find_callback(&shapes, callback_address(&filter_object));
        assert_eq!(index, Some(3));
        let selection = select_beneath(&shapes[..3], capture);
        assert_eq!(selection.shapes.len(), 1);
        assert_eq!(selection.skipped_callbacks, 0);
        assert_eq!(selection.shapes[0].clip_rect, capture, "clip rects are narrowed to the capture");
    }

    #[test]
    fn clip_rects_are_intersected_and_fully_clipped_shapes_dropped() {
        let capture = Rect::from_min_size(pos2(100.0, 100.0), vec2(100.0, 100.0));
        let partial = ClippedShape { clip_rect: Rect::from_min_max(pos2(0.0, 0.0), pos2(150.0, 150.0)), shape: rect_shape((0.0, 0.0), (400.0, 400.0)) };
        let hidden = ClippedShape { clip_rect: Rect::from_min_max(pos2(0.0, 0.0), pos2(50.0, 50.0)), shape: rect_shape((0.0, 0.0), (400.0, 400.0)) };
        let selection = select_beneath(&[partial, hidden], capture);
        assert_eq!(selection.shapes.len(), 1);
        assert_eq!(selection.shapes[0].clip_rect, Rect::from_min_max(pos2(100.0, 100.0), pos2(150.0, 150.0)));
    }

    #[test]
    fn callbacks_beneath_are_skipped_and_counted() {
        let capture = Rect::from_min_size(pos2(0.0, 0.0), vec2(100.0, 100.0));
        let (lower_filter, _object) = callback(capture);
        let selection = select_beneath(&[clipped(rect_shape((0.0, 0.0), (50.0, 50.0))), clipped(lower_filter)], capture);
        assert_eq!(selection.shapes.len(), 1);
        assert_eq!(selection.skipped_callbacks, 1);

        let mut primitives = vec![
            ClippedPrimitive { clip_rect: capture, primitive: Primitive::Mesh(Mesh::default()) },
            ClippedPrimitive { clip_rect: capture, primitive: Primitive::Callback(PaintCallback { rect: capture, callback: Arc::new(1_u8) }) },
        ];
        assert_eq!(strip_callbacks(&mut primitives), 1);
        assert_eq!(primitives.len(), 1);
    }

    #[test]
    fn the_capture_rect_is_padded_in_global_points() {
        let rect = Rect::from_min_size(pos2(100.0, 100.0), vec2(200.0, 100.0));
        let clip = Rect::from_min_max(pos2(0.0, 0.0), pos2(250.0, 1000.0));
        let Some(capture) = capture_points(rect, clip, 10.0, 0.5, 2.0) else { panic!("visible") };
        // visible = x 100..250, padded by 10 * 0.5 + 2 / 2 = 6 points.
        assert_eq!(capture, Rect::from_min_max(pos2(94.0, 94.0), pos2(256.0, 206.0)));
        assert!(capture_points(rect, Rect::from_min_size(pos2(500.0, 0.0), vec2(10.0, 10.0)), 0.0, 1.0, 1.0).is_none());
        assert!(capture_points(rect, clip, 0.0, 1.0, 0.0).is_none());
        let Some(unpadded) = capture_points(rect, Rect::EVERYTHING, f32::NAN, 1.0, 1.0) else { panic!("visible") };
        assert_eq!(unpadded, rect.expand(2.0));
    }
}
