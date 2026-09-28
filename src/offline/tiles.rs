/*
File: offline/tiles.rs

Purpose:
Pure planning of how one offline frame is split into GPU runs: a single whole-image run when the
image fits the texture limit, else a grid of tiles whose windows overlap by the chain's reach, all
allocated at one power-of-two texture size.

Key structures:
- `Tile`: the output rect a run writes and the window (output + overlap) it computes.
- `TilePlan`: the tiles plus the texture allocation shared by all of them.

Key functions:
- `plan()`: the plan for an image size, overlap and size limit.

Notes:
Exactness (see `wgpu_backend/chain.rs` and the prelude's window notes): a window's pixels are
computed with the same float operations as the whole image's, given power-of-two allocations
(then `scale = image / alloc` and `offset = window / alloc` are exact). The whole-image run
allocates exactly the image size, like the UI's capture.
*/

use crate::geometry::PxRect;

use super::error::OfflineError;

/// One GPU run of a frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Tile {
    /// Image pixels this run writes to the result.
    pub output: PxRect,
    /// Image pixels this run computes: `output` grown by the overlap, clamped to the image.
    pub window: PxRect,
}

/// How a frame is split.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TilePlan {
    /// Row-major tiles (one tile covering the image when it fits).
    pub tiles: Vec<Tile>,
    /// Size every texture of the run is allocated at: the image size when untiled, else the
    /// power-of-two size covering the largest window.
    pub alloc: [u32; 2],
}

/// Plans a `size` image (both sides in `1..2^24`) for textures of at most `max_side` pixels:
/// untiled when both sides fit, else tiles of windows at most `prev_power_of_two(max_side)`
/// pixels, overlapping by `overlap` pixels on every inner side.
///
/// # Errors
/// [`OfflineError::PaddingTooLarge`] when `2 * overlap` leaves no room for output pixels, and
/// [`OfflineError::InvalidOption`] for a zero `max_side`.
pub(crate) fn plan(size: [u32; 2], overlap: u32, max_side: u32) -> Result<TilePlan, OfflineError> {
    if max_side == 0 {
        return Err(OfflineError::InvalidOption { option: "max_tile_size", reason: "must be at least 1 pixel".to_owned() });
    }
    let image = PxRect { x: 0, y: 0, w: to_i32(size[0])?, h: to_i32(size[1])? };
    if size[0] <= max_side && size[1] <= max_side {
        return Ok(TilePlan { tiles: vec![Tile { output: image, window: image }], alloc: size });
    }
    // Power-of-two windows keep the storage mapping exact (see the file notes).
    let window_limit = 1_u32 << (u32::BITS - 1 - max_side.leading_zeros());
    let content = overlap.checked_mul(2).and_then(|both| window_limit.checked_sub(both)).filter(|&content| content > 0).ok_or(OfflineError::PaddingTooLarge { overlap_px: overlap, max_tile_px: window_limit })?;
    let columns = spans(size[0], content, overlap, window_limit)?;
    let rows = spans(size[1], content, overlap, window_limit)?;
    let largest = |spans: &[(i32, i32, i32, i32)]| spans.iter().map(|&(_, _, _, window_len)| window_len).max().unwrap_or(1);
    let alloc = [next_power_of_two(largest(&columns))?, next_power_of_two(largest(&rows))?];
    let tiles = rows
        .iter()
        .flat_map(|&(y, h, window_y, window_h)| columns.iter().map(move |&(x, w, window_x, window_w)| Tile { output: PxRect { x, y, w, h }, window: PxRect { x: window_x, y: window_y, w: window_w, h: window_h } }))
        .collect();
    Ok(TilePlan { tiles, alloc })
}

/// Splits one axis of `length` pixels: `(start, len, window_start, window_len)` per span. An
/// axis that fits `window_limit` is one span with no overlap; otherwise spans of `content`
/// pixels whose windows reach `overlap` further on each side (clamped to the axis).
fn spans(length: u32, content: u32, overlap: u32, window_limit: u32) -> Result<Vec<(i32, i32, i32, i32)>, OfflineError> {
    let length = to_i32(length)?;
    if length <= to_i32(window_limit)? {
        return Ok(vec![(0, length, 0, length)]);
    }
    let (content, overlap) = (to_i32(content)?, to_i32(overlap)?);
    let mut spans = Vec::new();
    let mut start = 0;
    while start < length {
        let end = start.saturating_add(content).min(length);
        let window_start = start.saturating_sub(overlap).max(0);
        let window_end = end.saturating_add(overlap).min(length);
        spans.push((start, end - start, window_start, window_end - window_start));
        start = end;
    }
    Ok(spans)
}

/// The smallest power of two `>= value` (`value` is a window length, `1..=2^31`).
fn next_power_of_two(value: i32) -> Result<u32, OfflineError> {
    u32::try_from(value.max(1)).ok().and_then(u32::checked_next_power_of_two).ok_or(OfflineError::SizeOverflow { what: "a tile texture" })
}

/// A pixel count as `i32` (image sides are below 2^24).
fn to_i32(value: u32) -> Result<i32, OfflineError> {
    i32::try_from(value).map_err(|_| OfflineError::SizeOverflow { what: "an image side" })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every image pixel is written by exactly one tile, and every window contains its output
    /// grown by `overlap` (clamped to the image) and fits the allocation.
    fn check_cover(size: [u32; 2], overlap: u32, plan: &TilePlan) {
        let [w, h] = size.map(|side| usize::try_from(side).unwrap_or(0));
        let mut hits = vec![0_u8; w * h];
        for tile in &plan.tiles {
            let (o, win) = (tile.output, tile.window);
            let grown = |start: i32, len: i32, limit: u32| ((start - i32::try_from(overlap).unwrap_or(0)).max(0), (start + len + i32::try_from(overlap).unwrap_or(0)).min(i32::try_from(limit).unwrap_or(0)));
            let (gx0, gx1) = grown(o.x, o.w, size[0]);
            let (gy0, gy1) = grown(o.y, o.h, size[1]);
            let single_x = plan.tiles.iter().all(|t| t.window.w == i32::try_from(size[0]).unwrap_or(0));
            let single_y = plan.tiles.iter().all(|t| t.window.h == i32::try_from(size[1]).unwrap_or(0));
            assert!(single_x || (win.x <= gx0 && win.x + win.w >= gx1), "{tile:?}");
            assert!(single_y || (win.y <= gy0 && win.y + win.h >= gy1), "{tile:?}");
            assert!(u32::try_from(win.w).unwrap_or(u32::MAX) <= plan.alloc[0] && u32::try_from(win.h).unwrap_or(u32::MAX) <= plan.alloc[1]);
            for y in o.y..o.y + o.h {
                for x in o.x..o.x + o.w {
                    hits[usize::try_from(y).unwrap_or(0) * w + usize::try_from(x).unwrap_or(0)] += 1;
                }
            }
        }
        assert!(hits.iter().all(|&count| count == 1), "every pixel exactly once");
    }

    #[test]
    fn an_image_that_fits_is_one_exact_size_run() {
        let Ok(plan) = plan([300, 200], 10, 300) else { panic!("fits") };
        assert_eq!(plan.tiles.len(), 1);
        assert_eq!(plan.alloc, [300, 200]);
        assert_eq!(plan.tiles[0].output, plan.tiles[0].window);
    }

    #[test]
    fn large_images_are_tiled_with_overlapping_power_of_two_windows() {
        for (size, overlap, max) in [([300, 200], 10, 100), ([1000, 50], 3, 256), ([97, 1013], 0, 64), ([500, 500], 31, 64)] {
            let Ok(plan) = plan(size, overlap, max) else { panic!("{size:?} {overlap} {max}") };
            assert!(plan.tiles.len() > 1, "{size:?}");
            assert!(plan.alloc.iter().all(|side| side.is_power_of_two() && *side <= max), "{:?}", plan.alloc);
            check_cover(size, overlap, &plan);
        }
    }

    #[test]
    fn the_cap_holds_on_an_axis_that_fits_it() {
        // Width 300 fits the cap 300, height 1000 does not: the plan is tiled, and the width must
        // not be allocated at next_power_of_two(300) = 512 > 300.
        for (size, overlap, max) in [([300, 1000], 10, 300), ([1000, 300], 10, 300), ([257, 5000], 4, 4096), ([300, 1000], 0, 300)] {
            let Ok(plan) = plan(size, overlap, max) else { panic!("{size:?} {overlap} {max}") };
            assert!(plan.tiles.len() > 1, "{size:?}");
            assert!(plan.alloc.iter().all(|side| *side <= max), "{size:?} cap {max}: alloc {:?}", plan.alloc);
            check_cover(size, overlap, &plan);
        }
    }

    #[test]
    fn an_overlap_that_leaves_no_content_is_rejected() {
        assert!(matches!(plan([300, 300], 32, 64), Err(OfflineError::PaddingTooLarge { overlap_px: 32, max_tile_px: 64 })));
        assert!(matches!(plan([300, 300], 1, 0), Err(OfflineError::InvalidOption { .. })));
        // Fits untiled: the overlap does not matter.
        assert!(plan([64, 64], 1000, 64).is_ok());
    }
}
