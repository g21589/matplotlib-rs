//! `clip_line_segment` from `agg_clip_liang_barsky.h`.

use super::basics::RectD;
use super::rasterizer::{clipping_flags, clipping_flags_y};

const CLIPPING_FLAGS_X1_CLIPPED: u32 = 4;
const CLIPPING_FLAGS_X_CLIPPED: u32 = 4 | 1;
const CLIPPING_FLAGS_Y1_CLIPPED: u32 = 8;
const CLIPPING_FLAGS_Y_CLIPPED: u32 = 8 | 2;

#[allow(clippy::too_many_arguments)]
fn clip_move_point(x1: f64, y1: f64, x2: f64, y2: f64, cb: &RectD, x: &mut f64, y: &mut f64, flags: u32) -> bool {
    if flags & CLIPPING_FLAGS_X_CLIPPED != 0 {
        if x1 == x2 {
            return false;
        }
        let bound = if flags & CLIPPING_FLAGS_X1_CLIPPED != 0 { cb.x1 } else { cb.x2 };
        *y = (bound - x1) * (y2 - y1) / (x2 - x1) + y1;
        *x = bound;
    }

    let flags = clipping_flags_y(*y, cb);
    if flags & CLIPPING_FLAGS_Y_CLIPPED != 0 {
        if y1 == y2 {
            return false;
        }
        let bound = if flags & CLIPPING_FLAGS_Y1_CLIPPED != 0 { cb.y1 } else { cb.y2 };
        *x = (bound - y1) * (x2 - x1) / (y2 - y1) + x1;
        *y = bound;
    }
    true
}

/// Returns: ret >= 4 fully clipped; (ret & 1) first point moved;
/// (ret & 2) second point moved.
pub fn clip_line_segment(x1: &mut f64, y1: &mut f64, x2: &mut f64, y2: &mut f64, cb: &RectD) -> u32 {
    let f1 = clipping_flags(*x1, *y1, cb);
    let f2 = clipping_flags(*x2, *y2, cb);
    let mut ret = 0;

    if (f2 | f1) == 0 {
        // Fully visible
        return 0;
    }

    if (f1 & CLIPPING_FLAGS_X_CLIPPED) != 0 && (f1 & CLIPPING_FLAGS_X_CLIPPED) == (f2 & CLIPPING_FLAGS_X_CLIPPED) {
        // Fully clipped
        return 4;
    }

    if (f1 & CLIPPING_FLAGS_Y_CLIPPED) != 0 && (f1 & CLIPPING_FLAGS_Y_CLIPPED) == (f2 & CLIPPING_FLAGS_Y_CLIPPED) {
        // Fully clipped
        return 4;
    }

    let tx1 = *x1;
    let ty1 = *y1;
    let tx2 = *x2;
    let ty2 = *y2;
    if f1 != 0 {
        if !clip_move_point(tx1, ty1, tx2, ty2, cb, x1, y1, f1) {
            return 4;
        }
        if *x1 == *x2 && *y1 == *y2 {
            return 4;
        }
        ret |= 1;
    }
    if f2 != 0 {
        if !clip_move_point(tx1, ty1, tx2, ty2, cb, x2, y2, f2) {
            return 4;
        }
        if *x1 == *x2 && *y1 == *y2 {
            return 4;
        }
        ret |= 2;
    }
    ret
}
