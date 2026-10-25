//! Basic definitions ported from Anti-Grain Geometry 2.4 (`agg_basics.h`,
//! `agg_math.h`).  The numeric behaviour (rounding, epsilons) intentionally
//! mirrors the C++ originals so that rasterization results are bit-identical
//! to matplotlib's Agg backend.

pub const PATH_CMD_STOP: u32 = 0;
pub const PATH_CMD_MOVE_TO: u32 = 1;
pub const PATH_CMD_LINE_TO: u32 = 2;
pub const PATH_CMD_CURVE3: u32 = 3;
pub const PATH_CMD_CURVE4: u32 = 4;
pub const PATH_CMD_END_POLY: u32 = 0x0F;
pub const PATH_CMD_MASK: u32 = 0x0F;

pub const PATH_FLAGS_NONE: u32 = 0;
pub const PATH_FLAGS_CCW: u32 = 0x10;
pub const PATH_FLAGS_CW: u32 = 0x20;
pub const PATH_FLAGS_CLOSE: u32 = 0x40;

/// matplotlib's CLOSEPOLY code (`path_cmd_end_poly | path_flags_close`).
pub const CLOSEPOLY: u32 = PATH_CMD_END_POLY | PATH_FLAGS_CLOSE;

pub const POLY_SUBPIXEL_SHIFT: i32 = 8;
pub const POLY_SUBPIXEL_SCALE: i32 = 1 << POLY_SUBPIXEL_SHIFT;
pub const POLY_SUBPIXEL_MASK: i32 = POLY_SUBPIXEL_SCALE - 1;

pub const COVER_SHIFT: u32 = 8;
pub const COVER_MASK: u32 = (1 << COVER_SHIFT) - 1;
pub const COVER_FULL: u8 = 255;

pub const PI: f64 = std::f64::consts::PI;
pub const VERTEX_DIST_EPSILON: f64 = 1e-14;
pub const INTERSECTION_EPSILON: f64 = 1.0e-30;

#[inline(always)]
pub fn is_vertex(c: u32) -> bool {
    c >= PATH_CMD_MOVE_TO && c < PATH_CMD_END_POLY
}
#[inline(always)]
pub fn is_drawing(c: u32) -> bool {
    c >= PATH_CMD_LINE_TO && c < PATH_CMD_END_POLY
}
#[inline(always)]
pub fn is_stop(c: u32) -> bool {
    c == PATH_CMD_STOP
}
#[inline(always)]
pub fn is_move_to(c: u32) -> bool {
    c == PATH_CMD_MOVE_TO
}
#[inline(always)]
pub fn is_line_to(c: u32) -> bool {
    c == PATH_CMD_LINE_TO
}
#[inline(always)]
pub fn is_curve(c: u32) -> bool {
    c == PATH_CMD_CURVE3 || c == PATH_CMD_CURVE4
}
#[inline(always)]
pub fn is_end_poly(c: u32) -> bool {
    (c & PATH_CMD_MASK) == PATH_CMD_END_POLY
}
#[inline(always)]
pub fn is_close(c: u32) -> bool {
    (c & !(PATH_FLAGS_CW | PATH_FLAGS_CCW)) == (PATH_CMD_END_POLY | PATH_FLAGS_CLOSE)
}
#[inline(always)]
pub fn get_close_flag(c: u32) -> u32 {
    c & PATH_FLAGS_CLOSE
}

/// `agg::iround` (non-FISTP variant): round half away from zero then truncate.
#[inline(always)]
pub fn iround(v: f64) -> i32 {
    (if v < 0.0 { v - 0.5 } else { v + 0.5 }) as i32
}

/// `agg::uround`: `unsigned(v + 0.5)`.
#[inline(always)]
pub fn uround(v: f64) -> u32 {
    let t = v + 0.5;
    if t <= 0.0 {
        0
    } else {
        t as u32
    }
}

/// matplotlib's `mpl_round_to_int`.
#[inline(always)]
pub fn mpl_round_to_int(v: f64) -> i32 {
    (v + if v >= 0.0 { 0.5 } else { -0.5 }) as i32
}

#[inline(always)]
pub fn mpl_round(v: f64) -> f64 {
    mpl_round_to_int(v) as f64
}

#[inline(always)]
pub fn cross_product(x1: f64, y1: f64, x2: f64, y2: f64, x: f64, y: f64) -> f64 {
    (x - x2) * (y2 - y1) - (y - y2) * (x2 - x1)
}

#[inline(always)]
pub fn calc_distance(x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    let dx = x2 - x1;
    let dy = y2 - y1;
    (dx * dx + dy * dy).sqrt()
}

#[inline(always)]
pub fn calc_sq_distance(x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    let dx = x2 - x1;
    let dy = y2 - y1;
    dx * dx + dy * dy
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn calc_intersection(
    ax: f64,
    ay: f64,
    bx: f64,
    by: f64,
    cx: f64,
    cy: f64,
    dx: f64,
    dy: f64,
    x: &mut f64,
    y: &mut f64,
) -> bool {
    let num = (ay - cy) * (dx - cx) - (ax - cx) * (dy - cy);
    let den = (bx - ax) * (dy - cy) - (by - ay) * (dx - cx);
    if den.abs() < INTERSECTION_EPSILON {
        return false;
    }
    let r = num / den;
    *x = ax + r * (bx - ax);
    *y = ay + r * (by - ay);
    true
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RectI {
    pub x1: i32,
    pub y1: i32,
    pub x2: i32,
    pub y2: i32,
}

impl RectI {
    pub fn new(x1: i32, y1: i32, x2: i32, y2: i32) -> Self {
        RectI { x1, y1, x2, y2 }
    }
    pub fn normalize(&mut self) {
        if self.x1 > self.x2 {
            std::mem::swap(&mut self.x1, &mut self.x2);
        }
        if self.y1 > self.y2 {
            std::mem::swap(&mut self.y1, &mut self.y2);
        }
    }
    /// `rect_base::clip`: intersect with `r`, returning whether the result is valid.
    pub fn clip(&mut self, r: &RectI) -> bool {
        if self.x2 > r.x2 {
            self.x2 = r.x2;
        }
        if self.y2 > r.y2 {
            self.y2 = r.y2;
        }
        if self.x1 < r.x1 {
            self.x1 = r.x1;
        }
        if self.y1 < r.y1 {
            self.y1 = r.y1;
        }
        self.x1 <= self.x2 && self.y1 <= self.y2
    }
    pub fn is_valid(&self) -> bool {
        self.x1 <= self.x2 && self.y1 <= self.y2
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RectD {
    pub x1: f64,
    pub y1: f64,
    pub x2: f64,
    pub y2: f64,
}

impl RectD {
    pub fn new(x1: f64, y1: f64, x2: f64, y2: f64) -> Self {
        RectD { x1, y1, x2, y2 }
    }
    pub fn normalize(&mut self) {
        if self.x1 > self.x2 {
            std::mem::swap(&mut self.x1, &mut self.x2);
        }
        if self.y1 > self.y2 {
            std::mem::swap(&mut self.y1, &mut self.y2);
        }
    }
    #[inline]
    pub fn hit_test(&self, x: f64, y: f64) -> bool {
        x >= self.x1 && x <= self.x2 && y >= self.y1 && y <= self.y2
    }
}

/// The Agg "vertex source" protocol.
pub trait VertexSource {
    fn rewind(&mut self, path_id: u32);
    fn vertex(&mut self, x: &mut f64, y: &mut f64) -> u32;
}

impl<T: VertexSource + ?Sized> VertexSource for &mut T {
    #[inline(always)]
    fn rewind(&mut self, path_id: u32) {
        (**self).rewind(path_id)
    }
    #[inline(always)]
    fn vertex(&mut self, x: &mut f64, y: &mut f64) -> u32 {
        (**self).vertex(x, y)
    }
}

impl<T: VertexSource + ?Sized> VertexSource for Box<T> {
    #[inline(always)]
    fn rewind(&mut self, path_id: u32) {
        (**self).rewind(path_id)
    }
    #[inline(always)]
    fn vertex(&mut self, x: &mut f64, y: &mut f64) -> u32 {
        (**self).vertex(x, y)
    }
}

/// A simple stored path (vertices + commands), the analogue of
/// `agg::path_storage` and of matplotlib's `PathIterator`.
#[derive(Clone, Debug, Default)]
pub struct PathStorage {
    pub cmds: Vec<u8>,
    pub xy: Vec<[f64; 2]>,
    iter: usize,
}

impl PathStorage {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn move_to(&mut self, x: f64, y: f64) {
        self.cmds.push(PATH_CMD_MOVE_TO as u8);
        self.xy.push([x, y]);
    }
    pub fn line_to(&mut self, x: f64, y: f64) {
        self.cmds.push(PATH_CMD_LINE_TO as u8);
        self.xy.push([x, y]);
    }
    pub fn close_polygon(&mut self) {
        self.cmds.push(CLOSEPOLY as u8);
        self.xy.push([0.0, 0.0]);
    }
}

impl VertexSource for PathStorage {
    fn rewind(&mut self, path_id: u32) {
        self.iter = path_id as usize;
    }
    fn vertex(&mut self, x: &mut f64, y: &mut f64) -> u32 {
        if self.iter >= self.cmds.len() {
            *x = 0.0;
            *y = 0.0;
            return PATH_CMD_STOP;
        }
        let i = self.iter;
        self.iter += 1;
        *x = self.xy[i][0];
        *y = self.xy[i][1];
        self.cmds[i] as u32
    }
}
