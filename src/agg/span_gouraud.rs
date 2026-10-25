//! `span_gouraud` and `span_gouraud_rgba` (agg_span_gouraud.h,
//! agg_span_gouraud_rgba.h): a triangle, dilated into a 6-vertex polygon,
//! filled with linearly interpolated colours.

use super::basics::{calc_intersection, cross_product, iround, VertexSource, INTERSECTION_EPSILON};
use super::basics::{PATH_CMD_LINE_TO, PATH_CMD_MOVE_TO, PATH_CMD_STOP};
use super::pixfmt::Rgba8;
use super::renderer::SpanGenerator;

#[derive(Clone, Copy, Default)]
struct Coord {
    x: f64,
    y: f64,
    color: Rgba8,
}

const SUBPIXEL_SHIFT: i32 = 4;
const SUBPIXEL_SCALE: i32 = 1 << SUBPIXEL_SHIFT;

/// `calc_orthogonal` (agg_math.h).
fn calc_orthogonal(thickness: f64, x1: f64, y1: f64, x2: f64, y2: f64) -> (f64, f64) {
    let dx = x2 - x1;
    let dy = y2 - y1;
    let d = (dx * dx + dy * dy).sqrt();
    (thickness * dy / d, -thickness * dx / d)
}

/// `dilate_triangle` (agg_math.h).
#[allow(clippy::too_many_arguments)]
fn dilate_triangle(x1: f64, y1: f64, x2: f64, y2: f64, x3: f64, y3: f64, x: &mut [f64; 8], y: &mut [f64; 8], mut d: f64) {
    let (mut dx1, mut dy1, mut dx2, mut dy2, mut dx3, mut dy3) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let loc = cross_product(x1, y1, x2, y2, x3, y3);
    if loc.abs() > INTERSECTION_EPSILON {
        if cross_product(x1, y1, x2, y2, x3, y3) > 0.0 {
            d = -d;
        }
        (dx1, dy1) = calc_orthogonal(d, x1, y1, x2, y2);
        (dx2, dy2) = calc_orthogonal(d, x2, y2, x3, y3);
        (dx3, dy3) = calc_orthogonal(d, x3, y3, x1, y1);
    }
    x[0] = x1 + dx1;
    y[0] = y1 + dy1;
    x[1] = x2 + dx1;
    y[1] = y2 + dy1;
    x[2] = x2 + dx2;
    y[2] = y2 + dy2;
    x[3] = x3 + dx2;
    y[3] = y3 + dy2;
    x[4] = x3 + dx3;
    y[4] = y3 + dy3;
    x[5] = x1 + dx3;
    y[5] = y1 + dy3;
}

/// `span_gouraud_rgba::rgba_calc`.
#[derive(Clone, Copy, Default)]
struct RgbaCalc {
    x1: f64,
    y1: f64,
    dx: f64,
    inv_dy: f64,
    r1: i32,
    g1: i32,
    b1: i32,
    a1: i32,
    dr: i32,
    dg: i32,
    db: i32,
    da: i32,
    r: i32,
    g: i32,
    b: i32,
    a: i32,
    x: i32,
}

impl RgbaCalc {
    fn init(&mut self, c1: &Coord, c2: &Coord) {
        self.x1 = c1.x - 0.5;
        self.y1 = c1.y - 0.5;
        self.dx = c2.x - c1.x;
        let dy = c2.y - c1.y;
        self.inv_dy = if dy < 1e-5 { 1e5 } else { 1.0 / dy };
        self.r1 = c1.color.r as i32;
        self.g1 = c1.color.g as i32;
        self.b1 = c1.color.b as i32;
        self.a1 = c1.color.a as i32;
        self.dr = c2.color.r as i32 - self.r1;
        self.dg = c2.color.g as i32 - self.g1;
        self.db = c2.color.b as i32 - self.b1;
        self.da = c2.color.a as i32 - self.a1;
    }

    fn calc(&mut self, y: f64) {
        let k = ((y - self.y1) * self.inv_dy).clamp(0.0, 1.0);
        self.r = self.r1 + iround(self.dr as f64 * k);
        self.g = self.g1 + iround(self.dg as f64 * k);
        self.b = self.b1 + iround(self.db as f64 * k);
        self.a = self.a1 + iround(self.da as f64 * k);
        self.x = iround((self.x1 + self.dx * k) * SUBPIXEL_SCALE as f64);
    }
}

/// `dda_line_interpolator<14>`, with C's wrapping integer arithmetic.
struct Dda14 {
    y: i32,
    inc: i32,
    dy: i32,
}

impl Dda14 {
    fn new(y1: i32, y2: i32, count: i32) -> Self {
        Dda14 { y: y1, inc: (y2 - y1).wrapping_shl(14) / count, dy: 0 }
    }

    fn add(&mut self, n: i32) {
        self.dy = self.dy.wrapping_add(self.inc.wrapping_mul(n));
    }

    fn sub(&mut self, n: i32) {
        self.dy = self.dy.wrapping_sub(self.inc.wrapping_mul(n));
    }

    fn y(&self) -> i32 {
        self.y.wrapping_add(self.dy >> 14)
    }
}

/// `span_gouraud_rgba<rgba8>`: both the vertex source of the (dilated)
/// triangle and the span generator filling it.
pub struct SpanGouraudRgba {
    coord: [Coord; 3],
    x: [f64; 8],
    y: [f64; 8],
    cmd: [u32; 8],
    vertex: usize,
    swap: bool,
    y2: i32,
    rgba1: RgbaCalc,
    rgba2: RgbaCalc,
    rgba3: RgbaCalc,
}

impl SpanGouraudRgba {
    #[allow(clippy::too_many_arguments)]
    pub fn new(c1: Rgba8, c2: Rgba8, c3: Rgba8, x1: f64, y1: f64, x2: f64, y2: f64, x3: f64, y3: f64, d: f64) -> Self {
        let mut s = SpanGouraudRgba {
            coord: [Coord::default(); 3],
            x: [0.0; 8],
            y: [0.0; 8],
            cmd: [PATH_CMD_STOP; 8],
            vertex: 0,
            swap: false,
            y2: 0,
            rgba1: RgbaCalc::default(),
            rgba2: RgbaCalc::default(),
            rgba3: RgbaCalc::default(),
        };
        s.coord[0].color = c1;
        s.coord[1].color = c2;
        s.coord[2].color = c3;
        s.triangle(x1, y1, x2, y2, x3, y3, d);
        s
    }

    /// Set the triangle, dilated by `d` (the coordinates interpolating the
    /// colours are the miter joins of the dilated edges).
    #[allow(clippy::too_many_arguments)]
    fn triangle(&mut self, x1: f64, y1: f64, x2: f64, y2: f64, x3: f64, y3: f64, d: f64) {
        (self.coord[0].x, self.coord[0].y) = (x1, y1);
        (self.coord[1].x, self.coord[1].y) = (x2, y2);
        (self.coord[2].x, self.coord[2].y) = (x3, y3);
        (self.x[0], self.y[0]) = (x1, y1);
        (self.x[1], self.y[1]) = (x2, y2);
        (self.x[2], self.y[2]) = (x3, y3);
        self.cmd[0] = PATH_CMD_MOVE_TO;
        self.cmd[1] = PATH_CMD_LINE_TO;
        self.cmd[2] = PATH_CMD_LINE_TO;
        self.cmd[3] = PATH_CMD_STOP;
        if d != 0.0 {
            let c = self.coord;
            dilate_triangle(c[0].x, c[0].y, c[1].x, c[1].y, c[2].x, c[2].y, &mut self.x, &mut self.y, d);
            let (x, y) = (self.x, self.y);
            let [c0, c1, c2] = &mut self.coord;
            calc_intersection(x[4], y[4], x[5], y[5], x[0], y[0], x[1], y[1], &mut c0.x, &mut c0.y);
            calc_intersection(x[0], y[0], x[1], y[1], x[2], y[2], x[3], y[3], &mut c1.x, &mut c1.y);
            calc_intersection(x[2], y[2], x[3], y[3], x[4], y[4], x[5], y[5], &mut c2.x, &mut c2.y);
            self.cmd[3] = PATH_CMD_LINE_TO;
            self.cmd[4] = PATH_CMD_LINE_TO;
            self.cmd[5] = PATH_CMD_LINE_TO;
            self.cmd[6] = PATH_CMD_STOP;
        }
    }

    /// The vertices sorted by y.
    fn arrange_vertices(&self) -> [Coord; 3] {
        let mut coord = self.coord;
        if self.coord[0].y > self.coord[2].y {
            coord[0] = self.coord[2];
            coord[2] = self.coord[0];
        }
        if coord[0].y > coord[1].y {
            coord.swap(0, 1);
        }
        if coord[1].y > coord[2].y {
            coord.swap(1, 2);
        }
        coord
    }
}

impl VertexSource for SpanGouraudRgba {
    fn rewind(&mut self, _path_id: u32) {
        self.vertex = 0;
    }

    fn vertex(&mut self, x: &mut f64, y: &mut f64) -> u32 {
        *x = self.x[self.vertex];
        *y = self.y[self.vertex];
        let c = self.cmd[self.vertex];
        self.vertex += 1;
        c
    }
}

impl SpanGenerator for SpanGouraudRgba {
    fn prepare(&mut self) {
        let coord = self.arrange_vertices();
        self.y2 = coord[1].y as i32;
        self.swap = cross_product(coord[0].x, coord[0].y, coord[2].x, coord[2].y, coord[1].x, coord[1].y) < 0.0;
        self.rgba1.init(&coord[0], &coord[2]);
        self.rgba2.init(&coord[0], &coord[1]);
        self.rgba3.init(&coord[1], &coord[2]);
    }

    fn generate(&mut self, span: &mut [Rgba8], x: i32, y: i32) {
        self.rgba1.calc(y as f64);
        let mut pc1 = self.rgba1;
        let mut pc2 = if y <= self.y2 {
            self.rgba2.calc(y as f64 + self.rgba2.inv_dy);
            self.rgba2
        } else {
            self.rgba3.calc(y as f64 - self.rgba3.inv_dy);
            self.rgba3
        };
        if self.swap {
            std::mem::swap(&mut pc1, &mut pc2);
        }

        // the horizontal length with subpixel accuracy
        let mut nlen = (pc2.x - pc1.x).abs();
        if nlen <= 0 {
            nlen = 1;
        }
        let mut r = Dda14::new(pc1.r, pc2.r, nlen);
        let mut g = Dda14::new(pc1.g, pc2.g, nlen);
        let mut b = Dda14::new(pc1.b, pc2.b, nlen);
        let mut a = Dda14::new(pc1.a, pc2.a, nlen);

        // roll back the interpolators to the start of the span
        let mut start = pc1.x - (x << SUBPIXEL_SHIFT);
        r.sub(start);
        g.sub(start);
        b.sub(start);
        a.sub(start);
        nlen += start;

        let lim = 255;
        let clamped = |v: i32| v.clamp(0, lim) as u8;
        let mut i = 0;
        let len = span.len();
        // beginning (may overflow)
        while i < len && start > 0 {
            span[i] = Rgba8 { r: clamped(r.y()), g: clamped(g.y()), b: clamped(b.y()), a: clamped(a.y()) };
            r.add(SUBPIXEL_SCALE);
            g.add(SUBPIXEL_SCALE);
            b.add(SUBPIXEL_SCALE);
            a.add(SUBPIXEL_SCALE);
            nlen -= SUBPIXEL_SCALE;
            start -= SUBPIXEL_SCALE;
            i += 1;
        }
        // middle (no overflow)
        while i < len && nlen > 0 {
            span[i] = Rgba8 { r: r.y() as u8, g: g.y() as u8, b: b.y() as u8, a: a.y() as u8 };
            r.add(SUBPIXEL_SCALE);
            g.add(SUBPIXEL_SCALE);
            b.add(SUBPIXEL_SCALE);
            a.add(SUBPIXEL_SCALE);
            nlen -= SUBPIXEL_SCALE;
            i += 1;
        }
        // end (may overflow)
        while i < len {
            span[i] = Rgba8 { r: clamped(r.y()), g: clamped(g.y()), b: clamped(b.y()), a: clamped(a.y()) };
            r.add(SUBPIXEL_SCALE);
            g.add(SUBPIXEL_SCALE);
            b.add(SUBPIXEL_SCALE);
            a.add(SUBPIXEL_SCALE);
            i += 1;
        }
    }
}
