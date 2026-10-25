//! Port of FreeType 2.14's anti-aliasing rasterizer (`src/smooth/ftgrays.c`,
//! the `FT_INT64` code paths), producing bit-identical coverage values.
//!
//! Portions of this file are derived from FreeType:
//! Copyright (C) 2000-2025 by David Turner, Robert Wilhelm, and Werner Lemberg.
//! Used under the FreeType License (FTL).

use super::outline::Outline;

const PIXEL_BITS: i32 = 8;
const ONE_PIXEL: i64 = 1 << PIXEL_BITS;

#[inline(always)]
fn trunc(x: i64) -> i32 {
    (x >> PIXEL_BITS) as i32
}
#[inline(always)]
fn fract(x: i64) -> i32 {
    (x & (ONE_PIXEL - 1)) as i32
}
#[inline(always)]
fn upscale(x: i64) -> i64 {
    x * (ONE_PIXEL >> 6)
}

#[derive(Clone, Copy, Debug)]
struct Cell {
    x: i32,
    cover: i32,
    area: i32,
}

/// Where the current cell lives.
#[derive(Clone, Copy)]
enum CurCell {
    Null,
    At(usize, usize), // (row index, position in row)
}

struct Worker {
    min_ex: i32,
    max_ex: i32,
    min_ey: i32,
    max_ey: i32,
    count_ey: i32,
    rows: Vec<Vec<Cell>>,
    cell: CurCell,
    x: i64,
    y: i64,
}

impl Worker {
    fn new(min_ex: i32, min_ey: i32, max_ex: i32, max_ey: i32) -> Self {
        let count_ey = max_ey - min_ey;
        Worker {
            min_ex,
            max_ex,
            min_ey,
            max_ey,
            count_ey,
            rows: vec![Vec::new(); count_ey.max(0) as usize],
            cell: CurCell::Null,
            x: 0,
            y: 0,
        }
    }

    #[inline]
    fn set_cell(&mut self, ex: i32, ey: i32) {
        let ey_index = ey - self.min_ey;
        if ey_index < 0 || ey_index >= self.count_ey || ex >= self.max_ex {
            self.cell = CurCell::Null;
            return;
        }
        let ex = ex.max(self.min_ex - 1);
        let row = &mut self.rows[ey_index as usize];
        // the list is ordered by x
        let mut pos = 0;
        while pos < row.len() {
            let cx = row[pos].x;
            if cx > ex {
                break;
            }
            if cx == ex {
                self.cell = CurCell::At(ey_index as usize, pos);
                return;
            }
            pos += 1;
        }
        row.insert(pos, Cell { x: ex, cover: 0, area: 0 });
        self.cell = CurCell::At(ey_index as usize, pos);
    }

    #[inline(always)]
    fn integrate(&mut self, a: i32, b: i32) {
        if let CurCell::At(r, p) = self.cell {
            let c = &mut self.rows[r][p];
            c.cover = c.cover.wrapping_add(a);
            c.area = c.area.wrapping_add(a.wrapping_mul(b));
        }
    }

    fn render_line(&mut self, to_x: i64, to_y: i64) {
        let mut ey1 = trunc(self.y);
        let ey2 = trunc(to_y);

        // perform vertical clipping
        if (ey1 >= self.max_ey && ey2 >= self.max_ey) || (ey1 < self.min_ey && ey2 < self.min_ey) {
            self.x = to_x;
            self.y = to_y;
            return;
        }

        let mut ex1 = trunc(self.x);
        let ex2 = trunc(to_x);

        let mut fx1 = fract(self.x);
        let mut fy1 = fract(self.y);

        let dx = to_x - self.x;
        let dy = to_y - self.y;

        if ex1 == ex2 && ey1 == ey2 {
            // inside one cell
        } else if dy == 0 {
            // any horizontal line
            self.set_cell(ex2, ey2);
            self.x = to_x;
            self.y = to_y;
            return;
        } else if dx == 0 {
            if dy > 0 {
                // vertical line up
                loop {
                    let fy2 = ONE_PIXEL as i32;
                    self.integrate(fy2 - fy1, fx1 * 2);
                    fy1 = 0;
                    ey1 += 1;
                    self.set_cell(ex1, ey1);
                    if ey1 == ey2 {
                        break;
                    }
                }
            } else {
                // vertical line down
                loop {
                    let fy2 = 0;
                    self.integrate(fy2 - fy1, fx1 * 2);
                    fy1 = ONE_PIXEL as i32;
                    ey1 -= 1;
                    self.set_cell(ex1, ey1);
                    if ey1 == ey2 {
                        break;
                    }
                }
            }
        } else {
            // any other line
            let mut prod: i64 = dx * fy1 as i64 - dy * fx1 as i64;
            let dx_r: i64 = if ex1 != ex2 { 0xFFFFFFFFi64 / dx } else { 0 };
            let dy_r: i64 = if ey1 != ey2 { 0xFFFFFFFFi64 / dy } else { 0 };
            #[inline(always)]
            fn udiv(a: i64, r: i64) -> i32 {
                (((a as u64).wrapping_mul(r as u64)) >> 32) as i32
            }

            // The fundamental value `prod' determines which side and the
            // exact coordinate where the line exits current cell.  It is
            // also easily updated when moving from one cell to the next.
            loop {
                let fx2;
                let fy2;
                if prod - dx * ONE_PIXEL > 0 && prod <= 0 {
                    // left
                    fx2 = 0;
                    fy2 = udiv(-prod, -dx_r);
                    prod -= dy * ONE_PIXEL;
                    self.integrate(fy2 - fy1, fx1 + fx2);
                    fx1 = ONE_PIXEL as i32;
                    fy1 = fy2;
                    ex1 -= 1;
                } else if prod - dx * ONE_PIXEL + dy * ONE_PIXEL > 0 && prod - dx * ONE_PIXEL <= 0 {
                    // up
                    prod -= dx * ONE_PIXEL;
                    fx2 = udiv(-prod, dy_r);
                    fy2 = ONE_PIXEL as i32;
                    self.integrate(fy2 - fy1, fx1 + fx2);
                    fx1 = fx2;
                    fy1 = 0;
                    ey1 += 1;
                } else if prod + dy * ONE_PIXEL >= 0 && prod - dx * ONE_PIXEL + dy * ONE_PIXEL <= 0 {
                    // right
                    prod += dy * ONE_PIXEL;
                    fx2 = ONE_PIXEL as i32;
                    fy2 = udiv(prod, dx_r);
                    self.integrate(fy2 - fy1, fx1 + fx2);
                    fx1 = 0;
                    fy1 = fy2;
                    ex1 += 1;
                } else {
                    // down
                    fx2 = udiv(prod, -dy_r);
                    fy2 = 0;
                    prod += dx * ONE_PIXEL;
                    self.integrate(fy2 - fy1, fx1 + fx2);
                    fx1 = fx2;
                    fy1 = ONE_PIXEL as i32;
                    ey1 -= 1;
                }

                self.set_cell(ex1, ey1);

                if !(ex1 != ex2 || ey1 != ey2) {
                    break;
                }
            }
        }

        let fx2 = fract(to_x);
        let fy2 = fract(to_y);

        self.integrate(fy2 - fy1, fx1 + fx2);

        self.x = to_x;
        self.y = to_y;
    }

    fn render_conic(&mut self, control: (i64, i64), to: (i64, i64)) {
        let p0 = (self.x, self.y);
        let p1 = (upscale(control.0), upscale(control.1));
        let p2 = (upscale(to.0), upscale(to.1));

        // short-cut the arc that crosses the current band
        if (trunc(p0.1) >= self.max_ey && trunc(p1.1) >= self.max_ey && trunc(p2.1) >= self.max_ey)
            || (trunc(p0.1) < self.min_ey && trunc(p1.1) < self.min_ey && trunc(p2.1) < self.min_ey)
        {
            self.x = p2.0;
            self.y = p2.1;
            return;
        }

        let bx = p1.0 - p0.0;
        let by = p1.1 - p0.1;
        let ax = p2.0 - p1.0 - bx; // p0.x + p2.x - 2 * p1.x
        let ay = p2.1 - p1.1 - by; // p0.y + p2.y - 2 * p1.y

        let mut dx = ax.abs();
        let dy = ay.abs();
        if dx < dy {
            dx = dy;
        }

        if dx <= ONE_PIXEL / 4 {
            self.render_line(p2.0, p2.1);
            return;
        }

        // We can calculate the number of necessary segments because
        // each bisection predictably reduces deviation exactly 4-fold.
        // Even 32-bit deviation would vanish after 16 bisections.
        let mut shift: u32 = 16;
        loop {
            dx >>= 2;
            shift -= 1;
            if dx <= ONE_PIXEL / 4 {
                break;
            }
        }
        let mut count: u32 = 0x10000u32 >> shift;

        #[inline(always)]
        fn left_shift(a: i64, b: u32) -> i64 {
            ((a as u64) << b) as i64
        }

        let mut rx = left_shift(ax, shift + shift);
        let mut ry = left_shift(ay, shift + shift);

        let mut qx = left_shift(bx, shift + 17).wrapping_add(rx);
        let mut qy = left_shift(by, shift + 17).wrapping_add(ry);

        rx = rx.wrapping_mul(2);
        ry = ry.wrapping_mul(2);

        let mut px = left_shift(p0.0, 32);
        let mut py = left_shift(p0.1, 32);

        loop {
            px = px.wrapping_add(qx);
            py = py.wrapping_add(qy);
            qx = qx.wrapping_add(rx);
            qy = qy.wrapping_add(ry);

            self.render_line(px >> 32, py >> 32);
            count -= 1;
            if count == 0 {
                break;
            }
        }
    }

    fn render_cubic(&mut self, control1: (i64, i64), control2: (i64, i64), to: (i64, i64)) {
        // bez_stack[16 * 3 + 1]
        let mut stack = [(0i64, 0i64); 16 * 3 + 1];
        let mut arc = 0usize;
        stack[0] = (upscale(to.0), upscale(to.1));
        stack[1] = (upscale(control2.0), upscale(control2.1));
        stack[2] = (upscale(control1.0), upscale(control1.1));
        stack[3] = (self.x, self.y);

        // short-cut the arc that crosses the current band
        if (trunc(stack[0].1) >= self.max_ey
            && trunc(stack[1].1) >= self.max_ey
            && trunc(stack[2].1) >= self.max_ey
            && trunc(stack[3].1) >= self.max_ey)
            || (trunc(stack[0].1) < self.min_ey
                && trunc(stack[1].1) < self.min_ey
                && trunc(stack[2].1) < self.min_ey
                && trunc(stack[3].1) < self.min_ey)
        {
            self.x = stack[0].0;
            self.y = stack[0].1;
            return;
        }

        loop {
            let a = &stack[arc..arc + 4];
            // with each split, control points quickly converge towards
            // chord trisection points and the vanishing distances below
            // indicate when the segment is flat enough to draw
            let split = (2 * a[0].0 - 3 * a[1].0 + a[3].0).abs() > ONE_PIXEL / 2
                || (2 * a[0].1 - 3 * a[1].1 + a[3].1).abs() > ONE_PIXEL / 2
                || (a[0].0 - 3 * a[2].0 + 2 * a[3].0).abs() > ONE_PIXEL / 2
                || (a[0].1 - 3 * a[2].1 + 2 * a[3].1).abs() > ONE_PIXEL / 2;
            if !split {
                let (x0, y0) = stack[arc];
                self.render_line(x0, y0);
                if arc == 0 {
                    return;
                }
                arc -= 3;
                continue;
            }
            split_cubic(&mut stack[arc..arc + 7]);
            arc += 3;
        }
    }

    fn move_to(&mut self, to: (i64, i64)) {
        let x = upscale(to.0);
        let y = upscale(to.1);
        self.set_cell(trunc(x), trunc(y));
        self.x = x;
        self.y = y;
    }

    fn line_to(&mut self, to: (i64, i64)) {
        self.render_line(upscale(to.0), upscale(to.1));
    }

    /// `FT_Outline_Decompose` with shift = 0 and delta = 0.
    fn decompose(&mut self, outline: &Outline) -> bool {
        const TAG_ON: u8 = 1;
        const TAG_CONIC: u8 = 0;
        const TAG_CUBIC: u8 = 2;
        let tag_of = |t: u8| t & 3;

        let pts = &outline.points;
        let tags = &outline.tags;
        let mut last: i32 = -1;
        for &c in &outline.contours {
            let first = (last + 1) as usize;
            last = c as i32;
            if (last as usize) < first {
                return false;
            }
            let mut limit = last as usize;

            let mut v_start = pts[first];
            let v_last = pts[last as usize];
            let mut v_control;

            let mut point = first as isize;
            let mut tag = tag_of(tags[first]);

            // A contour cannot start with a cubic control point!
            if tag == TAG_CUBIC {
                return false;
            }

            // check first point to determine origin
            if tag == TAG_CONIC {
                // first point is conic control.  Yes, this happens.
                if tag_of(tags[last as usize]) == TAG_ON {
                    // start at last point if it is on the curve
                    v_start = v_last;
                    limit -= 1;
                } else {
                    // if both first and last points are conic,
                    // start at their middle and record its position
                    // for closure
                    v_start.0 = (v_start.0 + v_last.0) / 2;
                    v_start.1 = (v_start.1 + v_last.1) / 2;
                }
                point -= 1;
            }

            self.move_to(v_start);

            let mut closed_by_curve = false;
            'outer: while point < limit as isize {
                point += 1;
                tag = tag_of(tags[point as usize]);
                match tag {
                    TAG_ON => {
                        self.line_to(pts[point as usize]);
                        continue;
                    }
                    TAG_CONIC => {
                        v_control = pts[point as usize];
                        loop {
                            // Do_Conic:
                            if point < limit as isize {
                                point += 1;
                                tag = tag_of(tags[point as usize]);
                                let vec = pts[point as usize];
                                if tag == TAG_ON {
                                    self.render_conic(v_control, vec);
                                    continue 'outer;
                                }
                                if tag != TAG_CONIC {
                                    return false;
                                }
                                let v_middle = ((v_control.0 + vec.0) / 2, (v_control.1 + vec.1) / 2);
                                self.render_conic(v_control, v_middle);
                                v_control = vec;
                                continue;
                            }
                            self.render_conic(v_control, v_start);
                            closed_by_curve = true;
                            break 'outer;
                        }
                    }
                    _ => {
                        // FT_CURVE_TAG_CUBIC
                        if point + 1 > limit as isize || tag_of(tags[point as usize + 1]) != TAG_CUBIC {
                            return false;
                        }
                        point += 2;
                        let vec1 = pts[point as usize - 2];
                        let vec2 = pts[point as usize - 1];
                        if point <= limit as isize {
                            let vec = pts[point as usize];
                            self.render_cubic(vec1, vec2, vec);
                            continue;
                        }
                        self.render_cubic(vec1, vec2, v_start);
                        closed_by_curve = true;
                        break 'outer;
                    }
                }
            }
            if !closed_by_curve {
                // close the contour with a line segment
                self.line_to(v_start);
            }
        }
        true
    }
}

fn split_cubic(base: &mut [(i64, i64)]) {
    // x
    base[6].0 = base[3].0;
    let mut a = base[0].0 + base[1].0;
    let b = base[1].0 + base[2].0;
    let mut c = base[2].0 + base[3].0;
    base[5].0 = c >> 1;
    c += b;
    base[4].0 = c >> 2;
    base[1].0 = a >> 1;
    a += b;
    base[2].0 = a >> 2;
    base[3].0 = (a + c) >> 3;
    // y
    base[6].1 = base[3].1;
    let mut a = base[0].1 + base[1].1;
    let b = base[1].1 + base[2].1;
    let mut c = base[2].1 + base[3].1;
    base[5].1 = c >> 1;
    c += b;
    base[4].1 = c >> 2;
    base[1].1 = a >> 1;
    a += b;
    base[2].1 = a >> 2;
    base[3].1 = (a + c) >> 3;
}

#[inline(always)]
fn fill_rule(area: i32, even_odd: bool) -> i32 {
    let fill: i32 = if even_odd { 0x100 } else { i32::MIN };
    let mut coverage = area >> (PIXEL_BITS * 2 + 1 - 8);
    if coverage & fill != 0 {
        coverage = !coverage;
    }
    if coverage > 255 && (fill & i32::MIN) != 0 {
        coverage = 255;
    }
    coverage
}

/// A span produced in "direct" mode.
#[derive(Clone, Copy, Debug)]
pub struct GraySpan {
    pub x: u16,
    pub len: u16,
    pub coverage: u8,
}

/// Render `outline` (26.6 coordinates) into an 8-bit coverage bitmap of
/// `width` x `rows` pixels whose row 0 is the *top* row.  The outline is
/// expected to be already translated so that the bitmap's bottom-left
/// corner is the origin.
pub fn render_to_bitmap(outline: &Outline, width: u32, rows: u32, buffer: &mut [u8]) {
    if outline.points.is_empty() || outline.contours.is_empty() || width == 0 || rows == 0 {
        return;
    }
    let mut w = Worker::new(0, 0, width as i32, rows as i32);
    if !w.decompose(outline) {
        return;
    }
    let even_odd = outline.even_odd;
    let pitch = width as usize;
    for y in w.min_ey..w.max_ey {
        let row = &w.rows[(y - w.min_ey) as usize];
        // line = origin - pitch * y, origin = last row
        let line_off = (rows as usize - 1 - y as usize) * pitch;
        let line = &mut buffer[line_off..line_off + pitch];
        let mut x = w.min_ex;
        let mut cover: i32 = 0;
        for cell in row {
            if cover != 0 && cell.x > x {
                let coverage = fill_rule(cover, even_odd) as u8;
                for v in &mut line[x as usize..cell.x as usize] {
                    *v = coverage;
                }
            }
            cover = cover.wrapping_add(cell.cover.wrapping_mul((ONE_PIXEL * 2) as i32));
            let area = cover.wrapping_sub(cell.area);
            if area != 0 && cell.x >= w.min_ex {
                line[cell.x as usize] = fill_rule(area, even_odd) as u8;
            }
            x = cell.x + 1;
        }
        if cover != 0 {
            // only if cropped
            let coverage = fill_rule(cover, even_odd) as u8;
            for v in &mut line[x as usize..w.max_ex as usize] {
                *v = coverage;
            }
        }
    }
}

/// Direct-mode rendering (`gray_sweep_direct`) with clip box
/// (0, 0, clip_w, clip_h); calls `f(y, spans)` for each scanline.
pub fn render_direct<F: FnMut(i32, &[GraySpan])>(outline: &Outline, clip_w: i32, clip_h: i32, mut f: F) {
    if outline.points.is_empty() || outline.contours.is_empty() || clip_w <= 0 || clip_h <= 0 {
        return;
    }
    let mut w = Worker::new(0, 0, clip_w, clip_h);
    if !w.decompose(outline) {
        return;
    }
    let even_odd = outline.even_odd;
    let mut spans: Vec<GraySpan> = Vec::new();
    for y in w.min_ey..w.max_ey {
        spans.clear();
        let row = &w.rows[(y - w.min_ey) as usize];
        let mut x = w.min_ex;
        let mut cover: i32 = 0;
        for cell in row {
            if cover != 0 && cell.x > x {
                let coverage = fill_rule(cover, even_odd);
                spans.push(GraySpan { coverage: coverage as u8, x: x as u16, len: (cell.x - x) as u16 });
            }
            cover = cover.wrapping_add(cell.cover.wrapping_mul((ONE_PIXEL * 2) as i32));
            let area = cover.wrapping_sub(cell.area);
            if area != 0 && cell.x >= w.min_ex {
                let coverage = fill_rule(area, even_odd);
                spans.push(GraySpan { coverage: coverage as u8, x: cell.x as u16, len: 1 });
            }
            x = cell.x + 1;
        }
        if cover != 0 {
            let coverage = fill_rule(cover, even_odd);
            spans.push(GraySpan { coverage: coverage as u8, x: x as u16, len: (w.max_ex - x) as u16 });
        }
        if !spans.is_empty() {
            f(y, &spans);
        }
    }
}
