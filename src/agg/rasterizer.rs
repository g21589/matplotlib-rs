//! Port of `agg::rasterizer_cells_aa`, `agg::rasterizer_sl_clip<ras_conv_dbl>`
//! and `agg::rasterizer_scanline_aa`.

use super::basics::*;
use super::scanline::Scanline;

#[derive(Clone, Copy, Debug)]
pub struct Cell {
    pub x: i32,
    pub y: i32,
    pub cover: i32,
    pub area: i32,
}

impl Cell {
    const INITIAL: Cell = Cell { x: 0x7FFFFFFF, y: 0x7FFFFFFF, cover: 0, area: 0 };
}

pub struct CellsAa {
    cells: Vec<Cell>,
    sorted_cells: Vec<Cell>,
    /// For each scanline (y - min_y): (start, num) into `sorted_cells`.
    sorted_y: Vec<(u32, u32)>,
    curr: Cell,
    min_x: i32,
    min_y: i32,
    max_x: i32,
    max_y: i32,
    sorted: bool,
}

impl Default for CellsAa {
    fn default() -> Self {
        Self::new()
    }
}

impl CellsAa {
    pub fn new() -> Self {
        CellsAa {
            cells: Vec::new(),
            sorted_cells: Vec::new(),
            sorted_y: Vec::new(),
            curr: Cell::INITIAL,
            min_x: 0x7FFFFFFF,
            min_y: 0x7FFFFFFF,
            max_x: -0x7FFFFFFF,
            max_y: -0x7FFFFFFF,
            sorted: false,
        }
    }

    pub fn reset(&mut self) {
        self.cells.clear();
        self.curr = Cell::INITIAL;
        self.sorted = false;
        self.min_x = 0x7FFFFFFF;
        self.min_y = 0x7FFFFFFF;
        self.max_x = -0x7FFFFFFF;
        self.max_y = -0x7FFFFFFF;
    }

    #[inline(always)]
    pub fn min_x(&self) -> i32 {
        self.min_x
    }
    #[inline(always)]
    pub fn min_y(&self) -> i32 {
        self.min_y
    }
    #[inline(always)]
    pub fn max_x(&self) -> i32 {
        self.max_x
    }
    #[inline(always)]
    pub fn max_y(&self) -> i32 {
        self.max_y
    }
    #[inline(always)]
    pub fn sorted(&self) -> bool {
        self.sorted
    }
    #[inline(always)]
    pub fn total_cells(&self) -> usize {
        self.sorted_cells.len()
    }

    #[inline(always)]
    fn add_curr_cell(&mut self) {
        if (self.curr.area | self.curr.cover) != 0 {
            self.cells.push(self.curr);
        }
    }

    #[inline(always)]
    fn set_curr_cell(&mut self, x: i32, y: i32) {
        if self.curr.x != x || self.curr.y != y {
            self.add_curr_cell();
            self.curr.x = x;
            self.curr.y = y;
            self.curr.cover = 0;
            self.curr.area = 0;
        }
    }

    #[inline]
    fn render_hline(&mut self, ey: i32, x1: i32, y1: i32, x2: i32, y2: i32) {
        let mut ex1 = x1 >> POLY_SUBPIXEL_SHIFT;
        let ex2 = x2 >> POLY_SUBPIXEL_SHIFT;
        let fx1 = x1 & POLY_SUBPIXEL_MASK;
        let fx2 = x2 & POLY_SUBPIXEL_MASK;
        let mut y1 = y1;

        if y1 == y2 {
            self.set_curr_cell(ex2, ey);
            return;
        }

        if ex1 == ex2 {
            let delta = y2 - y1;
            self.curr.cover = self.curr.cover.wrapping_add(delta);
            self.curr.area = self.curr.area.wrapping_add((fx1 + fx2).wrapping_mul(delta));
            return;
        }

        let mut p = (POLY_SUBPIXEL_SCALE - fx1) * (y2 - y1);
        let mut first = POLY_SUBPIXEL_SCALE;
        let mut incr = 1;
        let mut dx = x2 - x1;

        if dx < 0 {
            p = fx1 * (y2 - y1);
            first = 0;
            incr = -1;
            dx = -dx;
        }

        let mut delta = p / dx;
        let mut modv = p % dx;
        if modv < 0 {
            delta -= 1;
            modv += dx;
        }

        self.curr.cover = self.curr.cover.wrapping_add(delta);
        self.curr.area = self.curr.area.wrapping_add((fx1 + first).wrapping_mul(delta));

        ex1 += incr;
        self.set_curr_cell(ex1, ey);
        y1 += delta;

        if ex1 != ex2 {
            p = POLY_SUBPIXEL_SCALE * (y2 - y1 + delta);
            let mut lift = p / dx;
            let mut rem = p % dx;
            if rem < 0 {
                lift -= 1;
                rem += dx;
            }
            modv -= dx;

            while ex1 != ex2 {
                delta = lift;
                modv += rem;
                if modv >= 0 {
                    modv -= dx;
                    delta += 1;
                }
                self.curr.cover = self.curr.cover.wrapping_add(delta);
                self.curr.area = self.curr.area.wrapping_add(POLY_SUBPIXEL_SCALE.wrapping_mul(delta));
                y1 += delta;
                ex1 += incr;
                self.set_curr_cell(ex1, ey);
            }
        }
        delta = y2 - y1;
        self.curr.cover = self.curr.cover.wrapping_add(delta);
        self.curr.area = self
            .curr
            .area
            .wrapping_add((fx2 + POLY_SUBPIXEL_SCALE - first).wrapping_mul(delta));
    }

    pub fn line(&mut self, x1: i32, y1: i32, x2: i32, y2: i32) {
        const DX_LIMIT: i32 = 16384 << POLY_SUBPIXEL_SHIFT;
        let dx = x2.wrapping_sub(x1);
        if dx >= DX_LIMIT || dx <= -DX_LIMIT {
            let cx = (x1 >> 1) + (x2 >> 1) + ((x1 & 1) & (x2 & 1));
            let cy = (y1 >> 1) + (y2 >> 1) + ((y1 & 1) & (y2 & 1));
            self.line(x1, y1, cx, cy);
            self.line(cx, cy, x2, y2);
            return;
        }

        let mut dy = y2.wrapping_sub(y1);
        let ex1 = x1 >> POLY_SUBPIXEL_SHIFT;
        let ex2 = x2 >> POLY_SUBPIXEL_SHIFT;
        let mut ey1 = y1 >> POLY_SUBPIXEL_SHIFT;
        let ey2 = y2 >> POLY_SUBPIXEL_SHIFT;
        let fy1 = y1 & POLY_SUBPIXEL_MASK;
        let fy2 = y2 & POLY_SUBPIXEL_MASK;

        if ex1 < self.min_x {
            self.min_x = ex1;
        }
        if ex1 > self.max_x {
            self.max_x = ex1;
        }
        if ey1 < self.min_y {
            self.min_y = ey1;
        }
        if ey1 > self.max_y {
            self.max_y = ey1;
        }
        if ex2 < self.min_x {
            self.min_x = ex2;
        }
        if ex2 > self.max_x {
            self.max_x = ex2;
        }
        if ey2 < self.min_y {
            self.min_y = ey2;
        }
        if ey2 > self.max_y {
            self.max_y = ey2;
        }

        self.set_curr_cell(ex1, ey1);

        // everything is on a single hline
        if ey1 == ey2 {
            self.render_hline(ey1, x1, fy1, x2, fy2);
            return;
        }

        let mut incr = 1;
        if dx == 0 {
            let ex = x1 >> POLY_SUBPIXEL_SHIFT;
            let two_fx = (x1 - (ex << POLY_SUBPIXEL_SHIFT)) << 1;
            let mut first = POLY_SUBPIXEL_SCALE;
            if dy < 0 {
                first = 0;
                incr = -1;
            }
            let mut delta = first - fy1;
            self.curr.cover = self.curr.cover.wrapping_add(delta);
            self.curr.area = self.curr.area.wrapping_add(two_fx.wrapping_mul(delta));

            ey1 += incr;
            self.set_curr_cell(ex, ey1);

            delta = first + first - POLY_SUBPIXEL_SCALE;
            let area = two_fx.wrapping_mul(delta);
            while ey1 != ey2 {
                self.curr.cover = delta;
                self.curr.area = area;
                ey1 += incr;
                self.set_curr_cell(ex, ey1);
            }
            delta = fy2 - POLY_SUBPIXEL_SCALE + first;
            self.curr.cover = self.curr.cover.wrapping_add(delta);
            self.curr.area = self.curr.area.wrapping_add(two_fx.wrapping_mul(delta));
            return;
        }

        // ok, we have to render several hlines
        let mut p = (POLY_SUBPIXEL_SCALE - fy1).wrapping_mul(dx);
        let mut first = POLY_SUBPIXEL_SCALE;

        if dy < 0 {
            p = fy1.wrapping_mul(dx);
            first = 0;
            incr = -1;
            dy = -dy;
        }

        let mut delta = p / dy;
        let mut modv = p % dy;

        if modv < 0 {
            delta -= 1;
            modv += dy;
        }

        let mut x_from = x1 + delta;
        self.render_hline(ey1, x1, fy1, x_from, first);

        ey1 += incr;
        self.set_curr_cell(x_from >> POLY_SUBPIXEL_SHIFT, ey1);

        if ey1 != ey2 {
            p = POLY_SUBPIXEL_SCALE.wrapping_mul(dx);
            let mut lift = p / dy;
            let mut rem = p % dy;

            if rem < 0 {
                lift -= 1;
                rem += dy;
            }
            modv -= dy;

            while ey1 != ey2 {
                delta = lift;
                modv += rem;
                if modv >= 0 {
                    modv -= dy;
                    delta += 1;
                }

                let x_to = x_from + delta;
                self.render_hline(ey1, x_from, POLY_SUBPIXEL_SCALE - first, x_to, first);
                x_from = x_to;

                ey1 += incr;
                self.set_curr_cell(x_from >> POLY_SUBPIXEL_SHIFT, ey1);
            }
        }
        self.render_hline(ey1, x_from, POLY_SUBPIXEL_SCALE - first, x2, fy2);
    }

    pub fn sort_cells(&mut self) {
        if self.sorted {
            return; // Perform sort only the first time.
        }

        self.add_curr_cell();
        self.curr.x = 0x7FFFFFFF;
        self.curr.y = 0x7FFFFFFF;
        self.curr.cover = 0;
        self.curr.area = 0;

        self.sorted_cells.clear();
        self.sorted_y.clear();
        if self.cells.is_empty() {
            return;
        }

        // Counting sort by y, then sort each row by x.  The relative order of
        // cells sharing the same x is irrelevant because the sweep sums them.
        let rows = (self.max_y - self.min_y + 1) as usize;
        self.sorted_y.resize(rows, (0, 0));
        for c in &self.cells {
            self.sorted_y[(c.y - self.min_y) as usize].0 += 1;
        }
        let mut start = 0u32;
        for sy in self.sorted_y.iter_mut() {
            let v = sy.0;
            sy.0 = start;
            start += v;
        }
        self.sorted_cells.resize(self.cells.len(), Cell::INITIAL);
        for c in &self.cells {
            let sy = &mut self.sorted_y[(c.y - self.min_y) as usize];
            self.sorted_cells[(sy.0 + sy.1) as usize] = *c;
            sy.1 += 1;
        }
        for &(s, n) in &self.sorted_y {
            if n > 1 {
                self.sorted_cells[s as usize..(s + n) as usize].sort_unstable_by_key(|c| c.x);
            }
        }
        self.sorted = true;
    }

    #[inline(always)]
    pub fn scanline_cells(&self, y: i32) -> &[Cell] {
        let (s, n) = self.sorted_y[(y - self.min_y) as usize];
        &self.sorted_cells[s as usize..(s + n) as usize]
    }
}

/// `clipping_flags` from `agg_clip_liang_barsky.h`.
#[inline(always)]
pub fn clipping_flags(x: f64, y: f64, cb: &RectD) -> u32 {
    (x > cb.x2) as u32 | (((y > cb.y2) as u32) << 1) | (((x < cb.x1) as u32) << 2) | (((y < cb.y1) as u32) << 3)
}

#[inline(always)]
pub fn clipping_flags_y(y: f64, cb: &RectD) -> u32 {
    (((y > cb.y2) as u32) << 1) | (((y < cb.y1) as u32) << 3)
}

/// `rasterizer_sl_clip<ras_conv_dbl>`.
#[derive(Default)]
struct SlClipDbl {
    clip_box: RectD,
    x1: f64,
    y1: f64,
    f1: u32,
    clipping: bool,
}

#[inline(always)]
fn xi(v: f64) -> i32 {
    iround(v * POLY_SUBPIXEL_SCALE as f64)
}

#[inline(always)]
fn mul_div(a: f64, b: f64, c: f64) -> f64 {
    a * b / c
}

impl SlClipDbl {
    fn reset_clipping(&mut self) {
        self.clipping = false;
    }

    fn clip_box(&mut self, x1: f64, y1: f64, x2: f64, y2: f64) {
        self.clip_box = RectD::new(x1, y1, x2, y2);
        self.clip_box.normalize();
        self.clipping = true;
    }

    fn move_to(&mut self, x1: f64, y1: f64) {
        self.x1 = x1;
        self.y1 = y1;
        if self.clipping {
            self.f1 = clipping_flags(x1, y1, &self.clip_box);
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn line_clip_y(&self, ras: &mut CellsAa, x1: f64, y1: f64, x2: f64, y2: f64, f1: u32, f2: u32) {
        let f1 = f1 & 10;
        let f2 = f2 & 10;
        if (f1 | f2) == 0 {
            // Fully visible
            ras.line(xi(x1), xi(y1), xi(x2), xi(y2));
        } else {
            if f1 == f2 {
                // Invisible by Y
                return;
            }
            let cb = &self.clip_box;
            let mut tx1 = x1;
            let mut ty1 = y1;
            let mut tx2 = x2;
            let mut ty2 = y2;

            if f1 & 8 != 0 {
                tx1 = x1 + mul_div(cb.y1 - y1, x2 - x1, y2 - y1);
                ty1 = cb.y1;
            }
            if f1 & 2 != 0 {
                tx1 = x1 + mul_div(cb.y2 - y1, x2 - x1, y2 - y1);
                ty1 = cb.y2;
            }
            if f2 & 8 != 0 {
                tx2 = x1 + mul_div(cb.y1 - y1, x2 - x1, y2 - y1);
                ty2 = cb.y1;
            }
            if f2 & 2 != 0 {
                tx2 = x1 + mul_div(cb.y2 - y1, x2 - x1, y2 - y1);
                ty2 = cb.y2;
            }
            ras.line(xi(tx1), xi(ty1), xi(tx2), xi(ty2));
        }
    }

    fn line_to(&mut self, ras: &mut CellsAa, x2: f64, y2: f64) {
        if self.clipping {
            let f2 = clipping_flags(x2, y2, &self.clip_box);

            if (self.f1 & 10) == (f2 & 10) && (self.f1 & 10) != 0 {
                // Invisible by Y
                self.x1 = x2;
                self.y1 = y2;
                self.f1 = f2;
                return;
            }

            let x1 = self.x1;
            let y1 = self.y1;
            let f1 = self.f1;
            let cb = self.clip_box;

            match ((f1 & 5) << 1) | (f2 & 5) {
                0 => {
                    self.line_clip_y(ras, x1, y1, x2, y2, f1, f2);
                }
                1 => {
                    let y3 = y1 + mul_div(cb.x2 - x1, y2 - y1, x2 - x1);
                    let f3 = clipping_flags_y(y3, &cb);
                    self.line_clip_y(ras, x1, y1, cb.x2, y3, f1, f3);
                    self.line_clip_y(ras, cb.x2, y3, cb.x2, y2, f3, f2);
                }
                2 => {
                    let y3 = y1 + mul_div(cb.x2 - x1, y2 - y1, x2 - x1);
                    let f3 = clipping_flags_y(y3, &cb);
                    self.line_clip_y(ras, cb.x2, y1, cb.x2, y3, f1, f3);
                    self.line_clip_y(ras, cb.x2, y3, x2, y2, f3, f2);
                }
                3 => {
                    self.line_clip_y(ras, cb.x2, y1, cb.x2, y2, f1, f2);
                }
                4 => {
                    let y3 = y1 + mul_div(cb.x1 - x1, y2 - y1, x2 - x1);
                    let f3 = clipping_flags_y(y3, &cb);
                    self.line_clip_y(ras, x1, y1, cb.x1, y3, f1, f3);
                    self.line_clip_y(ras, cb.x1, y3, cb.x1, y2, f3, f2);
                }
                6 => {
                    let y3 = y1 + mul_div(cb.x2 - x1, y2 - y1, x2 - x1);
                    let y4 = y1 + mul_div(cb.x1 - x1, y2 - y1, x2 - x1);
                    let f3 = clipping_flags_y(y3, &cb);
                    let f4 = clipping_flags_y(y4, &cb);
                    self.line_clip_y(ras, cb.x2, y1, cb.x2, y3, f1, f3);
                    self.line_clip_y(ras, cb.x2, y3, cb.x1, y4, f3, f4);
                    self.line_clip_y(ras, cb.x1, y4, cb.x1, y2, f4, f2);
                }
                8 => {
                    let y3 = y1 + mul_div(cb.x1 - x1, y2 - y1, x2 - x1);
                    let f3 = clipping_flags_y(y3, &cb);
                    self.line_clip_y(ras, cb.x1, y1, cb.x1, y3, f1, f3);
                    self.line_clip_y(ras, cb.x1, y3, x2, y2, f3, f2);
                }
                9 => {
                    let y3 = y1 + mul_div(cb.x1 - x1, y2 - y1, x2 - x1);
                    let y4 = y1 + mul_div(cb.x2 - x1, y2 - y1, x2 - x1);
                    let f3 = clipping_flags_y(y3, &cb);
                    let f4 = clipping_flags_y(y4, &cb);
                    self.line_clip_y(ras, cb.x1, y1, cb.x1, y3, f1, f3);
                    self.line_clip_y(ras, cb.x1, y3, cb.x2, y4, f3, f4);
                    self.line_clip_y(ras, cb.x2, y4, cb.x2, y2, f4, f2);
                }
                12 => {
                    self.line_clip_y(ras, cb.x1, y1, cb.x1, y2, f1, f2);
                }
                _ => {}
            }
            self.f1 = f2;
        } else {
            ras.line(xi(self.x1), xi(self.y1), xi(x2), xi(y2));
        }
        self.x1 = x2;
        self.y1 = y2;
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FillingRule {
    NonZero,
    EvenOdd,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Status {
    Initial,
    MoveTo,
    LineTo,
    Closed,
}

pub const AA_SHIFT: i32 = 8;
pub const AA_SCALE: i32 = 1 << AA_SHIFT;
pub const AA_MASK: i32 = AA_SCALE - 1;
pub const AA_SCALE2: i32 = AA_SCALE * 2;
pub const AA_MASK2: i32 = AA_SCALE2 - 1;

/// `agg::rasterizer_scanline_aa<agg::rasterizer_sl_clip_dbl>`.
pub struct RasterizerScanlineAa {
    outline: CellsAa,
    clipper: SlClipDbl,
    gamma: [i32; AA_SCALE as usize],
    filling_rule: FillingRule,
    auto_close: bool,
    start_x: f64,
    start_y: f64,
    status: Status,
    scan_y: i32,
}

impl Default for RasterizerScanlineAa {
    fn default() -> Self {
        Self::new()
    }
}

impl RasterizerScanlineAa {
    pub fn new() -> Self {
        let mut gamma = [0i32; AA_SCALE as usize];
        for (i, g) in gamma.iter_mut().enumerate() {
            *g = i as i32;
        }
        RasterizerScanlineAa {
            outline: CellsAa::new(),
            clipper: SlClipDbl::default(),
            gamma,
            filling_rule: FillingRule::NonZero,
            auto_close: true,
            start_x: 0.0,
            start_y: 0.0,
            status: Status::Initial,
            scan_y: 0,
        }
    }

    pub fn reset(&mut self) {
        self.outline.reset();
        self.status = Status::Initial;
    }

    pub fn reset_clipping(&mut self) {
        self.reset();
        self.clipper.reset_clipping();
    }

    pub fn clip_box(&mut self, x1: f64, y1: f64, x2: f64, y2: f64) {
        self.reset();
        self.clipper.clip_box(x1, y1, x2, y2);
    }

    pub fn filling_rule(&mut self, r: FillingRule) {
        self.filling_rule = r;
    }

    pub fn gamma_fn<F: Fn(f64) -> f64>(&mut self, f: F) {
        for i in 0..AA_SCALE as usize {
            self.gamma[i] = uround(f(i as f64 / AA_MASK as f64) * AA_MASK as f64) as i32;
        }
    }

    pub fn close_polygon(&mut self) {
        if self.status == Status::LineTo {
            self.clipper.line_to(&mut self.outline, self.start_x, self.start_y);
            self.status = Status::Closed;
        }
    }

    pub fn move_to_d(&mut self, x: f64, y: f64) {
        if self.outline.sorted() {
            self.reset();
        }
        if self.auto_close {
            self.close_polygon();
        }
        self.start_x = x;
        self.start_y = y;
        self.clipper.move_to(x, y);
        self.status = Status::MoveTo;
    }

    pub fn line_to_d(&mut self, x: f64, y: f64) {
        self.clipper.line_to(&mut self.outline, x, y);
        self.status = Status::LineTo;
    }

    #[inline]
    pub fn add_vertex(&mut self, x: f64, y: f64, cmd: u32) {
        if is_move_to(cmd) {
            self.move_to_d(x, y);
        } else if is_vertex(cmd) {
            self.line_to_d(x, y);
        } else if is_close(cmd) {
            self.close_polygon();
        }
    }

    pub fn add_path<V: VertexSource>(&mut self, vs: &mut V) {
        self.add_path_id(vs, 0)
    }

    pub fn add_path_id<V: VertexSource>(&mut self, vs: &mut V, path_id: u32) {
        let mut x = 0.0;
        let mut y = 0.0;
        vs.rewind(path_id);
        if self.outline.sorted() {
            self.reset();
        }
        loop {
            let cmd = vs.vertex(&mut x, &mut y);
            if is_stop(cmd) {
                break;
            }
            self.add_vertex(x, y, cmd);
        }
    }

    pub fn min_x(&self) -> i32 {
        self.outline.min_x()
    }
    pub fn min_y(&self) -> i32 {
        self.outline.min_y()
    }
    pub fn max_x(&self) -> i32 {
        self.outline.max_x()
    }
    pub fn max_y(&self) -> i32 {
        self.outline.max_y()
    }

    pub fn sort(&mut self) {
        if self.auto_close {
            self.close_polygon();
        }
        self.outline.sort_cells();
    }

    pub fn rewind_scanlines(&mut self) -> bool {
        if self.auto_close {
            self.close_polygon();
        }
        self.outline.sort_cells();
        if self.outline.total_cells() == 0 {
            return false;
        }
        self.scan_y = self.outline.min_y();
        true
    }

    #[inline(always)]
    pub fn calculate_alpha(&self, area: i32) -> u32 {
        let mut cover = area >> (POLY_SUBPIXEL_SHIFT * 2 + 1 - AA_SHIFT);
        if cover < 0 {
            cover = -cover;
        }
        if self.filling_rule == FillingRule::EvenOdd {
            cover &= AA_MASK2;
            if cover > AA_SCALE {
                cover = AA_SCALE2 - cover;
            }
        }
        if cover > AA_MASK {
            cover = AA_MASK;
        }
        self.gamma[cover as usize] as u32
    }

    pub fn sweep_scanline<S: Scanline>(&mut self, sl: &mut S) -> bool {
        loop {
            if self.scan_y > self.outline.max_y() {
                return false;
            }
            sl.reset_spans();
            let cells = self.outline.scanline_cells(self.scan_y);
            let num = cells.len();
            let mut i = 0usize;
            let mut cover: i32 = 0;

            while i < num {
                let cur = cells[i];
                let mut x = cur.x;
                let mut area = cur.area;
                cover = cover.wrapping_add(cur.cover);
                i += 1;
                // accumulate all cells with the same X
                while i < num {
                    let c = cells[i];
                    if c.x != x {
                        break;
                    }
                    area = area.wrapping_add(c.area);
                    cover = cover.wrapping_add(c.cover);
                    i += 1;
                }

                if area != 0 {
                    let alpha = self.calculate_alpha((cover << (POLY_SUBPIXEL_SHIFT + 1)).wrapping_sub(area));
                    if alpha != 0 {
                        sl.add_cell(x, alpha);
                    }
                    x += 1;
                }

                if i < num && cells[i].x > x {
                    let alpha = self.calculate_alpha(cover << (POLY_SUBPIXEL_SHIFT + 1));
                    if alpha != 0 {
                        sl.add_span(x, (cells[i].x - x) as u32, alpha);
                    }
                }
            }

            if sl.num_spans() != 0 {
                break;
            }
            self.scan_y += 1;
        }
        sl.finalize(self.scan_y);
        self.scan_y += 1;
        true
    }
}
