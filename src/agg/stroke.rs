//! Port of `agg::vertex_sequence`, `agg::shorten_path`, `agg::math_stroke`,
//! `agg::vcgen_stroke`, `agg::vcgen_dash` and `agg::conv_adaptor_vcgen`.

use super::basics::*;

#[derive(Clone, Copy, Debug, Default)]
pub struct VertexDist {
    pub x: f64,
    pub y: f64,
    pub dist: f64,
}

impl VertexDist {
    #[inline(always)]
    pub fn new(x: f64, y: f64) -> Self {
        VertexDist { x, y, dist: 0.0 }
    }
    /// `vertex_dist::operator()`: compute (and store) the distance to `val`;
    /// return whether the points are not coincident.
    #[inline(always)]
    fn calc(&mut self, val: &VertexDist) -> bool {
        self.dist = calc_distance(self.x, self.y, val.x, val.y);
        let ret = self.dist > VERTEX_DIST_EPSILON;
        if !ret {
            self.dist = 1.0 / VERTEX_DIST_EPSILON;
        }
        ret
    }
}

#[derive(Clone, Default)]
pub struct VertexSequence {
    pub v: Vec<VertexDist>,
}

impl VertexSequence {
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.v.len()
    }
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.v.is_empty()
    }
    #[inline(always)]
    pub fn remove_all(&mut self) {
        self.v.clear();
    }
    #[inline(always)]
    pub fn remove_last(&mut self) {
        self.v.pop();
    }

    #[inline(always)]
    fn check_pair(&mut self, i: usize, j: usize) -> bool {
        let vj = self.v[j];
        self.v[i].calc(&vj)
    }

    pub fn add(&mut self, val: VertexDist) {
        let n = self.v.len();
        if n > 1 && !self.check_pair(n - 2, n - 1) {
            self.v.pop();
        }
        self.v.push(val);
    }

    pub fn modify_last(&mut self, val: VertexDist) {
        self.v.pop();
        self.add(val);
    }

    pub fn close(&mut self, closed: bool) {
        while self.v.len() > 1 {
            let n = self.v.len();
            if self.check_pair(n - 2, n - 1) {
                break;
            }
            let t = self.v[n - 1];
            self.v.pop();
            self.modify_last(t);
        }
        if closed {
            while self.v.len() > 1 {
                let n = self.v.len();
                if self.check_pair(n - 1, 0) {
                    break;
                }
                self.v.pop();
            }
        }
    }

    #[inline(always)]
    pub fn prev(&self, idx: usize) -> &VertexDist {
        let n = self.v.len();
        &self.v[(idx + n - 1) % n]
    }
    #[inline(always)]
    pub fn curr(&self, idx: usize) -> &VertexDist {
        &self.v[idx]
    }
    #[inline(always)]
    pub fn next(&self, idx: usize) -> &VertexDist {
        let n = self.v.len();
        &self.v[(idx + 1) % n]
    }
}

pub fn shorten_path(vs: &mut VertexSequence, s: f64, closed: u32) {
    let mut s = s;
    if s > 0.0 && vs.len() > 1 {
        let mut n = vs.len() as isize - 2;
        while n != 0 {
            let d = vs.v[n as usize].dist;
            if d > s {
                break;
            }
            vs.remove_last();
            s -= d;
            n -= 1;
        }
        if vs.len() < 2 {
            vs.remove_all();
        } else {
            let n = vs.len() - 1;
            let prev = vs.v[n - 1];
            let last = vs.v[n];
            let d = (prev.dist - s) / prev.dist;
            let x = prev.x + (last.x - prev.x) * d;
            let y = prev.y + (last.y - prev.y) * d;
            vs.v[n].x = x;
            vs.v[n].y = y;
            let lastv = vs.v[n];
            if !vs.v[n - 1].calc(&lastv) {
                vs.remove_last();
            }
            vs.close(closed != 0);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LineCap {
    Butt,
    Square,
    Round,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LineJoin {
    Miter,
    MiterRevert,
    Round,
    Bevel,
    MiterRound,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InnerJoin {
    Bevel,
    Miter,
    Jag,
    Round,
}

#[derive(Clone, Copy, Debug)]
pub struct PointD {
    pub x: f64,
    pub y: f64,
}

pub struct MathStroke {
    width: f64,
    width_abs: f64,
    width_eps: f64,
    width_sign: i32,
    miter_limit: f64,
    inner_miter_limit: f64,
    approx_scale: f64,
    line_cap: LineCap,
    line_join: LineJoin,
    inner_join: InnerJoin,
}

impl Default for MathStroke {
    fn default() -> Self {
        MathStroke {
            width: 0.5,
            width_abs: 0.5,
            width_eps: 0.5 / 1024.0,
            width_sign: 1,
            miter_limit: 4.0,
            inner_miter_limit: 1.01,
            approx_scale: 1.0,
            line_cap: LineCap::Butt,
            line_join: LineJoin::Miter,
            inner_join: InnerJoin::Miter,
        }
    }
}

impl MathStroke {
    pub fn set_width(&mut self, w: f64) {
        self.width = w * 0.5;
        if self.width < 0.0 {
            self.width_abs = -self.width;
            self.width_sign = -1;
        } else {
            self.width_abs = self.width;
            self.width_sign = 1;
        }
        self.width_eps = self.width / 1024.0;
    }

    #[inline(always)]
    fn add_vertex(vc: &mut Vec<PointD>, x: f64, y: f64) {
        vc.push(PointD { x, y });
    }

    #[allow(clippy::too_many_arguments)]
    fn calc_arc(&self, vc: &mut Vec<PointD>, x: f64, y: f64, dx1: f64, dy1: f64, dx2: f64, dy2: f64) {
        let ws = self.width_sign as f64;
        let mut a1 = (dy1 * ws).atan2(dx1 * ws);
        let mut a2 = (dy2 * ws).atan2(dx2 * ws);
        let mut da = (self.width_abs / (self.width_abs + 0.125 / self.approx_scale)).acos() * 2.0;

        Self::add_vertex(vc, x + dx1, y + dy1);
        if self.width_sign > 0 {
            if a1 > a2 {
                a2 += 2.0 * PI;
            }
            let n = ((a2 - a1) / da) as i32;
            da = (a2 - a1) / (n + 1) as f64;
            a1 += da;
            for _ in 0..n {
                Self::add_vertex(vc, x + a1.cos() * self.width, y + a1.sin() * self.width);
                a1 += da;
            }
        } else {
            if a1 < a2 {
                a2 -= 2.0 * PI;
            }
            let n = ((a1 - a2) / da) as i32;
            da = (a1 - a2) / (n + 1) as f64;
            a1 -= da;
            for _ in 0..n {
                Self::add_vertex(vc, x + a1.cos() * self.width, y + a1.sin() * self.width);
                a1 -= da;
            }
        }
        Self::add_vertex(vc, x + dx2, y + dy2);
    }

    #[allow(clippy::too_many_arguments)]
    fn calc_miter(
        &self,
        vc: &mut Vec<PointD>,
        v0: &VertexDist,
        v1: &VertexDist,
        v2: &VertexDist,
        dx1: f64,
        dy1: f64,
        dx2: f64,
        dy2: f64,
        lj: LineJoin,
        mlimit: f64,
        dbevel: f64,
    ) {
        let mut xi = v1.x;
        let mut yi = v1.y;
        let mut di = 1.0;
        let lim = self.width_abs * mlimit;
        let mut miter_limit_exceeded = true;
        let mut intersection_failed = true;

        if calc_intersection(
            v0.x + dx1,
            v0.y - dy1,
            v1.x + dx1,
            v1.y - dy1,
            v1.x + dx2,
            v1.y - dy2,
            v2.x + dx2,
            v2.y - dy2,
            &mut xi,
            &mut yi,
        ) {
            di = calc_distance(v1.x, v1.y, xi, yi);
            if di <= lim {
                Self::add_vertex(vc, xi, yi);
                miter_limit_exceeded = false;
            }
            intersection_failed = false;
        } else {
            let x2 = v1.x + dx1;
            let y2 = v1.y - dy1;
            if (cross_product(v0.x, v0.y, v1.x, v1.y, x2, y2) < 0.0)
                == (cross_product(v1.x, v1.y, v2.x, v2.y, x2, y2) < 0.0)
            {
                Self::add_vertex(vc, v1.x + dx1, v1.y - dy1);
                miter_limit_exceeded = false;
            }
        }

        if miter_limit_exceeded {
            match lj {
                LineJoin::MiterRevert => {
                    Self::add_vertex(vc, v1.x + dx1, v1.y - dy1);
                    Self::add_vertex(vc, v1.x + dx2, v1.y - dy2);
                }
                LineJoin::MiterRound => {
                    self.calc_arc(vc, v1.x, v1.y, dx1, -dy1, dx2, -dy2);
                }
                _ => {
                    if intersection_failed {
                        let mlimit = mlimit * self.width_sign as f64;
                        Self::add_vertex(vc, v1.x + dx1 + dy1 * mlimit, v1.y - dy1 + dx1 * mlimit);
                        Self::add_vertex(vc, v1.x + dx2 - dy2 * mlimit, v1.y - dy2 - dx2 * mlimit);
                    } else {
                        let x1 = v1.x + dx1;
                        let y1 = v1.y - dy1;
                        let x2 = v1.x + dx2;
                        let y2 = v1.y - dy2;
                        di = (lim - dbevel) / (di - dbevel);
                        Self::add_vertex(vc, x1 + (xi - x1) * di, y1 + (yi - y1) * di);
                        Self::add_vertex(vc, x2 + (xi - x2) * di, y2 + (yi - y2) * di);
                    }
                }
            }
        }
    }

    pub fn calc_cap(&self, vc: &mut Vec<PointD>, v0: &VertexDist, v1: &VertexDist, len: f64) {
        vc.clear();

        let mut dx1 = (v1.y - v0.y) / len;
        let mut dy1 = (v1.x - v0.x) / len;
        let mut dx2 = 0.0;
        let mut dy2 = 0.0;

        dx1 *= self.width;
        dy1 *= self.width;

        if self.line_cap != LineCap::Round {
            if self.line_cap == LineCap::Square {
                dx2 = dy1 * self.width_sign as f64;
                dy2 = dx1 * self.width_sign as f64;
            }
            Self::add_vertex(vc, v0.x - dx1 - dx2, v0.y + dy1 - dy2);
            Self::add_vertex(vc, v0.x + dx1 - dx2, v0.y - dy1 - dy2);
        } else {
            let mut da = (self.width_abs / (self.width_abs + 0.125 / self.approx_scale)).acos() * 2.0;
            let n = (PI / da) as i32;
            da = PI / (n + 1) as f64;
            Self::add_vertex(vc, v0.x - dx1, v0.y + dy1);
            if self.width_sign > 0 {
                let mut a1 = dy1.atan2(-dx1);
                a1 += da;
                for _ in 0..n {
                    Self::add_vertex(vc, v0.x + a1.cos() * self.width, v0.y + a1.sin() * self.width);
                    a1 += da;
                }
            } else {
                let mut a1 = (-dy1).atan2(dx1);
                a1 -= da;
                for _ in 0..n {
                    Self::add_vertex(vc, v0.x + a1.cos() * self.width, v0.y + a1.sin() * self.width);
                    a1 -= da;
                }
            }
            Self::add_vertex(vc, v0.x + dx1, v0.y - dy1);
        }
    }

    pub fn calc_join(
        &self,
        vc: &mut Vec<PointD>,
        v0: &VertexDist,
        v1: &VertexDist,
        v2: &VertexDist,
        len1: f64,
        len2: f64,
    ) {
        let dx1 = self.width * (v1.y - v0.y) / len1;
        let dy1 = self.width * (v1.x - v0.x) / len1;
        let dx2 = self.width * (v2.y - v1.y) / len2;
        let dy2 = self.width * (v2.x - v1.x) / len2;

        vc.clear();

        let mut cp = cross_product(v0.x, v0.y, v1.x, v1.y, v2.x, v2.y);
        if (cp > VERTEX_DIST_EPSILON && self.width > 0.0) || (cp < -VERTEX_DIST_EPSILON && self.width < 0.0) {
            // Inner join
            let mut limit = (if len1 < len2 { len1 } else { len2 }) / self.width_abs;
            if limit < self.inner_miter_limit {
                limit = self.inner_miter_limit;
            }

            match self.inner_join {
                InnerJoin::Bevel => {
                    Self::add_vertex(vc, v1.x + dx1, v1.y - dy1);
                    Self::add_vertex(vc, v1.x + dx2, v1.y - dy2);
                }
                InnerJoin::Miter => {
                    self.calc_miter(vc, v0, v1, v2, dx1, dy1, dx2, dy2, LineJoin::MiterRevert, limit, 0.0);
                }
                InnerJoin::Jag | InnerJoin::Round => {
                    cp = (dx1 - dx2) * (dx1 - dx2) + (dy1 - dy2) * (dy1 - dy2);
                    if cp < len1 * len1 && cp < len2 * len2 {
                        self.calc_miter(vc, v0, v1, v2, dx1, dy1, dx2, dy2, LineJoin::MiterRevert, limit, 0.0);
                    } else if self.inner_join == InnerJoin::Jag {
                        Self::add_vertex(vc, v1.x + dx1, v1.y - dy1);
                        Self::add_vertex(vc, v1.x, v1.y);
                        Self::add_vertex(vc, v1.x + dx2, v1.y - dy2);
                    } else {
                        Self::add_vertex(vc, v1.x + dx1, v1.y - dy1);
                        Self::add_vertex(vc, v1.x, v1.y);
                        self.calc_arc(vc, v1.x, v1.y, dx2, -dy2, dx1, -dy1);
                        Self::add_vertex(vc, v1.x, v1.y);
                        Self::add_vertex(vc, v1.x + dx2, v1.y - dy2);
                    }
                }
            }
        } else {
            // Outer join
            let mut dx = (dx1 + dx2) / 2.0;
            let mut dy = (dy1 + dy2) / 2.0;
            let dbevel = (dx * dx + dy * dy).sqrt();

            if (self.line_join == LineJoin::Round || self.line_join == LineJoin::Bevel)
                && self.approx_scale * (self.width_abs - dbevel) < self.width_eps
            {
                if calc_intersection(
                    v0.x + dx1,
                    v0.y - dy1,
                    v1.x + dx1,
                    v1.y - dy1,
                    v1.x + dx2,
                    v1.y - dy2,
                    v2.x + dx2,
                    v2.y - dy2,
                    &mut dx,
                    &mut dy,
                ) {
                    Self::add_vertex(vc, dx, dy);
                } else {
                    Self::add_vertex(vc, v1.x + dx1, v1.y - dy1);
                }
                return;
            }

            match self.line_join {
                LineJoin::Miter | LineJoin::MiterRevert | LineJoin::MiterRound => {
                    self.calc_miter(vc, v0, v1, v2, dx1, dy1, dx2, dy2, self.line_join, self.miter_limit, dbevel);
                }
                LineJoin::Round => {
                    self.calc_arc(vc, v1.x, v1.y, dx1, -dy1, dx2, -dy2);
                }
                LineJoin::Bevel => {
                    Self::add_vertex(vc, v1.x + dx1, v1.y - dy1);
                    Self::add_vertex(vc, v1.x + dx2, v1.y - dy2);
                }
            }
        }
    }
}

/// A vertex generator used by [`ConvAdaptorVcgen`].
pub trait Generator {
    fn remove_all(&mut self);
    fn add_vertex(&mut self, x: f64, y: f64, cmd: u32);
    fn rewind(&mut self, path_id: u32);
    fn vertex(&mut self, x: &mut f64, y: &mut f64) -> u32;
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum StrokeStatus {
    Initial,
    Ready,
    Cap1,
    Cap2,
    Outline1,
    CloseFirst,
    Outline2,
    OutVertices,
    EndPoly1,
    EndPoly2,
    Stop,
}

pub struct VcgenStroke {
    pub stroker: MathStroke,
    src_vertices: VertexSequence,
    out_vertices: Vec<PointD>,
    shorten: f64,
    closed: u32,
    status: StrokeStatus,
    prev_status: StrokeStatus,
    src_vertex: usize,
    out_vertex: usize,
}

impl Default for VcgenStroke {
    fn default() -> Self {
        VcgenStroke {
            stroker: MathStroke::default(),
            src_vertices: VertexSequence::default(),
            out_vertices: Vec::new(),
            shorten: 0.0,
            closed: 0,
            status: StrokeStatus::Initial,
            prev_status: StrokeStatus::Initial,
            src_vertex: 0,
            out_vertex: 0,
        }
    }
}

impl VcgenStroke {
    pub fn width(&mut self, w: f64) {
        self.stroker.set_width(w);
    }
    pub fn line_cap(&mut self, lc: LineCap) {
        self.stroker.line_cap = lc;
    }
    pub fn line_join(&mut self, lj: LineJoin) {
        self.stroker.line_join = lj;
    }
    pub fn inner_join(&mut self, ij: InnerJoin) {
        self.stroker.inner_join = ij;
    }
    pub fn miter_limit(&mut self, ml: f64) {
        self.stroker.miter_limit = ml;
    }
    pub fn approximation_scale(&mut self, s: f64) {
        self.stroker.approx_scale = s;
    }
}

impl Generator for VcgenStroke {
    fn remove_all(&mut self) {
        self.src_vertices.remove_all();
        self.closed = 0;
        self.status = StrokeStatus::Initial;
    }

    fn add_vertex(&mut self, x: f64, y: f64, cmd: u32) {
        self.status = StrokeStatus::Initial;
        if is_move_to(cmd) {
            self.src_vertices.modify_last(VertexDist::new(x, y));
        } else if is_vertex(cmd) {
            self.src_vertices.add(VertexDist::new(x, y));
        } else {
            self.closed = get_close_flag(cmd);
        }
    }

    fn rewind(&mut self, _path_id: u32) {
        if self.status == StrokeStatus::Initial {
            self.src_vertices.close(self.closed != 0);
            shorten_path(&mut self.src_vertices, self.shorten, self.closed);
            if self.src_vertices.len() < 3 {
                self.closed = 0;
            }
        }
        self.status = StrokeStatus::Ready;
        self.src_vertex = 0;
        self.out_vertex = 0;
    }

    fn vertex(&mut self, x: &mut f64, y: &mut f64) -> u32 {
        let mut cmd = PATH_CMD_LINE_TO;
        while !is_stop(cmd) {
            match self.status {
                StrokeStatus::Initial | StrokeStatus::Ready => {
                    if self.status == StrokeStatus::Initial {
                        self.rewind(0);
                    }
                    if self.src_vertices.len() < 2 + (self.closed != 0) as usize {
                        cmd = PATH_CMD_STOP;
                        continue;
                    }
                    self.status = if self.closed != 0 { StrokeStatus::Outline1 } else { StrokeStatus::Cap1 };
                    cmd = PATH_CMD_MOVE_TO;
                    self.src_vertex = 0;
                    self.out_vertex = 0;
                }
                StrokeStatus::Cap1 => {
                    let v0 = self.src_vertices.v[0];
                    let v1 = self.src_vertices.v[1];
                    self.stroker.calc_cap(&mut self.out_vertices, &v0, &v1, v0.dist);
                    self.src_vertex = 1;
                    self.prev_status = StrokeStatus::Outline1;
                    self.status = StrokeStatus::OutVertices;
                    self.out_vertex = 0;
                }
                StrokeStatus::Cap2 => {
                    let n = self.src_vertices.len();
                    let v0 = self.src_vertices.v[n - 1];
                    let v1 = self.src_vertices.v[n - 2];
                    self.stroker.calc_cap(&mut self.out_vertices, &v0, &v1, v1.dist);
                    self.prev_status = StrokeStatus::Outline2;
                    self.status = StrokeStatus::OutVertices;
                    self.out_vertex = 0;
                }
                StrokeStatus::Outline1 => {
                    if self.closed != 0 {
                        if self.src_vertex >= self.src_vertices.len() {
                            self.prev_status = StrokeStatus::CloseFirst;
                            self.status = StrokeStatus::EndPoly1;
                            continue;
                        }
                    } else if self.src_vertex >= self.src_vertices.len() - 1 {
                        self.status = StrokeStatus::Cap2;
                        continue;
                    }
                    let i = self.src_vertex;
                    let v0 = *self.src_vertices.prev(i);
                    let v1 = *self.src_vertices.curr(i);
                    let v2 = *self.src_vertices.next(i);
                    self.stroker.calc_join(&mut self.out_vertices, &v0, &v1, &v2, v0.dist, v1.dist);
                    self.src_vertex += 1;
                    self.prev_status = self.status;
                    self.status = StrokeStatus::OutVertices;
                    self.out_vertex = 0;
                }
                StrokeStatus::CloseFirst | StrokeStatus::Outline2 => {
                    if self.status == StrokeStatus::CloseFirst {
                        self.status = StrokeStatus::Outline2;
                        cmd = PATH_CMD_MOVE_TO;
                    }
                    if self.src_vertex <= (self.closed == 0) as usize {
                        self.status = StrokeStatus::EndPoly2;
                        self.prev_status = StrokeStatus::Stop;
                        continue;
                    }
                    self.src_vertex -= 1;
                    let i = self.src_vertex;
                    let v0 = *self.src_vertices.next(i);
                    let v1 = *self.src_vertices.curr(i);
                    let v2 = *self.src_vertices.prev(i);
                    self.stroker.calc_join(&mut self.out_vertices, &v0, &v1, &v2, v1.dist, v2.dist);
                    self.prev_status = self.status;
                    self.status = StrokeStatus::OutVertices;
                    self.out_vertex = 0;
                }
                StrokeStatus::OutVertices => {
                    if self.out_vertex >= self.out_vertices.len() {
                        self.status = self.prev_status;
                    } else {
                        let c = self.out_vertices[self.out_vertex];
                        self.out_vertex += 1;
                        *x = c.x;
                        *y = c.y;
                        return cmd;
                    }
                }
                StrokeStatus::EndPoly1 => {
                    self.status = self.prev_status;
                    return PATH_CMD_END_POLY | PATH_FLAGS_CLOSE | PATH_FLAGS_CCW;
                }
                StrokeStatus::EndPoly2 => {
                    self.status = self.prev_status;
                    return PATH_CMD_END_POLY | PATH_FLAGS_CLOSE | PATH_FLAGS_CW;
                }
                StrokeStatus::Stop => {
                    cmd = PATH_CMD_STOP;
                }
            }
        }
        cmd
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum DashStatus {
    Initial,
    Ready,
    Polyline,
    Stop,
}

const MAX_DASHES: usize = 32;

pub struct VcgenDash {
    dashes: [f64; MAX_DASHES],
    total_dash_len: f64,
    num_dashes: usize,
    dash_start: f64,
    shorten: f64,
    curr_dash_start: f64,
    curr_dash: usize,
    curr_rest: f64,
    v1: usize,
    v2: usize,
    src_vertices: VertexSequence,
    closed: u32,
    status: DashStatus,
    src_vertex: usize,
}

impl Default for VcgenDash {
    fn default() -> Self {
        VcgenDash {
            dashes: [0.0; MAX_DASHES],
            total_dash_len: 0.0,
            num_dashes: 0,
            dash_start: 0.0,
            shorten: 0.0,
            curr_dash_start: 0.0,
            curr_dash: 0,
            curr_rest: 0.0,
            v1: 0,
            v2: 0,
            src_vertices: VertexSequence::default(),
            closed: 0,
            status: DashStatus::Initial,
            src_vertex: 0,
        }
    }
}

impl VcgenDash {
    pub fn remove_all_dashes(&mut self) {
        self.total_dash_len = 0.0;
        self.num_dashes = 0;
        self.curr_dash_start = 0.0;
        self.curr_dash = 0;
    }

    pub fn add_dash(&mut self, dash_len: f64, gap_len: f64) {
        if self.num_dashes < MAX_DASHES {
            self.total_dash_len += dash_len + gap_len;
            self.dashes[self.num_dashes] = dash_len;
            self.num_dashes += 1;
            self.dashes[self.num_dashes] = gap_len;
            self.num_dashes += 1;
        }
    }

    pub fn dash_start(&mut self, ds: f64) {
        self.dash_start = ds;
        self.calc_dash_start(ds.abs());
    }

    fn calc_dash_start(&mut self, ds: f64) {
        let mut ds = ds;
        self.curr_dash = 0;
        self.curr_dash_start = 0.0;
        while ds > 0.0 {
            if ds > self.dashes[self.curr_dash] {
                ds -= self.dashes[self.curr_dash];
                self.curr_dash += 1;
                self.curr_dash_start = 0.0;
                if self.curr_dash >= self.num_dashes {
                    self.curr_dash = 0;
                }
            } else {
                self.curr_dash_start = ds;
                ds = 0.0;
            }
        }
    }
}

impl Generator for VcgenDash {
    fn remove_all(&mut self) {
        self.status = DashStatus::Initial;
        self.src_vertices.remove_all();
        self.closed = 0;
    }

    fn add_vertex(&mut self, x: f64, y: f64, cmd: u32) {
        self.status = DashStatus::Initial;
        if is_move_to(cmd) {
            self.src_vertices.modify_last(VertexDist::new(x, y));
        } else if is_vertex(cmd) {
            self.src_vertices.add(VertexDist::new(x, y));
        } else {
            self.closed = get_close_flag(cmd);
        }
    }

    fn rewind(&mut self, _path_id: u32) {
        if self.status == DashStatus::Initial {
            self.src_vertices.close(self.closed != 0);
            shorten_path(&mut self.src_vertices, self.shorten, self.closed);
        }
        self.status = DashStatus::Ready;
        self.src_vertex = 0;
    }

    fn vertex(&mut self, x: &mut f64, y: &mut f64) -> u32 {
        let mut cmd = PATH_CMD_MOVE_TO;
        while !is_stop(cmd) {
            match self.status {
                DashStatus::Initial | DashStatus::Ready => {
                    if self.status == DashStatus::Initial {
                        self.rewind(0);
                    }
                    if self.num_dashes < 2 || self.src_vertices.len() < 2 {
                        cmd = PATH_CMD_STOP;
                        continue;
                    }
                    self.status = DashStatus::Polyline;
                    self.src_vertex = 1;
                    self.v1 = 0;
                    self.v2 = 1;
                    self.curr_rest = self.src_vertices.v[0].dist;
                    *x = self.src_vertices.v[0].x;
                    *y = self.src_vertices.v[0].y;
                    if self.dash_start >= 0.0 {
                        self.calc_dash_start(self.dash_start);
                    }
                    return PATH_CMD_MOVE_TO;
                }
                DashStatus::Polyline => {
                    let dash_rest = self.dashes[self.curr_dash] - self.curr_dash_start;
                    let cmd2 = if self.curr_dash & 1 != 0 { PATH_CMD_MOVE_TO } else { PATH_CMD_LINE_TO };
                    let v1 = self.src_vertices.v[self.v1];
                    let v2 = self.src_vertices.v[self.v2];
                    if self.curr_rest > dash_rest {
                        self.curr_rest -= dash_rest;
                        self.curr_dash += 1;
                        if self.curr_dash >= self.num_dashes {
                            self.curr_dash = 0;
                        }
                        self.curr_dash_start = 0.0;
                        *x = v2.x - (v2.x - v1.x) * self.curr_rest / v1.dist;
                        *y = v2.y - (v2.y - v1.y) * self.curr_rest / v1.dist;
                    } else {
                        self.curr_dash_start += self.curr_rest;
                        *x = v2.x;
                        *y = v2.y;
                        self.src_vertex += 1;
                        self.v1 = self.v2;
                        self.curr_rest = self.src_vertices.v[self.v1].dist;
                        if self.closed != 0 {
                            if self.src_vertex > self.src_vertices.len() {
                                self.status = DashStatus::Stop;
                            } else {
                                self.v2 = if self.src_vertex >= self.src_vertices.len() { 0 } else { self.src_vertex };
                            }
                        } else if self.src_vertex >= self.src_vertices.len() {
                            self.status = DashStatus::Stop;
                        } else {
                            self.v2 = self.src_vertex;
                        }
                    }
                    return cmd2;
                }
                DashStatus::Stop => {
                    cmd = PATH_CMD_STOP;
                }
            }
        }
        PATH_CMD_STOP
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum AdaptorStatus {
    Initial,
    Accumulate,
    Generate,
}

/// `agg::conv_adaptor_vcgen<VertexSource, Generator>` (with null markers).
pub struct ConvAdaptorVcgen<S: VertexSource, G: Generator> {
    pub source: S,
    pub generator: G,
    status: AdaptorStatus,
    last_cmd: u32,
    start_x: f64,
    start_y: f64,
}

impl<S: VertexSource, G: Generator> ConvAdaptorVcgen<S, G> {
    pub fn new(source: S, generator: G) -> Self {
        ConvAdaptorVcgen { source, generator, status: AdaptorStatus::Initial, last_cmd: 0, start_x: 0.0, start_y: 0.0 }
    }
}

impl<S: VertexSource, G: Generator> VertexSource for ConvAdaptorVcgen<S, G> {
    fn rewind(&mut self, path_id: u32) {
        self.source.rewind(path_id);
        self.status = AdaptorStatus::Initial;
    }

    fn vertex(&mut self, x: &mut f64, y: &mut f64) -> u32 {
        let mut cmd;
        loop {
            match self.status {
                AdaptorStatus::Initial | AdaptorStatus::Accumulate => {
                    if self.status == AdaptorStatus::Initial {
                        self.last_cmd = self.source.vertex(&mut self.start_x, &mut self.start_y);
                        self.status = AdaptorStatus::Accumulate;
                    }
                    if is_stop(self.last_cmd) {
                        return PATH_CMD_STOP;
                    }
                    self.generator.remove_all();
                    self.generator.add_vertex(self.start_x, self.start_y, PATH_CMD_MOVE_TO);

                    loop {
                        cmd = self.source.vertex(x, y);
                        if is_vertex(cmd) {
                            self.last_cmd = cmd;
                            if is_move_to(cmd) {
                                self.start_x = *x;
                                self.start_y = *y;
                                break;
                            }
                            self.generator.add_vertex(*x, *y, cmd);
                        } else {
                            if is_stop(cmd) {
                                self.last_cmd = PATH_CMD_STOP;
                                break;
                            }
                            if is_end_poly(cmd) {
                                self.generator.add_vertex(*x, *y, cmd);
                                break;
                            }
                        }
                    }
                    self.generator.rewind(0);
                    self.status = AdaptorStatus::Generate;
                }
                AdaptorStatus::Generate => {
                    cmd = self.generator.vertex(x, y);
                    if is_stop(cmd) {
                        self.status = AdaptorStatus::Accumulate;
                        continue;
                    }
                    return cmd;
                }
            }
        }
    }
}

pub type ConvStroke<S> = ConvAdaptorVcgen<S, VcgenStroke>;
pub type ConvDash<S> = ConvAdaptorVcgen<S, VcgenDash>;

pub fn conv_stroke<S: VertexSource>(source: S) -> ConvStroke<S> {
    ConvAdaptorVcgen::new(source, VcgenStroke::default())
}

pub fn conv_dash<S: VertexSource>(source: S) -> ConvDash<S> {
    ConvAdaptorVcgen::new(source, VcgenDash::default())
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ContourStatus {
    Initial,
    Ready,
    Outline,
    OutVertices,
    EndPoly,
    Stop,
}

/// `agg::vcgen_contour` (without orientation auto-detection).
pub struct VcgenContour {
    stroker: MathStroke,
    width: f64,
    src_vertices: VertexSequence,
    out_vertices: Vec<PointD>,
    status: ContourStatus,
    src_vertex: usize,
    out_vertex: usize,
    closed: u32,
    orientation: u32,
}

impl Default for VcgenContour {
    fn default() -> Self {
        VcgenContour {
            stroker: MathStroke::default(),
            width: 1.0,
            src_vertices: VertexSequence::default(),
            out_vertices: Vec::new(),
            status: ContourStatus::Initial,
            src_vertex: 0,
            out_vertex: 0,
            closed: 0,
            orientation: 0,
        }
    }
}

impl VcgenContour {
    pub fn width(&mut self, w: f64) {
        self.width = w;
        self.stroker.set_width(w);
    }
}

impl Generator for VcgenContour {
    fn remove_all(&mut self) {
        self.src_vertices.remove_all();
        self.closed = 0;
        self.orientation = 0;
        self.status = ContourStatus::Initial;
    }

    fn add_vertex(&mut self, x: f64, y: f64, cmd: u32) {
        self.status = ContourStatus::Initial;
        if is_move_to(cmd) {
            self.src_vertices.modify_last(VertexDist::new(x, y));
        } else if is_vertex(cmd) {
            self.src_vertices.add(VertexDist::new(x, y));
        } else if is_end_poly(cmd) {
            self.closed = get_close_flag(cmd);
            if self.orientation == PATH_FLAGS_NONE {
                self.orientation = cmd & (PATH_FLAGS_CW | PATH_FLAGS_CCW);
            }
        }
    }

    fn rewind(&mut self, _path_id: u32) {
        if self.status == ContourStatus::Initial {
            self.src_vertices.close(true);
            if self.orientation & (PATH_FLAGS_CW | PATH_FLAGS_CCW) != 0 {
                let w = if self.orientation & PATH_FLAGS_CCW != 0 { self.width } else { -self.width };
                self.stroker.set_width(w);
            }
        }
        self.status = ContourStatus::Ready;
        self.src_vertex = 0;
    }

    fn vertex(&mut self, x: &mut f64, y: &mut f64) -> u32 {
        let mut cmd = PATH_CMD_LINE_TO;
        while !is_stop(cmd) {
            match self.status {
                ContourStatus::Initial | ContourStatus::Ready => {
                    if self.status == ContourStatus::Initial {
                        self.rewind(0);
                    }
                    let min = 2 + usize::from(self.closed != 0);
                    if self.src_vertices.len() < min {
                        cmd = PATH_CMD_STOP;
                        continue;
                    }
                    self.status = ContourStatus::Outline;
                    cmd = PATH_CMD_MOVE_TO;
                    self.src_vertex = 0;
                    self.out_vertex = 0;
                }
                ContourStatus::Outline => {
                    if self.src_vertex >= self.src_vertices.len() {
                        self.status = ContourStatus::EndPoly;
                        continue;
                    }
                    let i = self.src_vertex;
                    let (prev, curr, next) =
                        (*self.src_vertices.prev(i), *self.src_vertices.curr(i), *self.src_vertices.next(i));
                    self.stroker.calc_join(&mut self.out_vertices, &prev, &curr, &next, prev.dist, curr.dist);
                    self.src_vertex += 1;
                    self.status = ContourStatus::OutVertices;
                    self.out_vertex = 0;
                }
                ContourStatus::OutVertices => {
                    if self.out_vertex >= self.out_vertices.len() {
                        self.status = ContourStatus::Outline;
                    } else {
                        let c = self.out_vertices[self.out_vertex];
                        self.out_vertex += 1;
                        *x = c.x;
                        *y = c.y;
                        return cmd;
                    }
                }
                ContourStatus::EndPoly => {
                    if self.closed == 0 {
                        return PATH_CMD_STOP;
                    }
                    self.status = ContourStatus::Stop;
                    return PATH_CMD_END_POLY | PATH_FLAGS_CLOSE | PATH_FLAGS_CCW;
                }
                ContourStatus::Stop => return PATH_CMD_STOP,
            }
        }
        cmd
    }
}

pub type ConvContour<S> = ConvAdaptorVcgen<S, VcgenContour>;
