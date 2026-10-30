//! Port of ContourPy's `mpl2014` contour generator (contourpy 1.4,
//! `src/mpl2014.cpp`, BSD 3-clause license; see licenses/LICENSE_CONTOURPY),
//! matplotlib's default `contour.algorithm`, used when contourpy is not
//! installed.  The traversal order, the cache flags and the interpolation
//! arithmetic follow the C++ code, so that the vertices are identical.

pub const MOVETO: u8 = 1;
pub const LINETO: u8 = 2;
pub const CLOSEPOLY: u8 = 79;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Edge {
    None,
    E,
    N,
    W,
    S,
    NE,
    NW,
    SW,
    SE,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct QuadEdge {
    quad: isize,
    edge: Edge,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Dir {
    Right,
    Straight,
    Left,
}

const MASK_Z_LEVEL: u32 = 0x0003;
const MASK_Z_LEVEL_1: u32 = 0x0001;
const MASK_Z_LEVEL_2: u32 = 0x0002;
const MASK_VISITED_1: u32 = 0x0004;
const MASK_VISITED_2: u32 = 0x0008;
const MASK_SADDLE_1: u32 = 0x0010;
const MASK_SADDLE_2: u32 = 0x0020;
const MASK_SADDLE_LEFT_1: u32 = 0x0040;
const MASK_SADDLE_LEFT_2: u32 = 0x0080;
const MASK_SADDLE_START_SW_1: u32 = 0x0100;
const MASK_SADDLE_START_SW_2: u32 = 0x0200;
const MASK_BOUNDARY_S: u32 = 0x0400;
const MASK_BOUNDARY_W: u32 = 0x0800;
const MASK_EXISTS_QUAD: u32 = 0x1000;
const MASK_EXISTS_SW_CORNER: u32 = 0x2000;
const MASK_EXISTS_SE_CORNER: u32 = 0x3000;
const MASK_EXISTS_NW_CORNER: u32 = 0x4000;
const MASK_EXISTS_NE_CORNER: u32 = 0x5000;
const MASK_EXISTS: u32 = 0x7000;
const MASK_VISITED_S: u32 = 0x10000;
const MASK_VISITED_W: u32 = 0x20000;
const MASK_VISITED_CORNER: u32 = 0x40000;

/// A line of a contour; holes refer to their parent, non-holes list their
/// holes (indices into `Lines`).
struct Line {
    points: Vec<(f64, f64)>,
    is_hole: bool,
    parent: Option<usize>,
    children: Vec<usize>,
}

/// The output of one call: a polygon or line per element, as (x, y) pairs
/// and matplotlib path codes.
#[derive(Default, Debug, PartialEq)]
pub struct Output {
    pub vertices: Vec<Vec<(f64, f64)>>,
    pub codes: Vec<Vec<u8>>,
}

struct ParentCache {
    nx: isize,
    x_chunk_points: isize,
    y_chunk_points: isize,
    lines: Vec<Option<usize>>,
    istart: isize,
    jstart: isize,
}

impl ParentCache {
    fn index(&self, quad: isize) -> usize {
        let i = quad % self.nx;
        let j = quad / self.nx;
        ((i - self.istart) + (j - self.jstart) * self.x_chunk_points) as usize
    }
    fn get_parent(&self, quad: isize) -> usize {
        let mut index = self.index(quad) as isize;
        loop {
            if let Some(Some(p)) = self.lines.get(index as usize) {
                return *p;
            }
            index -= self.x_chunk_points;
            assert!(index >= 0, "Failed to find parent in chunk ParentCache");
        }
    }
    fn set_chunk_starts(&mut self, istart: isize, jstart: isize) {
        self.istart = istart;
        self.jstart = jstart;
        let n = (self.x_chunk_points * self.y_chunk_points) as usize;
        self.lines.clear();
        self.lines.resize(n, None);
    }
}

pub struct Mpl2014 {
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    nx: isize,
    ny: isize,
    n: isize,
    corner_mask: bool,
    x_chunk_size: isize,
    y_chunk_size: isize,
    nxchunk: isize,
    nychunk: isize,
    chunk_count: isize,
    cache: Vec<u32>,
    parent_cache: ParentCache,
    lines: Vec<Line>,
}

fn calc_chunk_count(point_count: isize, chunk_size: isize) -> isize {
    if chunk_size > 0 && point_count > 1 {
        let mut count = (point_count - 1) / chunk_size;
        if count * chunk_size < point_count - 1 {
            count += 1;
        }
        count
    } else {
        1
    }
}

fn calc_chunk_size(point_count: isize, chunk_size: isize) -> isize {
    let mut ret = point_count - 1;
    if chunk_size > 0 {
        ret = chunk_size.min(ret);
    }
    ret.max(1)
}

impl Mpl2014 {
    /// `x`, `y`, `z` (and the optional `mask`) have `ny * nx` points in row
    /// major order; nx, ny >= 2 and the chunk sizes >= 0 (checked by caller).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        x: Vec<f64>,
        y: Vec<f64>,
        z: Vec<f64>,
        nx: usize,
        ny: usize,
        mask: Option<&[bool]>,
        corner_mask: bool,
        x_chunk_size: usize,
        y_chunk_size: usize,
    ) -> Self {
        let (nx, ny) = (nx as isize, ny as isize);
        let x_chunk_size = calc_chunk_size(nx, x_chunk_size as isize);
        let y_chunk_size = calc_chunk_size(ny, y_chunk_size as isize);
        let nxchunk = calc_chunk_count(nx, x_chunk_size);
        let nychunk = calc_chunk_count(ny, y_chunk_size);
        let mut g = Mpl2014 {
            x,
            y,
            z,
            nx,
            ny,
            n: nx * ny,
            corner_mask,
            x_chunk_size,
            y_chunk_size,
            nxchunk,
            nychunk,
            chunk_count: nxchunk * nychunk,
            cache: vec![0; (nx * ny) as usize],
            parent_cache: ParentCache {
                nx,
                x_chunk_points: if x_chunk_size > 0 { x_chunk_size + 1 } else { nx },
                y_chunk_points: if y_chunk_size > 0 { y_chunk_size + 1 } else { ny },
                lines: Vec::new(),
                istart: 0,
                jstart: 0,
            },
            lines: Vec::new(),
        };
        g.init_cache_grid(mask);
        g
    }

    pub fn chunk_size(&self) -> (usize, usize) {
        (self.y_chunk_size as usize, self.x_chunk_size as usize)
    }

    pub fn chunk_count(&self) -> (usize, usize) {
        (self.nychunk as usize, self.nxchunk as usize)
    }

    // ---- cache accessors (out-of-range quads have no flags) ----
    #[inline]
    fn c(&self, quad: isize) -> u32 {
        if quad >= 0 && quad < self.n {
            self.cache[quad as usize]
        } else {
            0
        }
    }
    #[inline]
    fn set(&mut self, quad: isize, mask: u32) {
        self.cache[quad as usize] |= mask;
    }
    #[inline]
    fn z_level(&self, point: isize) -> u32 {
        self.c(point) & MASK_Z_LEVEL
    }
    #[inline]
    fn z_ne(&self, quad: isize) -> u32 {
        self.z_level(quad + self.nx + 1)
    }
    #[inline]
    fn z_nw(&self, quad: isize) -> u32 {
        self.z_level(quad + self.nx)
    }
    #[inline]
    fn z_se(&self, quad: isize) -> u32 {
        self.z_level(quad + 1)
    }
    #[inline]
    fn z_sw(&self, quad: isize) -> u32 {
        self.z_level(quad)
    }
    #[inline]
    fn visited(&self, quad: isize, li: u32) -> bool {
        self.c(quad) & (if li == 1 { MASK_VISITED_1 } else { MASK_VISITED_2 }) != 0
    }
    #[inline]
    fn visited_s(&self, quad: isize) -> bool {
        self.c(quad) & MASK_VISITED_S != 0
    }
    #[inline]
    fn visited_w(&self, quad: isize) -> bool {
        self.c(quad) & MASK_VISITED_W != 0
    }
    #[inline]
    fn visited_corner(&self, quad: isize) -> bool {
        self.c(quad) & MASK_VISITED_CORNER != 0
    }
    #[inline]
    fn saddle(&self, quad: isize, li: u32) -> bool {
        self.c(quad) & (if li == 1 { MASK_SADDLE_1 } else { MASK_SADDLE_2 }) != 0
    }
    #[inline]
    fn saddle_left(&self, quad: isize, li: u32) -> bool {
        self.c(quad) & (if li == 1 { MASK_SADDLE_LEFT_1 } else { MASK_SADDLE_LEFT_2 }) != 0
    }
    #[inline]
    fn saddle_start_sw(&self, quad: isize, li: u32) -> bool {
        self.c(quad) & (if li == 1 { MASK_SADDLE_START_SW_1 } else { MASK_SADDLE_START_SW_2 }) != 0
    }
    #[inline]
    fn boundary_s(&self, quad: isize) -> bool {
        self.c(quad) & MASK_BOUNDARY_S != 0
    }
    #[inline]
    fn boundary_w(&self, quad: isize) -> bool {
        self.c(quad) & MASK_BOUNDARY_W != 0
    }
    #[inline]
    fn boundary_n(&self, quad: isize) -> bool {
        self.boundary_s(quad + self.nx)
    }
    #[inline]
    fn boundary_e(&self, quad: isize) -> bool {
        self.boundary_w(quad + 1)
    }
    #[inline]
    fn exists(&self, quad: isize) -> u32 {
        self.c(quad) & MASK_EXISTS
    }
    #[inline]
    fn exists_quad(&self, quad: isize) -> bool {
        self.exists(quad) == MASK_EXISTS_QUAD
    }
    #[inline]
    fn exists_none(&self, quad: isize) -> bool {
        self.exists(quad) == 0
    }
    #[inline]
    fn exists_sw_corner(&self, quad: isize) -> bool {
        self.exists(quad) == MASK_EXISTS_SW_CORNER
    }
    #[inline]
    fn exists_se_corner(&self, quad: isize) -> bool {
        self.exists(quad) == MASK_EXISTS_SE_CORNER
    }
    #[inline]
    fn exists_nw_corner(&self, quad: isize) -> bool {
        self.exists(quad) == MASK_EXISTS_NW_CORNER
    }
    #[inline]
    fn exists_ne_corner(&self, quad: isize) -> bool {
        self.exists(quad) == MASK_EXISTS_NE_CORNER
    }
    #[inline]
    fn exists_any_corner(&self, quad: isize) -> bool {
        !self.exists_none(quad) && !self.exists_quad(quad)
    }
    fn exists_w_edge(&self, quad: isize) -> bool {
        self.exists_quad(quad) || self.exists_sw_corner(quad) || self.exists_nw_corner(quad)
    }
    fn exists_e_edge(&self, quad: isize) -> bool {
        self.exists_quad(quad) || self.exists_se_corner(quad) || self.exists_ne_corner(quad)
    }
    fn exists_s_edge(&self, quad: isize) -> bool {
        self.exists_quad(quad) || self.exists_sw_corner(quad) || self.exists_se_corner(quad)
    }
    fn exists_n_edge(&self, quad: isize) -> bool {
        self.exists_quad(quad) || self.exists_nw_corner(quad) || self.exists_ne_corner(quad)
    }

    fn init_cache_grid(&mut self, mask: Option<&[bool]>) {
        let (nx, ny) = (self.nx, self.ny);
        match mask {
            None => {
                let mut quad = 0usize;
                for j in 0..ny {
                    for i in 0..nx {
                        let mut v = 0;
                        if i < nx - 1 && j < ny - 1 {
                            v |= MASK_EXISTS_QUAD;
                        }
                        if (i % self.x_chunk_size == 0 || i == nx - 1) && j < ny - 1 {
                            v |= MASK_BOUNDARY_W;
                        }
                        if (j % self.y_chunk_size == 0 || j == ny - 1) && i < nx - 1 {
                            v |= MASK_BOUNDARY_S;
                        }
                        self.cache[quad] = v;
                        quad += 1;
                    }
                }
            }
            Some(mask) => {
                let mut quad = 0isize;
                for j in 0..ny {
                    for i in 0..nx {
                        self.cache[quad as usize] = 0;
                        if i < nx - 1 && j < ny - 1 {
                            let m = |p: isize| mask[p as usize] as u32;
                            let config = (m(quad + nx) << 3) | (m(quad + nx + 1) << 2) | (m(quad) << 1) | m(quad + 1);
                            if self.corner_mask {
                                self.cache[quad as usize] = match config {
                                    0 => MASK_EXISTS_QUAD,
                                    1 => MASK_EXISTS_NW_CORNER,
                                    2 => MASK_EXISTS_NE_CORNER,
                                    4 => MASK_EXISTS_SW_CORNER,
                                    8 => MASK_EXISTS_SE_CORNER,
                                    _ => 0,
                                };
                            } else if config == 0 {
                                self.cache[quad as usize] = MASK_EXISTS_QUAD;
                            }
                        }
                        quad += 1;
                    }
                }
                let mut quad = 0isize;
                for j in 0..ny {
                    for i in 0..nx {
                        if self.corner_mask {
                            let w_exists_none = i == 0 || self.exists_none(quad - 1);
                            let s_exists_none = j == 0 || self.exists_none(quad - nx);
                            let w_exists_e_edge = i > 0 && self.exists_e_edge(quad - 1);
                            let s_exists_n_edge = j > 0 && self.exists_n_edge(quad - nx);
                            if (self.exists_w_edge(quad) && w_exists_none)
                                || (self.exists_none(quad) && w_exists_e_edge)
                                || (i % self.x_chunk_size == 0 && self.exists_w_edge(quad) && w_exists_e_edge)
                            {
                                self.set(quad, MASK_BOUNDARY_W);
                            }
                            if (self.exists_s_edge(quad) && s_exists_none)
                                || (self.exists_none(quad) && s_exists_n_edge)
                                || (j % self.y_chunk_size == 0 && self.exists_s_edge(quad) && s_exists_n_edge)
                            {
                                self.set(quad, MASK_BOUNDARY_S);
                            }
                        } else {
                            let w_exists_quad = i > 0 && self.exists_quad(quad - 1);
                            let s_exists_quad = j > 0 && self.exists_quad(quad - nx);
                            if (self.exists_quad(quad) != w_exists_quad)
                                || (i % self.x_chunk_size == 0 && self.exists_quad(quad) && w_exists_quad)
                            {
                                self.set(quad, MASK_BOUNDARY_W);
                            }
                            if (self.exists_quad(quad) != s_exists_quad)
                                || (j % self.y_chunk_size == 0 && self.exists_quad(quad) && s_exists_quad)
                            {
                                self.set(quad, MASK_BOUNDARY_S);
                            }
                        }
                        quad += 1;
                    }
                }
            }
        }
    }

    fn init_cache_levels(&mut self, lower_level: f64, upper_level: f64) {
        let two_levels = lower_level != upper_level;
        let keep_mask = if self.corner_mask {
            MASK_EXISTS | MASK_BOUNDARY_S | MASK_BOUNDARY_W
        } else {
            MASK_EXISTS_QUAD | MASK_BOUNDARY_S | MASK_BOUNDARY_W
        };
        for (c, &z) in self.cache.iter_mut().zip(&self.z) {
            *c &= keep_mask;
            if two_levels {
                if z > upper_level {
                    *c |= MASK_Z_LEVEL_2;
                } else if z > lower_level {
                    *c |= MASK_Z_LEVEL_1;
                }
            } else if z > lower_level {
                *c |= MASK_Z_LEVEL_1;
            }
        }
    }

    fn chunk_limits(&self, ijchunk: isize) -> (isize, isize, isize, isize, isize, isize) {
        let ichunk = ijchunk % self.nxchunk;
        let jchunk = ijchunk / self.nxchunk;
        let istart = ichunk * self.x_chunk_size;
        let iend = if ichunk == self.nxchunk - 1 { self.nx } else { (ichunk + 1) * self.x_chunk_size };
        let jstart = jchunk * self.y_chunk_size;
        let jend = if jchunk == self.nychunk - 1 { self.ny } else { (jchunk + 1) * self.y_chunk_size };
        (ichunk, jchunk, istart, iend, jstart, jend)
    }

    fn edge_point_index(&self, qe: QuadEdge, start: bool) -> isize {
        let quad = qe.quad;
        let (sw, se, nw, ne) = (quad, quad + 1, quad + self.nx, quad + self.nx + 1);
        match qe.edge {
            Edge::E => if start { se } else { ne },
            Edge::N => if start { ne } else { nw },
            Edge::W => if start { nw } else { sw },
            Edge::S => if start { sw } else { se },
            Edge::NE => if start { se } else { nw },
            Edge::NW => if start { ne } else { sw },
            Edge::SW => if start { nw } else { se },
            Edge::SE => if start { sw } else { ne },
            Edge::None => 0,
        }
    }

    fn interp(&self, point1: isize, point2: isize, level: f64, line: &mut Vec<(f64, f64)>) {
        let (p1, p2) = (point1 as usize, point2 as usize);
        let fraction = (self.z[p2] - level) / (self.z[p2] - self.z[p1]);
        line.push((
            self.x[p1] * fraction + self.x[p2] * (1.0 - fraction),
            self.y[p1] * fraction + self.y[p2] * (1.0 - fraction),
        ));
    }

    fn edge_interp(&self, qe: QuadEdge, level: f64, line: &mut Vec<(f64, f64)>) {
        self.interp(self.edge_point_index(qe, true), self.edge_point_index(qe, false), level, line);
    }

    fn is_edge_a_boundary(&self, qe: QuadEdge) -> bool {
        match qe.edge {
            Edge::E => self.boundary_e(qe.quad),
            Edge::N => self.boundary_n(qe.quad),
            Edge::W => self.boundary_w(qe.quad),
            Edge::S => self.boundary_s(qe.quad),
            Edge::NE => self.exists_sw_corner(qe.quad),
            Edge::NW => self.exists_se_corner(qe.quad),
            Edge::SW => self.exists_ne_corner(qe.quad),
            Edge::SE => self.exists_nw_corner(qe.quad),
            Edge::None => true,
        }
    }

    fn exit_edge(&self, qe: QuadEdge, dir: Dir) -> Edge {
        let quad = qe.quad;
        if self.exists_any_corner(quad) {
            match qe.edge {
                Edge::E => {
                    if self.exists_se_corner(quad) {
                        if dir == Dir::Left { Edge::S } else { Edge::NW }
                    } else if dir == Dir::Right {
                        Edge::N
                    } else {
                        Edge::SW
                    }
                }
                Edge::N => {
                    if self.exists_nw_corner(quad) {
                        if dir == Dir::Right { Edge::W } else { Edge::SE }
                    } else if dir == Dir::Left {
                        Edge::E
                    } else {
                        Edge::SW
                    }
                }
                Edge::W => {
                    if self.exists_sw_corner(quad) {
                        if dir == Dir::Right { Edge::S } else { Edge::NE }
                    } else if dir == Dir::Left {
                        Edge::N
                    } else {
                        Edge::SE
                    }
                }
                Edge::S => {
                    if self.exists_sw_corner(quad) {
                        if dir == Dir::Left { Edge::W } else { Edge::NE }
                    } else if dir == Dir::Right {
                        Edge::E
                    } else {
                        Edge::NW
                    }
                }
                Edge::NE => if dir == Dir::Left { Edge::S } else { Edge::W },
                Edge::NW => if dir == Dir::Left { Edge::E } else { Edge::S },
                Edge::SW => if dir == Dir::Left { Edge::N } else { Edge::E },
                Edge::SE => if dir == Dir::Left { Edge::W } else { Edge::N },
                Edge::None => Edge::None,
            }
        } else {
            let pick = |left, straight, right| match dir {
                Dir::Left => left,
                Dir::Right => right,
                Dir::Straight => straight,
            };
            match qe.edge {
                Edge::E => pick(Edge::S, Edge::W, Edge::N),
                Edge::N => pick(Edge::E, Edge::S, Edge::W),
                Edge::W => pick(Edge::N, Edge::E, Edge::S),
                Edge::S => pick(Edge::W, Edge::N, Edge::E),
                _ => Edge::None,
            }
        }
    }

    fn corner_start_edge(&self, quad: isize, level_index: u32) -> Edge {
        let (sw, se, nw, ne) = (quad, quad + 1, quad + self.nx, quad + self.nx + 1);
        let (p1, p2, p3, e12, e23, e31) = match self.exists(quad) {
            MASK_EXISTS_SW_CORNER => (se, sw, nw, Edge::S, Edge::W, Edge::NE),
            MASK_EXISTS_SE_CORNER => (ne, se, sw, Edge::E, Edge::S, Edge::NW),
            MASK_EXISTS_NW_CORNER => (sw, nw, ne, Edge::W, Edge::N, Edge::SE),
            MASK_EXISTS_NE_CORNER => (nw, ne, se, Edge::N, Edge::E, Edge::SW),
            _ => return Edge::None,
        };
        let ge = |p| (self.z_level(p) >= level_index) as u32;
        let mut config = (ge(p1) << 2) | (ge(p2) << 1) | ge(p3);
        if level_index == 2 {
            config = 7 - config;
        }
        match config {
            1 => e23,
            2 | 3 => e12,
            4 => e31,
            5 => e23,
            6 => e31,
            _ => Edge::None,
        }
    }

    fn quad_start_edge(&self, quad: isize, level_index: u32) -> Edge {
        let ge = |z: u32| (z >= level_index) as u32;
        let mut config = (ge(self.z_nw(quad)) << 3)
            | (ge(self.z_ne(quad)) << 2)
            | (ge(self.z_sw(quad)) << 1)
            | ge(self.z_se(quad));
        if level_index == 2 {
            config = 15 - config;
        }
        match config {
            1 => Edge::E,
            2 => Edge::S,
            3 => Edge::E,
            4 => Edge::N,
            5 => Edge::N,
            6 => {
                if !self.saddle(quad, level_index) || self.saddle_start_sw(quad, level_index) {
                    Edge::S
                } else {
                    Edge::N
                }
            }
            7 => Edge::N,
            8 => Edge::W,
            9 => {
                if !self.saddle(quad, level_index) || self.saddle_start_sw(quad, level_index) {
                    Edge::W
                } else {
                    Edge::E
                }
            }
            10 => Edge::S,
            11 => Edge::E,
            12 => Edge::W,
            13 => Edge::W,
            14 => Edge::S,
            _ => Edge::None,
        }
    }

    fn start_edge(&self, quad: isize, level_index: u32) -> Edge {
        if self.exists_any_corner(quad) {
            self.corner_start_edge(quad, level_index)
        } else {
            self.quad_start_edge(quad, level_index)
        }
    }

    fn move_to_next_boundary_edge(&self, qe: &mut QuadEdge) {
        let nx = self.nx;
        qe.quad = self.edge_point_index(*qe, false);
        let mut index: i32 = match qe.edge {
            Edge::E => 0,
            Edge::SE => 1,
            Edge::S => 2,
            Edge::SW => 3,
            Edge::W => 4,
            Edge::NW => 5,
            Edge::N => 6,
            Edge::NE => 7,
            Edge::None => 0,
        };
        if !self.corner_mask {
            index += 1;
        }
        let start_index = index;
        let quad = qe.quad;
        loop {
            match index {
                0 => {
                    if self.exists_se_corner(quad - nx - 1) {
                        *qe = QuadEdge { quad: quad - nx - 1, edge: Edge::NW };
                        return;
                    }
                }
                1 => {
                    if self.boundary_n(quad - nx - 1) {
                        *qe = QuadEdge { quad: quad - nx - 1, edge: Edge::N };
                        return;
                    }
                }
                2 => {
                    if self.exists_sw_corner(quad - 1) {
                        *qe = QuadEdge { quad: quad - 1, edge: Edge::NE };
                        return;
                    }
                }
                3 => {
                    if self.boundary_e(quad - 1) {
                        *qe = QuadEdge { quad: quad - 1, edge: Edge::E };
                        return;
                    }
                }
                4 => {
                    if self.exists_nw_corner(quad) {
                        qe.edge = Edge::SE;
                        return;
                    }
                }
                5 => {
                    if self.boundary_s(quad) {
                        qe.edge = Edge::S;
                        return;
                    }
                }
                6 => {
                    if self.exists_ne_corner(quad - nx) {
                        *qe = QuadEdge { quad: quad - nx, edge: Edge::SW };
                        return;
                    }
                }
                7 => {
                    if self.boundary_w(quad - nx) {
                        *qe = QuadEdge { quad: quad - nx, edge: Edge::W };
                        return;
                    }
                }
                _ => {}
            }
            index = if self.corner_mask { (index + 1) % 8 } else { (index + 2) % 8 };
            if index == start_index {
                return;
            }
        }
    }

    fn move_to_next_quad(&self, qe: &mut QuadEdge) {
        match qe.edge {
            Edge::E => *qe = QuadEdge { quad: qe.quad + 1, edge: Edge::W },
            Edge::N => *qe = QuadEdge { quad: qe.quad + self.nx, edge: Edge::S },
            Edge::W => *qe = QuadEdge { quad: qe.quad - 1, edge: Edge::E },
            Edge::S => *qe = QuadEdge { quad: qe.quad - self.nx, edge: Edge::N },
            _ => {}
        }
    }

    fn set_parent(&mut self, quad: isize, line: usize) {
        let index = self.parent_cache.index(quad);
        if self.parent_cache.lines[index].is_none() {
            let l = &self.lines[line];
            self.parent_cache.lines[index] = if l.is_hole { l.parent } else { Some(line) };
        }
    }

    /// Appends to `points` and, when `line` is given, records parents (only
    /// for filled contours).
    #[allow(clippy::too_many_arguments)]
    fn follow_interior(
        &mut self,
        points: &mut Vec<(f64, f64)>,
        line: Option<usize>,
        qe: &mut QuadEdge,
        level_index: u32,
        level: f64,
        want_initial_point: bool,
        start_quad_edge: Option<QuadEdge>,
        start_level_index: u32,
    ) {
        let nx = self.nx;
        if want_initial_point {
            self.edge_interp(*qe, level, points);
        }
        let visited_mask = if level_index == 1 { MASK_VISITED_1 } else { MASK_VISITED_2 };
        let saddle_mask = if level_index == 1 { MASK_SADDLE_1 } else { MASK_SADDLE_2 };
        loop {
            let quad = qe.quad;
            let (sw, se, nw, ne) = (quad, quad + 1, quad + nx, quad + nx + 1);
            let dir;
            if self.c(quad) & saddle_mask != 0 {
                dir = if self.saddle_left(quad, level_index) { Dir::Left } else { Dir::Right };
                self.set(quad, visited_mask);
            } else if self.exists_any_corner(quad) {
                let point_opposite = match qe.edge {
                    Edge::E => if self.exists_se_corner(quad) { sw } else { nw },
                    Edge::N => if self.exists_nw_corner(quad) { sw } else { se },
                    Edge::W => if self.exists_sw_corner(quad) { se } else { ne },
                    Edge::S => if self.exists_sw_corner(quad) { nw } else { ne },
                    Edge::NE => sw,
                    Edge::NW => se,
                    Edge::SW => ne,
                    Edge::SE => nw,
                    Edge::None => -1,
                };
                dir = if (self.z_level(point_opposite) >= level_index) ^ (level_index == 2) {
                    Dir::Right
                } else {
                    Dir::Left
                };
                self.set(quad, visited_mask);
            } else {
                let (point_left, point_right) = match qe.edge {
                    Edge::E => (sw, nw),
                    Edge::N => (se, sw),
                    Edge::W => (ne, se),
                    Edge::S => (nw, ne),
                    _ => (-1, -1),
                };
                let mut config = (((self.z_level(point_left) >= level_index) as u32) << 1)
                    | ((self.z_level(point_right) >= level_index) as u32);
                if level_index == 2 {
                    config = 3 - config;
                }
                if config == 1 {
                    let z = |p: isize| self.z[p as usize];
                    let zmid = 0.25 * (z(sw) + z(se) + z(nw) + z(ne));
                    self.set(quad, saddle_mask);
                    if (zmid > level) ^ (level_index == 2) {
                        dir = Dir::Right;
                    } else {
                        dir = Dir::Left;
                        self.set(quad, if level_index == 1 { MASK_SADDLE_LEFT_1 } else { MASK_SADDLE_LEFT_2 });
                    }
                    if qe.edge == Edge::N || qe.edge == Edge::E {
                        self.set(quad, if level_index == 1 { MASK_SADDLE_START_SW_1 } else { MASK_SADDLE_START_SW_2 });
                    }
                } else {
                    dir = match config {
                        0 => Dir::Left,
                        3 => Dir::Right,
                        _ => Dir::Straight,
                    };
                    self.set(quad, visited_mask);
                }
            }
            qe.edge = self.exit_edge(*qe, dir);
            if let Some(line) = line {
                if qe.edge == Edge::E {
                    self.set_parent(quad + 1, line);
                } else if qe.edge == Edge::W {
                    self.set_parent(quad, line);
                }
            }
            self.edge_interp(*qe, level, points);
            if self.is_edge_a_boundary(*qe) {
                break;
            }
            self.move_to_next_quad(qe);
            if let Some(start) = start_quad_edge {
                if *qe == start && level_index == start_level_index {
                    break;
                }
            }
        }
    }

    fn follow_boundary(
        &mut self,
        line: usize,
        qe: &mut QuadEdge,
        lower_level: f64,
        upper_level: f64,
        mut level_index: u32,
        start_quad_edge: QuadEdge,
    ) -> u32 {
        let nx = self.nx;
        let mut end_level = 0;
        let mut first_edge = true;
        let mut stop = false;
        loop {
            let start_level = if first_edge { self.z_level(self.edge_point_index(*qe, true)) } else { end_level };
            let end_point = self.edge_point_index(*qe, false);
            end_level = self.z_level(end_point);
            if level_index == 1 {
                if start_level <= level_index && end_level == 2 {
                    level_index = 2;
                    stop = true;
                } else if start_level >= 1 && end_level == 0 {
                    stop = true;
                }
            } else if start_level <= level_index && end_level == 2 {
                stop = true;
            } else if start_level >= 1 && end_level == 0 {
                level_index = 1;
                stop = true;
            }
            if !first_edge && !stop && *qe == start_quad_edge {
                break;
            }
            let quad = qe.quad;
            match qe.edge {
                Edge::E => self.set(quad + 1, MASK_VISITED_W),
                Edge::N => self.set(quad + nx, MASK_VISITED_S),
                Edge::W => self.set(quad, MASK_VISITED_W),
                Edge::S => self.set(quad, MASK_VISITED_S),
                Edge::NE | Edge::NW | Edge::SW | Edge::SE => self.set(quad, MASK_VISITED_CORNER),
                Edge::None => {}
            }
            if stop {
                let level = if level_index == 1 { lower_level } else { upper_level };
                let mut points = std::mem::take(&mut self.lines[line].points);
                self.edge_interp(*qe, level, &mut points);
                self.lines[line].points = points;
                break;
            }
            self.move_to_next_boundary_edge(qe);
            let quad = qe.quad;
            match qe.edge {
                Edge::W | Edge::SW | Edge::S | Edge::SE => {
                    if !self.exists_se_corner(quad) {
                        self.set_parent(quad, line);
                    }
                }
                Edge::E | Edge::NE | Edge::N | Edge::NW => {
                    if !self.exists_sw_corner(quad) {
                        self.set_parent(quad + 1, line);
                    }
                }
                Edge::None => {}
            }
            let p = end_point as usize;
            self.lines[line].points.push((self.x[p], self.y[p]));
            first_edge = false;
        }
        level_index
    }

    #[allow(clippy::too_many_arguments)]
    fn start_filled(
        &mut self,
        quad: isize,
        edge: Edge,
        start_level_index: u32,
        hole: bool,
        mut boundary: bool,
        lower_level: f64,
        upper_level: f64,
    ) -> usize {
        let line = self.lines.len();
        self.lines.push(Line { points: Vec::new(), is_hole: hole, parent: None, children: Vec::new() });
        if hole {
            let parent = self.parent_cache.get_parent(quad + 1);
            self.lines[line].parent = Some(parent);
            self.lines[parent].children.push(line);
        }
        let mut qe = QuadEdge { quad, edge };
        let start_quad_edge = qe;
        let mut level_index = start_level_index;
        loop {
            if !boundary {
                let level = if level_index == 1 { lower_level } else { upper_level };
                let mut points = std::mem::take(&mut self.lines[line].points);
                self.follow_interior(
                    &mut points,
                    Some(line),
                    &mut qe,
                    level_index,
                    level,
                    false,
                    Some(start_quad_edge),
                    start_level_index,
                );
                self.lines[line].points = points;
            } else {
                level_index = self.follow_boundary(line, &mut qe, lower_level, upper_level, level_index, start_quad_edge);
            }
            if qe == start_quad_edge && (boundary || level_index == start_level_index) {
                break;
            }
            boundary = !boundary;
        }
        line
    }

    fn single_quad_filled(&mut self, contour: &mut Vec<usize>, quad: isize, lower: f64, upper: f64) {
        let nx = self.nx;
        macro_rules! start {
            ($edge:expr, $li:expr, $hole:expr, $boundary:expr) => {
                contour.push(self.start_filled(quad, $edge, $li, $hole, $boundary, lower, upper))
            };
        }
        let (zsw, zse) = (|g: &Self| g.z_sw(quad), |g: &Self| g.z_se(quad));
        let (znw, zne) = (|g: &Self| g.z_nw(quad), |g: &Self| g.z_ne(quad));
        if self.boundary_s(quad) && self.exists_s_edge(quad) {
            if !self.visited_s(quad) && zsw(self) >= 1 && zse(self) == 0 {
                start!(Edge::S, 1, false, false);
            }
            if !self.visited_s(quad) && zsw(self) < 2 && zse(self) == 2 {
                start!(Edge::S, 2, false, false);
            }
            if !self.visited_s(quad) && zsw(self) <= 1 && zse(self) == 1 {
                start!(Edge::S, 1, false, true);
            }
            if !self.visited_s(quad) && zsw(self) == 2 && zse(self) == 1 {
                start!(Edge::S, 2, false, true);
            }
        }
        if self.boundary_w(quad) && self.exists_w_edge(quad) {
            if !self.visited_w(quad) && znw(self) >= 1 && zsw(self) == 0 {
                start!(Edge::W, 1, false, false);
            }
            if !self.visited_w(quad) && znw(self) < 2 && zsw(self) == 2 {
                start!(Edge::W, 2, false, false);
            }
            if !self.visited_w(quad) && znw(self) <= 1 && zsw(self) == 1 {
                start!(Edge::W, 1, false, true);
            }
            if !self.visited_w(quad) && znw(self) == 2 && zsw(self) == 1 {
                start!(Edge::W, 2, false, true);
            }
        }
        if self.exists_sw_corner(quad) {
            if !self.visited_corner(quad) && znw(self) == 1 && zse(self) == 1 {
                start!(Edge::NE, 1, true, true);
            }
        } else if self.exists_nw_corner(quad) {
            if !self.visited(quad, 1) && znw(self) == 0 && zsw(self) == 0 && zne(self) >= 1 {
                start!(Edge::N, 1, false, false);
            }
            if !self.visited(quad, 2) && znw(self) < 2 && zsw(self) < 2 && zne(self) == 2 {
                start!(Edge::SE, 2, true, false);
            }
            if !self.visited(quad, 2) && znw(self) == 2 && zsw(self) == 2 && zne(self) < 2 {
                start!(Edge::N, 2, false, false);
            }
            if !self.visited(quad, 1) && znw(self) >= 1 && zsw(self) >= 1 && zne(self) == 0 {
                start!(Edge::SE, 1, true, false);
            }
        } else if self.exists_se_corner(quad) {
            if !self.visited(quad, 1) && zsw(self) == 0 && zse(self) == 0 && zne(self) >= 1 {
                start!(Edge::NW, 1, false, false);
            }
            if !self.visited(quad, 2) && zsw(self) < 2 && zse(self) < 2 && zne(self) == 2 {
                start!(Edge::E, 2, true, false);
            }
            if !self.visited(quad, 2) && zsw(self) == 2 && zse(self) == 2 && zne(self) < 2 {
                start!(Edge::NW, 2, false, false);
            }
            if !self.visited(quad, 1) && zsw(self) >= 1 && zse(self) >= 1 && zne(self) == 0 {
                start!(Edge::E, 1, true, false);
            }
        } else if self.exists_ne_corner(quad) {
            if !self.visited_corner(quad) && znw(self) >= 1 && zse(self) == 0 {
                start!(Edge::SW, 1, false, false);
            }
            if !self.visited_corner(quad) && znw(self) < 2 && zse(self) == 2 {
                start!(Edge::SW, 2, false, false);
            }
            if !self.visited_corner(quad) && znw(self) <= 1 && zse(self) == 1 {
                start!(Edge::SW, 1, false, true);
            }
            if !self.visited_corner(quad) && znw(self) == 2 && zse(self) == 1 {
                start!(Edge::SW, 2, false, true);
            }
        }
        if self.exists_quad(quad) || self.exists_ne_corner(quad) {
            if !self.visited(quad, 1)
                && znw(self) == 0
                && zse(self) == 0
                && zne(self) >= 1
                && (!self.saddle(quad, 1) || self.saddle_left(quad, 1))
            {
                start!(Edge::N, 1, false, false);
            }
            if !self.visited(quad, 2)
                && znw(self) < 2
                && zse(self) < 2
                && zne(self) == 2
                && (!self.saddle(quad, 2) || !self.saddle_left(quad, 2))
            {
                start!(Edge::E, 2, true, false);
            }
            if !self.visited(quad, 2)
                && znw(self) == 2
                && zse(self) == 2
                && zne(self) < 2
                && (!self.saddle(quad, 2) || self.saddle_left(quad, 2))
            {
                start!(Edge::N, 2, false, false);
            }
            if !self.visited(quad, 1)
                && znw(self) >= 1
                && zse(self) >= 1
                && zne(self) == 0
                && (!self.saddle(quad, 1) || !self.saddle_left(quad, 1))
            {
                start!(Edge::E, 1, true, false);
            }
        }
        if self.boundary_n(quad)
            && self.exists_n_edge(quad)
            && !self.visited_s(quad + nx)
            && znw(self) == 1
            && zne(self) == 1
        {
            start!(Edge::N, 1, true, true);
        }
    }

    /// Filled contours between `lower_level` and `upper_level` (which the
    /// caller checks: not NaN, lower < upper), as outer boundaries followed
    /// by their holes.
    pub fn filled(&mut self, lower_level: f64, upper_level: f64) -> Output {
        self.init_cache_levels(lower_level, upper_level);
        let mut out = Output::default();
        let mut contour = Vec::new();
        for ijchunk in 0..self.chunk_count {
            let (ichunk, jchunk, istart, iend, jstart, jend) = self.chunk_limits(ijchunk);
            self.parent_cache.set_chunk_starts(istart, jstart);
            for j in jstart..jend {
                let quad_end = iend + j * self.nx;
                for quad in (istart + j * self.nx)..quad_end {
                    if !self.exists_none(quad) {
                        self.single_quad_filled(&mut contour, quad, lower_level, upper_level);
                    }
                }
            }
            if jchunk < self.nychunk - 1 {
                let quad_end = iend + jend * self.nx;
                for quad in (istart + jend * self.nx)..quad_end {
                    self.cache[quad as usize] &= !MASK_VISITED_S;
                }
            }
            if ichunk < self.nxchunk - 1 {
                let quad_end = iend + jend * self.nx;
                let mut quad = iend + jstart * self.nx;
                while quad < quad_end {
                    self.cache[quad as usize] &= !MASK_VISITED_W;
                    quad += self.nx;
                }
            }
            for &id in &contour {
                let line = &self.lines[id];
                if line.is_hole {
                    continue;
                }
                let mut npoints = line.points.len() + 1;
                for &c in &line.children {
                    npoints += self.lines[c].points.len() + 1;
                }
                let mut vertices = Vec::with_capacity(npoints);
                let mut codes = Vec::with_capacity(npoints);
                for l in std::iter::once(id).chain(line.children.iter().copied()) {
                    let points = &self.lines[l].points;
                    for (k, &p) in points.iter().enumerate() {
                        vertices.push(p);
                        codes.push(if k == 0 { MOVETO } else { LINETO });
                    }
                    if let Some(&first) = points.first() {
                        vertices.push(first);
                        codes.push(CLOSEPOLY);
                    }
                }
                out.vertices.push(vertices);
                out.codes.push(codes);
            }
            contour.clear();
            self.lines.clear();
        }
        out
    }

    fn push_line(points: &mut Vec<(f64, f64)>, out: &mut Output) {
        let n = points.len();
        let mut codes = Vec::with_capacity(n);
        for k in 0..n {
            codes.push(if k == 0 { MOVETO } else { LINETO });
        }
        if n > 1 && points[0] == points[n - 1] {
            codes[n - 1] = CLOSEPOLY;
        }
        out.vertices.push(std::mem::take(points));
        out.codes.push(codes);
    }

    fn start_line(&mut self, out: &mut Output, quad: isize, edge: Edge, level: f64) -> bool {
        let mut qe = QuadEdge { quad, edge };
        let mut points = Vec::new();
        self.follow_interior(&mut points, None, &mut qe, 1, level, true, None, 1);
        Self::push_line(&mut points, out);
        self.visited(quad, 1)
    }

    /// Contour lines at `level`: open line strips, then closed loops.
    pub fn lines(&mut self, level: f64) -> Output {
        self.init_cache_levels(level, level);
        let mut out = Output::default();
        let nx = self.nx;
        for ijchunk in 0..self.chunk_count {
            let (_, _, istart, iend, jstart, jend) = self.chunk_limits(ijchunk);
            for j in jstart..jend {
                let quad_end = iend + j * nx;
                for quad in (istart + j * nx)..quad_end {
                    if self.exists_none(quad) || self.visited(quad, 1) {
                        continue;
                    }
                    let (zsw, zse, znw, zne) = (self.z_sw(quad), self.z_se(quad), self.z_nw(quad), self.z_ne(quad));
                    if self.boundary_s(quad) && zsw >= 1 && zse < 1 && self.start_line(&mut out, quad, Edge::S, level) {
                        continue;
                    }
                    if self.boundary_w(quad) && znw >= 1 && zsw < 1 && self.start_line(&mut out, quad, Edge::W, level) {
                        continue;
                    }
                    if self.boundary_n(quad) && zne >= 1 && znw < 1 && self.start_line(&mut out, quad, Edge::N, level) {
                        continue;
                    }
                    if self.boundary_e(quad) && zse >= 1 && zne < 1 && self.start_line(&mut out, quad, Edge::E, level) {
                        continue;
                    }
                    if self.corner_mask {
                        if self.exists_sw_corner(quad) && zse >= 1 && znw < 1 && self.start_line(&mut out, quad, Edge::NE, level) {
                            continue;
                        }
                        if self.exists_se_corner(quad) && zne >= 1 && zsw < 1 && self.start_line(&mut out, quad, Edge::NW, level) {
                            continue;
                        }
                        if self.exists_nw_corner(quad) && zsw >= 1 && zne < 1 && self.start_line(&mut out, quad, Edge::SE, level) {
                            continue;
                        }
                        if self.exists_ne_corner(quad) && znw >= 1 && zse < 1 && self.start_line(&mut out, quad, Edge::SW, level) {
                            continue;
                        }
                    }
                }
            }
        }
        let mut points = Vec::new();
        for ijchunk in 0..self.chunk_count {
            let (_, _, istart, iend, jstart, jend) = self.chunk_limits(ijchunk);
            for j in jstart..jend {
                let quad_end = iend + j * nx;
                let mut quad = istart + j * nx;
                while quad < quad_end {
                    if self.exists_none(quad) || self.visited(quad, 1) {
                        quad += 1;
                        continue;
                    }
                    let start_edge = self.start_edge(quad, 1);
                    if start_edge == Edge::None {
                        quad += 1;
                        continue;
                    }
                    let mut qe = QuadEdge { quad, edge: start_edge };
                    let start_quad_edge = qe;
                    let ignore_first = start_edge == Edge::N;
                    self.follow_interior(&mut points, None, &mut qe, 1, level, !ignore_first, Some(start_quad_edge), 1);
                    if ignore_first && !points.is_empty() {
                        points.push(points[0]);
                    }
                    Self::push_line(&mut points, &mut out);
                    // Repeat if saddle point but not visited.
                    if !(self.saddle(quad, 1) && !self.visited(quad, 1)) {
                        quad += 1;
                    }
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(nx: usize, ny: usize, f: impl Fn(f64, f64) -> f64) -> Mpl2014 {
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut z = Vec::new();
        for j in 0..ny {
            for i in 0..nx {
                x.push(i as f64);
                y.push(j as f64);
                z.push(f(i as f64, j as f64));
            }
        }
        Mpl2014::new(x, y, z, nx, ny, None, true, 0, 0)
    }

    #[test]
    fn closed_loop_and_strip() {
        // A bump in the middle gives a closed loop around it.
        let mut g = grid(5, 5, |x, y| -((x - 2.0).powi(2) + (y - 2.0).powi(2)));
        let out = g.lines(-1.5);
        assert_eq!(out.vertices.len(), 1);
        let codes = &out.codes[0];
        assert_eq!(codes[0], MOVETO);
        assert_eq!(*codes.last().unwrap(), CLOSEPOLY);
        assert_eq!(out.vertices[0].len(), 13);
        assert_eq!(out.vertices[0][0], (2.0, 0.8333333333333334));
        // A ramp gives one open line strip, higher values on its left (as
        // produced by contourpy).
        let mut g = grid(4, 3, |x, _| x);
        let out = g.lines(1.5);
        assert_eq!(out.vertices, vec![vec![(1.5, 2.0), (1.5, 1.0), (1.5, 0.0)]]);
        assert_eq!(out.codes, vec![vec![MOVETO, LINETO, LINETO]]);
    }

    #[test]
    fn filled_ramp() {
        let mut g = grid(3, 2, |x, _| x);
        let out = g.filled(0.5, 1.5);
        assert_eq!(out.vertices.len(), 1);
        let v = &out.vertices[0];
        assert_eq!(v.first(), v.last());
        assert_eq!(out.codes[0][0], MOVETO);
        assert_eq!(*out.codes[0].last().unwrap(), CLOSEPOLY);
        assert!(v.iter().all(|p| p.0 >= 0.5 && p.0 <= 1.5));
    }
}
