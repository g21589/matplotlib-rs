//! Port of matplotlib's `_tri` extension (`src/tri/_tri.cpp`): triangulation
//! topology (edges, neighbors, boundaries), triangle contouring and the
//! trapezoid map triangle finder.

use std::collections::{BTreeMap, BTreeSet, HashMap};

pub const MOVETO: u8 = 1;
pub const LINETO: u8 = 2;
pub const CLOSEPOLY: u8 = 79;

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct XY {
    pub x: f64,
    pub y: f64,
}

impl XY {
    pub fn new(x: f64, y: f64) -> XY {
        XY { x, y }
    }
    fn cross_z(&self, o: &XY) -> f64 {
        self.x * o.y - self.y * o.x
    }
    /// Compares x then y.
    fn is_right_of(&self, o: &XY) -> bool {
        if self.x == o.x {
            self.y > o.y
        } else {
            self.x > o.x
        }
    }
    fn sub(&self, o: &XY) -> XY {
        XY::new(self.x - o.x, self.y - o.y)
    }
    fn add(&self, o: &XY) -> XY {
        XY::new(self.x + o.x, self.y + o.y)
    }
    fn mul(&self, m: f64) -> XY {
        XY::new(self.x * m, self.y * m)
    }
}

struct XYZ {
    x: f64,
    y: f64,
    z: f64,
}

impl XYZ {
    fn cross(&self, o: &XYZ) -> XYZ {
        XYZ { x: self.y * o.z - self.z * o.y, y: self.z * o.x - self.x * o.z, z: self.x * o.y - self.y * o.x }
    }
    fn dot(&self, o: &XYZ) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }
    fn sub(&self, o: &XYZ) -> XYZ {
        XYZ { x: self.x - o.x, y: self.y - o.y, z: self.z - o.z }
    }
}

/// A triangle edge: (triangle, edge index 0..3).
pub type TriEdge = (i32, i32);

// ---------------------------------------------------------------------------
// Triangulation

#[derive(Clone, Debug)]
pub struct Triangulation {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    /// (ntri, 3) point indices
    pub triangles: Vec<i32>,
    /// empty or ntri
    pub mask: Vec<bool>,
    /// empty or (nedges, 2)
    pub edges: Vec<i32>,
    /// empty or (ntri, 3)
    pub neighbors: Vec<i32>,
    boundaries: Vec<Vec<TriEdge>>,
    tri_edge_to_boundary: BTreeMap<TriEdge, (usize, usize)>,
}

impl Triangulation {
    pub fn new(x: Vec<f64>, y: Vec<f64>, triangles: Vec<i32>, mask: Vec<bool>, edges: Vec<i32>, neighbors: Vec<i32>) -> Self {
        Triangulation { x, y, triangles, mask, edges, neighbors, boundaries: Vec::new(), tri_edge_to_boundary: BTreeMap::new() }
    }

    pub fn npoints(&self) -> usize {
        self.x.len()
    }

    pub fn ntri(&self) -> usize {
        self.triangles.len() / 3
    }

    pub fn is_masked(&self, tri: usize) -> bool {
        !self.mask.is_empty() && self.mask[tri]
    }

    fn has_neighbors(&self) -> bool {
        !self.neighbors.is_empty()
    }

    pub fn point(&self, p: i32) -> XY {
        XY::new(self.x[p as usize], self.y[p as usize])
    }

    pub fn tri_point(&self, tri: i32, edge: i32) -> i32 {
        self.triangles[3 * tri as usize + edge as usize]
    }

    /// Make all triangles anticlockwise (swapping their neighbors too).
    /// Returns the indices of the triangles that were flipped.
    pub fn correct_triangles(x: &[f64], y: &[f64], triangles: &mut [i32], neighbors: &mut [i32]) {
        let ntri = triangles.len() / 3;
        let pt = |p: i32| XY::new(x[p as usize], y[p as usize]);
        for tri in 0..ntri {
            let p0 = pt(triangles[3 * tri]);
            let p1 = pt(triangles[3 * tri + 1]);
            let p2 = pt(triangles[3 * tri + 2]);
            if p1.sub(&p0).cross_z(&p2.sub(&p0)) < 0.0 {
                triangles.swap(3 * tri + 1, 3 * tri + 2);
                if !neighbors.is_empty() {
                    neighbors.swap(3 * tri + 1, 3 * tri + 2);
                }
            }
        }
    }

    pub fn calculate_edges(&mut self) {
        let mut set: BTreeSet<(i32, i32)> = BTreeSet::new();
        for tri in 0..self.ntri() {
            if !self.is_masked(tri) {
                for edge in 0..3 {
                    let start = self.tri_point(tri as i32, edge);
                    let end = self.tri_point(tri as i32, (edge + 1) % 3);
                    set.insert(if start > end { (start, end) } else { (end, start) });
                }
            }
        }
        self.edges = set.into_iter().flat_map(|(a, b)| [a, b]).collect();
    }

    pub fn get_edges(&mut self) -> &[i32] {
        if self.edges.is_empty() {
            self.calculate_edges();
        }
        &self.edges
    }

    pub fn calculate_neighbors(&mut self) {
        let ntri = self.ntri();
        let mut neighbors = vec![-1i32; 3 * ntri];
        let mut map: HashMap<(i32, i32), TriEdge> = HashMap::new();
        for tri in 0..ntri {
            if !self.is_masked(tri) {
                for edge in 0..3 {
                    let start = self.tri_point(tri as i32, edge);
                    let end = self.tri_point(tri as i32, (edge + 1) % 3);
                    match map.remove(&(end, start)) {
                        None => {
                            map.insert((start, end), (tri as i32, edge));
                        }
                        Some((t, e)) => {
                            neighbors[3 * tri + edge as usize] = t;
                            neighbors[3 * t as usize + e as usize] = tri as i32;
                        }
                    }
                }
            }
        }
        self.neighbors = neighbors;
    }

    pub fn get_neighbors(&mut self) -> &[i32] {
        if !self.has_neighbors() {
            self.calculate_neighbors();
        }
        &self.neighbors
    }

    pub fn ensure_neighbors(&mut self) {
        if !self.has_neighbors() {
            self.calculate_neighbors();
        }
    }

    /// Requires the neighbors (see [`Triangulation::ensure_neighbors`]).
    pub fn neighbor(&self, tri: i32, edge: i32) -> i32 {
        self.neighbors[3 * tri as usize + edge as usize]
    }

    fn edge_in_triangle(&self, tri: i32, point: i32) -> i32 {
        (0..3).find(|&e| self.tri_point(tri, e) == point).unwrap_or(-1)
    }

    pub fn neighbor_edge(&self, tri: i32, edge: i32) -> TriEdge {
        let n = self.neighbor(tri, edge);
        if n == -1 {
            (-1, -1)
        } else {
            (n, self.edge_in_triangle(n, self.tri_point(tri, (edge + 1) % 3)))
        }
    }

    pub fn set_mask(&mut self, mask: Vec<bool>) {
        self.mask = mask;
        self.edges.clear();
        self.neighbors.clear();
        self.boundaries.clear();
        self.tri_edge_to_boundary.clear();
    }

    fn calculate_boundaries(&mut self) {
        self.ensure_neighbors();
        let mut boundary_edges: BTreeSet<TriEdge> = BTreeSet::new();
        for tri in 0..self.ntri() {
            if !self.is_masked(tri) {
                for edge in 0..3 {
                    if self.neighbor(tri as i32, edge) == -1 {
                        boundary_edges.insert((tri as i32, edge));
                    }
                }
            }
        }
        while let Some(&first) = boundary_edges.iter().next() {
            let mut it = Some(first);
            let (mut tri, mut edge) = first;
            let mut boundary: Vec<TriEdge> = Vec::new();
            let bidx = self.boundaries.len();
            loop {
                boundary.push((tri, edge));
                if let Some(k) = it {
                    boundary_edges.remove(&k);
                }
                self.tri_edge_to_boundary.insert((tri, edge), (bidx, boundary.len() - 1));
                edge = (edge + 1) % 3;
                let point = self.tri_point(tri, edge);
                while self.neighbor(tri, edge) != -1 {
                    tri = self.neighbor(tri, edge);
                    edge = self.edge_in_triangle(tri, point);
                }
                if (tri, edge) == boundary[0] {
                    break;
                }
                it = boundary_edges.get(&(tri, edge)).copied();
            }
            self.boundaries.push(boundary);
        }
    }

    pub fn ensure_boundaries(&mut self) {
        if self.boundaries.is_empty() {
            self.calculate_boundaries();
        }
    }

    pub fn boundaries(&self) -> &Vec<Vec<TriEdge>> {
        &self.boundaries
    }

    fn boundary_edge(&self, te: TriEdge) -> (usize, usize) {
        self.tri_edge_to_boundary[&te]
    }

    /// Plane coefficients (a, b, c) with z = a*x + b*y + c for each triangle.
    pub fn calculate_plane_coefficients(&self, z: &[f64]) -> Vec<f64> {
        let ntri = self.ntri();
        let mut planes = vec![0.0; 3 * ntri];
        for tri in 0..ntri {
            if self.is_masked(tri) {
                continue;
            }
            let p = |e: usize| {
                let i = self.triangles[3 * tri + e] as usize;
                XYZ { x: self.x[i], y: self.y[i], z: z[i] }
            };
            let point0 = p(0);
            let side01 = p(1).sub(&point0);
            let side02 = p(2).sub(&point0);
            let normal = side01.cross(&side02);
            if normal.z == 0.0 {
                let sum2 = side01.x * side01.x + side01.y * side01.y + side02.x * side02.x + side02.y * side02.y;
                let a = (side01.x * side01.z + side02.x * side02.z) / sum2;
                let b = (side01.y * side01.z + side02.y * side02.z) / sum2;
                planes[3 * tri] = a;
                planes[3 * tri + 1] = b;
                planes[3 * tri + 2] = point0.z - a * point0.x - b * point0.y;
            } else {
                planes[3 * tri] = -normal.x / normal.z;
                planes[3 * tri + 1] = -normal.y / normal.z;
                planes[3 * tri + 2] = normal.dot(&point0) / normal.z;
            }
        }
        planes
    }
}

// ---------------------------------------------------------------------------
// TriContourGenerator

/// A contour line; consecutive duplicate points are dropped.
#[derive(Default, Clone, Debug)]
pub struct ContourLine(pub Vec<XY>);

impl ContourLine {
    fn push(&mut self, p: XY) {
        if self.0.last() != Some(&p) {
            self.0.push(p);
        }
    }
}

pub struct TriContourGenerator {
    tri: Triangulation,
    z: Vec<f64>,
    interior_visited: Vec<bool>,
    boundaries_visited: Vec<Vec<bool>>,
    boundaries_used: Vec<bool>,
}

impl TriContourGenerator {
    pub fn new(mut tri: Triangulation, z: Vec<f64>) -> Self {
        tri.ensure_neighbors();
        let n = 2 * tri.ntri();
        TriContourGenerator { tri, z, interior_visited: vec![false; n], boundaries_visited: Vec::new(), boundaries_used: Vec::new() }
    }

    fn clear_visited_flags(&mut self, include_boundaries: bool) {
        self.interior_visited.iter_mut().for_each(|v| *v = false);
        if include_boundaries {
            if self.boundaries_visited.is_empty() {
                self.tri.ensure_boundaries();
                let b = self.tri.boundaries();
                self.boundaries_visited = b.iter().map(|bd| vec![false; bd.len()]).collect();
                self.boundaries_used = vec![false; b.len()];
            }
            for v in self.boundaries_visited.iter_mut() {
                v.iter_mut().for_each(|x| *x = false);
            }
            self.boundaries_used.iter_mut().for_each(|x| *x = false);
        }
    }

    fn get_z(&self, p: i32) -> f64 {
        self.z[p as usize]
    }

    fn interp(&self, p1: i32, p2: i32, level: f64) -> XY {
        let fraction = (self.get_z(p2) - level) / (self.get_z(p2) - self.get_z(p1));
        self.tri.point(p1).mul(fraction).add(&self.tri.point(p2).mul(1.0 - fraction))
    }

    fn edge_interp(&self, tri: i32, edge: i32, level: f64) -> XY {
        self.interp(self.tri.tri_point(tri, edge), self.tri.tri_point(tri, (edge + 1) % 3), level)
    }

    fn get_exit_edge(&self, tri: i32, level: f64, on_upper: bool) -> i32 {
        let mut config = (self.get_z(self.tri.tri_point(tri, 0)) >= level) as u32
            | ((self.get_z(self.tri.tri_point(tri, 1)) >= level) as u32) << 1
            | ((self.get_z(self.tri.tri_point(tri, 2)) >= level) as u32) << 2;
        if on_upper {
            config = 7 - config;
        }
        match config {
            1 => 2,
            2 => 0,
            3 => 2,
            4 => 1,
            5 => 1,
            6 => 0,
            _ => -1,
        }
    }

    /// Contour lines at `level`: one (points, closed) entry per line.
    pub fn create_contour(&mut self, level: f64) -> Vec<ContourLine> {
        self.clear_visited_flags(false);
        let mut contour = Vec::new();
        self.find_boundary_lines(&mut contour, level);
        self.find_interior_lines(&mut contour, level, false);
        contour
    }

    pub fn create_filled_contour(&mut self, lower: f64, upper: f64) -> Vec<ContourLine> {
        self.clear_visited_flags(true);
        let mut contour = Vec::new();
        self.find_boundary_lines_filled(&mut contour, lower, upper);
        self.find_interior_lines(&mut contour, lower, false);
        self.find_interior_lines(&mut contour, upper, true);
        contour
    }

    fn find_boundary_lines(&mut self, contour: &mut Vec<ContourLine>, level: f64) {
        self.tri.ensure_boundaries();
        let boundaries = self.tri.boundaries().clone();
        for boundary in &boundaries {
            let mut end_above = false;
            for (k, &te) in boundary.iter().enumerate() {
                let start_above =
                    if k == 0 { self.get_z(self.tri.tri_point(te.0, te.1)) >= level } else { end_above };
                end_above = self.get_z(self.tri.tri_point(te.0, (te.1 + 1) % 3)) >= level;
                if start_above && !end_above {
                    let mut line = ContourLine::default();
                    let mut tri_edge = te;
                    self.follow_interior(&mut line, &mut tri_edge, true, level, false);
                    contour.push(line);
                }
            }
        }
    }

    fn find_boundary_lines_filled(&mut self, contour: &mut Vec<ContourLine>, lower: f64, upper: f64) {
        let boundaries = self.tri.boundaries().clone();
        for i in 0..boundaries.len() {
            let boundary = &boundaries[i];
            for j in 0..boundary.len() {
                if self.boundaries_visited[i][j] {
                    continue;
                }
                let z_start = self.get_z(self.tri.tri_point(boundary[j].0, boundary[j].1));
                let z_end = self.get_z(self.tri.tri_point(boundary[j].0, (boundary[j].1 + 1) % 3));
                let incr_upper = z_start < upper && z_end >= upper;
                let decr_lower = z_start >= lower && z_end < lower;
                if decr_lower || incr_upper {
                    let mut line = ContourLine::default();
                    let start_tri_edge = boundary[j];
                    let mut tri_edge = start_tri_edge;
                    let mut on_upper = incr_upper;
                    loop {
                        self.follow_interior(&mut line, &mut tri_edge, true, if on_upper { upper } else { lower }, on_upper);
                        on_upper = self.follow_boundary(&mut line, &mut tri_edge, lower, upper, on_upper);
                        if tri_edge == start_tri_edge {
                            break;
                        }
                    }
                    let first = line.0[0];
                    line.push(first);
                    contour.push(line);
                }
            }
        }
        for i in 0..boundaries.len() {
            if !self.boundaries_used[i] {
                let boundary = &boundaries[i];
                let z = self.get_z(self.tri.tri_point(boundary[0].0, boundary[0].1));
                if z >= lower && z < upper {
                    let mut line = ContourLine::default();
                    for &(t, e) in boundary {
                        line.push(self.tri.point(self.tri.tri_point(t, e)));
                    }
                    let first = line.0[0];
                    line.push(first);
                    contour.push(line);
                }
            }
        }
    }

    fn find_interior_lines(&mut self, contour: &mut Vec<ContourLine>, level: f64, on_upper: bool) {
        let ntri = self.tri.ntri();
        for tri in 0..ntri {
            let visited_index = if on_upper { tri + ntri } else { tri };
            if self.interior_visited[visited_index] || self.tri.is_masked(tri) {
                continue;
            }
            self.interior_visited[visited_index] = true;
            let edge = self.get_exit_edge(tri as i32, level, on_upper);
            if edge == -1 {
                continue;
            }
            let mut line = ContourLine::default();
            let mut tri_edge = self.tri.neighbor_edge(tri as i32, edge);
            self.follow_interior(&mut line, &mut tri_edge, false, level, on_upper);
            let first = line.0[0];
            line.push(first);
            contour.push(line);
        }
    }

    fn follow_boundary(&mut self, line: &mut ContourLine, tri_edge: &mut TriEdge, lower: f64, upper: f64, mut on_upper: bool) -> bool {
        let (boundary, mut edge) = self.tri.boundary_edge(*tri_edge);
        self.boundaries_used[boundary] = true;
        let mut stop = false;
        let mut first_edge = true;
        let mut z_end = 0.0;
        while !stop {
            self.boundaries_visited[boundary][edge] = true;
            let z_start = if first_edge { self.get_z(self.tri.tri_point(tri_edge.0, tri_edge.1)) } else { z_end };
            z_end = self.get_z(self.tri.tri_point(tri_edge.0, (tri_edge.1 + 1) % 3));
            if z_end > z_start {
                if !(!on_upper && first_edge) && z_end >= lower && z_start < lower {
                    stop = true;
                    on_upper = false;
                } else if z_end >= upper && z_start < upper {
                    stop = true;
                    on_upper = true;
                }
            } else if !(on_upper && first_edge) && z_start >= upper && z_end < upper {
                stop = true;
                on_upper = true;
            } else if z_start >= lower && z_end < lower {
                stop = true;
                on_upper = false;
            }
            first_edge = false;
            if !stop {
                let n = self.tri.boundaries()[boundary].len();
                edge = (edge + 1) % n;
                *tri_edge = self.tri.boundaries()[boundary][edge];
                line.push(self.tri.point(self.tri.tri_point(tri_edge.0, tri_edge.1)));
            }
        }
        on_upper
    }

    fn follow_interior(&mut self, line: &mut ContourLine, tri_edge: &mut TriEdge, end_on_boundary: bool, level: f64, on_upper: bool) {
        line.push(self.edge_interp(tri_edge.0, tri_edge.1, level));
        loop {
            let mut visited_index = tri_edge.0 as usize;
            if on_upper {
                visited_index += self.tri.ntri();
            }
            if !end_on_boundary && self.interior_visited[visited_index] {
                break;
            }
            tri_edge.1 = self.get_exit_edge(tri_edge.0, level, on_upper);
            self.interior_visited[visited_index] = true;
            line.push(self.edge_interp(tri_edge.0, tri_edge.1, level));
            let next = self.tri.neighbor_edge(tri_edge.0, tri_edge.1);
            if end_on_boundary && next.0 == -1 {
                break;
            }
            *tri_edge = next;
        }
    }
}

/// `contour_line_to_segs_and_kinds`: per line, its vertices and codes.
pub fn contour_lines_to_segs(contour: &[ContourLine]) -> Vec<(Vec<[f64; 2]>, Vec<u8>)> {
    contour
        .iter()
        .map(|line| {
            let pts = &line.0;
            let segs: Vec<[f64; 2]> = pts.iter().map(|p| [p.x, p.y]).collect();
            let mut codes = vec![LINETO; pts.len()];
            if !pts.is_empty() {
                codes[0] = MOVETO;
            }
            if pts.len() > 1 && pts[0] == pts[pts.len() - 1] {
                *codes.last_mut().unwrap() = CLOSEPOLY;
            }
            (segs, codes)
        })
        .collect()
}

/// `contour_to_segs_and_kinds`: all polygons in one vertices/codes pair.
pub fn contour_to_segs(contour: &[ContourLine]) -> (Vec<[f64; 2]>, Vec<u8>) {
    let mut segs = Vec::new();
    let mut codes = Vec::new();
    for line in contour {
        for (k, p) in line.0.iter().enumerate() {
            segs.push([p.x, p.y]);
            codes.push(if k == 0 { MOVETO } else { LINETO });
        }
        if line.0.len() > 1 {
            *codes.last_mut().unwrap() = CLOSEPOLY;
        }
    }
    (segs, codes)
}

// ---------------------------------------------------------------------------
// std::mt19937 and libstdc++'s std::shuffle

pub struct Mt19937 {
    mt: [u32; 624],
    idx: usize,
}

impl Mt19937 {
    pub fn new(seed: u32) -> Self {
        let mut mt = [0u32; 624];
        mt[0] = seed;
        for i in 1..624 {
            mt[i] = 1812433253u32.wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_add(i as u32);
        }
        Mt19937 { mt, idx: 624 }
    }

    pub fn next_u32(&mut self) -> u32 {
        if self.idx >= 624 {
            for i in 0..624 {
                let y = (self.mt[i] & 0x8000_0000) | (self.mt[(i + 1) % 624] & 0x7fff_ffff);
                let mut v = self.mt[(i + 397) % 624] ^ (y >> 1);
                if y & 1 != 0 {
                    v ^= 0x9908_b0df;
                }
                self.mt[i] = v;
            }
            self.idx = 0;
        }
        let mut y = self.mt[self.idx];
        self.idx += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// libstdc++'s `uniform_int_distribution<unsigned long>{0, range-1}`
    /// as of GCC 10 (which matplotlib's wheels are built with): downscaling
    /// by division with rejection.
    fn uniform(&mut self, range: u64) -> u64 {
        let urngrange = u32::MAX as u64;
        let scaling = urngrange / range;
        let past = range * scaling;
        loop {
            let ret = self.next_u32() as u64;
            if ret < past {
                return ret / scaling;
            }
        }
    }
}

/// libstdc++'s `std::shuffle(first, last, mt19937&)`.
pub fn std_shuffle<T>(v: &mut [T], g: &mut Mt19937) {
    let n = v.len() as u64;
    if n == 0 {
        return;
    }
    let urngrange: u64 = u32::MAX as u64;
    if urngrange / n >= n {
        let mut i = 1u64;
        if n % 2 == 0 {
            let j = g.uniform(2);
            v.swap(i as usize, j as usize);
            i += 1;
        }
        while i != n {
            let swap_range = i + 1;
            let x = g.uniform(swap_range * (swap_range + 1));
            let (p0, p1) = (x / (swap_range + 1), x % (swap_range + 1));
            v.swap(i as usize, p0 as usize);
            i += 1;
            v.swap(i as usize, p1 as usize);
            i += 1;
        }
        return;
    }
    for i in 1..n {
        let j = g.uniform(i + 1);
        v.swap(i as usize, j as usize);
    }
}

// ---------------------------------------------------------------------------
// TrapezoidMapTriFinder

#[derive(Clone, Copy)]
struct TPoint {
    xy: XY,
    tri: i32,
}

#[derive(Clone, Copy)]
struct TEdge {
    left: usize,
    right: usize,
    triangle_below: i32,
    triangle_above: i32,
    point_below: Option<usize>,
    point_above: Option<usize>,
}

#[derive(Clone, Copy)]
struct Trapezoid {
    left: usize,
    right: usize,
    below: usize,
    above: usize,
    lower_left: Option<usize>,
    lower_right: Option<usize>,
    upper_left: Option<usize>,
    upper_right: Option<usize>,
    node: usize,
}

#[derive(Clone)]
enum NodeKind {
    X { point: usize, left: usize, right: usize },
    Y { edge: usize, below: usize, above: usize },
    Trap(usize),
}

#[derive(Clone)]
struct Node {
    kind: NodeKind,
    parents: Vec<usize>,
}

pub struct TrapezoidMapTriFinder {
    points: Vec<TPoint>,
    edges: Vec<TEdge>,
    traps: Vec<Trapezoid>,
    nodes: Vec<Node>,
    tree: usize,
}

impl TrapezoidMapTriFinder {
    fn edge_orient(&self, e: usize, xy: &XY) -> i32 {
        let ed = &self.edges[e];
        let l = self.points[ed.left].xy;
        let r = self.points[ed.right].xy;
        let cross_z = xy.sub(&l).cross_z(&r.sub(&l));
        if cross_z > 0.0 {
            1
        } else if cross_z < 0.0 {
            -1
        } else {
            0
        }
    }

    fn edge_slope(&self, e: usize) -> f64 {
        let ed = &self.edges[e];
        let d = self.points[ed.right].xy.sub(&self.points[ed.left].xy);
        d.y / d.x
    }

    fn new_node(&mut self, kind: NodeKind) -> usize {
        let id = self.nodes.len();
        match kind {
            NodeKind::X { left, right, .. } => {
                self.nodes[left].parents.push(id);
                self.nodes[right].parents.push(id);
            }
            NodeKind::Y { below, above, .. } => {
                self.nodes[below].parents.push(id);
                self.nodes[above].parents.push(id);
            }
            NodeKind::Trap(t) => self.traps[t].node = id,
        }
        self.nodes.push(Node { kind, parents: Vec::new() });
        id
    }

    fn new_trap(&mut self, left: usize, right: usize, below: usize, above: usize) -> usize {
        self.traps.push(Trapezoid {
            left,
            right,
            below,
            above,
            lower_left: None,
            lower_right: None,
            upper_left: None,
            upper_right: None,
            node: usize::MAX,
        });
        self.traps.len() - 1
    }

    fn set_lower_left(&mut self, t: usize, v: Option<usize>) {
        self.traps[t].lower_left = v;
        if let Some(o) = v {
            self.traps[o].lower_right = Some(t);
        }
    }
    fn set_lower_right(&mut self, t: usize, v: Option<usize>) {
        self.traps[t].lower_right = v;
        if let Some(o) = v {
            self.traps[o].lower_left = Some(t);
        }
    }
    fn set_upper_left(&mut self, t: usize, v: Option<usize>) {
        self.traps[t].upper_left = v;
        if let Some(o) = v {
            self.traps[o].upper_right = Some(t);
        }
    }
    fn set_upper_right(&mut self, t: usize, v: Option<usize>) {
        self.traps[t].upper_right = v;
        if let Some(o) = v {
            self.traps[o].upper_left = Some(t);
        }
    }

    fn remove_parent(&mut self, node: usize, parent: usize) {
        let p = &mut self.nodes[node].parents;
        if let Some(i) = p.iter().position(|&x| x == parent) {
            p.remove(i);
        }
    }

    fn replace_child(&mut self, node: usize, old: usize, new: usize) {
        match &mut self.nodes[node].kind {
            NodeKind::X { left, right, .. } => {
                if *left == old {
                    *left = new;
                } else {
                    *right = new;
                }
            }
            NodeKind::Y { below, above, .. } => {
                if *below == old {
                    *below = new;
                } else {
                    *above = new;
                }
            }
            NodeKind::Trap(_) => {}
        }
        self.remove_parent(old, node);
        self.nodes[new].parents.push(node);
    }

    fn replace_with(&mut self, node: usize, new: usize) {
        while let Some(&p) = self.nodes[node].parents.first() {
            self.replace_child(p, node, new);
        }
    }

    fn node_tri(&self, n: usize) -> i32 {
        match self.nodes[n].kind {
            NodeKind::X { point, .. } => self.points[point].tri,
            NodeKind::Y { edge, .. } => {
                let e = &self.edges[edge];
                if e.triangle_above != -1 {
                    e.triangle_above
                } else {
                    e.triangle_below
                }
            }
            NodeKind::Trap(t) => self.edges[self.traps[t].below].triangle_above,
        }
    }

    fn search_xy(&self, xy: &XY) -> usize {
        let mut n = self.tree;
        loop {
            match self.nodes[n].kind {
                NodeKind::X { point, left, right } => {
                    let p = self.points[point].xy;
                    if *xy == p {
                        return n;
                    } else if xy.is_right_of(&p) {
                        n = right;
                    } else {
                        n = left;
                    }
                }
                NodeKind::Y { edge, below, above } => {
                    let o = self.edge_orient(edge, xy);
                    if o == 0 {
                        return n;
                    } else if o < 0 {
                        n = above;
                    } else {
                        n = below;
                    }
                }
                NodeKind::Trap(_) => return n,
            }
        }
    }

    fn search_edge(&self, e: usize) -> Option<usize> {
        let edge = self.edges[e];
        let mut n = self.tree;
        loop {
            match self.nodes[n].kind {
                NodeKind::X { point, left, right } => {
                    if edge.left == point || self.points[edge.left].xy.is_right_of(&self.points[point].xy) {
                        n = right;
                    } else {
                        n = left;
                    }
                }
                NodeKind::Y { edge: ne, below, above } => {
                    let nedge = self.edges[ne];
                    if edge.left == nedge.left {
                        if self.edge_slope(e) == self.edge_slope(ne) {
                            if nedge.triangle_above == edge.triangle_below {
                                n = above;
                            } else if nedge.triangle_below == edge.triangle_above {
                                n = below;
                            } else {
                                return None;
                            }
                            continue;
                        }
                        n = if self.edge_slope(e) > self.edge_slope(ne) { above } else { below };
                    } else if edge.right == nedge.right {
                        if self.edge_slope(e) == self.edge_slope(ne) {
                            if nedge.triangle_above == edge.triangle_below {
                                n = above;
                            } else if nedge.triangle_below == edge.triangle_above {
                                n = below;
                            } else {
                                return None;
                            }
                            continue;
                        }
                        n = if self.edge_slope(e) > self.edge_slope(ne) { below } else { above };
                    } else {
                        let mut orient = self.edge_orient(ne, &self.points[edge.left].xy);
                        if orient == 0 {
                            let has = |p: usize| edge.left == p || edge.right == p;
                            if nedge.point_above.map(has).unwrap_or(false) {
                                orient = -1;
                            } else if nedge.point_below.map(has).unwrap_or(false) {
                                orient = 1;
                            } else {
                                return None;
                            }
                        }
                        n = if orient < 0 { above } else { below };
                    }
                }
                NodeKind::Trap(t) => return Some(t),
            }
        }
    }

    fn find_trapezoids_intersecting_edge(&self, e: usize) -> Option<Vec<usize>> {
        let edge = self.edges[e];
        let mut trap = self.search_edge(e)?;
        let mut out = vec![trap];
        while self.points[edge.right].xy.is_right_of(&self.points[self.traps[trap].right].xy) {
            let tr = self.traps[trap].right;
            let mut orient = self.edge_orient(e, &self.points[tr].xy);
            if orient == 0 {
                if edge.point_below == Some(tr) {
                    orient = 1;
                } else if edge.point_above == Some(tr) {
                    orient = -1;
                } else {
                    return None;
                }
            }
            let next = if orient == -1 { self.traps[trap].lower_right } else { self.traps[trap].upper_right };
            trap = next?;
            out.push(trap);
        }
        Some(out)
    }

    fn add_edge_to_tree(&mut self, e: usize) -> bool {
        let Some(trapezoids) = self.find_trapezoids_intersecting_edge(e) else { return false };
        let p = self.edges[e].left;
        let q = self.edges[e].right;
        let mut left_old: Option<usize> = None;
        let mut left_below: Option<usize> = None;
        let mut left_above: Option<usize> = None;
        let ntraps = trapezoids.len();
        for i in 0..ntraps {
            let old = trapezoids[i];
            let o = self.traps[old];
            let start_trap = i == 0;
            let end_trap = i == ntraps - 1;
            let have_left = start_trap && self.edges[e].left != o.left;
            let have_right = end_trap && self.edges[e].right != o.right;
            let mut left = None;
            let below;
            let above;
            let mut right = None;
            if start_trap && end_trap {
                if have_left {
                    left = Some(self.new_trap(o.left, p, o.below, o.above));
                }
                below = self.new_trap(p, q, o.below, e);
                above = self.new_trap(p, q, e, o.above);
                if have_right {
                    right = Some(self.new_trap(q, o.right, o.below, o.above));
                }
                if let Some(l) = left {
                    self.set_lower_left(l, o.lower_left);
                    self.set_upper_left(l, o.upper_left);
                    self.set_lower_right(l, Some(below));
                    self.set_upper_right(l, Some(above));
                } else {
                    self.set_lower_left(below, o.lower_left);
                    self.set_upper_left(above, o.upper_left);
                }
                if let Some(r) = right {
                    self.set_lower_right(r, o.lower_right);
                    self.set_upper_right(r, o.upper_right);
                    self.set_lower_right(below, Some(r));
                    self.set_upper_right(above, Some(r));
                } else {
                    self.set_lower_right(below, o.lower_right);
                    self.set_upper_right(above, o.upper_right);
                }
            } else if start_trap {
                if have_left {
                    left = Some(self.new_trap(o.left, p, o.below, o.above));
                }
                below = self.new_trap(p, o.right, o.below, e);
                above = self.new_trap(p, o.right, e, o.above);
                if let Some(l) = left {
                    self.set_lower_left(l, o.lower_left);
                    self.set_upper_left(l, o.upper_left);
                    self.set_lower_right(l, Some(below));
                    self.set_upper_right(l, Some(above));
                } else {
                    self.set_lower_left(below, o.lower_left);
                    self.set_upper_left(above, o.upper_left);
                }
                self.set_lower_right(below, o.lower_right);
                self.set_upper_right(above, o.upper_right);
            } else {
                let lb = left_below.unwrap();
                let la = left_above.unwrap();
                let (bright, aright) = if end_trap { (q, q) } else { (o.right, o.right) };
                if self.traps[lb].below == o.below {
                    below = lb;
                    self.traps[below].right = bright;
                } else {
                    below = self.new_trap(o.left, bright, o.below, e);
                }
                if self.traps[la].above == o.above {
                    above = la;
                    self.traps[above].right = aright;
                } else {
                    above = self.new_trap(o.left, aright, e, o.above);
                }
                // (end trapezoid) connect to the right first, as matplotlib does
                if end_trap {
                    if have_right {
                        right = Some(self.new_trap(q, o.right, o.below, o.above));
                    }
                    if let Some(r) = right {
                        self.set_lower_right(r, o.lower_right);
                        self.set_upper_right(r, o.upper_right);
                        self.set_lower_right(below, Some(r));
                        self.set_upper_right(above, Some(r));
                    } else {
                        self.set_lower_right(below, o.lower_right);
                        self.set_upper_right(above, o.upper_right);
                    }
                }
                // connect to the new trapezoids replacing the previous old one
                if below != lb {
                    self.set_upper_left(below, Some(lb));
                    if o.lower_left == left_old {
                        self.set_lower_left(below, Some(lb));
                    } else {
                        self.set_lower_left(below, o.lower_left);
                    }
                }
                if above != la {
                    self.set_lower_left(above, Some(la));
                    if o.upper_left == left_old {
                        self.set_upper_left(above, Some(la));
                    } else {
                        self.set_upper_left(above, o.upper_left);
                    }
                }
                if !end_trap {
                    self.set_lower_right(below, o.lower_right);
                    self.set_upper_right(above, o.upper_right);
                }
            }

            let below_node =
                if Some(below) == left_below { self.traps[below].node } else { self.new_node(NodeKind::Trap(below)) };
            let above_node =
                if Some(above) == left_above { self.traps[above].node } else { self.new_node(NodeKind::Trap(above)) };
            let mut top = self.new_node(NodeKind::Y { edge: e, below: below_node, above: above_node });
            if let Some(r) = right {
                let rn = self.new_node(NodeKind::Trap(r));
                top = self.new_node(NodeKind::X { point: q, left: top, right: rn });
            }
            if let Some(l) = left {
                let ln = self.new_node(NodeKind::Trap(l));
                top = self.new_node(NodeKind::X { point: p, left: ln, right: top });
            }
            let old_node = o.node;
            if old_node == self.tree {
                self.tree = top;
            } else {
                self.replace_with(old_node, top);
            }
            if !end_trap {
                left_old = Some(old);
                left_above = Some(above);
                left_below = Some(below);
            }
        }
        true
    }

    /// `initialize()`: build the trapezoid map of a triangulation (which
    /// must have its neighbors).
    pub fn new(triang: &Triangulation) -> Result<Self, String> {
        let npoints = triang.npoints();
        let mut points: Vec<TPoint> = Vec::with_capacity(npoints + 4);
        let mut empty = true;
        let (mut lower, mut upper) = (XY::default(), XY::default());
        for i in 0..npoints {
            let mut xy = triang.point(i as i32);
            if xy.x == -0.0 {
                xy.x = 0.0;
            }
            if xy.y == -0.0 {
                xy.y = 0.0;
            }
            points.push(TPoint { xy, tri: -1 });
            if empty {
                empty = false;
                lower = xy;
                upper = xy;
            } else {
                if xy.x < lower.x {
                    lower.x = xy.x;
                } else if xy.x > upper.x {
                    upper.x = xy.x;
                }
                if xy.y < lower.y {
                    lower.y = xy.y;
                } else if xy.y > upper.y {
                    upper.y = xy.y;
                }
            }
        }
        if empty {
            lower = XY::new(0.0, 0.0);
            upper = XY::new(1.0, 1.0);
        } else {
            let delta = upper.sub(&lower).mul(0.1);
            lower = lower.sub(&delta);
            upper = upper.add(&delta);
        }
        points.push(TPoint { xy: lower, tri: -1 });
        points.push(TPoint { xy: XY::new(upper.x, lower.y), tri: -1 });
        points.push(TPoint { xy: XY::new(lower.x, upper.y), tri: -1 });
        points.push(TPoint { xy: upper, tri: -1 });
        let np = npoints;
        let mut edges = vec![
            TEdge { left: np, right: np + 1, triangle_below: -1, triangle_above: -1, point_below: None, point_above: None },
            TEdge { left: np + 2, right: np + 3, triangle_below: -1, triangle_above: -1, point_below: None, point_above: None },
        ];
        for tri in 0..triang.ntri() {
            if triang.is_masked(tri) {
                continue;
            }
            let t = tri as i32;
            for edge in 0..3 {
                let start = triang.tri_point(t, edge) as usize;
                let end = triang.tri_point(t, (edge + 1) % 3) as usize;
                let other = triang.tri_point(t, (edge + 2) % 3) as usize;
                let neighbor = triang.neighbor_edge(t, edge);
                if points[end].xy.is_right_of(&points[start].xy) {
                    let nb = if neighbor.0 == -1 {
                        None
                    } else {
                        Some(triang.tri_point(neighbor.0, (neighbor.1 + 2) % 3) as usize)
                    };
                    edges.push(TEdge {
                        left: start,
                        right: end,
                        triangle_below: neighbor.0,
                        triangle_above: t,
                        point_below: nb,
                        point_above: Some(other),
                    });
                } else if neighbor.0 == -1 {
                    edges.push(TEdge {
                        left: end,
                        right: start,
                        triangle_below: t,
                        triangle_above: -1,
                        point_below: Some(other),
                        point_above: None,
                    });
                }
                if points[start].tri == -1 {
                    points[start].tri = t;
                }
            }
        }
        let mut f = TrapezoidMapTriFinder { points, edges, traps: Vec::new(), nodes: Vec::new(), tree: 0 };
        let t0 = f.new_trap(np, np + 1, 0, 1);
        f.tree = f.new_node(NodeKind::Trap(t0));
        let mut rng = Mt19937::new(1234);
        std_shuffle(&mut f.edges[2..], &mut rng);
        for index in 2..f.edges.len() {
            if !f.add_edge_to_tree(index) {
                return Err("Triangulation is invalid".into());
            }
        }
        Ok(f)
    }

    pub fn find_one(&self, xy: XY) -> i32 {
        let n = self.search_xy(&xy);
        self.node_tri(n)
    }

    /// `get_tree_stats()`: node count, unique nodes, trapezoid count, unique
    /// trapezoid nodes, max parent count, max depth, mean trapezoid depth.
    pub fn tree_stats(&self) -> (i64, i64, i64, i64, i64, i64, f64) {
        let mut node_count = 0i64;
        let mut trapezoid_count = 0i64;
        let mut max_parent_count = 0i64;
        let mut max_depth = 0i64;
        let mut sum_depth = 0.0f64;
        let mut unique = std::collections::HashSet::new();
        let mut unique_traps = std::collections::HashSet::new();
        let mut stack = vec![(self.tree, 0i64)];
        while let Some((n, depth)) = stack.pop() {
            node_count += 1;
            if depth > max_depth {
                max_depth = depth;
            }
            if unique.insert(n) {
                max_parent_count = max_parent_count.max(self.nodes[n].parents.len() as i64);
            }
            match self.nodes[n].kind {
                NodeKind::X { left, right, .. } => {
                    stack.push((right, depth + 1));
                    stack.push((left, depth + 1));
                }
                NodeKind::Y { below, above, .. } => {
                    stack.push((above, depth + 1));
                    stack.push((below, depth + 1));
                }
                NodeKind::Trap(_) => {
                    unique_traps.insert(n);
                    trapezoid_count += 1;
                    sum_depth += depth as f64;
                }
            }
        }
        (
            node_count,
            unique.len() as i64,
            trapezoid_count,
            unique_traps.len() as i64,
            max_parent_count,
            max_depth,
            sum_depth / trapezoid_count as f64,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mt19937_reference() {
        // The 10000th output of a default-seeded (5489) mt19937 is 4123659995.
        let mut g = Mt19937::new(5489);
        let mut v = 0;
        for _ in 0..10000 {
            v = g.next_u32();
        }
        assert_eq!(v, 4123659995);
    }
}

// ---------------------------------------------------------------------------
// Delaunay triangulation (used when matplotlib's qhull is not available)

/// Delaunay triangulation of the points, in matplotlib's `_qhull.delaunay`
/// conventions: anticlockwise triangles and `neighbors[i][j]` the triangle
/// across the edge from point `j` to point `j+1` of triangle `i` (-1 if none).
/// The triangles are the same as qhull's but may come in another order.
pub fn delaunay(x: &[f64], y: &[f64]) -> Result<(Vec<i32>, Vec<i32>), String> {
    let n = x.len();
    let (mut xm, mut ym) = (0.0, 0.0);
    for i in 0..n {
        xm += x[i];
        ym += y[i];
    }
    xm /= n as f64;
    ym /= n as f64;
    let pts: Vec<delaunator::Point> = (0..n).map(|i| delaunator::Point { x: x[i] - xm, y: y[i] - ym }).collect();
    let t = delaunator::triangulate(&pts);
    if t.triangles.is_empty() {
        return Err("Error in qhull Delaunay triangulation calculation: input is degenerate (exitcode=2)".into());
    }
    let mut triangles: Vec<i32> = t.triangles.iter().map(|&v| v as i32).collect();
    let mut no_neighbors: Vec<i32> = Vec::new();
    Triangulation::correct_triangles(x, y, &mut triangles, &mut no_neighbors);
    let mut tri = Triangulation::new(x.to_vec(), y.to_vec(), triangles, Vec::new(), Vec::new(), Vec::new());
    tri.calculate_neighbors();
    Ok((tri.triangles, tri.neighbors))
}
