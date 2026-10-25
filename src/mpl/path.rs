//! `matplotlib.path.Path` constructors and helpers.

use std::f64::consts::PI;

use super::transforms::{Bbox, Transform};
pub use crate::backend::renderer_agg::MplPath as Path;

pub const STOP: u8 = 0;
pub const MOVETO: u8 = 1;
pub const LINETO: u8 = 2;
pub const CURVE3: u8 = 3;
pub const CURVE4: u8 = 4;
pub const CLOSEPOLY: u8 = 79;

/// Default `path.simplify_threshold` / `path.simplify` (overridden from rc).
#[derive(Clone, Copy, Debug)]
pub struct SimplifyParams {
    pub simplify: bool,
    pub threshold: f64,
}

impl Default for SimplifyParams {
    fn default() -> Self {
        SimplifyParams { simplify: true, threshold: 1.0 / 9.0 }
    }
}

/// `Path(vertices, codes)` with `_update_values` semantics.
pub fn new_path(vertices: Vec<[f64; 2]>, codes: Option<Vec<u8>>, sp: SimplifyParams) -> Path {
    let should_simplify = sp.threshold > 0.0
        && sp.simplify
        && vertices.len() >= 128
        && codes.as_ref().map(|c| c.iter().all(|&v| v <= LINETO)).unwrap_or(true);
    Path { vertices, codes, should_simplify, simplify_threshold: sp.threshold }
}

pub fn closed_path(vertices: Vec<[f64; 2]>, sp: SimplifyParams) -> Path {
    let n = vertices.len();
    let codes = if n > 0 {
        let mut c = vec![LINETO; n];
        c[0] = MOVETO;
        c[n - 1] = CLOSEPOLY;
        Some(c)
    } else {
        None
    };
    new_path(vertices, codes, sp)
}

pub fn unit_rectangle() -> Path {
    closed_path(vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]], SimplifyParams::default())
}

pub fn unit_regular_polygon(n: usize) -> Path {
    let step = 2.0 * PI / n as f64;
    let verts: Vec<[f64; 2]> = (0..=n)
        .map(|k| {
            let theta = step * k as f64 + PI / 2.0;
            [theta.cos(), theta.sin()]
        })
        .collect();
    closed_path(verts, SimplifyParams::default())
}

pub fn unit_regular_star(n: usize, inner_circle: f64) -> Path {
    let ns2 = n * 2;
    let step = 2.0 * PI / ns2 as f64;
    let verts: Vec<[f64; 2]> = (0..=ns2)
        .map(|k| {
            let mut theta = step * k as f64;
            theta += PI / 2.0;
            let r = if k % 2 == 1 { inner_circle } else { 1.0 };
            [r * theta.cos(), r * theta.sin()]
        })
        .collect();
    closed_path(verts, SimplifyParams::default())
}

pub fn unit_regular_asterisk(n: usize) -> Path {
    unit_regular_star(n, 0.0)
}

const MAGIC: f64 = 0.2652031;

fn circle_vertices() -> Vec<[f64; 2]> {
    let sqrthalf = 0.5f64.sqrt();
    let magic45 = sqrthalf * MAGIC;
    vec![
        [0.0, -1.0],
        [MAGIC, -1.0],
        [sqrthalf - magic45, -sqrthalf - magic45],
        [sqrthalf, -sqrthalf],
        [sqrthalf + magic45, -sqrthalf + magic45],
        [1.0, -MAGIC],
        [1.0, 0.0],
        [1.0, MAGIC],
        [sqrthalf + magic45, sqrthalf - magic45],
        [sqrthalf, sqrthalf],
        [sqrthalf - magic45, sqrthalf + magic45],
        [MAGIC, 1.0],
        [0.0, 1.0],
        [-MAGIC, 1.0],
        [-sqrthalf + magic45, sqrthalf + magic45],
        [-sqrthalf, sqrthalf],
        [-sqrthalf - magic45, sqrthalf - magic45],
        [-1.0, MAGIC],
        [-1.0, 0.0],
        [-1.0, -MAGIC],
        [-sqrthalf - magic45, -sqrthalf + magic45],
        [-sqrthalf, -sqrthalf],
        [-sqrthalf + magic45, -sqrthalf - magic45],
        [-MAGIC, -1.0],
        [0.0, -1.0],
        [0.0, -1.0],
    ]
}

/// `Path.circle(center, radius)`.
pub fn circle(center: (f64, f64), radius: f64) -> Path {
    let verts: Vec<[f64; 2]> =
        circle_vertices().iter().map(|v| [v[0] * radius + center.0, v[1] * radius + center.1]).collect();
    let mut codes = vec![CURVE4; 26];
    codes[0] = MOVETO;
    codes[25] = CLOSEPOLY;
    new_path(verts, Some(codes), SimplifyParams::default())
}

pub fn unit_circle() -> Path {
    circle((0.0, 0.0), 1.0)
}

pub fn unit_circle_righthalf() -> Path {
    let mut v = circle_vertices();
    v.truncate(13);
    v.push([0.0, -1.0]);
    let mut codes = vec![CURVE4; 14];
    codes[0] = MOVETO;
    codes[13] = CLOSEPOLY;
    new_path(v, Some(codes), SimplifyParams::default())
}

/// numpy `rint` (round half to even).
#[inline]
pub fn rint(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 {
        2.0 * (x / 2.0).round()
    } else {
        r
    }
}

/// `Path.arc(theta1, theta2, n, is_wedge)` (degrees).
pub fn arc(theta1: f64, theta2: f64, n: Option<usize>, is_wedge: bool) -> Path {
    let halfpi = PI * 0.5;
    let eta1 = theta1;
    let n_turns = (theta2 - theta1) / 360.0;
    let nearest_turn = rint(n_turns);
    let is_full_circle = nearest_turn != 0.0 && (n_turns - nearest_turn).abs() <= 1e-12;
    let eta2 = if is_full_circle { theta1 + 360.0 } else { theta2 - 360.0 * n_turns.floor() };
    // np.deg2rad: x * (pi / 180)
    let eta1 = eta1 * (PI / 180.0);
    let eta2 = eta2 * (PI / 180.0);
    let n = n.unwrap_or_else(|| 2f64.powf(((eta2 - eta1) / halfpi).ceil()) as usize);
    let n = n.max(1);
    let deta = (eta2 - eta1) / n as f64;
    let t = (0.5 * deta).tan();
    let alpha = deta.sin() * ((4.0 + 3.0 * t * t).sqrt() - 1.0) / 3.0;
    let steps = linspace(eta1, eta2, n + 1);
    let cos_eta: Vec<f64> = steps.iter().map(|s| s.cos()).collect();
    let sin_eta: Vec<f64> = steps.iter().map(|s| s.sin()).collect();

    let (length, vertex_offset, end);
    let mut vertices;
    let mut codes;
    if is_wedge {
        length = n * 3 + 4;
        vertices = vec![[0.0, 0.0]; length];
        codes = vec![CURVE4; length];
        vertices[1] = [cos_eta[0], sin_eta[0]];
        codes[0] = MOVETO;
        codes[1] = LINETO;
        codes[length - 2] = LINETO;
        codes[length - 1] = CLOSEPOLY;
        vertex_offset = 2;
        end = length - 2;
    } else {
        length = n * 3 + 1;
        vertices = vec![[0.0, 0.0]; length];
        codes = vec![CURVE4; length];
        vertices[0] = [cos_eta[0], sin_eta[0]];
        codes[0] = MOVETO;
        vertex_offset = 1;
        end = length;
    }
    let _ = end;
    for i in 0..n {
        let xa = cos_eta[i];
        let ya = sin_eta[i];
        let xa_dot = -ya;
        let ya_dot = xa;
        let xb = cos_eta[i + 1];
        let yb = sin_eta[i + 1];
        let xb_dot = -yb;
        let yb_dot = xb;
        let o = vertex_offset + 3 * i;
        vertices[o] = [xa + alpha * xa_dot, ya + alpha * ya_dot];
        vertices[o + 1] = [xb - alpha * xb_dot, yb - alpha * yb_dot];
        vertices[o + 2] = [xb, yb];
    }
    new_path(vertices, Some(codes), SimplifyParams::default())
}

pub fn wedge(theta1: f64, theta2: f64, n: Option<usize>) -> Path {
    arc(theta1, theta2, n, true)
}

/// `np.linspace(start, stop, num, endpoint=True)`.
pub fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    if num == 0 {
        return Vec::new();
    }
    if num == 1 {
        return vec![start];
    }
    let div = (num - 1) as f64;
    let delta = stop - start;
    let step = delta / div;
    let mut out: Vec<f64> = (0..num)
        .map(|i| {
            let y = i as f64;
            if step == 0.0 {
                y / div * delta + start
            } else {
                y * step + start
            }
        })
        .collect();
    out[num - 1] = stop;
    out
}

/// Number of vertices per code (`Path.NUM_VERTICES_FOR_CODE`).
pub fn num_vertices_for_code(code: u8) -> usize {
    match code {
        STOP => 1,
        MOVETO => 1,
        LINETO => 1,
        CURVE3 => 2,
        CURVE4 => 3,
        CLOSEPOLY => 1,
        _ => 1,
    }
}

/// Real roots of a polynomial (coefficients lowest order first) in [0, 1].
fn real_roots_in_01(c: &[f64]) -> Vec<f64> {
    // trim trailing zeros (highest order)
    let mut n = c.len();
    while n > 0 && c[n - 1] == 0.0 {
        n -= 1;
    }
    let mut roots = Vec::new();
    match n {
        0 | 1 => {}
        2 => {
            roots.push(-c[0] / c[1]);
        }
        3 => {
            let (a, b, cc) = (c[2], c[1], c[0]);
            let disc = b * b - 4.0 * a * cc;
            if disc >= 0.0 {
                let s = disc.sqrt();
                roots.push((-b + s) / (2.0 * a));
                roots.push((-b - s) / (2.0 * a));
            }
        }
        _ => {}
    }
    roots.retain(|&r| (0.0..=1.0).contains(&r));
    roots.sort_by(|a, b| a.partial_cmp(b).unwrap());
    roots
}

fn bezier_eval(cp: &[[f64; 2]], t: f64) -> [f64; 2] {
    let n = cp.len() - 1;
    let binom = |n: usize, k: usize| -> f64 {
        let mut r = 1.0;
        for i in 0..k {
            r = r * (n - i) as f64 / (i + 1) as f64;
        }
        r
    };
    let mut x = 0.0;
    let mut y = 0.0;
    for (i, p) in cp.iter().enumerate() {
        let b = binom(n, i) * (1.0 - t).powi((n - i) as i32) * t.powi(i as i32);
        x += b * p[0];
        y += b * p[1];
    }
    [x, y]
}

fn bezier_extrema_points(cp: &[[f64; 2]], out: &mut Vec<[f64; 2]>) {
    let n = cp.len() - 1;
    let mut ts = vec![0.0];
    if n >= 2 {
        for d in 0..2 {
            // polynomial coefficients of the curve along dimension d
            let p: Vec<f64> = cp.iter().map(|v| v[d]).collect();
            let coeffs: Vec<f64> = match n {
                2 => vec![p[0], 2.0 * (p[1] - p[0]), p[0] - 2.0 * p[1] + p[2]],
                3 => vec![
                    p[0],
                    3.0 * (p[1] - p[0]),
                    3.0 * (p[0] - 2.0 * p[1] + p[2]),
                    -p[0] + 3.0 * p[1] - 3.0 * p[2] + p[3],
                ],
                _ => vec![],
            };
            let deriv: Vec<f64> = coeffs.iter().enumerate().skip(1).map(|(j, c)| j as f64 * c).collect();
            ts.extend(real_roots_in_01(&deriv));
        }
    }
    ts.push(1.0);
    for t in ts {
        out.push(bezier_eval(cp, t));
    }
}

/// `Path.get_extents(transform)`.
pub fn get_extents(path: &Path, transform: Option<&Transform>) -> Bbox {
    let verts: Vec<[f64; 2]> = match transform {
        Some(t) => t.transform_points(&path.vertices),
        None => path.vertices.clone(),
    };
    let xys: Vec<[f64; 2]> = match &path.codes {
        None => verts,
        Some(codes) => {
            if !codes.iter().any(|&c| c == CURVE3 || c == CURVE4) {
                verts
                    .iter()
                    .zip(codes.iter())
                    .filter(|(_, &c)| c == MOVETO || c == LINETO)
                    .map(|(v, _)| *v)
                    .collect()
            } else {
                let mut out = Vec::new();
                let mut first: Option<[f64; 2]> = None;
                let mut prev = [0.0, 0.0];
                let mut i = 0;
                while i < codes.len() {
                    let c = codes[i];
                    let nv = num_vertices_for_code(c);
                    match c {
                        MOVETO => {
                            first = Some(verts[i]);
                            out.push(verts[i]);
                            prev = verts[i];
                        }
                        LINETO => {
                            bezier_extrema_points(&[prev, verts[i]], &mut out);
                            prev = verts[i];
                        }
                        CURVE3 if i + 1 < verts.len() => {
                            bezier_extrema_points(&[prev, verts[i], verts[i + 1]], &mut out);
                            prev = verts[i + 1];
                        }
                        CURVE4 if i + 2 < verts.len() => {
                            bezier_extrema_points(&[prev, verts[i], verts[i + 1], verts[i + 2]], &mut out);
                            prev = verts[i + 2];
                        }
                        CLOSEPOLY => {
                            if let Some(f) = first {
                                bezier_extrema_points(&[prev, f], &mut out);
                                prev = verts[i];
                            }
                        }
                        STOP => break,
                        _ => {}
                    }
                    i += nv;
                }
                out
            }
        }
    };
    if xys.is_empty() {
        return Bbox::null();
    }
    let mut b = Bbox::new(f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    let mut nan = false;
    for p in &xys {
        if p[0].is_nan() || p[1].is_nan() {
            nan = true;
        }
        b.x0 = b.x0.min(p[0]);
        b.y0 = b.y0.min(p[1]);
        b.x1 = b.x1.max(p[0]);
        b.y1 = b.y1.max(p[1]);
    }
    if nan {
        // numpy min/max propagate NaN
        let xs_nan = xys.iter().any(|p| p[0].is_nan());
        let ys_nan = xys.iter().any(|p| p[1].is_nan());
        if xs_nan {
            b.x0 = f64::NAN;
            b.x1 = f64::NAN;
        }
        if ys_nan {
            b.y0 = f64::NAN;
            b.y1 = f64::NAN;
        }
    }
    b
}

/// Transform a path's vertices (all codes kept).
pub fn transform_path(path: &Path, t: &Transform) -> Path {
    Path {
        vertices: t.transform_points(&path.vertices),
        codes: path.codes.clone(),
        should_simplify: path.should_simplify,
        simplify_threshold: path.simplify_threshold,
    }
}

pub fn transform_path_non_affine(path: &Path, t: &Transform) -> Path {
    if t.is_affine() {
        return path.clone();
    }
    Path {
        vertices: t.transform_points_non_affine(&path.vertices),
        codes: path.codes.clone(),
        should_simplify: path.should_simplify,
        simplify_threshold: path.simplify_threshold,
    }
}

/// Vertices used by `Axes._update_patch_limits`: every Bézier segment
/// evaluated at 0, its axis-aligned extrema and 1.
pub fn bezier_vertices_for_limits(path: &Path, out: &mut Vec<[f64; 2]>) {
    let verts = &path.vertices;
    let n = verts.len();
    let codes: Vec<u8> = match &path.codes {
        Some(c) => c.clone(),
        None => (0..n).map(|i| if i == 0 { MOVETO } else { LINETO }).collect(),
    };
    let mut first: Option<[f64; 2]> = None;
    let mut prev = [0.0, 0.0];
    let mut i = 0;
    let finite = |p: &[f64; 2]| p[0].is_finite() && p[1].is_finite();
    while i < codes.len() {
        let c = codes[i];
        let nv = num_vertices_for_code(c).max(1);
        if i + nv > n {
            break;
        }
        let seg = &verts[i..i + nv];
        if !seg.iter().all(finite) {
            // NaN removal: the next segment restarts with a MOVETO
            i += nv;
            first = None;
            continue;
        }
        match c {
            STOP => break,
            CLOSEPOLY => {
                if let Some(f) = first {
                    bezier_extrema_points(&[prev, f], out);
                    prev = f;
                }
            }
            _ if first.is_none() || c == MOVETO => {
                let p = seg[nv - 1];
                first = Some(p);
                out.push(p);
                out.push(p);
                prev = p;
            }
            LINETO => {
                bezier_extrema_points(&[prev, seg[0]], out);
                prev = seg[0];
            }
            CURVE3 => {
                bezier_extrema_points(&[prev, seg[0], seg[1]], out);
                prev = seg[1];
            }
            CURVE4 => {
                bezier_extrema_points(&[prev, seg[0], seg[1], seg[2]], out);
                prev = seg[2];
            }
            _ => {}
        }
        i += nv;
    }
}
