//! Geometric queries on paths: port of matplotlib's `_path.h`
//! (`point_in_path`, `path_intersects_path`, `clip_path_to_rect`, ...).

use crate::agg::basics::{is_stop, VertexSource, PATH_CMD_END_POLY, PATH_CMD_MOVE_TO, PATH_CMD_STOP};
use crate::agg::curves::ConvCurve;
use crate::agg::stroke::{conv_stroke, ConvAdaptorVcgen, VcgenContour};
use crate::agg::{ConvTransform, TransAffine};
use crate::backend::path_converters::PathNanRemover;
use crate::backend::renderer_agg::MplPath;

fn is_end(code: u32) -> bool {
    code == PATH_CMD_STOP || (code & PATH_CMD_END_POLY) == PATH_CMD_END_POLY
}

/// `point_in_path_impl`: crossing-number test of `points` against all
/// subpaths of `path`; ORs the result into `inside`.
pub fn point_in_path_impl<S: VertexSource>(points: &[[f64; 2]], path: &mut S, inside: &mut [bool]) {
    let n = points.len();
    let mut yflag0 = vec![false; n];
    let mut subpath_flag = vec![false; n];
    for v in inside.iter_mut() {
        *v = false;
    }
    path.rewind(0);
    let (mut x, mut y) = (0.0, 0.0);
    let mut code: u32 = 0;
    loop {
        if code != PATH_CMD_MOVE_TO {
            code = path.vertex(&mut x, &mut y);
            if is_end(code) {
                if code == PATH_CMD_STOP {
                    break;
                }
                continue;
            }
        }
        let (sx, sy) = (x, y);
        let (mut vtx0, mut vty0) = (x, y);
        let (mut vtx1, mut vty1) = (x, y);
        for i in 0..n {
            let ty = points[i][1];
            if ty.is_finite() {
                yflag0[i] = vty0 >= ty;
                subpath_flag[i] = false;
            }
        }
        loop {
            code = path.vertex(&mut x, &mut y);
            if is_end(code) {
                x = sx;
                y = sy;
            } else if code == PATH_CMD_MOVE_TO {
                break;
            }
            for i in 0..n {
                let (tx, ty) = (points[i][0], points[i][1]);
                if !(tx.is_finite() && ty.is_finite()) {
                    continue;
                }
                let yflag1 = vty1 >= ty;
                if yflag0[i] != yflag1 && (((vty1 - ty) * (vtx0 - vtx1) >= (vtx1 - tx) * (vty0 - vty1)) == yflag1) {
                    subpath_flag[i] ^= true;
                }
                yflag0[i] = yflag1;
            }
            vtx0 = vtx1;
            vty0 = vty1;
            vtx1 = x;
            vty1 = y;
            if is_end(code) {
                break;
            }
        }
        let mut all_done = true;
        for i in 0..n {
            let (tx, ty) = (points[i][0], points[i][1]);
            if !(tx.is_finite() && ty.is_finite()) {
                continue;
            }
            let yflag1 = vty1 >= ty;
            if yflag0[i] != yflag1 && (((vty1 - ty) * (vtx0 - vtx1) >= (vtx1 - tx) * (vty0 - vty1)) == yflag1) {
                subpath_flag[i] ^= true;
            }
            inside[i] |= subpath_flag[i];
            if !inside[i] {
                all_done = false;
            }
        }
        if all_done || code == PATH_CMD_STOP {
            break;
        }
    }
}

/// `points_in_path(points, r, path, trans)`.
pub fn points_in_path(points: &[[f64; 2]], r: f64, path: &MplPath, trans: &TransAffine) -> Vec<bool> {
    let mut result = vec![false; points.len()];
    if path.total_vertices() < 3 {
        return result;
    }
    let tp = ConvTransform::new(path.iter(), *trans);
    let nn = PathNanRemover::new(tp, true, path.has_codes());
    let mut curved = ConvCurve::new(nn);
    if r != 0.0 {
        let mut gen = VcgenContour::default();
        gen.width(r);
        let mut contoured = ConvAdaptorVcgen::new(curved, gen);
        point_in_path_impl(points, &mut contoured, &mut result);
    } else {
        point_in_path_impl(points, &mut curved, &mut result);
    }
    result
}

pub fn point_in_path(x: f64, y: f64, r: f64, path: &MplPath, trans: &TransAffine) -> bool {
    points_in_path(&[[x, y]], r, path, trans)[0]
}

/// `point_on_path`: the point is within `r` of the path's outline.
pub fn point_on_path(x: f64, y: f64, r: f64, path: &MplPath, trans: &TransAffine) -> bool {
    let tp = ConvTransform::new(path.iter(), *trans);
    let nn = PathNanRemover::new(tp, true, path.has_codes());
    let curved = ConvCurve::new(nn);
    let mut stroked = conv_stroke(curved);
    stroked.generator.width(r * 2.0);
    let mut result = [false];
    point_in_path_impl(&[[x, y]], &mut stroked, &mut result);
    result[0]
}

/// `path_in_path(a, atrans, b, btrans)`: all vertices of b inside a.
pub fn path_in_path(a: &MplPath, atrans: &TransAffine, b: &MplPath, btrans: &TransAffine) -> bool {
    if a.total_vertices() < 3 {
        return false;
    }
    let tp = ConvTransform::new(b.iter(), *btrans);
    let nn = PathNanRemover::new(tp, true, b.has_codes());
    let mut curved = ConvCurve::new(nn);
    curved.rewind(0);
    let (mut x, mut y) = (0.0, 0.0);
    while curved.vertex(&mut x, &mut y) != PATH_CMD_STOP {
        if !point_in_path(x, y, 0.0, a, atrans) {
            return false;
        }
    }
    true
}

/// `point_in_path_collection`: indices of the collection paths that
/// contain (filled) or pass near (not filled) the point.
#[allow(clippy::too_many_arguments)]
pub fn point_in_path_collection(
    x: f64,
    y: f64,
    radius: f64,
    master: &TransAffine,
    paths: &[MplPath],
    transforms: &[TransAffine],
    offsets: &[[f64; 2]],
    offset_trans: &TransAffine,
    filled: bool,
) -> Vec<usize> {
    let mut result = Vec::new();
    let npaths = paths.len();
    if npaths == 0 {
        return result;
    }
    let noffsets = offsets.len();
    let n = npaths.max(noffsets);
    let ntransforms = transforms.len().min(n);
    for i in 0..n {
        let path = &paths[i % npaths];
        let mut trans = if ntransforms > 0 {
            let mut t = transforms[i % ntransforms];
            t.multiply(master);
            t
        } else {
            *master
        };
        if noffsets > 0 {
            let (mut xo, mut yo) = (offsets[i % noffsets][0], offsets[i % noffsets][1]);
            offset_trans.transform(&mut xo, &mut yo);
            trans.multiply(&TransAffine::translation(xo, yo));
        }
        let hit = if filled { point_in_path(x, y, radius, path, &trans) } else { point_on_path(x, y, radius, path, &trans) };
        if hit {
            result.push(i);
        }
    }
    result
}

fn isclose(a: f64, b: f64) -> bool {
    let rtol = 1e-10;
    let atol = 1e-13;
    (a - b).abs() <= f64::max(rtol * f64::max(a.abs(), b.abs()), atol)
}

#[allow(clippy::too_many_arguments)]
fn segments_intersect(x1: f64, y1: f64, x2: f64, y2: f64, x3: f64, y3: f64, x4: f64, y4: f64) -> bool {
    let den = ((y4 - y3) * (x2 - x1)) - ((x4 - x3) * (y2 - y1));
    if isclose(den, 0.0) {
        let t_area = (x2 * y3 - x3 * y2) - x1 * (y3 - y2) + y1 * (x3 - x2);
        if isclose(t_area, 0.0) {
            if x1 == x2 && x2 == x3 {
                return (y1.min(y2) <= y3.min(y4) && y3.min(y4) <= y1.max(y2))
                    || (y3.min(y4) <= y1.min(y2) && y1.min(y2) <= y3.max(y4));
            }
            return (x1.min(x2) <= x3.min(x4) && x3.min(x4) <= x1.max(x2))
                || (x3.min(x4) <= x1.min(x2) && x1.min(x2) <= x3.max(x4));
        }
        return false;
    }
    let n1 = ((x4 - x3) * (y1 - y3)) - ((y4 - y3) * (x1 - x3));
    let n2 = ((x2 - x1) * (y1 - y3)) - ((y2 - y1) * (x1 - x3));
    let u1 = n1 / den;
    let u2 = n2 / den;
    (u1 > 0.0 || isclose(u1, 0.0))
        && (u1 < 1.0 || isclose(u1, 1.0))
        && (u2 > 0.0 || isclose(u2, 0.0))
        && (u2 < 1.0 || isclose(u2, 1.0))
}

/// `path_intersects_path(p1, p2)` (outlines only).
pub fn path_intersects_path(p1: &MplPath, p2: &MplPath) -> bool {
    if p1.total_vertices() < 2 || p2.total_vertices() < 2 {
        return false;
    }
    let mut c1 = ConvCurve::new(PathNanRemover::new(p1.iter(), true, p1.has_codes()));
    let mut c2 = ConvCurve::new(PathNanRemover::new(p2.iter(), true, p2.has_codes()));
    let (mut x11, mut y11, mut x12, mut y12) = (0.0, 0.0, 0.0, 0.0);
    let (mut x21, mut y21, mut x22, mut y22) = (0.0, 0.0, 0.0, 0.0);
    c1.rewind(0);
    c1.vertex(&mut x11, &mut y11);
    while c1.vertex(&mut x12, &mut y12) != PATH_CMD_STOP {
        if isclose((x11 - x12) * (x11 - x12) + (y11 - y12) * (y11 - y12), 0.0) {
            continue;
        }
        c2.rewind(0);
        c2.vertex(&mut x21, &mut y21);
        while c2.vertex(&mut x22, &mut y22) != PATH_CMD_STOP {
            if isclose((x21 - x22) * (x21 - x22) + (y21 - y22) * (y21 - y22), 0.0) {
                continue;
            }
            if segments_intersect(x11, y11, x12, y12, x21, y21, x22, y22) {
                return true;
            }
            x21 = x22;
            y21 = y22;
        }
        x11 = x12;
        y11 = y12;
    }
    false
}

#[allow(clippy::too_many_arguments)]
fn segment_intersects_rectangle(x1: f64, y1: f64, x2: f64, y2: f64, cx: f64, cy: f64, w: f64, h: f64) -> bool {
    (x1 + x2 - 2.0 * cx).abs() < (x1 - x2).abs() + w
        && (y1 + y2 - 2.0 * cy).abs() < (y1 - y2).abs() + h
        && 2.0 * ((x1 - cx) * (y1 - y2) - (y1 - cy) * (x1 - x2)).abs() < w * (y1 - y2).abs() + h * (x1 - x2).abs()
}

/// `path_intersects_rectangle(path, x1, y1, x2, y2, filled)`.
pub fn path_intersects_rectangle(path: &MplPath, rx1: f64, ry1: f64, rx2: f64, ry2: f64, filled: bool) -> bool {
    if path.total_vertices() == 0 {
        return false;
    }
    let mut curve = ConvCurve::new(PathNanRemover::new(path.iter(), true, path.has_codes()));
    let cx = (rx1 + rx2) * 0.5;
    let cy = (ry1 + ry2) * 0.5;
    let w = (rx1 - rx2).abs();
    let h = (ry1 - ry2).abs();
    let (mut x1, mut y1, mut x2, mut y2) = (0.0, 0.0, 0.0, 0.0);
    curve.rewind(0);
    curve.vertex(&mut x1, &mut y1);
    if 2.0 * (x1 - cx).abs() <= w && 2.0 * (y1 - cy).abs() <= h {
        return true;
    }
    while curve.vertex(&mut x2, &mut y2) != PATH_CMD_STOP {
        if segment_intersects_rectangle(x1, y1, x2, y2, cx, cy, w, h) {
            return true;
        }
        x1 = x2;
        y1 = y2;
    }
    if filled && point_in_path(cx, cy, 0.0, path, &TransAffine::new(1.0, 0.0, 0.0, 1.0, 0.0, 0.0)) {
        return true;
    }
    false
}

fn clip_one_step(polygon: &[[f64; 2]], result: &mut Vec<[f64; 2]>, inside: impl Fn(f64, f64) -> bool, bisect: impl Fn(f64, f64, f64, f64) -> [f64; 2]) {
    result.clear();
    let Some(last) = polygon.last() else { return };
    let (mut sx, mut sy) = (last[0], last[1]);
    for p in polygon {
        let (px, py) = (p[0], p[1]);
        let sinside = inside(sx, sy);
        let pinside = inside(px, py);
        if sinside ^ pinside {
            result.push(bisect(sx, sy, px, py));
        }
        if pinside {
            result.push([px, py]);
        }
        sx = px;
        sy = py;
    }
}

fn finalize_polygon(results: &mut Vec<Vec<[f64; 2]>>) {
    let Some(p) = results.last_mut() else { return };
    if p.is_empty() || p.len() < 3 {
        results.pop();
    } else if p.first() != p.last() {
        let f = p[0];
        p.push(f);
    }
}

/// `clip_path_to_rect(path, rect, inside)`: Sutherland-Hodgman clipping of
/// each subpath; returns closed polygons.
pub fn clip_path_to_rect(path: &MplPath, rect: (f64, f64, f64, f64), inside: bool) -> Vec<Vec<[f64; 2]>> {
    let (mut xmin, mut xmax) = if rect.0 < rect.2 { (rect.0, rect.2) } else { (rect.2, rect.0) };
    let (mut ymin, mut ymax) = if rect.1 < rect.3 { (rect.1, rect.3) } else { (rect.3, rect.1) };
    if !inside {
        std::mem::swap(&mut xmin, &mut xmax);
        std::mem::swap(&mut ymin, &mut ymax);
    }
    let mut curve = ConvCurve::new(path.iter());
    let mut results: Vec<Vec<[f64; 2]>> = Vec::new();
    let mut p1: Vec<[f64; 2]> = Vec::new();
    let mut p2: Vec<[f64; 2]> = Vec::new();
    let (mut x, mut y) = (0.0, 0.0);
    let mut code: u32 = 0;
    curve.rewind(0);
    loop {
        p1.clear();
        loop {
            if code == PATH_CMD_MOVE_TO {
                p1.push([x, y]);
            }
            code = curve.vertex(&mut x, &mut y);
            if code == PATH_CMD_STOP {
                break;
            }
            if code != PATH_CMD_MOVE_TO {
                p1.push([x, y]);
            }
            if (code & PATH_CMD_END_POLY) == PATH_CMD_END_POLY {
                break;
            }
        }
        let bx = |m: f64| move |sx: f64, sy: f64, px: f64, py: f64| [m, sy + (py - sy) * ((m - sx) / (px - sx))];
        let by = |m: f64| move |sx: f64, sy: f64, px: f64, py: f64| [sx + (px - sx) * ((m - sy) / (py - sy)), m];
        clip_one_step(&p1, &mut p2, |x, _| x <= xmax, bx(xmax));
        clip_one_step(&p2, &mut p1, |x, _| x >= xmin, bx(xmin));
        clip_one_step(&p1, &mut p2, |_, y| y <= ymax, by(ymax));
        clip_one_step(&p2, &mut p1, |_, y| y >= ymin, by(ymin));
        if !p1.is_empty() {
            finalize_polygon(&mut results);
            results.push(p1.clone());
        }
        if is_stop(code) {
            break;
        }
    }
    finalize_polygon(&mut results);
    results
}
