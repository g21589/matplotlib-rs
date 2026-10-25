//! Rust implementations of matplotlib's `_path` extension functions.

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::agg::basics::{RectD, VertexSource, PATH_CMD_END_POLY, PATH_CMD_MOVE_TO, PATH_CMD_STOP};
use crate::agg::curves::ConvCurve;
use crate::agg::{ConvTransform, TransAffine};
use crate::backend::path_converters::{PathClipper, PathNanRemover, PathSimplifier, PathSnapper, Sketch, SnapMode};
use crate::backend::renderer_agg::MplPath;

/// Extract a matplotlib `Path` object.
pub fn path_from_py(obj: &Bound<'_, PyAny>) -> PyResult<MplPath> {
    let verts = obj.getattr("vertices")?;
    let arr: PyReadonlyArray2<f64> = verts.extract().or_else(|_| {
        let np = obj.py().import("numpy")?;
        np.call_method1("asarray", (verts.clone(), "float64"))?.extract()
    })?;
    let a = arr.as_array();
    let n = if a.ndim() == 2 { a.shape()[0] } else { 0 };
    let vertices: Vec<[f64; 2]> = (0..n).map(|i| [a[[i, 0]], a[[i, 1]]]).collect();
    let codes_obj = obj.getattr("codes")?;
    let codes = if codes_obj.is_none() {
        None
    } else {
        let c: Vec<u8> = codes_obj.call_method1("astype", ("uint8",))?.call_method0("tolist")?.extract()?;
        Some(c)
    };
    let should_simplify = obj.getattr("should_simplify").and_then(|v| v.extract()).unwrap_or(false);
    let simplify_threshold = obj.getattr("simplify_threshold").and_then(|v| v.extract()).unwrap_or(1.0 / 9.0);
    Ok(MplPath { vertices, codes, should_simplify, simplify_threshold })
}

pub fn trans_from_py(obj: &Bound<'_, PyAny>) -> PyResult<TransAffine> {
    if obj.is_none() {
        return Ok(TransAffine::new(1.0, 0.0, 0.0, 1.0, 0.0, 0.0));
    }
    let m = if obj.hasattr("get_matrix")? { obj.call_method0("get_matrix")? } else { obj.clone() };
    let v: Vec<Vec<f64>> = m.call_method0("tolist").and_then(|l| l.extract()).or_else(|_| m.extract())?;
    if v.len() != 3 || v[0].len() != 3 {
        return Err(PyValueError::new_err("Invalid affine transformation matrix"));
    }
    Ok(TransAffine { sx: v[0][0], shx: v[0][1], tx: v[0][2], shy: v[1][0], sy: v[1][1], ty: v[1][2] })
}

/// `affine_transform(points, trans)`: a*x + c*y + e (no FMA).
#[pyfunction]
fn affine_transform<'py>(py: Python<'py>, points: &Bound<'py, PyAny>, trans: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let t = trans_from_py(trans)?;
    let np = py.import("numpy")?;
    let arr = np.call_method1("asarray", (points, "float64"))?;
    let ndim: usize = arr.getattr("ndim")?.extract()?;
    if ndim == 1 {
        let v: Vec<f64> = arr.extract()?;
        if v.len() != 2 {
            return Err(PyValueError::new_err("Invalid vertices array"));
        }
        let (x, y) = (v[0], v[1]);
        let out = vec![t.sx * x + t.shx * y + t.tx, t.shy * x + t.sy * y + t.ty];
        return Ok(PyArray1::from_vec(py, out).into_any());
    }
    let a: PyReadonlyArray2<f64> = np.call_method1("ascontiguousarray", (arr,))?.extract()?;
    let s = a.as_slice()?;
    let n = a.shape()[0];
    if n > 0 && a.shape()[1] != 2 {
        return Err(PyValueError::new_err("Invalid vertices array"));
    }
    let mut out = Vec::with_capacity(n * 2);
    for i in 0..n {
        let (x, y) = (s[2 * i], s[2 * i + 1]);
        out.push(t.sx * x + t.shx * y + t.tx);
        out.push(t.shy * x + t.sy * y + t.ty);
    }
    Ok(PyArray1::from_vec(py, out).reshape([n, 2])?.into_any())
}

#[pyfunction]
fn count_bboxes_overlapping_bbox(bbox: &Bound<'_, PyAny>, bboxes: &Bound<'_, PyAny>) -> PyResult<usize> {
    let b: Vec<Vec<f64>> = bbox.call_method0("tolist").and_then(|l| l.extract()).or_else(|_| bbox.extract())?;
    let (mut ax0, mut ay0, mut ax1, mut ay1) = (b[0][0], b[0][1], b[1][0], b[1][1]);
    if ax1 < ax0 {
        std::mem::swap(&mut ax0, &mut ax1);
    }
    if ay1 < ay0 {
        std::mem::swap(&mut ay0, &mut ay1);
    }
    let bs: Vec<Vec<Vec<f64>>> = bboxes.call_method0("tolist").and_then(|l| l.extract()).or_else(|_| bboxes.extract())?;
    let mut count = 0;
    for bb in bs {
        let (mut bx0, mut by0, mut bx1, mut by1) = (bb[0][0], bb[0][1], bb[1][0], bb[1][1]);
        if bx1 < bx0 {
            std::mem::swap(&mut bx0, &mut bx1);
        }
        if by1 < by0 {
            std::mem::swap(&mut by0, &mut by1);
        }
        if !((bx1 <= ax0) || (by1 <= ay0) || (bx0 >= ax1) || (by0 >= ay1)) {
            count += 1;
        }
    }
    Ok(count)
}

#[pyfunction]
fn is_sorted_and_has_non_nan(array: &Bound<'_, PyAny>) -> PyResult<bool> {
    let v: Vec<f64> = array.call_method0("tolist").and_then(|l| l.extract()).or_else(|_| array.extract())?;
    Ok(crate::mpl::lines::is_sorted_and_has_non_nan(&v))
}

fn snap_mode_from(obj: &Bound<'_, PyAny>) -> PyResult<SnapMode> {
    if obj.is_none() {
        return Ok(SnapMode::Auto);
    }
    Ok(if obj.extract::<bool>()? { SnapMode::True } else { SnapMode::False })
}

fn collect<S: VertexSource>(mut src: S) -> (Vec<f64>, Vec<u8>) {
    let mut v = Vec::new();
    let mut c = Vec::new();
    src.rewind(0);
    let (mut x, mut y) = (0.0, 0.0);
    loop {
        let code = src.vertex(&mut x, &mut y);
        if code == PATH_CMD_STOP {
            break;
        }
        v.push(x);
        v.push(y);
        c.push(code as u8);
    }
    v.push(0.0);
    v.push(0.0);
    c.push(0);
    (v, c)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn cleanup_path<'py>(
    py: Python<'py>,
    path: &Bound<'py, PyAny>,
    trans: &Bound<'py, PyAny>,
    remove_nans: bool,
    clip_rect: &Bound<'py, PyAny>,
    snap_mode: &Bound<'py, PyAny>,
    stroke_width: f64,
    simplify: Option<bool>,
    return_curves: bool,
    sketch: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let p = path_from_py(path)?;
    let t = trans_from_py(trans)?;
    let rect = if clip_rect.is_none() {
        RectD::new(0.0, 0.0, 0.0, 0.0)
    } else {
        let r: Vec<f64> = clip_rect.call_method0("tolist").and_then(|l| l.extract()).or_else(|_| {
            let v: Vec<Vec<f64>> = clip_rect.extract()?;
            Ok::<Vec<f64>, PyErr>(v.into_iter().flatten().collect())
        })?;
        let r: Vec<f64> = if r.len() == 4 { r } else { vec![0.0; 4] };
        RectD::new(r[0], r[1], r[2], r[3])
    };
    let do_clip = rect.x1 < rect.x2 && rect.y1 < rect.y2;
    let sm = snap_mode_from(snap_mode)?;
    let simplify = simplify.unwrap_or(p.should_simplify);
    let sk: Option<(f64, f64, f64)> = if sketch.is_none() { None } else { Some(sketch.extract()?) };
    let tp = ConvTransform::new(p.iter(), t);
    let nr = PathNanRemover::new(tp, remove_nans, p.has_codes());
    let cl = PathClipper::from_rect(nr, do_clip, rect);
    let sn = PathSnapper::new(cl, sm, p.total_vertices(), stroke_width);
    let si = PathSimplifier::new(sn, simplify, p.simplify_threshold);
    let (v, c) = if return_curves && sk.map(|s| s.0 == 0.0).unwrap_or(true) {
        collect(si)
    } else {
        let curve = ConvCurve::new(si);
        let (s, l, r) = sk.unwrap_or((0.0, 0.0, 0.0));
        collect(Sketch::new(curve, s, l, r))
    };
    let n = c.len();
    let va = PyArray1::from_vec(py, v).reshape([n, 2])?.into_any();
    let ca = PyArray1::from_vec(py, c).into_any();
    Ok((va, ca))
}

#[pyfunction]
#[pyo3(signature = (path, trans, width=0.0, height=0.0, closed_only=false))]
fn convert_path_to_polygons<'py>(
    py: Python<'py>,
    path: &Bound<'py, PyAny>,
    trans: &Bound<'py, PyAny>,
    width: f64,
    height: f64,
    closed_only: bool,
) -> PyResult<Bound<'py, PyList>> {
    let p = path_from_py(path)?;
    let t = trans_from_py(trans)?;
    let do_clip = width != 0.0 && height != 0.0;
    let tp = ConvTransform::new(p.iter(), t);
    let nr = PathNanRemover::new(tp, true, p.has_codes());
    let cl = PathClipper::new(nr, do_clip, width, height);
    let si = PathSimplifier::new(cl, p.should_simplify, p.simplify_threshold);
    let mut curve = ConvCurve::new(si);
    let mut result: Vec<Vec<[f64; 2]>> = vec![Vec::new()];
    fn finalize(result: &mut Vec<Vec<[f64; 2]>>, closed_only: bool) {
        if result.is_empty() {
            return;
        }
        let last = result.last_mut().unwrap();
        if last.is_empty() {
            result.pop();
        } else if closed_only {
            if last.len() < 3 {
                result.pop();
            } else if last[0] != last[last.len() - 1] {
                let f = last[0];
                last.push(f);
            }
        }
    }
    curve.rewind(0);
    let (mut x, mut y) = (0.0, 0.0);
    loop {
        let code = curve.vertex(&mut x, &mut y);
        if code == PATH_CMD_STOP {
            break;
        }
        if (code & PATH_CMD_END_POLY) == PATH_CMD_END_POLY {
            finalize(&mut result, true);
            result.push(Vec::new());
        } else {
            if code == PATH_CMD_MOVE_TO {
                finalize(&mut result, closed_only);
                result.push(Vec::new());
            }
            result.last_mut().unwrap().push([x, y]);
        }
    }
    finalize(&mut result, closed_only);
    let out = PyList::empty(py);
    for poly in result {
        let n = poly.len();
        let flat: Vec<f64> = poly.into_iter().flat_map(|p| [p[0], p[1]]).collect();
        out.append(PyArray1::from_vec(py, flat).reshape([n, 2])?)?;
    }
    Ok(out)
}

#[pyfunction]
fn get_path_collection_extents<'py>(
    py: Python<'py>,
    master_transform: &Bound<'py, PyAny>,
    paths: &Bound<'py, PyAny>,
    transforms: &Bound<'py, PyAny>,
    offsets: &Bound<'py, PyAny>,
    offset_transform: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>)> {
    let master = trans_from_py(master_transform)?;
    let paths: Vec<MplPath> = paths.try_iter()?.map(|p| path_from_py(&p?)).collect::<PyResult<_>>()?;
    let np = py.import("numpy")?;
    let tr: Vec<Vec<Vec<f64>>> = np.call_method1("asarray", (transforms, "float64"))?.call_method0("tolist")?.extract().unwrap_or_default();
    let offs: Vec<Vec<f64>> = np
        .call_method1("asarray", (offsets, "float64"))?
        .call_method1("reshape", ((-1i64, 2i64),))?
        .call_method0("tolist")?
        .extract()?;
    let otrans = trans_from_py(offset_transform)?;
    let npaths = paths.len();
    let noffsets = offs.len();
    let n = npaths.max(noffsets);
    let ntrans = tr.len().min(n);
    let mut e = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::INFINITY];
    if npaths > 0 {
        for i in 0..n {
            let path = &paths[i % npaths];
            let mut trans = if ntrans > 0 {
                let s = &tr[i % ntrans];
                TransAffine::new(s[0][0], s[1][0], s[0][1], s[1][1], s[0][2], s[1][2])
            } else {
                master
            };
            if noffsets > 0 {
                let (mut xo, mut yo) = (offs[i % noffsets][0], offs[i % noffsets][1]);
                otrans.transform(&mut xo, &mut yo);
                trans.multiply(&TransAffine::translation(xo, yo));
            }
            let tp = ConvTransform::new(path.iter(), trans);
            let mut nr = PathNanRemover::new(tp, true, path.has_codes());
            nr.rewind(0);
            let (mut x, mut y) = (0.0, 0.0);
            loop {
                let code = nr.vertex(&mut x, &mut y);
                if code == PATH_CMD_STOP {
                    break;
                }
                if (code & PATH_CMD_END_POLY) == PATH_CMD_END_POLY {
                    continue;
                }
                if x < e[0] {
                    e[0] = x;
                }
                if y < e[1] {
                    e[1] = y;
                }
                if x > e[2] {
                    e[2] = x;
                }
                if y > e[3] {
                    e[3] = y;
                }
                if x > 0.0 && x < e[4] {
                    e[4] = x;
                }
                if y > 0.0 && y < e[5] {
                    e[5] = y;
                }
            }
        }
    }
    let ext = PyArray1::from_vec(py, vec![e[0], e[1], e[2], e[3]]).reshape([2, 2])?;
    let minpos = PyArray1::from_vec(py, vec![e[4], e[5]]);
    Ok((ext, minpos))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let sub = PyModule::new(m.py(), "_path")?;
    sub.add_function(wrap_pyfunction!(affine_transform, &sub)?)?;
    sub.add_function(wrap_pyfunction!(count_bboxes_overlapping_bbox, &sub)?)?;
    sub.add_function(wrap_pyfunction!(is_sorted_and_has_non_nan, &sub)?)?;
    sub.add_function(wrap_pyfunction!(cleanup_path, &sub)?)?;
    sub.add_function(wrap_pyfunction!(convert_path_to_polygons, &sub)?)?;
    sub.add_function(wrap_pyfunction!(get_path_collection_extents, &sub)?)?;
    m.add_submodule(&sub)?;
    Ok(())
}
