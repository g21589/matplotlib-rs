//! Low-level bindings exposing the ported `RendererAgg` with an interface
//! close to matplotlib's `_backend_agg.RendererAgg`.  Used for differential
//! testing against matplotlib's C++ implementation.

use numpy::{PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods};

use crate::agg::pixfmt::Rgba;
use crate::agg::stroke::{LineCap, LineJoin};
use crate::agg::TransAffine;
use crate::backend::path_converters::SnapMode;
use crate::backend::renderer_agg::{ClipPath, Dashes, Gc, Hatch, MplPath, RendererAgg};

fn trans_from(arr: &PyReadonlyArray2<f64>) -> PyResult<TransAffine> {
    let a = arr.as_array();
    if a.shape() != [3, 3] {
        return Err(PyValueError::new_err("Invalid affine transformation matrix"));
    }
    Ok(TransAffine { sx: a[[0, 0]], shx: a[[0, 1]], tx: a[[0, 2]], shy: a[[1, 0]], sy: a[[1, 1]], ty: a[[1, 2]] })
}

fn path_from(
    vertices: &PyReadonlyArray2<f64>,
    codes: Option<&PyReadonlyArray1<u8>>,
    should_simplify: bool,
    simplify_threshold: f64,
) -> PyResult<MplPath> {
    let v = vertices.as_array();
    if v.ndim() != 2 || (v.shape()[0] > 0 && v.shape()[1] != 2) {
        return Err(PyValueError::new_err("Invalid vertices array"));
    }
    let verts: Vec<[f64; 2]> = (0..v.shape()[0]).map(|i| [v[[i, 0]], v[[i, 1]]]).collect();
    let codes = codes.map(|c| c.as_array().iter().copied().collect::<Vec<u8>>());
    if let Some(c) = &codes {
        if c.len() != verts.len() {
            return Err(PyValueError::new_err("Invalid codes array"));
        }
    }
    Ok(MplPath { vertices: verts, codes, should_simplify, simplify_threshold })
}

fn rgba_from(obj: &Bound<'_, PyAny>) -> PyResult<Rgba> {
    let t: Vec<f64> = obj.extract()?;
    match t.len() {
        4 => Ok(Rgba::new(t[0], t[1], t[2], t[3])),
        3 => Ok(Rgba::new(t[0], t[1], t[2], 1.0)),
        _ => Err(PyValueError::new_err("RGBA value must be 3- or 4-tuple")),
    }
}

fn gc_from(d: &Bound<'_, PyDict>) -> PyResult<Gc> {
    let mut gc = Gc::default();
    if let Some(v) = d.get_item("linewidth")? {
        gc.linewidth = v.extract()?;
    }
    if let Some(v) = d.get_item("alpha")? {
        gc.alpha = v.extract()?;
    }
    if let Some(v) = d.get_item("forced_alpha")? {
        gc.forced_alpha = v.extract()?;
    }
    if let Some(v) = d.get_item("rgb")? {
        gc.color = rgba_from(&v)?;
    }
    if let Some(v) = d.get_item("antialiased")? {
        gc.isaa = v.extract()?;
    }
    if let Some(v) = d.get_item("capstyle")? {
        let s: String = v.extract()?;
        gc.cap = match s.as_str() {
            "butt" => LineCap::Butt,
            "round" => LineCap::Round,
            "projecting" => LineCap::Square,
            _ => return Err(PyValueError::new_err("bad capstyle")),
        };
    }
    if let Some(v) = d.get_item("joinstyle")? {
        let s: String = v.extract()?;
        gc.join = match s.as_str() {
            "miter" => LineJoin::MiterRevert,
            "round" => LineJoin::Round,
            "bevel" => LineJoin::Bevel,
            _ => return Err(PyValueError::new_err("bad joinstyle")),
        };
    }
    if let Some(v) = d.get_item("dashes")? {
        if !v.is_none() {
            let (offset, seq): (f64, Option<Vec<f64>>) = v.extract()?;
            if let Some(seq) = seq {
                gc.dashes = Dashes::from_seq(offset, &seq);
            }
        }
    }
    if let Some(v) = d.get_item("cliprect")? {
        if !v.is_none() {
            let r: Vec<f64> = v.extract()?;
            gc.cliprect = Some([r[0], r[1], r[2], r[3]]);
        }
    }
    if let Some(v) = d.get_item("snap")? {
        gc.snap_mode = if v.is_none() {
            SnapMode::Auto
        } else if v.extract::<bool>()? {
            SnapMode::True
        } else {
            SnapMode::False
        };
    }
    if let Some(v) = d.get_item("clippath")? {
        if !v.is_none() {
            let (verts, codes, trans): (PyReadonlyArray2<f64>, Option<PyReadonlyArray1<u8>>, PyReadonlyArray2<f64>) =
                v.extract()?;
            let p = path_from(&verts, codes.as_ref(), false, 1.0 / 9.0)?;
            gc.clippath = Some(ClipPath { path: std::sync::Arc::new(p), trans: trans_from(&trans)? });
        }
    }
    if let Some(v) = d.get_item("hatch")? {
        if !v.is_none() {
            let (verts, codes, color, lw): (PyReadonlyArray2<f64>, Option<PyReadonlyArray1<u8>>, Bound<'_, PyAny>, f64) =
                v.extract()?;
            let p = path_from(&verts, codes.as_ref(), false, 1.0 / 9.0)?;
            gc.hatch = Some(Hatch { path: p, color: rgba_from(&color)?, linewidth: lw });
        }
    }
    if let Some(v) = d.get_item("sketch")? {
        if !v.is_none() {
            let s: (f64, f64, f64) = v.extract()?;
            gc.sketch = Some(s);
        }
    }
    Ok(gc)
}

fn face_from(gc: &Gc, face: Option<Bound<'_, PyAny>>) -> PyResult<Option<Rgba>> {
    match face {
        None => Ok(None),
        Some(f) if f.is_none() => Ok(None),
        Some(f) => {
            let n = f.len()?;
            let mut c = rgba_from(&f)?;
            if gc.forced_alpha || n == 3 {
                c.a = gc.alpha;
            }
            Ok(Some(c))
        }
    }
}

#[pyclass(name = "RendererAgg", module = "matplotlib_rs._mplrs", unsendable)]
pub struct PyRendererAgg {
    inner: RendererAgg,
}

#[pymethods]
impl PyRendererAgg {
    #[new]
    fn new(width: u32, height: u32, dpi: f64) -> PyResult<Self> {
        if dpi <= 0.0 {
            return Err(PyValueError::new_err("dpi must be positive"));
        }
        Ok(PyRendererAgg { inner: RendererAgg::new(width, height, dpi) })
    }

    #[pyo3(signature = (gc, vertices, codes, should_simplify, simplify_threshold, trans, face=None))]
    #[allow(clippy::too_many_arguments)]
    fn draw_path(
        &mut self,
        gc: &Bound<'_, PyDict>,
        vertices: PyReadonlyArray2<f64>,
        codes: Option<PyReadonlyArray1<u8>>,
        should_simplify: bool,
        simplify_threshold: f64,
        trans: PyReadonlyArray2<f64>,
        face: Option<Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let gc = gc_from(gc)?;
        let path = path_from(&vertices, codes.as_ref(), should_simplify, simplify_threshold)?;
        let trans = trans_from(&trans)?;
        let face = face_from(&gc, face)?;
        self.inner.draw_path(&gc, &path, &trans, face);
        Ok(())
    }

    #[pyo3(signature = (gc, marker_vertices, marker_codes, marker_trans, vertices, codes, trans, face=None))]
    #[allow(clippy::too_many_arguments)]
    fn draw_markers(
        &mut self,
        gc: &Bound<'_, PyDict>,
        marker_vertices: PyReadonlyArray2<f64>,
        marker_codes: Option<PyReadonlyArray1<u8>>,
        marker_trans: PyReadonlyArray2<f64>,
        vertices: PyReadonlyArray2<f64>,
        codes: Option<PyReadonlyArray1<u8>>,
        trans: PyReadonlyArray2<f64>,
        face: Option<Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let gc = gc_from(gc)?;
        let mpath = path_from(&marker_vertices, marker_codes.as_ref(), false, 1.0 / 9.0)?;
        let path = path_from(&vertices, codes.as_ref(), false, 1.0 / 9.0)?;
        let face = face_from(&gc, face)?;
        self.inner.draw_markers(&gc, &mpath, &trans_from(&marker_trans)?, &path, &trans_from(&trans)?, face);
        Ok(())
    }

    /// paths: list of (vertices, codes)
    #[allow(clippy::too_many_arguments)]
    fn draw_path_collection(
        &mut self,
        gc: &Bound<'_, PyDict>,
        master_transform: PyReadonlyArray2<f64>,
        paths: Vec<(PyReadonlyArray2<f64>, Option<PyReadonlyArray1<u8>>)>,
        transforms: PyReadonlyArray3<f64>,
        offsets: PyReadonlyArray2<f64>,
        offset_trans: PyReadonlyArray2<f64>,
        facecolors: PyReadonlyArray2<f64>,
        edgecolors: PyReadonlyArray2<f64>,
        linewidths: PyReadonlyArray1<f64>,
        dashes: Vec<(f64, Option<Vec<f64>>)>,
        antialiaseds: PyReadonlyArray1<u8>,
    ) -> PyResult<()> {
        let gc = gc_from(gc)?;
        let paths: Vec<MplPath> =
            paths.iter().map(|(v, c)| path_from(v, c.as_ref(), false, 1.0 / 9.0)).collect::<PyResult<_>>()?;
        let t = transforms.as_array();
        let transforms: Vec<TransAffine> = if t.shape()[0] == 0 || t.len() == 0 {
            Vec::new()
        } else {
            (0..t.shape()[0])
                .map(|i| TransAffine {
                    sx: t[[i, 0, 0]],
                    shy: t[[i, 1, 0]],
                    shx: t[[i, 0, 1]],
                    sy: t[[i, 1, 1]],
                    tx: t[[i, 0, 2]],
                    ty: t[[i, 1, 2]],
                })
                .collect()
        };
        let o = offsets.as_array();
        let offsets: Vec<[f64; 2]> =
            if o.len() == 0 { Vec::new() } else { (0..o.shape()[0]).map(|i| [o[[i, 0]], o[[i, 1]]]).collect() };
        let colors = |a: &PyReadonlyArray2<f64>| -> Vec<Rgba> {
            let a = a.as_array();
            if a.len() == 0 {
                return Vec::new();
            }
            (0..a.shape()[0]).map(|i| Rgba::new(a[[i, 0]], a[[i, 1]], a[[i, 2]], a[[i, 3]])).collect()
        };
        let fc = colors(&facecolors);
        let ec = colors(&edgecolors);
        let lw: Vec<f64> = linewidths.as_array().iter().copied().collect();
        let ls: Vec<Dashes> = dashes
            .iter()
            .map(|(off, seq)| match seq {
                Some(s) => Dashes::from_seq(*off, s),
                None => Dashes::default(),
            })
            .collect();
        let aa: Vec<bool> = antialiaseds.as_array().iter().map(|&v| v != 0).collect();
        self.inner.draw_path_collection(
            &gc,
            &trans_from(&master_transform)?,
            &paths,
            &transforms,
            &offsets,
            &trans_from(&offset_trans)?,
            &fc,
            &ec,
            &lw,
            &ls,
            &aa,
            true,
            true,
        );
        Ok(())
    }

    fn draw_text_image(&mut self, image: PyReadonlyArray2<u8>, x: i32, y: i32, gc: &Bound<'_, PyDict>) -> PyResult<()> {
        let gc = gc_from(gc)?;
        let a = image.as_array();
        let (h, w) = (a.shape()[0], a.shape()[1]);
        let data: Vec<u8> = a.iter().copied().collect();
        self.inner.draw_text_image(&gc, &data, w, h, x, y);
        Ok(())
    }

    fn draw_image(&mut self, gc: &Bound<'_, PyDict>, x: f64, y: f64, image: PyReadonlyArray3<u8>) -> PyResult<()> {
        let gc = gc_from(gc)?;
        let a = image.as_array();
        let (h, w) = (a.shape()[0], a.shape()[1]);
        let data: Vec<u8> = a.iter().copied().collect();
        self.inner.draw_image(&gc, x, y, &data, w, h);
        Ok(())
    }

    fn clear(&mut self) {
        self.inner.clear();
    }

    fn buffer_rgba<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<u8>>> {
        let (w, h) = (self.inner.width as usize, self.inner.height as usize);
        let arr = numpy::PyArray1::from_slice(py, self.inner.buffer());
        Ok(arr.reshape([h, w, 4])?)
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRendererAgg>()?;
    Ok(())
}
