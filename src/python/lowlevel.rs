//! Low-level bindings exposing the ported `RendererAgg` with an interface
//! close to matplotlib's `_backend_agg.RendererAgg`.  Used for differential
//! testing against matplotlib's C++ implementation.

use numpy::{PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyDictMethods};

use crate::agg::pixfmt::Rgba;
use crate::agg::stroke::{LineCap, LineJoin};
use crate::agg::TransAffine;
use std::cell::Cell;
use std::rc::Rc;

use pyo3::exceptions::PyRuntimeError;

use crate::backend::path_converters::SnapMode;
use crate::backend::renderer_agg::{ClipPath, Dashes, Gc, Hatch, MplPath, RendererAgg};
use crate::mpl::artist::GcState;
use crate::mpl::renderer::{GlyphDraw, Renderer};
use crate::mpl::transforms::Bbox;

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

/// The same graphics-context dict as the renderer-level `GcState` (used by
/// the text drawing code).
pub fn gcstate_from(d: &Bound<'_, PyDict>) -> PyResult<GcState> {
    let g = gc_from(d)?;
    let mut gc = GcState::default();
    gc.linewidth = g.linewidth;
    gc.alpha = g.alpha;
    gc.forced_alpha = g.forced_alpha;
    gc.rgb = g.color;
    gc.antialiased = g.isaa;
    gc.capstyle = g.cap;
    gc.joinstyle = g.join;
    gc.cliprect = g.cliprect.map(|r| Bbox::new(r[0], r[1], r[2], r[3]));
    gc.clippath = g.clippath;
    if let Some(v) = d.get_item("dashes")? {
        if !v.is_none() {
            gc.dashes = v.extract()?;
        }
    }
    if let Some(v) = d.get_item("snap")? {
        gc.snap = if v.is_none() { None } else { Some(v.extract::<bool>()?) };
    }
    gc.sketch = g.sketch;
    gc.hatch = g.hatch.map(|h| (h.path, h.color, h.linewidth));
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

enum Target {
    Owned(Box<Renderer>),
    /// The renderer of a figure draw in progress (valid while `alive`).
    Borrowed(*mut Renderer, Rc<Cell<bool>>),
}

/// matplotlib's `_backend_agg.RendererAgg`: either a standalone renderer or
/// a proxy for the renderer of a figure being drawn by the Rust core.
#[pyclass(name = "RendererAgg", module = "matplotlib_rs._mplrs", unsendable)]
pub struct PyRendererAgg {
    t: Target,
}

impl PyRendererAgg {
    /// A proxy valid while `alive` is set (the caller clears it before the
    /// borrow of `r` ends).
    pub fn borrowed(r: &mut Renderer, alive: Rc<Cell<bool>>) -> Self {
        PyRendererAgg { t: Target::Borrowed(r as *mut Renderer, alive) }
    }

    pub fn r(&mut self) -> PyResult<&mut Renderer> {
        match &mut self.t {
            Target::Owned(b) => Ok(b),
            Target::Borrowed(p, alive) => {
                if alive.get() {
                    // SAFETY: the pointer comes from a `&mut Renderer` that its
                    // owner does not use while `alive` is set.
                    Ok(unsafe { &mut **p })
                } else {
                    Err(PyRuntimeError::new_err("the renderer is no longer valid (the draw has finished)"))
                }
            }
        }
    }

    fn rr(&self) -> PyResult<&Renderer> {
        match &self.t {
            Target::Owned(b) => Ok(b),
            Target::Borrowed(p, alive) => {
                if alive.get() {
                    // SAFETY: as in `r`.
                    Ok(unsafe { &**p })
                } else {
                    Err(PyRuntimeError::new_err("the renderer is no longer valid (the draw has finished)"))
                }
            }
        }
    }

    /// The Agg renderer to draw on, or None if drawing is disabled.
    fn agg(&mut self) -> PyResult<Option<&mut RendererAgg>> {
        let r = self.r()?;
        Ok(if r.disabled { None } else { Some(&mut r.agg) })
    }
}

#[pymethods]
impl PyRendererAgg {
    #[new]
    fn new(width: u32, height: u32, dpi: f64) -> PyResult<Self> {
        if dpi <= 0.0 {
            return Err(PyValueError::new_err("dpi must be positive"));
        }
        Ok(PyRendererAgg { t: Target::Owned(Box::new(Renderer::new(width as f64, height as f64, dpi))) })
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
        if let Some(r) = self.agg()? {
            r.draw_path(&gc, &path, &trans, face);
        }
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
        if let Some(r) = self.agg()? {
            r.draw_markers(&gc, &mpath, &trans_from(&marker_trans)?, &path, &trans_from(&trans)?, face);
        }
        Ok(())
    }

    /// paths: list of (vertices, codes)
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (gc, master_transform, paths, transforms, offsets, offset_trans, facecolors,
                        edgecolors, linewidths, dashes, antialiaseds, hatchcolors=None))]
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
        hatchcolors: Option<PyReadonlyArray2<f64>>,
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
        let hc = hatchcolors.as_ref().map(colors).unwrap_or_default();
        let lw: Vec<f64> = linewidths.as_array().iter().copied().collect();
        let ls: Vec<Dashes> = dashes
            .iter()
            .map(|(off, seq)| match seq {
                Some(s) => Dashes::from_seq(*off, s),
                None => Dashes::default(),
            })
            .collect();
        let aa: Vec<bool> = antialiaseds.as_array().iter().map(|&v| v != 0).collect();
        let Some(r) = self.agg()? else { return Ok(()) };
        r.draw_path_collection(
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
            &hc,
        );
        Ok(())
    }

    /// `copy_from_bbox((x1, y1, x2, y2))`.
    fn copy_from_bbox(&mut self, x1: f64, y1: f64, x2: f64, y2: f64) -> PyResult<PyBufferRegion> {
        let r = &self.r()?.agg;
        let (w, h) = (r.width as i32, r.height as i32);
        let rect = [x1 as i32, h - y2 as i32, x2 as i32, h - y1 as i32];
        let (rw, rh) = (rect[2] - rect[0], rect[3] - rect[1]);
        let mut data = vec![0u8; (rw.max(0) * rh.max(0) * 4) as usize];
        copy_rgba(&mut data, rw, rh, &r.pixf.buf, w, h, Some(rect), -rect[0], -rect[1]);
        Ok(PyBufferRegion { rect, width: rw, height: rh, data })
    }

    /// `restore_region(region)` or `restore_region(region, x1, y1, x2, y2,
    /// x, y)` (the part x1..x2, y1..y2 of the region, placed at x, y).
    #[pyo3(signature = (region, x1=None, y1=None, x2=None, y2=None, x=None, y=None))]
    #[allow(clippy::too_many_arguments)]
    fn restore_region(
        &mut self,
        region: &PyBufferRegion,
        x1: Option<i32>,
        y1: Option<i32>,
        x2: Option<i32>,
        y2: Option<i32>,
        x: Option<i32>,
        y: Option<i32>,
    ) -> PyResult<()> {
        let Some(r) = self.agg()? else { return Ok(()) };
        let (w, h) = (r.width as i32, r.height as i32);
        let rr = region.rect;
        match (x1, y1, x2, y2, x, y) {
            (Some(x1), Some(y1), Some(x2), Some(y2), Some(x), Some(y)) => {
                let rect = [x1 - rr[0], y1 - rr[1], x2 - rr[0], y2 - rr[1]];
                copy_rgba(&mut r.pixf.buf, w, h, &region.data, region.width, region.height, Some(rect), x, y);
            }
            (None, None, None, None, None, None) => {
                copy_rgba(&mut r.pixf.buf, w, h, &region.data, region.width, region.height, None, rr[0], rr[1]);
            }
            _ => return Err(PyValueError::new_err("restore_region takes a region and optionally 6 integers")),
        }
        Ok(())
    }

    /// `RendererAgg.draw_gouraud_triangles(gc, points (N, 3, 2), colors
    /// (N, 3, 4), transform)`.
    fn draw_gouraud_triangles(
        &mut self,
        gc: &Bound<'_, PyDict>,
        points: PyReadonlyArray3<f64>,
        colors: PyReadonlyArray3<f64>,
        transform: PyReadonlyArray2<f64>,
    ) -> PyResult<()> {
        let gc = gc_from(gc)?;
        let (p, c) = (points.as_array(), colors.as_array());
        if p.shape()[0] != 0 && (p.shape()[1] != 3 || p.shape()[2] != 2) {
            return Err(PyValueError::new_err(format!("points must have shape (N, 3, 2), got {:?}", p.shape())));
        }
        if c.shape()[0] != 0 && (c.shape()[1] != 3 || c.shape()[2] != 4) {
            return Err(PyValueError::new_err(format!("colors must have shape (N, 3, 4), got {:?}", c.shape())));
        }
        if p.shape()[0] != c.shape()[0] {
            return Err(PyValueError::new_err(format!(
                "points and colors arrays must be the same length, got {} points and {} colors",
                p.shape()[0],
                c.shape()[0]
            )));
        }
        let pts: Vec<[[f64; 2]; 3]> =
            (0..p.shape()[0]).map(|i| std::array::from_fn(|k| [p[[i, k, 0]], p[[i, k, 1]]])).collect();
        let cols: Vec<[[f64; 4]; 3]> =
            (0..c.shape()[0]).map(|i| std::array::from_fn(|k| std::array::from_fn(|j| c[[i, k, j]]))).collect();
        let trans = trans_from(&transform)?;
        let Some(r) = self.agg()? else { return Ok(()) };
        r.draw_gouraud_triangles(&gc, &pts, &cols, &trans);
        Ok(())
    }

    /// `RendererAgg.draw_quad_mesh`: each cell is a 5-vertex path in the
    /// order of matplotlib's `QuadMeshPathIterator` (the winding matters to
    /// the non-antialiased coverage of the cell edges).
    #[allow(clippy::too_many_arguments)]
    fn draw_quad_mesh(
        &mut self,
        gc: &Bound<'_, PyDict>,
        master_transform: PyReadonlyArray2<f64>,
        mesh_width: usize,
        mesh_height: usize,
        coordinates: PyReadonlyArray3<f64>,
        offsets: PyReadonlyArray2<f64>,
        offset_trans: PyReadonlyArray2<f64>,
        facecolors: PyReadonlyArray2<f64>,
        antialiased: bool,
        edgecolors: PyReadonlyArray2<f64>,
    ) -> PyResult<()> {
        let gc = gc_from(gc)?;
        let c = coordinates.as_array();
        if c.shape()[0] < mesh_height + 1 || c.shape()[1] < mesh_width + 1 || c.shape()[2] < 2 {
            return Err(PyValueError::new_err("coordinates do not match the mesh size"));
        }
        let mut paths = Vec::with_capacity(mesh_width * mesh_height);
        for i in 0..mesh_width * mesh_height {
            let (m, n) = (i % mesh_width, i / mesh_width);
            let v = |idx: usize| {
                let mm = m + ((idx & 2) >> 1);
                let nn = n + (((idx + 1) & 2) >> 1);
                [c[[nn, mm, 0]], c[[nn, mm, 1]]]
            };
            paths.push(MplPath::new((0..5).map(v).collect(), None));
        }
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
        let lw = [gc.linewidth];
        let Some(r) = self.agg()? else { return Ok(()) };
        r.draw_path_collection(
            &gc,
            &trans_from(&master_transform)?,
            &paths,
            &[],
            &offsets,
            &trans_from(&offset_trans)?,
            &fc,
            &ec,
            &lw,
            &[],
            &[antialiased],
            true,
            false,
            &[],
        );
        Ok(())
    }

    fn draw_text_image(&mut self, image: PyReadonlyArray2<u8>, x: i32, y: i32, gc: &Bound<'_, PyDict>) -> PyResult<()> {
        let gc = gc_from(gc)?;
        let a = image.as_array();
        let (h, w) = (a.shape()[0], a.shape()[1]);
        let data: Vec<u8> = a.iter().copied().collect();
        if let Some(r) = self.agg()? {
            r.draw_text_image(&gc, &data, w, h, x, y);
        }
        Ok(())
    }

    fn draw_image(&mut self, gc: &Bound<'_, PyDict>, x: f64, y: f64, image: PyReadonlyArray3<u8>) -> PyResult<()> {
        let gc = gc_from(gc)?;
        let a = image.as_array();
        let (h, w) = (a.shape()[0], a.shape()[1]);
        let data: Vec<u8> = a.iter().copied().collect();
        if let Some(r) = self.agg()? {
            r.draw_image(&gc, x, y, &data, w, h);
        }
        Ok(())
    }

    fn clear(&mut self) -> PyResult<()> {
        self.r()?.clear();
        Ok(())
    }

    fn buffer_rgba<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<u8>>> {
        let r = self.rr()?;
        let (w, h) = (r.agg.width as usize, r.agg.height as usize);
        let arr = numpy::PyArray1::from_slice(py, r.agg.buffer());
        Ok(arr.reshape([h, w, 4])?)
    }

    #[getter]
    fn width(&self) -> PyResult<f64> {
        Ok(self.rr()?.width)
    }

    #[getter]
    fn height(&self) -> PyResult<f64> {
        Ok(self.rr()?.height)
    }

    #[getter]
    fn dpi(&self) -> PyResult<f64> {
        Ok(self.rr()?.dpi)
    }

    /// Whether drawing calls are ignored (`_draw_disabled`).
    #[getter]
    fn disabled(&self) -> PyResult<bool> {
        Ok(self.rr()?.disabled)
    }

    /// Whether this proxy can still be used.
    #[getter]
    fn valid(&self) -> bool {
        match &self.t {
            Target::Owned(_) => true,
            Target::Borrowed(_, alive) => alive.get(),
        }
    }

    /// `RendererAgg.draw_text` (prop is the Rust font-properties tuple).
    #[pyo3(signature = (gc, x, y, s, prop, angle, ismath=false))]
    #[allow(clippy::too_many_arguments)]
    fn draw_text(&mut self, gc: &Bound<'_, PyDict>, x: f64, y: f64, s: &str, prop: &Bound<'_, PyAny>, angle: f64, ismath: bool) -> PyResult<()> {
        let gc = gcstate_from(gc)?;
        let fp = super::api::fontprops_from(prop)?;
        self.r()?.draw_text(&gc, x, y, s, &fp, angle, ismath);
        Ok(())
    }

    fn get_text_width_height_descent(&mut self, s: &str, prop: &Bound<'_, PyAny>, ismath: bool) -> PyResult<(f64, f64, f64)> {
        let fp = super::api::fontprops_from(prop)?;
        Ok(self.r()?.get_text_width_height_descent(s, &fp, ismath))
    }

    /// `_draw_text_glyphs_and_boxes`; glyphs are (font path, face index,
    /// size, glyph index, slant, extend, dx, dy).
    #[allow(clippy::too_many_arguments)]
    fn draw_glyphs_and_boxes(
        &mut self,
        gc: &Bound<'_, PyDict>,
        x: f64,
        y: f64,
        angle: f64,
        glyphs: Vec<(String, u32, f64, u32, f64, f64, f64, f64)>,
        boxes: Vec<(f64, f64, f64, f64)>,
    ) -> PyResult<()> {
        let gc = gcstate_from(gc)?;
        let mut fonts: Vec<std::sync::Arc<crate::font::FontData>> = Vec::new();
        let mut keys: Vec<(String, u32)> = Vec::new();
        let mut gd = Vec::with_capacity(glyphs.len());
        for (path, index, size, gid, slant, extend, dx, dy) in glyphs {
            let k = (path, index);
            let fi = match keys.iter().position(|x| *x == k) {
                Some(i) => i,
                None => {
                    let data = crate::mpl::fonts::load_font_data(&k.0, k.1)
                        .ok_or_else(|| PyValueError::new_err(format!("cannot load font {}", k.0)))?;
                    fonts.push(data);
                    keys.push(k);
                    fonts.len() - 1
                }
            };
            gd.push(GlyphDraw { font: fi, size, gid, slant, extend, dx, dy });
        }
        let r = self.r()?;
        if !r.disabled {
            r.draw_text_glyphs_and_boxes(&gc, x, y, angle, &fonts, &gd, &boxes);
        }
        Ok(())
    }
}

/// `FT2Font.layout` with a fallback list: (font index, glyph index, cluster,
/// x_advance, y_advance, x_offset, y_offset) for each glyph.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn layout_fonts(fonts: Vec<PyRef<'_, PyFont>>, text: &str) -> Vec<(usize, u32, u32, i32, i32, i32, i32)> {
    let refs: Vec<&crate::font::FtFont> = fonts.iter().map(|f| &f.inner).collect();
    let cps: Vec<u32> = text.chars().map(|c| c as u32).collect();
    crate::font::face::layout_with_fallbacks(&refs, &cps)
        .into_iter()
        .map(|(f, g)| (f, g.index, g.cluster, g.x_advance, g.y_advance, g.x_offset, g.y_offset))
        .collect()
}

/// `FT2Font.set_text` with a fallback list: (width, height, descent) in
/// pixels and the characters to warn about (`warn[i]`: last-resort font).
#[pyfunction]
#[allow(clippy::type_complexity)]
fn set_text_fonts(fonts: Vec<PyRef<'_, PyFont>>, text: &str, warn: Vec<bool>) -> ((f64, f64, f64), Vec<u32>, Vec<usize>) {
    let refs: Vec<&crate::font::FtFont> = fonts.iter().map(|f| &f.inner).collect();
    let cps: Vec<u32> = text.chars().map(|c| c as u32).collect();
    let (adv, bbox, missing, seen) = crate::font::face::set_text_with_fallbacks(&refs, &cps, &warn);
    ((adv as f64 / 64.0, (bbox.y_max - bbox.y_min) as f64 / 64.0, (-bbox.y_min) as f64 / 64.0), missing, seen)
}

/// Missing-glyph warnings recorded by the Rust text engine: (codepoint, fonts).
#[pyfunction]
fn take_missing_glyphs() -> Vec<(u32, String)> {
    crate::mpl::renderer::take_missing_glyphs()
}

/// `renderer_base::copy_from(src, rect_src, dx, dy)` between RGBA32
/// buffers, clipped to the destination buffer.
#[allow(clippy::too_many_arguments)]
fn copy_rgba(dst: &mut [u8], dw: i32, dh: i32, src: &[u8], sw: i32, sh: i32, rect: Option<[i32; 4]>, dx: i32, dy: i32) {
    let mut rsrc = [0, 0, sw, sh];
    if let Some(r) = rect {
        rsrc = [r[0], r[1], r[2] + 1, r[3] + 1];
    }
    let mut rdst = [rsrc[0] + dx, rsrc[1] + dy, rsrc[2] + dx, rsrc[3] + dy];
    // clip_rect_area
    let cb = [0, 0, dw, dh];
    if rsrc[0] < 0 {
        rdst[0] -= rsrc[0];
        rsrc[0] = 0;
    }
    if rsrc[1] < 0 {
        rdst[1] -= rsrc[1];
        rsrc[1] = 0;
    }
    rsrc[2] = rsrc[2].min(sw);
    rsrc[3] = rsrc[3].min(sh);
    if rdst[0] < cb[0] {
        rsrc[0] += cb[0] - rdst[0];
        rdst[0] = cb[0];
    }
    if rdst[1] < cb[1] {
        rsrc[1] += cb[1] - rdst[1];
        rdst[1] = cb[1];
    }
    rdst[2] = rdst[2].min(cb[2]);
    rdst[3] = rdst[3].min(cb[3]);
    let w = (rdst[2] - rdst[0]).min(rsrc[2] - rsrc[0]);
    let h = (rdst[3] - rdst[1]).min(rsrc[3] - rsrc[1]);
    if w <= 0 {
        return;
    }
    for k in 0..h.max(0) {
        let (sy, ty) = ((rsrc[1] + k) as usize, (rdst[1] + k) as usize);
        let s = (sy * sw as usize + rsrc[0] as usize) * 4;
        let t = (ty * dw as usize + rdst[0] as usize) * 4;
        let n = w as usize * 4;
        dst[t..t + n].copy_from_slice(&src[s..s + n]);
    }
}

/// `BufferRegion`: a copy of part of a renderer's buffer.
#[pyclass(name = "BufferRegion", module = "matplotlib_rs._mplrs")]
pub struct PyBufferRegion {
    rect: [i32; 4],
    width: i32,
    height: i32,
    data: Vec<u8>,
}

#[pymethods]
impl PyBufferRegion {
    fn set_x(&mut self, x: i32) {
        self.rect[0] = x;
    }

    fn set_y(&mut self, y: i32) {
        self.rect[1] = y;
    }

    fn get_extents(&self) -> (i32, i32, i32, i32) {
        (self.rect[0], self.rect[1], self.rect[2], self.rect[3])
    }

    #[pyo3(signature = (dtype=None, copy=None))]
    fn __array__<'py>(
        &self,
        py: Python<'py>,
        dtype: Option<&Bound<'py, PyAny>>,
        copy: Option<bool>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let _ = copy;
        let a = numpy::ndarray::Array3::from_shape_vec(
            (self.height.max(0) as usize, self.width.max(0) as usize, 4),
            self.data.clone(),
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let arr = numpy::PyArray3::from_owned_array(py, a).into_any();
        match dtype {
            Some(d) => arr.call_method1("astype", (d,)),
            None => Ok(arr),
        }
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRendererAgg>()?;
    m.add_class::<PyBufferRegion>()?;
    m.add_class::<PyFont>()?;
    m.add_function(wrap_pyfunction!(layout_fonts, m)?)?;
    m.add_function(wrap_pyfunction!(set_text_fonts, m)?)?;
    m.add_function(wrap_pyfunction!(take_missing_glyphs, m)?)?;
    Ok(())
}

#[pyclass(name = "FT2Font", module = "matplotlib_rs._mplrs", unsendable)]
pub struct PyFont {
    inner: crate::font::FtFont,
    /// the outline of the last `load` (`face->glyph->outline`)
    last: Option<crate::font::outline::Outline>,
}

#[pymethods]
impl PyFont {
    #[new]
    #[pyo3(signature = (path, face_index=0))]
    fn new(path: &str, face_index: u32) -> PyResult<Self> {
        let data = crate::mpl::fonts::load_font_data(path, face_index)
            .ok_or_else(|| PyValueError::new_err(format!("cannot load font {path}")))?;
        Ok(PyFont { inner: crate::font::FtFont::new(data), last: None })
    }

    /// A font from the bytes of a font file (`FT2Font(file-like)`).
    #[staticmethod]
    #[pyo3(signature = (data, face_index=0))]
    fn from_bytes(data: Vec<u8>, face_index: u32) -> PyResult<Self> {
        let d = crate::font::FontData::from_bytes(data, face_index).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyFont { inner: crate::font::FtFont::new(std::sync::Arc::new(d)), last: None })
    }

    /// `load_glyph(gid, flags)` (unhinted; `no_scale` for `LoadFlags.NO_SCALE`):
    /// (width, height, horiBearingX, horiBearingY, horiAdvance,
    /// linearHoriAdvance, bbox).  The outline is kept for `get_path`.
    #[allow(clippy::type_complexity)]
    fn load(&mut self, gid: u32, no_scale: bool) -> PyResult<(i64, i64, i64, i64, i64, i64, (i64, i64, i64, i64))> {
        let (o, m) = self.inner.load_glyph_full(gid, no_scale).map_err(|e| PyValueError::new_err(e.to_string()))?;
        self.last = Some(o);
        Ok((m.width, m.height, m.hori_bearing_x, m.hori_bearing_y, m.hori_advance, m.linear_hori_advance,
            (m.bbox.x_min, m.bbox.y_min, m.bbox.x_max, m.bbox.y_max)))
    }

    /// `get_path()` of the last loaded glyph: (vertices, codes).
    fn get_path<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, numpy::PyArray2<f64>>, Bound<'py, numpy::PyArray1<u8>>)> {
        let o = self.last.as_ref().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("No glyph loaded"))?;
        let (v, c) = o.decompose().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("FT_Outline_Decompose failed"))?;
        let n = v.len();
        let flat: Vec<f64> = v.into_iter().flatten().collect();
        Ok((numpy::PyArray1::from_vec(py, flat).reshape([n, 2])?, numpy::PyArray1::from_vec(py, c)))
    }

    /// `FT_Get_Glyph_Name` (None when the font has no glyph names).
    fn glyph_name(&self, gid: u32) -> Option<String> {
        self.inner.sfnt().glyph_name(gid)
    }

    /// The raw bytes of an SFNT table.
    fn table<'py>(&self, py: Python<'py>, tag: &str) -> Option<Bound<'py, PyBytes>> {
        let t: [u8; 4] = tag.as_bytes().try_into().ok()?;
        self.inner.sfnt().table_data(&t).map(|d| PyBytes::new(py, d))
    }

    /// The `name` table records: (platform, encoding, language, name id, bytes).
    fn name_records<'py>(&self, py: Python<'py>) -> Vec<(u16, u16, u16, u16, Bound<'py, PyBytes>)> {
        self.inner.sfnt().names.iter().map(|(p, e, l, n, b)| (*p, *e, *l, *n, PyBytes::new(py, b))).collect()
    }

    /// Whether the font has CFF (not TrueType) outlines, is CID-keyed, and
    /// has an encoding FreeType makes a charmap of.
    fn cff_kind(&self) -> (bool, bool, bool) {
        match &self.inner.sfnt().cff {
            Some(c) => (true, c.is_cid, c.has_encoding_cmap),
            None => (false, false, false),
        }
    }

    /// libraqm's glyphs for *text*: (index, cluster, x_advance, y_advance,
    /// x_offset, y_offset) in 26.6.
    fn _shape(&self, text: &str) -> Vec<(u32, u32, i32, i32, i32, i32)> {
        let cps: Vec<u32> = text.chars().map(|c| c as u32).collect();
        self.inner.shape(&cps).into_iter().map(|g| (g.index, g.cluster, g.x_advance, g.y_advance, g.x_offset, g.y_offset)).collect()
    }

    /// `load_glyph(gid, NO_HINTING)` metrics: (width, height, horiBearingX,
    /// horiBearingY, horiAdvance, linearHoriAdvance, bbox).
    #[allow(clippy::type_complexity)]
    fn glyph_metrics(&self, gid: u32) -> PyResult<(i64, i64, i64, i64, i64, i64, (i64, i64, i64, i64))> {
        let m = self.inner.glyph_metrics(gid).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((m.width, m.height, m.hori_bearing_x, m.hori_bearing_y, m.hori_advance, m.linear_hori_advance,
            (m.bbox.x_min, m.bbox.y_min, m.bbox.x_max, m.bbox.y_max)))
    }

    /// `get_kerning(left, right, mode)` in 26.6 (mode: 0 default, 1 unfitted, 2 unscaled).
    fn get_kerning(&self, left: u32, right: u32, mode: u32) -> i64 {
        self.inner.kerning(left, right, mode)
    }

    #[getter]
    fn postscript_name(&self) -> String {
        self.inner.sfnt().postscript_name.clone()
    }

    #[getter]
    fn family_name(&self) -> String {
        self.inner.sfnt().family_name.clone()
    }

    #[getter]
    fn style_name(&self) -> String {
        self.inner.sfnt().style_name.clone()
    }

    #[getter]
    fn num_glyphs(&self) -> u32 {
        self.inner.sfnt().num_glyphs as u32
    }

    #[getter]
    fn units_per_em(&self) -> u32 {
        self.inner.sfnt().units_per_em as u32
    }

    #[getter]
    fn underline_thickness(&self) -> i32 {
        self.inner.sfnt().underline_thickness as i32
    }

    #[getter]
    fn underline_position(&self) -> i32 {
        self.inner.sfnt().underline_position as i32
    }

    fn set_size(&mut self, ptsize: f64, dpi: f64) {
        self.inner.set_size(ptsize, dpi);
    }

    fn _set_transform(&mut self, matrix: [[i64; 2]; 2], delta: [i64; 2]) {
        let m = crate::font::outline::FtMatrix { xx: matrix[0][0], xy: matrix[0][1], yx: matrix[1][0], yy: matrix[1][1] };
        self.inner.set_transform(m, (delta[0], delta[1]));
    }

    /// Returns (left, top, buffer[rows, width]).
    fn _render_glyph<'py>(&self, py: Python<'py>, gid: u32) -> PyResult<(i32, i32, Bound<'py, numpy::PyArray2<u8>>)> {
        let bm = self.inner.render_glyph(gid).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let arr = numpy::PyArray1::from_slice(py, &bm.buffer).reshape([bm.rows as usize, bm.width as usize])?;
        Ok((bm.left, bm.top, arr))
    }

    /// Returns list of (glyph_index, x, y).
    fn _layout(&self, text: &str) -> Vec<(u32, f64, f64)> {
        let cps: Vec<u32> = text.chars().map(|c| c as u32).collect();
        self.inner.layout(&cps)
    }

    fn get_width_height_descent(&self, text: &str) -> (f64, f64, f64) {
        let cps: Vec<u32> = text.chars().map(|c| c as u32).collect();
        self.inner.text_width_height_descent(&cps)
    }

    /// `get_sfnt_table(name)` for 'OS/2', 'head' and 'hhea'.
    fn get_sfnt_table<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Option<Bound<'py, PyDict>>> {
        let sf = self.inner.sfnt();
        let d = PyDict::new(py);
        match name {
            "OS/2" => match sf.os2 {
                Some(o) if o.version != 0xffff => {
                    d.set_item("version", o.version)?;
                    d.set_item("usWeightClass", o.us_weight_class)?;
                    d.set_item("fsSelection", o.fs_selection)?;
                    d.set_item("sTypoAscender", o.s_typo_ascender)?;
                    d.set_item("sTypoDescender", o.s_typo_descender)?;
                    d.set_item("sTypoLineGap", o.s_typo_line_gap)?;
                }
                _ => return Ok(None),
            },
            "head" => {
                d.set_item("unitsPerEm", sf.units_per_em)?;
                d.set_item("xMin", sf.bbox.0)?;
                d.set_item("yMin", sf.bbox.1)?;
                d.set_item("xMax", sf.bbox.2)?;
                d.set_item("yMax", sf.bbox.3)?;
                d.set_item("macStyle", sf.mac_style)?;
            }
            "hhea" => {
                d.set_item("ascent", sf.ascender)?;
                d.set_item("descent", sf.descender)?;
                d.set_item("lineGap", sf.line_gap)?;
                d.set_item("advanceWidthMax", sf.max_advance_width)?;
            }
            _ => return Ok(None),
        }
        Ok(Some(d))
    }

    fn get_char_index(&self, c: u32) -> u32 {
        self.inner.glyph_index(c)
    }

    /// `set_charmap(i)`.
    fn set_charmap(&mut self, i: usize) -> PyResult<()> {
        if i >= self.inner.data.cmaps.len() {
            return Err(PyValueError::new_err("Could not set the charmap"));
        }
        self.inner.charmap = Some(i);
        Ok(())
    }

    fn outline_points(&self, gid: u32) -> PyResult<Vec<(i64, i64, u8)>> {
        let o = self.inner.load_glyph(gid).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(o.points.iter().zip(o.tags.iter()).map(|(p, t)| (p.0, p.1, *t)).collect())
    }

    /// The loaded outline as (points, tags, contour ends), in 26.6.
    fn outline(&self, gid: u32) -> PyResult<(Vec<(i64, i64)>, Vec<u8>, Vec<u16>)> {
        let o = self.inner.load_glyph(gid).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((o.points, o.tags, o.contours))
    }
}
