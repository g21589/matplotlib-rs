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

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRendererAgg>()?;
    m.add_class::<PyFont>()?;
    Ok(())
}

#[pyclass(name = "FT2Font", module = "matplotlib_rs._mplrs", unsendable)]
pub struct PyFont {
    inner: crate::font::FtFont,
}

#[pymethods]
impl PyFont {
    #[new]
    #[pyo3(signature = (path, face_index=0))]
    fn new(path: &str, face_index: u32) -> PyResult<Self> {
        let data = crate::mpl::fonts::load_font_data(path, face_index)
            .ok_or_else(|| PyValueError::new_err(format!("cannot load font {path}")))?;
        Ok(PyFont { inner: crate::font::FtFont::new(data) })
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

    fn outline_points(&self, gid: u32) -> PyResult<Vec<(i64, i64, u8)>> {
        let o = self.inner.load_glyph(gid).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(o.points.iter().zip(o.tags.iter()).map(|(p, t)| (p.0, p.1, *t)).collect())
    }
}
