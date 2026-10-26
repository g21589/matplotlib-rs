//! Figure / Axes / Text / Patch handles for the Python API layer.

use std::rc::Rc;
use std::sync::Arc;

use numpy::{PyArray3, PyArrayMethods};
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods, PyList, PyTuple};

use super::model::{base_set, color_from, f64_vec, opt_f64, PyLine2D};
use crate::agg::stroke::LineCap;
use crate::mpl::artist::{shared, Lock, Shared, SpinePosition, TransCtx};
use crate::mpl::axes::{anchor_coefs, scale_from_name, Aspect, Axes, Child};
use crate::mpl::axis::{GridKw, TickKw};
use crate::mpl::colors::ColorSpec;
use crate::mpl::figure::{snapshot_for, DrawSnapshot, FigChild, Figure, LayoutEngine};
use crate::mpl::fonts::{self, FontProperties, NumOrName};
use crate::mpl::logticker::{LogFormatter, LogFormatterKind, LogLocator, Subs};
use crate::mpl::patches::{BoxStyle, Patch, PatchKind, RotationPoint};
use crate::mpl::path::{new_path, Path};
use crate::mpl::renderer::Renderer;
use crate::mpl::scale::Scale;
use crate::mpl::text::Text;
use crate::mpl::ticker::{AxisInfo, ExternalFormatter, ExternalLocator, Formatter, Locator, MaxNLocator, MinorNDivs, NBins, ScalarFormatter};
use crate::mpl::transforms::Bbox;

fn err<T>(m: impl Into<String>) -> PyResult<T> {
    Err(PyValueError::new_err(m.into()))
}

// ---------------------------------------------------------------------------
// font properties

pub fn fontprops_from(obj: &Bound<'_, PyAny>) -> PyResult<FontProperties> {
    let t = obj.cast::<PyTuple>()?;
    let family: Vec<String> = t.get_item(0)?.extract()?;
    let style: String = t.get_item(1)?.extract()?;
    let variant: String = t.get_item(2)?.extract()?;
    let w = t.get_item(3)?;
    let weight = if let Ok(n) = w.extract::<i32>() { NumOrName::Num(n) } else { NumOrName::Name(w.extract()?) };
    let st = t.get_item(4)?;
    let stretch = if let Ok(n) = st.extract::<i32>() { NumOrName::Num(n) } else { NumOrName::Name(st.extract()?) };
    let size: f64 = t.get_item(5)?.extract()?;
    let file: Option<String> = t.get_item(6)?.extract()?;
    let math: String = t.get_item(7)?.extract()?;
    Ok(FontProperties { family, style, variant, weight, stretch, size, file, math_fontfamily: math })
}

pub fn fontprops_to<'py>(py: Python<'py>, p: &FontProperties) -> PyResult<Bound<'py, PyTuple>> {
    let w: Bound<'py, PyAny> = match &p.weight {
        NumOrName::Num(n) => n.into_pyobject(py)?.into_any(),
        NumOrName::Name(s) => s.into_pyobject(py)?.into_any(),
    };
    let st: Bound<'py, PyAny> = match &p.stretch {
        NumOrName::Num(n) => n.into_pyobject(py)?.into_any(),
        NumOrName::Name(s) => s.into_pyobject(py)?.into_any(),
    };
    PyTuple::new(
        py,
        [
            PyList::new(py, &p.family)?.into_any(),
            p.style.clone().into_pyobject(py)?.into_any(),
            p.variant.clone().into_pyobject(py)?.into_any(),
            w,
            st,
            p.size.into_pyobject(py)?.into_any(),
            p.file.clone().into_pyobject(py)?.into_any(),
            p.math_fontfamily.clone().into_pyobject(py)?.into_any(),
        ],
    )
}

fn rgba_tuple<'py>(py: Python<'py>, c: crate::agg::pixfmt::Rgba) -> PyResult<Bound<'py, PyTuple>> {
    PyTuple::new(py, [c.r, c.g, c.b, c.a])
}

#[allow(dead_code)]
fn colorspec_to<'py>(py: Python<'py>, c: ColorSpec) -> PyResult<Bound<'py, PyAny>> {
    Ok(match c {
        ColorSpec::Rgba(c) => rgba_tuple(py, c)?.into_any(),
        ColorSpec::None => "none".into_pyobject(py)?.into_any(),
        ColorSpec::Auto => "auto".into_pyobject(py)?.into_any(),
    })
}

// ---------------------------------------------------------------------------
// Text

fn text_set(t: &mut Text, name: &str, v: &Bound<'_, PyAny>) -> PyResult<()> {
    if base_set(&mut t.base, name, v)? {
        return Ok(());
    }
    match name {
        "text" => t.text = v.extract()?,
        "x" => t.x = v.extract()?,
        "y" => t.y = v.extract()?,
        "color" => t.color = color_from(v)?.to_rgba(None),
        "fontproperties" => t.fontproperties = fontprops_from(v)?,
        "fontsize" => t.fontproperties.size = v.extract()?,
        "ha" => t.ha = v.extract()?,
        "va" => t.va = v.extract()?,
        "multialignment" => t.multialignment = v.extract()?,
        "rotation" => t.set_rotation_deg(v.extract()?),
        "rotation_mode" => t.rotation_mode = v.extract::<Option<String>>()?.unwrap_or_else(|| "default".into()),
        "linespacing" => t.linespacing = opt_f64(v)?,
        "wrap" => t.wrap = v.extract()?,
        "usetex" => t.usetex = v.extract()?,
        "parse_math" => t.parse_math = v.extract()?,
        "antialiased" => t.antialiased = v.extract()?,
        "transform_rotates_text" => t.transform_rotates_text = v.extract()?,
        "url" => t.url = v.extract()?,
        "bbox" => {
            // None, a patch, or (patch, box style): with a box style, the
            // text sizes the box when drawn
            if v.is_none() {
                t.bbox_patch = None;
            } else {
                let (p, spec) = match v.cast::<PyTuple>() {
                    Ok(tp) => (tp.get_item(0)?, tp.get_item(1)?.extract::<Option<String>>()?),
                    Err(_) => (v.clone(), None),
                };
                let mut patch = p.cast::<PyPatch>()?.borrow().inner.lk().clone();
                if let Some(spec) = spec {
                    let boxstyle = BoxStyle::parse(&spec).ok_or_else(|| PyValueError::new_err(format!("unknown boxstyle {spec}")))?;
                    patch.kind = PatchKind::FancyBbox { x: 0.0, y: 0.0, width: 1.0, height: 1.0, boxstyle };
                }
                t.bbox_patch = Some(Box::new(crate::mpl::text::TextBbox { patch, pad: 0.0 }));
            }
        }
        _ => return Err(PyKeyError::new_err(format!("Text has no property {name:?}"))),
    }
    Ok(())
}

fn text_get<'py>(py: Python<'py>, t: &Text, name: &str) -> PyResult<Bound<'py, PyAny>> {
    Ok(match name {
        "text" => t.text.clone().into_pyobject(py)?.into_any(),
        "x" => t.x.into_pyobject(py)?.into_any(),
        "y" => t.y.into_pyobject(py)?.into_any(),
        "color" => rgba_tuple(py, t.color)?.into_any(),
        "fontproperties" => fontprops_to(py, &t.fontproperties)?.into_any(),
        "fontsize" => t.fontproperties.size.into_pyobject(py)?.into_any(),
        "ha" => t.ha.clone().into_pyobject(py)?.into_any(),
        "va" => t.va.clone().into_pyobject(py)?.into_any(),
        "rotation" => t.rotation.into_pyobject(py)?.into_any(),
        "rotation_mode" => t.rotation_mode.clone().into_pyobject(py)?.into_any(),
        "visible" => t.base.visible.into_pyobject(py)?.to_owned().into_any(),
        "alpha" => t.base.alpha.into_pyobject(py)?.into_any(),
        "zorder" => t.base.zorder.into_pyobject(py)?.into_any(),
        "label" => t.base.label.clone().into_pyobject(py)?.into_any(),
        "multialignment" => t.multialignment.clone().into_pyobject(py)?.into_any(),
        "linespacing" => t.linespacing.into_pyobject(py)?.into_any(),
        "wrap" => t.wrap.into_pyobject(py)?.to_owned().into_any(),
        "usetex" => t.usetex.into_pyobject(py)?.to_owned().into_any(),
        "parse_math" => t.parse_math.into_pyobject(py)?.to_owned().into_any(),
        "antialiased" => t.antialiased.into_pyobject(py)?.to_owned().into_any(),
        "clip_on" => t.base.clip_on.into_pyobject(py)?.to_owned().into_any(),
        "in_layout" => t.base.in_layout.into_pyobject(py)?.to_owned().into_any(),
        "transform" => super::model::tspec_to(py, &t.base.transform)?,
        _ => return Err(PyKeyError::new_err(format!("Text has no property {name:?}"))),
    })
}

#[pyclass(name = "Text", module = "matplotlib_rs._mplrs")]
pub struct PyText {
    pub inner: Shared<Text>,
}

impl PyText {
    fn lk(&self) -> std::sync::MutexGuard<'_, Text> {
        Lock::lk(&self.inner)
    }
}

#[pymethods]
impl PyText {
    /// A fingerprint of the text's state (changes whenever the text does).
    fn state_hash(&self) -> u64 {
        super::model::state_hash(&*self.inner.lk())
    }

    #[new]
    #[pyo3(signature = (x=0.0, y=0.0, text=""))]
    fn new(x: f64, y: f64, text: &str) -> Self {
        PyText { inner: shared(Text::new(x, y, text)) }
    }

    /// The Python `Text` this text mirrors (a weak reference), passed as
    /// `mtext` to renderers.
    fn set_owner(&self, weak: Py<PyAny>) {
        self.lk().owner = Some(super::foreign::owner_ref(weak));
    }

    fn set(&self, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        text_set(&mut self.lk(), name, value)
    }

    fn set_many(&self, props: &Bound<'_, PyDict>) -> PyResult<()> {
        let mut t = self.lk();
        for (k, v) in props.iter() {
            text_set(&mut t, &k.extract::<String>()?, &v)?;
        }
        Ok(())
    }

    fn get<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
        text_get(py, &self.lk(), name)
    }

    fn same(&self, other: &PyText) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    fn ident(&self) -> usize {
        Arc::as_ptr(&self.inner) as usize
    }
}

// ---------------------------------------------------------------------------
// Patch

fn path_from_parts(verts: Vec<[f64; 2]>, codes: Option<Vec<u8>>) -> Path {
    new_path(verts, codes, crate::mpl::lines::simplify_params())
}

fn patch_set(p: &mut Patch, name: &str, v: &Bound<'_, PyAny>) -> PyResult<()> {
    if name == "alpha" {
        p.set_alpha(opt_f64(v)?);
        return Ok(());
    }
    if base_set(&mut p.base, name, v)? {
        return Ok(());
    }
    let opt_color = |v: &Bound<'_, PyAny>| -> PyResult<Option<ColorSpec>> {
        if v.is_none() {
            Ok(None)
        } else {
            Ok(Some(color_from(v)?))
        }
    };
    match name {
        "facecolor_rgba" => p.facecolor = color_from(v)?.to_rgba(None),
        "edgecolor_rgba" => p.edgecolor = color_from(v)?.to_rgba(None),
        "hatchcolor_rgba" => p.hatch_color = crate::mpl::patches::HatchColor::Color(color_from(v)?.to_rgba(None)),
        "dash_pattern" => {
            let (o, d): (f64, Option<Vec<f64>>) = v.extract()?;
            p.dash_pattern = (o, d);
        }
        "linestyle_name" => p.linestyle = v.extract()?,
        "facecolor" => p.set_facecolor(opt_color(v)?),
        "edgecolor" => p.set_edgecolor(opt_color(v)?),
        "hatchcolor" => {
            if v.is_none() {
                p.set_hatchcolor(None)
            } else if v.extract::<String>().map(|s| s == "edge").unwrap_or(false) {
                p.set_hatchcolor(Some(None))
            } else {
                p.set_hatchcolor(Some(Some(color_from(v)?)))
            }
        }
        "color" => p.set_color(color_from(v)?),
        "linewidth" => p.set_linewidth(opt_f64(v)?),
        "linestyle" => {
            if let Ok(s) = v.extract::<String>() {
                p.set_linestyle(&s);
            } else {
                let (o, d): (f64, Option<Vec<f64>>) = v.extract()?;
                p.set_dashes(o, d);
            }
        }
        "fill" => p.set_fill(v.extract()?),
        "antialiased" => p.antialiased = v.extract()?,
        "hatch" => p.hatch = v.extract()?,
        "hatch_np" => p.hatch_np = v.extract()?,
        "hatch_linewidth" => p.hatch_linewidth = v.extract()?,
        "capstyle" => p.capstyle = crate::mpl::artist::parse_capstyle(&v.extract::<String>()?),
        "joinstyle" => p.joinstyle = crate::mpl::artist::parse_joinstyle(&v.extract::<String>()?),
        "edgegapcolor" => p.gapcolor = opt_color(v)?.map(|c| c.to_rgba(p.base.alpha)),
        "mutation_scale" => p.mutation_scale = v.extract()?,
        "mutation_aspect" => p.mutation_aspect = v.extract::<Option<f64>>()?.unwrap_or(1.0),
        "sticky_x" => p.sticky_x = v.extract()?,
        "sticky_y" => p.sticky_y = v.extract()?,
        "boxstyle" => {
            let s: String = v.extract()?;
            let bs = BoxStyle::parse(&s).ok_or_else(|| PyValueError::new_err(format!("unknown boxstyle {s}")))?;
            if let PatchKind::FancyBbox { boxstyle, .. } = &mut p.kind {
                *boxstyle = bs;
            }
        }
        "bounds" => {
            let (l, b, w, h): (f64, f64, f64, f64) = v.extract()?;
            p.set_bounds(l, b, w, h);
        }
        "xy" => {
            let (x, y): (f64, f64) = v.extract()?;
            match &mut p.kind {
                PatchKind::Rectangle { x0, y0, .. } => {
                    *x0 = x;
                    *y0 = y;
                }
                PatchKind::FancyBbox { x: px, y: py, .. } => {
                    *px = x;
                    *py = y;
                }
                PatchKind::RegularPolygon { xy, .. } => *xy = (x, y),
                PatchKind::Ellipse { center, .. } => *center = (x, y),
                _ => {}
            }
        }
        "width" | "height" | "angle" => {
            let val: f64 = v.extract()?;
            match &mut p.kind {
                PatchKind::Rectangle { width, height, angle, .. } | PatchKind::Ellipse { width, height, angle, .. } => match name {
                    "width" => *width = val,
                    "height" => *height = val,
                    _ => *angle = val,
                },
                PatchKind::FancyBbox { width, height, .. } => match name {
                    "width" => *width = val,
                    "height" => *height = val,
                    _ => {}
                },
                _ => {}
            }
        }
        "path" => {
            let (verts, codes): (Vec<[f64; 2]>, Option<Vec<u8>>) = v.extract()?;
            p.kind = PatchKind::Path { path: path_from_parts(verts, codes) };
        }
        "generic" => {
            let (verts, codes, mtx): (Vec<[f64; 2]>, Option<Vec<u8>>, Option<Vec<Vec<f64>>>) = v.extract()?;
            let pt = mtx.map(|m| [[m[0][0], m[0][1], m[0][2]], [m[1][0], m[1][1], m[1][2]], [0.0, 0.0, 1.0]]);
            p.kind = PatchKind::Generic { path: path_from_parts(verts, codes), patch_transform: pt };
        }
        "edge_default" => {
            p.edge_default = v.extract()?;
        }
        "rotation_point" => {
            let rp = if let Ok(s) = v.extract::<String>() {
                if s == "center" {
                    RotationPoint::Center
                } else {
                    RotationPoint::Xy
                }
            } else {
                let (a, b): (f64, f64) = v.extract()?;
                RotationPoint::Point(a, b)
            };
            if let PatchKind::Rectangle { rotation_point, .. } = &mut p.kind {
                *rotation_point = rp;
            }
        }
        _ => return Err(PyKeyError::new_err(format!("Patch has no property {name:?}"))),
    }
    Ok(())
}

fn patch_get<'py>(py: Python<'py>, p: &Patch, name: &str) -> PyResult<Bound<'py, PyAny>> {
    Ok(match name {
        "facecolor" => rgba_tuple(py, p.facecolor)?.into_any(),
        "edgecolor" => rgba_tuple(py, p.edgecolor)?.into_any(),
        "linewidth" => p.linewidth.into_pyobject(py)?.into_any(),
        "linestyle" => p.linestyle.clone().into_pyobject(py)?.into_any(),
        "fill" => p.fill.into_pyobject(py)?.to_owned().into_any(),
        "visible" => p.base.visible.into_pyobject(py)?.to_owned().into_any(),
        "alpha" => p.base.alpha.into_pyobject(py)?.into_any(),
        "zorder" => p.base.zorder.into_pyobject(py)?.into_any(),
        "label" => p.base.label.clone().into_pyobject(py)?.into_any(),
        "hatch" => p.hatch.clone().into_pyobject(py)?.into_any(),
        "antialiased" => p.antialiased.into_pyobject(py)?.to_owned().into_any(),
        "clip_on" => p.base.clip_on.into_pyobject(py)?.to_owned().into_any(),
        "transform" => super::model::tspec_to(py, &p.base.transform)?,
        "bounds" | "xy" | "width" | "height" | "angle" => {
            let (x, y, w, h, a) = match &p.kind {
                PatchKind::Rectangle { x0, y0, width, height, angle, .. } => (*x0, *y0, *width, *height, *angle),
                PatchKind::Ellipse { center, width, height, angle } => (center.0, center.1, *width, *height, *angle),
                PatchKind::FancyBbox { x, y, width, height, .. } => (*x, *y, *width, *height, 0.0),
                PatchKind::RegularPolygon { xy, radius, orientation, .. } => (xy.0, xy.1, *radius, *radius, *orientation),
                _ => (0.0, 0.0, 0.0, 0.0, 0.0),
            };
            match name {
                "bounds" => PyTuple::new(py, [x, y, w, h])?.into_any(),
                "xy" => PyTuple::new(py, [x, y])?.into_any(),
                "width" => w.into_pyobject(py)?.into_any(),
                "height" => h.into_pyobject(py)?.into_any(),
                _ => a.into_pyobject(py)?.into_any(),
            }
        }
        "path" => {
            let path = p.get_path();
            let v = PyList::new(py, path.vertices.iter().map(|v| (v[0], v[1])))?;
            let c = path.codes.clone().into_pyobject(py)?;
            PyTuple::new(py, [v.into_any(), c.into_any()])?.into_any()
        }
        _ => return Err(PyKeyError::new_err(format!("Patch has no property {name:?}"))),
    })
}

#[pyclass(name = "Patch", module = "matplotlib_rs._mplrs")]
pub struct PyPatch {
    pub inner: Shared<Patch>,
}

impl PyPatch {
    fn lk(&self) -> std::sync::MutexGuard<'_, Patch> {
        Lock::lk(&self.inner)
    }
}

#[pymethods]
impl PyPatch {
    /// Patch(kind, *params) with kind one of rectangle, polygon, path,
    /// ellipse, wedge, regular_polygon, fancybbox.
    #[new]
    #[pyo3(signature = (kind, *args))]
    fn new(kind: &str, args: &Bound<'_, PyTuple>) -> PyResult<Self> {
        let k = match kind {
            "rectangle" => {
                let (x, y, w, h, angle): (f64, f64, f64, f64, f64) = args.extract()?;
                PatchKind::Rectangle { x0: x, y0: y, width: w, height: h, angle, rotation_point: RotationPoint::Xy }
            }
            "polygon" => {
                let (xy, closed): (Vec<[f64; 2]>, bool) = args.extract()?;
                PatchKind::Path { path: crate::mpl::patches::polygon_path(xy, closed) }
            }
            "path" => {
                let (verts, codes): (Vec<[f64; 2]>, Option<Vec<u8>>) = args.extract()?;
                PatchKind::Path { path: path_from_parts(verts, codes) }
            }
            "ellipse" => {
                let (cx, cy, w, h, angle): (f64, f64, f64, f64, f64) = args.extract()?;
                PatchKind::Ellipse { center: (cx, cy), width: w, height: h, angle }
            }
            "wedge" => {
                let (cx, cy, r, t1, t2, width): (f64, f64, f64, f64, f64, Option<f64>) = args.extract()?;
                PatchKind::Path { path: crate::mpl::patches::wedge_path((cx, cy), r, t1, t2, width) }
            }
            "regular_polygon" => {
                let (x, y, n, r, o): (f64, f64, usize, f64, f64) = args.extract()?;
                PatchKind::RegularPolygon { xy: (x, y), numvertices: n, radius: r, orientation: o }
            }
            "generic" => {
                let (verts, codes, mtx): (Vec<[f64; 2]>, Option<Vec<u8>>, Option<Vec<Vec<f64>>>) = args.extract()?;
                let pt = mtx.map(|m| [[m[0][0], m[0][1], m[0][2]], [m[1][0], m[1][1], m[1][2]], [0.0, 0.0, 1.0]]);
                PatchKind::Generic { path: path_from_parts(verts, codes), patch_transform: pt }
            }
            "fancybbox" => {
                let (x, y, w, h, bs): (f64, f64, f64, f64, String) = args.extract()?;
                let boxstyle = BoxStyle::parse(&bs).ok_or_else(|| PyValueError::new_err(format!("unknown boxstyle {bs}")))?;
                PatchKind::FancyBbox { x, y, width: w, height: h, boxstyle }
            }
            _ => return err(format!("unknown patch kind {kind}")),
        };
        let is_pathpatch = kind == "path";
        let mut p = Patch::new(k);
        if is_pathpatch {
            p.edge_default = true;
            p.set_edgecolor(None);
        }
        Ok(PyPatch { inner: shared(p) })
    }

    fn set(&self, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        patch_set(&mut self.lk(), name, value)
    }

    fn set_many(&self, props: &Bound<'_, PyDict>) -> PyResult<()> {
        let mut p = self.lk();
        for (k, v) in props.iter() {
            patch_set(&mut p, &k.extract::<String>()?, &v)?;
        }
        Ok(())
    }

    /// Copy a PathPatch's style with independent replacement geometry.
    fn copy_with_path(&self, verts: Vec<[f64; 2]>, codes: Option<Vec<u8>>) -> Self {
        let mut patch = self.lk().clone();
        patch.kind = PatchKind::Generic {
            path: path_from_parts(verts, codes), patch_transform: None,
        };
        Self { inner: shared(patch) }
    }

    fn get<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
        patch_get(py, &self.lk(), name)
    }

    fn ident(&self) -> usize {
        Arc::as_ptr(&self.inner) as usize
    }
}

// ---------------------------------------------------------------------------
// external tick helpers

#[derive(Debug)]
struct PyLocator {
    obj: Py<PyAny>,
}

fn info_tuple<'py>(py: Python<'py>, info: &AxisInfo) -> PyResult<Bound<'py, PyTuple>> {
    PyTuple::new(
        py,
        [
            info.view.0.into_pyobject(py)?.into_any(),
            info.view.1.into_pyobject(py)?.into_any(),
            info.data.0.into_pyobject(py)?.into_any(),
            info.data.1.into_pyobject(py)?.into_any(),
            info.minpos.into_pyobject(py)?.into_any(),
            info.tick_space.into_pyobject(py)?.into_any(),
            info.axis_name.to_string().into_pyobject(py)?.into_any(),
            info.scale.clone().into_pyobject(py)?.into_any(),
            PyList::new(py, &info.major_locs)?.into_any(),
        ],
    )
}

impl ExternalLocator for PyLocator {
    fn call(&self, info: &AxisInfo) -> Vec<f64> {
        Python::attach(|py| {
            let r = info_tuple(py, info).and_then(|t| self.obj.bind(py).call_method1("rs_call", (t,)));
            match r.and_then(|v| v.extract::<Vec<f64>>()) {
                Ok(v) => v,
                Err(e) => {
                    e.print(py);
                    Vec::new()
                }
            }
        })
    }
    fn tick_values(&self, vmin: f64, vmax: f64, info: &AxisInfo) -> Vec<f64> {
        Python::attach(|py| {
            let r = info_tuple(py, info).and_then(|t| self.obj.bind(py).call_method1("rs_tick_values", (vmin, vmax, t)));
            match r.and_then(|v| v.extract::<Vec<f64>>()) {
                Ok(v) => v,
                Err(e) => {
                    e.print(py);
                    Vec::new()
                }
            }
        })
    }
    fn nonsingular(&self, v0: f64, v1: f64) -> (f64, f64) {
        Python::attach(|py| match self.obj.bind(py).call_method1("rs_nonsingular", (v0, v1)).and_then(|v| v.extract()) {
            Ok(v) => v,
            Err(e) => {
                e.print(py);
                (v0, v1)
            }
        })
    }
    fn view_limits(&self, v0: f64, v1: f64) -> (f64, f64) {
        Python::attach(|py| match self.obj.bind(py).call_method1("rs_view_limits", (v0, v1)).and_then(|v| v.extract()) {
            Ok(v) => v,
            Err(e) => {
                e.print(py);
                (v0, v1)
            }
        })
    }
    fn is_null(&self) -> bool {
        Python::attach(|py| self.obj.bind(py).getattr("rs_is_null").and_then(|v| v.extract()).unwrap_or(false))
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[derive(Debug)]
struct PyFormatter {
    obj: Py<PyAny>,
}

impl ExternalFormatter for PyFormatter {
    fn format_ticks(&self, values: &[f64], info: &AxisInfo) -> Vec<String> {
        Python::attach(|py| {
            let r = info_tuple(py, info).and_then(|t| self.obj.bind(py).call_method1("rs_format_ticks", (values.to_vec(), t)));
            match r.and_then(|v| v.extract::<Vec<String>>()) {
                Ok(v) => v,
                Err(e) => {
                    e.print(py);
                    vec![String::new(); values.len()]
                }
            }
        })
    }
    fn get_offset(&self) -> String {
        Python::attach(|py| self.obj.bind(py).call_method0("rs_get_offset").and_then(|v| v.extract()).unwrap_or_default())
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

fn locator_from(spec: &Bound<'_, PyAny>) -> PyResult<Locator> {
    let t = spec.cast::<PyTuple>()?;
    let kind: String = t.get_item(0)?.extract()?;
    Ok(match kind.as_str() {
        "auto" => Locator::MaxN(MaxNLocator::auto()),
        "maxn" => {
            let nb = t.get_item(1)?;
            let nbins = if nb.extract::<String>().is_ok() { NBins::Auto } else { NBins::N(nb.extract()?) };
            let steps: Option<Vec<f64>> = t.get_item(2)?.extract()?;
            let mut m = MaxNLocator::new(nbins, steps);
            m.integer = t.get_item(3)?.extract()?;
            m.symmetric = t.get_item(4)?.extract()?;
            m.prune = t.get_item(5)?.extract()?;
            m.min_n_ticks = t.get_item(6)?.extract()?;
            Locator::MaxN(m)
        }
        "fixed" => Locator::Fixed { locs: t.get_item(1)?.extract()?, nbins: t.get_item(2)?.extract()? },
        "null" => Locator::Null,
        "multiple" => Locator::Multiple { step: t.get_item(1)?.extract()?, offset: t.get_item(2)?.extract()? },
        "linear" => Locator::Linear { numticks: t.get_item(1)?.extract()? },
        "index" => Locator::Index { base: t.get_item(1)?.extract()?, offset: t.get_item(2)?.extract()? },
        "autominor" => {
            let n = t.get_item(1)?;
            let nd = if n.is_none() {
                None
            } else if n.extract::<String>().is_ok() {
                Some(MinorNDivs::Auto)
            } else {
                Some(MinorNDivs::N(n.extract()?))
            };
            Locator::AutoMinor { ndivs: nd }
        }
        "log" => {
            let base: f64 = t.get_item(1)?.extract()?;
            let s = t.get_item(2)?;
            let subs = if s.is_none() {
                Subs::Auto
            } else if let Ok(st) = s.extract::<String>() {
                if st == "all" {
                    Subs::All
                } else {
                    Subs::Auto
                }
            } else {
                Subs::List(s.extract()?)
            };
            let nt = t.get_item(3)?;
            let numticks = if nt.is_none() || nt.extract::<String>().is_ok() { None } else { Some(nt.extract()?) };
            Locator::Log(LogLocator::new(base, subs, numticks))
        }
        "py" => Locator::External(Arc::new(PyLocator { obj: t.get_item(1)?.unbind() })),
        _ => return err(format!("unknown locator {kind}")),
    })
}

fn formatter_from(spec: &Bound<'_, PyAny>) -> PyResult<Formatter> {
    let t = spec.cast::<PyTuple>()?;
    let kind: String = t.get_item(0)?.extract()?;
    Ok(match kind.as_str() {
        "scalar" => {
            let mut f = ScalarFormatter::new();
            let uo = t.get_item(1)?;
            if let Ok(b) = uo.extract::<bool>() {
                f.set_use_offset(b);
            } else if !uo.is_none() {
                f.set_offset_value(uo.extract()?);
            }
            if let Some(v) = t.get_item(2)?.extract::<Option<bool>>()? {
                f.use_mathtext = v;
            }
            if let Some(v) = t.get_item(3)?.extract::<Option<bool>>()? {
                f.use_locale = v;
            }
            if let Some(v) = t.get_item(4)?.extract::<Option<bool>>()? {
                f.usetex = v;
            }
            f.scientific = t.get_item(5)?.extract()?;
            if let Some((a, b)) = t.get_item(6)?.extract::<Option<(i64, i64)>>()? {
                f.powerlimits = (a, b);
            }
            Formatter::Scalar(f)
        }
        "null" => Formatter::Null,
        "fixed" => Formatter::Fixed { seq: t.get_item(1)?.extract()?, offset_string: t.get_item(2)?.extract()? },
        "log" | "logsci" | "logmath" | "logexp" => {
            let base: f64 = t.get_item(1)?.extract()?;
            let lob: bool = t.get_item(2)?.extract()?;
            let k = match kind.as_str() {
                "logsci" => LogFormatterKind::SciNotation,
                "logmath" => LogFormatterKind::Mathtext,
                "logexp" => LogFormatterKind::Exponent,
                _ => LogFormatterKind::Plain,
            };
            let mut f = LogFormatter::new(k, base, lob);
            if let Some(mt) = t.get_item(3)?.extract::<Option<(f64, f64)>>()? {
                f.minor_thresholds = mt;
            }
            f.linthresh = t.get_item(4)?.extract()?;
            Formatter::Log(f)
        }
        "py" => Formatter::External(Arc::new(PyFormatter { obj: t.get_item(1)?.unbind() })),
        _ => return err(format!("unknown formatter {kind}")),
    })
}

// ---------------------------------------------------------------------------
// tick params

fn tickkw_from(d: &Bound<'_, PyDict>) -> PyResult<TickKw> {
    let mut kw = TickKw::default();
    for (k, v) in d.iter() {
        let key: String = k.extract()?;
        match key.as_str() {
            "size" => kw.size = Some(v.extract()?),
            "width" => kw.width = Some(v.extract()?),
            "color" => kw.color = Some(color_from(&v)?),
            "tickdir" => kw.tickdir = Some(v.extract()?),
            "pad" => kw.pad = Some(v.extract()?),
            "labelsize" => kw.labelsize = Some(v.extract()?),
            "labelcolor" => kw.labelcolor = Some(color_from(&v)?),
            "labelfontfamily" => {
                kw.labelfontfamily = Some(if let Ok(s) = v.extract::<String>() { vec![s] } else { v.extract()? })
            }
            "zorder" => kw.zorder = Some(v.extract()?),
            "gridOn" => kw.grid_on = Some(v.extract()?),
            "tick1On" => kw.tick1_on = Some(v.extract()?),
            "tick2On" => kw.tick2_on = Some(v.extract()?),
            "label1On" => kw.label1_on = Some(v.extract()?),
            "label2On" => kw.label2_on = Some(v.extract()?),
            "labelrotation" => {
                kw.labelrotation = Some(if let Ok((m, a)) = v.extract::<(String, f64)>() {
                    (m, a)
                } else if let Ok(s) = v.extract::<String>() {
                    (s, 0.0)
                } else {
                    ("default".into(), v.extract()?)
                })
            }
            "labelrotation_mode" => kw.labelrotation_mode = Some(v.extract()?),
            "grid_color" => kw.grid.color = Some(color_from(&v)?),
            "grid_alpha" => kw.grid.alpha = Some(opt_f64(&v)?),
            "grid_linestyle" => {
                if let Ok(s) = v.extract::<String>() {
                    kw.grid.linestyle = Some(s);
                } else {
                    kw.grid.dashes = Some(v.extract()?);
                }
            }
            "grid_linewidth" => kw.grid.linewidth = Some(v.extract()?),
            "grid_zorder" => kw.grid.zorder = Some(v.extract()?),
            "grid_visible" => kw.grid.visible = Some(v.extract()?),
            "grid_antialiased" => kw.grid.antialiased = Some(v.extract()?),
            _ => return Err(PyKeyError::new_err(format!("unsupported tick parameter {key}"))),
        }
    }
    Ok(kw)
}

fn gridkw_from(d: &Bound<'_, PyDict>) -> PyResult<GridKw> {
    let pref = PyDict::new(d.py());
    for (k, v) in d.iter() {
        let key: String = k.extract()?;
        pref.set_item(format!("grid_{key}"), v)?;
    }
    Ok(tickkw_from(&pref)?.grid)
}

// ---------------------------------------------------------------------------
// Figure

#[pyclass(name = "Figure", module = "matplotlib_rs._mplrs")]
pub struct PyFigure {
    pub inner: Shared<Figure>,
    /// Whether this is the figure's own handle (the one created by the
    /// Python Figure), which alone reports its Python objects to the GC.
    primary: bool,
}

fn axes_mut<'a>(f: &'a mut Figure, uid: u64) -> PyResult<&'a mut Axes> {
    f.ax(uid).ok_or_else(|| PyValueError::new_err("the Axes does not belong to this figure"))
}

impl PyFigure {
    fn lk(&self) -> std::sync::MutexGuard<'_, Figure> {
        check_not_drawing(&self.inner);
        Lock::lk(&self.inner)
    }

    fn snap(&self) -> Option<Rc<DrawSnapshot>> {
        snapshot_for(Arc::as_ptr(&self.inner) as usize)
    }

    /// Run `f` on the figure, releasing the GIL unless Python artists need it.
    fn locked_maybe_detached<R: Send>(&self, py: Python<'_>, f: impl FnOnce(&mut Figure) -> R + Send) -> PyResult<R> {
        let ext = self.lk().has_ext();
        let r = if ext {
            f(&mut self.lk())
        } else {
            py.detach(|| f(&mut self.lk()))
        };
        super::ext::take_error()?;
        Ok(r)
    }
}

#[pymethods]
impl PyFigure {
    /// Expose the Python objects held by the Rust figure (Python-drawn
    /// artists and axes, tick helpers, transforms) to the garbage collector:
    /// they usually refer back to the figure.  The figure is skipped while
    /// it is locked (e.g. being drawn), which only makes the collector more
    /// conservative.
    fn __traverse__(&self, visit: pyo3::PyVisit<'_>) -> Result<(), pyo3::PyTraverseError> {
        use crate::mpl::figure::ExtRef;
        if !self.primary {
            return Ok(());
        }
        let Ok(f) = self.inner.try_lock() else { return Ok(()) };
        let mut seen = std::collections::HashSet::new();
        let mut objs: Vec<&Py<PyAny>> = Vec::new();
        f.visit_ext(&mut |r| {
            let (holder, obj): (usize, Option<&Py<PyAny>>) = match r {
                ExtRef::Artist(e) => {
                    let o = e.as_any().downcast_ref::<super::ext::PyExt>().map(|p| &p.obj);
                    (Arc::as_ptr(e) as *const () as usize, o)
                }
                ExtRef::Locator(e) => (Arc::as_ptr(e) as *const () as usize, e.as_any().downcast_ref::<PyLocator>().map(|p| &p.obj)),
                ExtRef::Formatter(e) => {
                    (Arc::as_ptr(e) as *const () as usize, e.as_any().downcast_ref::<PyFormatter>().map(|p| &p.obj))
                }
                ExtRef::Transform(t) => (
                    Arc::as_ptr(&t.0) as *const () as usize,
                    t.0.as_any().downcast_ref::<super::ext::PyTransform>().map(|p| &p.obj),
                ),
            };
            if let Some(o) = obj {
                if seen.insert(holder) {
                    // SAFETY: the reference lives as long as the figure guard.
                    objs.push(unsafe { &*(o as *const Py<PyAny>) });
                }
            }
        });
        for o in objs {
            visit.call(o)?;
        }
        Ok(())
    }

    fn __clear__(&mut self) {
        if !self.primary {
            return;
        }
        if let Ok(mut f) = self.inner.try_lock() {
            f.release_ext();
        }
    }

    #[new]
    #[pyo3(signature = (figsize=None, dpi=None, facecolor=None, edgecolor=None, linewidth=0.0, frameon=None))]
    fn new(
        figsize: Option<(f64, f64)>,
        dpi: Option<f64>,
        facecolor: Option<&Bound<'_, PyAny>>,
        edgecolor: Option<&Bound<'_, PyAny>>,
        linewidth: f64,
        frameon: Option<bool>,
    ) -> PyResult<Self> {
        let fc = facecolor.map(color_from).transpose()?;
        let ec = edgecolor.map(color_from).transpose()?;
        let inner = shared(Figure::new(figsize, dpi, fc, ec, linewidth, frameon));
        Lock::lk(&inner).key = Arc::as_ptr(&inner) as usize;
        Ok(PyFigure { inner, primary: true })
    }

    /// add_subplot(nrows, ncols, num1, num2[, gridspec params]) -> uid
    #[pyo3(signature = (nrows, ncols, num1, num2, gs=None))]
    fn add_subplot(&self, nrows: usize, ncols: usize, num1: usize, num2: usize, gs: Option<&Bound<'_, PyDict>>) -> PyResult<u64> {
        let mut f = self.lk();
        let g = match gs {
            None => f.gridspec_for(nrows, ncols),
            Some(d) => {
                let mut spec = crate::mpl::axes::GridSpec::new(nrows, ncols);
                for (k, v) in d.iter() {
                    let key: String = k.extract()?;
                    match key.as_str() {
                        "left" => spec.left = v.extract()?,
                        "right" => spec.right = v.extract()?,
                        "bottom" => spec.bottom = v.extract()?,
                        "top" => spec.top = v.extract()?,
                        "wspace" => spec.wspace = v.extract()?,
                        "hspace" => spec.hspace = v.extract()?,
                        "width_ratios" => {
                            if !v.is_none() {
                                spec.width_ratios = v.extract()?
                            }
                        }
                        "height_ratios" => {
                            if !v.is_none() {
                                spec.height_ratios = v.extract()?
                            }
                        }
                        "id" => {}
                        _ => {}
                    }
                }
                let id: Option<usize> = d.get_item("id")?.map(|v| v.extract()).transpose()?;
                match id.and_then(|i| f.gridspecs.get(i).cloned()) {
                    Some(existing) => {
                        *existing.lk() = spec;
                        existing
                    }
                    None => {
                        let s = shared(spec);
                        f.gridspecs.push(s.clone());
                        s
                    }
                }
            }
        };
        Ok(f.add_subplot(g, num1, num2))
    }

    /// Index of a new (or updated) gridspec for later `add_subplot(gs=...)`.
    fn new_gridspec_id(&self, nrows: usize, ncols: usize) -> usize {
        let mut f = self.lk();
        f.gridspecs.push(shared(crate::mpl::axes::GridSpec::new(nrows, ncols)));
        f.gridspecs.len() - 1
    }

    fn add_axes(&self, rect: (f64, f64, f64, f64)) -> u64 {
        self.lk().add_axes_rect(rect)
    }

    fn new_axes(&self, rect: (f64, f64, f64, f64)) -> u64 {
        self.lk().new_detached(rect)
    }

    fn attach_axes(&self, uid: u64) -> bool {
        self.lk().attach(uid)
    }

    fn detach_axes(&self, uid: u64) {
        self.lk().detach(uid)
    }

    /// Reorder the attached axes to the given uid order.
    fn order_axes(&self, uids: Vec<u64>) {
        let mut f = self.lk();
        let mut ax = std::mem::take(&mut f.axes);
        let mut out = Vec::new();
        for u in &uids {
            if let Some(i) = ax.iter().position(|a| a.uid == *u) {
                out.push(ax.remove(i));
            }
        }
        out.extend(ax);
        f.axes = out;
    }

    fn delaxes(&self, uid: u64) {
        self.lk().delaxes(uid);
    }

    fn sca(&self, uid: u64) {
        self.lk().sca(uid);
    }

    fn gca(&self) -> Option<u64> {
        self.lk().gca_uid()
    }

    fn axes_uids(&self) -> Vec<u64> {
        self.lk().axes.iter().map(|a| a.uid).collect()
    }

    fn clear(&self) {
        let mut f = self.lk();
        f.axes.clear();
        f.axstack.clear();
        f.children.clear();
        f.subfigs.clear();
        f.root_subfigs.clear();
        f.suptitle = None;
        f.supxlabel = None;
        f.supylabel = None;
        f.gridspecs.clear();
    }

    #[getter]
    fn get_dpi(&self) -> f64 {
        if let Some(s) = self.snap() {
            return s.fig.dpi;
        }
        self.lk().dpi
    }

    #[setter]
    fn set_dpi(&self, dpi: f64) {
        self.lk().dpi = dpi;
    }

    fn get_size_inches(&self) -> (f64, f64) {
        if let Some(s) = self.snap() {
            return s.figsize;
        }
        self.lk().figsize
    }

    fn set_size_inches(&self, w: f64, h: f64) {
        self.lk().figsize = (w, h);
    }

    fn patch_set(&self, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        patch_set(&mut self.lk().patch, name, value)
    }

    fn patch_get<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
        patch_get(py, &self.lk().patch, name)
    }

    #[pyo3(signature = (left=None, bottom=None, right=None, top=None, wspace=None, hspace=None))]
    fn subplots_adjust(&self, left: Option<f64>, bottom: Option<f64>, right: Option<f64>, top: Option<f64>, wspace: Option<f64>, hspace: Option<f64>) {
        self.lk().subplots_adjust(left, bottom, right, top, wspace, hspace);
    }

    fn get_subplotpars(&self) -> (f64, f64, f64, f64, f64, f64) {
        let p = self.lk().subplotpars;
        (p.left, p.bottom, p.right, p.top, p.wspace, p.hspace)
    }

    /// Add a figure-level artist: a text, line or patch handle, or any
    /// Python artist (drawn by its own `draw`), with its `get_children`
    /// category (0 artists, 2 lines, 3 patches, 4 texts, 5 images, 6 legends).
    #[pyo3(signature = (a, category=0, subfig=None))]
    fn add_artist(&self, a: &Bound<'_, PyAny>, category: u8, subfig: Option<u64>) -> PyResult<()> {
        let c = if let Ok(t) = a.cast::<PyText>() {
            FigChild::Text(t.borrow().inner.clone())
        } else if let Ok(l) = a.cast::<PyLine2D>() {
            FigChild::Line(l.borrow().inner.clone())
        } else if let Ok(p) = a.cast::<PyPatch>() {
            FigChild::Patch(p.borrow().inner.clone())
        } else {
            FigChild::Ext(super::ext::ext_from(a), category)
        };
        let mut f = self.lk();
        let children = f.children_of_mut(subfig).ok_or_else(|| PyValueError::new_err("no such subfigure"))?;
        children.push(c);
        Ok(())
    }

    /// `add_subfigure`: a new subfigure of the figure or of subfigure
    /// *parent*; returns its id.
    #[pyo3(signature = (parent=None))]
    fn new_subfigure(&self, parent: Option<u64>) -> u64 {
        self.lk().new_subfig(parent)
    }

    fn remove_subfigure(&self, uid: u64) {
        self.lk().remove_subfig(uid)
    }

    fn subfig_set_patch(&self, uid: u64, patch: &PyPatch) -> PyResult<()> {
        let p = patch.inner.clone();
        let mut f = self.lk();
        let s = f.subfig_mut(uid).ok_or_else(|| PyValueError::new_err("no such subfigure"))?;
        s.patch = Some(p);
        Ok(())
    }

    fn subfig_state(&self, uid: u64, zorder: f64, visible: bool) -> PyResult<()> {
        let mut f = self.lk();
        let s = f.subfig_mut(uid).ok_or_else(|| PyValueError::new_err("no such subfigure"))?;
        s.zorder = zorder;
        s.visible = visible;
        Ok(())
    }

    /// Make axes *uid* a local axes of subfigure *sf*.
    fn subfig_add_axes(&self, sf: u64, uid: u64) -> bool {
        self.lk().subfig_add_axes(sf, uid)
    }

    /// Replace figure child `old` by `new` at the same position.
    #[pyo3(signature = (old, new, subfig=None))]
    fn replace_artist(&self, old: &Bound<'_, PyAny>, new: &Bound<'_, PyAny>, subfig: Option<u64>) -> PyResult<bool> {
        let key = |o: &Bound<'_, PyAny>| -> usize {
            if let Ok(t) = o.cast::<PyText>() {
                Arc::as_ptr(&t.borrow().inner) as usize
            } else if let Ok(l) = o.cast::<PyLine2D>() {
                Arc::as_ptr(&l.borrow().inner) as usize
            } else if let Ok(p) = o.cast::<PyPatch>() {
                Arc::as_ptr(&p.borrow().inner) as usize
            } else {
                o.as_ptr() as usize
            }
        };
        let ok = key(old);
        let mut f = self.lk();
        let Some(children) = f.children_of_mut(subfig) else { return Ok(false) };
        let pos = children.iter().position(|c| match c {
            FigChild::Text(x) => Arc::as_ptr(x) as usize == ok,
            FigChild::Line(x) => Arc::as_ptr(x) as usize == ok,
            FigChild::Patch(x) => Arc::as_ptr(x) as usize == ok,
            FigChild::Ext(e, _) => e.ident() == ok,
            FigChild::Legend(_) => false,
        });
        let Some(i) = pos else { return Ok(false) };
        let cat = match &children[i] {
            FigChild::Ext(_, c) => *c,
            FigChild::Line(_) => 2,
            FigChild::Patch(_) => 3,
            _ => 4,
        };
        children[i] = if let Ok(t) = new.cast::<PyText>() {
            FigChild::Text(t.borrow().inner.clone())
        } else if let Ok(l) = new.cast::<PyLine2D>() {
            FigChild::Line(l.borrow().inner.clone())
        } else if let Ok(p) = new.cast::<PyPatch>() {
            FigChild::Patch(p.borrow().inner.clone())
        } else {
            FigChild::Ext(super::ext::ext_from(new), cat)
        };
        Ok(true)
    }

    #[pyo3(signature = (a, subfig=None))]
    fn remove_artist(&self, a: &Bound<'_, PyAny>, subfig: Option<u64>) -> PyResult<bool> {
        let mut f = self.lk();
        let Some(children) = f.children_of_mut(subfig) else { return Ok(false) };
        let before = children.len();
        if let Ok(t) = a.cast::<PyText>() {
            let p = t.borrow().inner.clone();
            children.retain(|c| !matches!(c, FigChild::Text(x) if Arc::ptr_eq(x, &p)));
        } else if let Ok(l) = a.cast::<PyLine2D>() {
            let p = l.borrow().inner.clone();
            children.retain(|c| !matches!(c, FigChild::Line(x) if Arc::ptr_eq(x, &p)));
        } else if let Ok(pp) = a.cast::<PyPatch>() {
            let p = pp.borrow().inner.clone();
            children.retain(|c| !matches!(c, FigChild::Patch(x) if Arc::ptr_eq(x, &p)));
        } else {
            let id = a.as_ptr() as usize;
            children.retain(|c| !matches!(c, FigChild::Ext(e, _) if e.ident() == id));
        }
        Ok(children.len() != before)
    }

    /// Render and return the RGBA buffer (h, w, 4).
    fn render<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<u8>>> {
        let r = self.locked_maybe_detached(py, |f| f.render())?;
        let (w, h) = (r.agg.width as usize, r.agg.height as usize);
        let arr = numpy::PyArray1::from_slice(py, r.agg.buffer());
        Ok(arr.reshape([h, w, 4])?)
    }

    /// Render and write a PNG file; `bbox` (inches, x0, y0, x1, y1) crops.
    #[pyo3(signature = (path, dpi=None, facecolor=None, edgecolor=None, bbox=None, tight=None, metadata=None, compression=None))]
    #[allow(clippy::too_many_arguments)]
    fn savefig_png(
        &self,
        py: Python<'_>,
        path: &str,
        dpi: Option<f64>,
        facecolor: Option<&Bound<'_, PyAny>>,
        edgecolor: Option<&Bound<'_, PyAny>>,
        bbox: Option<(f64, f64, f64, f64)>,
        tight: Option<f64>,
        metadata: Option<Vec<(String, String)>>,
        compression: Option<u8>,
    ) -> PyResult<()> {
        let fc = facecolor.map(color_from).transpose()?;
        let ec = edgecolor.map(color_from).transpose()?;
        let meta = metadata.unwrap_or_default();
        let path = path.to_string();
        let buf = self.locked_maybe_detached(py, |f| render_for_save(f, dpi, fc, ec, bbox, tight))?;
        py.detach(|| -> PyResult<()> {
            write_png(&path, &buf.0, buf.1, buf.2, buf.3, &meta, compression.unwrap_or(1))
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })
    }

    /// Render for saving and return (rgba bytes, width, height).
    #[pyo3(signature = (dpi=None, facecolor=None, edgecolor=None, bbox=None, tight=None))]
    fn render_for_save<'py>(
        &self,
        py: Python<'py>,
        dpi: Option<f64>,
        facecolor: Option<&Bound<'py, PyAny>>,
        edgecolor: Option<&Bound<'py, PyAny>>,
        bbox: Option<(f64, f64, f64, f64)>,
        tight: Option<f64>,
    ) -> PyResult<Bound<'py, PyArray3<u8>>> {
        let fc = facecolor.map(color_from).transpose()?;
        let ec = edgecolor.map(color_from).transpose()?;
        let (buf, w, h, _) = self.locked_maybe_detached(py, |f| render_for_save(f, dpi, fc, ec, bbox, tight))?;
        let arr = numpy::PyArray1::from_vec(py, buf);
        Ok(arr.reshape([h as usize, w as usize, 4])?)
    }

    /// `get_tightbbox()` in inches: (x0, y0, x1, y1).
    #[pyo3(signature = (dpi=None))]
    fn get_tightbbox(&self, dpi: Option<f64>) -> PyResult<(f64, f64, f64, f64)> {
        let b = {
            let mut f = self.lk();
            let old = f.dpi;
            if let Some(d) = dpi {
                f.dpi = d;
            }
            let (w, h) = f.canvas_size();
            let mut r = Renderer::metrics_only(w as f64, h as f64, f.dpi);
            for i in 0..f.axes.len() {
                f.unstale_viewlim(i);
            }
            let b = f.get_tightbbox(&mut r);
            f.dpi = old;
            b
        };
        super::ext::take_error()?;
        Ok((b.x0, b.y0, b.x1, b.y1))
    }

    /// `draw_without_rendering`: run a disabled draw (layout, aspect, ticks).
    fn draw_disabled(&self) -> PyResult<()> {
        self.lk().draw_disabled();
        super::ext::take_error()
    }

    /// `Figure.draw(renderer)` into a Python-side `RendererAgg`.
    fn draw_into(&self, target: &Bound<'_, super::lowlevel::PyRendererAgg>) -> PyResult<()> {
        {
            let mut t = target.borrow_mut();
            let r = t.r()?;
            self.lk().draw(r);
        }
        super::ext::take_error()
    }

    /// `Figure.draw(renderer)` into any other Python renderer (matplotlib's
    /// vector backends): drawing calls and text metrics are forwarded to it.
    fn draw_foreign(&self, renderer: &Bound<'_, PyAny>) -> PyResult<()> {
        let fr = std::sync::Arc::new(super::foreign::PyForeign::new(renderer)?);
        {
            let mut f = self.lk();
            let (w, h) = f.canvas_size();
            let mut r = Renderer::foreign(fr, w as f64, h as f64, f.dpi);
            f.draw(&mut r);
        }
        super::ext::take_error()
    }

    /// Mirror the figure geometry owned by the Python layer (size in inches,
    /// dpi and the `bbox_inches` adjustment of `transFigure`).
    #[pyo3(signature = (figsize, dpi, boxout=None))]
    fn mirror_state(&self, figsize: (f64, f64), dpi: f64, boxout: Option<(f64, f64, f64, f64)>) {
        let mut f = self.lk();
        f.figsize = figsize;
        f.dpi = dpi;
        f.boxout = boxout.map(|b| Bbox::new(b.0, b.1, b.2, b.3));
        f.layout = LayoutEngine::None;
    }

    /// `canvas.is_saving()` for the next draws (unchanged while one runs).
    fn set_saving(&self, saving: bool) {
        if self.snap().is_none() {
            self.lk().saving = saving;
        }
    }

    #[pyo3(signature = (kind, pad=1.08, h_pad=None, w_pad=None, rect=None))]
    fn set_layout(&self, kind: &str, pad: f64, h_pad: Option<f64>, w_pad: Option<f64>, rect: Option<(f64, f64, f64, f64)>) {
        self.lk().layout = match kind {
            "tight" => LayoutEngine::Tight { pad, h_pad, w_pad, rect },
            "constrained" => LayoutEngine::Constrained,
            _ => LayoutEngine::None,
        };
    }

    fn get_layout(&self) -> String {
        match self.lk().layout {
            LayoutEngine::None => "none".into(),
            LayoutEngine::Tight { .. } => "tight".into(),
            LayoutEngine::Constrained => "constrained".into(),
        }
    }

    /// Run tight_layout once now.
    #[pyo3(signature = (pad=1.08, h_pad=None, w_pad=None, rect=None))]
    fn tight_layout(&self, pad: f64, h_pad: Option<f64>, w_pad: Option<f64>, rect: Option<(f64, f64, f64, f64)>) {
        let mut f = self.lk();
        let (w, h) = f.canvas_size();
        let mut r = Renderer::metrics_only(w as f64, h as f64, f.dpi);
        for i in 0..f.axes.len() {
            f.unstale_viewlim(i);
        }
        crate::mpl::tight_layout::tight_layout(&mut f, &mut r, pad, h_pad, w_pad, rect);
    }

    fn axes(&self, uid: u64) -> PyAxes {
        PyAxes { fig: self.inner.clone(), uid }
    }

    /// Display coordinates of a figure-level text's extent.
    fn text_extent(&self, t: &PyText) -> (f64, f64, f64, f64) {
        if let Some(s) = self.snap() {
            let mut r = Renderer::metrics_only(s.canvas.0, s.canvas.1, s.fig.dpi);
            let b = t.lk().get_window_extent(&mut r, &s.tc);
            return (b.x0, b.y0, b.x1, b.y1);
        }
        let f = self.lk();
        let tc = f.trans_ctx();
        let (w, h) = f.canvas_size();
        let mut r = Renderer::metrics_only(w as f64, h as f64, f.dpi);
        let b = t.inner.lk().get_window_extent(&mut r, &tc);
        (b.x0, b.y0, b.x1, b.y1)
    }

    /// The figure's transFigure matrix rows.
    fn trans_figure(&self) -> Vec<Vec<f64>> {
        let m = match self.snap() {
            Some(s) => s.fig.trans_figure.get_affine(),
            None => self.lk().ctx().trans_figure.get_affine(),
        };
        m.iter().map(|r| r.to_vec()).collect()
    }

    /// Share an axis between two axes (x if `x`).
    fn share(&self, a: u64, b: u64, x: bool) -> PyResult<()> {
        let mut f = self.lk();
        let ia = f.axes_index(a).ok_or_else(|| PyValueError::new_err("no axes"))?;
        let ib = f.axes_index(b).ok_or_else(|| PyValueError::new_err("no axes"))?;
        let mut group: Vec<u64> = if x { f.axes[ib].shared_x.clone() } else { f.axes[ib].shared_y.clone() };
        if group.is_empty() {
            group.push(b);
        }
        let ga: Vec<u64> = if x { f.axes[ia].shared_x.clone() } else { f.axes[ia].shared_y.clone() };
        for u in ga {
            if !group.contains(&u) {
                group.push(u);
            }
        }
        if !group.contains(&a) {
            group.push(a);
        }
        for ax in f.axes.iter_mut() {
            if group.contains(&ax.uid) {
                if x {
                    ax.shared_x = group.clone();
                } else {
                    ax.shared_y = group.clone();
                }
            }
        }
        // follow the other's axis state (locators, scale, limits)
        let (src_axis, lim, auto) = {
            let o = &f.axes[ib];
            let ax = if x { &o.xaxis } else { &o.yaxis };
            (ax.clone(), o.get_lim(x), ax.autoscale_on)
        };
        let me = &mut f.axes[ia];
        {
            let ax = if x { &mut me.xaxis } else { &mut me.yaxis };
            ax.major_locator = src_axis.major_locator.clone();
            ax.minor_locator = src_axis.minor_locator.clone();
            ax.major_formatter = src_axis.major_formatter.clone();
            ax.minor_formatter = src_axis.minor_formatter.clone();
            ax.scale = src_axis.scale.clone();
        }
        me.set_lim(x, lim.0, lim.1, Some(auto));
        Ok(())
    }

    fn twin(&self, a: u64, x: bool) -> PyResult<u64> {
        // twinx: shares x, y on the right
        let mut f = self.lk();
        let i = f.axes_index(a).ok_or_else(|| PyValueError::new_err("no axes"))?;
        let (pos, ss) = (f.axes[i].original_position, f.axes[i].subplotspec.clone());
        let sp = f.subplotpars;
        let mut ax = Axes::new(pos, ss, &sp);
        ax.position = f.axes[i].position;
        ax.original_position = pos;
        let uid = ax.uid;
        f.add_axes_internal(ax);
        drop(f);
        self.share(uid, a, x)?;
        let mut f = self.lk();
        let ib = f.axes_index(uid).unwrap();
        let ia = f.axes_index(a).unwrap();
        let tw: Vec<u64> = {
            let mut t = f.axes[ia].twinned.clone();
            if t.is_empty() {
                t.push(a);
            }
            t.push(uid);
            t
        };
        for ax in f.axes.iter_mut() {
            if tw.contains(&ax.uid) {
                ax.twinned = tw.clone();
            }
        }
        let new = &mut f.axes[ib];
        if x {
            // twinx: ticks of the y axis on the right
            new.yaxis.label_position = "right".into();
            new.yaxis.tick_position = "right".into();
            let kw = TickKw { tick1_on: Some(false), tick2_on: Some(true), label1_on: Some(false), label2_on: Some(true), ..TickKw::default() };
            new.yaxis.set_tick_params("both", false, &kw);
            new.yaxis.offset_text.lk().ha = "right".into();
            new.yaxis.offset_text.lk().x = 1.0;
            {
                let mut l = new.yaxis.label.lk();
                l.va = "top".into();
                l.rotation_mode = "anchor".into();
            }
            new.xaxis.visible = false;
        } else {
            new.xaxis.label_position = "top".into();
            new.xaxis.tick_position = "top".into();
            let kw = TickKw { tick1_on: Some(false), tick2_on: Some(true), label1_on: Some(false), label2_on: Some(true), ..TickKw::default() };
            new.xaxis.set_tick_params("both", false, &kw);
            new.xaxis.offset_text.lk().va = "bottom".into();
            new.xaxis.label.lk().va = "baseline".into();
            new.yaxis.visible = false;
        }
        new.patch.base.visible = false;
        Ok(uid)
    }

    fn unique_id(&self) -> usize {
        Arc::as_ptr(&self.inner) as usize
    }
}

/// Render the figure for saving with temporary dpi/face/edge colours,
/// `bbox` (inches) or tight bbox with `pad` inches.
fn render_for_save(f: &mut Figure, dpi: Option<f64>, fc: Option<ColorSpec>, ec: Option<ColorSpec>, bbox: Option<(f64, f64, f64, f64)>, tight: Option<f64>) -> (Vec<u8>, u32, u32, f64) {
    let old_dpi = f.dpi;
    let old_patch = f.patch.clone();
    if let Some(d) = dpi {
        f.dpi = d;
    }
    if let Some(c) = fc {
        f.patch.set_facecolor(Some(c));
    }
    if let Some(c) = ec {
        f.patch.set_edgecolor(Some(c));
    }
    let r = f.render_for_save(bbox.map(|b| Bbox::new(b.0, b.1, b.2, b.3)), tight);
    let d = f.dpi;
    f.dpi = old_dpi;
    f.patch = old_patch;
    (r.agg.buffer().to_vec(), r.agg.width, r.agg.height, d)
}

/// Write an RGBA buffer as PNG with matplotlib's metadata keys.
pub fn write_png(path: &str, rgba: &[u8], w: u32, h: u32, dpi: f64, meta: &[(String, String)], level: u8) -> std::io::Result<()> {
    let file = std::fs::File::create(path)?;
    let bw = std::io::BufWriter::new(file);
    let mut enc = png::Encoder::new(bw, w, h);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    enc.set_compression(match level {
        0 => png::Compression::NoCompression,
        1 => png::Compression::Fastest,
        2 => png::Compression::Fast,
        3 => png::Compression::Balanced,
        _ => png::Compression::High,
    });
    let ppm = (dpi / 0.0254).round() as u32;
    enc.set_pixel_dims(Some(png::PixelDimensions { xppu: ppm, yppu: ppm, unit: png::Unit::Meter }));
    for (k, v) in meta {
        enc.add_text_chunk(k.clone(), v.clone()).map_err(std::io::Error::other)?;
    }
    let mut writer = enc.write_header().map_err(std::io::Error::other)?;
    writer.write_image_data(rgba).map_err(std::io::Error::other)?;
    writer.finish().map_err(std::io::Error::other)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Axes

#[pyclass(name = "Axes", module = "matplotlib_rs._mplrs")]
pub struct PyAxes {
    pub fig: Shared<Figure>,
    pub uid: u64,
}

fn axis_of(x: &str) -> PyResult<bool> {
    match x {
        "x" => Ok(true),
        "y" => Ok(false),
        _ => err(format!("unknown axis {x}")),
    }
}

/// Fail loudly instead of deadlocking when Python code running inside a
/// draw of `f` (an external artist) tries to access the locked figure.
fn check_not_drawing(f: &Shared<Figure>) {
    if snapshot_for(Arc::as_ptr(f) as usize).is_some() {
        if std::env::var_os("MPLRS_DEBUG").is_some() {
            Python::attach(|py| {
                let _ = py.import("traceback").and_then(|t| t.call_method0("print_stack"));
            });
        }
        panic!("matplotlib_rs: this figure's state cannot be changed while the figure is being drawn");
    }
}

impl PyAxes {
    fn lk(&self) -> std::sync::MutexGuard<'_, Figure> {
        check_not_drawing(&self.fig);
        Lock::lk(&self.fig)
    }

    /// The draw-time snapshot of this axes, if its figure is being drawn.
    fn snap(&self) -> Option<(Rc<DrawSnapshot>, usize)> {
        let s = snapshot_for(Arc::as_ptr(&self.fig) as usize)?;
        let i = s.axes.iter().position(|a| a.uid == self.uid)?;
        Some((s, i))
    }

    /// This axes if it is a child axes lent to the Python draw of its parent.
    fn lent(&self) -> Option<crate::mpl::figure::LentAxes> {
        crate::mpl::figure::lent_take(Arc::as_ptr(&self.fig) as usize, self.uid)
    }

    fn with<R>(&self, f: impl FnOnce(&mut Axes, &crate::mpl::axes::FigCtx) -> R) -> PyResult<R> {
        if let Some(mut l) = self.lent() {
            let ctx = l.fig.clone();
            return Ok(f(l.axes(), &ctx));
        }
        let mut fig = self.lk();
        let ctx = fig.ctx();
        let a = axes_mut(&mut fig, self.uid)?;
        Ok(f(a, &ctx))
    }

    fn unstaled<R>(&self, f: impl FnOnce(&mut Axes, &crate::mpl::axes::FigCtx) -> R) -> PyResult<R> {
        if let Some(mut l) = self.lent() {
            let ctx = l.fig.clone();
            let a = l.axes();
            a.unstale_viewlim(&ctx);
            return Ok(f(a, &ctx));
        }
        let mut fig = self.lk();
        if let Some(i) = fig.axes_index(self.uid) {
            fig.unstale_viewlim(i);
        }
        let ctx = fig.ctx();
        let a = axes_mut(&mut fig, self.uid)?;
        Ok(f(a, &ctx))
    }

    fn after_lim_change(&self) -> PyResult<()> {
        if self.lent().is_some() {
            return Ok(());
        }
        let mut fig = self.lk();
        if let Some(i) = fig.axes_index(self.uid) {
            fig.sync_shared(i);
        }
        Ok(())
    }
}

fn child_from(obj: &Bound<'_, PyAny>) -> PyResult<Child> {
    if let Ok(l) = obj.cast::<PyLine2D>() {
        return Ok(Child::Line(l.borrow().inner.clone()));
    }
    if let Ok(p) = obj.cast::<PyPatch>() {
        return Ok(Child::Patch(p.borrow().inner.clone()));
    }
    if let Ok(t) = obj.cast::<PyText>() {
        return Ok(Child::Text(t.borrow().inner.clone()));
    }
    if let Ok(c) = obj.cast::<super::collections::PyCollection>() {
        return Ok(Child::Collection(c.borrow().inner.clone()));
    }
    if let Ok(c) = obj.cast::<super::collections::PyImage>() {
        return Ok(Child::Image(c.borrow().inner.clone()));
    }
    // any other object is a Python artist drawn by its own `draw`
    Ok(Child::Ext(super::ext::ext_from(obj)))
}

#[pymethods]
impl PyAxes {
    #[getter]
    fn uid(&self) -> u64 {
        self.uid
    }

    fn add_line(&self, line: &PyLine2D) -> PyResult<()> {
        self.with(|a, _| a.add_line(line.inner.clone()))
    }

    fn add_patch(&self, p: &PyPatch) -> PyResult<()> {
        self.with(|a, _| a.add_patch(p.inner.clone()))
    }

    fn add_text(&self, t: &PyText) -> PyResult<()> {
        self.with(|a, _| a.add_text(t.inner.clone()))
    }

    #[pyo3(signature = (c, autolim=true))]
    fn add_collection(&self, c: &super::collections::PyCollection, autolim: bool) -> PyResult<()> {
        let mut fig = self.lk();
        let i = fig.axes_index(self.uid).ok_or_else(|| PyValueError::new_err("no such axes"))?;
        fig.unstale_viewlim(i);
        let ctx = fig.ctx();
        fig.axes[i].add_collection(c.inner.clone(), autolim, &ctx);
        Ok(())
    }

    fn add_image(&self, im: &super::collections::PyImage) -> PyResult<()> {
        self.with(|a, _| a.add_image(im.inner.clone()))
    }

    /// Append an artist without the add_* side effects (zorder order kept).
    fn add_child_raw(&self, obj: &Bound<'_, PyAny>) -> PyResult<()> {
        let c = child_from(obj)?;
        self.with(|a, _| a.children.push(c))
    }

    fn remove_child(&self, obj: &Bound<'_, PyAny>) -> PyResult<bool> {
        let c = child_from(obj)?;
        self.with(|a, _| {
            let n = a.children.len();
            a.children.retain(|x| !x.same(&c));
            n != a.children.len()
        })
    }

    /// Replace child `old` by `new` at the same position (e.g. when an
    /// artist switches between native and Python drawing).
    fn replace_child(&self, old: &Bound<'_, PyAny>, new: &Bound<'_, PyAny>) -> PyResult<bool> {
        let o = child_from(old)?;
        let n = child_from(new)?;
        self.with(|a, _| match a.children.iter().position(|x| x.same(&o)) {
            Some(i) => {
                a.children[i] = n;
                true
            }
            None => false,
        })
    }

    /// Kinds of children in order ("line", "patch", "text", "collection", "image").
    fn children(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        let ch = self.with(|a, _| a.children.clone())?;
        let mut out = Vec::new();
        for c in ch {
            let o: Py<PyAny> = match c {
                Child::Line(l) => Py::new(py, PyLine2D { inner: l })?.into_any(),
                Child::Patch(p) => Py::new(py, PyPatch { inner: p })?.into_any(),
                Child::Text(t) => Py::new(py, PyText { inner: t })?.into_any(),
                Child::Collection(c) => Py::new(py, super::collections::PyCollection { inner: c })?.into_any(),
                Child::Image(c) => Py::new(py, super::collections::PyImage { inner: c })?.into_any(),
                Child::Ext(e) => match super::ext::py_object(&e) {
                    Some(o) => o.clone_ref(py),
                    None => continue,
                },
            };
            out.push(o);
        }
        Ok(out)
    }

    #[pyo3(signature = (x=true, y=true, tight=None))]
    fn request_autoscale_view(&self, x: bool, y: bool, tight: Option<bool>) -> PyResult<()> {
        self.with(|a, _| a.request_autoscale_view(x, y, tight))
    }

    #[pyo3(signature = (tight=None, scalex=true, scaley=true))]
    fn autoscale_view(&self, tight: Option<bool>, scalex: bool, scaley: bool) -> PyResult<()> {
        {
            let mut fig = self.lk();
            let i = fig.axes_index(self.uid).ok_or_else(|| PyValueError::new_err("no such axes"))?;
            fig.unstale_viewlim(i);
            let ctx = fig.ctx();
            let shared = {
                let ax = &fig.axes[i];
                if ax.shared_x.is_empty() && ax.shared_y.is_empty() {
                    None
                } else {
                    Some(())
                }
            };
            let _ = shared;
            fig.axes[i].autoscale_view(tight, scalex, scaley, &ctx, None);
        }
        self.after_lim_change()
    }

    #[pyo3(signature = (visible_only=false))]
    fn relim(&self, visible_only: bool) -> PyResult<()> {
        self.with(|a, c| a.relim(visible_only, c))
    }

    fn get_lim(&self, axis: &str) -> PyResult<(f64, f64)> {
        let x = axis_of(axis)?;
        if let Some((s, i)) = self.snap() {
            let v = s.axes[i].view_lim;
            return Ok(if x { (v.x0, v.x1) } else { (v.y0, v.y1) });
        }
        self.unstaled(|a, _| a.get_lim(x))
    }

    #[pyo3(signature = (axis, v0, v1, auto=Some(false)))]
    fn set_lim(&self, axis: &str, v0: Option<f64>, v1: Option<f64>, auto: Option<bool>) -> PyResult<(f64, f64)> {
        let x = axis_of(axis)?;
        let r = self.unstaled(|a, _| a.set_lim(x, v0.unwrap_or(f64::NAN), v1.unwrap_or(f64::NAN), auto))?;
        self.after_lim_change()?;
        Ok(r)
    }

    #[pyo3(signature = (axis, lower=None, upper=None))]
    fn set_bound(&self, axis: &str, lower: Option<f64>, upper: Option<f64>) -> PyResult<()> {
        let x = axis_of(axis)?;
        self.unstaled(|a, _| a.set_bound(x, lower, upper))?;
        self.after_lim_change()
    }

    fn get_autoscale_on(&self, axis: &str) -> PyResult<bool> {
        let x = axis_of(axis)?;
        self.with(|a, _| a.axis(x).autoscale_on)
    }

    fn set_autoscale_on(&self, axis: &str, b: bool) -> PyResult<()> {
        let x = axis_of(axis)?;
        self.with(|a, _| a.axis_mut(x).autoscale_on = b)
    }

    fn get_margin(&self, axis: &str) -> PyResult<f64> {
        let x = axis_of(axis)?;
        self.with(|a, _| if x { a.xmargin } else { a.ymargin })
    }

    fn set_margin(&self, axis: &str, m: f64) -> PyResult<()> {
        let x = axis_of(axis)?;
        self.with(|a, _| {
            if x {
                a.xmargin = m;
            } else {
                a.ymargin = m;
            }
            a.request_autoscale_view(x, !x, None);
        })
    }

    fn set_tight(&self, t: Option<bool>) -> PyResult<()> {
        self.with(|a, _| a.tight = t)
    }

    fn get_tight(&self) -> PyResult<Option<bool>> {
        self.with(|a, _| a.tight)
    }

    fn set_use_sticky_edges(&self, b: bool) -> PyResult<()> {
        self.with(|a, _| a.use_sticky_edges = b)
    }

    /// set_scale(axis, name, base, subs, linthresh, linscale, nonpositive_clip,
    /// linear_width, custom_id); name "custom" is a Python scale whose transform
    /// the scale hook evaluates by *custom_id*.
    #[pyo3(signature = (axis, name, base=10.0, subs=None, linthresh=2.0, linscale=1.0, clip=true, linear_width=1.0, custom_id=0))]
    #[allow(clippy::too_many_arguments)]
    fn set_scale(&self, axis: &str, name: &str, base: f64, subs: Option<Vec<f64>>, linthresh: f64, linscale: f64, clip: bool, linear_width: f64, custom_id: u64) -> PyResult<()> {
        let x = axis_of(axis)?;
        let mut sc = if name == "custom" {
            Scale::Custom { id: custom_id }
        } else {
            scale_from_name(name).ok_or_else(|| PyValueError::new_err(format!("unknown scale {name}")))?
        };
        match &mut sc {
            Scale::Log { base: b, subs: s, clip: c } => {
                *b = base;
                *s = subs;
                *c = clip;
            }
            Scale::Symlog { base: b, linthresh: lt, linscale: ls, subs: s } => {
                *b = base;
                *lt = linthresh;
                *ls = linscale;
                *s = subs;
            }
            Scale::Logit { clip: c } => *c = clip,
            Scale::Asinh { linear_width: lw } => *lw = linear_width,
            Scale::Linear | Scale::Custom { .. } => {}
        }
        // The Python layer applies the scale to shared axes and autoscales.
        self.with(|a, _| {
            let ax = a.axis_mut(x);
            ax.set_scale(sc);
            ax.reset_ticks();
        })
    }

    fn get_scale(&self, axis: &str) -> PyResult<String> {
        let x = axis_of(axis)?;
        self.with(|a, _| a.axis(x).scale.name().to_string())
    }

    fn get_datalim(&self) -> PyResult<(f64, f64, f64, f64, f64, f64)> {
        if let Some((s, i)) = self.snap() {
            let d = s.axes[i].data_lim;
            let b = d.bbox;
            return Ok((b.x0, b.y0, b.x1, b.y1, d.minpos[0], d.minpos[1]));
        }
        self.with(|a, _| {
            let b = a.data_lim.bbox;
            (b.x0, b.y0, b.x1, b.y1, a.data_lim.minpos[0], a.data_lim.minpos[1])
        })
    }

    /// Replace the data limits (and their minpos when given).
    #[pyo3(signature = (x0, y0, x1, y1, minposx=None, minposy=None))]
    #[allow(clippy::too_many_arguments)]
    fn set_datalim(&self, x0: f64, y0: f64, x1: f64, y1: f64, minposx: Option<f64>, minposy: Option<f64>) -> PyResult<()> {
        self.with(|a, _| {
            a.data_lim.bbox = Bbox::new(x0, y0, x1, y1);
            if let Some(m) = minposx {
                a.data_lim.minpos[0] = m;
            }
            if let Some(m) = minposy {
                a.data_lim.minpos[1] = m;
            }
        })
    }

    /// Mirror the state the Python layer owns: active and original
    /// position, view limits and data limits (with minpos); the Rust axes
    /// then neither autoscales nor applies its aspect by itself.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (position, original, viewlim, datalim, minpos, zorder=0.0, visible=true,
                        axison=true, frameon=true, autotitlepos=None, rasterization_zorder=None))]
    fn mirror_state(
        &self,
        position: (f64, f64, f64, f64),
        original: (f64, f64, f64, f64),
        viewlim: (f64, f64, f64, f64),
        datalim: (f64, f64, f64, f64),
        minpos: (f64, f64),
        zorder: f64,
        visible: bool,
        axison: bool,
        frameon: bool,
        autotitlepos: Option<bool>,
        rasterization_zorder: Option<f64>,
    ) -> PyResult<()> {
        self.with(|a, _| {
            a.rasterization_zorder = rasterization_zorder;
            a.base.zorder = zorder;
            a.base.visible = visible;
            a.axison = axison;
            a.frameon = frameon;
            a.autotitlepos = autotitlepos;
            a.position = Bbox::new(position.0, position.1, position.2, position.3);
            a.original_position = Bbox::new(original.0, original.1, original.2, original.3);
            a.view_lim = Bbox::new(viewlim.0, viewlim.1, viewlim.2, viewlim.3);
            a.data_lim.bbox = Bbox::new(datalim.0, datalim.1, datalim.2, datalim.3);
            a.data_lim.minpos = [minpos.0, minpos.1];
            a.stale_viewlims = [false, false];
            a.aspect_frozen = true;
        })
    }

    fn set_aspect_frozen(&self, b: bool) -> PyResult<()> {
        self.with(|a, _| a.aspect_frozen = b)
    }

    #[pyo3(signature = (xys, updatex=true, updatey=true))]
    fn update_datalim(&self, xys: Vec<[f64; 2]>, updatex: bool, updatey: bool) -> PyResult<()> {
        self.with(|a, _| a.update_datalim(&xys, updatex, updatey))
    }

    fn get_viewlim(&self) -> PyResult<(f64, f64, f64, f64)> {
        if let Some((s, i)) = self.snap() {
            let b = s.axes[i].view_lim;
            return Ok((b.x0, b.y0, b.x1, b.y1));
        }
        self.unstaled(|a, _| {
            let b = a.view_lim;
            (b.x0, b.y0, b.x1, b.y1)
        })
    }

    /// The view limits without triggering a pending autoscale.
    fn get_viewlim_raw(&self) -> PyResult<(f64, f64, f64, f64)> {
        if let Some((s, i)) = self.snap() {
            let b = s.axes[i].view_lim;
            return Ok((b.x0, b.y0, b.x1, b.y1));
        }
        self.with(|a, _| {
            let b = a.view_lim;
            (b.x0, b.y0, b.x1, b.y1)
        })
    }

    /// Whether the view limits of the x / y axis await an autoscale.
    fn get_stale_viewlims(&self) -> PyResult<(bool, bool)> {
        self.with(|a, _| (a.stale_viewlims[0], a.stale_viewlims[1]))
    }

    fn ignore_existing_data_limits(&self, b: bool) -> PyResult<()> {
        self.with(|a, _| a.ignore_existing_data_limits = b)
    }

    /// Title text handle for loc in left/center/right.
    fn title(&self, loc: &str) -> PyResult<PyText> {
        self.with(|a, _| {
            let t = match loc {
                "left" => a.left_title.clone(),
                "right" => a.right_title.clone(),
                _ => a.title.clone(),
            };
            PyText { inner: t }
        })
    }

    /// Use a Python-created text as the left/center/right title.
    fn set_title_handle(&self, loc: &str, t: &PyText) -> PyResult<()> {
        let h = t.inner.clone();
        self.with(|a, _| match loc {
            "left" => a.left_title = h,
            "right" => a.right_title = h,
            _ => a.title = h,
        })
    }

    fn set_title_offset(&self, pts: f64) -> PyResult<()> {
        self.with(|a, _| {
            a.title_offset_points = pts;
            a.set_title_offset_trans(pts);
        })
    }

    fn set_autotitlepos(&self, v: Option<bool>) -> PyResult<()> {
        self.with(|a, _| a.autotitlepos = v)
    }

    fn axis_label(&self, axis: &str) -> PyResult<PyText> {
        let x = axis_of(axis)?;
        self.with(|a, _| PyText { inner: a.axis(x).label.clone() })
    }

    fn offset_text(&self, axis: &str) -> PyResult<PyText> {
        let x = axis_of(axis)?;
        self.with(|a, _| PyText { inner: a.axis(x).offset_text.clone() })
    }

    /// Axis-level properties: labelpad, visible, label_position,
    /// tick_position, autolabelpos, remove_overlapping_locs, is_default_label.
    fn axis_set(&self, axis: &str, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let x = axis_of(axis)?;
        self.with(|a, _| -> PyResult<_> {
            let ax = a.axis_mut(x);
            match name {
                "labelpad" => ax.labelpad = value.extract()?,
                "visible" => ax.visible = value.extract()?,
                "label_position" => {
                    let p: String = value.extract()?;
                    {
                        let mut l = ax.label.lk();
                        if x {
                            l.va = if p == "top" { "baseline".into() } else { "top".into() };
                        } else {
                            l.rotation_mode = "anchor".into();
                            l.va = if p == "right" { "top".into() } else { "bottom".into() };
                        }
                    }
                    ax.label_position = p;
                }
                "tick_position" => ax.tick_position = value.extract()?,
                "autolabelpos" => ax.autolabelpos = value.extract()?,
                "remove_overlapping_locs" => ax.remove_overlapping_locs = value.extract()?,
                "is_default_label" => ax.is_default_label = value.extract()?,
                "zorder" => ax.zorder = value.extract()?,
                _ => return Err(PyKeyError::new_err(format!("unknown axis property {name}"))),
            }
            Ok(())
        })?
    }

    fn axis_get<'py>(&self, py: Python<'py>, axis: &str, name: &str) -> PyResult<Bound<'py, PyAny>> {
        let x = axis_of(axis)?;
        self.with(|a, _| -> PyResult<_> {
            let ax = a.axis_mut(x);
            Ok(match name {
                "labelpad" => ax.labelpad.into_pyobject(py)?.into_any(),
                "visible" => ax.visible.into_pyobject(py)?.to_owned().into_any(),
                "label_position" => ax.label_position.clone().into_pyobject(py)?.into_any(),
                "tick_position" => ax.tick_position.clone().into_pyobject(py)?.into_any(),
                "ticks_position" => ax.ticks_position().into_pyobject(py)?.into_any(),
                "remove_overlapping_locs" => ax.remove_overlapping_locs.into_pyobject(py)?.to_owned().into_any(),
                "zorder" => ax.zorder.into_pyobject(py)?.into_any(),
                "offset" => ax.major_formatter.get_offset().into_pyobject(py)?.into_any(),
                _ => return Err(PyKeyError::new_err(format!("unknown axis property {name}"))),
            })
        })?
    }

    /// `set_tick_params(which, reset, **kw)` with matplotlib-translated keys.
    #[pyo3(signature = (axis, which, reset, kw))]
    fn tick_params(&self, axis: &str, which: &str, reset: bool, kw: &Bound<'_, PyDict>) -> PyResult<()> {
        let x = axis_of(axis)?;
        let k = tickkw_from(kw)?;
        self.with(|a, _| a.axis_mut(x).set_tick_params(which, reset, &k))
    }

    #[pyo3(signature = (visible, which, axis, kw))]
    fn grid(&self, visible: Option<bool>, which: &str, axis: &str, kw: &Bound<'_, PyDict>) -> PyResult<()> {
        let g = gridkw_from(kw)?;
        self.with(|a, _| a.grid(visible, which, axis, &g))
    }

    fn set_axison(&self, b: bool) -> PyResult<()> {
        self.with(|a, _| a.axison = b)
    }

    fn get_axison(&self) -> PyResult<bool> {
        self.with(|a, _| a.axison)
    }

    fn set_frameon(&self, b: bool) -> PyResult<()> {
        self.with(|a, _| a.frameon = b)
    }

    fn get_frameon(&self) -> PyResult<bool> {
        self.with(|a, _| a.frameon)
    }

    fn set_axisbelow(&self, b: &str) -> PyResult<()> {
        self.with(|a, _| a.set_axisbelow(b))
    }

    fn set_facecolor(&self, c: &Bound<'_, PyAny>) -> PyResult<()> {
        let cs = color_from(c)?;
        self.with(|a, _| {
            a.facecolor = cs;
            a.patch.set_facecolor(Some(cs));
        })
    }

    fn patch_set(&self, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.with(|a, _| -> PyResult<_> {
            patch_set(&mut a.patch, name, value)
        })?
    }

    fn patch_get<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
        self.with(|a, _| -> PyResult<_> {
            patch_get(py, &a.patch, name)
        })?
    }

    /// set_aspect(aspect 'auto' or float, adjustable, anchor, box_aspect)
    #[pyo3(signature = (aspect=None, adjustable=None, anchor=None))]
    fn set_aspect(&self, aspect: Option<&Bound<'_, PyAny>>, adjustable: Option<String>, anchor: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        let asp = match aspect {
            None => None,
            Some(v) => Some(if v.extract::<String>().is_ok() { Aspect::Auto } else { Aspect::Num(v.extract()?) }),
        };
        let anc = match anchor {
            None => None,
            Some(v) => Some(if let Ok(s) = v.extract::<String>() {
                anchor_coefs(&s).ok_or_else(|| PyValueError::new_err("bad anchor"))?
            } else {
                v.extract::<(f64, f64)>()?
            }),
        };
        self.with(|a, _| {
            if let Some(x) = asp {
                a.aspect = x;
            }
            if let Some(x) = adjustable {
                a.adjustable = x;
            }
            if let Some(x) = anc {
                a.anchor = x;
            }
        })
    }

    fn get_aspect<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let asp = self.with(|a, _| a.aspect.clone())?;
        Ok(match asp {
            Aspect::Auto => "auto".into_pyobject(py)?.into_any(),
            Aspect::Num(v) => v.into_pyobject(py)?.into_any(),
        })
    }

    fn set_box_aspect(&self, v: Option<f64>) -> PyResult<()> {
        self.with(|a, _| {
            a.box_aspect = v;
            if v.is_some() {
                a.adjustable = "datalim".into();
            }
        })
    }

    /// `get_position(original)`; `apply` = False reads the active position
    /// without applying the aspect first (the `_position` attribute).
    #[pyo3(signature = (original=false, apply=true))]
    fn get_position(&self, original: bool, apply: bool) -> PyResult<(f64, f64, f64, f64)> {
        if let Some((s, i)) = self.snap() {
            let a = &s.axes[i];
            let p = if original { a.original_position } else { a.position };
            return Ok((p.x0, p.y0, p.x1, p.y1));
        }
        self.with(|a, ctx| -> PyResult<_> {
            let p = if original {
                a.original_position
            } else {
                if apply {
                    a.apply_aspect(None, &ctx);
                }
                a.position
            };
            Ok((p.x0, p.y0, p.x1, p.y1))
        })?
    }

    /// `apply_aspect(position)`.
    #[pyo3(signature = (position=None))]
    fn apply_aspect(&self, position: Option<(f64, f64, f64, f64)>) -> PyResult<()> {
        self.with(|a, ctx| -> PyResult<_> {
            a.apply_aspect(position.map(|p| Bbox::new(p.0, p.1, p.2, p.3)), &ctx);
            Ok(())
        })?
    }

    #[pyo3(signature = (rect, which="both"))]
    fn set_position(&self, rect: (f64, f64, f64, f64), which: &str) -> PyResult<()> {
        let b = Bbox::new(rect.0, rect.1, rect.2, rect.3);
        let mut fig = self.lk();
        let tw: Vec<u64> = {
            let a = axes_mut(&mut fig, self.uid)?;
            if a.twinned.is_empty() {
                vec![self.uid]
            } else {
                a.twinned.clone()
            }
        };
        for a in fig.all_axes_mut() {
            if tw.contains(&a.uid) {
                if which == "both" || which == "active" {
                    a.position = b;
                }
                if which == "both" || which == "original" {
                    a.original_position = b;
                }
            }
        }
        Ok(())
    }

    fn clear_subplotspec(&self) -> PyResult<()> {
        self.with(|a, _| a.subplotspec = None)
    }

    /// (nrows, ncols, num1, num2) of the subplotspec, if any.
    fn subplotspec(&self) -> PyResult<Option<(usize, usize, usize, usize)>> {
        self.with(|a, _| a.subplotspec.as_ref().map(|s| (s.nrows, s.ncols, s.num1, s.num2)))
    }

    fn spine_set(&self, name: &str, prop: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let idx = spine_idx(name)?;
        self.with(|a, _| -> PyResult<_> {
            match prop {
                "position" if value.is_none() => {
                    a.spines[idx].position = None;
                }
                "position" => {
                    let s = &mut a.spines[idx];
                    s.position = Some(if let Ok(st) = value.extract::<String>() {
                        match st.as_str() {
                            "center" => SpinePosition::Axes(0.5),
                            "zero" => SpinePosition::Data(0.0),
                            _ => return err("bad spine position"),
                        }
                    } else {
                        let (k, amt): (String, f64) = value.extract()?;
                        match k.as_str() {
                            "outward" => SpinePosition::Outward(amt),
                            "axes" => SpinePosition::Axes(amt),
                            "data" => SpinePosition::Data(amt),
                            _ => return err("bad spine position"),
                        }
                    });
                    let x = idx >= 2;
                    a.axis_mut(x).reset_ticks();
                }
                "bounds" => a.spines[idx].bounds = if value.is_none() { None } else { Some(value.extract()?) },
                "capstyle" => {
                    let s: String = value.extract()?;
                    a.spines[idx].patch.capstyle = match s.as_str() {
                        "butt" => LineCap::Butt,
                        "round" => LineCap::Round,
                        _ => LineCap::Square,
                    };
                }
                _ => patch_set(&mut a.spines[idx].patch, prop, value)?,
            }
            Ok(())
        })?
    }

    fn spine_get<'py>(&self, py: Python<'py>, name: &str, prop: &str) -> PyResult<Bound<'py, PyAny>> {
        let idx = spine_idx(name)?;
        self.with(|a, _| -> PyResult<_> {
            match prop {
                "position" => Ok(match a.spines[idx].position() {
                    SpinePosition::Outward(v) => ("outward", v).into_pyobject(py)?.into_any(),
                    SpinePosition::Axes(v) => ("axes", v).into_pyobject(py)?.into_any(),
                    SpinePosition::Data(v) => ("data", v).into_pyobject(py)?.into_any(),
                }),
                "bounds" => Ok(a.spines[idx].bounds.into_pyobject(py)?.into_any()),
                _ => patch_get(py, &a.spines[idx].patch, prop),
            }
        })?
    }

    /// Tick locations (`get_ticklocs`).
    #[pyo3(signature = (axis, minor=false))]
    fn get_ticklocs(&self, axis: &str, minor: bool) -> PyResult<Vec<f64>> {
        let x = axis_of(axis)?;
        self.unstaled(|a, ctx| {
            let tc = a.trans_ctx(ctx);
            let ac = a.axis_ctx(x, &tc, ctx);
            if minor {
                a.axis(x).get_minorticklocs(&ac)
            } else {
                a.axis(x).get_majorticklocs(&ac)
            }
        })
    }

    /// Update ticks and return the tick label handles (label1 and label2 of
    /// each tick), like `get_ticklabels(minor, which)`; `visible_only`
    /// filters like matplotlib.
    #[pyo3(signature = (axis, which="major", visible_only=true, count=None))]
    fn get_ticklabels(&self, axis: &str, which: &str, visible_only: bool, count: Option<usize>) -> PyResult<Vec<PyText>> {
        let x = axis_of(axis)?;
        self.unstaled(|a, ctx| {
            let tc = a.trans_ctx(ctx);
            let ac = a.axis_ctx(x, &tc, ctx);
            let ax = if x { &mut a.xaxis } else { &mut a.yaxis };
            ax.update_ticks(&ac);
            let mut out = Vec::new();
            let mut collect = |ticks: &Vec<crate::mpl::axis::Tick>, n: usize| {
                let mut l1 = Vec::new();
                let mut l2 = Vec::new();
                for t in ticks.iter().take(n) {
                    if !visible_only || t.label1.lk().base.visible {
                        l1.push(PyText { inner: t.label1.clone() });
                    }
                    if !visible_only || t.label2.lk().base.visible {
                        l2.push(PyText { inner: t.label2.clone() });
                    }
                }
                out.extend(l1);
                out.extend(l2);
            };
            if which == "major" || which == "both" {
                let n = ax.get_majorticklocs(&ac).len().max(count.unwrap_or(0));
                ax.ensure_major_ticks(n);
                let t = ax.major_ticks.clone().unwrap_or_default();
                collect(&t, n);
            }
            if which == "minor" || which == "both" {
                let n = ax.get_minorticklocs(&ac).len().max(count.unwrap_or(0));
                ax.ensure_minor_ticks(n);
                let t = ax.minor_ticks.clone().unwrap_or_default();
                collect(&t, n);
            }
            out
        })
    }

    /// Update the ticks (as `get_ticklabels` does) and return a fingerprint
    /// of the tick objects: their identities and states.  Equal fingerprints
    /// mean the tick handles and everything about them are unchanged.
    #[pyo3(signature = (axis, major=true, count=None))]
    fn ticks_signature(&self, axis: &str, major: bool, count: Option<usize>) -> PyResult<u64> {
        use std::hash::{Hash, Hasher};
        let x = axis_of(axis)?;
        self.unstaled(|a, ctx| {
            let tc = a.trans_ctx(ctx);
            let ac = a.axis_ctx(x, &tc, ctx);
            let ax = if x { &mut a.xaxis } else { &mut a.yaxis };
            ax.update_ticks(&ac);
            let (n, ticks) = if major {
                let n = ax.get_majorticklocs(&ac).len().max(count.unwrap_or(0));
                ax.ensure_major_ticks(n);
                (n, ax.major_ticks.clone().unwrap_or_default())
            } else {
                let n = ax.get_minorticklocs(&ac).len().max(count.unwrap_or(0));
                ax.ensure_minor_ticks(n);
                (n, ax.minor_ticks.clone().unwrap_or_default())
            };
            let mut h = std::collections::hash_map::DefaultHasher::new();
            n.hash(&mut h);
            for t in ticks.iter().take(n) {
                for p in [&t.label1, &t.label2] {
                    (Arc::as_ptr(p) as usize).hash(&mut h);
                    super::model::state_hash(&*p.lk()).hash(&mut h);
                }
                for p in [&t.tick1line, &t.tick2line, &t.gridline] {
                    (Arc::as_ptr(p) as usize).hash(&mut h);
                    super::model::state_hash(&*p.lk()).hash(&mut h);
                }
            }
            h.finish()
        })
    }

    /// Tick line handles (tick1line, tick2line per tick).
    #[pyo3(signature = (axis, minor=false, count=None))]
    fn get_ticklines(&self, axis: &str, minor: bool, count: Option<usize>) -> PyResult<Vec<PyLine2D>> {
        let x = axis_of(axis)?;
        self.unstaled(|a, ctx| {
            let tc = a.trans_ctx(ctx);
            let ac = a.axis_ctx(x, &tc, ctx);
            let ax = if x { &mut a.xaxis } else { &mut a.yaxis };
            let n = if minor { ax.get_minorticklocs(&ac).len().max(count.unwrap_or(0)) } else { ax.get_majorticklocs(&ac).len().max(count.unwrap_or(0)) };
            let ticks = if minor {
                ax.ensure_minor_ticks(n);
                ax.minor_ticks.clone().unwrap_or_default()
            } else {
                ax.ensure_major_ticks(n);
                ax.major_ticks.clone().unwrap_or_default()
            };
            let mut out = Vec::new();
            for t in ticks.iter().take(n) {
                out.push(PyLine2D { inner: t.tick1line.clone() });
                out.push(PyLine2D { inner: t.tick2line.clone() });
            }
            out
        })
    }

    #[pyo3(signature = (axis, minor=false, count=None))]
    fn get_gridlines(&self, axis: &str, minor: bool, count: Option<usize>) -> PyResult<Vec<PyLine2D>> {
        let x = axis_of(axis)?;
        self.unstaled(|a, ctx| {
            let tc = a.trans_ctx(ctx);
            let ac = a.axis_ctx(x, &tc, ctx);
            let ax = if x { &mut a.xaxis } else { &mut a.yaxis };
            let n = if minor { ax.get_minorticklocs(&ac).len().max(count.unwrap_or(0)) } else { ax.get_majorticklocs(&ac).len().max(count.unwrap_or(0)) };
            let ticks = if minor {
                ax.ensure_minor_ticks(n);
                ax.minor_ticks.clone().unwrap_or_default()
            } else {
                ax.ensure_major_ticks(n);
                ax.major_ticks.clone().unwrap_or_default()
            };
            ticks.iter().take(n).map(|t| PyLine2D { inner: t.gridline.clone() }).collect()
        })
    }

    #[pyo3(signature = (axis, spec, minor=false))]
    fn set_locator(&self, axis: &str, spec: &Bound<'_, PyAny>, minor: bool) -> PyResult<()> {
        let x = axis_of(axis)?;
        let loc = locator_from(spec)?;
        let mut fig = self.lk();
        let group: Vec<u64> = {
            let a = axes_mut(&mut fig, self.uid)?;
            let g = if x { a.shared_x.clone() } else { a.shared_y.clone() };
            if g.is_empty() {
                vec![self.uid]
            } else {
                g
            }
        };
        for a in fig.all_axes_mut() {
            if group.contains(&a.uid) {
                let ax = a.axis_mut(x);
                if minor {
                    ax.minor_locator = loc.clone();
                    ax.default_minloc = false;
                } else {
                    ax.major_locator = loc.clone();
                    ax.default_majloc = false;
                }
            }
        }
        Ok(())
    }

    #[pyo3(signature = (axis, spec, minor=false))]
    fn set_formatter(&self, axis: &str, spec: &Bound<'_, PyAny>, minor: bool) -> PyResult<()> {
        let x = axis_of(axis)?;
        let fmt = formatter_from(spec)?;
        let mut fig = self.lk();
        let group: Vec<u64> = {
            let a = axes_mut(&mut fig, self.uid)?;
            let g = if x { a.shared_x.clone() } else { a.shared_y.clone() };
            if g.is_empty() {
                vec![self.uid]
            } else {
                g
            }
        };
        for a in fig.all_axes_mut() {
            if group.contains(&a.uid) {
                let ax = a.axis_mut(x);
                if minor {
                    ax.minor_formatter = fmt.clone();
                    ax.default_minfmt = false;
                } else {
                    ax.major_formatter = fmt.clone();
                    ax.default_majfmt = false;
                }
            }
        }
        Ok(())
    }

    /// Describe the current locator/formatter kind ("auto", "scalar", ...).
    #[pyo3(signature = (axis, minor=false))]
    fn locator_kind(&self, axis: &str, minor: bool) -> PyResult<String> {
        let x = axis_of(axis)?;
        self.with(|a, _| {
            let ax = a.axis(x);
            let l = if minor { &ax.minor_locator } else { &ax.major_locator };
            match l {
                Locator::MaxN(m) if *m == MaxNLocator::auto() => "auto",
                Locator::MaxN(_) => "maxn",
                Locator::Fixed { .. } => "fixed",
                Locator::Null => "null",
                Locator::Multiple { .. } => "multiple",
                Locator::Linear { .. } => "linear",
                Locator::Index { .. } => "index",
                Locator::AutoMinor { .. } => "autominor",
                Locator::Log(_) => "log",
                Locator::External(_) => "py",
            }
            .to_string()
        })
    }

    #[pyo3(signature = (axis, minor=false))]
    fn formatter_kind(&self, axis: &str, minor: bool) -> PyResult<String> {
        let x = axis_of(axis)?;
        self.with(|a, _| {
            let ax = a.axis(x);
            let f = if minor { &ax.minor_formatter } else { &ax.major_formatter };
            match f {
                Formatter::Scalar(_) => "scalar",
                Formatter::Null => "null",
                Formatter::Fixed { .. } => "fixed",
                Formatter::Log(l) => match l.kind {
                    LogFormatterKind::SciNotation => "logsci",
                    LogFormatterKind::Mathtext => "logmath",
                    LogFormatterKind::Exponent => "logexp",
                    LogFormatterKind::Plain => "log",
                },
                Formatter::External(_) => "py",
            }
            .to_string()
        })
    }

    /// State of a native ScalarFormatter: (offset, orderOfMagnitude, sigfigs, useOffset, useMathText, scientific, powerlimits).
    fn scalar_formatter_state(&self, axis: &str) -> PyResult<Option<(f64, i64, usize, bool, bool, bool, (i64, i64))>> {
        let x = axis_of(axis)?;
        self.with(|a, _| match &a.axis(x).major_formatter {
            Formatter::Scalar(s) => Some((s.offset, s.order_of_magnitude, s.sigfigs, s.use_offset, s.use_mathtext, s.scientific, s.powerlimits)),
            _ => None,
        })
    }

    /// Axis info tuple for Python tick helpers.
    fn axis_info<'py>(&self, py: Python<'py>, axis: &str) -> PyResult<Bound<'py, PyTuple>> {
        let x = axis_of(axis)?;
        let info = self.unstaled(|a, ctx| a.axis_info(x, ctx))?;
        info_tuple(py, &info)
    }

    fn reset_ticks(&self, axis: &str) -> PyResult<()> {
        let x = axis_of(axis)?;
        self.with(|a, _| a.axis_mut(x).reset_ticks())
    }

    /// Display-space window extent of the axes (`ax.bbox`).
    fn get_window_extent(&self) -> PyResult<(f64, f64, f64, f64)> {
        if let Some((s, i)) = self.snap() {
            let b = s.axes[i].bbox;
            return Ok((b.x0, b.y0, b.x1, b.y1));
        }
        self.with(|a, c| {
            let b = a.bbox(c);
            (b.x0, b.y0, b.x1, b.y1)
        })
    }

    #[pyo3(signature = (for_layout_only=false))]
    fn get_tightbbox(&self, for_layout_only: bool) -> PyResult<Option<(f64, f64, f64, f64)>> {
        let b = if let Some(mut l) = self.lent() {
            let s = crate::mpl::figure::snapshot_for(l.key).ok_or_else(|| PyValueError::new_err("no draw in progress"))?;
            let mut r = Renderer::metrics_only(s.canvas.0, s.canvas.1, l.fig.dpi);
            let ctx = l.fig.clone();
            l.axes().get_tightbbox(&mut r, &ctx, for_layout_only)
        } else {
            let mut fig = self.lk();
            let ctx = fig.ctx();
            let (w, h) = fig.canvas_size();
            let mut r = Renderer::metrics_only(w as f64, h as f64, ctx.dpi);
            let _snap = fig.publish_snapshot();
            axes_mut(&mut fig, self.uid)?.get_tightbbox(&mut r, &ctx, for_layout_only)
        };
        super::ext::take_error()?;
        Ok(b.map(|b| (b.x0, b.y0, b.x1, b.y1)))
    }

    /// Matrices of the axes transforms: "data", "axes", "xaxis", "yaxis".
    fn transform_matrix(&self, which: &str) -> PyResult<Vec<Vec<f64>>> {
        let pick = |tc: &TransCtx| {
            let t = match which {
                "axes" => tc.trans_axes.clone().unwrap(),
                "xaxis" => tc.resolve(&crate::mpl::artist::TSpec::XAxis(crate::mpl::artist::Which::Grid)),
                "yaxis" => tc.resolve(&crate::mpl::artist::TSpec::YAxis(crate::mpl::artist::Which::Grid)),
                _ => tc.trans_data.clone().unwrap(),
            };
            t.get_affine()
        };
        let m = if let Some((s, i)) = self.snap() {
            pick(&s.axes[i].tc)
        } else if which == "axes" {
            self.with(|a, c| pick(&a.trans_ctx(c)))?
        } else {
            self.unstaled(|a, c| pick(&a.trans_ctx(c)))?
        };
        Ok(m.iter().map(|r| r.to_vec()).collect())
    }

    /// Transform points with the axes' transData (incl. non-affine scales).
    fn transform_data(&self, pts: Vec<[f64; 2]>) -> PyResult<Vec<[f64; 2]>> {
        if let Some((s, i)) = self.snap() {
            return Ok(s.axes[i].tc.trans_data.as_ref().unwrap().transform_points(&pts));
        }
        self.unstaled(|a, c| {
            let tc: TransCtx = a.trans_ctx(c);
            tc.trans_data.as_ref().unwrap().transform_points(&pts)
        })
    }

    fn text_extent(&self, t: &PyText) -> PyResult<(f64, f64, f64, f64)> {
        if let Some((s, i)) = self.snap() {
            let mut r = Renderer::metrics_only(s.canvas.0, s.canvas.1, s.fig.dpi);
            let b = t.lk().get_window_extent(&mut r, &s.axes[i].tc);
            return Ok((b.x0, b.y0, b.x1, b.y1));
        }
        let mut fig = self.lk();
        let ctx = fig.ctx();
        let (w, h) = fig.canvas_size();
        let mut r = Renderer::metrics_only(w as f64, h as f64, ctx.dpi);
        let tc = axes_mut(&mut fig, self.uid)?.trans_ctx(&ctx);
        let b = t.inner.lk().get_window_extent(&mut r, &tc);
        Ok((b.x0, b.y0, b.x1, b.y1))
    }

    fn base_set(&self, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.with(|a, _| -> PyResult<_> {
            if name == "label" {
                a.label = value.extract()?;
                return Ok(());
            }
            if !base_set(&mut a.base, name, value)? {
                return Err(PyKeyError::new_err(format!("unknown axes property {name}")));
            }
            Ok(())
        })?
    }

    fn base_get<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
        self.with(|a, _| -> PyResult<_> {
            Ok(match name {
                "visible" => a.base.visible.into_pyobject(py)?.to_owned().into_any(),
                "zorder" => a.base.zorder.into_pyobject(py)?.into_any(),
                "label" => a.label.clone().into_pyobject(py)?.into_any(),
                "in_layout" => a.base.in_layout.into_pyobject(py)?.to_owned().into_any(),
                _ => return Err(PyKeyError::new_err(format!("unknown axes property {name}"))),
            })
        })?
    }

    fn cla(&self) -> PyResult<()> {
        self.with(|a, _| {
            a.clear();
            a.child_axes.clear();
        })
    }

    /// `transSubfigure` of the subfigure holding this axes (a `("py", t)`
    /// transform spec), or None for the figure's.
    fn set_trans_subfigure(&self, spec: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        let t = match spec {
            Some(s) => match super::model::tspec_from(s)? {
                crate::mpl::artist::TSpec::Ext(e) => Some(e),
                _ => return Err(PyValueError::new_err("expected a ('py', transform) spec")),
            },
            None => None,
        };
        self.with(|a, _| a.trans_subfigure = t)
    }

    /// Have the axes drawn by *obj*'s `_rs_draw` (None: by the Rust core).
    fn set_ext_draw(&self, obj: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        let e = obj.map(super::ext::ext_from);
        self.with(|a, _| a.ext_draw = e)
    }

    /// `add_child_axes`: draw the axes *uid* among this axes' children.
    fn add_child_axes(&self, uid: u64) -> PyResult<()> {
        self.with(|a, _| {
            a.child_axes.retain(|&u| u != uid);
            a.child_axes.push(uid);
        })
    }

    /// `Axes.draw(renderer)` of this axes alone into a Python `RendererAgg`.
    fn draw_into(&self, target: &Bound<'_, super::lowlevel::PyRendererAgg>) -> PyResult<()> {
        {
            let mut t = target.borrow_mut();
            let r = t.r()?;
            if let Some(mut l) = self.lent() {
                l.draw(r);
                drop(l);
                drop(t);
                return super::ext::take_error();
            }
            let mut fig = self.lk();
            let _snap = fig.publish_snapshot();
            if !fig.draw_axes(self.uid, r) {
                return Err(PyValueError::new_err("no such axes"));
            }
        }
        super::ext::take_error()
    }

    /// `Axes.draw(renderer)` of this axes alone into any other Python renderer.
    fn draw_foreign(&self, renderer: &Bound<'_, PyAny>) -> PyResult<()> {
        let fr = std::sync::Arc::new(super::foreign::PyForeign::new(renderer)?);
        if let Some(mut l) = self.lent() {
            let s = crate::mpl::figure::snapshot_for(l.key).ok_or_else(|| PyValueError::new_err("no draw in progress"))?;
            let mut r = Renderer::foreign(fr, s.canvas.0, s.canvas.1, l.fig.dpi);
            l.draw(&mut r);
            drop(l);
            return super::ext::take_error();
        }
        {
            let mut fig = self.lk();
            let (w, h) = fig.canvas_size();
            let mut r = Renderer::foreign(fr, w as f64, h as f64, fig.dpi);
            let _snap = fig.publish_snapshot();
            if !fig.draw_axes(self.uid, &mut r) {
                return Err(PyValueError::new_err("no such axes"));
            }
        }
        super::ext::take_error()
    }

    fn shared(&self, axis: &str) -> PyResult<Vec<u64>> {
        let x = axis_of(axis)?;
        self.with(|a, _| if x { a.shared_x.clone() } else { a.shared_y.clone() })
    }

    fn twinned(&self) -> PyResult<Vec<u64>> {
        self.with(|a, _| a.twinned.clone())
    }

    fn set_legend(&self, l: Option<&super::collections::PyLegend>) -> PyResult<()> {
        let leg = l.map(|l| l.inner.clone());
        self.with(|a, _| a.legend = leg)
    }

    fn cycle_index(&self, kind: &str, advance: bool) -> PyResult<usize> {
        self.with(|a, _| {
            let r = if kind == "patch" { &mut a.patch_cycle_index } else { &mut a.prop_cycle_index };
            let v = *r;
            if advance {
                *r += 1;
            }
            v
        })
    }

    fn reset_cycle(&self) -> PyResult<()> {
        self.with(|a, _| {
            a.prop_cycle_index = 0;
            a.patch_cycle_index = 0;
        })
    }

    #[getter]
    fn figure(&self) -> PyFigure {
        PyFigure { inner: self.fig.clone(), primary: false }
    }
}

fn spine_idx(name: &str) -> PyResult<usize> {
    Ok(match name {
        "left" => 0,
        "right" => 1,
        "bottom" => 2,
        "top" => 3,
        _ => return Err(PyKeyError::new_err(format!("unknown spine {name}"))),
    })
}

#[pyfunction]
fn findfont_props(prop: &Bound<'_, PyAny>) -> PyResult<Vec<(String, u32)>> {
    let p = fontprops_from(prop)?;
    Ok(fonts::FONT_MANAGER.write().unwrap().find_fonts_by_props(&p))
}

#[pyfunction]
fn text_metrics(s: &str, prop: &Bound<'_, PyAny>, ismath: bool, dpi: f64) -> PyResult<(f64, f64, f64)> {
    let p = fontprops_from(prop)?;
    Ok(crate::mpl::renderer::text_metrics(s, &p, ismath, dpi))
}

/// Encode an (h, w, 4) uint8 RGBA array as PNG bytes with the pixel
/// density of `dpi` and text chunks `meta`.
#[pyfunction]
#[pyo3(signature = (rgba, dpi, meta, compression=1))]
fn encode_png<'py>(
    py: Python<'py>,
    rgba: numpy::PyReadonlyArray3<'py, u8>,
    dpi: f64,
    meta: Vec<(String, String)>,
    compression: u8,
) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
    let a = rgba.as_array();
    let (h, w, c) = (a.shape()[0], a.shape()[1], a.shape()[2]);
    if c != 4 {
        return err("expected an RGBA array");
    }
    let data: Vec<u8> = match a.as_slice() {
        Some(s) => s.to_vec(),
        None => a.iter().copied().collect(),
    };
    let out = py
        .detach(|| encode_png_bytes(&data, w as u32, h as u32, dpi, &meta, compression))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(pyo3::types::PyBytes::new(py, &out))
}

fn encode_png_bytes(rgba: &[u8], w: u32, h: u32, dpi: f64, meta: &[(String, String)], level: u8) -> std::io::Result<Vec<u8>> {
    let mut out = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut out, w, h);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        enc.set_compression(match level {
            0 => png::Compression::NoCompression,
            1 => png::Compression::Fastest,
            2 => png::Compression::Fast,
            3 => png::Compression::Balanced,
            _ => png::Compression::High,
        });
        let ppm = (dpi / 0.0254).round() as u32;
        enc.set_pixel_dims(Some(png::PixelDimensions { xppu: ppm, yppu: ppm, unit: png::Unit::Meter }));
        for (k, v) in meta {
            enc.add_text_chunk(k.clone(), v.clone()).map_err(std::io::Error::other)?;
        }
        let mut writer = enc.write_header().map_err(std::io::Error::other)?;
        writer.write_image_data(rgba).map_err(std::io::Error::other)?;
        writer.finish().map_err(std::io::Error::other)?;
    }
    Ok(out)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFigure>()?;
    m.add_class::<PyAxes>()?;
    m.add_class::<PyText>()?;
    m.add_class::<PyPatch>()?;
    m.add_function(wrap_pyfunction!(findfont_props, m)?)?;
    m.add_function(wrap_pyfunction!(text_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(encode_png, m)?)?;
    let _ = f64_vec;
    Ok(())
}
