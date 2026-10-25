//! Bindings for the artist model.  The Python layer (`matplotlib_rs`)
//! wraps these handles with matplotlib's public API.

use numpy::{PyArray3, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods, PyList, PyTuple};

use crate::agg::pixfmt::Rgba;
use crate::mpl::artist::{shared, ClipBoxSpec, Lock, Shared, TSpec};
use crate::mpl::axes::Child;
use crate::mpl::colors::ColorSpec;
use crate::mpl::figure::Figure;
use crate::mpl::fonts::{self, FontEntry, NumOrName};
use crate::mpl::lines::{DrawStyle, Line2D};
use crate::mpl::markers::{FillStyle, MarkerSpec, MarkerStyle};
use crate::mpl::patches::Patch;
use crate::mpl::rc::{rc_update, RcValue};
use crate::mpl::text::Text;

// ---------------------------------------------------------------------------
// conversions

pub fn color_from(obj: &Bound<'_, PyAny>) -> PyResult<ColorSpec> {
    if obj.is_none() {
        return Ok(ColorSpec::None);
    }
    if let Ok(s) = obj.extract::<String>() {
        return match s.to_lowercase().as_str() {
            "none" => Ok(ColorSpec::None),
            "auto" => Ok(ColorSpec::Auto),
            _ => Err(PyValueError::new_err(format!("unresolved colour {s:?}"))),
        };
    }
    let t: Vec<f64> = obj.extract()?;
    match t.len() {
        4 => Ok(ColorSpec::rgba(t[0], t[1], t[2], t[3])),
        3 => Ok(ColorSpec::rgba(t[0], t[1], t[2], 1.0)),
        _ => Err(PyValueError::new_err("RGBA value must be 3- or 4-tuple")),
    }
}

fn rgba_of(obj: &Bound<'_, PyAny>) -> PyResult<Rgba> {
    Ok(color_from(obj)?.to_rgba(None))
}

fn opt_f64(obj: &Bound<'_, PyAny>) -> PyResult<Option<f64>> {
    if obj.is_none() {
        Ok(None)
    } else {
        Ok(Some(obj.extract()?))
    }
}

fn tspec_from(obj: &Bound<'_, PyAny>) -> PyResult<TSpec> {
    if let Ok(s) = obj.extract::<String>() {
        return Ok(match s.as_str() {
            "data" => TSpec::Data,
            "axes" => TSpec::Axes,
            "figure" => TSpec::Figure,
            "identity" => TSpec::Identity,
            "dpi" => TSpec::DpiScale,
            "xaxis" => TSpec::XAxis(crate::mpl::artist::Which::Grid),
            "yaxis" => TSpec::YAxis(crate::mpl::artist::Which::Grid),
            _ => return Err(PyValueError::new_err(format!("unknown transform {s}"))),
        });
    }
    let t = obj.cast::<PyTuple>()?;
    let kind: String = t.get_item(0)?.extract()?;
    Ok(match kind.as_str() {
        "affine" => {
            let m: Vec<Vec<f64>> = t.get_item(1)?.extract()?;
            TSpec::Affine([[m[0][0], m[0][1], m[0][2]], [m[1][0], m[1][1], m[1][2]], [0.0, 0.0, 1.0]])
        }
        "then" => TSpec::Then(Box::new(tspec_from(&t.get_item(1)?)?), Box::new(tspec_from(&t.get_item(2)?)?)),
        "blend" => TSpec::Blend(Box::new(tspec_from(&t.get_item(1)?)?), Box::new(tspec_from(&t.get_item(2)?)?)),
        "scaled_translation" => TSpec::ScaledTranslation(t.get_item(1)?.extract()?, t.get_item(2)?.extract()?),
        _ => return Err(PyValueError::new_err(format!("unknown transform kind {kind}"))),
    })
}

fn marker_from(obj: &Bound<'_, PyAny>, fs: FillStyle) -> PyResult<MarkerStyle> {
    if obj.is_none() {
        return Ok(MarkerStyle::new(MarkerSpec::Named("none".into()), fs));
    }
    if let Ok(s) = obj.extract::<String>() {
        if !MarkerStyle::is_valid_name(&s) {
            return Err(PyValueError::new_err(format!("Unrecognized marker style {s:?}")));
        }
        return Ok(MarkerStyle::new(MarkerSpec::Named(s), fs));
    }
    if let Ok((n, st, a)) = obj.extract::<(usize, u8, f64)>() {
        return Ok(MarkerStyle::new(MarkerSpec::Tuple(n, st, a), fs));
    }
    Err(PyTypeError::new_err("unsupported marker"))
}

fn base_set(base: &mut crate::mpl::artist::ArtistBase, name: &str, v: &Bound<'_, PyAny>) -> PyResult<bool> {
    match name {
        "visible" => base.visible = v.extract()?,
        "alpha" => base.alpha = opt_f64(v)?,
        "zorder" => base.zorder = v.extract()?,
        "label" => base.label = v.extract()?,
        "clip_on" => base.clip_on = v.extract()?,
        "clip_box" => {
            base.clip_box = if v.is_none() {
                ClipBoxSpec::None
            } else if let Ok(s) = v.extract::<String>() {
                if s == "axes" {
                    ClipBoxSpec::AxesBbox
                } else {
                    return Err(PyValueError::new_err("bad clip box"));
                }
            } else {
                let b: (f64, f64, f64, f64) = v.extract()?;
                ClipBoxSpec::Display(crate::mpl::transforms::Bbox::new(b.0, b.1, b.2, b.3))
            }
        }
        "snap" => base.snap = if v.is_none() { None } else { Some(v.extract()?) },
        "transform" => {
            base.transform = tspec_from(v)?;
            base.transform_set = true;
        }
        "in_layout" => base.in_layout = v.extract()?,
        "gid" => base.gid = if v.is_none() { None } else { Some(v.extract()?) },
        "animated" => base.animated = v.extract()?,
        "rasterized" => base.rasterized = v.extract()?,
        "sketch_params" => base.sketch = if v.is_none() { None } else { Some(v.extract()?) },
        _ => return Ok(false),
    }
    Ok(true)
}

fn line_set(l: &mut Line2D, name: &str, v: &Bound<'_, PyAny>) -> PyResult<()> {
    if base_set(&mut l.base, name, v)? {
        if name == "transform" {
            l.invalidate();
        }
        return Ok(());
    }
    match name {
        "xdata" => l.set_xdata(f64_vec(v)?),
        "ydata" => l.set_ydata(f64_vec(v)?),
        "linewidth" => l.set_linewidth(v.extract()?),
        "linestyle" => {
            if let Ok(s) = v.extract::<String>() {
                l.set_linestyle(&s);
            } else {
                let (o, d): (f64, Option<Vec<f64>>) = v.extract()?;
                l.set_dashes(o, d);
            }
        }
        "color" => l.color = color_from(v)?,
        "gapcolor" => l.gapcolor = if v.is_none() { None } else { Some(color_from(v)?) },
        "marker" => {
            let fs = l.marker.fillstyle;
            l.set_marker(marker_from(v, fs)?);
        }
        "fillstyle" => {
            let s: String = v.extract()?;
            let fs = FillStyle::parse(&s).ok_or_else(|| PyValueError::new_err("bad fillstyle"))?;
            let spec = l.marker.spec.clone();
            l.set_marker(MarkerStyle::new(spec, fs));
        }
        "markersize" => l.markersize = v.extract()?,
        "markeredgewidth" => l.markeredgewidth = v.extract()?,
        "markeredgecolor" => l.markeredgecolor = color_from(v)?,
        "markerfacecolor" => l.markerfacecolor = color_from(v)?,
        "markerfacecoloralt" => l.markerfacecoloralt = color_from(v)?,
        "antialiased" => l.antialiased = v.extract()?,
        "drawstyle" => {
            let s: String = v.extract()?;
            l.set_drawstyle(DrawStyle::parse(&s).ok_or_else(|| PyValueError::new_err("bad drawstyle"))?);
        }
        "dash_capstyle" => l.dash_capstyle = crate::mpl::artist::parse_capstyle(&v.extract::<String>()?),
        "solid_capstyle" => l.solid_capstyle = crate::mpl::artist::parse_capstyle(&v.extract::<String>()?),
        "dash_joinstyle" => l.dash_joinstyle = crate::mpl::artist::parse_joinstyle(&v.extract::<String>()?),
        "solid_joinstyle" => l.solid_joinstyle = crate::mpl::artist::parse_joinstyle(&v.extract::<String>()?),
        _ => return Err(PyKeyError::new_err(format!("Line2D has no property {name:?}"))),
    }
    Ok(())
}

fn f64_vec(v: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    if let Ok(a) = v.extract::<PyReadonlyArray1<f64>>() {
        return Ok(a.as_array().iter().copied().collect());
    }
    v.extract()
}

// ---------------------------------------------------------------------------
// handles

#[pyclass(name = "Line2D", module = "matplotlib_rs._mplrs")]
pub struct PyLine2D {
    pub inner: Shared<Line2D>,
}

#[pymethods]
impl PyLine2D {
    #[new]
    fn new(x: &Bound<'_, PyAny>, y: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(PyLine2D { inner: shared(Line2D::new(f64_vec(x)?, f64_vec(y)?)) })
    }

    fn set(&self, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        line_set(&mut self.inner.lk(), name, value)
    }

    fn set_many(&self, props: &Bound<'_, PyDict>) -> PyResult<()> {
        let mut l = self.inner.lk();
        for (k, v) in props.iter() {
            line_set(&mut l, &k.extract::<String>()?, &v)?;
        }
        Ok(())
    }
}

#[pyclass(name = "Figure", module = "matplotlib_rs._mplrs")]
pub struct PyFigure {
    pub inner: Shared<Figure>,
}

#[pymethods]
impl PyFigure {
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
        Ok(PyFigure { inner: shared(Figure::new(figsize, dpi, fc, ec, linewidth, frameon)) })
    }

    /// add_subplot(nrows, ncols, num1, num2) with 0-based cell indices.
    fn add_subplot(&self, nrows: usize, ncols: usize, num1: usize, num2: usize) -> u64 {
        let mut f = self.inner.lk();
        let gs = f.gridspec_for(nrows, ncols);
        f.add_subplot(gs, num1, num2)
    }

    fn add_axes(&self, rect: (f64, f64, f64, f64)) -> u64 {
        self.inner.lk().add_axes_rect(rect)
    }

    fn add_line(&self, ax: u64, line: &PyLine2D) -> PyResult<()> {
        let mut f = self.inner.lk();
        let a = f.ax(ax).ok_or_else(|| PyValueError::new_err("no such axes"))?;
        a.add_line(line.inner.clone());
        Ok(())
    }

    #[pyo3(signature = (ax, x=true, y=true, tight=None))]
    fn request_autoscale_view(&self, ax: u64, x: bool, y: bool, tight: Option<bool>) -> PyResult<()> {
        let mut f = self.inner.lk();
        let a = f.ax(ax).ok_or_else(|| PyValueError::new_err("no such axes"))?;
        a.request_autoscale_view(x, y, tight);
        Ok(())
    }

    fn get_lim(&self, ax: u64, x: bool) -> PyResult<(f64, f64)> {
        let mut f = self.inner.lk();
        let i = f.axes_index(ax).ok_or_else(|| PyValueError::new_err("no such axes"))?;
        f.unstale_viewlim(i);
        Ok(f.axes[i].get_lim(x))
    }

    fn render<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<u8>>> {
        let r = py.detach(|| self.inner.lk().render());
        let (w, h) = (r.agg.width as usize, r.agg.height as usize);
        let arr = numpy::PyArray1::from_slice(py, r.agg.buffer());
        Ok(arr.reshape([h, w, 4])?)
    }

    fn line_count(&self, ax: u64) -> PyResult<usize> {
        let mut f = self.inner.lk();
        let a = f.ax(ax).ok_or_else(|| PyValueError::new_err("no such axes"))?;
        Ok(a.children.iter().filter(|c| matches!(c, Child::Line(_))).count())
    }
}

// ---------------------------------------------------------------------------
// module-level functions

/// Set rcParams: `values` maps key -> (kind, value) with kind one of
/// bool/num/str/color/numlist/strlist/colorlist/none.
#[pyfunction]
fn rc_set(values: &Bound<'_, PyDict>) -> PyResult<()> {
    let mut items: Vec<(String, RcValue)> = Vec::new();
    for (k, v) in values.iter() {
        let key: String = k.extract()?;
        let t = v.cast::<PyTuple>()?;
        let kind: String = t.get_item(0)?.extract()?;
        let val = t.get_item(1)?;
        let rv = match kind.as_str() {
            "bool" => RcValue::Bool(val.extract()?),
            "num" => RcValue::Num(val.extract()?),
            "str" => RcValue::Str(val.extract()?),
            "color" => RcValue::Color(rgba_of(&val)?),
            "numlist" => RcValue::NumList(val.extract()?),
            "strlist" => RcValue::StrList(val.extract()?),
            "colorlist" => {
                let l = val.cast::<PyList>()?;
                let mut v = Vec::new();
                for c in l.iter() {
                    v.push(rgba_of(&c)?);
                }
                RcValue::ColorList(v)
            }
            "none" => RcValue::None,
            _ => return Err(PyValueError::new_err(format!("bad rc kind {kind}"))),
        };
        items.push((key, rv));
    }
    rc_update(|rc| {
        for (k, v) in items {
            rc.map.insert(k, v);
        }
    });
    Ok(())
}

#[pyfunction]
fn rc_clear() {
    crate::mpl::rc::rc_replace(crate::mpl::rc::Rc::default());
}

/// Font list entries: (fname, index, name, style, variant, weight, stretch, size).
#[pyfunction]
fn set_font_list(entries: Vec<(String, u32, String, String, String, i32, String, String)>) {
    let list = entries
        .into_iter()
        .map(|(fname, index, name, style, variant, weight, stretch, size)| FontEntry {
            fname,
            index,
            name,
            style,
            variant,
            weight: NumOrName::Num(weight),
            stretch,
            size,
        })
        .collect();
    fonts::set_font_list(list);
}

/// `ttfFontProperty` (+ alternative names) for every face of a font file.
#[pyfunction]
fn font_entries(path: &str) -> PyResult<Vec<(String, u32, String, String, String, i32, String, String)>> {
    let bytes = std::fs::read(path).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let nfaces = if bytes.len() >= 12 && &bytes[0..4] == b"ttcf" { u32::from_be_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) } else { 1 };
    let mut out = Vec::new();
    for idx in 0..nfaces {
        let data = crate::font::FontData::from_bytes(bytes.clone(), idx).map_err(|e| PyValueError::new_err(e.to_string()))?;
        for e in fonts::ttf_font_properties(path, idx, &data) {
            out.push((e.fname, e.index, e.name, e.style, e.variant, e.weight.weight(), e.stretch, e.size));
        }
    }
    Ok(out)
}

#[pyfunction]
fn set_dot_uses_fma(v: bool) {
    crate::mpl::transforms::set_dot_uses_fma(v);
}

/// Resolve a font: (family list, style, variant, weight, stretch, size) -> paths.
#[pyfunction]
fn findfont(family: Vec<String>, style: String, variant: String, weight: &Bound<'_, PyAny>, stretch: String, size: f64) -> PyResult<Vec<(String, u32)>> {
    let w = if let Ok(n) = weight.extract::<i32>() { NumOrName::Num(n) } else { fonts::parse_weight(&weight.extract::<String>()?) };
    let prop = fonts::FontProperties {
        family,
        style,
        variant,
        weight: w,
        stretch: fonts::parse_stretch(&stretch),
        size,
        file: None,
        math_fontfamily: "dejavusans".into(),
    };
    Ok(fonts::FONT_MANAGER.write().unwrap().find_fonts_by_props(&prop))
}

#[allow(dead_code)]
fn text_handle(_t: &Shared<Text>) {}
#[allow(dead_code)]
fn patch_handle(_p: &Shared<Patch>) {}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFigure>()?;
    m.add_class::<PyLine2D>()?;
    m.add_function(wrap_pyfunction!(rc_set, m)?)?;
    m.add_function(wrap_pyfunction!(rc_clear, m)?)?;
    m.add_function(wrap_pyfunction!(set_font_list, m)?)?;
    m.add_function(wrap_pyfunction!(font_entries, m)?)?;
    m.add_function(wrap_pyfunction!(set_dot_uses_fma, m)?)?;
    m.add_function(wrap_pyfunction!(findfont, m)?)?;
    Ok(())
}
