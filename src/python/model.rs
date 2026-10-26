//! Bindings for the artist model.  The Python layer (`matplotlib_rs`)
//! wraps these handles with matplotlib's public API.

use numpy::PyReadonlyArray1;
use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods, PyList, PyTuple};

use crate::agg::pixfmt::Rgba;
use crate::mpl::artist::{shared, ClipBoxSpec, Lock, Shared, TSpec};
use crate::mpl::colors::ColorSpec;
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

pub fn opt_f64(obj: &Bound<'_, PyAny>) -> PyResult<Option<f64>> {
    if obj.is_none() {
        Ok(None)
    } else {
        Ok(Some(obj.extract()?))
    }
}

pub fn tspec_from(obj: &Bound<'_, PyAny>) -> PyResult<TSpec> {
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
        "py" => TSpec::Ext(crate::mpl::artist::ExtT(std::sync::Arc::new(super::ext::PyTransform {
            obj: t.get_item(1)?.unbind(),
        }))),
        _ => return Err(PyValueError::new_err(format!("unknown transform kind {kind}"))),
    })
}

/// The Python form of a transform spec (inverse of `tspec_from`, plus the
/// read-only forms used by Rust-owned artists).
pub fn tspec_to<'py>(py: Python<'py>, t: &TSpec) -> PyResult<Bound<'py, PyAny>> {
    use crate::mpl::artist::Which;
    let w = |w: &Which| match w {
        Which::Grid => "grid",
        Which::Tick1 => "tick1",
        Which::Tick2 => "tick2",
    };
    Ok(match t {
        TSpec::Identity => "identity".into_pyobject(py)?.into_any(),
        TSpec::Data => "data".into_pyobject(py)?.into_any(),
        TSpec::Axes => "axes".into_pyobject(py)?.into_any(),
        TSpec::Figure => "figure".into_pyobject(py)?.into_any(),
        TSpec::DpiScale => "dpi".into_pyobject(py)?.into_any(),
        TSpec::XAxis(x) => ("xaxis", w(x)).into_pyobject(py)?.into_any(),
        TSpec::YAxis(x) => ("yaxis", w(x)).into_pyobject(py)?.into_any(),
        TSpec::XText1(p) => ("xtext1", *p).into_pyobject(py)?.into_any(),
        TSpec::XText2(p) => ("xtext2", *p).into_pyobject(py)?.into_any(),
        TSpec::YText1(p) => ("ytext1", *p).into_pyobject(py)?.into_any(),
        TSpec::YText2(p) => ("ytext2", *p).into_pyobject(py)?.into_any(),
        TSpec::Title(p) => ("title", *p).into_pyobject(py)?.into_any(),
        TSpec::Affine(m) => ("affine", vec![m[0].to_vec(), m[1].to_vec()]).into_pyobject(py)?.into_any(),
        TSpec::ScaledTranslation(x, y) => ("scaled_translation", *x, *y).into_pyobject(py)?.into_any(),
        TSpec::Then(a, b) => ("then", tspec_to(py, a)?, tspec_to(py, b)?).into_pyobject(py)?.into_any(),
        TSpec::Blend(a, b) => ("blend", tspec_to(py, a)?, tspec_to(py, b)?).into_pyobject(py)?.into_any(),
        TSpec::Fixed(tr) => {
            let m = tr.get_affine();
            ("affine", vec![m[0].to_vec(), m[1].to_vec()]).into_pyobject(py)?.into_any()
        }
        TSpec::Spine(i) => ("spine", *i).into_pyobject(py)?.into_any(),
        TSpec::Ext(e) => match e.0.as_any().downcast_ref::<super::ext::PyTransform>() {
            Some(p) => ("py", p.obj.bind(py).clone()).into_pyobject(py)?.into_any(),
            None => "identity".into_pyobject(py)?.into_any(),
        },
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
    if let Ok((verts, codes)) = obj.extract::<(Vec<[f64; 2]>, Option<Vec<u8>>)>() {
        let p = crate::mpl::path::new_path(verts, codes, crate::mpl::path::SimplifyParams::default());
        return Ok(MarkerStyle::new(MarkerSpec::Path(p), fs));
    }
    Err(PyTypeError::new_err("unsupported marker"))
}

pub fn base_set(base: &mut crate::mpl::artist::ArtistBase, name: &str, v: &Bound<'_, PyAny>) -> PyResult<bool> {
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
        "url" => base.url = if v.is_none() { None } else { Some(v.extract()?) },
        "animated" => base.animated = v.extract()?,
        "rasterized" => base.rasterized = v.extract()?,
        "sketch_params" => base.sketch = if v.is_none() { None } else { Some(v.extract()?) },
        "clip_path" => {
            // None or (Path, transform spec, clip path object):
            // `TransformedPath(path, transform)`
            base.clip_path = if v.is_none() {
                None
            } else {
                let t = v.cast::<PyTuple>()?;
                Some(crate::mpl::artist::ClipPathSpec {
                    path: std::sync::Arc::new(super::pathmod::path_from_py(&t.get_item(0)?)?),
                    trans: tspec_from(&t.get_item(1)?)?,
                    source: Some(crate::mpl::artist::ClipSource(std::sync::Arc::new(t.get_item(2)?.unbind()))),
                })
            };
        }
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
        "dash_pattern" => {
            let (o, d): (f64, Option<Vec<f64>>) = v.extract()?;
            l.dash_pattern = (o, d);
        }
        "color" => l.color = color_from(v)?,
        "gapcolor" => l.gapcolor = if v.is_none() { None } else { Some(color_from(v)?) },
        "marker" => {
            let fs = l.marker.fillstyle;
            l.set_marker(marker_from(v, fs)?);
        }
        "marker_parts" => {
            let t = v.cast::<PyTuple>()?;
            let mk_path = |vv: &Bound<'_, PyAny>, cc: &Bound<'_, PyAny>| -> PyResult<crate::mpl::path::Path> {
                let verts: Vec<[f64; 2]> = vv.extract()?;
                let codes: Option<Vec<u8>> = cc.extract()?;
                Ok(crate::mpl::path::Path::new(verts, codes))
            };
            let mtx = |m: &Bound<'_, PyAny>| -> PyResult<crate::mpl::transforms::Affine2D> {
                let r: Vec<Vec<f64>> = m.extract()?;
                Ok(crate::mpl::transforms::Affine2D([[r[0][0], r[0][1], r[0][2]], [r[1][0], r[1][1], r[1][2]], [0.0, 0.0, 1.0]]))
            };
            let path = mk_path(&t.get_item(0)?, &t.get_item(1)?)?;
            let tr = mtx(&t.get_item(2)?)?;
            let alt = t.get_item(3)?;
            let (alt_path, alt_tr) = if alt.is_none() {
                (None, None)
            } else {
                (Some(mk_path(&alt, &t.get_item(4)?)?), Some(mtx(&t.get_item(5)?)?))
            };
            let join = crate::mpl::artist::parse_joinstyle(&t.get_item(6)?.extract::<String>()?);
            let cap = crate::mpl::artist::parse_capstyle(&t.get_item(7)?.extract::<String>()?);
            let sn = t.get_item(8)?;
            let snap = if sn.is_none() {
                crate::mpl::markers::SnapThreshold::None
            } else {
                let v: f64 = sn.extract()?;
                if v.is_infinite() {
                    crate::mpl::markers::SnapThreshold::Inf
                } else {
                    crate::mpl::markers::SnapThreshold::Value(v)
                }
            };
            let filled: bool = t.get_item(9)?.extract()?;
            let fs = FillStyle::parse(&t.get_item(10)?.extract::<String>()?).unwrap_or(FillStyle::Full);
            let is_pixel: bool = t.get_item(11)?.extract()?;
            l.set_marker(MarkerStyle::from_parts(path, tr, alt_path, alt_tr, join, cap, snap, filled, fs, is_pixel));
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
        "markevery" => {
            use crate::mpl::lines::MarkEvery;
            l.markevery = if v.is_none() {
                None
            } else {
                let t = v.cast::<PyTuple>()?;
                let kind: String = t.get_item(0)?.extract()?;
                Some(match kind.as_str() {
                    "every" => MarkEvery::Every(t.get_item(1)?.extract()?, t.get_item(2)?.extract()?),
                    "fraction" => MarkEvery::Fraction(t.get_item(1)?.extract()?, t.get_item(2)?.extract()?),
                    _ => MarkEvery::Indices(t.get_item(1)?.extract()?),
                })
            };
        }
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

pub fn colorspec_py<'py>(py: Python<'py>, c: ColorSpec) -> PyResult<Bound<'py, PyAny>> {
    Ok(match c {
        ColorSpec::Rgba(c) => PyTuple::new(py, [c.r, c.g, c.b, c.a])?.into_any(),
        ColorSpec::None => "none".into_pyobject(py)?.into_any(),
        ColorSpec::Auto => "auto".into_pyobject(py)?.into_any(),
    })
}

fn line_get<'py>(py: Python<'py>, l: &mut Line2D, name: &str) -> PyResult<Bound<'py, PyAny>> {
    Ok(match name {
        "xdata" => PyList::new(py, &l.xorig)?.into_any(),
        "ydata" => PyList::new(py, &l.yorig)?.into_any(),
        "xydata" => {
            let xy = l.get_xydata(false).to_vec();
            PyList::new(py, xy.iter().map(|p| (p[0], p[1])))?.into_any()
        }
        "linewidth" => l.linewidth.into_pyobject(py)?.into_any(),
        "linestyle" => l.linestyle.clone().into_pyobject(py)?.into_any(),
        "color" => colorspec_py(py, l.color)?,
        "markersize" => l.markersize.into_pyobject(py)?.into_any(),
        "markeredgewidth" => l.markeredgewidth.into_pyobject(py)?.into_any(),
        "markeredgecolor" => colorspec_py(py, l.markeredgecolor)?,
        "markerfacecolor" => colorspec_py(py, l.markerfacecolor)?,
        "markerfacecoloralt" => colorspec_py(py, l.markerfacecoloralt)?,
        "marker" => match &l.marker.spec {
            MarkerSpec::Named(s) => s.clone().into_pyobject(py)?.into_any(),
            MarkerSpec::Tuple(n, st, a) => (*n, *st, *a).into_pyobject(py)?.into_any(),
            MarkerSpec::Path(_) => "path".into_pyobject(py)?.into_any(),
        },
        "fillstyle" => format!("{:?}", l.marker.fillstyle).to_lowercase().into_pyobject(py)?.into_any(),
        "visible" => l.base.visible.into_pyobject(py)?.to_owned().into_any(),
        "alpha" => l.base.alpha.into_pyobject(py)?.into_any(),
        "zorder" => l.base.zorder.into_pyobject(py)?.into_any(),
        "label" => l.base.label.clone().into_pyobject(py)?.into_any(),
        "antialiased" => l.antialiased.into_pyobject(py)?.to_owned().into_any(),
        "drawstyle" => match l.drawstyle {
            DrawStyle::Default => "default",
            DrawStyle::StepsPre => "steps-pre",
            DrawStyle::StepsMid => "steps-mid",
            DrawStyle::StepsPost => "steps-post",
        }
        .into_pyobject(py)?
        .into_any(),
        "dash_pattern" => (l.dash_pattern.0, l.dash_pattern.1.clone()).into_pyobject(py)?.into_any(),
        "clip_on" => l.base.clip_on.into_pyobject(py)?.to_owned().into_any(),
        "transform" => tspec_to(py, &l.base.transform)?,
        _ => return Err(PyKeyError::new_err(format!("Line2D has no property {name:?}"))),
    })
}

pub fn f64_vec(v: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
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

/// A fingerprint of a value's state (its `Debug` form, hashed as written).
pub fn state_hash<T: std::fmt::Debug>(v: &T) -> u64 {
    use std::hash::Hasher;
    struct W(std::collections::hash_map::DefaultHasher);
    impl std::fmt::Write for W {
        fn write_str(&mut self, s: &str) -> std::fmt::Result {
            self.0.write(s.as_bytes());
            Ok(())
        }
    }
    let mut w = W(std::collections::hash_map::DefaultHasher::new());
    let _ = std::fmt::write(&mut w, format_args!("{v:?}"));
    w.0.finish()
}

#[pymethods]
impl PyLine2D {
    /// A fingerprint of the line's state (changes whenever the line does).
    fn state_hash(&self) -> u64 {
        state_hash(&*self.inner.lk())
    }

    #[new]
    fn new(x: &Bound<'_, PyAny>, y: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(PyLine2D { inner: shared(Line2D::new(f64_vec(x)?, f64_vec(y)?)) })
    }

    fn set(&self, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        line_set(&mut self.inner.lk(), name, value)
    }

    fn get<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
        line_get(py, &mut self.inner.lk(), name)
    }

    fn ident(&self) -> usize {
        std::sync::Arc::as_ptr(&self.inner) as usize
    }

    fn set_many(&self, props: &Bound<'_, PyDict>) -> PyResult<()> {
        let mut l = self.inner.lk();
        for (k, v) in props.iter() {
            line_set(&mut l, &k.extract::<String>()?, &v)?;
        }
        Ok(())
    }

    /// Copy a validated style while giving the new artist independent data.
    fn copy_with_data(&self, x: &Bound<'_, PyAny>, y: &Bound<'_, PyAny>) -> PyResult<Self> {
        let mut line = self.inner.lk().clone();
        line.set_data(f64_vec(x)?, f64_vec(y)?);
        Ok(Self { inner: shared(line) })
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
    m.add_class::<PyLine2D>()?;
    m.add_function(wrap_pyfunction!(rc_set, m)?)?;
    m.add_function(wrap_pyfunction!(rc_clear, m)?)?;
    m.add_function(wrap_pyfunction!(set_font_list, m)?)?;
    m.add_function(wrap_pyfunction!(font_entries, m)?)?;
    m.add_function(wrap_pyfunction!(set_dot_uses_fma, m)?)?;
    m.add_function(wrap_pyfunction!(findfont, m)?)?;
    Ok(())
}
