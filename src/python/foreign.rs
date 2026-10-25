//! Drawing the Rust model into a Python renderer (matplotlib's vector
//! backends): every renderer call becomes a call of
//! `matplotlib_rs._foreign.Target`, which rebuilds matplotlib's objects.

use std::sync::Arc;

use numpy::{PyArray1, PyArray2, PyArray3, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

use super::api::fontprops_to;
use super::ext::record_error;
use crate::agg::pixfmt::Rgba;
use crate::agg::stroke::{LineCap, LineJoin};
use crate::mpl::artist::GcState;
use crate::mpl::fonts::FontProperties;
use crate::mpl::path::Path;
use crate::mpl::renderer::{ForeignRenderer, MText, OwnerRef};
use crate::mpl::transforms::Mat;

/// A Python renderer wrapped for the Rust core.
pub struct PyForeign {
    pub target: Py<PyAny>,
}

/// The Python object an artist mirrors (held weakly).
pub struct PyOwner(pub Py<PyAny>);

impl PyForeign {
    pub fn new(renderer: &Bound<'_, PyAny>) -> PyResult<PyForeign> {
        let m = renderer.py().import("matplotlib_rs._foreign")?;
        let t = m.getattr("Target")?.call1((renderer,))?;
        Ok(PyForeign { target: t.unbind() })
    }

    /// The wrapped Python renderer.
    pub fn renderer<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.target.bind(py).getattr("renderer")
    }

    fn call<'py, A: pyo3::call::PyCallArgs<'py>>(&self, py: Python<'py>, name: &str, args: A) -> Option<Bound<'py, PyAny>> {
        match self.target.bind(py).call_method1(name, args) {
            Ok(v) => Some(v),
            Err(e) => {
                record_error(e);
                None
            }
        }
    }
}

fn rgba<'py>(py: Python<'py>, c: Rgba) -> PyResult<Bound<'py, PyTuple>> {
    PyTuple::new(py, [c.r, c.g, c.b, c.a])
}

fn mat<'py>(py: Python<'py>, m: &Mat) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let flat: Vec<f64> = m.iter().flatten().copied().collect();
    Ok(PyArray1::from_vec(py, flat).reshape([3, 3])?)
}

fn path<'py>(py: Python<'py>, p: &Path) -> PyResult<Bound<'py, PyTuple>> {
    let n = p.vertices.len();
    let flat: Vec<f64> = p.vertices.iter().flatten().copied().collect();
    let verts = PyArray1::from_vec(py, flat).reshape([n, 2])?;
    let codes: Bound<'py, PyAny> = match &p.codes {
        Some(c) => PyArray1::from_slice(py, c).into_any(),
        None => py.None().into_bound(py),
    };
    PyTuple::new(
        py,
        [
            verts.into_any(),
            codes,
            p.should_simplify.into_pyobject(py)?.to_owned().into_any(),
            p.simplify_threshold.into_pyobject(py)?.into_any(),
        ],
    )
}

fn gc<'py>(py: Python<'py>, g: &GcState) -> PyResult<Bound<'py, PyTuple>> {
    let cap = match g.capstyle {
        LineCap::Butt => "butt",
        LineCap::Round => "round",
        LineCap::Square => "projecting",
    };
    let join = match g.joinstyle {
        LineJoin::Round => "round",
        LineJoin::Bevel => "bevel",
        _ => "miter",
    };
    let none = || py.None().into_bound(py);
    let cliprect = match g.cliprect {
        Some(b) => (g.clip_owner, (b.x0, b.y0, b.x1, b.y1)).into_pyobject(py)?.into_any(),
        None => none(),
    };
    let clippath = match &g.clippath {
        Some(cp) => {
            let t = &cp.trans;
            let m: Mat = [[t.sx, t.shx, t.tx], [t.shy, t.sy, t.ty], [0.0, 0.0, 1.0]];
            let key = (Arc::as_ptr(&cp.path) as usize, m.iter().flatten().map(|v| v.to_bits()).collect::<Vec<u64>>());
            let p = path(py, &cp.path)?;
            let mut items: Vec<Bound<'py, PyAny>> = vec![key.into_pyobject(py)?.into_any()];
            items.extend(p.iter());
            items.push(mat(py, &m)?.into_any());
            PyTuple::new(py, items)?.into_any()
        }
        None => none(),
    };
    let dashes = match &g.dashes.1 {
        Some(d) => PyTuple::new(py, [g.dashes.0.into_pyobject(py)?.into_any(), PyList::new(py, d)?.into_any()])?,
        None => PyTuple::new(py, [g.dashes.0.into_pyobject(py)?.into_any(), none()])?,
    };
    let snap = match g.snap {
        Some(b) => b.into_pyobject(py)?.to_owned().into_any(),
        None => none(),
    };
    let sketch = match g.sketch {
        Some(s) => s.into_pyobject(py)?.into_any(),
        None => none(),
    };
    let hatch = match (&g.hatch_pattern, &g.hatch) {
        (Some(pat), Some((_, color, lw))) => {
            PyTuple::new(
                py,
                [
                    pat.into_pyobject(py)?.into_any(),
                    rgba(py, *color)?.into_any(),
                    lw.into_pyobject(py)?.into_any(),
                    g.hatch_np.into_pyobject(py)?.to_owned().into_any(),
                ],
            )?
            .into_any()
        }
        _ => none(),
    };
    PyTuple::new(
        py,
        [
            g.alpha.into_pyobject(py)?.into_any(),
            g.forced_alpha.into_pyobject(py)?.to_owned().into_any(),
            g.antialiased.into_pyobject(py)?.to_owned().into_any(),
            cap.into_pyobject(py)?.into_any(),
            join.into_pyobject(py)?.into_any(),
            cliprect,
            clippath,
            dashes.into_any(),
            g.linewidth.into_pyobject(py)?.into_any(),
            rgba(py, g.rgb)?.into_any(),
            snap,
            sketch,
            hatch,
        ],
    )
}

fn face<'py>(py: Python<'py>, f: Option<Rgba>) -> PyResult<Bound<'py, PyAny>> {
    Ok(match f {
        Some(c) => rgba(py, c)?.into_any(),
        None => py.None().into_bound(py),
    })
}

macro_rules! tryp {
    ($e:expr) => {
        match $e {
            Ok(v) => v,
            Err(e) => {
                record_error(e);
                return Default::default();
            }
        }
    };
}

impl ForeignRenderer for PyForeign {
    fn draw_path(&self, g: &GcState, p: &Path, trans: &Mat, f: Option<Rgba>) {
        Python::attach(|py| {
            let args = (tryp!(gc(py, g)), tryp!(path(py, p)), tryp!(mat(py, trans)), tryp!(face(py, f)));
            self.call(py, "draw_path", args);
        })
    }

    fn draw_markers(&self, g: &GcState, marker: &Path, marker_trans: &Mat, p: &Path, trans: &Mat, f: Option<Rgba>) {
        Python::attach(|py| {
            let args = (
                tryp!(gc(py, g)),
                tryp!(path(py, marker)),
                tryp!(mat(py, marker_trans)),
                tryp!(path(py, p)),
                tryp!(mat(py, trans)),
                tryp!(face(py, f)),
            );
            self.call(py, "draw_markers", args);
        })
    }

    fn draw_path_collection(
        &self,
        g: &GcState,
        master_transform: &Mat,
        paths: &[Path],
        transforms: &[Mat],
        offsets: &[[f64; 2]],
        offset_trans: &Mat,
        facecolors: &[Rgba],
        edgecolors: &[Rgba],
        linewidths: &[f64],
        linestyles: &[(f64, Option<Vec<f64>>)],
        antialiaseds: &[bool],
    ) {
        Python::attach(|py| {
            let ps: Vec<Bound<'_, PyTuple>> = tryp!(paths.iter().map(|p| path(py, p)).collect::<PyResult<_>>());
            let ts: Vec<Bound<'_, PyArray2<f64>>> = tryp!(transforms.iter().map(|m| mat(py, m)).collect::<PyResult<_>>());
            let fc: Vec<(f64, f64, f64, f64)> = facecolors.iter().map(|c| (c.r, c.g, c.b, c.a)).collect();
            let ec: Vec<(f64, f64, f64, f64)> = edgecolors.iter().map(|c| (c.r, c.g, c.b, c.a)).collect();
            let args = (
                tryp!(gc(py, g)),
                tryp!(mat(py, master_transform)),
                ps,
                ts,
                offsets.to_vec(),
                tryp!(mat(py, offset_trans)),
                fc,
                ec,
                linewidths.to_vec(),
                linestyles.to_vec(),
                antialiaseds.to_vec(),
            );
            self.call(py, "draw_path_collection", args);
        })
    }

    fn draw_image(&self, g: &GcState, x: f64, y: f64, data: &[u8], w: usize, h: usize) {
        Python::attach(|py| {
            let im: Bound<'_, PyArray3<u8>> = tryp!(PyArray1::from_slice(py, data).reshape([h, w, 4]));
            self.call(py, "draw_image", (tryp!(gc(py, g)), x, y, im));
        })
    }

    fn draw_text(&self, g: &GcState, x: f64, y: f64, s: &str, prop: &FontProperties, angle: f64, ismath: bool, mtext: Option<&MText<'_>>) {
        Python::attach(|py| {
            let owner: Bound<'_, PyAny> = match mtext.and_then(|m| m.owner).and_then(|o| o.0.downcast_ref::<PyOwner>()) {
                Some(o) => o.0.bind(py).clone(),
                None => py.None().into_bound(py),
            };
            let anchor = mtext.map(|m| (m.x, m.y, m.ha, m.va, m.rotation_mode));
            let args = (tryp!(gc(py, g)), x, y, s, tryp!(fontprops_to(py, prop)), angle, ismath, owner, anchor);
            self.call(py, "draw_text", args);
        })
    }

    fn text_metrics(&self, s: &str, prop: &FontProperties, ismath: bool) -> (f64, f64, f64) {
        Python::attach(|py| {
            let fp = tryp!(fontprops_to(py, prop));
            match self.call(py, "text_metrics", (s, fp, ismath)) {
                Some(v) => tryp!(v.extract()),
                None => (0.0, 0.0, 0.0),
            }
        })
    }

    fn open_group(&self, name: &str, gid: Option<&str>) {
        Python::attach(|py| {
            self.call(py, "open_group", (name, gid));
        })
    }

    fn close_group(&self, name: &str) {
        Python::attach(|py| {
            self.call(py, "close_group", (name,));
        })
    }

    fn artist_start(&self) {
        Python::attach(|py| {
            self.call(py, "artist_start", ());
        })
    }

    fn flipy(&self) -> bool {
        self.flip().0
    }

    fn canvas_height(&self) -> f64 {
        self.flip().1
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl PyForeign {
    fn flip(&self) -> (bool, f64) {
        Python::attach(|py| match self.call(py, "flip", ()) {
            Some(v) => tryp!(v.extract()),
            None => (true, 0.0),
        })
    }
}

/// `push_foreign(renderer)`: layout by the Rust core uses *renderer*'s text
/// metrics until `pop_foreign()`.
#[pyfunction]
fn push_foreign(renderer: &Bound<'_, PyAny>) -> PyResult<()> {
    crate::mpl::renderer::push_foreign(Arc::new(PyForeign::new(renderer)?));
    Ok(())
}

#[pyfunction]
fn pop_foreign() {
    crate::mpl::renderer::pop_foreign();
}

/// The `mtext` reference of a text (a weak reference to its Python object).
pub fn owner_ref(weak: Py<PyAny>) -> OwnerRef {
    OwnerRef(Arc::new(PyOwner(weak)))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(push_foreign, m)?)?;
    m.add_function(wrap_pyfunction!(pop_foreign, m)?)?;
    Ok(())
}
