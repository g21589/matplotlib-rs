//! Python artists inside the Rust model: they are drawn by their own
//! `draw` method, through a renderer proxy for the figure's renderer.

use std::any::Any;
use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::Arc;

use pyo3::prelude::*;

use super::lowlevel::PyRendererAgg;
use crate::mpl::artist::ExtArtist;
use crate::mpl::renderer::Renderer;
use crate::mpl::transforms::Bbox;

#[derive(Debug)]
pub struct PyExt {
    pub obj: Py<PyAny>,
}

thread_local! {
    /// The first Python exception raised by an external artist during a
    /// draw; re-raised when the draw returns to Python.
    static DRAW_ERROR: RefCell<Option<PyErr>> = const { RefCell::new(None) };
}

pub(crate) fn record_error(e: PyErr) {
    DRAW_ERROR.with(|d| {
        let mut d = d.borrow_mut();
        if d.is_none() {
            *d = Some(e);
        }
    })
}

/// Raise the exception recorded during the last draw, if any.
pub fn take_error() -> PyResult<()> {
    match DRAW_ERROR.with(|d| d.borrow_mut().take()) {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// Run `f` with a Python proxy for `r`, invalidated when `f` returns.
///
/// A renderer forwarding to a Python renderer is passed as that renderer.
fn with_target<R>(py: Python<'_>, r: &mut Renderer, f: impl FnOnce(&Bound<'_, PyAny>) -> PyResult<R>) -> PyResult<R> {
    if let Some(fr) = r.foreign.as_ref().and_then(|f| f.as_any().downcast_ref::<super::foreign::PyForeign>()) {
        let target = fr.renderer(py)?;
        return f(&target);
    }
    let alive = Rc::new(Cell::new(true));
    let t = Bound::new(py, PyRendererAgg::borrowed(r, alive.clone()))?;
    let res = f(t.as_any());
    alive.set(false);
    res
}

impl ExtArtist for PyExt {
    fn zorder(&self) -> f64 {
        Python::attach(|py| self.obj.bind(py).getattr("zorder").and_then(|v| v.extract()).unwrap_or(0.0))
    }

    fn visible(&self) -> bool {
        Python::attach(|py| self.obj.bind(py).call_method0("get_visible").and_then(|v| v.extract()).unwrap_or(true))
    }

    fn draw(&self, r: &mut Renderer) {
        Python::attach(|py| {
            if let Err(e) = with_target(py, r, |t| self.obj.bind(py).call_method1("_rs_draw", (t,)).map(|_| ())) {
                record_error(e);
            }
        })
    }

    fn ident(&self) -> usize {
        self.obj.as_ptr() as usize
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn sticky_edges(&self) -> (Vec<f64>, Vec<f64>) {
        Python::attach(|py| match self.obj.bind(py).call_method0("_rs_sticky_edges").and_then(|v| v.extract()) {
            Ok(v) => v,
            Err(e) => {
                record_error(e);
                (Vec::new(), Vec::new())
            }
        })
    }

    fn tightbbox(&self, r: &mut Renderer, for_axes: bool) -> Option<Bbox> {
        Python::attach(|py| {
            let res = with_target(py, r, |t| {
                self.obj.bind(py).call_method1("_rs_tightbbox", (t, for_axes))?.extract::<Option<(f64, f64, f64, f64)>>()
            });
            match res {
                Ok(v) => v.map(|(x0, y0, x1, y1)| Bbox::new(x0, y0, x1, y1)),
                Err(e) => {
                    record_error(e);
                    None
                }
            }
        })
    }
}

/// The Python object of an external artist created from Python.
pub fn py_object(e: &Arc<dyn ExtArtist>) -> Option<&Py<PyAny>> {
    e.as_any().downcast_ref::<PyExt>().map(|p| &p.obj)
}

pub fn ext_from(obj: &Bound<'_, PyAny>) -> Arc<dyn ExtArtist> {
    Arc::new(PyExt { obj: obj.clone().unbind() })
}

/// A Python transform (any object with `get_matrix()`), evaluated when the
/// Rust core draws an artist using it.
pub struct PyTransform {
    pub obj: Py<PyAny>,
}

impl crate::mpl::artist::ExtTransform for PyTransform {
    fn matrix(&self) -> Option<crate::mpl::transforms::Mat> {
        Python::attach(|py| {
            let r = self
                .obj
                .bind(py)
                .call_method0("get_matrix")
                .and_then(|m| m.extract::<numpy::PyReadonlyArray2<f64>>().map(|a| a.as_array().to_owned()));
            match r {
                Ok(a) if a.shape() == [3, 3] => {
                    Some([[a[[0, 0]], a[[0, 1]], a[[0, 2]]], [a[[1, 0]], a[[1, 1]], a[[1, 2]]], [0.0, 0.0, 1.0]])
                }
                Ok(_) => None,
                Err(e) => {
                    record_error(e);
                    None
                }
            }
        })
    }

    fn ident(&self) -> usize {
        self.obj.as_ptr() as usize
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
