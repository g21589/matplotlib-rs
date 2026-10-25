//! The mathtext layout engine used by the Rust renderer: matplotlib's
//! mathtext parser, called from Python.

use std::sync::Arc;

use pyo3::prelude::*;

use crate::mpl::fonts::FontProperties;
use crate::mpl::mathtext::{set_engine, MathLayout, MathtextEngine};

struct PyMathtext {
    func: Py<PyAny>,
}

type Glyphs = Vec<(String, u32, f64, u32, f64, f64)>;

impl MathtextEngine for PyMathtext {
    fn layout(&self, s: &str, prop: &FontProperties, dpi: f64) -> Option<MathLayout> {
        Python::attach(|py| {
            let res = super::api::fontprops_to(py, prop).and_then(|p| {
                let r = self.func.bind(py).call1((s, dpi, p))?;
                r.extract::<(f64, f64, f64, Glyphs, Vec<(f64, f64, f64, f64)>)>()
            });
            match res {
                Ok((width, height, depth, glyphs, rects)) => Some(MathLayout { width, height, depth, glyphs, rects }),
                Err(e) => {
                    e.print(py);
                    None
                }
            }
        })
    }
}

/// Register `func(s, dpi, fontprops) -> (width, height, depth, glyphs,
/// rects)` as the mathtext layout engine (None unregisters it).
#[pyfunction]
fn set_mathtext_handler(func: Option<Py<PyAny>>) {
    set_engine(func.map(|f| Arc::new(PyMathtext { func: f }) as Arc<dyn MathtextEngine>));
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(set_mathtext_handler, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// scale transforms evaluated with numpy

fn scale_key(t: &crate::mpl::transforms::ScaleTransform) -> (&'static str, Vec<f64>) {
    use crate::mpl::transforms::ScaleTransform as S;
    match *t {
        S::Log { base, clip } => ("log", vec![base, if clip { 1.0 } else { 0.0 }]),
        S::InvertedLog { base } => ("invlog", vec![base]),
        S::Symlog { base, linthresh, linscale } => ("symlog", vec![base, linthresh, linscale]),
        S::InvertedSymlog { base, linthresh, linscale } => ("invsymlog", vec![base, linthresh, linscale]),
        S::Logit { clip } => ("logit", vec![if clip { 1.0 } else { 0.0 }]),
        S::Logistic => ("logistic", vec![]),
        S::AsinH { a } => ("asinh", vec![a]),
        S::InvertedAsinH { a } => ("invasinh", vec![a]),
    }
}

/// Register `func(kind, params, values) -> values` evaluating the
/// non-affine part of scale transforms (None unregisters it).
#[pyfunction]
fn set_scale_hook(func: Option<Py<PyAny>>) {
    use numpy::{PyArray1, PyReadonlyArray1};
    let Some(func) = func else {
        crate::mpl::transforms::set_scale_hook(None);
        return;
    };
    let hook = move |t: &crate::mpl::transforms::ScaleTransform, vals: &mut [f64]| -> bool {
        Python::attach(|py| {
            let (kind, params) = scale_key(t);
            let arr = PyArray1::from_slice(py, vals);
            let res = func
                .bind(py)
                .call1((kind, pyo3::types::PyTuple::new(py, &params).ok()?, arr))
                .and_then(|r| r.extract::<PyReadonlyArray1<f64>>().map(|a| a.as_array().to_vec()));
            match res {
                Ok(out) if out.len() == vals.len() => {
                    vals.copy_from_slice(&out);
                    Some(())
                }
                Ok(_) => None,
                Err(e) => {
                    e.print(py);
                    None
                }
            }
        })
        .is_some()
    };
    crate::mpl::transforms::set_scale_hook(Some(Arc::new(hook)));
}

pub fn register_scale(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(set_scale_hook, m)?)?;
    Ok(())
}
