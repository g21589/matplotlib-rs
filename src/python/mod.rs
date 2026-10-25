//! Python bindings (PyO3).

pub mod api;
pub mod collections;
pub mod ext;
pub mod lowlevel;
pub mod mathtext;
pub mod model;
pub mod pathmod;

use pyo3::prelude::*;

#[pymodule]
fn _mplrs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    lowlevel::register(m)?;
    model::register(m)?;
    api::register(m)?;
    collections::register(m)?;
    pathmod::register(m)?;
    mathtext::register(m)?;
    mathtext::register_scale(m)?;
    Ok(())
}
