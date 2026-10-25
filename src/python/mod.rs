//! Python bindings (PyO3).

pub mod lowlevel;
pub mod model;

use pyo3::prelude::*;

#[pymodule]
fn _mplrs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    lowlevel::register(m)?;
    model::register(m)?;
    Ok(())
}
