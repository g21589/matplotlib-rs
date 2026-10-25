//! Python bindings (PyO3).

pub mod lowlevel;

use pyo3::prelude::*;

#[pymodule]
fn _mplrs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    lowlevel::register(m)?;
    Ok(())
}
