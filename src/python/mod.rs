//! Python bindings (PyO3).

pub mod api;
pub mod collections;
pub mod contour;
pub mod ext;
pub mod foreign;
pub mod hist;
pub mod image;
pub mod kiwi;
pub mod lowlevel;
pub mod mathtext;
pub mod model;
pub mod pathmod;
pub mod tri;

use pyo3::prelude::*;

#[pymodule]
fn _mplrs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    lowlevel::register(m)?;
    model::register(m)?;
    api::register(m)?;
    collections::register(m)?;
    pathmod::register(m)?;
    image::register(m)?;
    tri::register(m)?;
    kiwi::register(m)?;
    contour::register(m)?;
    foreign::register(m)?;
    hist::register(m)?;
    mathtext::register(m)?;
    mathtext::register_scale(m)?;
    Ok(())
}
