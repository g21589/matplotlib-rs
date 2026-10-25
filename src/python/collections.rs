//! Bindings for collections, images and legends.

use pyo3::prelude::*;

use crate::mpl::artist::{shared, Shared};
use crate::mpl::collections::Collection;
use crate::mpl::image::AxesImage;
use crate::mpl::legend::Legend;

#[pyclass(name = "Collection", module = "matplotlib_rs._mplrs")]
pub struct PyCollection {
    pub inner: Shared<Collection>,
}

#[pymethods]
impl PyCollection {
    #[new]
    fn new() -> Self {
        PyCollection { inner: shared(Collection::default()) }
    }
}

#[pyclass(name = "AxesImage", module = "matplotlib_rs._mplrs")]
pub struct PyImage {
    pub inner: Shared<AxesImage>,
}

#[pymethods]
impl PyImage {
    #[new]
    fn new() -> Self {
        PyImage { inner: shared(AxesImage::default()) }
    }
}

#[pyclass(name = "Legend", module = "matplotlib_rs._mplrs")]
pub struct PyLegend {
    pub inner: Shared<Legend>,
}

#[pymethods]
impl PyLegend {
    #[new]
    fn new() -> Self {
        PyLegend { inner: shared(Legend { zorder: 5.0 }) }
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCollection>()?;
    m.add_class::<PyImage>()?;
    m.add_class::<PyLegend>()?;
    Ok(())
}
