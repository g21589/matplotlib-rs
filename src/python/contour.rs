//! Binding of the `mpl2014` contour generator port, wrapped by
//! `matplotlib_rs._contour`.

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

use crate::mpl::contour2014::{Mpl2014, Output};

#[pyclass(name = "Mpl2014ContourGenerator", module = "matplotlib_rs._mplrs", unsendable)]
pub struct PyMpl2014 {
    inner: Mpl2014,
    corner_mask: bool,
}

fn to_python<'py>(py: Python<'py>, out: Output) -> PyResult<Bound<'py, PyTuple>> {
    let vertices = PyList::empty(py);
    let codes = PyList::empty(py);
    for (v, c) in out.vertices.into_iter().zip(out.codes) {
        let n = v.len();
        let flat: Vec<f64> = v.into_iter().flat_map(|(x, y)| [x, y]).collect();
        vertices.append(PyArray1::from_vec(py, flat).reshape([n, 2])?)?;
        codes.append(PyArray1::from_vec(py, c))?;
    }
    PyTuple::new(py, [vertices.into_any(), codes.into_any()])
}

#[pymethods]
impl PyMpl2014 {
    /// contourpy's ``Mpl2014ContourGenerator(x, y, z, mask, *, corner_mask,
    /// x_chunk_size=0, y_chunk_size=0)``; *mask* may be None.
    #[new]
    #[pyo3(signature = (x, y, z, mask, *, corner_mask, x_chunk_size=0, y_chunk_size=0))]
    fn new(
        x: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray2<'_, f64>,
        z: PyReadonlyArray2<'_, f64>,
        mask: Option<PyReadonlyArray2<'_, bool>>,
        corner_mask: bool,
        x_chunk_size: i64,
        y_chunk_size: i64,
    ) -> PyResult<Self> {
        let shape = z.as_array().shape().to_vec();
        let (ny, nx) = (shape[0], shape[1]);
        if x.as_array().shape() != shape.as_slice() || y.as_array().shape() != shape.as_slice() {
            return Err(PyValueError::new_err("x, y and z arrays must have the same shape"));
        }
        if nx < 2 || ny < 2 {
            return Err(PyValueError::new_err("x, y and z must all be at least 2x2 arrays"));
        }
        if x_chunk_size < 0 || y_chunk_size < 0 {
            return Err(PyValueError::new_err("x_chunk_size and y_chunk_size cannot be negative"));
        }
        let mask: Option<Vec<bool>> = match mask {
            Some(m) => {
                if m.as_array().shape() != shape.as_slice() {
                    return Err(PyValueError::new_err(
                        "If mask is set it must be a 2D array with the same shape as z",
                    ));
                }
                Some(m.as_array().iter().copied().collect())
            }
            None => None,
        };
        let flat = |a: &PyReadonlyArray2<'_, f64>| a.as_array().iter().copied().collect::<Vec<f64>>();
        Ok(PyMpl2014 {
            inner: Mpl2014::new(
                flat(&x),
                flat(&y),
                flat(&z),
                nx,
                ny,
                mask.as_deref(),
                corner_mask,
                x_chunk_size as usize,
                y_chunk_size as usize,
            ),
            corner_mask,
        })
    }

    /// Contour lines at *level*: ``([vertices], [codes])``.
    fn lines<'py>(&mut self, py: Python<'py>, level: f64) -> PyResult<Bound<'py, PyTuple>> {
        to_python(py, self.inner.lines(level))
    }

    fn create_contour<'py>(&mut self, py: Python<'py>, level: f64) -> PyResult<Bound<'py, PyTuple>> {
        self.lines(py, level)
    }

    /// Filled contours between *lower_level* and *upper_level*:
    /// ``([vertices], [codes])``, each polygon followed by its holes.
    fn filled<'py>(&mut self, py: Python<'py>, lower_level: f64, upper_level: f64) -> PyResult<Bound<'py, PyTuple>> {
        if lower_level.is_nan() || upper_level.is_nan() {
            return Err(PyValueError::new_err("lower_level and upper_level cannot be NaN"));
        }
        if lower_level >= upper_level {
            return Err(PyValueError::new_err("upper_level must be larger than lower_level"));
        }
        to_python(py, self.inner.filled(lower_level, upper_level))
    }

    fn create_filled_contour<'py>(
        &mut self,
        py: Python<'py>,
        lower_level: f64,
        upper_level: f64,
    ) -> PyResult<Bound<'py, PyTuple>> {
        self.filled(py, lower_level, upper_level)
    }

    #[getter]
    fn chunk_size(&self) -> (usize, usize) {
        self.inner.chunk_size()
    }

    #[getter]
    fn chunk_count(&self) -> (usize, usize) {
        self.inner.chunk_count()
    }

    #[getter]
    fn corner_mask(&self) -> bool {
        self.corner_mask
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMpl2014>()?;
    Ok(())
}
