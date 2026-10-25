//! Bindings of the `_tri` port (`matplotlib_rs._tri`) and of the Delaunay
//! fallback of `_qhull`.

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::mpl::tri::{self, Triangulation, TriContourGenerator, TrapezoidMapTriFinder, XY};

#[pyclass(name = "Triangulation", module = "matplotlib_rs._mplrs")]
pub struct PyTriangulation {
    inner: Triangulation,
}

fn vec2<T: Copy + numpy::Element>(a: &PyReadonlyArray2<'_, T>) -> Vec<T> {
    a.as_array().iter().copied().collect()
}

#[pymethods]
impl PyTriangulation {
    /// `Triangulation(x, y, triangles, mask, edges, neighbors)` with
    /// already-validated arrays (see `matplotlib_rs._tri`); `mask`, `edges`
    /// and `neighbors` may be empty.
    #[new]
    fn new(
        x: PyReadonlyArray1<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
        triangles: PyReadonlyArray2<'_, i32>,
        mask: PyReadonlyArray1<'_, bool>,
        edges: PyReadonlyArray2<'_, i32>,
        neighbors: PyReadonlyArray2<'_, i32>,
    ) -> Self {
        PyTriangulation {
            inner: Triangulation::new(
                x.as_array().to_vec(),
                y.as_array().to_vec(),
                vec2(&triangles),
                mask.as_array().to_vec(),
                vec2(&edges),
                vec2(&neighbors),
            ),
        }
    }

    fn calculate_plane_coefficients<'py>(&self, py: Python<'py>, z: PyReadonlyArray1<'_, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let z = z.as_array().to_vec();
        if z.len() != self.inner.npoints() {
            return Err(PyValueError::new_err(
                "z must be a 1D array with the same length as the triangulation x and y arrays",
            ));
        }
        let p = self.inner.calculate_plane_coefficients(&z);
        Ok(PyArray1::from_vec(py, p).reshape([self.inner.ntri(), 3])?)
    }

    fn get_edges<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<i32>>> {
        let e = self.inner.get_edges().to_vec();
        let n = e.len() / 2;
        Ok(PyArray1::from_vec(py, e).reshape([n, 2])?)
    }

    fn get_neighbors<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<i32>>> {
        let e = self.inner.get_neighbors().to_vec();
        Ok(PyArray1::from_vec(py, e).reshape([self.inner.ntri(), 3])?)
    }

    fn set_mask(&mut self, mask: PyReadonlyArray1<'_, bool>) {
        self.inner.set_mask(mask.as_array().to_vec());
    }

    fn get_boundaries(&mut self) -> Vec<Vec<(i32, i32)>> {
        self.inner.ensure_boundaries();
        self.inner.boundaries().clone()
    }
}

/// Make the triangles anticlockwise in place (swapping neighbors too).
#[pyfunction]
fn tri_correct_triangles(
    x: PyReadonlyArray1<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    mut triangles: PyReadwriteArray2<'_, i32>,
    neighbors: Option<PyReadwriteArray2<'_, i32>>,
) -> PyResult<()> {
    let x = x.as_slice()?;
    let y = y.as_slice()?;
    let t = triangles.as_slice_mut()?;
    match neighbors {
        Some(mut n) => Triangulation::correct_triangles(x, y, t, n.as_slice_mut()?),
        None => Triangulation::correct_triangles(x, y, t, &mut []),
    }
    Ok(())
}

#[pyclass(name = "TriContourGenerator", module = "matplotlib_rs._mplrs", unsendable)]
pub struct PyTriContourGenerator {
    inner: TriContourGenerator,
}

#[pymethods]
impl PyTriContourGenerator {
    #[new]
    fn new(triangulation: &PyTriangulation, z: PyReadonlyArray1<'_, f64>) -> PyResult<Self> {
        let z = z.as_array().to_vec();
        if z.len() != triangulation.inner.npoints() {
            return Err(PyValueError::new_err("z must be a 1D array with the same length as the x and y arrays"));
        }
        Ok(PyTriContourGenerator { inner: TriContourGenerator::new(triangulation.inner.clone(), z) })
    }

    fn create_contour<'py>(&mut self, py: Python<'py>, level: f64) -> PyResult<(Bound<'py, PyList>, Bound<'py, PyList>)> {
        let lines = self.inner.create_contour(level);
        let segs = tri::contour_lines_to_segs(&lines);
        let verts = PyList::empty(py);
        let codes = PyList::empty(py);
        for (s, c) in segs {
            let n = s.len();
            let flat: Vec<f64> = s.into_iter().flatten().collect();
            verts.append(PyArray1::from_vec(py, flat).reshape([n, 2])?)?;
            codes.append(PyArray1::from_vec(py, c))?;
        }
        Ok((verts, codes))
    }

    fn create_filled_contour<'py>(
        &mut self,
        py: Python<'py>,
        lower_level: f64,
        upper_level: f64,
    ) -> PyResult<(Bound<'py, PyList>, Bound<'py, PyList>)> {
        if lower_level >= upper_level {
            return Err(PyValueError::new_err("filled contour levels must be increasing"));
        }
        let lines = self.inner.create_filled_contour(lower_level, upper_level);
        let (s, c) = tri::contour_to_segs(&lines);
        let n = s.len();
        let flat: Vec<f64> = s.into_iter().flatten().collect();
        let verts = PyList::new(py, [PyArray1::from_vec(py, flat).reshape([n, 2])?])?;
        let codes = PyList::new(py, [PyArray1::from_vec(py, c)])?;
        Ok((verts, codes))
    }
}

#[pyclass(name = "TrapezoidMapTriFinder", module = "matplotlib_rs._mplrs", unsendable)]
pub struct PyTrapezoidMapTriFinder {
    triangulation: Py<PyTriangulation>,
    inner: Option<TrapezoidMapTriFinder>,
}

#[pymethods]
impl PyTrapezoidMapTriFinder {
    #[new]
    fn new(triangulation: Py<PyTriangulation>) -> Self {
        PyTrapezoidMapTriFinder { triangulation, inner: None }
    }

    fn initialize(&mut self, py: Python<'_>) -> PyResult<()> {
        let mut t = self.triangulation.borrow_mut(py);
        t.inner.ensure_neighbors();
        self.inner = Some(TrapezoidMapTriFinder::new(&t.inner).map_err(PyRuntimeError::new_err)?);
        Ok(())
    }

    fn find_many<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<i32>>> {
        let (x, y) = (x.as_array(), y.as_array());
        if x.len() != y.len() {
            return Err(PyValueError::new_err("x and y must be array-like with same shape"));
        }
        let f = self.inner.as_ref().ok_or_else(|| PyRuntimeError::new_err("TriFinder is not initialized"))?;
        let out: Vec<i32> = x.iter().zip(y.iter()).map(|(&a, &b)| f.find_one(XY::new(a, b))).collect();
        Ok(PyArray1::from_vec(py, out))
    }

    fn get_tree_stats(&self) -> PyResult<(i64, i64, i64, i64, i64, i64, f64)> {
        let f = self.inner.as_ref().ok_or_else(|| PyRuntimeError::new_err("TriFinder is not initialized"))?;
        Ok(f.tree_stats())
    }

    fn print_tree(&self) {}
}

/// Delaunay triangulation (Rust fallback of `_qhull.delaunay`).
#[pyfunction]
fn delaunay<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
) -> PyResult<(Bound<'py, PyArray2<i32>>, Bound<'py, PyArray2<i32>>)> {
    let (t, n) = tri::delaunay(x.as_slice()?, y.as_slice()?).map_err(PyRuntimeError::new_err)?;
    let ntri = t.len() / 3;
    Ok((PyArray1::from_vec(py, t).reshape([ntri, 3])?, PyArray1::from_vec(py, n).reshape([ntri, 3])?))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTriangulation>()?;
    m.add_class::<PyTriContourGenerator>()?;
    m.add_class::<PyTrapezoidMapTriFinder>()?;
    m.add_function(wrap_pyfunction!(tri_correct_triangles, m)?)?;
    m.add_function(wrap_pyfunction!(delaunay, m)?)?;
    Ok(())
}
