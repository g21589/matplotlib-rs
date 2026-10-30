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

extern "C" {
    fn mplrs_qhull_delaunay(
        npoints: std::os::raw::c_int,
        x: *const f64,
        y: *const f64,
        hide_errors: std::os::raw::c_int,
        ntri: *mut std::os::raw::c_int,
        triangles: *mut *mut std::os::raw::c_int,
        neighbors: *mut *mut std::os::raw::c_int,
        leaked: *mut std::os::raw::c_int,
    ) -> std::os::raw::c_int;
    fn mplrs_qhull_free(p: *mut std::os::raw::c_int);
    fn mplrs_qhull_version() -> *const std::os::raw::c_char;
}

const QHULL_ERROR_MSG: [&str; 6] = [
    "",
    "input inconsistency",
    "singular input data",
    "precision error",
    "insufficient memory",
    "internal error",
];

/// matplotlib's `_qhull.delaunay(x, y, verbose)`, computed by the same qhull
/// (extern/qhull): ``(triangles, neighbors)``, int arrays of shape (ntri, 3).
/// *x* and *y* have been checked by `matplotlib_rs._qhull`.
#[pyfunction]
fn qhull_delaunay<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    verbose: i32,
) -> PyResult<(Bound<'py, PyArray2<i32>>, Bound<'py, PyArray2<i32>>)> {
    let (x, y) = (x.as_slice()?, y.as_slice()?);
    if x.len() != y.len() || x.len() > i32::MAX as usize {
        return Err(PyValueError::new_err("x and y must be 1D arrays of the same length"));
    }
    let (mut ntri, mut leaked) = (0, 0);
    let (mut tri, mut nbr) = (std::ptr::null_mut(), std::ptr::null_mut());
    let hide = (verbose == 0) as std::os::raw::c_int;
    let code = unsafe {
        mplrs_qhull_delaunay(x.len() as i32, x.as_ptr(), y.as_ptr(), hide, &mut ntri, &mut tri, &mut nbr, &mut leaked)
    };
    if leaked != 0 {
        PyErr::warn(
            py,
            &py.get_type::<pyo3::exceptions::PyRuntimeWarning>(),
            c"Qhull could not free all allocated memory",
            1,
        )?;
    }
    if code != 0 {
        if code < 0 {
            return Err(PyRuntimeError::new_err("Could not allocate memory for qhull"));
        }
        let mut msg = format!(
            "Error in qhull Delaunay triangulation calculation: {} (exitcode={})",
            QHULL_ERROR_MSG.get(code as usize).copied().unwrap_or("unknown error"),
            code
        );
        if hide != 0 {
            msg += "; use python verbose option (-v) to see original qhull error.";
        }
        return Err(PyRuntimeError::new_err(msg));
    }
    let n = ntri as usize;
    let take = |p: *mut std::os::raw::c_int| {
        let v = unsafe { std::slice::from_raw_parts(p, 3 * n).to_vec() };
        unsafe { mplrs_qhull_free(p) };
        v
    };
    let (t, nb) = (take(tri), take(nbr));
    Ok((PyArray1::from_vec(py, t).reshape([n, 3])?, PyArray1::from_vec(py, nb).reshape([n, 3])?))
}

/// qhull's version string.
#[pyfunction]
fn qhull_version() -> String {
    unsafe { std::ffi::CStr::from_ptr(mplrs_qhull_version()) }.to_string_lossy().into_owned()
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(qhull_delaunay, m)?)?;
    m.add_function(wrap_pyfunction!(qhull_version, m)?)?;
    m.add_class::<PyTriangulation>()?;
    m.add_class::<PyTriContourGenerator>()?;
    m.add_class::<PyTrapezoidMapTriFinder>()?;
    m.add_function(wrap_pyfunction!(tri_correct_triangles, m)?)?;
    m.add_function(wrap_pyfunction!(delaunay, m)?)?;
    Ok(())
}
