//! The counting loops of `numpy.histogram` for unweighted float64 data.
//!
//! The counts are integers, so the data can be split among threads.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// numpy's equal-bins algorithm (the same for NumPy 1.25 to 2.3): the
/// estimated index `((x - first) / (last - first)) * n` truncated, then
/// corrected by one against the edges.
fn count_uniform(a: &[f64], edges: &[f64], first: f64, last: f64, n: usize, counts: &mut [i64]) {
    let denom = last - first;
    let nf = n as f64;
    let last_bin = n as i64 - 1;
    for &x in a {
        // keep = (a >= first_edge) & (a <= last_edge) (drops NaN)
        if !(x >= first && x <= last) {
            continue;
        }
        let mut i = (((x - first) / denom) * nf) as i64;
        if i == n as i64 {
            i -= 1;
        }
        if x < edges[i as usize] {
            i -= 1;
        }
        if x >= edges[i as usize + 1] && i != last_bin {
            i += 1;
        }
        counts[i as usize] += 1;
    }
}

/// numpy's algorithm for arbitrary edges: per bin, the values below its
/// right edge minus those below its left edge (`searchsorted` of the sorted
/// data, 'right' for the last edge), i.e. edges[i] <= x < edges[i + 1] and
/// edges[-2] <= x <= edges[-1] for the last bin.
fn count_sorted_edges(a: &[f64], edges: &[f64], counts: &mut [i64]) {
    let n = edges.len() - 1;
    let (lo, hi) = (edges[0], edges[n]);
    let inner = &edges[..n];
    for &x in a {
        if !(x >= lo && x <= hi) {
            continue;
        }
        let k = inner.partition_point(|&e| e <= x);
        counts[k - 1] += 1;
    }
}

/// `histogram_counts(a, edges, uniform)`: the counts `numpy.histogram`
/// gives the 1D float64 array *a* for the float64 bin *edges* (which come
/// from numpy), using its equal-bins algorithm when *uniform* is
/// ``(first_edge, last_edge, n_bins)``.
#[pyfunction]
fn histogram_counts<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<'py, f64>,
    edges: PyReadonlyArray1<'py, f64>,
    uniform: Option<(f64, f64, usize)>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let a = a.as_slice()?;
    let edges = edges.as_slice()?.to_vec();
    let n = edges.len().saturating_sub(1);
    if n == 0 || edges.iter().any(|e| e.is_nan()) || uniform.is_some_and(|u| u.2 != n) {
        return Err(PyValueError::new_err("histogram_counts: invalid edges"));
    }
    let count = |chunk: &[f64]| {
        let mut c = vec![0i64; n];
        match uniform {
            Some((first, last, _)) => count_uniform(chunk, &edges, first, last, n, &mut c),
            None => count_sorted_edges(chunk, &edges, &mut c),
        }
        c
    };
    let counts = py.detach(|| {
        let threads = std::thread::available_parallelism().map_or(1, |t| t.get()).min(8);
        if threads < 2 || a.len() < 1 << 18 {
            return count(a);
        }
        let size = a.len().div_ceil(threads);
        std::thread::scope(|s| {
            let parts: Vec<_> = a.chunks(size).map(|chunk| s.spawn(move || count(chunk))).collect();
            let mut total = vec![0i64; n];
            for p in parts {
                for (t, c) in total.iter_mut().zip(p.join().expect("histogram thread")) {
                    *t += c;
                }
            }
            total
        })
    });
    Ok(PyArray1::from_vec(py, counts))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(histogram_counts, m)?)?;
    Ok(())
}
