//! `Colormap._get_rgba_and_mask` (and, for images, the linear `Normalize`
//! before it) for large float arrays: numpy's steps on each value, on
//! several threads.
//!
//! For a value `x` numpy computes `xa = x * N` (in the array's float type),
//! sets `xa == N` to `N - 1`, marks `xa < 0` as under, `xa >= N` as over and
//! masked (or, without a mask, NaN) values as bad, truncates `xa` to an
//! integer, sets the under, over and bad indices (in that order) and looks
//! the indices up in the lut with `take(..., mode='clip')`.  NaN that is not
//! masked truncates to `INT64_MIN` (x86) or 0 (arm), which the clipping
//! turns into 0 either way, as `as i64` does here.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::backend::renderer_agg::collection_threads;

/// Below this many values one thread is used.
const PARALLEL_MIN: usize = 1 << 16;

trait Value: Copy + PartialOrd + Send + Sync {
    const ZERO: Self;
    fn from_usize(n: usize) -> Self;
    fn mul(self, o: Self) -> Self;
    fn is_nan(self) -> bool;
    fn trunc(self) -> i64;
}

impl Value for f64 {
    const ZERO: f64 = 0.0;
    fn from_usize(n: usize) -> f64 {
        n as f64
    }
    #[inline(always)]
    fn mul(self, o: f64) -> f64 {
        self * o
    }
    #[inline(always)]
    fn is_nan(self) -> bool {
        f64::is_nan(self)
    }
    #[inline(always)]
    fn trunc(self) -> i64 {
        self as i64
    }
}

impl Value for f32 {
    const ZERO: f32 = 0.0;
    fn from_usize(n: usize) -> f32 {
        n as f32
    }
    #[inline(always)]
    fn mul(self, o: f32) -> f32 {
        self * o
    }
    #[inline(always)]
    fn is_nan(self) -> bool {
        f32::is_nan(self)
    }
    #[inline(always)]
    fn trunc(self) -> i64 {
        self as i64
    }
}

/// The colormap's N, its under/over/bad indices and the lut length.
#[derive(Clone, Copy)]
struct Lookup<V> {
    n: V,
    n_minus_1: V,
    under: i64,
    over: i64,
    bad: i64,
    last: i64,
}

impl<V: Value> Lookup<V> {
    /// The lut row of the (already normalized) value *x*; *masked* is its
    /// mask value, or None for an unmasked array (then NaN is bad).
    #[inline(always)]
    fn index(&self, x: V, masked: Option<bool>) -> u32 {
        let mut v = x.mul(self.n);
        if v == self.n {
            v = self.n_minus_1;
        }
        let bad = masked.unwrap_or_else(|| v.is_nan());
        let i = if bad {
            self.bad
        } else if v >= self.n {
            self.over
        } else if v < V::ZERO {
            self.under
        } else {
            v.trunc()
        };
        i.clamp(0, self.last) as u32
    }
}

/// The linear `Normalize`: `(x - vmin) / (vmax - vmin)` in float64.
#[derive(Clone, Copy)]
struct Linear {
    vmin: f64,
    scale: f64,
}

/// Call `f(range, out_chunk)` for consecutive pieces of `0..len`, with
/// `out` split accordingly (`per` output values per input value), on
/// several threads for large inputs.
fn split<T: Send>(len: usize, per: usize, out: &mut [T], f: impl Fn(std::ops::Range<usize>, &mut [T]) + Sync) {
    let threads = collection_threads();
    if threads < 2 || len < PARALLEL_MIN {
        f(0..len, out);
        return;
    }
    let size = len.div_ceil(threads);
    std::thread::scope(|s| {
        for (k, chunk) in out.chunks_mut(size * per).enumerate() {
            let f = &f;
            let start = k * size;
            s.spawn(move || f(start..(start + size).min(len), chunk));
        }
    });
}

fn indices<V: Value>(x: &[V], mask: Option<&[bool]>, l: Lookup<V>, out: &mut [u32]) {
    split(x.len(), 1, out, |r, out| match mask {
        Some(m) => {
            for ((o, &v), &m) in out.iter_mut().zip(&x[r.clone()]).zip(&m[r]) {
                *o = l.index(v, Some(m));
            }
        }
        None => {
            for (o, &v) in out.iter_mut().zip(&x[r]) {
                *o = l.index(v, None);
            }
        }
    });
}

fn normalized_indices(x: &[f64], mask: Option<&[bool]>, norm: Linear, l: Lookup<f64>, out: &mut [u32]) {
    let f = |v: f64| (v - norm.vmin) / norm.scale;
    split(x.len(), 1, out, |r, out| match mask {
        Some(m) => {
            for ((o, &v), &m) in out.iter_mut().zip(&x[r.clone()]).zip(&m[r]) {
                *o = l.index(f(v), Some(m));
            }
        }
        None => {
            for (o, &v) in out.iter_mut().zip(&x[r]) {
                *o = l.index(f(v), None);
            }
        }
    });
}

fn gather<V: Value, T: Copy + Send + Sync>(x: &[V], mask: Option<&[bool]>, l: Lookup<V>, lut: &[[T; 4]], out: &mut [T]) {
    split(x.len(), 4, out, |r, out| {
        let rows = out.chunks_exact_mut(4);
        match mask {
            Some(m) => {
                for ((o, &v), &m) in rows.zip(&x[r.clone()]).zip(&m[r]) {
                    o.copy_from_slice(&lut[l.index(v, Some(m)) as usize]);
                }
            }
            None => {
                for (o, &v) in rows.zip(&x[r]) {
                    o.copy_from_slice(&lut[l.index(v, None) as usize]);
                }
            }
        }
    });
}

fn lookup<V: Value>(n: usize, under: i64, over: i64, bad: i64, lut_len: usize) -> PyResult<Lookup<V>> {
    if n == 0 || lut_len == 0 {
        return Err(PyValueError::new_err("colormap: empty lut"));
    }
    Ok(Lookup {
        n: V::from_usize(n),
        n_minus_1: V::from_usize(n - 1),
        under,
        over,
        bad,
        last: lut_len as i64 - 1,
    })
}

fn check_mask(mask: &Option<PyReadonlyArray1<'_, bool>>, len: usize) -> PyResult<()> {
    match mask {
        Some(m) if m.len() != len => Err(PyValueError::new_err("colormap: mask and data sizes differ")),
        _ => Ok(()),
    }
}

/// `colormap_indices(x, mask, n, under, over, bad, lut_len, norm=None)`:
/// the lut rows `Colormap._get_rgba_and_mask` takes for the 1D float64 or
/// float32 array *x* (with the bool *mask* of a masked array that has
/// masked values, else None).  With *norm* ``(vmin, vmax - vmin)`` *x*
/// (float64) is first normalized like `Normalize` does.
#[pyfunction]
#[pyo3(signature = (x, mask, n, under, over, bad, lut_len, norm=None))]
#[allow(clippy::too_many_arguments)]
fn colormap_indices<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArray1<'py, bool>>,
    n: usize,
    under: i64,
    over: i64,
    bad: i64,
    lut_len: usize,
    norm: Option<(f64, f64)>,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    let mask_slice = match &mask {
        Some(m) => Some(m.as_slice()?),
        None => None,
    };
    if let Ok(a) = x.extract::<PyReadonlyArray1<'py, f64>>() {
        let a = a.as_slice()?;
        check_mask(&mask, a.len())?;
        let l = lookup::<f64>(n, under, over, bad, lut_len)?;
        let mut out = vec![0u32; a.len()];
        py.detach(|| match norm {
            Some((vmin, scale)) => normalized_indices(a, mask_slice, Linear { vmin, scale }, l, &mut out),
            None => indices(a, mask_slice, l, &mut out),
        });
        return Ok(PyArray1::from_vec(py, out));
    }
    if norm.is_some() {
        return Err(PyValueError::new_err("colormap_indices: norm needs float64 data"));
    }
    let a = x.extract::<PyReadonlyArray1<'py, f32>>()?;
    let a = a.as_slice()?;
    check_mask(&mask, a.len())?;
    let l = lookup::<f32>(n, under, over, bad, lut_len)?;
    let mut out = vec![0u32; a.len()];
    py.detach(|| indices(a, mask_slice, l, &mut out));
    Ok(PyArray1::from_vec(py, out))
}

fn rows<T: Copy>(lut: &[T]) -> Vec<[T; 4]> {
    lut.chunks_exact(4).map(|c| [c[0], c[1], c[2], c[3]]).collect()
}

fn rgba<'py, V: Value, T: Copy + Send + Sync + numpy::Element>(
    py: Python<'py>,
    x: &[V],
    mask: Option<&[bool]>,
    l: Lookup<V>,
    lut: &[[T; 4]],
    zero: T,
) -> Bound<'py, PyArray1<T>> {
    let mut out = vec![zero; x.len() * 4];
    py.detach(|| gather(x, mask, l, lut, &mut out));
    PyArray1::from_vec(py, out)
}

/// `colormap_rgba(x, mask, n, under, over, bad, lut)`: the RGBA values
/// (flattened) `Colormap._get_rgba_and_mask` gives the 1D float64 or float32
/// array *x* (with the bool *mask* of a masked array that has masked values,
/// else None) from the (K, 4) float64 or uint8 *lut*.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn colormap_rgba<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArray1<'py, bool>>,
    n: usize,
    under: i64,
    over: i64,
    bad: i64,
    lut: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let mask_slice = match &mask {
        Some(m) => Some(m.as_slice()?),
        None => None,
    };
    enum Lut {
        F(Vec<[f64; 4]>),
        B(Vec<[u8; 4]>),
    }
    let (table, len) = if let Ok(t) = lut.extract::<PyReadonlyArray2<'py, f64>>() {
        if t.shape()[1] != 4 {
            return Err(PyValueError::new_err("colormap_rgba: lut must be (K, 4)"));
        }
        let r = rows(t.as_slice()?);
        let len = r.len();
        (Lut::F(r), len)
    } else {
        let t = lut.extract::<PyReadonlyArray2<'py, u8>>()?;
        if t.shape()[1] != 4 {
            return Err(PyValueError::new_err("colormap_rgba: lut must be (K, 4)"));
        }
        let r = rows(t.as_slice()?);
        let len = r.len();
        (Lut::B(r), len)
    };
    macro_rules! run {
        ($v:ty, $a:expr) => {{
            let a: &[$v] = $a;
            check_mask(&mask, a.len())?;
            let l = lookup::<$v>(n, under, over, bad, len)?;
            Ok(match &table {
                Lut::F(t) => rgba(py, a, mask_slice, l, t, 0.0).into_any(),
                Lut::B(t) => rgba(py, a, mask_slice, l, t, 0u8).into_any(),
            })
        }};
    }
    if let Ok(a) = x.extract::<PyReadonlyArray1<'py, f64>>() {
        return run!(f64, a.as_slice()?);
    }
    let a = x.extract::<PyReadonlyArray1<'py, f32>>()?;
    run!(f32, a.as_slice()?)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(colormap_indices, m)?)?;
    m.add_function(wrap_pyfunction!(colormap_rgba, m)?)?;
    Ok(())
}
