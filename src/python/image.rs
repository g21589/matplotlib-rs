//! Bindings for the image resampler (matplotlib's `_image.resample`).

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::agg::trans_affine::TransAffine;
use crate::backend::image_resample::{resample, ColorType, Gray, ResampleParams, Rgba8Plain, RgbaPre};

/// View raw bytes as a slice of `T`, copying when they are not aligned.
fn cast<T: Copy + Default>(bytes: &[u8]) -> std::borrow::Cow<'_, [T]> {
    let size = std::mem::size_of::<T>();
    let n = bytes.len() / size;
    if bytes.as_ptr().align_offset(std::mem::align_of::<T>()) == 0 {
        // SAFETY: aligned, in bounds, and every bit pattern is a valid T
        // (T is one of u8, u16, f32, f64).
        std::borrow::Cow::Borrowed(unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const T, n) })
    } else {
        let mut v = vec![T::default(); n];
        // SAFETY: v has room for n * size bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), v.as_mut_ptr() as *mut u8, n * size);
        }
        std::borrow::Cow::Owned(v)
    }
}

#[allow(clippy::too_many_arguments)]
fn run<C: ColorType>(
    input: &[u8],
    output: &mut [u8],
    dims: (i32, i32, i32, i32),
    params: &ResampleParams,
) {
    let (in_w, in_h, out_w, out_h) = dims;
    let src = cast::<C::V>(input);
    let size = std::mem::size_of::<C::V>();
    if output.as_ptr().align_offset(std::mem::align_of::<C::V>()) == 0 {
        // SAFETY: aligned and in bounds; every bit pattern is a valid value.
        let out = unsafe {
            std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut C::V, output.len() / size)
        };
        resample::<C>(&src, in_w, in_h, out, out_w, out_h, params);
    } else {
        let mut out = cast::<C::V>(output).into_owned();
        resample::<C>(&src, in_w, in_h, &mut out, out_w, out_h, params);
        // SAFETY: out holds output.len() / size values.
        unsafe {
            std::ptr::copy_nonoverlapping(out.as_ptr() as *const u8, output.as_mut_ptr(), out.len() * size);
        }
    }
}

/// Resample the image in *input* (raw bytes of an `in_h` x `in_w` x
/// `channels` C-contiguous array of *kind* 'u1', 'u2', 'f4' or 'f8') into
/// *output* (raw bytes of the `out_h` x `out_w` output array), blending into
/// its contents.  *affine* is the 3x3 matrix of an affine transform (or
/// None, then *mesh* holds the input coordinates of each output pixel
/// center).
#[pyfunction]
#[pyo3(signature = (input, output, kind, channels, in_w, in_h, out_w, out_h, affine, mesh,
                    interpolation, resample, alpha, norm, radius))]
#[allow(clippy::too_many_arguments)]
fn image_resample(
    input: PyReadonlyArray1<'_, u8>,
    mut output: PyReadwriteArray1<'_, u8>,
    kind: &str,
    channels: usize,
    in_w: i32,
    in_h: i32,
    out_w: i32,
    out_h: i32,
    affine: Option<PyReadonlyArray2<'_, f64>>,
    mesh: Option<PyReadonlyArray2<'_, f64>>,
    interpolation: u32,
    resample: bool,
    alpha: f64,
    norm: bool,
    radius: f64,
) -> PyResult<()> {
    let input = input.as_slice()?;
    let output = output.as_slice_mut()?;
    let affine = match &affine {
        Some(a) => {
            let a = a.as_array();
            if a.shape() != [3, 3] {
                return Err(PyValueError::new_err("Invalid affine transformation matrix"));
            }
            Some(TransAffine { sx: a[[0, 0]], shx: a[[0, 1]], tx: a[[0, 2]], shy: a[[1, 0]], sy: a[[1, 1]], ty: a[[1, 2]] })
        }
        None => None,
    };
    let mesh = match &mesh {
        Some(m) => Some(m.as_slice()?),
        None => None,
    };
    if affine.is_none() {
        match mesh {
            Some(m) if m.len() >= (out_w as usize) * (out_h as usize) * 2 => {}
            _ => return Err(PyValueError::new_err("an undersized transform mesh")),
        }
    }
    let params = ResampleParams { interpolation, affine, mesh, resample, norm, radius, alpha };
    let dims = (in_w, in_h, out_w, out_h);
    match (kind, channels) {
        ("u1", 1) => run::<Gray<u8>>(input, output, dims, &params),
        ("u2", 1) => run::<Gray<u16>>(input, output, dims, &params),
        ("f4", 1) => run::<Gray<f32>>(input, output, dims, &params),
        ("f8", 1) => run::<Gray<f64>>(input, output, dims, &params),
        ("u1", 4) => run::<Rgba8Plain>(input, output, dims, &params),
        ("u2", 4) => run::<RgbaPre<u16>>(input, output, dims, &params),
        ("f4", 4) => run::<RgbaPre<f32>>(input, output, dims, &params),
        ("f8", 4) => run::<RgbaPre<f64>>(input, output, dims, &params),
        _ => return Err(PyValueError::new_err("arrays must be of dtype byte, short, float32 or float64")),
    }
    Ok(())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(image_resample, m)?)?;
    Ok(())
}
