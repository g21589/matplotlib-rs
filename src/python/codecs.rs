//! Python bindings of the image file codecs (`matplotlib_rs._imageio`).
//!
//! Pixel arrays are uint8, (h, w) or (h, w, channels); they are read in
//! place when C-contiguous, and the GIL is released while encoding.

use numpy::{PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::backend::image_codecs::{self as codecs, Samples};
use crate::backend::png_encode;
use crate::backend::renderer_agg::collection_threads;

/// (height, width, channels) and the bytes of an image array.
struct Pixels {
    h: usize,
    w: usize,
    c: usize,
    data: Vec<u8>,
}

/// The shape and pixels of `a` (copied: the arrays are small next to the
/// encoding work, and the copy lets every frame outlive the GIL).
fn pixels(a: &PyReadonlyArrayDyn<'_, u8>) -> PyResult<Pixels> {
    let (h, w, c) = match *a.shape() {
        [h, w] => (h, w, 1),
        [h, w, c] => (h, w, c),
        _ => return Err(PyValueError::new_err("expected an (h, w) or (h, w, channels) array")),
    };
    let v = a.as_array();
    let data = match v.as_slice() {
        Some(s) => s.to_vec(),
        None => v.iter().copied().collect(),
    };
    Ok(Pixels { h, w, c, data })
}

/// All frames as one size.
fn frames(frames: &[PyReadonlyArrayDyn<'_, u8>]) -> PyResult<Vec<Pixels>> {
    let out = frames.iter().map(pixels).collect::<PyResult<Vec<_>>>()?;
    let first = out.first().ok_or_else(|| PyValueError::new_err("no frames"))?;
    if out.iter().any(|f| (f.h, f.w, f.c) != (first.h, first.w, first.c)) {
        return Err(PyValueError::new_err("all frames must have the same shape"));
    }
    Ok(out)
}

fn bytes<'py>(py: Python<'py>, r: Result<Vec<u8>, String>) -> PyResult<Bound<'py, PyBytes>> {
    let out = r.map_err(PyValueError::new_err)?;
    Ok(PyBytes::new(py, &out))
}

/// The JPEG file of a gray, RGB or RGBA array (RGBA composited over the
/// RGB `background`, as by Pillow's ``paste``), with Pillow's option names.
#[pyfunction]
#[pyo3(signature = (pixels, quality=75, subsampling=2, optimize=false, progressive=false, dpi=None, icc_profile=None, exif=None, background=(255, 255, 255)))]
#[allow(clippy::too_many_arguments)]
fn encode_jpeg<'py>(
    py: Python<'py>,
    pixels: PyReadonlyArrayDyn<'py, u8>,
    quality: u8,
    subsampling: u8,
    optimize: bool,
    progressive: bool,
    dpi: Option<(u16, u16)>,
    icc_profile: Option<Vec<u8>>,
    exif: Option<Vec<u8>>,
    background: (u8, u8, u8),
) -> PyResult<Bound<'py, PyBytes>> {
    let p = self::pixels(&pixels)?;
    let o = codecs::JpegOptions {
        quality,
        subsampling,
        optimize,
        progressive,
        dpi,
        icc_profile: icc_profile.unwrap_or_default(),
        exif: exif.unwrap_or_default(),
    };
    let bg = [background.0, background.1, background.2];
    let threads = collection_threads();
    bytes(py, py.detach(|| codecs::encode_jpeg(&p.data, p.w, p.h, p.c, bg, &o, threads)))
}

/// The TIFF file of a gray, RGB or RGBA array; `compression` 0 none, 1 LZW,
/// 2 Deflate, 3 PackBits.
#[pyfunction]
#[pyo3(signature = (pixels, compression=0, dpi=None))]
fn encode_tiff<'py>(
    py: Python<'py>,
    pixels: PyReadonlyArrayDyn<'py, u8>,
    compression: u8,
    dpi: Option<(f64, f64)>,
) -> PyResult<Bound<'py, PyBytes>> {
    let p = self::pixels(&pixels)?;
    bytes(py, py.detach(|| codecs::encode_tiff(&p.data, p.w, p.h, p.c, compression, dpi)))
}

/// The GIF file of RGBA frames, each shown `duration` ms, looping
/// `loop_count` times (0: forever; None: no loop extension).
#[pyfunction]
#[pyo3(signature = (frames, duration=0, loop_count=None))]
fn encode_gif<'py>(
    py: Python<'py>,
    frames: Vec<PyReadonlyArrayDyn<'py, u8>>,
    duration: u32,
    loop_count: Option<u16>,
) -> PyResult<Bound<'py, PyBytes>> {
    let f = self::frames(&frames)?;
    if f[0].c != 4 {
        return Err(PyValueError::new_err("GIF frames must be RGBA"));
    }
    let (w, h) = (f[0].w, f[0].h);
    bytes(py, py.detach(|| {
        let refs: Vec<&[u8]> = f.iter().map(|p| &p.data[..]).collect();
        codecs::encode_gif(&refs, w, h, duration, loop_count)
    }))
}

/// The WebP file of RGB or RGBA frames: a still image for one frame without
/// `duration`, else an animation.
#[pyfunction]
#[pyo3(signature = (frames, lossless=false, quality=80.0, alpha_quality=100.0, method=4, exact=false, duration=None, loop_count=0))]
#[allow(clippy::too_many_arguments)]
fn encode_webp<'py>(
    py: Python<'py>,
    frames: Vec<PyReadonlyArrayDyn<'py, u8>>,
    lossless: bool,
    quality: f32,
    alpha_quality: f32,
    method: i32,
    exact: bool,
    duration: Option<u32>,
    loop_count: u32,
) -> PyResult<Bound<'py, PyBytes>> {
    let f = self::frames(&frames)?;
    let o = codecs::WebpOptions { lossless, quality, alpha_quality, method, exact };
    let (w, h, c) = (f[0].w, f[0].h, f[0].c);
    bytes(py, py.detach(|| match duration {
        None if f.len() == 1 => codecs::encode_webp(&f[0].data, w, h, c, &o),
        _ => {
            let refs: Vec<&[u8]> = f.iter().map(|p| &p.data[..]).collect();
            codecs::encode_webp_animation(&refs, w, h, c, duration.unwrap_or(0), loop_count, &o)
        }
    }))
}

/// The AVIF file of an RGBA array.
#[pyfunction]
#[pyo3(signature = (pixels, quality=75.0, speed=6))]
fn encode_avif<'py>(py: Python<'py>, pixels: PyReadonlyArrayDyn<'py, u8>, quality: f32, speed: u8) -> PyResult<Bound<'py, PyBytes>> {
    let p = self::pixels(&pixels)?;
    if p.c != 4 {
        return Err(PyValueError::new_err("AVIF images must be RGBA"));
    }
    let threads = collection_threads();
    bytes(py, py.detach(|| codecs::encode_avif(&p.data, p.w, p.h, quality, speed, threads)))
}

/// The animated PNG of frames (gray, gray + alpha, RGB or RGBA), each shown
/// `duration` ms, played `loop_count` times (0: forever).
#[pyfunction]
#[pyo3(signature = (frames, duration, loop_count=0, level=png_encode::DEFAULT_LEVEL))]
fn encode_apng<'py>(
    py: Python<'py>,
    frames: Vec<PyReadonlyArrayDyn<'py, u8>>,
    duration: u16,
    loop_count: u32,
    level: u8,
) -> PyResult<Bound<'py, PyBytes>> {
    let f = self::frames(&frames)?;
    let (w, h, c) = (f[0].w, f[0].h, f[0].c);
    let opts = super::api::png_options(None, &[], level, 0);
    bytes(py, py.detach(|| {
        let refs: Vec<&[u8]> = f.iter().map(|p| &p.data[..]).collect();
        png_encode::encode_apng(&refs, w, h, c, duration, loop_count, &opts)
    }))
}

/// The zlib stream of PNG-filtered rows of a (h, row_bytes) array (the data
/// of a PDF image with a PNG predictor), `bpp` bytes per pixel.
#[pyfunction]
#[pyo3(signature = (rows, bpp, level=png_encode::DEFAULT_LEVEL))]
fn zlib_png_rows<'py>(py: Python<'py>, rows: PyReadonlyArrayDyn<'py, u8>, bpp: usize, level: u8) -> PyResult<Bound<'py, PyBytes>> {
    let p = self::pixels(&rows)?;
    if p.c != 1 {
        return Err(PyValueError::new_err("expected an (h, row_bytes) array"));
    }
    let threads = collection_threads();
    bytes(py, py.detach(|| png_encode::zlib_rows(&p.data, p.w, p.h, bpp, level, threads)))
}

/// The palette (bytes of RGB triples, in increasing order) and the (h, w)
/// uint8 indices of an (h, w, 3) RGB array of at most 256 colors, or None.
#[pyfunction]
fn indexed_colors<'py>(py: Python<'py>, pixels: PyReadonlyArrayDyn<'py, u8>) -> PyResult<Option<(Bound<'py, PyBytes>, Bound<'py, PyAny>)>> {
    let p = self::pixels(&pixels)?;
    if p.c != 3 {
        return Err(PyValueError::new_err("expected an (h, w, 3) array"));
    }
    let Some((palette, indices)) = py.detach(|| codecs::indexed_colors(&p.data, 256)) else {
        return Ok(None);
    };
    let indices = numpy::PyArray1::from_vec(py, indices).reshape([p.h, p.w])?.into_any();
    Ok(Some((PyBytes::new(py, &palette), indices)))
}

/// The format name and pixels of a JPEG, TIFF, GIF or WebP file: an (h, w)
/// or (h, w, channels) uint8 array, or uint16 for 16-bit gray TIFF files.
#[pyfunction]
fn decode_image<'py>(py: Python<'py>, data: &[u8]) -> PyResult<(&'static str, Bound<'py, PyAny>)> {
    let (format, image) = py.detach(|| codecs::decode(data)).map_err(PyValueError::new_err)?;
    let shape: Vec<usize> = if image.channels == 1 {
        vec![image.height, image.width]
    } else {
        vec![image.height, image.width, image.channels]
    };
    let array = match image.samples {
        Samples::U8(v) => numpy::PyArray1::from_vec(py, v).reshape(shape)?.into_any(),
        Samples::U16(v) => numpy::PyArray1::from_vec(py, v).reshape(shape)?.into_any(),
    };
    Ok((format.name(), array))
}

/// The format name of a file from its first bytes, or None.
#[pyfunction]
fn sniff_image(data: &[u8]) -> Option<&'static str> {
    codecs::sniff(data).map(|f| f.name())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(encode_jpeg, m)?)?;
    m.add_function(wrap_pyfunction!(encode_tiff, m)?)?;
    m.add_function(wrap_pyfunction!(encode_gif, m)?)?;
    m.add_function(wrap_pyfunction!(encode_webp, m)?)?;
    m.add_function(wrap_pyfunction!(encode_avif, m)?)?;
    m.add_function(wrap_pyfunction!(encode_apng, m)?)?;
    m.add_function(wrap_pyfunction!(zlib_png_rows, m)?)?;
    m.add_function(wrap_pyfunction!(decode_image, m)?)?;
    m.add_function(wrap_pyfunction!(indexed_colors, m)?)?;
    m.add_function(wrap_pyfunction!(sniff_image, m)?)?;
    Ok(())
}
