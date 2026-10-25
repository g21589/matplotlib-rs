"""
Replacement for matplotlib's ``_image`` extension module: image resampling
by the Rust port of matplotlib's Agg resampler.
"""

import numpy as np

from . import _mplrs

NEAREST = 0
BILINEAR = 1
BICUBIC = 2
SPLINE16 = 3
SPLINE36 = 4
HANNING = 5
HAMMING = 6
HERMITE = 7
KAISER = 8
QUADRIC = 9
CATROM = 10
GAUSSIAN = 11
BESSEL = 12
MITCHELL = 13
SINC = 14
LANCZOS = 15
BLACKMAN = 16

_KINDS = {
    np.dtype(np.uint8): "u1", np.dtype(np.int8): "u1",
    np.dtype(np.uint16): "u2", np.dtype(np.int16): "u2",
    np.dtype(np.float32): "f4", np.dtype(np.float64): "f8",
}


def _transform_mesh(transform, out_h, out_w):
    """For each output pixel center, the input coordinates it maps from."""
    inverse = transform.inverted()
    y, x = np.mgrid[:out_h, :out_w]
    mesh = np.empty((out_h * out_w, 2))
    mesh[:, 0] = x.ravel() + 0.5
    mesh[:, 1] = y.ravel() + 0.5
    out = np.ascontiguousarray(inverse.transform(mesh), dtype=np.float64)
    if out.ndim != 2:
        raise RuntimeError(
            f"Inverse transformed mesh array should be 2D not {out.ndim}D")
    if out.shape != mesh.shape:
        raise RuntimeError(
            f"Inverse transformed mesh array should have shape {mesh.shape} "
            f"not {out.shape}")
    return out


def resample(input_array, output_array, transform, interpolation=NEAREST,
             resample=False, alpha=1, norm=False, radius=1):
    """
    Resample input_array, blending it in-place into output_array, using an
    affine transform.
    """
    input_array = np.asarray(input_array)
    ndim = input_array.ndim
    if ndim not in (2, 3):
        raise ValueError("Input array must be a 2D or 3D array")
    if ndim == 3 and input_array.shape[2] != 4:
        raise ValueError(
            "3D input array must be RGBA with shape (M, N, 4), has trailing "
            f"dimension of {input_array.shape[2]}")
    input_array = np.ascontiguousarray(input_array)
    out_ndim = output_array.ndim
    if out_ndim != ndim:
        raise ValueError(
            f"Input ({ndim}D) and output ({out_ndim}D) arrays have different "
            "dimensionalities")
    if out_ndim == 3 and output_array.shape[2] != 4:
        raise ValueError(
            "3D output array must be RGBA with shape (M, N, 4), has trailing "
            f"dimension of {output_array.shape[2]}")
    if output_array.dtype != input_array.dtype:
        raise ValueError("Input and output arrays have mismatched types")
    if not output_array.flags.c_contiguous:
        raise ValueError("Output array must be C-contiguous")
    if not output_array.flags.writeable:
        raise ValueError("Output array must be writeable")
    kind = _KINDS.get(input_array.dtype)
    out_h, out_w = output_array.shape[:2]
    if transform is None:
        affine, mesh = np.eye(3), None
    elif transform.is_affine:
        affine, mesh = np.ascontiguousarray(transform, dtype=np.float64), None
    else:
        affine, mesh = None, _transform_mesh(transform, out_h, out_w)
    if kind is None:
        raise ValueError(
            "arrays must be of dtype byte, short, float32 or float64")
    in_h, in_w = input_array.shape[:2]
    if input_array.size == 0 or output_array.size == 0:
        return
    _mplrs.image_resample(
        input_array.reshape(-1).view(np.uint8),
        output_array.reshape(-1).view(np.uint8),
        kind, 1 if ndim == 2 else 4, in_w, in_h, out_w, out_h,
        affine, mesh, int(interpolation), bool(resample),
        float(np.float32(alpha)), bool(norm), float(np.float32(radius)))


def calculate_rms_and_diff(expected_image, actual_image):
    """RMS of the difference of two images and a difference image."""
    from .testing.exceptions import ImageComparisonFailure
    for image, name in ((expected_image, "Expected"), (actual_image, "Actual")):
        if image.ndim != 3:
            raise ImageComparisonFailure(
                f"{name} image must be 3-dimensional, but is "
                f"{image.ndim}-dimensional")
    height, width, depth = expected_image.shape
    if depth not in (3, 4):
        raise ImageComparisonFailure(
            f"Image must be RGB or RGBA but has depth {depth}")
    if expected_image.shape != actual_image.shape:
        raise ImageComparisonFailure(
            f"Image sizes do not match expected size: {expected_image.shape} "
            f"actual size {actual_image.shape}")
    diff = expected_image.astype(np.float64) - actual_image.astype(np.float64)
    rms = np.sqrt((diff ** 2).sum() / (width * height * depth))
    diff_image = np.clip(np.abs(diff[..., :3]) * 10, 0, 255).astype(np.uint8)
    return rms, diff_image
