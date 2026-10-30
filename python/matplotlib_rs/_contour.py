"""
Contour generators for `.QuadContourSet`.

matplotlib's default algorithm, ``mpl2014``, is computed by the Rust port of
ContourPy's implementation (licenses/LICENSE_CONTOURPY), which gives the same
vertices, so that matplotlib_rs needs only NumPy.  The other algorithms
(``serial``, ``threaded``, ``mpl2005``) require ContourPy, which is also used
for ``mpl2014`` if ``MPLRS_CONTOUR=contourpy``.
"""

import os

import numpy as np

from . import _mplrs


def _contourpy(name="mpl2014"):
    """contourpy if the algorithm *name* is computed by it (else None)."""
    if name == "mpl2014" and os.environ.get("MPLRS_CONTOUR") != "contourpy":
        return None
    try:
        import contourpy
    except ImportError:
        raise ImportError(
            f"contour.algorithm {name!r} requires ContourPy (pip install contourpy); "
            f"the default 'mpl2014' algorithm is built in") from None
    return contourpy


def _calc_chunk_sizes(chunk_size, ny, nx):
    # contourpy.chunk.calc_chunk_sizes for chunk_size only.
    if nx < 2 or ny < 2:
        raise ValueError(f"(ny, nx) must be at least (2, 2), not ({ny}, {nx})")
    if chunk_size is None:
        y_chunk_size = x_chunk_size = 0
    elif isinstance(chunk_size, tuple):
        y_chunk_size, x_chunk_size = chunk_size
    else:
        y_chunk_size = x_chunk_size = chunk_size
    if x_chunk_size < 0 or y_chunk_size < 0:
        raise ValueError("chunk_size cannot be negative")
    return y_chunk_size, x_chunk_size


def _remove_z_mask(z):
    # Preserve mask if present.
    z_array = np.ma.asarray(z, dtype=np.float64)
    z_masked = np.ma.masked_invalid(z_array, copy=False)
    if np.ma.is_masked(z_masked):
        mask = np.ma.getmask(z_masked)
    else:
        mask = None
    return np.ma.getdata(z_masked), mask


def contour_generator(x, y, z, *, name, corner_mask, chunk_size):
    """
    ``contourpy.contour_generator(x, y, z, name=name, corner_mask=corner_mask,
    line_type=LineType.SeparateCode, fill_type=FillType.OuterCode,
    chunk_size=chunk_size)``.
    """
    contourpy = _contourpy(name)
    if contourpy is not None:
        return contourpy.contour_generator(
            x, y, z, name=name, corner_mask=corner_mask,
            line_type=contourpy.LineType.SeparateCode,
            fill_type=contourpy.FillType.OuterCode,
            chunk_size=chunk_size)

    # Argument processing of contourpy.contour_generator.
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z, mask = _remove_z_mask(z)
    if z.ndim != 2:
        raise TypeError(f"Input z must be 2D, not {z.ndim}D")
    if z.shape[0] < 2 or z.shape[1] < 2:
        raise TypeError(f"Input z must be at least a (2, 2) shaped array, but has shape {z.shape}")
    ny, nx = z.shape
    if x.ndim != y.ndim:
        raise TypeError(f"Number of dimensions of x ({x.ndim}) and y ({y.ndim}) do not match")
    if x.ndim == 0:
        x = np.arange(nx, dtype=np.float64)
        y = np.arange(ny, dtype=np.float64)
        x, y = np.meshgrid(x, y)
    elif x.ndim == 1:
        if len(x) != nx:
            raise TypeError(f"Length of x ({len(x)}) must match number of columns in z ({nx})")
        if len(y) != ny:
            raise TypeError(f"Length of y ({len(y)}) must match number of rows in z ({ny})")
        x, y = np.meshgrid(x, y)
    elif x.ndim == 2:
        if x.shape != z.shape:
            raise TypeError(f"Shapes of x {x.shape} and z {z.shape} do not match")
        if y.shape != z.shape:
            raise TypeError(f"Shapes of y {y.shape} and z {z.shape} do not match")
    else:
        raise TypeError(f"Inputs x and y must be None, 1D or 2D, not {x.ndim}D")
    if mask is not None and mask.shape != z.shape:
        raise ValueError("If mask is set it must be a 2D array with the same shape as z")
    y_chunk_size, x_chunk_size = _calc_chunk_sizes(chunk_size, ny, nx)
    if corner_mask is None:
        corner_mask = True
    return _mplrs.Mpl2014ContourGenerator(
        np.ascontiguousarray(x), np.ascontiguousarray(y), np.ascontiguousarray(z),
        None if mask is None else np.ascontiguousarray(mask, dtype=bool),
        corner_mask=bool(corner_mask), x_chunk_size=int(x_chunk_size),
        y_chunk_size=int(y_chunk_size))
