"""Colormap lookups of large float arrays in Rust, with numpy's results.

`get_rgba_and_mask` is `Colormap._get_rgba_and_mask` for float64 and float32
arrays.  `image_rgba` is the norm and colormap step of an image's 'rgba'
interpolation stage: it gives the colormap's lut rows of the pixels instead
of their RGBA values (the image is then resampled through the lut), and for
float64 data and a plain linear `Normalize` it normalizes in the same pass.

Anything else (other dtypes, subclasses that override the steps, alpha
arrays, small arrays) is left to matplotlib's code (None is returned).
"""
import operator

import numpy as np

from . import _mplrs

# Below this many values numpy is used as it is.
MIN_SIZE = 1 << 14


class Indexed:
    """An RGBA image as lut rows: ``lut[idx]``."""
    __slots__ = ("idx", "lut")

    def __init__(self, idx, lut):
        self.idx = idx
        self.lut = lut


def _plain_cmap(cmap):
    from .colors import Colormap
    cls = type(cmap)
    return (cls.__call__ is Colormap.__call__
            and cls._get_rgba_and_mask is Colormap._get_rgba_and_mask)


def _lookup(cmap, X):
    """The flat float data of *X*, its flat mask (when *X* has masked
    values), whether it has, and the colormap's N and under, over and bad
    indices; None for other inputs."""
    if type(X) not in (np.ndarray, np.ma.MaskedArray) or X.size < MIN_SIZE:
        return None
    data = np.asarray(X)
    if data.dtype not in (np.float64, np.float32) or not data.dtype.isnative:
        return None
    try:
        n, under, over, bad = (operator.index(v) for v in
                               (cmap.N, cmap._i_under, cmap._i_over, cmap._i_bad))
    except TypeError:
        return None
    # (float32: N and N - 1 are exact)
    if n < 1 or (data.dtype == np.float32 and n > 1 << 24):
        return None
    masked = np.ma.is_masked(X)
    mask = None
    if masked:
        mask = np.ma.getmaskarray(X)
        if mask.dtype != np.bool_ or mask.shape != X.shape:
            return None
        mask = np.ascontiguousarray(mask).reshape(-1)
    return np.ascontiguousarray(data).reshape(-1), mask, masked, (n, under, over, bad)


def _lut(cmap, dtype=np.float64):
    lut = cmap._lut
    if (type(lut) is not np.ndarray or lut.dtype != dtype or lut.ndim != 2
            or lut.shape[1] != 4 or len(lut) == 0):
        return None
    return np.ascontiguousarray(lut)


def get_rgba_and_mask(cmap, X, alpha, bytes):
    """`Colormap._get_rgba_and_mask(X, alpha, bytes)` (*cmap* is inited),
    or None to compute it with numpy."""
    if alpha is not None or not _plain_cmap(cmap):
        return None
    found = _lookup(cmap, X)
    lut = _lut(cmap)
    if found is None or lut is None:
        return None
    data, mask, masked, params = found
    if bytes:
        lut = (lut * 255).astype(np.uint8)
    rgba = _mplrs.colormap_rgba(data, mask, *params, lut).reshape(X.shape + (4,))
    # (numpy: np.isnan(X * N), the same as np.isnan(X) for N >= 1)
    mask_bad = X.mask if masked else np.isnan(data).reshape(X.shape)
    return rgba, mask_bad


def _linear_norm(norm, x):
    """``(vmin, vmax - vmin)`` when calling *norm* on the float64 masked
    array *x* computes ``(x - vmin) / (vmax - vmin)``, else None."""
    from .colors import Normalize
    cls = type(norm)
    if (cls.__call__ is not Normalize.__call__
            or cls.process_value is not Normalize.process_value
            or norm.clip or x.dtype != np.float64 or not x.dtype.isnative
            or norm.vmin is None or norm.vmax is None):
        return None
    (vmin,), _ = norm.process_value(norm.vmin)
    (vmax,), _ = norm.process_value(norm.vmax)
    if type(vmin) is not np.float64 or type(vmax) is not np.float64 or not vmin < vmax:
        return None
    return float(vmin), float(vmax - vmin)


def image_rgba(image, A):
    """``image.to_rgba(A)`` for the 2D data *A* of `_ImageBase._make_image`:
    the RGBA array, or an `Indexed` image with the same values."""
    from .colorizer import Colorizer, _ColorizerInterface
    colorizer = getattr(image, "_colorizer", None)
    if (A.size < MIN_SIZE or np.ndim(image.get_alpha()) > 0
            or type(image).to_rgba is not _ColorizerInterface.to_rgba
            or type(colorizer).to_rgba is not Colorizer.to_rgba
            or not _plain_cmap(colorizer.cmap)):
        return image.to_rgba(A)
    # Colorizer.to_rgba(A): cmap(norm(ma.asarray(A)))
    x = np.ma.asarray(A)
    norm = colorizer.norm
    linear = _linear_norm(norm, x)
    if linear is None:
        x = norm(x)
    cmap = colorizer.cmap
    cmap._ensure_inited()
    found = _lookup(cmap, x)
    lut = _lut(cmap)
    if found is None or lut is None:
        return image.to_rgba(A) if linear is not None else cmap(x)
    data, mask, _, params = found
    idx = _mplrs.colormap_indices(data, mask, *params, len(lut), linear)
    return Indexed(idx.reshape(x.shape), lut)
