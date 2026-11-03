"""`numpy.histogram` and `numpy.histogram2d` for `Axes.hist` and
`Axes.hist2d`, counting large unweighted float64 data in Rust.

The bin edges (and the choice of numpy's equal-bins algorithm) come from
numpy's own ``_get_bin_edges``; the Rust loops are numpy's algorithms, so the
counts are the same.
"""
import operator

import numpy as np

from .. import _mplrs

try:  # NumPy 2
    from numpy.lib._histograms_impl import _get_bin_edges, _get_outer_edges
except ImportError:
    try:  # NumPy 1.25, 1.26
        from numpy.lib.histograms import _get_bin_edges, _get_outer_edges
    except ImportError:  # (numpy.histogram is used as it is)
        _get_bin_edges = _get_outer_edges = None

# Below this many values numpy is used as it is.
MIN_SIZE = 1 << 16


def histogram(a, bins=10, range=None, density=None, weights=None):
    """`numpy.histogram`, with the same results."""
    if (weights is not None or density or type(a) is not np.ndarray
            or a.dtype != np.float64 or a.size < MIN_SIZE or _get_bin_edges is None):
        return np.histogram(a, bins, range=range, density=density, weights=weights)
    a = a.ravel()
    try:
        edges, uniform = _get_bin_edges(a, bins, range, None)
    except Exception:  # numpy raises the same
        return np.histogram(a, bins, range=range, density=density, weights=weights)
    if edges.dtype != np.float64 or edges.ndim != 1 or len(edges) < 2:
        return np.histogram(a, bins, range=range, density=density, weights=weights)
    if uniform is not None:
        first, last, n = uniform
        # (numpy computes with these in float64)
        if type(first) not in (float, np.float64) or type(last) not in (float, np.float64):
            return np.histogram(a, bins, range=range, density=density, weights=weights)
        uniform = (float(first), float(last), int(n))
    elif np.isnan(edges).any():
        return np.histogram(a, bins, range=range, density=density, weights=weights)
    counts = _mplrs.histogram_counts(np.ascontiguousarray(a), np.ascontiguousarray(edges), uniform)
    return counts.astype(np.intp, copy=False), edges


def _histdd_edges(sample, bins, range_):
    """The edges `numpy.histogramdd` computes along one axis, or None if it
    would raise (numpy then raises the same) or they are not float64."""
    if np.ndim(bins) == 0:
        if isinstance(bins, (bool, np.bool_)) or bins < 1:
            return None
        smin, smax = _get_outer_edges(sample, range_)
        edges = np.linspace(smin, smax, operator.index(bins) + 1)
    elif np.ndim(bins) == 1:
        edges = np.asarray(bins)
        if edges.dtype != np.float64 or np.any(edges[:-1] > edges[1:]):
            return None
    else:
        return None
    if edges.dtype != np.float64 or len(edges) < 2 or np.isnan(edges).any():
        return None
    return edges


def histogram2d(x, y, bins=10, range=None, density=None, weights=None):
    """`numpy.histogram2d`, with the same results."""
    def numpy():
        return np.histogram2d(x, y, bins=bins, range=range, density=density, weights=weights)

    if (weights is not None or density or _get_outer_edges is None
            or not all(type(v) is np.ndarray and v.dtype == np.float64 and v.ndim == 1
                       for v in (x, y))
            or len(x) != len(y) or len(x) < MIN_SIZE):
        return numpy()
    try:
        n = len(bins)
    except TypeError:
        n = 1
    b = bins
    if n not in {1, 2}:
        b = [np.asarray(bins)] * 2
    elif n == 1:
        b = [bins, bins] if np.ndim(bins) == 0 else None
    if b is None or (range is not None and len(range) != 2):
        return numpy()
    r = (None, None) if range is None else range
    try:
        edges = [_histdd_edges(v, bi, ri) for v, bi, ri in zip((x, y), b, r)]
    except Exception:  # numpy raises the same
        return numpy()
    if edges[0] is None or edges[1] is None:
        return numpy()
    h = _mplrs.histogram2d_counts(np.ascontiguousarray(x), np.ascontiguousarray(y),
                                  edges[0], edges[1])
    return h, edges[0], edges[1]
