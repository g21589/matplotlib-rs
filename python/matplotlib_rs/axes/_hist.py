"""`numpy.histogram` for `Axes.hist`, counting large unweighted float64 data
in Rust.

The bin edges (and the choice of numpy's equal-bins algorithm) come from
numpy's own ``_get_bin_edges``; the Rust loops are numpy's algorithms, so the
counts are the same.
"""
import numpy as np

from .. import _mplrs

try:  # NumPy 2
    from numpy.lib._histograms_impl import _get_bin_edges
except ImportError:
    try:  # NumPy 1.25, 1.26
        from numpy.lib.histograms import _get_bin_edges
    except ImportError:  # (numpy.histogram is used as it is)
        _get_bin_edges = None

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
