"""
Replacement for matplotlib's ``_qhull`` extension module (Delaunay
triangulation).

The triangulation is computed by the qhull library matplotlib builds
(extern/qhull, compiled into the Rust extension) with matplotlib's options and
post-processing, so that the triangles, their order and the order of their
vertices are identical.  Set the environment variable ``MPLRS_QHULL=rust`` to
use the Rust Delaunay triangulation instead: it has the same triangles,
possibly in another order.
"""

import os

import numpy as np

from . import _mplrs


def _at_least_3_unique_points(x, y):
    unique2 = 0
    for i in range(1, len(x)):
        if unique2 == 0:
            if x[i] != x[0] or y[i] != y[0]:
                unique2 = i
        elif ((x[i] != x[0] or y[i] != y[0])
              and (x[i] != x[unique2] or y[i] != y[unique2])):
            return True
    return False


def delaunay(x, y, verbose):
    """
    Compute a Delaunay triangulation.

    Parameters
    ----------
    x, y : 1d arrays
        The coordinates of the point set, which must consist of at least
        three unique points.
    verbose : int
        Python's verbosity level.

    Returns
    -------
    triangles, neighbors : int arrays, shape (ntri, 3)
        Indices of triangle vertices and indices of triangle neighbors.
    """
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D arrays")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must be 1D arrays of the same length")
    if x.shape[0] < 3:
        raise ValueError("x and y arrays must have a length of at least 3")
    if not _at_least_3_unique_points(x, y):
        raise ValueError("x and y arrays must consist of at least 3 unique points")
    if os.environ.get("MPLRS_QHULL") == "rust":
        return _mplrs.delaunay(x, y)
    return _mplrs.qhull_delaunay(x, y, verbose)


def version():
    """Return the qhull version string."""
    return _mplrs.qhull_version()
