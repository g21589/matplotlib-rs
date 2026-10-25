"""
Replacement for matplotlib's ``_qhull`` extension module (Delaunay
triangulation).

matplotlib triangulates with the qhull library.  When matplotlib itself is
installed next to matplotlib_rs, its compiled ``_qhull`` module is used, so
that triangulations (including the order of the triangles, which affects
drawing) are identical.  Otherwise a Rust Delaunay triangulation is used:
it has the same triangles, possibly in another order.  Set the environment
variable ``MPLRS_QHULL=rust`` to always use the Rust triangulation.
"""

import glob
import importlib.machinery
import importlib.util
import os
import sys

import numpy as np

from . import _mplrs


def _find_matplotlib_qhull():
    if os.environ.get("MPLRS_QHULL") == "rust":
        return None
    here = os.path.dirname(os.path.abspath(__file__))
    for entry in sys.path:
        d = os.path.join(entry or os.getcwd(), "matplotlib")
        if os.path.abspath(d) == here:
            continue
        for path in glob.glob(os.path.join(d, "_qhull*")):
            if not path.endswith(tuple(importlib.machinery.EXTENSION_SUFFIXES)):
                continue
            try:
                loader = importlib.machinery.ExtensionFileLoader("matplotlib._qhull", path)
                spec = importlib.util.spec_from_file_location(
                    "matplotlib._qhull", path, loader=loader)
                mod = importlib.util.module_from_spec(spec)
                loader.exec_module(mod)
                return mod
            except Exception:
                continue
    return None


_qhull = _find_matplotlib_qhull()


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
    if _qhull is not None:
        return _qhull.delaunay(x, y, verbose)
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
    return _mplrs.delaunay(x, y)


def version():
    return _qhull.version() if _qhull is not None else "matplotlib_rs (Rust Delaunay)"
