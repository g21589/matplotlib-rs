"""
Replacement for matplotlib's ``_tri`` extension module, backed by the Rust
port of ``src/tri/_tri.cpp``.
"""

import numpy as np

from . import _mplrs


def _coords(a):
    return np.ascontiguousarray(a, dtype=np.float64)


def _ints(a):
    return np.ascontiguousarray(a, dtype=np.int32)


class Triangulation:
    """
    Create a new C++ Triangulation object.
    This should not be called directly, use the python class
    matplotlib.tri.Triangulation instead.
    """

    def __init__(self, x, y, triangles, mask, edges, neighbors,
                 correct_triangle_orientations):
        x, y = _coords(x), _coords(y)
        triangles = _ints(triangles)
        mask = np.ascontiguousarray(mask, dtype=bool)
        edges, neighbors = _ints(edges), _ints(neighbors)
        if x.ndim != 1 or y.ndim != 1 or x.shape[0] != y.shape[0]:
            raise ValueError("x and y must be 1D arrays of the same length")
        if triangles.ndim != 2 or triangles.shape[1] != 3:
            raise ValueError("triangles must be a 2D array of shape (?,3)")
        if mask.size > 0 and (mask.ndim != 1 or mask.shape[0] != triangles.shape[0]):
            raise ValueError(
                "mask must be a 1D array with the same length as the triangles array")
        if edges.size > 0 and (edges.ndim != 2 or edges.shape[1] != 2):
            raise ValueError("edges must be a 2D array with shape (?,2)")
        if neighbors.size > 0 and (neighbors.ndim != 2
                                   or neighbors.shape != triangles.shape):
            raise ValueError(
                "neighbors must be a 2D array with the same shape as the triangles array")
        if correct_triangle_orientations:
            # in place, like the C++ code (which shares int32 C arrays)
            _mplrs.tri_correct_triangles(
                x, y, triangles, neighbors if neighbors.size > 0 else None)
        self._x = x
        self._ntri = triangles.shape[0]
        self._rs = _mplrs.Triangulation(
            x, y, triangles, mask.reshape(-1),
            edges.reshape(-1, 2), neighbors.reshape(-1, 3))
        self._edges = edges if edges.size > 0 else None
        self._neighbors = neighbors if neighbors.size > 0 else None

    def calculate_plane_coefficients(self, z):
        """Calculate plane equation coefficients for all unmasked triangles."""
        z = _coords(z)
        if z.ndim != 1 or z.shape[0] != self._x.shape[0]:
            raise ValueError(
                "z must be a 1D array with the same length as the triangulation "
                "x and y arrays")
        return self._rs.calculate_plane_coefficients(z)

    def get_edges(self):
        """Return edges array."""
        if self._edges is None:
            self._edges = self._rs.get_edges()
        return self._edges

    def get_neighbors(self):
        """Return neighbors array."""
        if self._neighbors is None:
            self._neighbors = self._rs.get_neighbors()
        return self._neighbors

    def set_mask(self, mask):
        """Set or clear the mask array."""
        mask = np.ascontiguousarray(mask, dtype=bool)
        if mask.size > 0 and (mask.ndim != 1 or mask.shape[0] != self._ntri):
            raise ValueError(
                "mask must be a 1D array with the same length as the triangles array")
        self._rs.set_mask(mask.reshape(-1))
        self._edges = None
        self._neighbors = None



class TriContourGenerator:
    """
    Create a new C++ TriContourGenerator object.
    This should not be called directly, use the functions
    matplotlib.axes.tricontour and tricontourf instead.
    """

    def __init__(self, triangulation, z):
        z = _coords(z)
        if z.ndim != 1:
            raise ValueError(
                "z must be a 1D array with the same length as the x and y arrays")
        self._rs = _mplrs.TriContourGenerator(triangulation._rs, z)

    def create_contour(self, level):
        """Create and return a non-filled contour."""
        return self._rs.create_contour(float(level))

    def create_filled_contour(self, lower_level, upper_level):
        """Create and return a filled contour."""
        return self._rs.create_filled_contour(float(lower_level), float(upper_level))


class TrapezoidMapTriFinder:
    """
    Create a new C++ TrapezoidMapTriFinder object.
    This should not be called directly, use the python class
    matplotlib.tri.TrapezoidMapTriFinder instead.
    """

    def __init__(self, triangulation):
        self._rs = _mplrs.TrapezoidMapTriFinder(triangulation._rs)

    def find_many(self, x, y):
        """Find indices of triangles containing the point coordinates (x, y)."""
        x, y = _coords(x), _coords(y)
        if x.ndim != 1 or y.ndim != 1 or x.shape[0] != y.shape[0]:
            raise ValueError("x and y must be array-like with same shape")
        return self._rs.find_many(x, y)

    def get_tree_stats(self):
        """Return statistics about the tree used by the trapezoid map."""
        return list(self._rs.get_tree_stats())

    def initialize(self):
        """Initialize this object, creating the trapezoid map from the triangulation."""
        self._rs.initialize()

    def print_tree(self):
        """Print the search tree as text to stdout; useful for debug purposes."""
        self._rs.print_tree()
