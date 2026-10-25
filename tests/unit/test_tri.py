"""Compare matplotlib_rs._tri with matplotlib's C++ _tri module."""

import sys

import numpy as np

sys.path.insert(0, "python")

from matplotlib import _qhull, _tri as mtri  # noqa: E402
from matplotlib_rs import _tri as rtri  # noqa: E402

rng = np.random.default_rng(1)
fails = 0
checks = 0


def same(a, b, what):
    global fails, checks
    checks += 1
    if isinstance(a, (list, tuple)):
        ok = len(a) == len(b) and all(
            np.array_equal(np.asarray(u), np.asarray(v)) for u, v in zip(a, b))
    else:
        ok = np.array_equal(np.asarray(a), np.asarray(b))
    if not ok:
        fails += 1
        if fails < 10:
            print("MISMATCH", what)


for case in range(40):
    n = int(rng.integers(5, 200))
    if case % 4 == 0:  # grid with duplicate-free regular points (degenerate cases)
        g = int(np.sqrt(n)) + 2
        xx, yy = np.meshgrid(np.arange(g, dtype=float), np.arange(g, dtype=float))
        x, y = xx.ravel(), yy.ravel()
    else:
        x, y = rng.random(n), rng.random(n)
    tris, neigh = _qhull.delaunay(x, y, 0)
    mask = rng.random(len(tris)) < 0.15 if case % 3 == 0 else None
    shuffled = tris[:, ::-1].copy() if case % 5 == 0 else tris.copy()
    args = (x, y, shuffled, mask if mask is not None else (), (), (), True)
    tm = mtri.Triangulation(*args[:2], shuffled.copy(), *args[3:])
    rs_tris = shuffled.copy()
    tr = rtri.Triangulation(x, y, rs_tris, *args[3:])
    same(tm.get_edges(), tr.get_edges(), "edges")
    same(tm.get_neighbors(), tr.get_neighbors(), "neighbors")
    z = np.sin(3 * x) * np.cos(4 * y) + (x if case % 2 else 0)
    same(tm.calculate_plane_coefficients(z), tr.calculate_plane_coefficients(z), "planes")
    cm = mtri.TriContourGenerator(tm, z)
    cr = rtri.TriContourGenerator(tr, z)
    for level in np.linspace(z.min(), z.max(), 7)[1:-1]:
        a = cm.create_contour(level)
        b = cr.create_contour(level)
        same(a[0], b[0], f"contour verts {level}")
        same(a[1], b[1], f"contour codes {level}")
    levels = np.linspace(z.min() - 0.1, z.max() + 0.1, 6)
    for lo, hi in zip(levels[:-1], levels[1:]):
        a = cm.create_filled_contour(lo, hi)
        b = cr.create_filled_contour(lo, hi)
        same(a[0], b[0], "filled verts")
        same(a[1], b[1], "filled codes")
    fm = mtri.TrapezoidMapTriFinder(tm)
    fr = rtri.TrapezoidMapTriFinder(tr)
    fm.initialize()
    fr.initialize()
    same(fm.get_tree_stats(), fr.get_tree_stats(), f"tree stats case {case} {fm.get_tree_stats()} {fr.get_tree_stats()}")
    px = np.concatenate([rng.random(300) * 1.2 - 0.1, x])
    py = np.concatenate([rng.random(300) * 1.2 - 0.1, y])
    same(fm.find_many(px, py), fr.find_many(px, py), "find_many")
    if mask is not None:
        m2 = rng.random(len(tris)) < 0.3
        tm.set_mask(m2)
        tr.set_mask(m2)
        same(tm.get_edges(), tr.get_edges(), "edges after set_mask")

# orientation correction happens in place on int32 arrays, as in C++
t = np.array([[0, 2, 1]], dtype=np.int32)
rtri.Triangulation(np.array([0., 1, 0]), np.array([0., 0, 1]), t, (), (), (), True)
same(t, [[0, 1, 2]], "in-place orientation")
print(f"{checks - fails}/{checks} identical")
sys.exit(1 if fails else 0)
