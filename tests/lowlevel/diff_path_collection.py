"""Differential test of large draw_path_collection calls (which matplotlib_rs
rasterizes on several threads and renders in order) against matplotlib's C++
RendererAgg: every call's RGBA buffer must be byte-for-byte identical.

Usage: python diff_path_collection.py [iterations] [seeds]
"""
import sys

import numpy as np
from matplotlib.backend_bases import GraphicsContextBase
from matplotlib.backends._backend_agg import RendererAgg as MplRenderer
from matplotlib.path import Path
from matplotlib.transforms import Affine2D

from matplotlib_rs._mplrs import RendererAgg as RsRenderer

from diff_renderer import compare, gc_to_dict, random_gc, random_trans

W, H, DPI = 200, 150, 100.0


def marker_path(rng):
    kind = int(rng.integers(0, 5))
    if kind == 0:
        return Path.unit_circle()
    if kind == 1:
        return Path.unit_rectangle()
    if kind == 2:
        return Path.unit_regular_star(int(rng.integers(3, 8)))
    if kind == 3:  # a short polyline (errorbar / LineCollection segments)
        return Path(rng.normal(size=(int(rng.integers(2, 5)), 2)))
    return Path.unit_regular_polygon(int(rng.integers(3, 9)))


def run(n_iter=40, seed=0):
    rng = np.random.default_rng(seed)
    failures = 0
    for it in range(n_iter):
        mr = MplRenderer(W, H, DPI)
        rr = RsRenderer(W, H, DPI)
        bg = Path([[0, 0], [W, 0], [W, H], [0, H], [0, 0]])
        g0 = GraphicsContextBase()
        g0.set_linewidth(0)
        mr.draw_path(g0, bg, Affine2D(), (1, 1, 1))
        rr.draw_path(gc_to_dict(g0), bg.vertices, bg.codes, False, 1 / 9, Affine2D().get_matrix(), (1, 1, 1))
        gc = random_gc(rng)
        d = gc_to_dict(gc)
        n = int(rng.choice([256, 300, 1000, 2048, 2049, 5000]))
        paths = [marker_path(rng) for _ in range(int(rng.choice([1, 1, 2, n])))]
        offsets = rng.random((n, 2)) * [W * 1.2, H * 1.2] - [W * 0.1, H * 0.1]
        if rng.random() < 0.1:
            offsets[rng.random(n) < 0.1] = np.nan
        nt = int(rng.choice([0, 1, 3, n]))
        transforms = np.array([Affine2D().scale(rng.random() * 6 + 0.3).get_matrix()
                               for _ in range(nt)]).reshape((nt, 3, 3))
        fcs = rng.random((int(rng.choice([0, 1, 2, n])), 4))
        if rng.random() < 0.3:
            fcs[:, 3] = 1
        ecs = rng.random((int(rng.choice([0, 1, 3, n])), 4))
        lws = rng.random(int(rng.choice([0, 1, 2, n]))) * 3
        lss = [(0, None)] if rng.random() < 0.7 else [(0.0, [3.0, 2.0]), (1.0, [1.0, 1.0])]
        aas = rng.random(int(rng.choice([1, 2, n]))) < 0.8
        master = random_trans(rng)
        ot = Affine2D() if rng.random() < 0.5 else random_trans(rng)
        mr.draw_path_collection(gc, master.frozen(), paths, transforms, offsets, ot.frozen(), fcs, ecs,
                                lws, lss, aas, None, 'screen')
        rr.draw_path_collection(d, master.get_matrix(), [(np.asarray(q.vertices, float), q.codes) for q in paths],
                                transforms, np.asarray(offsets, float), ot.get_matrix(), fcs, ecs, lws, lss,
                                aas.astype(np.uint8))
        if not compare(mr, rr.buffer_rgba(), f"it{it} n={n} paths={len(paths)} {d}"):
            failures += 1
    print(f"{n_iter} iterations, {failures} failures")
    return failures


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    seeds = range(int(sys.argv[2])) if len(sys.argv) > 2 else [0]
    sys.exit(1 if sum(run(n, s) for s in seeds) else 0)
