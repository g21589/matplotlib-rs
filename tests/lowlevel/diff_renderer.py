"""Differential tests: ported Rust RendererAgg vs matplotlib's C++ RendererAgg.

Every primitive is drawn with randomized parameters into both renderers and
the resulting RGBA buffers are compared byte-for-byte.
"""
import sys

import numpy as np
from matplotlib.backend_bases import GraphicsContextBase
from matplotlib.backends._backend_agg import RendererAgg as MplRenderer
from matplotlib.path import Path
from matplotlib.transforms import Affine2D, Bbox, TransformedPath

from matplotlib_rs._mplrs import RendererAgg as RsRenderer

W, H, DPI = 160, 120, 100.0


def gc_to_dict(gc):
    d = dict(
        linewidth=gc._linewidth,
        alpha=gc._alpha,
        forced_alpha=gc._forced_alpha,
        rgb=tuple(gc._rgb),
        antialiased=bool(gc._antialiased),
        capstyle=gc._capstyle.name,
        joinstyle=gc._joinstyle.name,
        dashes=gc.get_dashes(),
        cliprect=None if gc._cliprect is None else tuple(gc._cliprect.extents),
        snap=gc.get_snap(),
        sketch=gc.get_sketch_params(),
    )
    cp, ct = gc.get_clip_path()
    if cp is not None:
        d["clippath"] = (np.asarray(cp.vertices, float), cp.codes, ct.get_matrix())
    return d


def random_gc(rng):
    gc = GraphicsContextBase()
    gc.set_linewidth(float(rng.choice([0, 0.5, 1, 1.5, 2.3, 4, 7.7])))
    gc.set_foreground(tuple(rng.random(4)) if rng.random() < 0.5 else tuple(rng.random(3)))
    if rng.random() < 0.3:
        gc.set_alpha(float(rng.random()))
    gc.set_antialiased(bool(rng.random() < 0.85))
    gc.set_capstyle(str(rng.choice(["butt", "round", "projecting"])))
    gc.set_joinstyle(str(rng.choice(["miter", "round", "bevel"])))
    if rng.random() < 0.3:
        gc.set_dashes(float(rng.random() * 5), list(rng.random(int(rng.integers(1, 5))) * 6 + 0.5))
    if rng.random() < 0.4:
        x0, y0 = rng.random(2) * [W / 2, H / 2]
        gc.set_clip_rectangle(Bbox.from_bounds(x0, y0, rng.random() * W, rng.random() * H))
    gc.set_snap([None, True, False][int(rng.integers(0, 3))])
    if rng.random() < 0.15:
        cp = Affine2D().scale(rng.random() * 60 + 10).translate(*(rng.random(2) * [W, H]))
        gc.set_clip_path(TransformedPath(Path.unit_circle(), cp))
    if rng.random() < 0.05:
        gc.set_sketch_params(float(rng.random() * 3), float(rng.random() * 50 + 5), float(rng.random() * 8 + 1))
    return gc


def random_path(rng):
    kind = rng.integers(0, 6)
    n = int(rng.integers(2, 40))
    if kind == 0:  # polyline without codes
        v = rng.random((n, 2)) * [W, H] * 1.4 - [0.2 * W, 0.2 * H]
        return Path(v)
    if kind == 1:  # rectilinear (snappable)
        v = np.cumsum(rng.integers(-30, 30, (n, 2)), axis=0).astype(float)
        v[1::2, 0] = v[0::2, 0][: len(v[1::2])]
        return Path(v + [W / 2, H / 2])
    if kind == 2:  # with NaNs
        v = rng.random((n, 2)) * [W, H]
        v[rng.random(n) < 0.2] = np.nan
        return Path(v)
    if kind == 3:  # closed polygon
        v = rng.random((n, 2)) * [W, H]
        codes = np.full(n, Path.LINETO, dtype=np.uint8)
        codes[0] = Path.MOVETO
        codes[-1] = Path.CLOSEPOLY
        return Path(v, codes)
    if kind == 4:  # circle / curves
        p = Path.unit_circle() if rng.random() < 0.5 else Path.unit_regular_star(5)
        return Affine2D().scale(rng.random() * 50 + 1).translate(*(rng.random(2) * [W, H])).transform_path(p)
    # dense line for simplification
    x = np.linspace(-0.1, 1.1, 2000) * W
    y = H / 2 + np.sin(x / 3) * 30 + rng.normal(0, 3, x.size)
    p = Path(np.column_stack([x, y]))
    return p


def random_trans(rng):
    t = Affine2D()
    if rng.random() < 0.5:
        t.scale(*(rng.random(2) * 1.5 + 0.25))
    if rng.random() < 0.3:
        t.rotate(rng.random() * 6.28)
    t.translate(*(rng.normal(0, 10, 2)))
    return t


def compare(a, b, label):
    a = np.asarray(a)
    b = np.asarray(b)
    if not np.array_equal(a, b):
        diff = np.argwhere(np.any(a != b, axis=2))
        y, x = diff[0]
        print(f"MISMATCH {label}: {len(diff)} px differ; first at (x={x}, y={y}) "
              f"mpl={a[y, x]} rs={b[y, x]}")
        return False
    return True


def run(n_iter=300, seed=0):
    rng = np.random.default_rng(seed)
    failures = 0
    visible = 0
    for it in range(n_iter):
        mr = MplRenderer(W, H, DPI)
        rr = RsRenderer(W, H, DPI)
        # white background like a figure patch
        bg = Path([[0, 0], [W, 0], [W, H], [0, H], [0, 0]])
        g0 = GraphicsContextBase()
        g0.set_linewidth(0)
        mr.draw_path(g0, bg, Affine2D(), (1, 1, 1))
        rr.draw_path(gc_to_dict(g0), bg.vertices, bg.codes, False, 1 / 9, Affine2D().get_matrix(), (1, 1, 1))
        for k in range(int(rng.integers(1, 5))):
            what = rng.random()
            gc = random_gc(rng)
            d = gc_to_dict(gc)
            if what < 0.5:
                p = random_path(rng)
                p.should_simplify = bool(len(p.vertices) >= 128 and p.codes is None)
                t = random_trans(rng)
                face = None
                if rng.random() < 0.5:
                    face = tuple(rng.random(4)) if rng.random() < 0.5 else tuple(rng.random(3))
                mr.draw_path(gc, p, t, face)
                rr.draw_path(d, np.asarray(p.vertices, float), p.codes, p.should_simplify,
                             p.simplify_threshold, t.get_matrix(), face)
                label = f"it{it} draw_path"
            elif what < 0.6:
                npaths = int(rng.integers(1, 4))
                paths = [random_path(rng) for _ in range(npaths)]
                no = int(rng.integers(0, 20))
                offsets = rng.random((no, 2)) * [W, H] if no else np.zeros((0, 2))
                nt = int(rng.integers(0, 3))
                transforms = np.array([Affine2D().scale(rng.random() + 0.2).get_matrix() for _ in range(nt)]).reshape((nt, 3, 3))
                fcs = rng.random((int(rng.integers(0, 3)), 4))
                ecs = rng.random((int(rng.integers(0, 3)), 4))
                lws = rng.random(int(rng.integers(0, 3))) * 3
                lss = [(0, None)] if rng.random() < 0.7 else [(0.0, [3.0, 2.0])]
                aas = np.array([bool(rng.random() < 0.8)])
                master = random_trans(rng)
                ot = Affine2D() if rng.random() < 0.5 else random_trans(rng)
                mr.draw_path_collection(gc, master.frozen(), paths, transforms, offsets, ot.frozen(), fcs, ecs,
                                        lws, lss, aas, None, 'screen')
                rr.draw_path_collection(d, master.get_matrix(), [(np.asarray(q.vertices, float), q.codes) for q in paths],
                                        transforms, np.asarray(offsets, float), ot.get_matrix(), fcs, ecs, lws, lss,
                                        aas.astype(np.uint8))
                label = f"it{it} draw_path_collection"
            elif what < 0.85:
                marker = [Path.unit_circle(), Path.unit_rectangle(), Path([[0, 0], [0, -1]]),
                          Path.unit_regular_polygon(3)][int(rng.integers(0, 4))]
                mt = Affine2D().scale(rng.random() * 8 + 0.5)
                pts = Path(rng.random((int(rng.integers(1, 30)), 2)) * [W, H])
                t = random_trans(rng)
                face = None if rng.random() < 0.3 else tuple(rng.random(3))
                mr.draw_markers(gc, marker, mt, pts, t, face)
                rr.draw_markers(d, np.asarray(marker.vertices, float), marker.codes, mt.get_matrix(),
                                np.asarray(pts.vertices, float), pts.codes, t.get_matrix(), face)
                label = f"it{it} draw_markers {d} marker={len(marker.vertices)} face={face}"
            else:
                img = (rng.random((int(rng.integers(1, 30)), int(rng.integers(1, 30)))) * 255).astype(np.uint8)
                x, y = int(rng.integers(-20, W)), int(rng.integers(-20, H + 20))
                mr.draw_text_image(img, x, y, 0, gc)
                rr.draw_text_image(img, x, y, d)
                label = f"it{it} draw_text_image"
            if not compare(mr, rr.buffer_rgba(), label):
                failures += 1
                break
        if np.any(np.asarray(mr)[..., :3] != 255):
            visible += 1
    print(f"{n_iter} iterations, {failures} failures, {visible} with visible output")
    return failures


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    seeds = range(int(sys.argv[2])) if len(sys.argv) > 2 else [0]
    sys.exit(1 if sum(run(n, s) for s in seeds) else 0)
