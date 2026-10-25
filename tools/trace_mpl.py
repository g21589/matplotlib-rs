"""Trace the renderer calls matplotlib's Agg backend makes while drawing a
figure.  Usage: python tools/trace_mpl.py script.py  (script must define `fig`
or leave a current figure)."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import RendererAgg

np.set_printoptions(precision=17, linewidth=200, threshold=40)


def gcinfo(gc):
    return dict(lw=gc._linewidth, rgb=tuple(round(v, 6) for v in gc._rgb), alpha=gc._alpha,
                forced=gc._forced_alpha, aa=gc._antialiased, cap=gc._capstyle.name,
                join=gc._joinstyle.name, dashes=gc.get_dashes(),
                clip=None if gc._cliprect is None else tuple(gc._cliprect.extents),
                snap=gc.get_snap(), clippath=gc.get_clip_path()[0] is not None)


def wrap(name, fn):
    def inner(self, *a, **k):
        if name == "draw_path":
            gc, path, trans = a[:3]
            face = a[3] if len(a) > 3 else k.get("rgbFace")
            print(f"draw_path {gcinfo(gc)}\n   verts={np.asarray(path.vertices).tolist()[:8]} n={len(path.vertices)} codes={None if path.codes is None else path.codes.tolist()[:8]} simplify={path.should_simplify}\n   trans={trans.get_matrix().tolist()} face={face}")
        elif name == "draw_markers":
            gc, mpath, mtrans, path, trans = a[:5]
            face = a[5] if len(a) > 5 else None
            print(f"draw_markers {gcinfo(gc)}\n   marker={np.asarray(mpath.vertices).tolist()[:6]} mtrans={mtrans.get_matrix().tolist()}\n   pts={np.asarray(path.vertices).tolist()[:6]} n={len(path.vertices)} trans={trans.get_matrix().tolist()} face={face}")
        elif name == "draw_text":
            gc, x, y, s, prop, angle = a[:6]
            print(f"draw_text {s!r} x={x!r} y={y!r} angle={angle} size={prop.get_size_in_points()} family={prop.get_family()} {gcinfo(gc)}")
        elif name == "draw_path_collection":
            print(f"draw_path_collection n_paths={len(a[2])} offsets={np.asarray(a[4]).shape} fc={np.asarray(a[6]).tolist()[:2]} ec={np.asarray(a[7]).tolist()[:2]} lw={list(a[8])}")
        elif name == "draw_image":
            print(f"draw_image x={a[1]} y={a[2]} shape={a[3].shape}")
        else:
            print(name)
        return fn(self, *a, **k)
    return inner


for name in ["draw_path", "draw_markers", "draw_text", "draw_path_collection", "draw_image"]:
    orig = getattr(RendererAgg, name, None)
    if orig is not None:
        setattr(RendererAgg, name, wrap(name, orig))
_orig_update = RendererAgg._update_methods
def _update(self):
    _orig_update(self)
    for name in ["draw_markers", "draw_path_collection", "draw_image"]:
        fn = getattr(self._renderer, name)
        setattr(self, name, (lambda n, f: (lambda *a, **k: wrap(n, lambda s, *aa, **kk: f(*aa, **kk))(self, *a, **k)))(name, fn))
RendererAgg._update_methods = _update

src = open(sys.argv[1]).read()
ns = {"__name__": "__main__"}
exec(compile(src, sys.argv[1], "exec"), ns)
fig = ns.get("fig", plt.gcf())
fig.canvas.draw()
