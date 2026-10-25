"""Test helper: push a matplotlib rcParams mapping into the Rust rc store."""
import numpy as np


def _is_colorkey(k):
    return "color" in k or k.endswith("facecolor") or k.endswith("edgecolor")


def rc_items(rc, to_rgba):
    out = {}
    for k, v in rc.items():
        if k == "axes.prop_cycle":
            cols = [to_rgba(c) for c in v.by_key().get("color", [])]
            out[k] = ("colorlist", cols)
        elif isinstance(v, bool):
            out[k] = ("bool", v)
        elif isinstance(v, (int, float)):
            out[k] = ("num", float(v))
        elif v is None:
            out[k] = ("none", None)
        elif isinstance(v, str):
            if _is_colorkey(k) and v not in ("auto", "inherit", "none", "edge"):
                try:
                    out[k] = ("color", tuple(to_rgba(v)))
                    continue
                except Exception:
                    pass
            out[k] = ("str", v)
        elif isinstance(v, (list, tuple)):
            if _is_colorkey(k) and len(v) in (3, 4) and all(isinstance(x, (int, float)) for x in v):
                out[k] = ("color", tuple(float(x) for x in v))
            elif all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v):
                out[k] = ("numlist", [float(x) for x in v])
            elif all(isinstance(x, str) for x in v):
                out[k] = ("strlist", list(v))
    return out


def dot_uses_fma():
    rng = np.random.default_rng(0)
    for _ in range(200):
        a = rng.standard_normal((3, 3))
        b = rng.standard_normal((3, 3))
        r = np.dot(b, a)
        for i in range(3):
            for j in range(3):
                naive = b[i, 0] * a[0, j] + b[i, 1] * a[1, j] + b[i, 2] * a[2, j]
                if r[i, j] != naive:
                    return True
    return False


def setup(mplrs, matplotlib):
    import matplotlib.colors as mc
    from matplotlib import font_manager as fm
    mplrs.rc_clear()
    mplrs.rc_set(rc_items(matplotlib.rcParams, mc.to_rgba))
    mplrs.set_font_list([(f.fname, f.index, f.name, f.style, f.variant,
                          f.weight if isinstance(f.weight, int) else fm.weight_dict.get(f.weight, 400),
                          str(f.stretch), str(f.size)) for f in fm.fontManager.ttflist])
    mplrs.set_dot_uses_fma(dot_uses_fma())
