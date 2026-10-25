"""
The Rust side of `matplotlib.spines.Spine`.

The four spines of a rectilinear Axes are drawn by the Rust axes, which
positions them (and the ticks along them) at draw time.  The Python
`Spine` keeps matplotlib's state and logic; its drawing properties, position
and bounds are mirrored into the Rust spine.
"""

import functools

from . import artist as martist

_SPINES = ("left", "right", "bottom", "top")

# properties handled by the Rust spine itself
_SKIP = {"generic", "transform", "clip_box", "sticky_x", "sticky_y"}


class _RsSpineHandle:
    """`set_many` / `get` on one spine of a Rust axes."""

    def __init__(self, rs, name):
        self._rs = rs
        self._name = name

    def set(self, key, value):
        if key not in _SKIP:
            self._rs.spine_set(self._name, key, value)

    def set_many(self, d):
        for k, v in d.items():
            if k not in _SKIP:
                self._rs.spine_set(self._name, k, v)

    def get(self, key):
        return self._rs.spine_get(self._name, key)

    def ident(self):
        return (id(self._rs), self._name)


def _patch_spines(Spine):
    orig_init = Spine.__init__

    @functools.wraps(orig_init)
    def __init__(self, axes, spine_type, path, **kwargs):
        orig_init(self, axes, spine_type, path, **kwargs)
        rs = getattr(axes, "_rs", None)
        if rs is not None and spine_type in _SPINES:
            self.__dict__["_h"] = _RsSpineHandle(rs, spine_type)
            martist._DIRTY.add(self)

    Spine.__init__ = __init__

    orig_props = Spine._rs_props

    def _rs_props(self):
        d = orig_props(self)
        pos = self._position
        if pos is not None and not isinstance(pos, str):
            pos = (str(pos[0]), float(pos[1]))
        d["position"] = pos
        d["bounds"] = None if self._bounds is None else tuple(map(float, self._bounds))
        return d

    Spine._rs_props = _rs_props

    def draw(self, renderer):
        # drawn by the Rust core as part of its Axes
        self.stale = False

    Spine.draw = martist.allow_rasterization(draw)
