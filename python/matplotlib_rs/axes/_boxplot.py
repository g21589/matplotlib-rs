"""Create independent boxplot artists from validated style prototypes.

bxp knows every component's numeric coordinates and reuses a handful of styles.
It need not repeat plot's format/unit parsing, clipping setup and path limits
for each of its thousands of two-point lines. The returned objects remain
ordinary, independently mutable Line2D and PathPatch instances.
"""
import functools
import weakref

import numpy as np

from .. import artist, cbook, lines, patches
from ..path import Path
from ._base_rs import _RsChildren, _rs_native, _rs_py_drawn, _rs_switch_child


class _ArtistCallbacks(cbook.CallbackRegistry):
    """Allocate callback maps when this independent registry is first used."""

    def __init__(self):
        self._initialized = False

    def _initialize(self):
        if not self.__dict__.get("_initialized", False):
            pending = self.__dict__.copy()
            super().__init__(signals=["pchanged"])
            self.__dict__.update(pending)
            self._initialized = True

    def __getattr__(self, name):
        self._initialize()
        try:
            return self.__dict__[name]
        except KeyError:
            raise AttributeError(name) from None

    def __getstate__(self):
        self._initialize()
        return super().__getstate__()


class _BoxplotArtists:
    def __init__(self, axes, line_styles, patch_style):
        self.axes = axes
        self.styles = {}
        self.points = []
        self.switch = functools.partial(_rs_switch_child, weakref.ref(axes))
        for kw in line_styles:
            key = tuple(kw.items())
            hash(key)  # mutable/complex styles use the ordinary creation path
            if key in self.styles:
                continue
            prototype = lines.Line2D([], [], **kw)
            self._prepare(prototype)
            props = prototype._rs_props()
            prototype._h.set_many(props)
            self.styles[key] = prototype
        self.patch_style = None
        if patch_style is not None:
            prototype = patches.PathPatch(Path(np.empty((0, 2))), **patch_style)
            self._prepare(prototype)
            props = prototype._rs_props()
            prototype._h.set_many(props)
            self.patch_style = prototype

    def _prepare(self, prototype):
        self.axes._set_artist_props(prototype)
        if prototype.get_clip_path() is None:
            prototype.set_clip_path(self.axes.patch)
        if not _rs_native(prototype) or prototype.get_mouseover():
            raise TypeError("artist requires the general creation path")
        artist._DIRTY.discard(prototype)

    @staticmethod
    def create(axes, stats, line_styles, patch_style):
        from numbers import Real
        from ._axes import Axes
        if (len(stats) == 0 or type(axes) is not Axes
                or not isinstance(axes._children, _RsChildren) or _rs_py_drawn(axes)
                or getattr(axes.plot, "__func__", None) is not Axes.plot
                or getattr(axes.add_artist, "__func__", None) is not Axes.add_artist
                or any(axis.have_units() for axis in axes._axis_map.values())
                or any("transform" in kw for kw in line_styles)
                or patch_style is not None and "transform" in patch_style):
            return None
        cycle_keys = axes._get_lines._prop_cycle._cycler_items[0]
        if any(kw.get(key) is None for kw in line_styles for key in cycle_keys):
            return None  # Preserve plot's per-line advancement of property cycles.
        for stat in stats:
            if any(not isinstance(stat[k], Real) for k in ("q1", "q3", "med", "whislo", "whishi")):
                return None
            if any(not isinstance(stat[k], Real) for k in ("mean", "cilo", "cihi") if k in stat):
                return None
            if "fliers" in stat and np.asarray(stat["fliers"]).dtype.kind not in "biuf":
                return None
        try:
            return _BoxplotArtists(axes, line_styles, patch_style)
        except (TypeError, NotImplementedError):
            return None

    def _copy(self, prototype, handle):
        a = type(prototype).__new__(type(prototype))
        a.__dict__ = prototype.__dict__.copy()
        a._h = handle
        a._callbacks = _ArtistCallbacks()
        a._remove_method = self.axes._children.remove
        a._rs_native_in = True
        a._rs_switch = self.switch
        # State is already installed in the native handle. Future mutations
        # use the same dirty/switch machinery as conventionally added artists.
        list.append(self.axes._children, a)
        self.axes._rs.add_child_raw(handle)
        return a

    def line(self, xs, ys, kw):
        prototype = self.styles[tuple(kw.items())]
        x, y = np.array(xs, copy=True), np.array(ys, copy=True)
        h = prototype._h.copy_with_data(x, y)
        a = self._copy(prototype, h)
        a._xorig, a._yorig = x, y
        a._in_autoscale = True
        if not a._label:
            a._label = f"_child{len(self.axes._children) - 1}"
            h.set("label", a._label)
        self.points.extend(zip(x, y))
        return a

    def patch(self, xs, ys):
        prototype = self.patch_style
        path = Path._create_closed(np.column_stack([xs, ys]))
        h = prototype._h.copy_with_path(path.vertices.tolist(), path.codes.tolist())
        a = self._copy(prototype, h)
        a._path = path
        return a

    def finish(self):
        if self.points:
            self.axes.update_datalim(self.points)
        self.axes._request_autoscale_view()
        self.axes.stale = True
