"""
The Rust side of `matplotlib.axes._base._AxesBase`.

The Python Axes keeps all of matplotlib's state and logic (limits,
autoscaling, aspect, units, ...).  Each Axes owns an Axes of the Rust model
that mirrors the state needed for drawing (position, view and data limits)
and draws the axes with its spines, ticks, titles and children.  Children
the Rust core can draw itself (lines, patches, texts) are handed over as
Rust handles; any other artist is drawn by its own ``draw`` method.
"""

import functools
import weakref

from .. import _mplrs
from .. import artist as martist
from .. import lines as mlines
from .. import patches as mpatches
from .. import text as mtext
from .. import transforms as mtransforms


class _FlushBbox(mtransforms.Bbox):
    """A Bbox whose changes are mirrored to the Rust model at the next sync."""

    def __init__(self, points, owner, **kwargs):
        super().__init__(points, **kwargs)
        self._rs_owner_ref = weakref.ref(owner)

    def invalidate(self):
        owner = self._rs_owner_ref()
        if owner is not None:
            martist._DIRTY.add(owner)
        return super().invalidate()


_FlushBbox.__name__ = "Bbox"  # (as printed by transforms)


# methods whose overrides decide an artist's geometry, and the modules whose
# artists keep it up to date by marking themselves stale
_GEOMETRY_METHODS = ("get_path", "get_patch_transform", "get_xydata", "recache",
                     "_get_transformed_path", "get_text", "get_window_extent")
_CORE_MODULES = {"matplotlib_rs.lines", "matplotlib_rs.patches", "matplotlib_rs.text",
                 "matplotlib_rs.artist"}


def _rs_native(a):
    """Whether the Rust core draws *a* itself (instead of ``a.draw``)."""
    h = a.__dict__.get("_h")
    if not isinstance(h, (_mplrs.Line2D, _mplrs.Text, _mplrs.Patch)):
        return False
    cls = type(a)
    if isinstance(a, mlines.Line2D):
        base = mlines.Line2D
    elif isinstance(a, mtext.Text):
        base = mtext.Text
        if a.get_usetex() or a._bbox_patch is not None:
            return False
    elif isinstance(a, mpatches.Patch):
        base = mpatches.Patch
    else:
        return False
    # subclasses computing their transform at draw time (e.g. axline's)
    if cls.draw is not base.draw or cls.get_transform is not base.get_transform:
        return False
    # subclasses from elsewhere may compute their geometry when drawn (e.g.
    # the connectors of mpl_toolkits' mark_inset)
    for name in _GEOMETRY_METHODS:
        m = getattr(cls, name, None)
        if m is not None and getattr(m, "__module__", None) not in _CORE_MODULES:
            return False
    if a.get_path_effects() or a.get_agg_filter() is not None or a.get_rasterized():
        return False
    return True


def _rs_attach(ax, a):
    """Add child *a* (just appended to ``ax._children``) to the Rust axes."""
    native = _rs_native(a)
    a._rs_native_in = native
    a._rs_switch = functools.partial(_rs_switch_child, weakref.ref(ax))
    if native:
        martist._DIRTY.discard(a)
        a._rs_sync()
        ax._rs.add_child_raw(a._h)
    else:
        ax._rs.add_child_raw(a)


def _rs_detach(ax, a):
    """Remove child *a* (just removed from ``ax._children``) from the Rust axes."""
    rs = ax.__dict__.get("_rs") if ax is not None else None
    if rs is not None:
        rs.remove_child(a._h if a.__dict__.get("_rs_native_in") else a)


class _RsChildren(list):
    """``Axes._children``, keeping the children of the Rust axes in step.

    matplotlib adds children by appending to this list (and removes them
    through ``_remove_method = _children.remove``).  ``clear`` does not
    notify: matplotlib only clears a list it has already swapped out, after
    the Rust axes has been cleared.
    """

    __slots__ = ("_axref",)

    def __init__(self, ax, items=()):
        super().__init__()
        self._axref = weakref.ref(ax)
        self.extend(items)

    def append(self, a):
        super().append(a)
        ax = self._axref()
        if ax is not None:
            _rs_attach(ax, a)

    def extend(self, items):
        for a in items:
            self.append(a)

    def insert(self, i, a):
        super().insert(i, a)
        ax = self._axref()
        if ax is not None:
            _rs_attach(ax, a)

    def remove(self, a):
        super().remove(a)
        _rs_detach(self._axref(), a)

    def pop(self, i=-1):
        a = super().pop(i)
        _rs_detach(self._axref(), a)
        return a

    def __delitem__(self, i):
        items = self[i] if isinstance(i, slice) else [self[i]]
        super().__delitem__(i)
        for a in items:
            _rs_detach(self._axref(), a)


def _rs_switch_child(axref, a):
    """Swap a child between native and Python drawing (keeping its place)."""
    ax = axref()
    native = _rs_native(a)
    if ax is None or native == a.__dict__.get("_rs_native_in"):
        return
    old = a._h if a._rs_native_in else a
    new = a._h if native else a
    if ax._rs.replace_child(old, new):
        a._rs_native_in = native


def _rs_attr(name):
    """An attribute mirrored into the Rust axes (by ``_rs_sync``)."""
    key = "_rs_attr_" + name

    def getter(self):
        return self.__dict__.get(key)

    def setter(self, v):
        self.__dict__[key] = v
        martist._DIRTY.add(self)

    return property(getter, setter)


def _rs_py_drawn(ax):
    """Whether *ax* is drawn by matplotlib's Python code rather than by the
    Rust core: non-rectilinear projections (polar, geo, 3d) and axes with
    their own children (mpl_toolkits.axisartist)."""
    from ._base import _AxesBase
    return ax.name != "rectilinear" or type(ax).get_children is not _AxesBase.get_children


def _patch_axes_base(cls):
    orig_draw = cls.draw
    orig_set_figure = cls.set_figure
    orig_init_axis = cls._init_axis

    def _init_axis(self):
        if not _rs_py_drawn(self):
            return orig_init_axis(self)
        # axes drawn by Python use matplotlib's own Axis (see _axis_py.py)
        from .. import _axis_py
        self.xaxis = _axis_py.XAxis(self, clear=False)
        self.spines.bottom.register_axis(self.xaxis)
        self.spines.top.register_axis(self.xaxis)
        self.yaxis = _axis_py.YAxis(self, clear=False)
        self.spines.left.register_axis(self.yaxis)
        self.spines.right.register_axis(self.yaxis)

    _init_axis.__doc__ = orig_init_axis.__doc__
    cls._init_axis = _init_axis
    orig_set_lim_and_transforms = cls._set_lim_and_transforms
    orig_clear = cls._AxesBase__clear
    orig_gen_axes_patch = cls._gen_axes_patch

    def set_figure(self, fig):
        martist.Artist.set_figure(self, fig)
        if self.__dict__.get("_rs") is None:
            root = fig.get_figure(root=True)
            uid = root._rs.new_axes(tuple(map(float, self._position.bounds)))
            self._rs = root._rs.axes(uid)
            if _rs_py_drawn(self):
                self._rs.set_ext_draw(self)
        self._position = _FlushBbox(self._position.get_points(), self)
        if fig.transSubfigure is not fig.get_figure(root=True).transFigure:
            from .._live import PyTransformRef
            self._rs.set_trans_subfigure(("py", PyTransformRef(fig.transSubfigure)))
        self.bbox = mtransforms.TransformedBbox(self._position, fig.transSubfigure)
        self.bbox._rs_clip_spec = "axes"
        self.bbox._rs_bbox_spec = "axes"
        # these will be updated later as data is added
        self.dataLim = _FlushBbox(mtransforms.Bbox.null().get_points(), self)
        self._viewLim = _FlushBbox(mtransforms.Bbox.unit().get_points(), self)
        self.transScale = mtransforms.TransformWrapper(
            mtransforms.IdentityTransform())
        self._set_lim_and_transforms()
        martist._DIRTY.add(self)

    set_figure.__doc__ = orig_set_figure.__doc__
    cls.set_figure = set_figure

    def _set_lim_and_transforms(self):
        orig_set_lim_and_transforms(self)
        if _rs_py_drawn(self):
            return
        self.transAxes._rs_spec = "axes"
        self.transData._rs_spec = "data"

    _set_lim_and_transforms.__doc__ = orig_set_lim_and_transforms.__doc__
    cls._set_lim_and_transforms = _set_lim_and_transforms

    def _gen_axes_patch(self):
        p = orig_gen_axes_patch(self)
        p._rs_axes_patch = True
        return p

    _gen_axes_patch.__doc__ = orig_gen_axes_patch.__doc__
    cls._gen_axes_patch = _gen_axes_patch

    def __clear(self):
        rs = self._rs
        rs.cla()
        orig_clear(self)
        if not _rs_py_drawn(self):
            # (axes drawn by Python draw their children themselves, and may
            # change them while drawing, e.g. parasite axes)
            self._children = _RsChildren(self, self._children)
        for loc, t in (("center", self.title), ("left", self._left_title),
                       ("right", self._right_title)):
            rs.set_title_handle(loc, t._h)
            t._rs_native_in = True
        # matplotlib sets the spine positions when it creates the first
        # ticks; the Rust core creates the ticks itself.
        if not _rs_py_drawn(self):
            for spine in self.spines.values():
                spine._ensure_position_is_set()
        self.patch.__dict__["_h"] = mpatches._RsPatchHandle(rs)
        martist._DIRTY.add(self.patch)
        martist._DIRTY.add(self)

    cls._AxesBase__clear = __clear

    def _rs_sync(self):
        """Mirror position and limits into the Rust axes."""
        p = self._position
        o = self._originalPosition
        v = self._viewLim
        d = self.dataLim
        mp = d.minpos
        self._rs.mirror_state(
            tuple(map(float, p.extents)), tuple(map(float, o.extents)),
            tuple(map(float, v.extents)), tuple(map(float, d.extents)),
            (float(mp[0]), float(mp[1])), float(self.zorder), bool(self.get_visible()),
            bool(self.axison), bool(self._frameon),
            None if self._autotitlepos is None else bool(self._autotitlepos))

    cls._rs_sync = _rs_sync

    def _rs_prepare_draw(self, renderer):
        """What matplotlib's ``Axes.draw`` does before drawing children."""
        if _rs_py_drawn(self):
            martist._DIRTY.add(self)
            return
        self._unstale_viewLim()
        locator = self.get_axes_locator()
        self.apply_aspect(locator(self, renderer) if locator else None)
        for axis in self._axis_map.values():
            axis._rs_sync_tickers()
        martist._DIRTY.add(self)

    cls._rs_prepare_draw = _rs_prepare_draw

    def draw(self, renderer):
        # Normally drawn by the Rust core as part of the figure.
        if _rs_py_drawn(self):
            return orig_draw(self, renderer)
        if not self.get_visible():
            return
        from .._foreign import is_foreign, metrics
        with metrics(renderer):
            self._rs_prepare_draw(renderer)
            self.get_figure(root=True)._rs_sync_all()
            with martist._rs_drawing():
                if is_foreign(renderer):
                    self._rs.draw_foreign(renderer)
                elif not renderer._rs_disabled:
                    self._rs.draw_into(renderer._renderer)
        self.stale = False

    cls.draw = martist.allow_rasterization(draw)

    # child axes (insets, secondary axes) are drawn among the children
    orig_add_child_axes = cls.add_child_axes

    @functools.wraps(orig_add_child_axes)
    def add_child_axes(self, ax):
        ret = orig_add_child_axes(self, ax)
        crs = ax.__dict__.get("_rs")
        if crs is not None:
            self._rs.add_child_axes(crs.uid)
        return ret

    cls.add_child_axes = add_child_axes

    # the legend is drawn among the children
    def _get_legend_(self):
        return self.__dict__.get("_rs_legend_")

    def _set_legend_(self, leg):
        rs = self.__dict__.get("_rs")
        old = self.__dict__.get("_rs_legend_")
        if old is not None and rs is not None:
            rs.remove_child(old)
        self.__dict__["_rs_legend_"] = leg
        if leg is not None and rs is not None:
            rs.add_child_raw(leg)

    cls.legend_ = property(_get_legend_, _set_legend_)

    # -- plain attributes that the Rust axes needs ------------------------------
    cls.axison = _rs_attr("axison")
    cls._frameon = _rs_attr("frameon")
    cls._autotitlepos = _rs_attr("autotitlepos")

    def __del__(self):
        rs = self.__dict__.get("_rs")
        if rs is not None:
            try:
                rs.figure().delaxes(rs.uid)
            except Exception:
                pass

    cls.__del__ = __del__
