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
        if a.get_usetex() or (a.get_parse_math() and "$" in a.get_text()):
            return False
    elif isinstance(a, mpatches.Patch):
        base = mpatches.Patch
    else:
        return False
    if cls.draw is not base.draw:
        return False
    if a.get_path_effects() or a.get_agg_filter() is not None or a.get_rasterized():
        return False
    return True


def _rs_attach(ax, a):
    """Add child *a* (already in ``ax._children``) to the Rust axes."""
    native = _rs_native(a)
    a._rs_native_in = native
    if native:
        martist._DIRTY.discard(a)
        a._rs_sync()
        ax._rs.add_child_raw(a._h)
    else:
        ax._rs.add_child_raw(a)
    a._remove_method = functools.partial(_rs_remove_child, ax)


def _rs_remove_child(ax, a):
    ax._children.remove(a)
    rs = ax.__dict__.get("_rs")
    if rs is not None:
        rs.remove_child(a._h if a.__dict__.get("_rs_native_in") else a)


def _rs_attr(name, push):
    """An attribute whose value is also pushed to the Rust axes."""
    key = "_rs_attr_" + name

    def getter(self):
        return self.__dict__.get(key)

    def setter(self, v):
        self.__dict__[key] = v
        rs = self.__dict__.get("_rs")
        if rs is not None:
            push(rs, v)

    return property(getter, setter)


def _patch_axes_base(cls):
    orig_set_figure = cls.set_figure
    orig_set_lim_and_transforms = cls._set_lim_and_transforms
    orig_clear = cls._AxesBase__clear
    orig_gen_axes_patch = cls._gen_axes_patch

    def set_figure(self, fig):
        martist.Artist.set_figure(self, fig)
        if self.__dict__.get("_rs") is None:
            root = fig.get_figure(root=True)
            uid = root._rs.new_axes(tuple(map(float, self._position.bounds)))
            self._rs = root._rs.axes(uid)
        self._position = _FlushBbox(self._position.get_points(), self)
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
        for loc, t in (("center", self.title), ("left", self._left_title),
                       ("right", self._right_title)):
            rs.set_title_handle(loc, t._h)
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
            (float(mp[0]), float(mp[1])))

    cls._rs_sync = _rs_sync

    def _rs_prepare_draw(self, renderer):
        """What matplotlib's ``Axes.draw`` does before drawing children."""
        self._unstale_viewLim()
        locator = self.get_axes_locator()
        self.apply_aspect(locator(self, renderer) if locator else None)
        for axis in self._axis_map.values():
            axis._rs_sync_tickers()
        martist._DIRTY.add(self)

    cls._rs_prepare_draw = _rs_prepare_draw

    def draw(self, renderer):
        # Normally drawn by the Rust core as part of the figure.
        if not self.get_visible():
            return
        self._rs_prepare_draw(renderer)
        self.get_figure(root=True)._rs_sync_all()
        target = getattr(renderer, "_renderer", None)
        if isinstance(target, _mplrs.RendererAgg) and not renderer._rs_disabled:
            self._rs.draw_into(target)
        self.stale = False

    cls.draw = martist.allow_rasterization(draw)

    # -- children ------------------------------------------------------------
    def _wrap_adder(name):
        orig = getattr(cls, name)

        @functools.wraps(orig)
        def adder(self, artist, *args, **kwargs):
            ret = orig(self, artist, *args, **kwargs)
            _rs_attach(self, artist)
            return ret

        setattr(cls, name, adder)

    for name in ["add_artist", "add_collection", "add_image", "add_line",
                 "_add_text", "add_patch", "add_table"]:
        _wrap_adder(name)

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
    cls.axison = _rs_attr("axison", lambda rs, v: rs.set_axison(bool(v)))
    cls._frameon = _rs_attr("frameon", lambda rs, v: rs.set_frameon(bool(v)))
    cls._autotitlepos = _rs_attr(
        "autotitlepos", lambda rs, v: rs.set_autotitlepos(None if v is None else bool(v)))

    def __del__(self):
        rs = self.__dict__.get("_rs")
        if rs is not None:
            try:
                rs.figure().delaxes(rs.uid)
            except Exception:
                pass

    cls.__del__ = __del__
