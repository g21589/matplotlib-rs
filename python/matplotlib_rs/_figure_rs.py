"""
The Rust side of `matplotlib.figure.Figure`.

The Python Figure keeps matplotlib's state and logic (size, dpi, layout
engines, axes management, ``savefig``); its Rust twin mirrors the geometry
and draws the figure: the patch, the axes and the figure-level artists.
"""

import functools

from . import _mplrs
from . import artist as martist
from . import patches as mpatches
from . import transforms as mtransforms
from .backend_bases import DrawEvent
from .axes._base_rs import _FlushBbox, _rs_native


class _FlushAffine2D(mtransforms.Affine2D):
    """An Affine2D whose changes are mirrored to the Rust model."""

    def __init__(self, owner, matrix=None, **kwargs):
        import weakref
        super().__init__(matrix, **kwargs)
        self._rs_owner_ref = weakref.ref(owner)

    def invalidate(self):
        owner = self._rs_owner_ref()
        if owner is not None:
            martist._DIRTY.add(owner)
        return super().invalidate()


class _FlushBboxTransformTo(mtransforms.BboxTransformTo):
    """``transFigure``: its ``_boxout`` is replaced by ``bbox_inches='tight'``."""

    def __init__(self, boxout, owner, **kwargs):
        import weakref
        super().__init__(boxout, **kwargs)
        self._rs_owner_ref = weakref.ref(owner)
        self._rs_spec = "figure"

    def invalidate(self):
        owner = self._rs_owner_ref()
        if owner is not None:
            martist._DIRTY.add(owner)
        return super().invalidate()


# the category the Rust figure gives to natively drawn artists
_NATIVE_CATEGORY = {"Line2D": 2, "Patch": 3, "Text": 4}


def _rs_fig_native(a, category):
    if not _rs_native(a):
        return False
    return _NATIVE_CATEGORY.get(type(a._h).__name__) == category


def _rs_fig_switch(rootref, category, sfid, a):
    root = rootref()
    native = _rs_fig_native(a, category)
    if root is None or native == a.__dict__.get("_rs_native_in"):
        return
    old = a._h if a._rs_native_in else a
    new = a._h if native else a
    if root._rs.replace_artist(old, new, sfid):
        a._rs_native_in = native


def _rs_fig_attach(fig, a, category):
    """Add figure-level artist *a* of (sub)figure *fig* to the Rust figure."""
    import weakref
    root = fig.get_figure(root=True)
    sfid = fig.__dict__.get("_rs_sfid")
    native = _rs_fig_native(a, category)
    a._rs_native_in = native
    a._rs_switch = functools.partial(_rs_fig_switch, weakref.ref(root), category, sfid)
    if native:
        martist._DIRTY.discard(a)
        a._rs_sync()
        root._rs.add_artist(a._h, category, sfid)
    else:
        root._rs.add_artist(a, category, sfid)
    owner_list = a._remove_method

    def remove(artist):
        owner_list(artist)
        root._rs.remove_artist(
            artist._h if artist.__dict__.get("_rs_native_in") else artist, sfid)

    a._remove_method = remove


def _rs_sync_subfigs(fig):
    """Mirror the zorder and visibility of the subfigures of *fig*."""
    root = fig.get_figure(root=True)
    for sf in fig.subfigs:
        sfid = sf.__dict__.get("_rs_sfid")
        if sfid is not None:
            root._rs.subfig_state(sfid, float(sf.zorder), bool(sf.get_visible()))
            _rs_sync_subfigs(sf)


def _patch_figure(FigureBase, Figure, SubFigure):
    orig_init = Figure.__init__

    @functools.wraps(orig_init)
    def __init__(self, *args, **kwargs):
        self._rs = _mplrs.Figure()
        orig_init(self, *args, **kwargs)
        dpi = self._dpi
        self.bbox_inches = _FlushBbox(self.bbox_inches.get_points(), self)
        self.dpi_scale_trans = _FlushAffine2D(self).scale(dpi)
        self.dpi_scale_trans._rs_spec = "dpi"
        self.bbox = mtransforms.TransformedBbox(self.bbox_inches, self.dpi_scale_trans)
        self.bbox._rs_bbox_spec = "figure"
        self.figbbox = self.bbox
        self.transFigure = _FlushBboxTransformTo(self.bbox, self)
        self.transSubfigure = self.transFigure
        self.patch.set_transform(self.transSubfigure)
        self.patch.__dict__["_h"] = mpatches._RsPatchHandle(self._rs)
        martist._DIRTY.add(self.patch)
        martist._DIRTY.add(self)

    Figure.__init__ = __init__

    def _rs_sync(self):
        """Mirror the figure geometry into the Rust figure."""
        w, h = self.bbox_inches.size
        bo = getattr(self.transFigure, "_boxout", None)
        boxout = None if (bo is None or bo is self.bbox) else tuple(map(float, bo.extents))
        self._rs.mirror_state((float(w), float(h)), float(self._dpi), boxout)

    Figure._rs_sync = _rs_sync

    def _rs_sync_all(self):
        martist._DIRTY.add(self)
        martist._sync_dirty()

    Figure._rs_sync_all = _rs_sync_all

    def _rs_all_axes(self):
        out = []

        def walk(axs):
            for ax in axs:
                out.append(ax)
                walk(ax.child_axes)

        walk(self.axes)
        return out

    def draw(self, renderer):
        # docstring inherited
        if not self.get_visible():
            return
        with self._render_lock:
            # applies the aspect of all axes (as matplotlib does)
            self._get_draw_artists(renderer)
            try:
                renderer.open_group('figure', gid=self.get_gid())
                if self.axes and self.get_layout_engine() is not None:
                    try:
                        self.get_layout_engine().execute(self)
                    except ValueError:
                        pass
                    # We don't want to raise an error here, but we also don't
                    # want to fail to draw the figure.
                for ax in _rs_all_axes(self):
                    ax._rs_prepare_draw(renderer)
                _rs_sync_subfigs(self)
                self._rs_sync_all()
                target = getattr(renderer, "_renderer", None)
                if (isinstance(target, _mplrs.RendererAgg)
                        and not getattr(renderer, "_rs_disabled", False)):
                    self._rs.draw_into(target)
                else:
                    self._rs.draw_disabled()
                renderer.close_group('figure')
            finally:
                self.stale = False

            DrawEvent("draw_event", self.canvas, renderer)._process()

    draw.__doc__ = Figure.draw.__doc__
    Figure.draw = martist.allow_rasterization(draw)

    # -- axes ------------------------------------------------------------------
    orig_add_axes_internal = FigureBase._add_axes_internal

    @functools.wraps(orig_add_axes_internal)
    def _add_axes_internal(self, ax, key):
        ret = orig_add_axes_internal(self, ax, key)
        rs = ax.__dict__.get("_rs")
        if rs is not None:
            root = self.get_figure(root=True)
            sfid = self.__dict__.get("_rs_sfid")
            if sfid is None:
                root._rs.attach_axes(rs.uid)
            else:
                root._rs.subfig_add_axes(sfid, rs.uid)
        return ret

    FigureBase._add_axes_internal = _add_axes_internal

    orig_remove_axes = FigureBase._remove_axes

    @functools.wraps(orig_remove_axes)
    def _remove_axes(self, ax, owners):
        ret = orig_remove_axes(self, ax, owners)
        rs = ax.__dict__.get("_rs")
        if rs is not None:
            self.get_figure(root=True)._rs.detach_axes(rs.uid)
        return ret

    FigureBase._remove_axes = _remove_axes

    # -- figure-level artists ----------------------------------------------------
    def _wrap(name, category):
        orig = getattr(FigureBase, name)

        @functools.wraps(orig)
        def method(self, *args, **kwargs):
            ret = orig(self, *args, **kwargs)
            _rs_fig_attach(self, ret, category)
            return ret

        setattr(FigureBase, name, method)

    _wrap("add_artist", 0)
    _wrap("text", 4)
    _wrap("legend", 6)

    orig_figimage = Figure.figimage

    @functools.wraps(orig_figimage)
    def figimage(self, *args, **kwargs):
        ret = orig_figimage(self, *args, **kwargs)
        _rs_fig_attach(self, ret, 5)
        return ret

    Figure.figimage = figimage

    # -- subfigures -------------------------------------------------------------
    orig_add_subfigure = FigureBase.add_subfigure

    @functools.wraps(orig_add_subfigure)
    def add_subfigure(self, subplotspec, **kwargs):
        sf = orig_add_subfigure(self, subplotspec, **kwargs)
        root = self.get_figure(root=True)
        sfid = root._rs.new_subfigure(self.__dict__.get("_rs_sfid"))
        sf._rs_sfid = sfid
        root._rs.subfig_set_patch(sfid, sf.patch._h)
        martist._DIRTY.add(sf.patch)
        owner_remove = sf._remove_method

        def remove(s):
            owner_remove(s)
            root._rs.remove_subfigure(sfid)

        sf._remove_method = remove
        return sf

    FigureBase.add_subfigure = add_subfigure

    orig_clear = Figure.clear

    @functools.wraps(orig_clear)
    def clear(self, keep_observers=False):
        ret = orig_clear(self, keep_observers=keep_observers)
        self._rs.clear()
        return ret

    Figure.clear = clear
