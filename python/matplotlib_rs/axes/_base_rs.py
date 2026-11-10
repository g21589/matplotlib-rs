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
from .. import cbook
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
    try:
        martist._rs_clip_path(a)
    except NotImplementedError:
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


class _Pending:
    """
    The artists of a plotting call (the components of a boxplot), created
    when they are first needed.

    The Rust axes has them as one child, an `_mplrs.ArtistBatch`, which draws
    them from arrays.  ``Axes._children`` has this object in their place until
    the list is read (`_RsChildren._expand`); the lists a plotting method
    returns (`_PendingList`) create the artists of their component only.

    An artist created here is an ordinary, independent artist with its own
    Rust handle.  While none of them is changed, the batch goes on drawing
    all the artists.  The first change (an artist marked stale, hence synced,
    or removed) replaces the batch by the handles of all its artists
    (`explode`), which are drawn one by one from then on.

    Artist *i* of slot *s* is child ``base + i * len(slots) + s`` of the Axes.
    A slot is a dict with the keys

    - ``prototype``: an artist with the class and the properties of the
      artists of the slot;
    - ``make``: ``make(handles, switch, dead)`` returns the *n* artists for
      their Rust handles; they call ``switch(artist)`` when they are synced
      (``_rs_switch``); *dead* tells that the Axes was cleared;
    - ``points``: the (N, 2) data of lines, for the data limits (None: the
      slot has no part in them).
    """

    def __init__(self, axes, batch, n, slots):
        self.axes = axes
        self._axref = weakref.ref(axes)
        self.batch = batch
        self.n = n
        self.slots = slots
        self.made = [None] * len(slots)
        self.exploded = False
        self.dead = False
        #: (axis name, lower edges, upper edges, slot): the sticky edges
        #: of the artists of a slot (see `set_sticky`)
        self.sticky = None
        #: (slot, labels): the labels of the first artists of a slot (see
        #: `set_labels`)
        self.labels = None

    def __len__(self):
        return self.n * len(self.slots)

    def explode(self):
        """Hand the Rust axes the handles of the artists instead of the batch."""
        if self.exploded:
            return
        self.exploded = True
        children = self.axes._children
        if isinstance(children, _RsChildren) and self in children._packed:
            children._packed.remove(self)
        if not self.dead:
            try:
                self.axes._rs.explode_batch(self.batch)
            except ValueError:  # (the Rust axes is gone)
                pass

    def _switch(self, a):
        # An artist is about to be synced: it changed.
        if not self.exploded:
            self.explode()
        _rs_switch_child(self._axref, a)

    @cbook._gc_paused  # (many artists: see cbook._gc_paused)
    def slot_artists(self, s):
        """The artists of slot *s* (created if needed)."""
        made = self.made[s]
        if made is None:
            made = self.made[s] = self.slots[s]["make"](
                self.batch.handles(s), self._switch, self.dead)
            sticky = self.sticky
            if sticky is not None and sticky[3] == s:
                name, lower, upper, _ = sticky
                self.sticky = None
                pair = martist._XYPair
                if name == "x":
                    for a, lo, hi in zip(made, lower, upper):
                        a._sticky_edges = pair([lo, hi], [])
                else:
                    for a, lo, hi in zip(made, lower, upper):
                        a._sticky_edges = pair([], [lo, hi])
            labels = self.labels
            if labels is not None and labels[0] == s:
                self.labels = None
                for a, label in zip(made, labels[1]):  # (as Artist.set_label)
                    a._label = label
                    a._h.set("label", "" if label is None else label)
        return made

    @cbook._gc_paused
    def artists(self):
        """All the artists, in the order of the children."""
        nslots = len(self.slots)
        out = [None] * len(self)
        for s in range(nslots):
            out[s::nslots] = self.slot_artists(s)
        return out

    def existing(self):
        """The artists created so far, in the order of the children."""
        made = [m for m in self.made if m is not None]
        if not made:
            return ()
        if len(made) == 1:
            return made[0]
        return [a for group in zip(*made) for a in group]

    def set_sticky(self, s, name, lower, upper):
        """Give artist *i* of slot *s* the sticky edges ``[lower[i], upper[i]]``
        on the *name* axis (and no others)."""
        if self.made[s] is None and self.sticky is None:
            self.sticky = (name, lower, upper, s)
            return
        for a, lo, hi in zip(self.slot_artists(s), lower, upper):
            getattr(a.sticky_edges, name).extend([lo, hi])

    def set_labels(self, s, labels):
        """``artist.set_label(label)`` for the first artists of slot *s*."""
        if self.made[s] is None and self.labels is None:
            self.labels = (s, [None if label is None else str(label) for label in labels])
            return
        for a, label in zip(self.slot_artists(s), labels):
            a.set_label(label)

    def pending_sticky(self, name):
        """The sticky edges on the *name* axis of the artists not created."""
        sticky = self.sticky
        if sticky is not None and sticky[0] == name:
            return [*sticky[1], *sticky[2]]
        return ()

    def limit_points(self, visible_only):
        """The points that count for the data limits when no artist has been
        created (and so none changed), or None: the artists tell."""
        if any(m is not None for m in self.made):
            return None
        out = []
        for slot in self.slots:
            points = slot["points"]
            if points is not None and (not visible_only or slot["prototype"].get_visible()):
                if not slot["finite"]:
                    return None
                out.append(points)
        return out

    def make_labelled(self):
        """Create the artists that a legend may show (those of the slots
        whose label does not start with an underscore, or with labels of
        their own)."""
        for s, slot in enumerate(self.slots):
            label = slot["prototype"].get_label()
            if self.made[s] is None and (
                    label and not label.startswith("_")
                    or self.labels is not None and self.labels[0] == s):
                self.slot_artists(s)

    def count(self, check):
        """The number of artists that satisfy *check* (a test of their type)."""
        return self.n * sum(bool(check(slot["prototype"])) for slot in self.slots)


@cbook._fills_when_read
class _PendingList(list):
    """
    A list of artists a plotting method returns (``boxplot(...)["boxes"]``)
    that are created when the list is first read: the artists of the slots
    *slots* of *pending*, taken in turn (slot 0 of box 0, slot 1 of box 0, ...).
    """

    __slots__ = ("_pending", "_pending_slots")

    def __init__(self, pending, slots):
        super().__init__()
        self._pending = pending
        self._pending_slots = slots

    def _fill(self):
        pending = self._pending
        if pending is not None:
            self._pending = None
            parts = [pending.slot_artists(s) for s in self._pending_slots]
            if len(parts) == 1:
                list.extend(self, parts[0])
            else:
                list.extend(self, [a for group in zip(*parts) for a in group])

    def _set_labels(self, labels):
        """`_Pending.set_labels` for the artists of this (one-slot) list;
        False if the artists were already created."""
        if self._pending is None or len(self._pending_slots) != 1:
            return False
        self._pending.set_labels(self._pending_slots[0], labels)
        return True

    def _set_sticky(self, name, lower, upper):
        """`_Pending.set_sticky` for the artists of this (one-slot) list;
        False if the artists were already created."""
        if self._pending is None or len(self._pending_slots) != 1:
            return False
        self._pending.set_sticky(self._pending_slots[0], name, lower, upper)
        return True

    def __len__(self):
        pending = self._pending
        if pending is not None:
            return pending.n * len(self._pending_slots)
        return list.__len__(self)

    def __reduce_ex__(self, protocol):
        self._fill()
        return (list, (list(self),))


class _RsChildren(list):
    """``Axes._children``, keeping the children of the Rust axes in step.

    matplotlib adds children by appending to this list (and removes them
    through ``_remove_method = _children.remove``).  ``clear`` does not
    notify: matplotlib only clears a list it has already swapped out, after
    the Rust axes has been cleared.

    The list may hold `_Pending` objects in place of artists not created yet
    (``_extra`` counts the artists they stand for, less the objects
    themselves).  Reading the list creates the artists; ``len``, ``append``
    and the methods below do not.  ``_packed`` are the `_Pending` whose
    artists the Rust axes draws as one batch.
    """

    __slots__ = ("_axref", "_extra", "_holders", "_packed")

    def __init__(self, ax, items=()):
        super().__init__()
        self._axref = weakref.ref(ax)
        self._extra = 0
        self._holders = 0
        self._packed = []
        self.extend(items)

    # -- artists created when they are needed --------------------------------

    def _add_pending(self, pending):
        """Add the artists of *pending* (a `_Pending`), after the children."""
        list.append(self, pending)
        self._extra += len(pending) - 1
        self._holders += 1
        self._packed.append(pending)
        self._axref()._rs.add_child_raw(pending.batch)

    def _expand(self):
        """Create the artists that are not, in their places."""
        if self._holders:
            items = []
            for a in list.__iter__(self):
                if type(a) is _Pending:
                    items.extend(a.artists())
                else:
                    items.append(a)
            list.__setitem__(self, slice(None), items)
            self._extra = self._holders = 0

    def _unpack(self):
        """Create the artists and give their handles to the Rust axes: a
        child is about to be removed or moved."""
        self._expand()
        for pending in list(self._packed):
            pending.explode()

    def _detach_pending(self):
        """The Axes is being cleared: artists created later have no Axes."""
        for pending in self._packed:
            pending.exploded = True
        self._packed = []
        if self._holders:
            items = []
            for a in list.__iter__(self):
                if type(a) is _Pending:
                    a.dead = a.exploded = True
                    items.extend(a.existing())
                else:
                    items.append(a)
            list.__setitem__(self, slice(None), items)
            self._extra = self._holders = 0

    def _existing(self):
        """The children that exist as Python objects, in order."""
        if not self._holders:
            return list.__iter__(self)
        items = []
        for a in list.__iter__(self):
            if type(a) is _Pending:
                items.extend(a.existing())
            else:
                items.append(a)
        return items

    def _pending(self):
        """The `_Pending` whose artists are not all in the list."""
        if not self._holders:
            return ()
        return [a for a in list.__iter__(self) if type(a) is _Pending]

    def _count(self, check):
        """``sum(check(a) for a in self)`` for a *check* of the artist types."""
        return sum(a.count(check) if type(a) is _Pending else bool(check(a))
                   for a in list.__iter__(self))

    def __len__(self):
        return list.__len__(self) + self._extra

    def __reduce_ex__(self, protocol):
        self._expand()
        return super().__reduce_ex__(protocol)

    # -- the list ------------------------------------------------------------

    def append(self, a):
        super().append(a)
        ax = self._axref()
        if ax is not None:
            _rs_attach(ax, a)

    def extend(self, items):
        for a in items:
            self.append(a)

    def insert(self, i, a):
        self._unpack()
        super().insert(i, a)
        ax = self._axref()
        if ax is not None:
            _rs_attach(ax, a)

    def remove(self, a):
        self._unpack()
        super().remove(a)
        _rs_detach(self._axref(), a)

    def pop(self, i=-1):
        self._unpack()
        a = super().pop(i)
        _rs_detach(self._axref(), a)
        return a

    def __delitem__(self, i):
        self._unpack()
        items = list.__getitem__(self, i) if isinstance(i, slice) else [list.__getitem__(self, i)]
        super().__delitem__(i)
        for a in items:
            _rs_detach(self._axref(), a)

    def clear(self):
        # (of a list that is not an Axes' any more: see the class docstring)
        super().clear()
        self._extra = self._holders = 0
        self._packed = []


def _expanded(name, unpack=False):
    method = getattr(list, name)

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if unpack:
            self._unpack()
        else:
            self._expand()
        return method(self, *args, **kwargs)

    return wrapper


for _name in ("__iter__", "__getitem__", "__contains__", "__reversed__", "__eq__", "__ne__",
              "__lt__", "__le__", "__gt__", "__ge__", "__add__", "__mul__", "__rmul__",
              "__repr__", "__sizeof__", "index", "count", "copy"):
    setattr(_RsChildren, _name, _expanded(_name))
for _name in ("__setitem__", "__iadd__", "__imul__", "sort", "reverse"):
    setattr(_RsChildren, _name, _expanded(_name, unpack=True))
del _name


def _radd(self, other):
    # (``[...] + children``, which list would do by reading the items directly)
    self._expand()
    return other + list.copy(self) if isinstance(other, list) else NotImplemented


_RsChildren.__radd__ = _radd


def _existing_children(ax):
    """The children of *ax* that exist as Python objects: artists not yet
    created (see `_Pending`) are left out."""
    children = ax._children
    return children._existing() if isinstance(children, _RsChildren) else children


def _pending_children(ax):
    """The `_Pending` of *ax* whose artists are not all created."""
    children = ax._children
    return children._pending() if isinstance(children, _RsChildren) else ()


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
        children = self.__dict__.get("_children")
        if isinstance(children, _RsChildren):
            children._detach_pending()
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
            None if self._autotitlepos is None else bool(self._autotitlepos),
            None if self._rasterization_zorder is None else float(self._rasterization_zorder))

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
            self.get_figure(root=True)._rs_sync_all(renderer)
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
