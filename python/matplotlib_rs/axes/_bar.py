"""Create the Rectangles of a bar call from validated style prototypes.

`Axes.bar` creates, styles and adds one Rectangle per bar; each goes through
the constructor, `add_patch` (artist properties, clip path, data limits from
the transformed path) and a full Rust synchronisation.  For thousands of
bars (``hist`` with many bins, long bar charts) this dominates the time.

Bars of the same style differ only by their bounds and sticky edge.  Here one
Rectangle per style is created by the ordinary constructor; its Rust handle is
copied once per bar with the bar's patch transform (`copy_many_rects`) and its
attribute dict is copied with the bar's ``_x0``, ``_y0``, ``_width``,
``_height`` and sticky edges.  The data limits receive the transformed
corners of all bars at once (min/max and minpos do not depend on the order).
The returned objects are ordinary, independently mutable Rectangles.
"""
import functools
import weakref

import numpy as np

from .. import _mplrs, artist, cbook, patches
from ..path import Path
from ._base_rs import _RsChildren, _rs_native, _rs_py_drawn, _rs_switch_child

# Below this many bars the ordinary loop is used.
MIN_BARS = 32

# Rectangle properties that bar may pass on unchanged to every bar.
_KWARGS = frozenset({
    "alpha", "antialiased", "capstyle", "fill", "gid", "hatch_linewidth",
    "hatchcolor", "in_layout", "joinstyle", "linestyle", "snap", "url",
    "visible", "zorder",
})


def _numeric(a):
    return isinstance(a, np.ndarray) and a.dtype.kind in "iuf"


def _same(values):
    """The common element of the 1D array *values*, or raise ValueError."""
    first = values[0]
    if values.dtype == object:
        if not all(v is first for v in values.tolist()):
            raise ValueError
    elif values.dtype.kind in "iuf":
        if not (values == first).all():
            raise ValueError
    else:
        raise ValueError
    return first


@cbook._gc_paused
def rectangles(ax, left, bottom, width, height, facecolor, edgecolor,
               linewidth, hatch, patch_labels, kwargs, orientation):
    """
    The Rectangles of ``ax.bar``, created and added to *ax* as its loop over
    ``zip(left, bottom, width, height, facecolor, edgecolor, linewidth,
    hatch, patch_labels)`` does, or None (with nothing done) when the
    ordinary loop must be used.

    *facecolor* is an (n, 4) RGBA array and *edgecolor* None or one, both
    cycled over the bars.
    """
    from ._axes import Axes
    n = len(left)
    if (n < MIN_BARS or type(ax) is not Axes
            or not isinstance(ax._children, _RsChildren) or _rs_py_drawn(ax)
            or any(axis.have_units() for axis in ax._axis_map.values())
            or not kwargs.keys() <= _KWARGS
            or not all(map(_numeric, (left, bottom, width, height)))
            or len(facecolor) == 0 or edgecolor is not None and len(edgecolor) == 0
            or not isinstance(patch_labels, list) or len(patch_labels) != n
            or patch_labels.count("_nolegend_") != n):
        return None
    for name in ("add_patch", "_update_patch_limits", "_set_artist_props",
                 "update_datalim"):
        if getattr(type(ax), name) is not getattr(Axes, name):
            return None
    try:
        lw = _same(linewidth)
        htch = _same(hatch)
    except ValueError:
        return None
    # The bounds as Rectangle._convert_units and Bbox.from_extents give them.
    x0 = left.astype(float)
    y0 = bottom.astype(float)
    x1 = (left + width).astype(float)
    y1 = (bottom + height).astype(float)
    if not (np.isfinite(x0).all() and np.isfinite(y0).all()
            and np.isfinite(x1).all() and np.isfinite(y1).all()):
        return None

    # One prototype per (facecolor, edgecolor) combination of the cycles.
    nf = len(facecolor)
    ne = 1 if edgecolor is None else len(edgecolor)
    index = np.arange(n)
    keys = (index % nf) * ne + index % ne
    styles = np.unique(keys)
    if len(styles) > max(16, n // 8):
        return None  # (almost) one style per bar: nothing to share

    switch = functools.partial(_rs_switch_child, weakref.ref(ax))
    remove = ax._children.remove
    prototypes = {}
    for key in styles.tolist():
        i = int(np.argmax(keys == key))
        r = patches.Rectangle(
            xy=(left[i], bottom[i]), width=width[i], height=height[i],
            facecolor=facecolor[i % nf],
            edgecolor=None if edgecolor is None else edgecolor[i % ne],
            linewidth=lw, label="_nolegend_", hatch=htch)
        r._internal_update(kwargs)
        ax._set_artist_props(r)
        if r.get_clip_path() is None:
            r.set_clip_path(ax.patch)
        r._set_in_autoscale(True)
        if (not _rs_native(r) or r.get_mouseover() or r.angle != 0
                or r.rotation_point != "xy" or r._aspect_ratio_correction != 1):
            return None
        artist._DIRTY.discard(r)
        r._h.set_many(r._rs_props())
        prototypes[key] = r
    Path.unit_rectangle()._interpolation_steps = 100

    vertical = orientation == "vertical"
    sticky_values = bottom if vertical else left
    sticky = sticky_values.astype(float)
    cols = {"_x0": list(left), "_y0": list(bottom), "_width": list(width),
            "_height": list(height)}
    XYPair = artist._XYPair
    if vertical:
        cols["_sticky_edges"] = [XYPair([], [v]) for v in sticky_values]
    else:
        cols["_sticky_edges"] = [XYPair([v], []) for v in sticky_values]
    names = list(cols)

    out = [None] * n
    for key, r in prototypes.items():
        idx = np.flatnonzero(keys == key) if len(prototypes) > 1 else index
        handles = r._h.copy_many_rects(x0[idx], y0[idx], x1[idx], y1[idx],
                                       1 if vertical else 0, sticky[idx])
        template = r.__dict__.copy()
        del template["_callbacks"]  # (each copy creates its own on first use)
        template["_remove_method"] = remove
        template["_rs_native_in"] = True
        template["_rs_switch"] = switch
        if len(prototypes) == 1:
            columns = [cols[name] for name in names]
        else:
            columns = [[cols[name][i] for i in idx.tolist()] for name in names]
        clones = _mplrs.clone_instances(patches.Rectangle, template, handles, names, columns)
        if len(prototypes) == 1:
            out = clones
        else:
            for i, a in zip(idx.tolist(), clones):
                out[i] = a

    list.extend(ax._children, out)
    ax._rs.add_children_raw([a._h for a in out])

    # Data limits: the corners of the patch transform [[x1 - x0, 0, x0],
    # [0, y1 - y0, y0]] applied to the unit rectangle, skipping bars with
    # neither width nor height (as _update_patch_limits does).
    keep = ~((width == 0) & (height == 0))
    if keep.any():
        xa, ya = x0[keep], y0[keep]
        xb = (x1[keep] - xa) + xa
        yb = (y1[keep] - ya) + ya
        ax.update_datalim(np.column_stack([np.concatenate([xa, xb, xb, xa]),
                                           np.concatenate([ya, ya, yb, yb])]))
    return out
