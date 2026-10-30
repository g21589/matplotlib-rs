"""Create independent boxplot artists from validated style prototypes.

bxp knows every component's numeric coordinates and reuses a handful of styles.
It need not repeat plot's format/unit parsing, clipping setup and path limits
for each of its thousands of two-point lines. The returned objects remain
ordinary, independently mutable Line2D and PathPatch instances.

When the statistics are plain floats, `_BoxplotArtists.build` creates all the
artists of a bxp call at once: the coordinates are computed with NumPy (the
same operations as bxp's loop, element by element), the Rust handles are
copied in one call per component, and the children and data limits are
updated once.
"""
import functools
import weakref

import numpy as np

from .. import _mplrs, artist, cbook, lines, patches
from ..path import Path
from ._base_rs import _RsChildren, _rs_native, _rs_py_drawn, _rs_switch_child


_PLAIN_FLOATS = (float, np.float64)
_PLAIN_NUMBERS = (float, int, np.float64, np.int64)

_gc_paused = cbook._gc_paused


class _BoxplotArtists:
    def __init__(self, axes, line_styles, patch_style):
        self.axes = axes
        self.styles = {}
        self.points = []
        self.point_arrays = []
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
    @_gc_paused
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
            for k in ("q1", "q3", "med", "whislo", "whishi"):
                v = stat[k]
                if type(v) not in _PLAIN_FLOATS and not isinstance(v, Real):
                    return None
            for k in ("mean", "cilo", "cihi"):
                v = stat.get(k, 0.)
                if type(v) not in _PLAIN_FLOATS and not isinstance(v, Real):
                    return None
            if "fliers" in stat:
                f = stat["fliers"]
                if (f.dtype if type(f) is np.ndarray else np.asarray(f).dtype).kind not in "biuf":
                    return None
        try:
            return _BoxplotArtists(axes, line_styles, patch_style)
        except (TypeError, NotImplementedError):
            return None

    def _copy(self, prototype, handle):
        a = type(prototype).__new__(type(prototype))
        a.__dict__ = prototype.__dict__.copy()
        a._h = handle
        # An independent callback registry is created on first use
        # (artist._LazyCallbacks).
        del a._callbacks
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

    # -- all artists of a bxp call at once ---------------------------------

    def _template(self, prototype, line):
        """The attribute dict shared (by copy) by the copies of *prototype*."""
        d = prototype.__dict__.copy()
        del d["_callbacks"]  # (each copy creates its own on first use)
        d["_remove_method"] = self.axes._children.remove
        d["_rs_native_in"] = True
        d["_rs_switch"] = self.switch
        if line:
            d["_in_autoscale"] = True
        return d

    def _lines(self, kw, xs, ys, slot, nslots, base):
        """Line2D copies of the style *kw* with the rows of *xs*, *ys* as data
        (the original arrays, whose dtype is that of bxp's lists); the copy
        for box *i* will be child ``base + i * nslots + slot``."""
        prototype = self.styles[tuple(kw.items())]
        n = len(xs)
        labels = None
        if not prototype._label:
            labels = [f"_child{base + i * nslots + slot}" for i in range(n)]
        if isinstance(xs, list):  # ragged (fliers): one 1D array per box
            offsets = np.concatenate([[0], np.cumsum([len(x) for x in xs])]).tolist()
            handles = prototype._h.copy_many_ragged(
                np.concatenate(xs).astype(np.float64, copy=False),
                np.concatenate(ys).astype(np.float64, copy=False), offsets)
            if labels is not None:
                for h, label in zip(handles, labels):
                    h.set("label", label)
            self.point_arrays.append(np.column_stack([np.concatenate(xs), np.concatenate(ys)]))
        else:
            handles = prototype._h.copy_many(
                np.ascontiguousarray(xs, np.float64), np.ascontiguousarray(ys, np.float64),
                labels)
            self.point_arrays.append(np.column_stack([xs.ravel(), ys.ravel()]))
        names, columns = ["_xorig", "_yorig"], [list(xs), list(ys)]
        if labels is not None:
            names.append("_label")
            columns.append(labels)
        return self._clones(prototype, True, handles, names, columns)

    def _clones(self, prototype, line, handles, names, columns):
        """Copies of *prototype* with the Rust *handles* and, per copy, the
        attributes *names* from *columns*."""
        cls = type(prototype)
        template = self._template(prototype, line)
        if cls.__new__ is object.__new__:
            return _mplrs.clone_instances(cls, template, handles, names, columns)
        out = []
        for i, h in enumerate(handles):
            a = cls.__new__(cls)
            d = template.copy()
            d["_h"] = h
            d.update((name, column[i]) for name, column in zip(names, columns))
            a.__dict__ = d
            out.append(a)
        return out

    def _patches(self, verts):
        """PathPatch copies with the closed paths through each (m, 2) block
        of *verts*, as ``Path._create_closed`` makes them."""
        prototype = self.patch_style
        verts = np.concatenate([verts, verts[:, :1]], axis=1)
        first = Path._create_closed(verts[0, :-1])
        codes = first.codes
        handles = prototype._h.copy_many_paths(verts, codes.tolist())
        fast_path = Path._fast_from_codes_and_verts
        paths = [first] + [fast_path(v, codes.copy(), first) for v in verts[1:]]
        return self._clones(prototype, False, handles, ["_path"], [paths])

    @_gc_paused
    def build(self, bxpstats, positions, widths, capwidths, vertical, *,
              showbox, patch_artist, shownotches, showcaps, showmeans, meanline,
              showfliers, box_kw, whisker_kw, cap_kw, median_kw, mean_kw, flier_kw):
        """
        Create the artists of all boxes of a bxp call, in bxp's order.

        Returns ``(whiskers, caps, boxes, medians, means, fliers)``, or None
        (before creating anything) if the statistics are not plain floats,
        so that bxp's per-box loop computes them with its own types.
        """
        if positions.dtype not in (np.int64, np.float64):
            return None
        if not all(type(w) in _PLAIN_NUMBERS for w in widths):
            return None
        if not all(type(w) in _PLAIN_NUMBERS for w in capwidths):
            return None
        keys = ["q1", "q3", "med", "whislo", "whishi"]
        if shownotches:
            keys += ["cilo", "cihi"]
        if showmeans:
            keys.append("mean")
        for stat in bxpstats:
            for k in keys:
                if type(stat[k]) not in _PLAIN_FLOATS:
                    return None
            if showfliers:
                f = stat["fliers"]
                if type(f) is not np.ndarray or f.dtype != np.float64 or f.ndim != 1:
                    return None

        def column(key):
            return np.array([stat[key] for stat in bxpstats], np.float64)

        def pairs(a, b):
            return np.stack([a, b], axis=1)

        def oriented(xs, ys):
            return (xs, ys) if vertical else (ys, xs)

        pos = positions
        width = np.array(widths, np.float64)
        capwidth = np.array(capwidths, np.float64)
        q1, q3, med = column("q1"), column("q3"), column("med")
        whislo, whishi = column("whislo"), column("whishi")
        box_left = pos - width * 0.5
        box_right = pos + width * 0.5
        components = []  # (name, kind, data) in bxp's creation order per box
        if shownotches:
            cilo, cihi = column("cilo"), column("cihi")
            notch_left = pos - width * 0.25
            notch_right = pos + width * 0.25
            box_x = np.stack([box_left, box_right, box_right, notch_right, box_right,
                              box_right, box_left, box_left, notch_left, box_left,
                              box_left], axis=1)
            box_y = np.stack([q1, q1, cilo, med, cihi, q3, q3, cihi, med, cilo, q1], axis=1)
            med_x = pairs(notch_left, notch_right)
        else:
            box_x = np.stack([box_left, box_right, box_right, box_left, box_left], axis=1)
            box_y = np.stack([q1, q1, q3, q3, q1], axis=1)
            med_x = pairs(box_left, box_right)
        if showbox:
            if patch_artist:
                components.append(("boxes", "patch", np.stack(oriented(box_x, box_y), axis=2)))
            else:
                components.append(("boxes", box_kw, oriented(box_x, box_y)))
        # (each component gets its own arrays: every artist's data is a row
        # of its own, as bxp's loop copies its lists for every artist)
        whis_x = pairs(pos, pos)
        components.append(("whiskers", whisker_kw, oriented(whis_x, pairs(q1, whislo))))
        components.append(("whiskers", whisker_kw, oriented(whis_x.copy(), pairs(q3, whishi))))
        if showcaps:
            cap_x = pairs(pos - capwidth * 0.5, pos + capwidth * 0.5)
            components.append(("caps", cap_kw, oriented(cap_x, pairs(whislo, whislo))))
            components.append(("caps", cap_kw, oriented(cap_x.copy(), pairs(whishi, whishi))))
        components.append(("medians", median_kw, oriented(med_x, pairs(med, med))))
        if showmeans:
            mean = column("mean")
            if meanline:
                data = oriented(pairs(box_left, box_right), pairs(mean, mean))
            else:
                data = oriented(pos[:, None].copy(), mean[:, None])
            components.append(("means", mean_kw, data))
        if showfliers:
            fl = [stat["fliers"] for stat in bxpstats]
            lengths = [len(f) for f in fl]
            flier_x = np.repeat(pos.astype(np.float64), lengths)
            flier_y = np.concatenate(fl) if fl else np.empty(0)
            cuts = np.cumsum(lengths)[:-1]
            components.append(("fliers", flier_kw, oriented(np.split(flier_x, cuts),
                                                            np.split(flier_y, cuts))))

        nslots = len(components)
        base = len(self.axes._children)
        n = len(bxpstats)
        ordered = [None] * (n * nslots)
        result = {name: [] for name in
                  ("whiskers", "caps", "boxes", "medians", "means", "fliers")}
        for slot, (name, kw, data) in enumerate(components):
            if kw == "patch":
                artists = self._patches(data)
            else:
                artists = self._lines(kw, data[0], data[1], slot, nslots, base)
            ordered[slot::nslots] = artists
            result[name].append(artists)
        list.extend(self.axes._children, ordered)
        self.axes._rs.add_children_raw([a._h for a in ordered])
        out = []
        for name in ("whiskers", "caps", "boxes", "medians", "means", "fliers"):
            parts = result[name]
            if len(parts) == 2:  # lower and upper, alternating per box
                merged = [None] * (2 * n)
                merged[0::2], merged[1::2] = parts
                out.append(merged)
            else:
                out.append(parts[0] if parts else [])
        return tuple(out)

    def finish(self):
        if self.points:
            self.axes.update_datalim(self.points)
        if self.point_arrays:
            self.axes.update_datalim(np.concatenate(self.point_arrays))
        self.axes._request_autoscale_view()
        self.axes.stale = True
