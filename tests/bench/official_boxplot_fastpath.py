"""Experimental Python/NumPy boxplot optimization for Matplotlib 3.8.2/3.11.2.

Only the importing process is patched. Rendering, PNG encoding and returned
Artist classes remain those of official Matplotlib; matplotlib_rs is not used.
"""
from contextlib import contextmanager
import difflib
import inspect
from numbers import Real
from pathlib import Path as FilePath

import numpy as np
import matplotlib as mpl
from matplotlib import cbook, lines, patches
from matplotlib.axes import Axes
from matplotlib.path import Path
from matplotlib.transforms import Bbox, TransformedBbox

ORIGINAL_BXP = Axes.bxp
ORIGINAL_STATS = cbook.boxplot_stats
ORIGINAL_PLOT = Axes.plot
ORIGINAL_ADD_ARTIST = Axes.add_artist
ORIGINAL_ADD_LINE = Axes.add_line
COUNTS = {"fast_groups": 0, "fallback_groups": 0, "dense_stats": 0}
SUPPORTED_VERSIONS = ("3.8.2", "3.11.2")


class LazyCallbacks(cbook.CallbackRegistry):
    """Independent callback maps, allocated on first use as in the RS patch."""

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


def dense_stats(X, whis, labels):
    columns = np.ascontiguousarray(X.T)
    ncols, nsamples = columns.shape
    if labels is not None and len(labels) != ncols:
        if mpl.__version__ == "3.8.2":
            raise ValueError("Dimensions of labels and X must be compatible")
        raise ValueError(f"The number of labels ({len(labels)}) must match the"
                         f" number of columns ({ncols}).")
    q1, med, q3 = np.percentile(columns, [25, 50, 75], axis=1)
    means = np.mean(columns, axis=1)
    iqr = q3 - q1
    delta = 1.57 * iqr / np.sqrt(nsamples)
    cilo, cihi = med - delta, med + delta
    loval, hival = q1 - whis * iqr, q3 + whis * iqr
    lo = np.min(columns, axis=1, where=columns >= loval[:, None], initial=np.inf)
    hi = np.max(columns, axis=1, where=columns <= hival[:, None], initial=-np.inf)
    lo = np.where((lo == np.inf) | (lo > q1), q1, lo)
    hi = np.where((hi == -np.inf) | (hi < q3), q3, hi)
    result = []
    for i, x in enumerate(columns):
        stats = {}
        if labels is not None and labels[i] is not None:
            stats["label"] = labels[i]
        stats.update(mean=means[i], iqr=iqr[i], cilo=cilo[i], cihi=cihi[i],
                     whishi=hi[i], whislo=lo[i],
                     fliers=np.concatenate([x[x < lo[i]], x[x > hi[i]]]),
                     q1=q1[i], med=med[i], q3=q3[i])
        result.append(stats)
    return result


def boxplot_stats(X, whis=1.5, bootstrap=None, labels=None, autorange=False):
    if (type(X) is np.ndarray and X.ndim == 2 and X.dtype == np.float64
            and X.shape[0] and X.shape[1] > 1 and bootstrap is None
            and not autorange and isinstance(whis, Real) and np.isfinite(X).all()):
        COUNTS["dense_stats"] += 1
        return dense_stats(X, whis, labels)
    return ORIGINAL_STATS(X, whis=whis, bootstrap=bootstrap, labels=labels,
                          autorange=autorange)


class BoxplotArtists:
    def __init__(self, axes, line_styles, patch_style):
        self.axes = axes
        self.styles = {}
        self.points = []
        for kw in line_styles:
            key = tuple(kw.items())
            hash(key)
            if key not in self.styles:
                prototype = lines.Line2D([], [], **kw)
                if prototype.get_drawstyle() != "default":
                    raise TypeError("nondefault drawstyle uses ordinary creation")
                self._prepare(prototype)
                self.styles[key] = prototype
        self.patch_style = None
        if patch_style is not None:
            prototype = patches.PathPatch(Path(np.empty((0, 2))), **patch_style)
            self._prepare(prototype)
            self.patch_style = prototype

    def _prepare(self, prototype):
        self.axes._set_artist_props(prototype)
        if prototype.get_clip_path() is None:
            prototype.set_clip_path(self.axes.patch)
        if prototype.get_mouseover():
            raise TypeError("mouseover uses ordinary creation")

    @staticmethod
    def create(axes, stats, line_styles, patch_style):
        styles = list(line_styles) + ([patch_style] if patch_style is not None else [])
        eligible = (
            len(stats) > 0 and type(axes) is Axes
            and isinstance(axes.patch, patches.Rectangle)
            and getattr(axes.plot, "__func__", None) is ORIGINAL_PLOT
            and getattr(axes.add_artist, "__func__", None) is ORIGINAL_ADD_ARTIST
            and getattr(axes.add_line, "__func__", None) is ORIGINAL_ADD_LINE
            and not any(axis.have_units() for axis in axes._axis_map.values())
            and not any(key in kw for kw in styles
                        for key in ("transform", "clip_path", "clip_box")))
        if eligible:
            # 3.8 keeps cycle state on _get_lines; 3.11 uses _prop_cycle.
            cycle = (axes._get_lines if mpl.__version__ == "3.8.2"
                     else axes._get_lines._prop_cycle)
            cycle_keys = cycle._cycler_items[0]
            eligible = not any(kw.get(key) is None for kw in line_styles for key in cycle_keys)
        if eligible:
            for stat in stats:
                if (any(not isinstance(stat[k], Real)
                        for k in ("q1", "q3", "med", "whislo", "whishi"))
                        or any(not isinstance(stat[k], Real)
                               for k in ("mean", "cilo", "cihi") if k in stat)
                        or "fliers" in stat and np.asarray(stat["fliers"]).dtype.kind not in "biuf"):
                    eligible = False
                    break
        if eligible:
            try:
                factory = BoxplotArtists(axes, line_styles, patch_style)
            except (TypeError, NotImplementedError):
                pass
            else:
                COUNTS["fast_groups"] += 1
                return factory
        COUNTS["fallback_groups"] += 1
        return None

    def _copy(self, prototype):
        result = type(prototype).__new__(type(prototype))
        result.__dict__ = prototype.__dict__.copy()
        result._callbacks = LazyCallbacks()
        # 3.8 exposes this pair directly; 3.11 initializes it lazily from None.
        result._sticky_edges = (type(prototype.sticky_edges)([], [])
                                if mpl.__version__ == "3.8.2" else None)
        # Keep independent mutable clip bounds and their normal invalidation.
        if prototype.clipbox is not None:
            result.clipbox = TransformedBbox(Bbox.unit(), prototype.clipbox._transform)
        result._remove_method = self.axes._children.remove
        self.axes._children.append(result)
        return result

    def line(self, xs, ys, kw):
        prototype = self.styles[tuple(kw.items())]
        line = self._copy(prototype)
        x, y = np.array(xs, copy=True), np.array(ys, copy=True)
        line._xorig, line._yorig = x, y
        # Materialize geometry now, just as add_line/get_path normally does.
        # The experiment must not move path construction into first_draw.
        xy = np.empty((len(x), 2), dtype=float)
        xy[:, 0], xy[:, 1] = x, y
        line._xy = xy
        line._x, line._y = xy[:, 0], xy[:, 1]
        line._path = Path(xy)
        line._invalidx = line._invalidy = False
        line._in_autoscale = True
        if len(x) > line._subslice_optim_min_size:
            line.recache(always=True)
        if not line._label:
            line._label = f"_child{len(self.axes._children) - 1}"
        self.points.extend(zip(x, y))
        return line

    def patch(self, xs, ys):
        patch = self._copy(self.patch_style)
        patch._path = Path._create_closed(np.column_stack([xs, ys]))
        return patch

    def finish(self):
        if self.points:
            self.axes.update_datalim(self.points)
        self.axes._request_autoscale_view()
        self.axes.stale = True


def patched_bxp_source():
    original = inspect.getsource(ORIGINAL_BXP)
    anchor = "        for pos, width, stats, capwidth in zip(positions, widths, bxpstats,\n"
    insertion = '''        # Experimental validated-prototype path, using official Artist classes.
        line_styles = [whisker_kw, median_kw]
        if showcaps:
            line_styles.append(cap_kw)
        if showfliers:
            line_styles.append(flier_kw)
        if showmeans:
            line_styles.append(mean_kw)
        if showbox and not patch_artist:
            line_styles.append(box_kw)
LABEL_DEFAULTS
        fast = _official_boxplot_factory(
            self, bxpstats, line_styles, box_kw if showbox and patch_artist else None)
        if fast is not None:
            def do_plot(xs, ys, **kwargs):
                x, y = [xs, ys][maybe_swap]
                return fast.line(x, y, kwargs)

            def do_patch(xs, ys, **kwargs):
                x, y = [xs, ys][maybe_swap]
                return fast.patch(x, y)

'''
    # Legend-label defaults were added after 3.8. Preserve each version's
    # original labels, including _childN numbering on 3.8.
    label_defaults = "" if mpl.__version__ == "3.8.2" else '''        for kw in (whisker_kw, cap_kw, flier_kw):
            kw.setdefault('label', '_nolegend_')
        if showbox:
            median_kw.setdefault('label', '_nolegend_')'''
    insertion = insertion.replace("LABEL_DEFAULTS", label_defaults)
    finish_anchor = ("        if manage_ticks:\n" if mpl.__version__ == "3.8.2"
                     else "        # Set legend labels\n")
    if original.count(anchor) != 1 or original.count(finish_anchor) != 1:
        raise RuntimeError("Official bxp source does not match the supported version")
    updated = original.replace(anchor, insertion + anchor)
    updated = updated.replace(finish_anchor,
                              "        if fast is not None:\n            fast.finish()\n\n" + finish_anchor)
    return original, updated


def install(*, artists=True, statistics=True):
    if mpl.__version__ not in SUPPORTED_VERSIONS:
        raise RuntimeError("This experimental patch requires official Matplotlib "
                           + " or ".join(SUPPORTED_VERSIONS))
    COUNTS.update(fast_groups=0, fallback_groups=0, dense_stats=0)
    if artists:
        _, source = patched_bxp_source()
        namespace = dict(inspect.unwrap(ORIGINAL_BXP).__globals__)
        namespace["_official_boxplot_factory"] = BoxplotArtists.create
        # Some upstream docstring lines have less indentation than the method.
        source = "".join(line[4:] if line.startswith("    ") else line
                         for line in source.splitlines(keepends=True))
        exec(compile(source, "<official-boxplot-fastpath>", "exec"), namespace)
    # Repeated install calls must not retain components from an earlier mode.
    Axes.bxp = namespace["bxp"] if artists else ORIGINAL_BXP
    cbook.boxplot_stats = boxplot_stats if statistics else ORIGINAL_STATS


def uninstall():
    Axes.bxp = ORIGINAL_BXP
    cbook.boxplot_stats = ORIGINAL_STATS


@contextmanager
def enabled(**kwargs):
    install(**kwargs)
    try:
        yield
    finally:
        uninstall()


def export_patch(destination):
    """Save the exact bxp edit for inspection; the helper module is also needed."""
    before, after = patched_bxp_source()
    FilePath(destination).write_text("".join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True),
        fromfile=f"official-{mpl.__version__}/Axes.bxp", tofile="optimized/Axes.bxp")))
