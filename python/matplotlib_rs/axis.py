"""
`matplotlib.axis` for matplotlib_rs.

Limits, scales, units, locators and formatters live in Python exactly as in
matplotlib (much of this module is matplotlib's own code).  The ticks
themselves (tick lines, grid lines, tick labels) and the axis label and
offset text are owned by the Rust core, which updates and draws them; the
objects returned by ``get_ticklabels()`` & co. are Python views of them.
Locators and formatters of matplotlib's standard classes are mirrored by
native Rust implementations; any other is called back from Rust.

Parts of this file are from matplotlib 3.11 (see licenses/LICENSE_MATPLOTLIB).
"""
import datetime
import functools
import logging
from numbers import Real
import warnings

import numpy as np

import matplotlib_rs as mpl
from . import _api, cbook
from . import artist as martist
from . import colors as mcolors
from . import lines as mlines
from . import scale as mscale
from . import text as mtext
from . import ticker as mticker
from . import transforms as mtransforms
from . import units as munits
from .ticker import NullLocator

_log = logging.getLogger(__name__)

GRIDLINE_INTERPOLATION_STEPS = 180

_line_inspector = martist.ArtistInspector(mlines.Line2D)
_line_param_names = _line_inspector.get_setters()
_line_param_aliases = [next(iter(d)) for d in _line_inspector.aliasd.values()]
_gridline_param_names = ['grid_' + name
                         for name in _line_param_names + _line_param_aliases]

# --------------------------------------------------------------------------
# Native tick helpers


def _locator_spec(loc):
    """Rust spec for a locator, or None if it has to run in Python."""
    t = type(loc)
    if t is mticker.AutoLocator:
        if (loc._nbins == "auto" and list(loc._steps) == [1, 2, 2.5, 5, 10]
                and not loc._integer and not loc._symmetric and loc._prune is None
                and loc._min_n_ticks == 2):
            return ("auto",)
    if t in (mticker.MaxNLocator, mticker.AutoLocator):
        return ("maxn", loc._nbins if loc._nbins == "auto" else int(loc._nbins),
                [float(s) for s in loc._steps], bool(loc._integer), bool(loc._symmetric),
                loc._prune, int(loc._min_n_ticks))
    if t is mticker.NullLocator:
        return ("null",)
    if t is mticker.FixedLocator:
        return ("fixed", [float(v) for v in np.asarray(loc.locs, float).ravel()], loc.nbins)
    if t is mticker.MultipleLocator:
        return ("multiple", float(loc._edge.step), float(loc._offset))
    if t is mticker.LinearLocator and not loc.presets:
        return ("linear", int(loc.numticks))
    if t is mticker.IndexLocator:
        return ("index", float(loc._base), float(loc.offset))
    if t is mticker.AutoMinorLocator:
        n = loc.ndivs
        return ("autominor", None if n is None else ("auto" if n == "auto" else int(n)))
    if t is mticker.LogLocator:
        subs = loc._subs
        if isinstance(subs, str):
            s = subs
        else:
            s = [float(v) for v in np.asarray(subs).ravel()]
        return ("log", float(loc._base), s,
                None if loc.numticks == "auto" else int(loc.numticks))
    return None


def _formatter_spec(fmt):
    """Rust spec for a formatter, or None if it has to run in Python."""
    t = type(fmt)
    if t is mticker.ScalarFormatter:
        if isinstance(fmt._useOffset, bool) and not (fmt._useOffset is False and fmt.offset):
            uo = bool(fmt._useOffset)
        else:
            uo = float(fmt.offset)
        pl = fmt._powerlimits
        return ("scalar", uo, bool(fmt._useMathText), bool(fmt._useLocale),
                bool(fmt._usetex), bool(fmt._scientific), (int(pl[0]), int(pl[1])))
    if t is mticker.NullFormatter:
        return ("null",)
    if t is mticker.FixedFormatter:
        return ("fixed", [str(s) for s in fmt.seq], str(fmt.offset_string))
    kinds = {mticker.LogFormatterSciNotation: "logsci",
             mticker.LogFormatterMathtext: "logmath",
             mticker.LogFormatter: "log",
             mticker.LogFormatterExponent: "logexp"}
    if t in kinds and fmt._linthresh is None:
        return (kinds[t], float(fmt._base), bool(fmt.labelOnlyBase),
                tuple(map(float, fmt.minor_thresholds)), None)
    return None


class _PyLocatorAdapter:
    """Lets the Rust core call a Python locator."""

    def __init__(self, loc):
        self._loc = loc
        self.rs_is_null = isinstance(loc, mticker.NullLocator)

    def rs_call(self, info):
        return [float(v) for v in np.asarray(self._loc(), float).ravel()]

    def rs_tick_values(self, vmin, vmax, info):
        return [float(v) for v in np.asarray(self._loc.tick_values(vmin, vmax), float).ravel()]

    def rs_nonsingular(self, v0, v1):
        a, b = self._loc.nonsingular(v0, v1)
        return float(a), float(b)

    def rs_view_limits(self, v0, v1):
        a, b = self._loc.view_limits(v0, v1)
        return float(a), float(b)


class _PyFormatterAdapter:
    """Lets the Rust core call a Python formatter."""

    def __init__(self, fmt):
        self._fmt = fmt

    def rs_format_ticks(self, values, info):
        return [str(s) for s in self._fmt.format_ticks(np.asarray(values, float))]

    def rs_get_offset(self):
        return str(self._fmt.get_offset())


# --------------------------------------------------------------------------
# Ticks


class Tick:
    """
    One tick of an `Axis`: a view of the tick owned by the Rust core.

    ``tick1line``, ``tick2line``, ``gridline``, ``label1`` and ``label2``
    are views of the Rust artists, so changes to them are drawn.
    """

    def __init__(self, axis, idx, major, line_handles, label1, label2):
        self._axis_proxy = axis
        self._idx = idx
        self._major = major
        # Rust handles of tick1line, tick2line and gridline; their Python
        # views are made when first used
        self._rs_line_handles = line_handles
        self._rs_line_views = {}
        self.label1 = label1
        self.label2 = label2
        self.axes = axis.axes

    def _rs_line(self, i):
        v = self._rs_line_views.get(i)
        if v is not None and v._h is self._rs_line_handles[i]:
            v._rs_refresh()
            return v
        v = _wrap_rs_line(self._axis_proxy._line_cache, self._rs_line_handles[i], self.axes)
        self._rs_line_views[i] = v
        return v

    def _rs_line_property(i):
        def fget(self):
            return self._rs_line(i)

        def fset(self, line):
            self._rs_line_views[i] = line

        return property(fget, fset)

    tick1line = _rs_line_property(0)
    tick2line = _rs_line_property(1)
    gridline = _rs_line_property(2)
    del _rs_line_property

    def __repr__(self):
        return f"<{type(self).__name__} {self._idx} of {self._axis_proxy}>"

    def get_loc(self):
        loc = self.__dict__.get("_loc")
        if loc is None:
            locs = self._axis_proxy.get_ticklocs(minor=not self._major)
            loc = locs[self._idx] if self._idx < len(locs) else np.nan
        return loc

    @property
    def _size(self):
        return self._axis_proxy._get_tick_kw(self._major).get(
            "size", mpl.rcParams[f"{self._axis_proxy.axis_name}tick.{self._which()}.size"])

    @property
    def _tickdir(self):
        return self.get_tickdir()

    @property
    def label(self):
        return self.label1

    def get_children(self):
        return [self.tick1line, self.tick2line, self.gridline, self.label1, self.label2]

    def _which(self):
        return "major" if self._major else "minor"

    def set_label1(self, s):
        self.label1.set_text(s)

    set_label = set_label1

    def set_label2(self, s):
        self.label2.set_text(s)

    def set_pad(self, val):
        self._axis_proxy.set_tick_params(which=self._which(), pad=val)

    def get_pad(self):
        return self._axis_proxy._get_tick_kw(self._major).get("pad", mpl.rcParams[
            f"{self._axis_proxy.axis_name}tick.{self._which()}.pad"])

    def get_tickdir(self):
        return self._axis_proxy._get_tick_kw(self._major).get(
            "tickdir", mpl.rcParams[f"{self._axis_proxy.axis_name}tick.direction"])

    def get_tick_padding(self):
        size = self._axis_proxy._get_tick_kw(self._major).get(
            "size", mpl.rcParams[f"{self._axis_proxy.axis_name}tick.{self._which()}.size"])
        return size * {"in": 0.0, "inout": 0.5, "out": 1.0}[self.get_tickdir()]

    def set_visible(self, b):
        for a in self.get_children():
            a.set_visible(b)

    def get_visible(self):
        return True

    def _apply_params(self, **kwargs):
        self._axis_proxy.set_tick_params(which=self._which(), **kwargs)

    def update_position(self, loc):
        self._loc = loc

    def set_clip_path(self, path, transform=None):
        self.gridline.set_clip_path(path, transform)

    def get_view_interval(self):
        return self._axis_proxy.get_view_interval()


class XTick(Tick):
    __name__ = 'xtick'


class YTick(Tick):
    __name__ = 'ytick'


class Ticker:
    """
    A container for the objects defining tick position and format.
    """

    def __init__(self):
        self._locator = None
        self._formatter = None
        self._locator_is_default = True
        self._formatter_is_default = True

    @property
    def locator(self):
        return self._locator

    @locator.setter
    def locator(self, locator):
        if not isinstance(locator, mticker.Locator):
            raise TypeError('locator must be a subclass of '
                            'matplotlib.ticker.Locator')
        self._locator = locator

    @property
    def formatter(self):
        return self._formatter

    @formatter.setter
    def formatter(self, formatter):
        if not isinstance(formatter, mticker.Formatter):
            raise TypeError('formatter must be a subclass of '
                            'matplotlib.ticker.Formatter')
        self._formatter = formatter


# --------------------------------------------------------------------------
# Axis


def _wrap_rs_text(cache, h, owner):
    """A Python Text view of an existing Rust text handle."""
    key = h.ident()
    t = cache.get(key)
    if t is None:
        t = mtext._RsText._from_rs(h, owner)
        cache[key] = t
    else:
        t._rs_refresh()
    return t


def _wrap_rs_line(cache, h, owner):
    key = h.ident()
    ln = cache.get(key)
    if ln is None:
        ln = mlines._RsLine2D._from_rs(h, owner)
        cache[key] = ln
    else:
        ln._rs_refresh()
    return ln


def _rs_prop(name, axis_key, conv=lambda v: v, default=None):
    """A plain attribute that is also pushed to the Rust axis as `axis_key`."""
    attr = "_rs_" + name

    def getter(self):
        return self.__dict__.get(attr, default)

    def setter(self, v):
        self.__dict__[attr] = v
        if self.__dict__.get("_axes") is not None:
            self.axes._rs.axis_set(self.axis_name, axis_key, conv(v))

    return property(getter, setter)


# Axis classes of other implementations (the Python one used by polar and
# geo axes, see _axis_py.py); matplotlib treats them like Axis.
_rs_other_axis_classes = []


class Axis(martist.Artist):
    """
    Base class for `.XAxis` and `.YAxis`.
    """
    OFFSETTEXTPAD = 3
    _tick_class = None
    converter = _api.deprecate_privatize_attribute(
                    "3.10",
                    alternative="get_converter and set_converter methods"
                )

    labelpad = _rs_prop("labelpad", "labelpad", float)
    label_position = _rs_prop("label_position", "label_position", str)
    _tick_position = _rs_prop("tick_position", "tick_position", str, "bottom")
    _autolabelpos = _rs_prop("autolabelpos", "autolabelpos", bool)

    def __str__(self):
        return "{}({},{})".format(
            type(self).__name__, *self.axes.transAxes.transform((0, 0)))

    def __init__(self, axes, *, pickradius=15, clear=True):
        super().__init__()
        self._remove_overlapping_locs = True
        self.set_figure(axes.get_figure(root=False))
        self.isDefault_label = True
        self.axes = axes
        self.major = Ticker()
        self.minor = Ticker()
        self.callbacks = cbook.CallbackRegistry(signals=["units"])
        self._autolabelpos = True
        self._text_cache = {}
        self._line_cache = {}
        self._tick_cache = {}
        self._pushed = {}
        self.label = _wrap_rs_text(self._text_cache, axes._rs.axis_label(self.axis_name), axes)
        self.offsetText = _wrap_rs_text(self._text_cache, axes._rs.offset_text(self.axis_name), axes)
        self.labelpad = mpl.rcParams['axes.labelpad']
        self.pickradius = pickradius
        self._major_tick_kw = dict()
        self._minor_tick_kw = dict()
        self._scale = mscale.LinearScale(self)
        if clear:
            self.clear()
        else:
            self._converter = None
            self._converter_is_explicit = False
            self.units = None
        self._autoscale_on = True

    # -- Rust helpers ----------------------------------------------------
    @property
    def _rs(self):
        return self.axes._rs

    def _get_tick_kw(self, major):
        return self._major_tick_kw if major else self._minor_tick_kw

    def _rs_sync_tickers(self):
        """Push the current locators and formatters if they changed."""
        rs = self.axes._rs
        for which, tk in (("major", self.major), ("minor", self.minor)):
            minor = which == "minor"
            loc = tk.locator
            if loc is not None:
                spec = _locator_spec(loc)
                key = (which, "loc")
                if spec is None:
                    old = self._pushed.get(key)
                    if not (isinstance(old, tuple) and old[0] == "py" and old[1] is loc):
                        rs.set_locator(self.axis_name, ("py", _PyLocatorAdapter(loc)), minor)
                        self._pushed[key] = ("py", loc)
                elif self._pushed.get(key) != spec:
                    rs.set_locator(self.axis_name, spec, minor)
                    self._pushed[key] = spec
            fmt = tk.formatter
            if fmt is not None:
                spec = _formatter_spec(fmt)
                key = (which, "fmt")
                if spec is None:
                    old = self._pushed.get(key)
                    if not (isinstance(old, tuple) and old[0] == "py" and old[1] is fmt):
                        rs.set_formatter(self.axis_name, ("py", _PyFormatterAdapter(fmt)), minor)
                        self._pushed[key] = ("py", fmt)
                elif self._pushed.get(key) != spec:
                    rs.set_formatter(self.axis_name, spec, minor)
                    self._pushed[key] = spec

    def _rs_push_scale(self):
        s = self._scale
        name = s.name
        kw = {}
        if name == "log":
            kw = dict(base=float(s.base),
                      subs=None if s.subs is None else [float(v) for v in s.subs],
                      clip=bool(getattr(s._transform, "_clip", True)))
        elif name == "symlog":
            tr = s._transform
            kw = dict(base=float(tr.base), linthresh=float(tr.linthresh),
                      linscale=float(tr.linscale),
                      subs=None if s.subs is None else [float(v) for v in s.subs])
        elif name == "asinh":
            kw = dict(linear_width=float(s.linear_width))
        if name not in ("linear", "log", "symlog", "logit", "asinh"):
            # any other scale: the Rust axis evaluates its transform in Python
            from . import _rs_register_scale_transform
            kw = dict(custom_id=_rs_register_scale_transform(s.get_transform()))
            name = "custom"
        self.axes._rs.set_scale(self.axis_name, name, **kw)
        # the Rust axis resets its tick helpers with the scale
        self._pushed = {}

    def _rs_prepare(self):
        """Bring the Rust axis up to date before reading its ticks."""
        self.axes._unstale_viewLim()
        self._rs_sync_tickers()
        martist._sync_dirty()

    # -- properties ------------------------------------------------------
    @property
    def isDefault_majloc(self):
        return self.major._locator_is_default

    @isDefault_majloc.setter
    def isDefault_majloc(self, value):
        self.major._locator_is_default = value

    @property
    def isDefault_majfmt(self):
        return self.major._formatter_is_default

    @isDefault_majfmt.setter
    def isDefault_majfmt(self, value):
        self.major._formatter_is_default = value

    @property
    def isDefault_minloc(self):
        return self.minor._locator_is_default

    @isDefault_minloc.setter
    def isDefault_minloc(self, value):
        self.minor._locator_is_default = value

    @property
    def isDefault_minfmt(self):
        return self.minor._formatter_is_default

    @isDefault_minfmt.setter
    def isDefault_minfmt(self, value):
        self.minor._formatter_is_default = value

    def _get_shared_axes(self):
        """Return a list of shared Axes for current axis."""
        return self.axes._shared_axes[
            self._get_axis_name()].get_siblings(self.axes)

    def _get_shared_axis(self):
        """Return list of shared axis for current axis."""
        name = self._get_axis_name()
        return [ax._axis_map[name] for ax in self._get_shared_axes()]

    def _get_axis_name(self):
        """Return the axis name."""
        return self.axis_name

    def get_remove_overlapping_locs(self):
        return self._remove_overlapping_locs

    def set_remove_overlapping_locs(self, val):
        self._remove_overlapping_locs = bool(val)
        self.axes._rs.axis_set(self.axis_name, "remove_overlapping_locs", bool(val))

    remove_overlapping_locs = property(
        get_remove_overlapping_locs, set_remove_overlapping_locs,
        doc=('If minor ticker locations that overlap with major '
             'ticker locations should be trimmed.'))

    @property
    def pickradius(self):
        return self._pickradius

    @pickradius.setter
    def pickradius(self, pickradius):
        if not isinstance(pickradius, Real) or pickradius < 0:
            raise ValueError("pick radius should be a distance")
        self._pickradius = pickradius

    def get_pickradius(self):
        return self._pickradius

    def set_pickradius(self, pickradius):
        self.pickradius = pickradius

    def set_label_coords(self, x, y, transform=None):
        self._autolabelpos = False
        if transform is None:
            transform = self.axes.transAxes
        self.label.set_transform(transform)
        self.label.set_position((x, y))
        self.stale = True

    # -- visibility & zorder are also Rust state --------------------------
    def set_visible(self, b):
        super().set_visible(b)
        self.axes._rs.axis_set(self.axis_name, "visible", bool(b))

    def set_zorder(self, level):
        if level is None:
            level = self.__class__.zorder
        self.zorder = level
        self.axes._rs.axis_set(self.axis_name, "zorder", float(level))
        self.pchanged()
        self.stale = True

    # -- scale (matplotlib's code + Rust mirror) ---------------------------
    def get_transform(self):
        """Return the transform used in the Axis' scale"""
        return self._scale.get_transform()

    def get_scale(self):
        """Return this Axis' scale (as a str)."""
        return self._scale.name

    def _set_scale(self, value, **kwargs):
        if not isinstance(value, mscale.ScaleBase):
            self._scale = mscale.scale_factory(value, self, **kwargs)
        else:
            self._scale = value
        self._rs_push_scale()
        self._scale.set_default_locators_and_formatters(self)

        self.isDefault_majloc = True
        self.isDefault_minloc = True
        self.isDefault_majfmt = True
        self.isDefault_minfmt = True

    def _set_axes_scale(self, value, **kwargs):
        name = self._get_axis_name()
        old_default_lims = (self.get_major_locator()
                            .nonsingular(-np.inf, np.inf))
        for ax in self._get_shared_axes():
            ax._axis_map[name]._set_scale(value, **kwargs)
            ax._update_transScale()
            ax.stale = True
        new_default_lims = (self.get_major_locator()
                            .nonsingular(-np.inf, np.inf))
        if old_default_lims != new_default_lims:
            self.axes.autoscale_view(
                **{f"scale{k}": k == name for k in self.axes._axis_names})

    def limit_range_for_scale(self, vmin, vmax):
        return self._scale.limit_range_for_scale(vmin, vmax, self.get_minpos())

    def _nan_out_of_scale_range(self, data):
        data = np.asanyarray(data, dtype=float)
        valid = self._scale.val_in_range(data)
        return data if np.all(valid) else np.where(valid, data, np.nan)

    def _get_autoscale_on(self):
        """Return whether this Axis is autoscaled."""
        return self._autoscale_on

    def _set_autoscale_on(self, b):
        if b is not None:
            self._autoscale_on = b

    def get_children(self):
        return [self.label, self.offsetText,
                *self.get_major_ticks(), *self.get_minor_ticks()]

    def _reset_major_tick_kw(self):
        self._major_tick_kw.clear()
        self._major_tick_kw['gridOn'] = (
                mpl.rcParams['axes.grid'] and
                mpl.rcParams['axes.grid.which'] in ('both', 'major'))

    def _reset_minor_tick_kw(self):
        self._minor_tick_kw.clear()
        self._minor_tick_kw['gridOn'] = (
                mpl.rcParams['axes.grid'] and
                mpl.rcParams['axes.grid.which'] in ('both', 'minor'))

    def clear(self):
        """
        Clear the axis (the Rust axis is reset with its Axes).
        """
        # Rust-owned tick handles are recreated by the reset
        label, offset = self.label, self.offsetText
        self._text_cache = {label._h.ident(): label, offset._h.ident(): offset}
        self._line_cache = {}
        self._tick_cache = {}
        label._rs_refresh()
        offset._rs_refresh()
        self.labelpad = mpl.rcParams['axes.labelpad']
        self._init()
        self._set_scale('linear')
        self.callbacks = cbook.CallbackRegistry(signals=["units"])
        self._major_tick_kw['gridOn'] = (
                mpl.rcParams['axes.grid'] and
                mpl.rcParams['axes.grid.which'] in ('both', 'major'))
        self._minor_tick_kw['gridOn'] = (
                mpl.rcParams['axes.grid'] and
                mpl.rcParams['axes.grid.which'] in ('both', 'minor'))
        self.reset_ticks()
        self._converter = None
        self._converter_is_explicit = False
        self.units = None
        self.stale = True

    def reset_ticks(self):
        self.axes._rs.reset_ticks(self.axis_name)
        self._line_cache = {}
        self._tick_cache = {}

    def minorticks_on(self):
        scale = self.get_scale()
        if scale == 'log':
            s = self._scale
            self.set_minor_locator(mticker.LogLocator(s.base, s.subs))
        elif scale == 'symlog':
            s = self._scale
            self.set_minor_locator(
                mticker.SymmetricalLogLocator(s._transform, s.subs))
        elif scale == 'asinh':
            s = self._scale
            self.set_minor_locator(
                    mticker.AsinhLocator(s.linear_width, base=s._base,
                                         subs=s._subs))
        elif scale == 'logit':
            self.set_minor_locator(mticker.LogitLocator(minor=True))
        else:
            self.set_minor_locator(mticker.AutoMinorLocator())

    def minorticks_off(self):
        self.set_minor_locator(mticker.NullLocator())

    # -- tick parameters -------------------------------------------------
    def set_tick_params(self, which='major', reset=False, **kwargs):
        _api.check_in_list(['major', 'minor', 'both'], which=which)
        kwtrans = self._translate_tick_params(kwargs)
        if reset:
            if which in ['major', 'both']:
                self._reset_major_tick_kw()
                self._major_tick_kw.update(kwtrans)
            if which in ['minor', 'both']:
                self._reset_minor_tick_kw()
                self._minor_tick_kw.update(kwtrans)
        else:
            if which in ['major', 'both']:
                self._major_tick_kw.update(kwtrans)
            if which in ['minor', 'both']:
                self._minor_tick_kw.update(kwtrans)
        martist._sync_dirty()
        self.axes._rs.tick_params(self.axis_name, which, bool(reset), _rs_tick_kw(kwtrans))
        for t in list(self._text_cache.values()):
            t._rs_refresh()
        for ln in list(self._line_cache.values()):
            ln._rs_refresh()
        self.stale = True

    def get_tick_params(self, which='major'):
        _api.check_in_list(['major', 'minor'], which=which)
        if which == 'major':
            return self._translate_tick_params(
                self._major_tick_kw, reverse=True
            )
        return self._translate_tick_params(self._minor_tick_kw, reverse=True)

    @classmethod
    def _translate_tick_params(cls, kw, reverse=False):
        kw_ = {**kw}

        # The following lists may be moved to a more accessible location.
        allowed_keys = [
            'size', 'width', 'color', 'tickdir', 'pad',
            'labelsize', 'labelcolor', 'labelfontfamily', 'zorder', 'gridOn',
            'tick1On', 'tick2On', 'label1On', 'label2On',
            'length', 'direction', 'left', 'bottom', 'right', 'top',
            'labelleft', 'labelbottom', 'labelright', 'labeltop',
            'labelrotation', 'labelrotation_mode',
            *_gridline_param_names]

        keymap = {
            # tick_params key -> axis key
            'length': 'size',
            'direction': 'tickdir',
            'rotation': 'labelrotation',
            'rotation_mode': 'labelrotation_mode',
            'left': 'tick1On',
            'bottom': 'tick1On',
            'right': 'tick2On',
            'top': 'tick2On',
            'labelleft': 'label1On',
            'labelbottom': 'label1On',
            'labelright': 'label2On',
            'labeltop': 'label2On',
        }
        if reverse:
            kwtrans = {}
            is_x_axis = cls.axis_name == 'x'
            y_axis_keys = ['left', 'right', 'labelleft', 'labelright']
            for oldkey, newkey in keymap.items():
                if newkey in kw_:
                    if is_x_axis and oldkey in y_axis_keys:
                        continue
                    else:
                        kwtrans[oldkey] = kw_.pop(newkey)
        else:
            kwtrans = {
                newkey: kw_.pop(oldkey)
                for oldkey, newkey in keymap.items() if oldkey in kw_
            }
        if 'colors' in kw_:
            c = kw_.pop('colors')
            kwtrans['color'] = c
            kwtrans['labelcolor'] = c
        # Maybe move the checking up to the caller of this method.
        for key in kw_:
            if key not in allowed_keys:
                raise ValueError(
                    "keyword %s is not recognized; valid keywords are %s"
                    % (key, allowed_keys))
        kwtrans.update(kw_)
        return kwtrans

    def set_clip_path(self, path, transform=None):
        super().set_clip_path(path, transform)
        self.stale = True

    # -- intervals (Python state, mirrored to Rust at draw time) ------------
    def get_view_interval(self):
        """Return the ``(min, max)`` view limits of this axis."""
        raise NotImplementedError('Derived must override')

    def set_view_interval(self, vmin, vmax, ignore=False):
        raise NotImplementedError('Derived must override')

    def get_data_interval(self):
        """Return the ``(min, max)`` data limits of this axis."""
        raise NotImplementedError('Derived must override')

    def set_data_interval(self, vmin, vmax, ignore=False):
        raise NotImplementedError('Derived must override')

    def get_inverted(self):
        low, high = self.get_view_interval()
        return high < low

    def set_inverted(self, inverted):
        a, b = self.get_view_interval()
        # cast to bool to avoid bad interaction between python 3.8 and np.bool_
        self._set_lim(*sorted((a, b), reverse=bool(inverted)), auto=None)

    def set_default_intervals(self):
        """
        Set the default limits for the axis data and view interval if they
        have not been not mutated yet.
        """

    def _set_lim(self, v0, v1, *, emit=True, auto):
        name = self._get_axis_name()

        self.axes._process_unit_info([(name, (v0, v1))], convert=False)
        v0 = self.axes._validate_converted_limits(v0, self.convert_units)
        v1 = self.axes._validate_converted_limits(v1, self.convert_units)

        if v0 is None or v1 is None:
            # Axes init calls set_xlim(0, 1) before get_xlim() can be called,
            # so only grab the limits if we really need them.
            old0, old1 = self.get_view_interval()
            if v0 is None:
                v0 = old0
            if v1 is None:
                v1 = old1

        if self.get_scale() == 'log' and (v0 <= 0 or v1 <= 0):
            # Axes init calls set_xlim(0, 1) before get_xlim() can be called,
            # so only grab the limits if we really need them.
            old0, old1 = self.get_view_interval()
            if v0 <= 0:
                _api.warn_external(f"Attempt to set non-positive {name}lim on "
                                   f"a log-scaled axis will be ignored.")
                v0 = old0
            if v1 <= 0:
                _api.warn_external(f"Attempt to set non-positive {name}lim on "
                                   f"a log-scaled axis will be ignored.")
                v1 = old1
        if v0 == v1:
            _api.warn_external(
                f"Attempting to set identical low and high {name}lims "
                f"makes transformation singular; automatically expanding.")
        reverse = bool(v0 > v1)  # explicit cast needed for python3.8+np.bool_.
        v0, v1 = self.get_major_locator().nonsingular(v0, v1)
        v0, v1 = self.limit_range_for_scale(v0, v1)
        v0, v1 = sorted([v0, v1], reverse=bool(reverse))

        self.set_view_interval(v0, v1, ignore=True)
        # Mark viewlims as no longer stale without triggering an autoscale.
        for ax in self._get_shared_axes():
            ax._stale_viewlims[name] = False
        self._set_autoscale_on(auto)

        if emit:
            self.axes.callbacks.process(f"{name}lim_changed", self.axes)
            # Call all of the other Axes that are shared with this one
            for other in self._get_shared_axes():
                if other is self.axes:
                    continue
                other._axis_map[name]._set_lim(v0, v1, emit=False, auto=auto)
                if emit:
                    other.callbacks.process(f"{name}lim_changed", other)
                if ((other_fig := other.get_figure(root=False)) !=
                        self.get_figure(root=False)):
                    other_fig.canvas.draw_idle()

        self.stale = True
        return v0, v1

    def _set_artist_props(self, a):
        if a is None:
            return
        a.set_figure(self.get_figure(root=False))

    def _get_ticklabel_bboxes(self, ticks, renderer):
        """Return lists of bboxes for ticks' label1's and label2's."""
        return ([tick.label1.get_window_extent(renderer)
                 for tick in ticks
                 if tick.label1.get_visible() and tick.label1.get_in_layout()],
                [tick.label2.get_window_extent(renderer)
                 for tick in ticks
                 if tick.label2.get_visible() and tick.label2.get_in_layout()])

    def get_tightbbox(self, renderer=None, *, for_layout_only=False):
        """
        Return a bounding box that encloses the axis. It only accounts
        tick labels, axis label, and offsetText.
        """
        if not self.get_visible() or for_layout_only and not self.get_in_layout():
            return
        if renderer is None:
            renderer = self.get_figure(root=True)._get_renderer()
        ticks_to_draw = self._update_ticks()

        self._update_label_position(renderer)

        # go back to just this axis's tick labels
        tlb1, tlb2 = self._get_ticklabel_bboxes(ticks_to_draw, renderer)

        self._update_offset_text_position(tlb1, tlb2)
        self.offsetText.set_text(self.major.formatter.get_offset())

        bboxes = [
            *(a.get_window_extent(renderer)
              for a in [self.offsetText]
              if a.get_visible()),
            *tlb1, *tlb2,
        ]
        # take care of label
        if self.label.get_visible():
            bb = self.label.get_window_extent(renderer)
            # for constrained/tight_layout, we want to ignore the label's
            # width/height because the adjustments they make can't be improved.
            # this code collapses the relevant direction
            if for_layout_only:
                if self.axis_name == "x" and bb.width > 0:
                    bb.x0 = (bb.x0 + bb.x1) / 2 - 0.5
                    bb.x1 = bb.x0 + 1.0
                if self.axis_name == "y" and bb.height > 0:
                    bb.y0 = (bb.y0 + bb.y1) / 2 - 0.5
                    bb.y1 = bb.y0 + 1.0
            bboxes.append(bb)
        bboxes = [b for b in bboxes if b._is_finite()]
        if bboxes:
            return mtransforms.Bbox.union(bboxes)
        else:
            return None

    def get_tick_padding(self):
        values = []
        if len(self.get_majorticklocs()):
            values.append(self.get_major_ticks(1)[0].get_tick_padding())
        if len(self.get_minorticklocs()):
            values.append(self.get_minor_ticks(1)[0].get_tick_padding())
        return max(values, default=0)

    def draw(self, renderer):
        # drawn by the Rust core as part of its Axes
        self.stale = False

    def get_gridlines(self):
        """Return this Axis' grid lines as a list of `.Line2D`\\s."""
        self._rs_prepare()
        return cbook.silent_list('Line2D gridline', [
            _wrap_rs_line(self._line_cache, h, self.axes)
            for h in self.axes._rs.get_gridlines(self.axis_name, False)])

    def set_label(self, s):
        raise RuntimeError(
            "A legend label cannot be assigned to an Axis. Did you mean to "
            "set the axis label via set_label_text()?")

    def get_label(self):
        """Return the axis label as a Text instance."""
        return self.label

    def get_offset_text(self):
        """Return the axis offsetText as a Text instance."""
        return self.offsetText

    def get_pickradius_(self):
        return self._pickradius

    def _labels(self, which, visible_only=True):
        self._rs_prepare()
        return [_wrap_rs_text(self._text_cache, h, self.axes)
                for h in self.axes._rs.get_ticklabels(self.axis_name, which, visible_only)]

    def get_majorticklabels(self):
        return self._labels("major")

    def get_minorticklabels(self):
        return self._labels("minor")

    def get_ticklabels(self, minor=False, which=None):
        if which is not None:
            if which == 'minor':
                return self.get_minorticklabels()
            elif which == 'major':
                return self.get_majorticklabels()
            elif which == 'both':
                return self.get_majorticklabels() + self.get_minorticklabels()
            else:
                _api.check_in_list(['major', 'minor', 'both'], which=which)
        if minor:
            return self.get_minorticklabels()
        return self.get_majorticklabels()

    def get_majorticklines(self):
        self._rs_prepare()
        return cbook.silent_list('Line2D ticklines', [
            _wrap_rs_line(self._line_cache, h, self.axes)
            for h in self.axes._rs.get_ticklines(self.axis_name, False)])

    def get_minorticklines(self):
        self._rs_prepare()
        return cbook.silent_list('Line2D ticklines', [
            _wrap_rs_line(self._line_cache, h, self.axes)
            for h in self.axes._rs.get_ticklines(self.axis_name, True)])

    def get_ticklines(self, minor=False):
        if minor:
            return self.get_minorticklines()
        return self.get_majorticklines()

    def get_majorticklocs(self):
        """Return this Axis' major tick locations in data coordinates."""
        return self.get_major_locator()()

    def get_minorticklocs(self):
        """Return this Axis' minor tick locations in data coordinates."""
        # Remove minor ticks duplicating major ticks.
        minor_locs = np.asarray(self.get_minor_locator()())
        if self.remove_overlapping_locs:
            major_locs = self.get_majorticklocs()
            transform = self._scale.get_transform()
            tr_minor_locs = transform.transform(minor_locs)
            tr_major_locs = transform.transform(major_locs)
            lo, hi = sorted(transform.transform(self.get_view_interval()))
            # Use the transformed view limits as scale.  1e-5 is the default
            # rtol for np.isclose.
            tol = (hi - lo) * 1e-5
            mask = np.isclose(tr_minor_locs[:, None], tr_major_locs[None, :],
                              atol=tol, rtol=0).any(axis=1)
            minor_locs = minor_locs[~mask]
        return minor_locs

    def get_ticklocs(self, *, minor=False):
        return self.get_minorticklocs() if minor else self.get_majorticklocs()

    def get_ticks_direction(self, minor=False):
        kw = self._minor_tick_kw if minor else self._major_tick_kw
        d = kw.get("tickdir", mpl.rcParams[f"{self.axis_name}tick.direction"])
        n = len(self.get_ticklocs(minor=minor))
        return np.array([d] * n)

    def _make_ticks(self, major, n):
        self._rs_prepare()
        rs = self.axes._rs
        # reuse the tick views while the Rust ticks are unchanged
        sig = rs.ticks_signature(self.axis_name, major)
        made = self.__dict__.setdefault("_rs_made_ticks", {})
        prev = made.get(major)
        if prev is not None and prev[0] == sig:
            ticks = prev[1]
            return ticks[:n] if n is not None else ticks
        ticks = self._make_ticks_from_rs(major)
        made[major] = (sig, ticks)
        return ticks[:n] if n is not None else ticks

    def _make_ticks_from_rs(self, major):
        which = "major" if major else "minor"
        rs = self.axes._rs
        labels = [_wrap_rs_text(self._text_cache, h, self.axes)
                  for h in rs.get_ticklabels(self.axis_name, which, False)]
        lines = rs.get_ticklines(self.axis_name, not major)
        grids = rs.get_gridlines(self.axis_name, not major)
        nt = len(grids)
        cls = self._tick_class or Tick
        cache = self.__dict__.setdefault("_tick_cache", {})
        ticks = []
        for i in range(nt):
            key = (major, labels[i]._h.ident())
            t = cache.get(key)
            handles = (lines[2 * i], lines[2 * i + 1], grids[i])
            if (t is None or t.label1 is not labels[i]
                    or t._rs_line_handles[2].ident() != grids[i].ident()):
                t = cls(self, i, major, handles, labels[i], labels[nt + i])
            else:
                t._rs_line_handles = handles
                cache[key] = t
            t._idx = i
            ticks.append(t)
        return ticks

    def _update_ticks(self):
        """
        Update the ticks (position and labels) using the current data
        interval of the axes.  Return the list of ticks that will be drawn.
        """
        # The tick labels themselves are formatted by the Rust core; the
        # formatters are run for their state (e.g. the offset).
        major_locs = self.get_majorticklocs()
        self.major.formatter.format_ticks(major_locs)
        major_ticks = self.get_major_ticks(len(major_locs))
        for tick, loc in zip(major_ticks, major_locs):
            tick.update_position(loc)
        minor_locs = self.get_minorticklocs()
        self.minor.formatter.format_ticks(minor_locs)
        minor_ticks = self.get_minor_ticks(len(minor_locs))
        for tick, loc in zip(minor_ticks, minor_locs):
            tick.update_position(loc)
        ticks = [*major_ticks, *minor_ticks]

        view_low, view_high = self.get_view_interval()
        if view_low > view_high:
            view_low, view_high = view_high, view_low

        interval_t = self.get_transform().transform([view_low, view_high])

        ticks_to_draw = []
        for tick in ticks:
            try:
                loc_t = self.get_transform().transform(tick.get_loc())
            except AssertionError:
                # Transform.transform doesn't allow masked values but
                # some scales might make them, so we need this try/except.
                pass
            else:
                if mtransforms._interval_contains_close(interval_t, loc_t):
                    ticks_to_draw.append(tick)

        return ticks_to_draw

    @property
    def majorTicks(self):
        return self._make_ticks(True, None)

    @property
    def minorTicks(self):
        return self._make_ticks(False, None)

    def get_major_ticks(self, numticks=None):
        return self._make_ticks(True, numticks)

    def get_minor_ticks(self, numticks=None):
        return self._make_ticks(False, numticks)

    def grid(self, visible=None, which='major', **kwargs):
        if kwargs:
            if visible is None:
                visible = True
            elif not visible:  # something false-like but not None
                _api.warn_external('First parameter to grid() is false, '
                                   'but line properties are supplied. The '
                                   'grid will be enabled.')
                visible = True
        which = which.lower()
        _api.check_in_list(['major', 'minor', 'both'], which=which)
        gridkw = {f'grid_{name}': value for name, value in kwargs.items()}
        if which in ['minor', 'both']:
            gridkw['gridOn'] = (not self._minor_tick_kw['gridOn']
                                if visible is None else visible)
            self.set_tick_params(which='minor', **gridkw)
        if which in ['major', 'both']:
            gridkw['gridOn'] = (not self._major_tick_kw['gridOn']
                                if visible is None else visible)
            self.set_tick_params(which='major', **gridkw)
        self.stale = True

    # -- units (matplotlib's code) -------------------------------------------
    def update_units(self, data):
        if not self._converter_is_explicit:
            converter = munits.registry.get_converter(data)
        else:
            converter = self._converter

        if converter is None:
            return False

        neednew = self._converter != converter
        self._set_converter(converter)
        default = self._converter.default_units(data, self)
        if default is not None and self.units is None:
            self.set_units(default)

        elif neednew:
            self._update_axisinfo()
        self.stale = True
        return True

    def _update_axisinfo(self):
        if self._converter is None:
            return

        info = self._converter.axisinfo(self.units, self)

        if info is None:
            return
        if info.majloc is not None and \
           self.major.locator != info.majloc and self.isDefault_majloc:
            self.set_major_locator(info.majloc)
            self.isDefault_majloc = True
        if info.minloc is not None and \
           self.minor.locator != info.minloc and self.isDefault_minloc:
            self.set_minor_locator(info.minloc)
            self.isDefault_minloc = True
        if info.majfmt is not None and \
           self.major.formatter != info.majfmt and self.isDefault_majfmt:
            self.set_major_formatter(info.majfmt)
            self.isDefault_majfmt = True
        if info.minfmt is not None and \
           self.minor.formatter != info.minfmt and self.isDefault_minfmt:
            self.set_minor_formatter(info.minfmt)
            self.isDefault_minfmt = True
        if info.label is not None and self.isDefault_label:
            self.set_label_text(info.label)
            self.isDefault_label = True

        self.set_default_intervals()

    def have_units(self):
        return self._converter is not None or self.units is not None

    def convert_units(self, x):
        # If x is natively supported by Matplotlib, doesn't need converting
        if munits._is_natively_supported(x):
            return x

        if self._converter is None:
            self._set_converter(munits.registry.get_converter(x))

        if self._converter is None:
            return x
        try:
            ret = self._converter.convert(x, self.units, self)
        except Exception as e:
            raise munits.ConversionError('Failed to convert value(s) to axis '
                                         f'units: {x!r}') from e
        return ret

    def get_converter(self):
        return self._converter

    def set_converter(self, converter):
        self._set_converter(converter)
        self._converter_is_explicit = True

    def _set_converter(self, converter):
        if self._converter is converter or self._converter == converter:
            return
        if self._converter_is_explicit:
            raise RuntimeError("Axis already has an explicit converter set")
        elif (
            self._converter is not None and
            not isinstance(converter, type(self._converter)) and
            not isinstance(self._converter, type(converter))
        ):
            _api.warn_external(
                "This axis already has a converter set and "
                "is updating to a potentially incompatible converter"
            )
        self._converter = converter

    def set_units(self, u):
        if u == self.units:
            return
        for axis in self._get_shared_axis():
            axis.units = u
            axis._update_axisinfo()
            axis.callbacks.process('units')
            axis.stale = True

    def get_units(self):
        return self.units

    # -- labels ----------------------------------------------------------
    def set_label_text(self, label, fontdict=None, **kwargs):
        self.isDefault_label = False
        self.label.set_text(label)
        if fontdict is not None:
            self.label.update(fontdict)
        self.label.update(kwargs)
        self.stale = True
        return self.label

    def get_label_text(self):
        return self.label.get_text()

    def get_label_position(self):
        return self.label_position

    def set_label_position(self, position):
        raise NotImplementedError()

    # -- locators / formatters --------------------------------------------
    def get_major_locator(self):
        return self.major.locator

    def get_minor_locator(self):
        return self.minor.locator

    def get_major_formatter(self):
        return self.major.formatter

    def get_minor_formatter(self):
        return self.minor.formatter

    def set_major_formatter(self, formatter):
        self._set_formatter(formatter, self.major)

    def set_minor_formatter(self, formatter):
        self._set_formatter(formatter, self.minor)

    def _set_formatter(self, formatter, level):
        if isinstance(formatter, str):
            formatter = mticker.StrMethodFormatter(formatter)
        # Don't allow any other TickHelper to avoid easy-to-make errors,
        # like using a Locator instead of a Formatter.
        elif (callable(formatter) and
              not isinstance(formatter, mticker.TickHelper)):
            formatter = mticker.FuncFormatter(formatter)
        else:
            _api.check_isinstance(mticker.Formatter, formatter=formatter)

        if (isinstance(formatter, mticker.FixedFormatter)
                and len(formatter.seq) > 0
                and not isinstance(level.locator, mticker.FixedLocator)):
            _api.warn_external('FixedFormatter should only be used together '
                               'with FixedLocator')

        if level == self.major:
            self.isDefault_majfmt = False
        else:
            self.isDefault_minfmt = False

        level.formatter = formatter
        formatter.set_axis(self)
        self.stale = True

    def set_major_locator(self, locator):
        _api.check_isinstance(mticker.Locator, locator=locator)
        self.isDefault_majloc = False
        self.major.locator = locator
        if self.major.formatter:
            self.major.formatter._set_locator(locator)
        locator.set_axis(self)
        self.stale = True

    def set_minor_locator(self, locator):
        _api.check_isinstance(mticker.Locator, locator=locator)
        self.isDefault_minloc = False
        self.minor.locator = locator
        if self.minor.formatter:
            self.minor.formatter._set_locator(locator)
        locator.set_axis(self)
        self.stale = True

    @staticmethod
    def _format_with_dict(tickd, x, pos):
        return tickd.get(x, "")

    def set_ticklabels(self, labels, *, minor=False, fontdict=None, **kwargs):
        try:
            labels = [t.get_text() if hasattr(t, 'get_text') else t
                      for t in labels]
        except TypeError:
            raise TypeError(f"{labels:=} must be a sequence") from None
        locator = (self.get_minor_locator() if minor
                   else self.get_major_locator())
        if not labels:
            # eg labels=[]:
            formatter = mticker.NullFormatter()
        elif isinstance(locator, mticker.FixedLocator):
            # Passing [] as a list of labels is often used as a way to
            # remove all tick labels, so only error for > 0 labels
            if len(locator.locs) != len(labels) and len(labels) != 0:
                raise ValueError(
                    "The number of FixedLocator locations"
                    f" ({len(locator.locs)}), usually from a call to"
                    " set_ticks, does not match"
                    f" the number of labels ({len(labels)}).")
            tickd = {loc: lab for loc, lab in zip(locator.locs, labels)}
            func = functools.partial(self._format_with_dict, tickd)
            formatter = mticker.FuncFormatter(func)
        else:
            _api.warn_external(
                "set_ticklabels() should only be used with a fixed number of "
                "ticks, i.e. after set_ticks() or using a FixedLocator. "
                "Otherwise, ticks may be mislabeled.")
            formatter = mticker.FixedFormatter(labels)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="FixedFormatter should only be used together with "
                        "FixedLocator")
            if minor:
                self.set_minor_formatter(formatter)
            else:
                self.set_major_formatter(formatter)

        # Note: The tick labels are updated by the Rust core at draw time;
        # setting their text here returns the labels as matplotlib does.
        if minor:
            locs = self.get_minorticklocs()
            ticks = self.get_minor_ticks(len(locs))
        else:
            locs = self.get_majorticklocs()
            ticks = self.get_major_ticks(len(locs))

        ret = []
        if fontdict is not None:
            kwargs.update(fontdict)
        if len(kwargs) or len(labels):
            for pos, (loc, tick) in enumerate(zip(locs, ticks)):
                tick.update_position(loc)
                tick_label = formatter(loc, pos)
                # deal with label1
                tick.label1.set_text(tick_label)
                tick.label1._internal_update(kwargs)
                # deal with label2
                tick.label2.set_text(tick_label)
                tick.label2._internal_update(kwargs)
                # only return visible tick labels
                if tick.label1.get_visible():
                    ret.append(tick.label1)
                if tick.label2.get_visible():
                    ret.append(tick.label2)

        self.stale = True
        return ret

    def _set_tick_locations(self, ticks, *, minor=False):
        # XXX if the user changes units, the information will be lost here
        ticks = self.convert_units(ticks)
        locator = mticker.FixedLocator(ticks)  # validate ticks early.
        if len(ticks):
            for axis in self._get_shared_axis():
                # set_view_interval maintains any preexisting inversion.
                axis.set_view_interval(min(ticks), max(ticks))
        self.axes.stale = True
        if minor:
            self.set_minor_locator(locator)
            return self.get_minor_ticks(len(ticks))
        else:
            self.set_major_locator(locator)
            return self.get_major_ticks(len(ticks))

    def set_ticks(self, ticks, labels=None, *, minor=False, **kwargs):
        if labels is None and kwargs:
            first_key = next(iter(kwargs))
            raise ValueError(
                f"Incorrect use of keyword argument {first_key!r}. Keyword arguments "
                "other than 'minor' modify the text labels and can only be used if "
                "'labels' are passed as well.")
        result = self._set_tick_locations(ticks, minor=minor)
        if labels is not None:
            self.set_ticklabels(labels, minor=minor, **kwargs)
        return result

    def _get_tick_boxes_siblings(self, renderer):
        name = self._get_axis_name()
        if name not in self.get_figure(root=False)._align_label_groups:
            return [], []
        grouper = self.get_figure(root=False)._align_label_groups[name]
        bboxes = []
        bboxes2 = []
        for ax in grouper.get_siblings(self.axes):
            axis = ax._axis_map[name]
            ticks_to_draw = axis._update_ticks()
            tlb, tlb2 = axis._get_ticklabel_bboxes(ticks_to_draw, renderer)
            bboxes.extend(tlb)
            bboxes2.extend(tlb2)
        return bboxes, bboxes2

    def axis_date(self, tz=None):
        if isinstance(tz, str):
            import dateutil.tz
            tz = dateutil.tz.gettz(tz)
        self.update_units(datetime.datetime(2009, 1, 1, 0, 0, 0, 0, tz))

    def get_tick_space(self):
        """Return the estimated number of ticks that can fit on the axis."""
        # Must be overridden in the subclass
        raise NotImplementedError()

    def _get_ticks_position(self):
        representative_ticks = []
        if not isinstance(self.get_major_locator(), NullLocator):
            representative_ticks.append(self.get_major_ticks(1)[0])
        if not isinstance(self.get_minor_locator(), NullLocator):
            mt = self.get_minor_ticks(1)
            if mt:
                representative_ticks.append(mt[0])

        if all(tick.tick1line.get_visible()
               and not tick.tick2line.get_visible()
               and tick.label1.get_visible()
               and not tick.label2.get_visible()
               for tick in representative_ticks):
            return 1
        elif all(tick.tick2line.get_visible()
                 and not tick.tick1line.get_visible()
                 and tick.label2.get_visible()
                 and not tick.label1.get_visible()
                 for tick in representative_ticks):
            return 2
        elif all(tick.tick1line.get_visible()
                 and tick.tick2line.get_visible()
                 and tick.label1.get_visible()
                 and not tick.label2.get_visible()
                 for tick in representative_ticks):
            return "default"
        else:
            return "unknown"

    def _get_tick_label_size(self, axis_name):
        tick_kw = self._major_tick_kw
        size = tick_kw.get('labelsize',
                           mpl.rcParams[f'{axis_name}tick.labelsize'])
        return mtext.FontProperties(size=size).get_size_in_points()

    def get_minpos(self):
        raise NotImplementedError()

    def contains(self, mouseevent):
        return False, {}


def _make_getset_interval(method_name, lim_name, attr_name):
    """
    Helper to generate ``get_{data,view}_interval`` and
    ``set_{data,view}_interval`` implementations.
    """

    def getter(self):
        # docstring inherited.
        return getattr(getattr(self.axes, lim_name), attr_name)

    def setter(self, vmin, vmax, ignore=False):
        # docstring inherited.
        if ignore:
            setattr(getattr(self.axes, lim_name), attr_name, (vmin, vmax))
        else:
            oldmin, oldmax = getter(self)
            if oldmin < oldmax:
                setter(self, min(vmin, vmax, oldmin), max(vmin, vmax, oldmax),
                       ignore=True)
            else:
                setter(self, max(vmin, vmax, oldmin), min(vmin, vmax, oldmax),
                       ignore=True)
        self.stale = True

    getter.__name__ = f"get_{method_name}_interval"
    setter.__name__ = f"set_{method_name}_interval"

    return getter, setter


def _rs_tick_kw(kw):
    """Resolve tick parameters to the values the Rust core expects."""
    out = {}
    for k, v in kw.items():
        if k in ("color", "labelcolor", "grid_color", "grid_c"):
            key = "grid_color" if k.startswith("grid") else k
            if isinstance(v, str) and v.lower() == "inherit":
                continue
            out[key] = mcolors.to_rgba(v)
        elif k in ("size", "width", "pad", "zorder", "grid_linewidth", "grid_lw",
                   "grid_zorder"):
            key = {"grid_lw": "grid_linewidth"}.get(k, k)
            out[key] = float(v)
        elif k == "labelsize":
            out[k] = float(mtext.FontProperties(size=v).get_size_in_points())
        elif k in ("gridOn", "tick1On", "tick2On", "label1On", "label2On", "grid_visible",
                   "grid_antialiased", "grid_aa"):
            key = {"grid_aa": "grid_antialiased"}.get(k, k)
            out[key] = bool(v)
        elif k == "grid_alpha":
            out[k] = None if v is None else float(v)
        elif k in ("grid_linestyle", "grid_ls"):
            ls = v
            if isinstance(ls, str):
                if ls in (" ", "", "none"):
                    ls = "None"
                out["grid_linestyle"] = ls
            else:
                from .lines import _get_dash_pattern
                out["grid_linestyle"] = _get_dash_pattern(ls)
        elif k == "tickdir":
            out[k] = str(v)
        elif k == "labelrotation":
            out[k] = v if not isinstance(v, str) else (v, 0.0)
        elif k in ("labelrotation_mode", "labelfontfamily"):
            out[k] = v
    return out


class XAxis(Axis):
    __name__ = 'xaxis'
    axis_name = 'x'  #: Read-only name identifying the axis.
    _tick_class = XTick

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init()

    def _init(self):
        # x in axes coords, y in display coords (updated at draw time by the
        # Rust core).
        self.label.set(
            x=0.5, y=0,
            verticalalignment='top', horizontalalignment='center',
            transform=mtransforms.blended_transform_factory(
                self.axes.transAxes, mtransforms.IdentityTransform()),
        )
        self.label_position = 'bottom'

        if mpl.rcParams['xtick.labelcolor'] == 'inherit':
            tick_color = mpl.rcParams['xtick.color']
        else:
            tick_color = mpl.rcParams['xtick.labelcolor']

        self.offsetText.set(
            x=1, y=0,
            verticalalignment='top', horizontalalignment='right',
            transform=mtransforms.blended_transform_factory(
                self.axes.transAxes, mtransforms.IdentityTransform()),
            fontsize=mpl.rcParams['xtick.labelsize'],
            color=tick_color
        )
        self.offset_text_position = 'bottom'

    def contains(self, mouseevent):
        """Test whether the mouse event occurred in the x-axis."""
        if self._different_canvas(mouseevent):
            return False, {}
        x, y = mouseevent.x, mouseevent.y
        try:
            trans = self.axes.transAxes.inverted()
            xaxes, yaxes = trans.transform((x, y))
        except ValueError:
            return False, {}
        (l, b), (r, t) = self.axes.transAxes.transform([(0, 0), (1, 1)])
        inaxis = 0 <= xaxes <= 1 and (
            b - self._pickradius < y < b or
            t < y < t + self._pickradius)
        return inaxis, {}

    def set_label_position(self, position):
        self.label.set_verticalalignment(_api.getitem_checked({
            'top': 'baseline', 'bottom': 'top',
        }, position=position))
        self.label_position = position
        self.stale = True

    def _update_label_position(self, renderer):
        """
        Update the label position based on the bounding box enclosing
        all the ticklabels and axis spine
        """
        if not self._autolabelpos:
            return

        # get bounding boxes for this axis and any siblings
        # that have been set by `fig.align_xlabels()`
        bboxes, bboxes2 = self._get_tick_boxes_siblings(renderer=renderer)
        x, y = self.label.get_position()

        if self.label_position == 'bottom':
            # Union with extents of the bottom spine if present, of the axes otherwise.
            bbox = mtransforms.Bbox.union([
                *bboxes, self.axes.spines.get("bottom", self.axes).get_window_extent()])
            self.label.set_position(
                (x, bbox.y0 - self.labelpad * self.get_figure(root=True).dpi / 72))
        else:
            # Union with extents of the top spine if present, of the axes otherwise.
            bbox = mtransforms.Bbox.union([
                *bboxes2, self.axes.spines.get("top", self.axes).get_window_extent()])
            self.label.set_position(
                (x, bbox.y1 + self.labelpad * self.get_figure(root=True).dpi / 72))

    def _update_offset_text_position(self, bboxes, bboxes2):
        """
        Update the offset_text position based on the sequence of bounding
        boxes of all the ticklabels
        """
        x, y = self.offsetText.get_position()
        if not hasattr(self, '_tick_position'):
            self._tick_position = 'bottom'
        if self._tick_position == 'bottom':
            if not len(bboxes):
                bottom = self.axes.bbox.ymin
            else:
                bbox = mtransforms.Bbox.union(bboxes)
                bottom = bbox.y0
            y = bottom - self.OFFSETTEXTPAD * self.get_figure(root=True).dpi / 72
        else:
            if not len(bboxes2):
                top = self.axes.bbox.ymax
            else:
                bbox = mtransforms.Bbox.union(bboxes2)
                top = bbox.y1
            y = top + self.OFFSETTEXTPAD * self.get_figure(root=True).dpi / 72
        self.offsetText.set_position((x, y))

    def set_ticks_position(self, position):
        if position == 'top':
            self.set_tick_params(which='both', top=True, labeltop=True,
                                 bottom=False, labelbottom=False)
            self._tick_position = 'top'
            self.offsetText.set_verticalalignment('bottom')
        elif position == 'bottom':
            self.set_tick_params(which='both', top=False, labeltop=False,
                                 bottom=True, labelbottom=True)
            self._tick_position = 'bottom'
            self.offsetText.set_verticalalignment('top')
        elif position == 'both':
            self.set_tick_params(which='both', top=True,
                                 bottom=True)
        elif position == 'none':
            self.set_tick_params(which='both', top=False,
                                 bottom=False)
        elif position == 'default':
            self.set_tick_params(which='both', top=True, labeltop=False,
                                 bottom=True, labelbottom=True)
            self._tick_position = 'bottom'
            self.offsetText.set_verticalalignment('top')
        else:
            _api.check_in_list(['top', 'bottom', 'both', 'default', 'none'],
                               position=position)
        self.stale = True

    def tick_top(self):
        label = True
        if 'label1On' in self._major_tick_kw:
            label = (self._major_tick_kw['label1On']
                     or self._major_tick_kw['label2On'])
        self.set_ticks_position('top')
        # If labels were turned off before this was called, leave them off.
        self.set_tick_params(which='both', labeltop=label)

    def tick_bottom(self):
        label = True
        if 'label1On' in self._major_tick_kw:
            label = (self._major_tick_kw['label1On']
                     or self._major_tick_kw['label2On'])
        self.set_ticks_position('bottom')
        # If labels were turned off before this was called, leave them off.
        self.set_tick_params(which='both', labelbottom=label)

    def get_ticks_position(self):
        return {1: "bottom", 2: "top",
                "default": "default", "unknown": "unknown"}[
                    self._get_ticks_position()]

    get_view_interval, set_view_interval = _make_getset_interval(
        "view", "viewLim", "intervalx")
    get_data_interval, set_data_interval = _make_getset_interval(
        "data", "dataLim", "intervalx")

    def get_minpos(self):
        return self.axes.dataLim.minposx

    def set_default_intervals(self):
        # only change view if dataLim has not changed and user has
        # not changed the view:
        if (not self.axes.dataLim.mutatedx() and
                not self.axes.viewLim.mutatedx()):
            if self._converter is not None:
                info = self._converter.axisinfo(self.units, self)
                if info.default_limits is not None:
                    xmin, xmax = self.convert_units(info.default_limits)
                    self.axes.viewLim.intervalx = xmin, xmax
        self.stale = True

    def get_tick_space(self):
        ends = mtransforms.Bbox.unit().transformed(
            self.axes.transAxes - self.get_figure(root=False).dpi_scale_trans)
        length = ends.width * 72
        # There is a heuristic here that the aspect ratio of tick text
        # is no more than 3:1
        size = self._get_tick_label_size('x') * 3
        if size > 0:
            return int(np.floor(length / size))
        else:
            return 2**31 - 1


class YAxis(Axis):
    __name__ = 'yaxis'
    axis_name = 'y'  #: Read-only name identifying the axis.
    _tick_class = YTick

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init()

    def _init(self):
        # x in display coords, y in axes coords (updated at draw time by the
        # Rust core).
        self.label.set(
            x=0, y=0.5,
            verticalalignment='bottom', horizontalalignment='center',
            rotation='vertical', rotation_mode='anchor',
            transform=mtransforms.blended_transform_factory(
                mtransforms.IdentityTransform(), self.axes.transAxes),
        )
        self.label_position = 'left'

        if mpl.rcParams['ytick.labelcolor'] == 'inherit':
            tick_color = mpl.rcParams['ytick.color']
        else:
            tick_color = mpl.rcParams['ytick.labelcolor']

        # x in axes coords, y in display coords(!).
        self.offsetText.set(
            x=0, y=0.5,
            verticalalignment='baseline', horizontalalignment='left',
            transform=mtransforms.blended_transform_factory(
                self.axes.transAxes, mtransforms.IdentityTransform()),
            fontsize=mpl.rcParams['ytick.labelsize'],
            color=tick_color
        )
        self.offset_text_position = 'left'

    def contains(self, mouseevent):
        if self._different_canvas(mouseevent):
            return False, {}
        x, y = mouseevent.x, mouseevent.y
        try:
            trans = self.axes.transAxes.inverted()
            xaxes, yaxes = trans.transform((x, y))
        except ValueError:
            return False, {}
        (l, b), (r, t) = self.axes.transAxes.transform([(0, 0), (1, 1)])
        inaxis = 0 <= yaxes <= 1 and (
            l - self._pickradius < x < l or
            r < x < r + self._pickradius)
        return inaxis, {}

    def set_label_position(self, position):
        self.label.set_rotation_mode('anchor')
        self.label.set_verticalalignment(_api.getitem_checked({
            'left': 'bottom', 'right': 'top',
        }, position=position))
        self.label_position = position
        self.stale = True

    def _update_label_position(self, renderer):
        """
        Update the label position based on the bounding box enclosing
        all the ticklabels and axis spine
        """
        if not self._autolabelpos:
            return

        # get bounding boxes for this axis and any siblings
        # that have been set by `fig.align_ylabels()`
        bboxes, bboxes2 = self._get_tick_boxes_siblings(renderer=renderer)
        x, y = self.label.get_position()

        if self.label_position == 'left':
            # Union with extents of the left spine if present, of the axes otherwise.
            bbox = mtransforms.Bbox.union([
                *bboxes, self.axes.spines.get("left", self.axes).get_window_extent()])
            self.label.set_position(
                (bbox.x0 - self.labelpad * self.get_figure(root=True).dpi / 72, y))
        else:
            # Union with extents of the right spine if present, of the axes otherwise.
            bbox = mtransforms.Bbox.union([
                *bboxes2, self.axes.spines.get("right", self.axes).get_window_extent()])
            self.label.set_position(
                (bbox.x1 + self.labelpad * self.get_figure(root=True).dpi / 72, y))

    def _update_offset_text_position(self, bboxes, bboxes2):
        """
        Update the offset_text position based on the sequence of bounding
        boxes of all the ticklabels
        """
        x, _ = self.offsetText.get_position()
        if 'outline' in self.axes.spines:
            # Special case for colorbars:
            bbox = self.axes.spines['outline'].get_window_extent()
        else:
            bbox = self.axes.bbox
        top = bbox.ymax
        self.offsetText.set_position(
            (x, top + self.OFFSETTEXTPAD * self.get_figure(root=True).dpi / 72)
        )

    def set_offset_position(self, position):
        x, y = self.offsetText.get_position()
        x = _api.getitem_checked({'left': 0, 'right': 1}, position=position)

        self.offsetText.set_ha(position)
        self.offsetText.set_position((x, y))
        self.stale = True

    def set_ticks_position(self, position):
        if position == 'right':
            self.set_tick_params(which='both', right=True, labelright=True,
                                 left=False, labelleft=False)
            self.set_offset_position(position)
        elif position == 'left':
            self.set_tick_params(which='both', right=False, labelright=False,
                                 left=True, labelleft=True)
            self.set_offset_position(position)
        elif position == 'both':
            self.set_tick_params(which='both', right=True,
                                 left=True)
        elif position == 'none':
            self.set_tick_params(which='both', right=False,
                                 left=False)
        elif position == 'default':
            self.set_tick_params(which='both', right=True, labelright=False,
                                 left=True, labelleft=True)
        else:
            _api.check_in_list(['left', 'right', 'both', 'default', 'none'],
                               position=position)
        self.stale = True

    def tick_right(self):
        label = True
        if 'label1On' in self._major_tick_kw:
            label = (self._major_tick_kw['label1On']
                     or self._major_tick_kw['label2On'])
        self.set_ticks_position('right')
        # if labels were turned off before this was called
        # leave them off
        self.set_tick_params(which='both', labelright=label)

    def tick_left(self):
        label = True
        if 'label1On' in self._major_tick_kw:
            label = (self._major_tick_kw['label1On']
                     or self._major_tick_kw['label2On'])
        self.set_ticks_position('left')
        # if labels were turned off before this was called
        # leave them off
        self.set_tick_params(which='both', labelleft=label)

    def get_ticks_position(self):
        return {1: "left", 2: "right",
                "default": "default", "unknown": "unknown"}[
                    self._get_ticks_position()]

    get_view_interval, set_view_interval = _make_getset_interval(
        "view", "viewLim", "intervaly")
    get_data_interval, set_data_interval = _make_getset_interval(
        "data", "dataLim", "intervaly")

    def get_minpos(self):
        return self.axes.dataLim.minposy

    def set_default_intervals(self):
        # only change view if dataLim has not changed and user has
        # not changed the view:
        if (not self.axes.dataLim.mutatedy() and
                not self.axes.viewLim.mutatedy()):
            if self._converter is not None:
                info = self._converter.axisinfo(self.units, self)
                if info.default_limits is not None:
                    ymin, ymax = self.convert_units(info.default_limits)
                    self.axes.viewLim.intervaly = ymin, ymax
        self.stale = True

    def get_tick_space(self):
        ends = mtransforms.Bbox.unit().transformed(
            self.axes.transAxes - self.get_figure(root=False).dpi_scale_trans)
        length = ends.height * 72
        # Having a spacing of at least 2 just looks good.
        size = self._get_tick_label_size('y') * 2
        if size > 0:
            return int(np.floor(length / size))
        else:
            return 2**31 - 1
