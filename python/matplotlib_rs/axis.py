"""
`matplotlib.axis` for matplotlib_rs.

The tick machinery (locating, formatting, laying out and drawing ticks and
axis labels) runs in the Rust core.  The classes here are thin proxies that
keep matplotlib's Python API: they hold the user-visible locator / formatter
objects (instances of :mod:`matplotlib_rs.ticker`, i.e. matplotlib's own
classes), translate them to native Rust tick helpers when possible and call
back into Python otherwise.
"""
import datetime
import functools
import logging
from numbers import Real
import threading
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
from ._live import tspec

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
    t = type(fmt)
    if t is mticker.ScalarFormatter:
        uo = fmt._useOffset if not fmt._useOffset is False or not fmt.offset else None
        if fmt._useOffset is False and fmt.offset:
            uo = float(fmt.offset)
        elif isinstance(fmt._useOffset, bool):
            uo = bool(fmt._useOffset)
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


class _InfoOverride:
    """Axis information supplied by Rust while it calls back into Python."""
    local = threading.local()


def _with_info(axis, info, fn):
    prev = getattr(_InfoOverride.local, "map", None)
    m = dict(prev or {})
    m[id(axis)] = info
    _InfoOverride.local.map = m
    try:
        return fn()
    finally:
        _InfoOverride.local.map = prev


def _current_info(axis):
    m = getattr(_InfoOverride.local, "map", None)
    if m:
        return m.get(id(axis))
    return None


class _PyLocatorAdapter:
    """Lets the Rust core call a Python locator."""

    def __init__(self, axis, loc):
        self._axis = axis
        self._loc = loc
        self.rs_is_null = isinstance(loc, mticker.NullLocator)

    def rs_call(self, info):
        return [float(v) for v in np.asarray(_with_info(self._axis, info, self._loc), float).ravel()]

    def rs_tick_values(self, vmin, vmax, info):
        return [float(v) for v in np.asarray(
            _with_info(self._axis, info, lambda: self._loc.tick_values(vmin, vmax)), float).ravel()]

    def rs_nonsingular(self, v0, v1):
        a, b = self._loc.nonsingular(v0, v1)
        return float(a), float(b)

    def rs_view_limits(self, v0, v1):
        a, b = self._loc.view_limits(v0, v1)
        return float(a), float(b)


class _PyFormatterAdapter:
    def __init__(self, axis, fmt):
        self._axis = axis
        self._fmt = fmt

    def rs_format_ticks(self, values, info):
        vals = np.asarray(values, float)
        return [str(s) for s in _with_info(self._axis, info,
                                           lambda: self._fmt.format_ticks(vals))]

    def rs_get_offset(self):
        return str(self._fmt.get_offset())


# --------------------------------------------------------------------------
# Ticks


class Tick:
    """A proxy for one Rust tick (lines and labels are real artists)."""

    def __init__(self, axis, idx, major, tick1line, tick2line, gridline, label1, label2):
        self._axis_proxy = axis
        self._idx = idx
        self._major = major
        self.tick1line = tick1line
        self.tick2line = tick2line
        self.gridline = gridline
        self.label1 = label1
        self.label2 = label2
        self.axes = axis.axes

    def get_loc(self):
        return self._axis_proxy.get_ticklocs(minor=not self._major)[self._idx]

    @property
    def label(self):
        return self.label1

    def get_children(self):
        return [self.tick1line, self.tick2line, self.gridline, self.label1, self.label2]

    def set_label1(self, s):
        self.label1.set_text(s)

    def set_label2(self, s):
        self.label2.set_text(s)

    def set_pad(self, val):
        self._axis_proxy.set_tick_params(which="major" if self._major else "minor", pad=val)

    def get_pad(self):
        return self._axis_proxy._get_tick_kw(self._major).get("pad", mpl.rcParams[
            f"{self._axis_proxy.axis_name}tick.{'major' if self._major else 'minor'}.pad"])

    def get_tickdir(self):
        return self._axis_proxy._get_tick_kw(self._major).get(
            "tickdir", mpl.rcParams[f"{self._axis_proxy.axis_name}tick.direction"])

    def get_tick_padding(self):
        size = self._axis_proxy._get_tick_kw(self._major).get(
            "size", mpl.rcParams[f"{self._axis_proxy.axis_name}tick.{'major' if self._major else 'minor'}.size"])
        return size * {"in": 0.0, "inout": 0.5, "out": 1.0}[self.get_tickdir()]

    def set_visible(self, b):
        for a in self.get_children():
            a.set_visible(b)

    def get_visible(self):
        return True

    def _apply_params(self, **kwargs):
        self._axis_proxy.set_tick_params(which="major" if self._major else "minor", **kwargs)

    def update_position(self, loc):
        pass

    def set_clip_path(self, path, transform=None):
        self.gridline.set_clip_path(path, transform)


class Ticker:
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
            raise TypeError('locator must be a subclass of matplotlib.ticker.Locator')
        self._locator = locator

    @property
    def formatter(self):
        return self._formatter

    @formatter.setter
    def formatter(self, formatter):
        if not isinstance(formatter, mticker.Formatter):
            raise TypeError('formatter must be a subclass of matplotlib.ticker.Formatter')
        self._formatter = formatter


# --------------------------------------------------------------------------
# Axis


def _wrap_rs_text(cache, h):
    """A Python Text bound to an existing Rust text handle."""
    key = h.ident()
    t = cache.get(key)
    if t is None:
        t = mtext.Text._from_rs(h)
        cache[key] = t
    else:
        t._rs_refresh()
    return t


def _wrap_rs_line(cache, h, axes):
    key = h.ident()
    ln = cache.get(key)
    if ln is None:
        ln = mlines.Line2D._from_rs(h)
        ln.axes = axes
        cache[key] = ln
    return ln


class Axis(martist.Artist):
    OFFSETTEXTPAD = 3
    _tick_class = None

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
        self.label = _wrap_rs_text(self._text_cache, axes._rs.axis_label(self.axis_name))
        self.offsetText = _wrap_rs_text(self._text_cache, axes._rs.offset_text(self.axis_name))
        self._pickradius = pickradius
        self._major_tick_kw = dict()
        self._minor_tick_kw = dict()
        self._converter = None
        self._converter_is_explicit = False
        self.units = None
        self._scale = mscale.LinearScale(self)
        self._autoscale_on = True
        self._pushed = {}
        if clear:
            self.clear()

    # -- Rust helpers ----------------------------------------------------
    @property
    def _rs(self):
        return self.axes._rs

    def _get_tick_kw(self, major):
        return self._major_tick_kw if major else self._minor_tick_kw

    def _rs_sync_tickers(self):
        """Push the current locators / formatters if they changed."""
        for which, tk in (("major", self.major), ("minor", self.minor)):
            minor = which == "minor"
            loc = tk.locator
            if loc is not None:
                spec = _locator_spec(loc)
                key = (which, "loc")
                if spec is None:
                    old = self._pushed.get(key)
                    if not (isinstance(old, tuple) and old[0] == "py" and old[1] is loc):
                        self._rs.set_locator(self.axis_name, ("py", _PyLocatorAdapter(self, loc)), minor)
                        self._pushed[key] = ("py", loc)
                elif self._pushed.get(key) != spec:
                    self._rs.set_locator(self.axis_name, spec, minor)
                    self._pushed[key] = spec
            fmt = tk.formatter
            if fmt is not None:
                spec = _formatter_spec(fmt)
                key = (which, "fmt")
                if spec is None:
                    old = self._pushed.get(key)
                    if not (isinstance(old, tuple) and old[0] == "py" and old[1] is fmt):
                        self._rs.set_formatter(self.axis_name, ("py", _PyFormatterAdapter(self, fmt)), minor)
                        self._pushed[key] = ("py", fmt)
                elif self._pushed.get(key) != spec:
                    self._rs.set_formatter(self.axis_name, spec, minor)
                    self._pushed[key] = spec

    def _rs_info(self):
        info = _current_info(self)
        if info is None:
            info = self._rs.axis_info(self.axis_name)
        return info

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

    @property
    def labelpad(self):
        return self._rs.axis_get(self.axis_name, "labelpad")

    @labelpad.setter
    def labelpad(self, v):
        self._rs.axis_set(self.axis_name, "labelpad", float(v))

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

    @property
    def remove_overlapping_locs(self):
        return self._remove_overlapping_locs

    @remove_overlapping_locs.setter
    def remove_overlapping_locs(self, val):
        self._remove_overlapping_locs = bool(val)
        self._rs.axis_set(self.axis_name, "remove_overlapping_locs", bool(val))

    def get_remove_overlapping_locs(self):
        return self._remove_overlapping_locs

    def set_remove_overlapping_locs(self, val):
        self.remove_overlapping_locs = val

    def _get_shared_axes(self):
        return self.axes._shared_axes[self._get_axis_name()].get_siblings(self.axes)

    def _get_shared_axis(self):
        name = self._get_axis_name()
        return [ax._axis_map[name] for ax in self._get_shared_axes()]

    def _get_axis_name(self):
        return self.axis_name

    # -- visibility & zorder go to Rust -----------------------------------
    def set_visible(self, b):
        super().set_visible(b)
        self._rs.axis_set(self.axis_name, "visible", bool(b))

    def set_zorder(self, level):
        if level is None:
            level = 0
        self.zorder = level
        self._rs.axis_set(self.axis_name, "zorder", float(level))
        self.stale = True

    # -- scale -----------------------------------------------------------
    def get_transform(self):
        return self._scale.get_transform()

    def get_scale(self):
        return self._scale.name

    def _set_scale(self, value, **kwargs):
        if not isinstance(value, mscale.ScaleBase):
            self._scale = mscale.scale_factory(value, self, **kwargs)
        else:
            self._scale = value
        s = self._scale
        name = s.name
        kw = {}
        if name == "log":
            kw = dict(base=float(s.base), subs=None if s.subs is None else [float(v) for v in s.subs],
                      clip=getattr(s._transform, "_clip", True))
        elif name == "symlog":
            tr = s._transform
            kw = dict(base=float(tr.base), linthresh=float(tr.linthresh), linscale=float(tr.linscale),
                      subs=None if s.subs is None else [float(v) for v in s.subs])
        elif name == "asinh":
            kw = dict(linear_width=float(s.linear_width))
        if name in ("linear", "log", "symlog", "logit", "asinh"):
            self._rs.set_scale(self.axis_name, name, **kw)
        else:
            raise NotImplementedError(f"scale {name!r} is not supported by matplotlib_rs yet")
        self._pushed = {}
        self._scale.set_default_locators_and_formatters(self)
        self.isDefault_majloc = True
        self.isDefault_minloc = True
        self.isDefault_majfmt = True
        self.isDefault_minfmt = True
        self._rs_sync_tickers()

    def _set_axes_scale(self, value, **kwargs):
        name = self._get_axis_name()
        for ax in self._get_shared_axes():
            ax._axis_map[name]._set_scale(value, **kwargs)
            ax.stale = True

    def limit_range_for_scale(self, vmin, vmax):
        return self._scale.limit_range_for_scale(vmin, vmax, self.get_minpos())

    def _nan_out_of_scale_range(self, data):
        data = np.asanyarray(data, dtype=float)
        valid = self._scale.val_in_range(data)
        return data if np.all(valid) else np.where(valid, data, np.nan)

    def _get_autoscale_on(self):
        return self._rs.get_autoscale_on(self.axis_name)

    def _set_autoscale_on(self, b):
        if b is not None:
            self._rs.set_autoscale_on(self.axis_name, bool(b))

    def get_children(self):
        return [self.label, self.offsetText, *self.get_major_ticks(), *self.get_minor_ticks()]

    def _reset_major_tick_kw(self):
        self._major_tick_kw.clear()
        self._major_tick_kw['gridOn'] = (mpl.rcParams['axes.grid'] and
                                         mpl.rcParams['axes.grid.which'] in ('both', 'major'))

    def _reset_minor_tick_kw(self):
        self._minor_tick_kw.clear()
        self._minor_tick_kw['gridOn'] = (mpl.rcParams['axes.grid'] and
                                         mpl.rcParams['axes.grid.which'] in ('both', 'minor'))

    def clear(self):
        """Python-side reset; the Rust axis is cleared by the Axes."""
        self.label._rs_refresh()
        self.offsetText._rs_refresh()
        self._set_scale_py_only('linear')
        self.callbacks = cbook.CallbackRegistry(signals=["units"])
        self._major_tick_kw['gridOn'] = (mpl.rcParams['axes.grid'] and
                                         mpl.rcParams['axes.grid.which'] in ('both', 'major'))
        self._minor_tick_kw['gridOn'] = (mpl.rcParams['axes.grid'] and
                                         mpl.rcParams['axes.grid.which'] in ('both', 'minor'))
        self._converter = None
        self._converter_is_explicit = False
        self.units = None
        self.stale = True

    def _set_scale_py_only(self, value):
        """Set default locators/formatters matching the Rust defaults."""
        self._scale = mscale.scale_factory(value, self)
        self._scale.set_default_locators_and_formatters(self)
        self.isDefault_majloc = True
        self.isDefault_minloc = True
        self.isDefault_majfmt = True
        self.isDefault_minfmt = True
        self._pushed = {
            ("major", "loc"): _locator_spec(self.major.locator),
            ("major", "fmt"): _formatter_spec(self.major.formatter),
            ("minor", "loc"): _locator_spec(self.minor.locator),
            ("minor", "fmt"): _formatter_spec(self.minor.formatter),
        }

    def reset_ticks(self):
        self._rs.reset_ticks(self.axis_name)

    def minorticks_on(self):
        scale = self.get_scale()
        if scale == 'log':
            s = self._scale
            self.set_minor_locator(mticker.LogLocator(s.base, s.subs))
        elif scale == 'symlog':
            s = self._scale
            self.set_minor_locator(mticker.SymmetricalLogLocator(s._transform, s.subs))
        elif scale == 'asinh':
            s = self._scale
            self.set_minor_locator(
                mticker.AsinhLocator(s.linear_width, base=s._base, subs=s._subs))
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
        self._rs.tick_params(self.axis_name, which, bool(reset), _rs_tick_kw(kwtrans))
        for t in self._text_cache.values():
            t._rs_refresh()
        self.stale = True

    def get_tick_params(self, which='major'):
        _api.check_in_list(['major', 'minor'], which=which)
        if which == 'major':
            return self._translate_tick_params(self._major_tick_kw, reverse=True)
        return self._translate_tick_params(self._minor_tick_kw, reverse=True)

    @classmethod
    def _translate_tick_params(cls, kw, reverse=False):
        kw_ = {**kw}
        allowed_keys = [
            'size', 'width', 'color', 'tickdir', 'pad',
            'labelsize', 'labelcolor', 'labelfontfamily', 'zorder', 'gridOn',
            'tick1On', 'tick2On', 'label1On', 'label2On',
            'length', 'direction', 'left', 'bottom', 'right', 'top',
            'labelleft', 'labelbottom', 'labelright', 'labeltop',
            'labelrotation', 'labelrotation_mode',
            *_gridline_param_names]
        keymap = {
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

    # -- intervals -------------------------------------------------------
    def get_view_interval(self):
        info = _current_info(self)
        if info is not None:
            return np.array(info[0:2])
        return np.array(self.axes._rs.get_lim(self.axis_name))

    def set_view_interval(self, vmin, vmax, ignore=False):
        if ignore:
            self._rs.set_lim(self.axis_name, float(vmin), float(vmax), None)
        else:
            oldmin, oldmax = self.get_view_interval()
            if oldmin < oldmax:
                self.set_view_interval(min(vmin, vmax, oldmin), max(vmin, vmax, oldmax), ignore=True)
            else:
                self.set_view_interval(max(vmin, vmax, oldmin), min(vmin, vmax, oldmax), ignore=True)
        self.stale = True

    def get_data_interval(self):
        info = _current_info(self)
        if info is not None:
            return np.array(info[2:4])
        d = self._rs.get_datalim()
        return np.array([d[0], d[2]]) if self.axis_name == "x" else np.array([d[1], d[3]])

    def set_data_interval(self, vmin, vmax, ignore=False):
        if self.axis_name == "x":
            self._rs.update_datalim([[vmin, np.nan], [vmax, np.nan]], True, False)
        else:
            self._rs.update_datalim([[np.nan, vmin], [np.nan, vmax]], False, True)

    def get_minpos(self):
        info = _current_info(self)
        if info is not None:
            return info[4]
        d = self._rs.get_datalim()
        return d[4] if self.axis_name == "x" else d[5]

    def get_inverted(self):
        low, high = self.get_view_interval()
        return high < low

    def set_inverted(self, inverted):
        a, b = self.get_view_interval()
        self._set_lim(*sorted((a, b), reverse=bool(inverted)), auto=None)

    def set_default_intervals(self):
        pass

    def _set_lim(self, v0, v1, *, emit=True, auto):
        name = self._get_axis_name()
        self.axes._process_unit_info([(name, (v0, v1))], convert=False)
        v0 = self.axes._validate_converted_limits(v0, self.convert_units)
        v1 = self.axes._validate_converted_limits(v1, self.convert_units)
        if v0 is None or v1 is None:
            old0, old1 = self.get_view_interval()
            if v0 is None:
                v0 = old0
            if v1 is None:
                v1 = old1
        if self.get_scale() == 'log' and (v0 <= 0 or v1 <= 0):
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
        self._rs_sync_tickers()
        v0, v1 = self._rs.set_lim(name, float(v0), float(v1), auto)
        if emit:
            self.axes.callbacks.process(f"{name}lim_changed", self.axes)
            for other in self._get_shared_axes():
                if other is self.axes:
                    continue
                other.callbacks.process(f"{name}lim_changed", other)
        self.stale = True
        return v0, v1

    def _set_artist_props(self, a):
        if a is None:
            return
        a.set_figure(self.get_figure(root=False))

    # -- ticks -----------------------------------------------------------
    def get_tightbbox(self, renderer=None, *, for_layout_only=False):
        bb = self.axes.get_tightbbox(renderer)
        return bb

    def get_tick_padding(self):
        values = []
        values.append(self.get_major_ticks(1)[0].get_tick_padding() if self.get_ticklocs().size else 0)
        return max(values, default=0)

    def get_gridlines(self):
        return cbook.silent_list('Line2D gridline', [
            _wrap_rs_line(self._line_cache, h, self.axes)
            for h in self._rs.get_gridlines(self.axis_name, False)])

    def set_label(self, s):
        raise RuntimeError(
            "A legend label cannot be assigned to an Axis. Did you mean to "
            "set the axis label via set_label_text()?")

    def get_label(self):
        return self.label

    def get_offset_text(self):
        return self.offsetText

    def _labels(self, which, visible_only=True):
        martist._sync_dirty()
        self._rs_sync_tickers()
        return [_wrap_rs_text(self._text_cache, h)
                for h in self._rs.get_ticklabels(self.axis_name, which, visible_only)]

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
        return cbook.silent_list('Line2D ticklines', [
            _wrap_rs_line(self._line_cache, h, self.axes)
            for h in self._rs.get_ticklines(self.axis_name, False)])

    def get_minorticklines(self):
        return cbook.silent_list('Line2D ticklines', [
            _wrap_rs_line(self._line_cache, h, self.axes)
            for h in self._rs.get_ticklines(self.axis_name, True)])

    def get_ticklines(self, minor=False):
        if minor:
            return self.get_minorticklines()
        return self.get_majorticklines()

    def get_majorticklocs(self):
        martist._sync_dirty()
        self._rs_sync_tickers()
        return np.asarray(self._rs.get_ticklocs(self.axis_name, False))

    def get_minorticklocs(self):
        martist._sync_dirty()
        self._rs_sync_tickers()
        return np.asarray(self._rs.get_ticklocs(self.axis_name, True))

    def get_ticklocs(self, *, minor=False):
        return self.get_minorticklocs() if minor else self.get_majorticklocs()

    def get_ticks_direction(self, minor=False):
        kw = self._minor_tick_kw if minor else self._major_tick_kw
        d = kw.get("tickdir", mpl.rcParams[f"{self.axis_name}tick.direction"])
        n = len(self.get_ticklocs(minor=minor))
        return np.array([d] * n)

    def _make_ticks(self, major, n):
        martist._sync_dirty()
        self._rs_sync_tickers()
        which = "major" if major else "minor"
        labels = [_wrap_rs_text(self._text_cache, h)
                  for h in self._rs.get_ticklabels(self.axis_name, which, False)]
        lines = [_wrap_rs_line(self._line_cache, h, self.axes)
                 for h in self._rs.get_ticklines(self.axis_name, not major)]
        grids = [_wrap_rs_line(self._line_cache, h, self.axes)
                 for h in self._rs.get_gridlines(self.axis_name, not major)]
        nt = len(grids)
        l1 = labels[:nt]
        l2 = labels[nt:2 * nt]
        ticks = []
        for i in range(nt):
            ticks.append(Tick(self, i, major, lines[2 * i], lines[2 * i + 1], grids[i], l1[i], l2[i]))
        return ticks[:n] if n is not None else ticks

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
            elif not visible:
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

    # -- units (verbatim from matplotlib) ----------------------------------
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
        return self._rs.axis_get(self.axis_name, "label_position")

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
        if getattr(self, "axes", None) is not None and getattr(self.axes, "_rs", None) is not None:
            self._rs_sync_tickers()
        self.stale = True

    def set_major_locator(self, locator):
        _api.check_isinstance(mticker.Locator, locator=locator)
        self.isDefault_majloc = False
        self.major.locator = locator
        if self.major.formatter:
            self.major.formatter._set_locator(locator)
        locator.set_axis(self)
        if getattr(self, "axes", None) is not None and getattr(self.axes, "_rs", None) is not None:
            self._rs_sync_tickers()
        self.stale = True

    def set_minor_locator(self, locator):
        _api.check_isinstance(mticker.Locator, locator=locator)
        self.isDefault_minloc = False
        self.minor.locator = locator
        if self.minor.formatter:
            self.minor.formatter._set_locator(locator)
        locator.set_axis(self)
        if getattr(self, "axes", None) is not None and getattr(self.axes, "_rs", None) is not None:
            self._rs_sync_tickers()
        self.stale = True

    @staticmethod
    def _format_with_dict(tickd, x, pos):
        return tickd.get(x, "")

    def set_ticklabels(self, labels, *, minor=False, fontdict=None, **kwargs):
        try:
            labels = [t.get_text() if hasattr(t, 'get_text') else t
                      for t in labels]
        except TypeError:
            raise TypeError(f"{labels=!r} must be a sequence") from None
        locator = (self.get_minor_locator() if minor
                   else self.get_major_locator())
        if not labels:
            formatter = mticker.NullFormatter()
        elif isinstance(locator, mticker.FixedLocator):
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
                message="FixedFormatter should only be used together with FixedLocator")
            if minor:
                self.set_minor_formatter(formatter)
            else:
                self.set_major_formatter(formatter)
        if fontdict is not None:
            kwargs.update(fontdict)
        ret = self.get_ticklabels(minor=minor)
        if kwargs:
            for t in ret:
                t._internal_update(kwargs)
        self.stale = True
        return ret

    def _set_tick_locations(self, ticks, *, minor=False):
        ticks = self.convert_units(ticks)
        locator = mticker.FixedLocator(ticks)
        if len(ticks):
            for axis in self._get_shared_axis():
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

    def axis_date(self, tz=None):
        if isinstance(tz, str):
            import dateutil.tz
            tz = dateutil.tz.gettz(tz)
        self.update_units(datetime.datetime(2009, 1, 1, 0, 0, 0, 0, tz))

    def get_tick_space(self):
        info = _current_info(self)
        if info is not None:
            return info[5]
        return self._rs.axis_info(self.axis_name)[5]

    def _get_tick_label_size(self, axis_name):
        tick_kw = self._major_tick_kw
        size = tick_kw.get('labelsize', mpl.rcParams[f'{axis_name}tick.labelsize'])
        return mtext.FontProperties(size=size).get_size_in_points()

    def get_majorticklocs_info(self):
        return self._rs_info()

    def set_label_position(self, position):
        raise NotImplementedError()

    def get_ticks_position(self):
        return self._rs.axis_get(self.axis_name, "ticks_position")

    def _get_ticks_position(self):
        return self.get_ticks_position()

    def draw(self, renderer):
        pass

    def contains(self, mouseevent):
        return False, {}


def _rs_tick_kw(kw):
    """Resolve tick parameters to the values the Rust core expects."""
    out = {}
    for k, v in kw.items():
        if k in ("color", "labelcolor", "grid_color", "grid_c"):
            key = "grid_color" if k.startswith("grid") else k
            if isinstance(v, str) and v.lower() == "inherit":
                continue
            out[key] = mcolors.to_rgba(v)
        elif k in ("size", "width", "pad", "zorder", "grid_linewidth", "grid_lw", "grid_zorder"):
            key = {"grid_lw": "grid_linewidth"}.get(k, k)
            out[key] = float(v)
        elif k == "labelsize":
            out[k] = float(mtext.FontProperties(size=v).get_size_in_points())
        elif k in ("gridOn", "tick1On", "tick2On", "label1On", "label2On", "grid_visible",
                   "grid_antialiased", "grid_aa"):
            key = {"grid_aa": "grid_antialiased"}.get(k, k)
            out[key] = bool(v)
        elif k in ("grid_alpha",):
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
        elif k == "labelrotation_mode":
            out[k] = v
        elif k == "labelfontfamily":
            out[k] = v
    return out


class XAxis(Axis):
    __name__ = 'xaxis'
    axis_name = 'x'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def contains(self, mouseevent):
        return False, {}

    def set_label_position(self, position):
        _api.check_in_list(['top', 'bottom'], position=position)
        self._rs.axis_set("x", "label_position", position)
        self.label._rs_refresh()
        self.stale = True

    def _update_label_position(self, renderer):
        pass

    def _update_offset_text_position(self, bboxes, bboxes2):
        pass

    def set_ticks_position(self, position):
        if position == 'top':
            self.set_tick_params(which='both', top=True, labeltop=True,
                                 bottom=False, labelbottom=False)
            self._rs.axis_set("x", "tick_position", "top")
            self.offsetText.set_verticalalignment('bottom')
        elif position == 'bottom':
            self.set_tick_params(which='both', top=False, labeltop=False,
                                 bottom=True, labelbottom=True)
            self._rs.axis_set("x", "tick_position", "bottom")
            self.offsetText.set_verticalalignment('top')
        elif position == 'both':
            self.set_tick_params(which='both', top=True, bottom=True)
        elif position == 'none':
            self.set_tick_params(which='both', top=False, bottom=False)
        elif position == 'default':
            self.set_tick_params(which='both', top=True, labeltop=False,
                                 bottom=True, labelbottom=True)
            self._rs.axis_set("x", "tick_position", "bottom")
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
        self.set_tick_params(which='both', labeltop=label)

    def tick_bottom(self):
        label = True
        if 'label1On' in self._major_tick_kw:
            label = (self._major_tick_kw['label1On']
                     or self._major_tick_kw['label2On'])
        self.set_ticks_position('bottom')
        self.set_tick_params(which='both', labelbottom=label)


class YAxis(Axis):
    __name__ = 'yaxis'
    axis_name = 'y'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def contains(self, mouseevent):
        return False, {}

    def set_label_position(self, position):
        _api.check_in_list(['left', 'right'], position=position)
        self._rs.axis_set("y", "label_position", position)
        self.label._rs_refresh()
        self.stale = True

    def _update_label_position(self, renderer):
        pass

    def _update_offset_text_position(self, bboxes, bboxes2):
        pass

    def set_offset_position(self, position):
        x = _api.getitem_checked({'left': 0, 'right': 1}, position=position)
        self.offsetText.set_ha(position)
        self.offsetText.set_x(x)
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
            self.set_tick_params(which='both', right=True, left=True)
        elif position == 'none':
            self.set_tick_params(which='both', right=False, left=False)
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
        self.set_tick_params(which='both', labelright=label)

    def tick_left(self):
        label = True
        if 'label1On' in self._major_tick_kw:
            label = (self._major_tick_kw['label1On']
                     or self._major_tick_kw['label2On'])
        self.set_ticks_position('left')
        self.set_tick_params(which='both', labelleft=label)
