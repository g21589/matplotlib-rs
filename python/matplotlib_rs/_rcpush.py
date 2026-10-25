"""Mirror rcParams into the Rust rc store (typed values)."""
from enum import Enum
import re

from . import _mplrs

_SPECIAL_STR = {"auto", "inherit", "none", "edge", "face", "fill", "round", "None"}
_CN = re.compile(r"\AC[0-9]+\Z")


def _is_colorkey(k):
    return "color" in k


def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def convert(key, v):
    """Return (kind, value) for the Rust store, or None to skip."""
    if isinstance(v, Enum):
        v = v.value
    if key == "axes.prop_cycle":
        from .colors import to_rgba
        try:
            cols = v.by_key().get("color", [])
        except Exception:
            cols = []
        return ("colorlist", [tuple(map(float, to_rgba(c))) for c in cols])
    if isinstance(v, bool):
        return ("bool", v)
    if _is_num(v):
        return ("num", float(v))
    if v is None:
        return ("none", None)
    if isinstance(v, str):
        if _is_colorkey(key) and v not in _SPECIAL_STR:
            from .colors import to_rgba
            try:
                return ("color", tuple(map(float, to_rgba(v))))
            except (ValueError, TypeError, KeyError):
                pass
        return ("str", v)
    if isinstance(v, (list, tuple)):
        if _is_colorkey(key) and len(v) in (3, 4) and all(_is_num(x) for x in v):
            return ("color", tuple(float(x) for x in v))
        if all(_is_num(x) for x in v):
            return ("numlist", [float(x) for x in v])
        if all(isinstance(x, str) for x in v):
            return ("strlist", list(v))
        return None
    return None


def push(key, val):
    c = convert(key, val)
    if c is not None:
        _mplrs.rc_set({key: c})
    if key == "axes.prop_cycle":
        # colours given as "CN" depend on the property cycle
        from . import rcParams
        upd = {}
        for k in rcParams.keys():
            if _is_colorkey(k):
                raw = rcParams._get(k)
                if isinstance(raw, str) and _CN.match(raw):
                    cc = convert(k, raw)
                    if cc is not None:
                        upd[k] = cc
        if upd:
            _mplrs.rc_set(upd)


def push_all(rc):
    out = {}
    pc = rc._get("axes.prop_cycle") if "axes.prop_cycle" in rc else None
    if pc is not None:
        out["axes.prop_cycle"] = convert("axes.prop_cycle", pc)
        _mplrs.rc_set({"axes.prop_cycle": out["axes.prop_cycle"]})
    for k in rc.keys():
        c = convert(k, rc._get(k))
        if c is not None:
            out[k] = c
    _mplrs.rc_set(out)
