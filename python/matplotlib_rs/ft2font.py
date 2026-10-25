"""Minimal ``matplotlib.ft2font`` replacement backed by the Rust font engine."""
from enum import Flag, IntEnum, IntFlag

from . import _mplrs

__freetype_version__ = "2.14.3"
__freetype_build_type__ = "local"


class LoadFlags(IntFlag):
    DEFAULT = 0
    NO_SCALE = 1
    NO_HINTING = 2
    RENDER = 4
    NO_BITMAP = 8
    VERTICAL_LAYOUT = 16
    FORCE_AUTOHINT = 32
    CROP_BITMAP = 64
    PEDANTIC = 128
    IGNORE_GLOBAL_ADVANCE_WIDTH = 512
    NO_RECURSE = 1024
    IGNORE_TRANSFORM = 2048
    MONOCHROME = 4096
    LINEAR_DESIGN = 8192
    NO_AUTOHINT = 32768
    TARGET_NORMAL = 0
    TARGET_LIGHT = 65536
    TARGET_MONO = 131072
    TARGET_LCD = 196608
    TARGET_LCD_V = 262144


class StyleFlags(Flag):
    NORMAL = 0
    ITALIC = 1
    BOLD = 2


class Kerning(IntEnum):
    DEFAULT = 0
    UNFITTED = 1
    UNSCALED = 2


class RenderMode(IntEnum):
    NORMAL = 0
    LIGHT = 1
    MONO = 2
    LCD = 3
    LCD_V = 4
    SDF = 5


class FT2Font:
    """Subset of matplotlib's FT2Font API."""

    def __init__(self, filename, hinting_factor=8, *, face_index=0,
                 _fallback_list=None, _kerning_factor=None, _warn_if_used=False):
        self.fname = str(filename)
        self.face_index = face_index
        self._fallback_list = _fallback_list or []
        self._f = _mplrs.FT2Font(self.fname)
        entries = _mplrs.font_entries(self.fname)
        e = entries[0] if entries else (self.fname, 0, "", "normal", "normal", 400, "normal", "scalable")
        self.family_name = e[2]
        self.style_name = "Regular"
        self.num_faces = 1
        self.scalable = True

    def set_charmap(self, i):
        pass

    def clear(self):
        pass

    def set_size(self, ptsize, dpi):
        self._f.set_size(float(ptsize), float(dpi))

    def _set_transform(self, matrix, delta):
        self._f._set_transform([[int(v) for v in row] for row in matrix], [int(v) for v in delta])

    def get_char_index(self, codepoint):
        return self._f.get_char_index(codepoint)

    def set_text(self, string, angle=0.0, flags=LoadFlags.DEFAULT):
        self._last = self._f.get_width_height_descent(string)

    def get_width_height(self):
        w, h, d = self._last
        return int(round(w * 64)), int(round(h * 64))

    def get_descent(self):
        return int(round(self._last[2] * 64))
