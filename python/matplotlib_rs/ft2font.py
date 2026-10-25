"""Minimal ``matplotlib.ft2font`` replacement backed by the Rust font engine."""
from enum import Flag, IntEnum, IntFlag
from typing import NamedTuple

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


class Glyph:
    """The metrics of a loaded glyph (26.6 units, ``linearHoriAdvance`` 16.16)."""

    def __init__(self, metrics):
        (self.width, self.height, self.horiBearingX, self.horiBearingY,
         self.horiAdvance, self.linearHoriAdvance, self.bbox) = metrics
        self.vertBearingX = self.vertBearingY = self.vertAdvance = 0

    def __repr__(self):
        return (f"Glyph(width={self.width}, height={self.height}, "
                f"horiAdvance={self.horiAdvance}, bbox={self.bbox})")


class FT2Font:
    """matplotlib's ``FT2Font`` over the Rust font engine (scalable TrueType)."""

    def __init__(self, filename, hinting_factor=8, *, face_index=0,
                 _fallback_list=None, _kerning_factor=None, _warn_if_used=False):
        self.fname = str(filename)
        self.face_index = face_index
        self._fallback_list = _fallback_list or []
        self._f = _mplrs.FT2Font(self.fname, face_index)
        self.family_name = self._f.family_name
        self.style_name = self._f.style_name
        self.postscript_name = self._f.postscript_name
        self.num_glyphs = self._f.num_glyphs
        self.units_per_EM = self._f.units_per_em
        self.underline_thickness = self._f.underline_thickness
        self.underline_position = self._f.underline_position
        self.num_faces = 1
        self.scalable = True
        self._last_glyph = None

    def set_charmap(self, i):
        pass

    def select_charmap(self, i):
        pass

    def clear(self):
        pass

    def set_size(self, ptsize, dpi):
        self._f.set_size(float(ptsize), float(dpi))

    def _set_transform(self, matrix, delta):
        self._f._set_transform([[int(v) for v in row] for row in matrix], [int(v) for v in delta])

    def get_char_index(self, codepoint):
        return self._f.get_char_index(codepoint)

    def load_glyph(self, glyph_index, flags=LoadFlags.FORCE_AUTOHINT):
        g = Glyph(self._f.glyph_metrics(int(glyph_index)))
        self._last_glyph = int(glyph_index)
        return g

    def load_char(self, charcode, flags=LoadFlags.FORCE_AUTOHINT):
        return self.load_glyph(self.get_char_index(charcode), flags)

    def get_kerning(self, left, right, mode):
        return self._f.get_kerning(int(left), int(right), int(mode))

    def set_text(self, string, angle=0.0, flags=LoadFlags.DEFAULT, **kwargs):
        self._last = self._f.get_width_height_descent(string)

    def get_width_height(self):
        w, h, d = self._last
        return int(round(w * 64)), int(round(h * 64))

    def get_descent(self):
        return int(round(self._last[2] * 64))

    def get_sfnt_table(self, name):
        return self._f.get_sfnt_table(name)


# type aliases used in annotations of the vendored modules
CharacterCodeType = int
GlyphIndexType = int


class LayoutItem(NamedTuple):
    ft_object: "FT2Font"
    char: str
    glyph_index: int
    x: float
    y: float
    prev_kern: float
