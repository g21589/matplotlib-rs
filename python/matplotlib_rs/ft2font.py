"""Minimal ``matplotlib.ft2font`` replacement backed by the Rust font engine."""
from enum import Enum, Flag
from typing import NamedTuple

from . import _mplrs

__freetype_version__ = "2.14.3"
__freetype_build_type__ = "local"


class LoadFlags(Flag):
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
    COLOR = 1048576
    COMPUTE_METRICS = 2097152
    BITMAP_METRICS_ONLY = 4194304
    NO_SVG = 16777216
    TARGET_NORMAL = 0
    TARGET_LIGHT = 65536
    TARGET_MONO = 131072
    TARGET_LCD = 196608
    TARGET_LCD_V = 262144


class StyleFlags(Flag):
    NORMAL = 0
    ITALIC = 1
    BOLD = 2


class Kerning(Enum):
    DEFAULT = 0
    UNFITTED = 1
    UNSCALED = 2


class RenderMode(Enum):
    NORMAL = 0
    LIGHT = 1
    MONO = 2
    LCD = 3
    LCD_V = 4
    SDF = 5


class FaceFlags(Flag):
    SCALABLE = 1
    FIXED_SIZES = 2
    FIXED_WIDTH = 4
    SFNT = 8
    HORIZONTAL = 16
    VERTICAL = 32
    KERNING = 64
    FAST_GLYPHS = 128
    MULTIPLE_MASTERS = 256
    GLYPH_NAMES = 512
    EXTERNAL_STREAM = 1024
    HINTER = 2048
    CID_KEYED = 4096
    TRICKY = 8192
    COLOR = 16384
    VARIATION = 32768
    SVG = 65536
    SBIX = 131072
    SBIX_OVERLAY = 262144


class Glyph:
    """The metrics of a loaded glyph (26.6 units, ``linearHoriAdvance`` 16.16;
    font units for `LoadFlags.NO_SCALE` loads)."""

    def __init__(self, metrics):
        (self.width, self.height, self.horiBearingX, self.horiBearingY,
         self.horiAdvance, self.linearHoriAdvance, self.bbox) = metrics
        self.vertBearingX = self.vertBearingY = self.vertAdvance = 0

    def __repr__(self):
        return (f"Glyph(width={self.width}, height={self.height}, "
                f"horiAdvance={self.horiAdvance}, bbox={self.bbox})")


def _fixed(v):
    """FIXED_MAJOR/FIXED_MINOR of a 16.16 value."""
    v &= 0xffffffff
    major = (v >> 16) & 0xffff
    return (major - 0x10000 if major & 0x8000 else major, v & 0xffff)


class FT2Font:
    """matplotlib's ``FT2Font`` over the Rust font engine."""

    def __init__(self, filename, hinting_factor=8, *, face_index=0,
                 _fallback_list=None, _kerning_factor=None, _warn_if_used=False):
        self.face_index = face_index
        self._fallback_list = _fallback_list or []
        self._warn_if_used = _warn_if_used
        if hasattr(filename, "read"):
            # a file-like object (matplotlib reads it through a stream)
            self.fname = getattr(filename, "name", "<file-like>")
            filename.seek(0)
            self._f = _mplrs.FT2Font.from_bytes(filename.read(), face_index)
        else:
            self.fname = str(filename)
            self._f = _mplrs.FT2Font(self.fname, face_index)
        self.family_name = self._f.family_name
        self.style_name = self._f.style_name
        self.postscript_name = self._f.postscript_name
        self.num_glyphs = self._f.num_glyphs
        self.units_per_EM = self._f.units_per_em
        self.num_fixed_sizes = 0
        self.num_named_instances = 0
        self.scalable = True
        self._last = (0.0, 0.0, 0.0)
        self._init_face()

    def _table(self, tag):
        return self._f.table(tag)

    def _init_face(self):
        """Face fields as FreeType's sfnt loader sets them."""
        import struct
        head = self._table("head")
        hhea = self._table("hhea")
        post = self._table("post")
        os2 = self.get_sfnt_table("OS/2")
        self.bbox = struct.unpack(">4h", head[36:44])
        asc, desc, gap = struct.unpack(">3h", hhea[4:10])
        height = asc - desc + gap
        if os2 is not None and os2["fsSelection"] & 128:  # USE_TYPO_METRICS
            asc, desc = os2["sTypoAscender"], os2["sTypoDescender"]
            height = asc - desc + os2["sTypoLineGap"]
        elif not (asc or desc) and os2 is not None:
            if os2["sTypoAscender"] or os2["sTypoDescender"]:
                asc, desc = os2["sTypoAscender"], os2["sTypoDescender"]
                height = asc - desc + os2["sTypoLineGap"]
            else:
                asc, desc = os2["usWinAscent"], -os2["usWinDescent"]
                height = asc - desc
        self.ascender, self.descender, self.height = asc, desc, height
        self.max_advance_width = struct.unpack(">H", hhea[10:12])[0]
        vhea = self._table("vhea")
        self.max_advance_height = (struct.unpack(">H", vhea[10:12])[0]
                                   if vhea and self._table("vmtx") else height)
        if post is not None and len(post) >= 16:
            fmt, _, upos, uthick, fixed = struct.unpack(">IihhI", post[:16])
        else:
            fmt, upos, uthick, fixed = 0x30000, 0, 0, 0
        self.underline_position = upos - int(uthick / 2)
        self.underline_thickness = uthick
        cmap = self._table("cmap")
        is_cff, is_cid, enc_cmap = self._f.cff_kind()
        self.num_charmaps = len(_cmap_subtables(cmap)) + bool(enc_cmap)
        flags = (FaceFlags.SCALABLE | FaceFlags.SFNT | FaceFlags.HORIZONTAL
                 | FaceFlags.EXTERNAL_STREAM)
        if vhea and self._table("vmtx"):
            flags |= FaceFlags.VERTICAL
        if fixed:
            flags |= FaceFlags.FIXED_WIDTH
        kern = self._table("kern")
        if kern and len(kern) > 4 and struct.unpack(">H", kern[2:4])[0]:
            flags |= FaceFlags.KERNING
        if is_cff:
            if not is_cid:
                flags |= FaceFlags.GLYPH_NAMES
        else:
            flags |= FaceFlags.HINTER
            if fmt != 0x30000:
                flags |= FaceFlags.GLYPH_NAMES
        if self._table("fvar"):
            flags |= FaceFlags.MULTIPLE_MASTERS | FaceFlags.VARIATION
        if self._table("COLR") and self._table("CPAL") or self._table("sbix") or self._table("CBDT"):
            flags |= FaceFlags.COLOR
        self.face_flags = flags
        style = StyleFlags.NORMAL
        if os2 is not None:
            if os2["fsSelection"] & 512 or os2["fsSelection"] & 1:
                style |= StyleFlags.ITALIC
            if os2["fsSelection"] & 32:
                style |= StyleFlags.BOLD
        else:
            mac = struct.unpack(">H", head[44:46])[0]
            if mac & 1:
                style |= StyleFlags.BOLD
            if mac & 2:
                style |= StyleFlags.ITALIC
        self.style_flags = style
        self.num_faces = 1

    def set_charmap(self, i):
        pass

    def select_charmap(self, i):
        pass

    def clear(self):
        self._last = (0.0, 0.0, 0.0)

    def set_size(self, ptsize, dpi):
        self._f.set_size(float(ptsize), float(dpi))

    def _set_transform(self, matrix, delta):
        self._f._set_transform([[int(v) for v in row] for row in matrix], [int(v) for v in delta])

    def get_char_index(self, codepoint, _fallback=False):
        return self._f.get_char_index(codepoint)

    def get_num_glyphs(self):
        return self.num_glyphs

    def load_glyph(self, glyph_index, flags=LoadFlags.FORCE_AUTOHINT):
        if not isinstance(flags, LoadFlags):
            flags = LoadFlags(flags)
        return Glyph(self._f.load(int(glyph_index), bool(flags & LoadFlags.NO_SCALE)))

    def load_char(self, charcode, flags=LoadFlags.FORCE_AUTOHINT):
        return self.load_glyph(self.get_char_index(charcode), flags)

    def get_path(self):
        return self._f.get_path()

    def get_glyph_name(self, index):
        if not FaceFlags.GLYPH_NAMES & self.face_flags:
            return f"uni{index:08x}"
        name = self._f.glyph_name(int(index))
        return "" if name is None else name

    def get_name_index(self, name):
        for i in range(self.num_glyphs):
            if self._f.glyph_name(i) == name:
                return i
        return 0

    def get_kerning(self, left, right, mode):
        return self._f.get_kerning(int(left), int(right), int(getattr(mode, "value", mode)))

    def _layout(self, text, flags, *, features=None, language=None):
        items = []
        glyphs = self._f._shape(text)
        clusters = sorted({g[1] for g in glyphs})
        x = y = 0
        prev_advance = None
        prev_x = 0
        for gid, cluster, xa, ya, xo, yo in glyphs:
            g = self.load_glyph(gid, flags)
            prev_kern = 0.0
            if prev_advance is not None:
                prev_kern = (x + xo) - prev_x - prev_advance
            nxt = next((c for c in clusters if c > cluster), len(text))
            items.append(LayoutItem(self, text[cluster:nxt], gid, (x + xo) / 64.0,
                                    (y + yo) / 64.0, prev_kern / 64.0))
            prev_x = x + xo
            x += xa
            y += ya
            prev_advance = g.linearHoriAdvance / 1024.0
        return items

    def set_text(self, string, angle=0.0, flags=LoadFlags.DEFAULT, **kwargs):
        self._last = self._f.get_width_height_descent(string)
        import numpy as np
        return np.array([(it.x, it.y) for it in self._layout(string, flags)],
                        dtype=float).reshape(-1, 2)

    def get_width_height(self):
        w, h, d = self._last
        return int(round(w * 64)), int(round(h * 64))

    def get_descent(self):
        return int(round(self._last[2] * 64))

    def get_sfnt(self):
        # (FreeType skips empty records)
        return {(p, e, l, n): b for p, e, l, n, b in self._f.name_records() if b}

    def get_ps_font_info(self):
        raise ValueError("Could not get PS font info")

    def get_charmap(self):
        return _read_cmap(self._table("cmap"))

    def get_sfnt_table(self, name):
        import struct
        tag = {"head": "head", "maxp": "maxp", "OS/2": "OS/2", "hhea": "hhea",
               "vhea": "vhea", "post": "post", "pclt": "PCLT"}.get(name)
        d = self._table(tag) if tag else None
        if d is None:
            return None
        if name == "head":
            (ver, rev, csum, magic, flags, upem, c0, c1, m0, m1, xmin, ymin, xmax, ymax,
             mac, lrp, fdh, itlf, gdf) = struct.unpack(">iiiiHHIIIIhhhhHHhhh", d[:54])
            return dict(version=_fixed(ver), fontRevision=_fixed(rev),
                        checkSumAdjustment=csum, magicNumber=magic, flags=flags,
                        unitsPerEm=upem, created=(c0, c1), modified=(m0, m1),
                        xMin=xmin, yMin=ymin, xMax=xmax, yMax=ymax, macStyle=mac,
                        lowestRecPPEM=lrp, fontDirectionHint=fdh,
                        indexToLocFormat=itlf, glyphDataFormat=gdf)
        if name == "maxp":
            ver, n = struct.unpack(">iH", d[:6])
            rest = (struct.unpack(">13H", d[6:32]) if ver >= 0x10000 and len(d) >= 32
                    else (0,) * 13)
            keys = ["maxPoints", "maxContours", "maxComponentPoints",
                    "maxComponentContours", "maxZones", "maxTwilightPoints", "maxStorage",
                    "maxFunctionDefs", "maxInstructionDefs", "maxStackElements",
                    "maxSizeOfInstructions", "maxComponentElements", "maxComponentDepth"]
            r = dict(version=_fixed(ver), numGlyphs=n, **dict(zip(keys, rest)))
            if ver >= 0x10000:
                # (FreeType's adjustments for broken fonts)
                r["maxFunctionDefs"] = max(r["maxFunctionDefs"], 64)
                r["maxTwilightPoints"] = min(r["maxTwilightPoints"], 0xFFFF - 4)
            return r
        if name == "OS/2":
            if len(d) < 68:
                return None
            v = struct.unpack(">HhHHHhhhhhhhhhhh", d[:32])
            ver = v[0]
            r = dict(version=ver, xAvgCharWidth=v[1], usWeightClass=v[2], usWidthClass=v[3],
                     fsType=v[4], ySubscriptXSize=v[5], ySubscriptYSize=v[6],
                     ySubscriptXOffset=v[7], ySubscriptYOffset=v[8],
                     ySuperscriptXSize=v[9], ySuperscriptYSize=v[10],
                     ySuperscriptXOffset=v[11], ySuperscriptYOffset=v[12],
                     yStrikeoutSize=v[13], yStrikeoutPosition=v[14], sFamilyClass=v[15],
                     panose=bytes(d[32:42]),
                     ulUnicodeRange=struct.unpack(">4I", d[42:58]),
                     achVendID=bytes(d[58:62]))
            sel, first, last = struct.unpack(">3H", d[62:68])
            r.update(fsSelection=sel, usFirstCharIndex=first, usLastCharIndex=last)
            ta, td, tg, wa, wd = (struct.unpack(">hhhHH", d[68:78]) if len(d) >= 78
                                  else (0, 0, 0, 0, 0))
            r.update(sTypoAscender=ta, sTypoDescender=td, sTypoLineGap=tg,
                     usWinAscent=wa, usWinDescent=wd)
            if ver >= 1:
                r["ulCodePageRange"] = (struct.unpack(">2I", d[78:86]) if len(d) >= 86
                                        else (0, 0))
            if ver >= 2:
                x = struct.unpack(">hhHHH", d[86:96]) if len(d) >= 96 else (0,) * 5
                r.update(sxHeight=x[0], sCapHeight=x[1], usDefaultChar=x[2],
                         usBreakChar=x[3], usMaxContext=x[4])
            if ver >= 5:
                x = struct.unpack(">HH", d[96:100]) if len(d) >= 100 else (0, 0)
                r.update(usLowerOpticalPointSize=x[0], usUpperOpticalPointSize=x[1])
            return r
        if name in ("hhea", "vhea"):
            v = struct.unpack(">ihhhHhhhhhh8xhH", d[:36])
            if name == "hhea":
                keys = ["ascent", "descent", "lineGap", "advanceWidthMax", "minLeftBearing",
                        "minRightBearing", "xMaxExtent", "caretSlopeRise", "caretSlopeRun",
                        "caretOffset", "metricDataFormat", "numOfLongHorMetrics"]
            else:
                keys = ["vertTypoAscender", "vertTypoDescender", "vertTypoLineGap",
                        "advanceHeightMax", "minTopSideBearing", "minBottomSideBearing",
                        "yMaxExtent", "caretSlopeRise", "caretSlopeRun", "caretOffset",
                        "metricDataFormat", "numOfLongVerMetrics"]
            return dict(version=_fixed(v[0]), **dict(zip(keys, v[1:])))
        if name == "post":
            fmt, ia, up, ut, fixed, m42, x42, m1, x1 = struct.unpack(">iihhIIIII", d[:32])
            return dict(format=_fixed(fmt), italicAngle=_fixed(ia), underlinePosition=up,
                        underlineThickness=ut, isFixedPitch=fixed, minMemType42=m42,
                        maxMemType42=x42, minMemType1=m1, maxMemType1=x1)
        if name == "pclt":
            (ver, num, pitch, xh, style, fam, cap, sym) = struct.unpack(">iIHHHHHH", d[:20])
            return dict(version=_fixed(ver), fontNumber=num, pitch=pitch, xHeight=xh,
                        style=style, typeFamily=fam, capHeight=cap, symbolSet=sym,
                        typeFace=bytes(d[20:36]), characterComplement=bytes(d[36:44]),
                        strokeWeight=struct.unpack(">b", d[50:51])[0],
                        widthType=struct.unpack(">b", d[51:52])[0],
                        serifStyle=d[52])
        return None


def _cmap_subtables(cmap):
    """The cmap subtables FreeType loads: [(platform, encoding, format, offset)]."""
    import struct
    if not cmap or len(cmap) < 4:
        return []
    out = []
    n = struct.unpack(">H", cmap[2:4])[0]
    for i in range(n):
        if 12 + 8 * i > len(cmap):
            break
        pid, eid, off = struct.unpack(">HHI", cmap[4 + 8 * i:12 + 8 * i])
        if off + 2 <= len(cmap):
            out.append((pid, eid, struct.unpack(">H", cmap[off:off + 2])[0], off))
    return out


def _read_cmap(cmap):
    """The Unicode charmap FreeType selects (`find_unicode_charmap`):
    {codepoint: glyph index}."""
    import struct
    subs = [s for s in _cmap_subtables(cmap) if s[2] != 14]

    def unicode(pid, eid):
        return pid in (0, 2) or (pid == 3 and eid in (1, 10))

    chosen = None
    for pid, eid, fmt, off in reversed(subs):
        if unicode(pid, eid) and ((pid, eid) in ((3, 10), (0, 4)) or (pid, eid, fmt) == (0, 6, 13)):
            chosen = (fmt, off)
            break
    if chosen is None:
        for pid, eid, fmt, off in reversed(subs):
            if unicode(pid, eid):
                chosen = (fmt, off)
                break
    if chosen is None:
        return {}
    fmt, off = chosen
    out = {}
    if fmt == 4:
        segx2 = struct.unpack(">H", cmap[off + 6:off + 8])[0]
        seg = segx2 // 2
        ends = struct.unpack(f">{seg}H", cmap[off + 14:off + 14 + segx2])
        p = off + 16 + segx2
        starts = struct.unpack(f">{seg}H", cmap[p:p + segx2])
        deltas = struct.unpack(f">{seg}h", cmap[p + segx2:p + 2 * segx2])
        ro_pos = p + 2 * segx2
        ros = struct.unpack(f">{seg}H", cmap[ro_pos:ro_pos + segx2])
        for i in range(seg):
            for c in range(starts[i], ends[i] + 1):
                if c == 0xFFFF:
                    continue
                if ros[i] == 0:
                    g = (c + deltas[i]) & 0xFFFF
                else:
                    q = ro_pos + 2 * i + ros[i] + 2 * (c - starts[i])
                    g = struct.unpack(">H", cmap[q:q + 2])[0]
                    if g:
                        g = (g + deltas[i]) & 0xFFFF
                if g:
                    out[c] = g
    elif fmt in (12, 13):
        ng = struct.unpack(">I", cmap[off + 12:off + 16])[0]
        for i in range(ng):
            s, e, g0 = struct.unpack(">3I", cmap[off + 16 + 12 * i:off + 28 + 12 * i])
            for c in range(s, min(e, 0x10FFFF) + 1):
                g = g0 + c - s if fmt == 12 else g0
                if g:
                    out[c] = g
    elif fmt == 6:
        first, count = struct.unpack(">HH", cmap[off + 6:off + 10])
        gs = struct.unpack(f">{count}H", cmap[off + 10:off + 10 + 2 * count])
        out = {first + i: g for i, g in enumerate(gs) if g}
    elif fmt == 0:
        out = {c: g for c, g in enumerate(cmap[off + 6:off + 262]) if g}
    return out


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
