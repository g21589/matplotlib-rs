"""Differential test of matplotlib_rs.ft2font.FT2Font against matplotlib's
FT2Font: face properties, SFNT tables and names, glyph names, glyph
metrics (scaled and NO_SCALE), outlines (get_path) and _layout."""
import os
import sys

import numpy as np
import matplotlib.font_manager as fm
from matplotlib import ft2font as mft

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
from matplotlib_rs import ft2font as rft  # noqa: E402

PROPS = ["family_name", "style_name", "postscript_name", "num_glyphs", "units_per_EM",
         "bbox", "ascender", "descender", "height", "max_advance_width",
         "max_advance_height", "underline_position", "underline_thickness",
         "num_charmaps", "scalable", "num_fixed_sizes"]
TABLES = ["head", "maxp", "OS/2", "hhea", "vhea", "post", "pclt"]
STRINGS = ["Hello World", "AVATAR fi ffl", "0.25 −1e6", "ÅÉÎ ñ"]


def glyph_tuple(g):
    return (g.width, g.height, g.horiBearingX, g.horiBearingY, g.horiAdvance,
            g.linearHoriAdvance, tuple(g.bbox))


def main():
    rng = np.random.default_rng(2)
    nfail = 0
    paths = sorted({f.fname for f in fm.fontManager.ttflist if not f.fname.endswith(".ttc")})

    def fail(msg):
        nonlocal nfail
        nfail += 1
        if nfail < 30:
            print(msg)

    for path in paths:
        m = mft.FT2Font(path)
        r = rft.FT2Font(path)
        name = os.path.basename(path)
        for p in PROPS:
            if getattr(m, p) != getattr(r, p):
                fail(f"{name}: {p} mpl={getattr(m, p)!r} rs={getattr(r, p)!r}")
        if m.face_flags.value != r.face_flags.value or m.style_flags.value != r.style_flags.value:
            fail(f"{name}: flags mpl={m.face_flags!r} {m.style_flags!r} rs={r.face_flags!r} {r.style_flags!r}")
        for t in TABLES:
            if m.get_sfnt_table(t) != r.get_sfnt_table(t):
                fail(f"{name}: table {t}\n  mpl={m.get_sfnt_table(t)}\n  rs ={r.get_sfnt_table(t)}")
        if m.get_sfnt() != r.get_sfnt():
            fail(f"{name}: get_sfnt differs")
        if m.get_charmap() != r.get_charmap():
            fail(f"{name}: get_charmap differs")
        gids = rng.choice(m.num_glyphs, min(m.num_glyphs, 60), replace=False)
        for gid in map(int, gids):
            if m.get_glyph_name(gid) != r.get_glyph_name(gid):
                fail(f"{name}: glyph name {gid} mpl={m.get_glyph_name(gid)!r} rs={r.get_glyph_name(gid)!r}")
            for flags in [mft.LoadFlags.NO_SCALE, mft.LoadFlags.NO_HINTING]:
                m.set_size(12.5, 100)
                r.set_size(12.5, 100)
                gm = m.load_glyph(gid, flags)
                gr = r.load_glyph(gid, rft.LoadFlags(flags.value))
                if glyph_tuple(gm) != glyph_tuple(gr):
                    fail(f"{name}: glyph {gid} {flags!r} mpl={glyph_tuple(gm)} rs={glyph_tuple(gr)}")
                vm, cm = m.get_path()
                vr, cr = r.get_path()
                if not (np.array_equal(vm, vr) and np.array_equal(cm, cr)):
                    fail(f"{name}: path {gid} {flags!r}")
        m.set_size(10, 72)
        r.set_size(10, 72)
        for s in STRINGS:
            lm = [(it.char, it.glyph_index, it.x, it.y, it.prev_kern)
                  for it in m._layout(s, mft.LoadFlags.NO_HINTING)]
            lr = [(it.char, it.glyph_index, it.x, it.y, it.prev_kern)
                  for it in r._layout(s, rft.LoadFlags.NO_HINTING)]
            if lm != lr:
                fail(f"{name}: _layout {s!r}\n  mpl={lm}\n  rs ={lr}")
    print(f"ft2font api diff: {nfail} failures over {len(paths)} fonts")
    return nfail


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
