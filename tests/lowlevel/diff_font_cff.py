"""Differential tests of the Rust CFF (OpenType .otf) glyph loader against
matplotlib's FT2Font (FreeType 2.14's Adobe CFF engine): outlines, metrics
and bitmaps of every glyph (sampled for very large fonts)."""
import glob
import os
import sys

import numpy as np
from matplotlib.ft2font import FT2Font, LoadFlags, RenderMode

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
from matplotlib_rs._mplrs import FT2Font as RsFont  # noqa: E402

FONTS = sorted(glob.glob("/usr/share/fonts/opentype/**/*.otf", recursive=True))
MOVETO, LINETO, CURVE4, CLOSEPOLY = 1, 2, 4, 79


def decompose(points, tags, contours):
    """`FT_Outline_Decompose` (cubic-only outlines) as matplotlib's get_path."""
    verts, codes = [], []
    first = 0
    for last in contours:
        if verts:
            verts.append((0, 0))
            codes.append(CLOSEPOLY)
        start = points[first]
        verts.append(start)
        codes.append(MOVETO)
        i = first + 1
        while i <= last:
            if tags[i] & 1:
                verts.append(points[i])
                codes.append(LINETO)
                i += 1
            else:
                c1 = points[i]
                c2 = points[i + 1]
                to = points[i + 2] if i + 2 <= last else start
                verts += [c1, c2, to]
                codes += [CURVE4] * 3
                i += 3
        else:
            if tags[last] & 1:
                verts.append(start)
                codes.append(LINETO)
        first = last + 1
    if verts:
        verts.append((0, 0))
        codes.append(CLOSEPOLY)
    return np.array(verts, float).reshape(-1, 2) / 64, np.array(codes, np.uint8)


def main():
    rng = np.random.default_rng(1)
    nfail = nglyph = 0
    for path in FONTS:
        try:
            mf = FT2Font(path)
        except Exception:
            continue
        rf = RsFont(path, 0)
        if (mf.num_glyphs, mf.units_per_EM) != (rf.num_glyphs, rf.units_per_em):
            print(f"face mismatch {path}: {mf.num_glyphs} {rf.num_glyphs}")
            nfail += 1
            continue
        gids = np.arange(mf.num_glyphs)
        if len(gids) > 3000:
            gids = np.unique(rng.choice(gids, 3000))
        for size, dpi in [(10, 100), (33.3, 72), (120, 150)]:
            mf.set_size(size, dpi)
            rf.set_size(size, dpi)
            for gid in gids:
                gid = int(gid)
                g = mf.load_glyph(gid, LoadFlags.NO_HINTING)
                mm = (g.width, g.height, g.horiBearingX, g.horiBearingY, g.horiAdvance,
                      g.linearHoriAdvance, tuple(g.bbox))
                rm = rf.glyph_metrics(gid)
                mv, mc = mf.get_path()
                rv, rc = decompose(*rf.outline(gid))
                nglyph += 1
                if mm != tuple(rm) or mc.tolist() != rc.tolist() or not np.array_equal(mv, rv):
                    print(f"glyph mismatch {os.path.basename(path)} {size}/{dpi} gid={gid}: "
                          f"mpl={mm} rs={rm} ncodes {len(mc)} {len(rc)}")
                    nfail += 1
                    continue
                if size != 33.3 or gid % 7:
                    continue
                mb = mf._render_glyph(gid, LoadFlags.NO_HINTING, RenderMode.NORMAL)
                rl, rt, rbuf = rf._render_glyph(gid)
                if (mb.left, mb.top) != (rl, rt) or not np.array_equal(mb.buffer, rbuf):
                    print(f"bitmap mismatch {os.path.basename(path)} gid={gid}")
                    nfail += 1
            # rotated glyphs
            for _ in range(20):
                gid = int(rng.choice(gids))
                angle = float(rng.choice([30, 45, -60, 90]))
                c, s = np.cos(np.radians(angle)), np.sin(np.radians(angle))
                matrix = (0x10000 * np.array([[c, -s], [s, c]])).round().astype(int)
                delta = [int(rng.integers(-5000, 50000)), int(rng.integers(-5000, 50000))]
                mf._set_transform(matrix, delta)
                rf._set_transform(matrix.tolist(), delta)
                try:
                    mb = mf._render_glyph(gid, LoadFlags.NO_HINTING, RenderMode.NORMAL)
                except RuntimeError:
                    continue
                rl, rt, rbuf = rf._render_glyph(gid)
                if (mb.left, mb.top) != (rl, rt) or not np.array_equal(mb.buffer, rbuf):
                    print(f"rotated bitmap mismatch {os.path.basename(path)} gid={gid} angle={angle}")
                    nfail += 1
            mf.set_size(size, dpi)
            rf.set_size(size, dpi)
    print(f"cff font diff: {nfail} failures, {nglyph} glyphs compared over {len(FONTS)} fonts")
    return nfail


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
