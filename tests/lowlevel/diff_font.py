"""Differential tests of the Rust font engine against matplotlib's FT2Font
(FreeType 2.14 + libraqm/HarfBuzz): glyph bitmaps, layout, text metrics."""
import os
import sys

import numpy as np
import matplotlib
from matplotlib.ft2font import FT2Font, LoadFlags, RenderMode

from matplotlib_rs._mplrs import FT2Font as RsFont

FONTDIR = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
FONTS = ["DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans-Oblique.ttf",
         "DejaVuSansMono.ttf", "DejaVuSerif.ttf", "DejaVuSans-BoldOblique.ttf"]

STRINGS = ["0", "1.0", "\N{MINUS SIGN}0.5", "Hello World", "AVATAR Tj fi fl ffi",
           "x label", "Temperature (°C)", "sin(x)", "1e6", "−1.25", "ÅÉÎõü ñ",
           "Quick brown fox jumps; over the lazy dog!", "lp", "WAVE", "To", "Ty",
           "αβγ ΔΩ", "Кириллица", "2024-10-25", "→ ← ↑", " leading space", "trail "]


def main():
    rng = np.random.default_rng(0)
    nfail = 0
    nglyph = 0
    for fname in FONTS:
        path = os.path.join(FONTDIR, fname)
        mf = FT2Font(path, _kerning_factor=0)
        rf = RsFont(path)
        for size in [6, 8, 10, 12, 14.4, 20, 33.3]:
            for dpi in [72, 100, 150, 300]:
                mf.set_size(size, dpi)
                rf.set_size(size, dpi)
                for s in STRINGS:
                    # text metrics (set_text path)
                    mf.set_text(s, 0.0, flags=LoadFlags.NO_HINTING)
                    w, h = mf.get_width_height()
                    d = mf.get_descent()
                    mwhd = (w / 64, h / 64, d / 64)
                    rwhd = rf.get_width_height_descent(s)
                    if not np.allclose(mwhd, rwhd, rtol=0, atol=0):
                        print(f"WHD mismatch {fname} {size} {dpi} {s!r}: mpl={mwhd} rs={rwhd}")
                        nfail += 1
                    # layout
                    mitems = [(it.glyph_index, it.x, it.y) for it in mf._layout(s, LoadFlags.NO_HINTING)]
                    ritems = rf._layout(s)
                    if mitems != ritems:
                        print(f"layout mismatch {fname} {size} {dpi} {s!r}:\n  mpl={mitems}\n  rs ={ritems}")
                        nfail += 1
                # glyph bitmaps under random transforms
                for _ in range(12):
                    gid = int(rng.integers(1, 600))
                    angle = float(rng.choice([0, 0, 90, 30, 45, -60, 180]))
                    cos, sin = np.cos(np.radians(angle)), np.sin(np.radians(angle))
                    matrix = (0x10000 * np.array([[cos, -sin], [sin, cos]])).round().astype(int)
                    delta = [int(rng.integers(-5000, 50000)), int(rng.integers(-5000, 50000))]
                    mf.set_size(size, dpi)
                    rf.set_size(size, dpi)
                    mf._set_transform(matrix, delta)
                    rf._set_transform(matrix.tolist(), delta)
                    try:
                        mb = mf._render_glyph(gid, LoadFlags.NO_HINTING, RenderMode.NORMAL)
                    except RuntimeError:
                        continue
                    rl, rt, rbuf = rf._render_glyph(gid)
                    nglyph += 1
                    if (mb.left, mb.top) != (rl, rt) or not np.array_equal(mb.buffer, rbuf):
                        print(f"glyph mismatch {fname} size={size} dpi={dpi} gid={gid} angle={angle} "
                              f"delta={delta}: mpl=({mb.left},{mb.top},{mb.buffer.shape}) "
                              f"rs=({rl},{rt},{rbuf.shape})")
                        if mb.buffer.shape == rbuf.shape:
                            print("   maxdiff", np.abs(mb.buffer.astype(int) - rbuf).max())
                        nfail += 1
    print(f"font diff: {nfail} failures, {nglyph} glyph renders compared")
    return nfail


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
