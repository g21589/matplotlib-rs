"""Write <case>/diff.png: matplotlib | matplotlib_rs | differences (red)."""
import os
import sys

import numpy as np
from PIL import Image

here = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("COMPARE_OUT", os.path.join(here, "out"))

for case in sys.argv[1:]:
    name = os.path.splitext(os.path.basename(case))[0]
    d = os.path.join(OUT, name)
    for img in sorted(os.listdir(os.path.join(d, "mpl"))):
        a = np.asarray(Image.open(os.path.join(d, "mpl", img)).convert("RGBA"))
        b = np.asarray(Image.open(os.path.join(d, "rs", img)).convert("RGBA"))
        h = max(a.shape[0], b.shape[0])
        w = max(a.shape[1], b.shape[1])
        pa = np.zeros((h, w, 4), np.uint8); pa[:a.shape[0], :a.shape[1]] = a
        pb = np.zeros((h, w, 4), np.uint8); pb[:b.shape[0], :b.shape[1]] = b
        diff = np.full((h, w, 4), 255, np.uint8)
        m = np.any(pa != pb, axis=2)
        diff[m] = (255, 0, 0, 255)
        out = np.concatenate([pa, pb, diff], axis=1)
        p = os.path.join(d, "diff_" + img)
        Image.fromarray(out).save(p)
        print(p)
