"""Pixel comparison of matplotlib and matplotlib_rs on the case scripts.

Usage: python compare.py [case.py ...]   (default: all of cases/)

Each case is run in its own process with both implementations; every image
it saves must be pixel-identical.  Exit status 1 if any case differs.
"""
import glob
import os
import subprocess
import sys

import numpy as np
from PIL import Image

here = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("COMPARE_OUT", os.path.join(here, "out"))


def run(impl, case, outdir):
    p = subprocess.run([sys.executable, os.path.join(here, "runner.py"), impl, case, outdir],
                       capture_output=True, text=True)
    if p.returncode != 0:
        return None, p.stderr.strip().splitlines()[-1] if p.stderr.strip() else "failed"
    t = [line for line in p.stdout.splitlines() if line.startswith("time ")]
    return float(t[-1].split()[1]), None


def compare_images(a, b):
    ia = np.asarray(Image.open(a).convert("RGBA"))
    ib = np.asarray(Image.open(b).convert("RGBA"))
    if ia.shape != ib.shape:
        return f"shape {ia.shape} != {ib.shape}"
    d = np.any(ia != ib, axis=2)
    n = int(d.sum())
    if n:
        ys, xs = np.nonzero(d)
        md = int(np.abs(ia.astype(int) - ib.astype(int)).max())
        return f"{n} px differ (max {md}) in x {xs.min()}-{xs.max()} y {ys.min()}-{ys.max()}"
    return None


def main():
    cases = sys.argv[1:] or sorted(glob.glob(os.path.join(here, "cases", "*.py")))
    failed = 0
    tot = [0.0, 0.0]
    for case in cases:
        name = os.path.splitext(os.path.basename(case))[0]
        dm = os.path.join(OUT, name, "mpl")
        dr = os.path.join(OUT, name, "rs")
        for d in (dm, dr):
            os.makedirs(d, exist_ok=True)
            for f in glob.glob(os.path.join(d, "*")):
                os.remove(f)
        tm, em = run("mpl", case, dm)
        tr, er = run("rs", case, dr)
        if em or er:
            print(f"{name:30s} ERROR mpl={em} rs={er}")
            failed += 1
            continue
        tot[0] += tm
        tot[1] += tr
        imgs = sorted(os.path.basename(p) for p in glob.glob(os.path.join(dm, "*.png")))
        problems = []
        if not imgs:
            problems.append("no images")
        for img in imgs:
            pr = os.path.join(dr, img)
            if not os.path.exists(pr):
                problems.append(f"{img}: missing")
                continue
            r = compare_images(os.path.join(dm, img), pr)
            if r:
                problems.append(f"{img}: {r}")
        status = "OK  " if not problems else "DIFF"
        if problems:
            failed += 1
        print(f"{name:30s} {status} mpl {tm*1000:8.1f} ms  rs {tr*1000:8.1f} ms  "
              f"x{tm / tr if tr else 0:5.2f}  {'; '.join(problems)}")
    print(f"total: {len(cases) - failed}/{len(cases)} identical; "
          f"mpl {tot[0]:.3f}s rs {tot[1]:.3f}s")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
