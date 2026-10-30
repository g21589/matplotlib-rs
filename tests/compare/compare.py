"""Pixel comparison of matplotlib and matplotlib_rs on the case scripts.

Usage: python compare.py [case.py ...]   (default: all of cases/)

Each case is run in its own process with both implementations; every image
it saves must be pixel-identical.  Exit status 1 if any case differs.

COMPARE_MPL_PYTHON and COMPARE_RS_PYTHON select the interpreters of the two
sides (default: this one), e.g. matplotlib 3.11.2 on Python 3.12 against
matplotlib_rs on Python 3.10 with NumPy as its only installed package.
Without Pillow, matplotlib_rs encodes embedded raster images itself; with
COMPARE_DECODE_IMAGES=1, SVG/PDF files that differ are compared again with
their images decoded to pixels (see normalize.py), and all other content
must still be identical.
"""
import glob
import os
import subprocess
import sys

import numpy as np
from PIL import Image

here = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("COMPARE_OUT", os.path.join(here, "out"))
DECODE_IMAGES = os.environ.get("COMPARE_DECODE_IMAGES") == "1"
PYTHON = {"mpl": os.environ.get("COMPARE_MPL_PYTHON") or sys.executable,
          "rs": os.environ.get("COMPARE_RS_PYTHON") or sys.executable}


def run(impl, case, outdir):
    p = subprocess.run([PYTHON[impl], os.path.join(here, "runner.py"), impl, case, outdir],
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


def normalized(path):
    """File contents, without lines naming the output file (PostScript title)."""
    with open(path, "rb") as fh:
        data = fh.read()
    return b"\n".join(line for line in data.split(b"\n") if not line.startswith(b"%%Title:"))


def same_decoded(a, b):
    """Whether vector files *a* and *b* are identical apart from the encoding
    of their raster images."""
    import normalize
    with open(a, "rb") as fa, open(b, "rb") as fb:
        da, db = fa.read(), fb.read()
    if a.endswith(".svg"):
        return normalize.svg(da) == normalize.svg(db)
    if a.endswith(".pdf"):
        return normalize.pdf(da) == normalize.pdf(db)
    return False


def main():
    cases = sys.argv[1:] or sorted(glob.glob(os.path.join(here, "cases", "*.py")))
    for impl, python in PYTHON.items():
        if python != sys.executable:
            version = subprocess.run([python, "-c", "import platform; print(platform.python_version())"],
                                     capture_output=True, text=True, check=True).stdout.strip()
            print(f"{impl}: Python {version} ({python})")
    if os.environ.get("COMPARE_STABLE_EPS") == "1":
        print("Reference EPS: retaining temporary clip transforms to avoid upstream id-reuse bug; strict file comparison")
    failed = n_decoded = 0
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
        decoded = []
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
        extras = set(os.listdir(dr)) - set(os.listdir(dm))
        problems.extend(f"{name}: extra output" for name in sorted(extras))
        for f in sorted(glob.glob(os.path.join(dm, "*"))):
            base = os.path.basename(f)
            if base.endswith(".png"):
                continue
            pr = os.path.join(dr, base)
            if not os.path.exists(pr):
                problems.append(f"{base}: missing")
            elif normalized(f) != normalized(pr):
                if DECODE_IMAGES and same_decoded(f, pr):
                    decoded.append(base)
                else:
                    problems.append(f"{base}: differs")
        status = "OK  " if not problems else "DIFF"
        if problems:
            failed += 1
        notes = problems + ([f"same pixels in embedded images: {', '.join(decoded)}"]
                            if decoded else [])
        n_decoded += bool(decoded) and not problems
        print(f"{name:30s} {status} mpl {tm*1000:8.1f} ms  rs {tr*1000:8.1f} ms  "
              f"x{tm / tr if tr else 0:5.2f}  {'; '.join(notes)}")
    print(f"total: {len(cases) - failed}/{len(cases)} identical; "
          f"mpl {tot[0]:.3f}s rs {tot[1]:.3f}s"
          + (f"; {n_decoded} of them with differently encoded embedded images"
             if n_decoded else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
