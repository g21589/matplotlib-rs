"""Run matplotlib's gallery examples with matplotlib and matplotlib_rs and
compare the figures they leave open (saved as PNG at exit).

Usage: python run_gallery.py GALLERY_DIR [pattern ...] [-j N]

Each example runs in its own process per implementation, from its own
directory; results go to $GALLERY_OUT (default: tests/gallery/out).
Prints one line per example: OK, DIFF (pixels), ERR (which side failed)
or TIMEOUT, then a summary.
"""
import argparse
import concurrent.futures as cf
import glob
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image

here = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("GALLERY_OUT", os.path.join(here, "out"))
RUNNER = r'''
import os, sys, runpy, time, warnings, json
impl, script, outdir, rcdir = sys.argv[1:5]
os.environ["MATPLOTLIBRC"] = rcdir
os.environ["MPLBACKEND"] = "Agg"
if impl == "rs":
    sys.path.insert(0, os.path.join(%(root)r, "python"))
    import matplotlib_rs
    matplotlib_rs.install()
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *a, **k: None
os.chdir(os.path.dirname(script))
sys.argv = [script]
t0 = time.perf_counter()
warnings.simplefilter("ignore")
runpy.run_path(script, run_name="__main__")
t1 = time.perf_counter()
for num in plt.get_fignums():
    plt.figure(num).savefig(os.path.join(outdir, f"fig{num}.png"))
t2 = time.perf_counter()
print("TIMES", json.dumps([t1 - t0, t2 - t1]))
''' % {"root": os.path.abspath(os.path.join(here, "..", ".."))}


def run(impl, script, outdir, timeout):
    os.makedirs(outdir, exist_ok=True)
    for f in glob.glob(os.path.join(outdir, "*")):
        os.remove(f)
    rc = os.path.join(here, "..", "mplrc")
    try:
        p = subprocess.run([sys.executable, "-c", RUNNER, impl, script, outdir, rc],
                           capture_output=True, text=True, errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, "timeout"
    if p.returncode:
        lines = [l for l in p.stderr.strip().splitlines() if l.strip()]
        return None, lines[-1][:160] if lines else "failed"
    t = [l for l in p.stdout.splitlines() if l.startswith("TIMES ")]
    return (sum(json.loads(t[-1][6:])) if t else 0.0), None


def compare(dm, dr):
    problems = []
    for f in sorted(glob.glob(os.path.join(dm, "*.png"))):
        b = os.path.join(dr, os.path.basename(f))
        if not os.path.exists(b):
            problems.append(f"{os.path.basename(f)} missing")
            continue
        ia = np.asarray(Image.open(f).convert("RGBA"))
        ib = np.asarray(Image.open(b).convert("RGBA"))
        if ia.shape != ib.shape:
            problems.append(f"{os.path.basename(f)} shape {ia.shape[:2]} vs {ib.shape[:2]}")
            continue
        n = int(np.any(ia != ib, axis=2).sum())
        if n:
            problems.append(f"{os.path.basename(f)} {n}px")
    extra = set(map(os.path.basename, glob.glob(os.path.join(dr, "*.png")))) - set(
        map(os.path.basename, glob.glob(os.path.join(dm, "*.png"))))
    problems += [f"{e} extra" for e in sorted(extra)]
    return problems


def one(script, gallery, timeout):
    name = os.path.relpath(script, gallery)[:-3].replace(os.sep, "__")
    dm, dr = os.path.join(OUT, name, "mpl"), os.path.join(OUT, name, "rs")
    tm, em = run("mpl", script, dm, timeout)
    if em:
        return name, "SKIP", f"mpl: {em}", tm, None
    tr, er = run("rs", script, dr, timeout)
    if er:
        return name, "ERR", f"rs: {er}", tm, tr
    problems = compare(dm, dr)
    return name, ("DIFF" if problems else "OK"), "; ".join(problems), tm, tr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gallery")
    ap.add_argument("patterns", nargs="*")
    ap.add_argument("-j", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--timeout", type=float, default=120)
    a = ap.parse_args()
    scripts = sorted(glob.glob(os.path.join(a.gallery, "**", "*.py"), recursive=True))
    if a.patterns:
        scripts = [s for s in scripts if any(p in s for p in a.patterns)]
    counts = {}
    tot = [0.0, 0.0]
    with cf.ThreadPoolExecutor(a.j) as ex:
        for name, status, msg, tm, tr in ex.map(lambda s: one(s, a.gallery, a.timeout), scripts):
            counts[status] = counts.get(status, 0) + 1
            if status == "OK":
                tot[0] += tm
                tot[1] += tr
            print(f"{status:5s} {name:70s} {msg}", flush=True)
    print("summary:", counts, f"OK cases: mpl {tot[0]:.1f}s rs {tot[1]:.1f}s")


if __name__ == "__main__":
    main()
