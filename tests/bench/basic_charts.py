"""Large-data benchmark of common basic charts on Matplotlib and matplotlib-rs.

Run from the project root (Matplotlib 3.11.2 and the release extension
installed): python tests/bench/basic_charts.py [--cases line scatter ...]

Every case runs in fresh subprocesses, alternating which implementation
starts.  Data generation is excluded from the timings.  For each case the
worker measures

- ``build``: the plotting calls (figure creation included),
- ``draw``: the first ``canvas.draw()``,
- ``redraw``: a second ``canvas.draw()`` of the unchanged figure,
- ``save``: ``savefig()`` to PNG (another draw, compression and writing).

The RGBA canvases of both implementations must be identical; the script
exits with status 1 when a case differs or fails.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PHASES = ("build", "draw", "build_and_draw", "redraw", "save")


# --- cases: case(rng, n) generates the data and returns plot(ax) ----------

def _walk(rng, n):
    return {"x": np.arange(n, dtype=float), "y": np.cumsum(rng.normal(size=n))}


CASES = {}


def case(n):
    def register(fn):
        CASES[fn.__name__] = (n, fn)
        return fn
    return register


@case(1_000_000)
def line(rng, n):
    d = _walk(rng, n)
    return lambda ax: ax.plot(d["x"], d["y"], lw=0.8)


@case(20)
def lines_many(rng, n):
    # n lines of 100,000 points each
    ds = [_walk(rng, 100_000) for _ in range(n)]

    def plot(ax):
        for i, d in enumerate(ds):
            ax.plot(d["x"], d["y"] + 30 * i, lw=0.6)
    return plot


@case(200_000)
def line_markers(rng, n):
    d = _walk(rng, n)
    return lambda ax: ax.plot(d["x"], d["y"], "o-", ms=2, lw=0.5)


@case(1_000_000)
def line_dashed(rng, n):
    d = _walk(rng, n)
    return lambda ax: ax.plot(d["x"], d["y"], "--", lw=0.8)


@case(200_000)
def scatter(rng, n):
    x, y = rng.normal(size=(2, n))
    s, c = rng.uniform(1, 30, n), rng.uniform(size=n)
    return lambda ax: ax.scatter(x, y, s=s, c=c, alpha=0.5)


@case(1_000_000)
def scatter_uniform(rng, n):
    x, y = rng.normal(size=(2, n))
    return lambda ax: ax.scatter(x, y, s=1, c="tab:blue")


@case(20_000)
def bar(rng, n):
    h = rng.uniform(1, 10, n)
    return lambda ax: ax.bar(np.arange(n), h, width=0.8)


@case(20_000)
def barh(rng, n):
    w = rng.uniform(1, 10, n)
    return lambda ax: ax.barh(np.arange(n), w, height=0.8, color="tab:green",
                              edgecolor="k", linewidth=0.2)


@case(10_000_000)
def hist(rng, n):
    x = rng.normal(size=n)
    return lambda ax: ax.hist(x, bins=200)


@case(2_000_000)
def hist_many_bins(rng, n):
    x = rng.normal(size=n)
    return lambda ax: ax.hist(x, bins=10_000)


@case(2_000_000)
def hist_step(rng, n):
    x = rng.normal(size=(3, n)).T * [1, 1.5, 2]
    return lambda ax: ax.hist(x, bins=2_000, histtype="step")


@case(2_000_000)
def hist2d(rng, n):
    x, y = rng.normal(size=(2, n))
    return lambda ax: ax.hist2d(x, y, bins=500)


@case(1_000_000)
def hexbin(rng, n):
    x, y = rng.normal(size=(2, n))
    return lambda ax: ax.hexbin(x, y, gridsize=150)


@case(500)
def pie(rng, n):
    x = rng.uniform(1, 2, n)
    return lambda ax: ax.pie(x)


@case(4000)
def imshow(rng, n):
    z = rng.normal(size=(n, n))
    return lambda ax: ax.imshow(z)


@case(1000)
def pcolormesh(rng, n):
    z = rng.normal(size=(n, n))
    return lambda ax: ax.pcolormesh(z)


@case(1_000_000)
def fill_between(rng, n):
    d = _walk(rng, n)
    return lambda ax: ax.fill_between(d["x"], d["y"], d["y"] + 5, alpha=0.5)


@case(1_000_000)
def step(rng, n):
    d = _walk(rng, n)
    return lambda ax: ax.step(d["x"], d["y"], lw=0.8)


@case(20_000)
def errorbar(rng, n):
    d = _walk(rng, n)
    e = rng.uniform(0.5, 2, n)
    return lambda ax: ax.errorbar(d["x"], d["y"], yerr=e, fmt="o", ms=1, capsize=1, lw=0.4)


@case(20_000)
def stem(rng, n):
    y = rng.normal(size=n)
    return lambda ax: ax.stem(np.arange(n), y)


@case(200_000)
def stackplot(rng, n):
    ys = np.abs(rng.normal(size=(8, n))).cumsum(axis=1)
    return lambda ax: ax.stackplot(np.arange(n), ys)


@case(1000)
def contourf(rng, n):
    x = np.linspace(-3, 3, n)
    X, Y = np.meshgrid(x, x)
    Z = np.sin(X * 3) * np.cos(Y * 2) + 0.1 * rng.normal(size=X.shape)
    return lambda ax: ax.contourf(X, Y, Z, levels=20)


@case(1000)
def boxplot(rng, n):
    x = rng.normal(size=(200, n))
    return lambda ax: ax.boxplot(x)


@case(300)
def violinplot(rng, n):
    x = rng.normal(size=(2000, n))
    return lambda ax: ax.violinplot(x)


# --- worker ----------------------------------------------------------------

def worker(args):
    global np
    if args.impl == "rs":
        sys.path.insert(0, str(args.python_source.resolve()))
        import matplotlib_rs
        matplotlib_rs.install()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    matplotlib.rc_file(ROOT / "tests/mplrc/matplotlibrc")

    n0, fn = CASES[args.case]
    n = max(1, int(n0 * args.scale))
    plot = fn(np.random.default_rng(args.seed), n)  # data, not timed
    t0 = time.perf_counter()
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    plot(ax)
    t1 = time.perf_counter()
    fig.canvas.draw()
    t2 = time.perf_counter()
    fig.canvas.draw()
    t3 = time.perf_counter()
    out = Path(args.output) / f"{args.case}-{args.impl}.png"
    fig.savefig(out)
    t4 = time.perf_counter()
    buf = np.asarray(fig.canvas.buffer_rgba())
    np.save(Path(args.output) / f"{args.case}-{args.impl}.npy", buf)
    timings = {"build": t1 - t0, "draw": t2 - t1, "build_and_draw": t2 - t0,
               "redraw": t3 - t2, "save": t4 - t3}
    print(json.dumps({"n": n, "timings": timings,
                      "canvas": hashlib.sha256(buf.tobytes()).hexdigest()}))


def run_worker(impl, case, args):
    python = args.mpl_python if impl == "mpl" else args.rs_python
    cmd = [str(python), __file__, "--impl", impl, "--case", case, "--output", str(args.output),
           "--scale", str(args.scale), "--seed", str(args.seed),
           "--python-source", str(args.python_source)]
    env = dict(os.environ, MPLBACKEND="Agg")
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=args.timeout)
    if p.returncode:
        return None, (p.stderr.strip().splitlines() or ["failed"])[-1]
    return json.loads(p.stdout.strip().splitlines()[-1]), None


def compare(case, out):
    import numpy as np
    a = np.load(out / f"{case}-mpl.npy")
    b = np.load(out / f"{case}-rs.npy")
    if a.shape != b.shape:
        return f"shape {a.shape} != {b.shape}"
    d = np.any(a != b, axis=2)
    if d.any():
        return f"{int(d.sum())} px differ (max {int(np.abs(a.astype(int) - b.astype(int)).max())})"
    return None


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    cases = args.cases or list(CASES)
    results = {}
    failed = 0
    impls = [i for i in ("mpl", "rs") if i in args.impls]
    for case in cases:
        runs = {impl: [] for impl in impls}
        error = None
        for r in range(args.rounds):
            order = impls if r % 2 == 0 else impls[::-1]
            for impl in order:
                res, err = run_worker(impl, case, args)
                if err:
                    error = f"{impl}: {err}"
                    break
                runs[impl].append(res)
            if error:
                break
        if error:
            print(f"{case:18s} ERROR {error}", flush=True)
            failed += 1
            results[case] = {"error": error}
            continue
        med = {impl: {ph: statistics.median(r["timings"][ph] for r in runs[impl]) for ph in PHASES}
               for impl in impls}
        diff = compare(case, args.output) if len(impls) == 2 else None
        failed += diff is not None
        results[case] = {"n": runs[impls[0]][0]["n"], "median": med, "diff": diff,
                         "runs": {impl: [r["timings"] for r in runs[impl]] for impl in impls}}
        cells = []
        for ph in ("build", "draw", "build_and_draw", "save"):
            vals = [med[impl][ph] for impl in impls]
            cell = "/".join(f"{v:7.3f}" for v in vals)
            if len(vals) == 2:
                cell += f" ({vals[0] / vals[1]:6.1f}x)"
            cells.append(f"{ph} {cell}")
        print(f"{case:18s} {'OK  ' if diff is None else 'DIFF'} " + "  ".join(cells)
              + (f"  {diff}" if diff else ""), flush=True)
    (args.output / "results.json").write_text(json.dumps(results, indent=1))
    return 1 if failed else 0


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cases", nargs="*", choices=list(CASES))
    p.add_argument("--impls", nargs="*", default=["mpl", "rs"], choices=["mpl", "rs"])
    p.add_argument("--scale", type=float, default=1.0, help="multiply every data size")
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--seed", type=int, default=20261031)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--output", type=Path, default=ROOT / "tests/output/basic-charts")
    p.add_argument("--python-source", type=Path, default=ROOT / "python")
    p.add_argument("--mpl-python", type=Path, default=Path(sys.executable))
    p.add_argument("--rs-python", type=Path, default=Path(sys.executable))
    p.add_argument("--impl", choices=("mpl", "rs"), help=argparse.SUPPRESS)
    p.add_argument("--case", help=argparse.SUPPRESS)
    return p.parse_args(argv)


if __name__ == "__main__":
    a = arguments()
    if a.impl:
        worker(a)
    else:
        sys.exit(main(a))
