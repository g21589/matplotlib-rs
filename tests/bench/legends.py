"""The cost of legends in an ordinary figure, on Matplotlib and matplotlib-rs.

Run from the project root (Matplotlib 3.11.2 and the release extension
installed):

    python tests/bench/legends.py [--output DIR] [--rounds 5] [--base-source DIR]

The figure is the one of ``phases.py``: 2 x 2 Axes, each with two lines of
1,000 points, a scatter, a title, a grid, and

- ``best``: a legend at the default ``loc="best"``,
- ``fixed``: a legend at ``loc="upper right"``,
- ``none``: no legend.

After ``tight_layout()``, every ``savefig`` draws the figure twice (once to
lay it out, once into the canvas), as in Matplotlib.  Each worker process
saves the figure to memory once, then ``--saves`` more times (the title of
one Axes changes before every save); its result is the median of these.  The
printed times are the medians over ``--rounds`` fresh processes per
implementation, started in alternating order.

The RGBA canvases of matplotlib-rs must be identical to Matplotlib's; the
script exits with status 1 when they are not.  With ``--base-source DIR``
another matplotlib-rs (its ``python`` directory, with a built extension) is
measured too.
"""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
MODES = ("best", "fixed", "none")


def worker(args):
    if args.impl in ("rs", "base"):
        sys.path.insert(0, str(args.python_source.resolve()))
        import matplotlib_rs
        matplotlib_rs.install()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    matplotlib.rc_file(ROOT / "tests/mplrc/matplotlibrc")

    x = np.linspace(0, 10, 1000)
    fig, axs = plt.subplots(2, 2, figsize=(8, 6))
    for i, ax in enumerate(axs.flat):
        ax.plot(x, np.sin(x + i), label="sin")
        ax.plot(x, np.cos(x + i), "--", label="cos")
        ax.scatter(x[::50], np.sin(x[::50]), c=x[::50])
        ax.set_title(f"panel {i}")
        ax.set_xlabel("x")
        if args.mode == "best":
            ax.legend()
        elif args.mode == "fixed":
            ax.legend(loc="upper right")
        ax.grid(True)
    fig.tight_layout()
    fig.savefig(io.BytesIO(), format="png")
    times = []
    for k in range(args.saves):
        axs[0, 0].set_title(f"frame {k}")
        t = time.perf_counter()
        fig.savefig(io.BytesIO(), format="png")
        times.append(time.perf_counter() - t)
    t = time.perf_counter()
    fig.canvas.draw()
    draw = time.perf_counter() - t
    buf = np.asarray(fig.canvas.buffer_rgba())
    print(json.dumps({"save": statistics.median(times), "draw": draw,
                      "canvas": hashlib.sha256(buf.tobytes()).hexdigest()}))


def run_worker(impl, mode, args):
    source = args.base_source if impl == "base" else args.python_source
    cmd = [sys.executable, __file__, "--impl", impl, "--mode", mode, "--saves", str(args.saves),
           "--python-source", str(source)]
    p = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, MPLBACKEND="Agg"))
    if p.returncode:
        raise SystemExit(f"{impl} {mode}: {(p.stderr.strip().splitlines() or ['failed'])[-1]}")
    return json.loads(p.stdout.strip().splitlines()[-1])


def main(args):
    impls = ["mpl", "rs"] + (["base"] if args.base_source else [])
    results = {mode: {impl: [] for impl in impls} for mode in MODES}
    for r in range(args.rounds):
        for mode in MODES:
            for impl in (impls if r % 2 == 0 else impls[::-1]):
                results[mode][impl].append(run_worker(impl, mode, args))
    summary, failed = {}, False
    print(f"{'legends':8}" + "".join(f"{impl + ' save':>12}{impl + ' draw':>12}" for impl in impls) + "   canvases")
    for mode in MODES:
        row = {impl: {key: statistics.median(run[key] for run in runs) for key in ("save", "draw")}
               for impl, runs in results[mode].items()}
        canvases = {impl: {run["canvas"] for run in runs} for impl, runs in results[mode].items()}
        same = canvases["rs"] == canvases["mpl"] and len(canvases["mpl"]) == 1
        failed |= not same
        summary[mode] = {"median_seconds": row, "identical_to_matplotlib": same}
        print(f"{mode:8}" + "".join(f"{1e3 * row[i]['save']:10.1f}ms{1e3 * row[i]['draw']:10.1f}ms" for i in impls)
              + ("   identical" if same else "   DIFFERENT"))
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "results.json").write_text(json.dumps(
            {"rounds": args.rounds, "saves": args.saves, "summary": summary, "runs": results}, indent=1))
    return 1 if failed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--impl", choices=("mpl", "rs", "base"), help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=MODES, help=argparse.SUPPRESS)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--saves", type=int, default=15)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--python-source", type=Path, default=ROOT / "python")
    parser.add_argument("--base-source", type=Path)
    arguments = parser.parse_args()
    sys.exit(worker(arguments) if arguments.impl else main(arguments))
