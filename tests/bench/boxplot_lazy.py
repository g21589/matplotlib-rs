"""Benchmark boxplots whose artists are created when they are needed.

Run from the project root: python tests/bench/boxplot_lazy.py --boxes 5400 50000

One ``ax.boxplot`` call of N boxes (100 samples each, patch artists, fliers)
in a 3600 x 1200 pixel figure.  Every variant runs in fresh subprocesses,
one after the other, alternating the order between rounds:

- ``lazy``: matplotlib_rs as it is (the artists are created when read);
- ``eager``: ``MPLRS_LAZY=0``, the artists are created by ``boxplot``;
- ``lazy-1-thread``: ``MPLRS_THREADS=1``;
- ``base``: another matplotlib_rs (``--base-source``: its ``python`` directory
  with a built extension), e.g. the previous release;
- ``mpl``: Matplotlib (``--mpl``; slow: about 2.5 ms per box).

A worker measures, in this order,

- ``build``: figure, axes and the ``boxplot`` call;
- ``first_draw`` and ``redraw`` (the median of two): ``canvas.draw()``;
- ``save_png``: ``savefig`` (another draw, compression and writing);
- ``rss_after_draw_mib``: the peak resident memory so far;
- ``read_boxes``: reading every box (``get_facecolor``), which creates the
  artists of that component, and ``redraw_after_read``;
- ``read_all``: ``ax.get_children()``, which creates the other artists, and
  ``redraw_after_read_all`` (the figure is unchanged, the artists exist);
- ``change_boxes``: a new face colour for every box, and ``redraw_changed``
  (the first draw afterwards, which hands the artists to the renderer one by
  one) and ``redraw_one_by_one`` (the following one);
- ``peak_rss_mib``: the peak resident memory of the process.

The canvases of all the variants must be identical after the first draw and
after the change; the script exits with status 1 otherwise.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PHASES = ("build", "first_draw", "build_and_first_draw", "redraw", "save_png", "read_boxes",
          "redraw_after_read", "read_all", "redraw_after_read_all", "change_boxes",
          "redraw_changed", "redraw_one_by_one")
VARIANTS = {
    "lazy": {},
    "eager": {"MPLRS_LAZY": "0"},
    "lazy-1-thread": {"MPLRS_THREADS": "1"},
    "base": {},
    "mpl": {},
}


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--boxes", type=int, nargs="+", default=[5400, 50000])
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS),
                        default=["lazy", "eager", "lazy-1-thread"])
    parser.add_argument("--base-source", type=Path,
                        help="python directory of another matplotlib_rs (adds the 'base' variant)")
    parser.add_argument("--mpl", action="store_true", help="add the 'mpl' variant")
    parser.add_argument("--mpl-max-boxes", type=int, default=10 ** 9,
                        help="largest N the 'mpl' variant runs")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--output", type=Path, default=ROOT / "tests/output/boxplot-lazy")
    parser.add_argument("--worker", nargs=3, metavar=("VARIANT", "BOXES", "SOURCE"),
                        help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def worker(args):
    import resource
    variant, boxes, source = args.worker
    boxes = int(boxes)
    if variant != "mpl":
        sys.path.insert(0, source)
        import matplotlib_rs
        matplotlib_rs.install()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    matplotlib.rc_file(ROOT / "tests/mplrc/matplotlibrc")

    rng = np.random.default_rng(args.seed)
    locations = 2 * np.sin(np.linspace(0, 12 * np.pi, boxes)) + rng.normal(0, 0.3, boxes)
    data = rng.normal(locations, rng.uniform(0.5, 1.3, boxes), (args.samples, boxes))
    colors = plt.cm.viridis(np.linspace(0, 1, boxes))

    def peak():
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / (1024 * 1024 if sys.platform == "darwin" else 1024)

    def image():
        return hashlib.sha256(np.asarray(fig.canvas.buffer_rgba()).tobytes()).hexdigest()

    seconds = {}
    clock = time.perf_counter

    def timed(name, fn):
        t = clock()
        out = fn()
        seconds[name] = clock() - t
        return out

    def build():
        fig, ax = plt.subplots(figsize=(36, 12), dpi=100)
        result = ax.boxplot(data, widths=0.65, patch_artist=True, manage_ticks=False,
                            boxprops={"linewidth": 0.45}, whiskerprops={"linewidth": 0.45},
                            capprops={"linewidth": 0.45}, medianprops={"linewidth": 0.45},
                            flierprops={"marker": ".", "markersize": 1.1})
        ax.set_xlim(0, boxes + 1)
        ax.set_title(f"{boxes:,} boxplots")
        return fig, ax, result

    fig, ax, result = timed("build", build)
    timed("first_draw", fig.canvas.draw)
    seconds["build_and_first_draw"] = seconds["build"] + seconds["first_draw"]
    first = image()
    redraws = []
    for _ in range(2):
        t = clock()
        fig.canvas.draw()
        redraws.append(clock() - t)
    seconds["redraw"] = statistics.median(redraws)
    timed("save_png", lambda: fig.savefig(args.output / f"worker-{variant}-{boxes}.png"))
    rss_after_draw = peak()

    timed("read_boxes", lambda: [box.get_facecolor() for box in result["boxes"]])
    timed("redraw_after_read", fig.canvas.draw)
    timed("read_all", ax.get_children)
    timed("redraw_after_read_all", fig.canvas.draw)
    unchanged = image()

    def change():
        for box, color in zip(result["boxes"], colors):
            box.set_facecolor(color)

    timed("change_boxes", change)
    timed("redraw_changed", fig.canvas.draw)
    timed("redraw_one_by_one", fig.canvas.draw)
    print(json.dumps({
        "variant": variant, "boxes": boxes, "seconds": seconds,
        "rss_after_draw_mib": rss_after_draw, "peak_rss_mib": peak(),
        "first_sha256": first, "unchanged_sha256": unchanged, "changed_sha256": image(),
        "data_sha256": hashlib.sha256(data.tobytes()).hexdigest(),
        "children": len(ax._children), "lines": len(ax.lines), "patches": len(ax.patches),
        "numpy_version": np.__version__, "python_version": platform.python_version(),
        "matplotlib_api_version": matplotlib.__version__,
    }))


def run(args, variant, boxes):
    source = args.base_source if variant == "base" else ROOT / "python"
    env = {k: v for k, v in os.environ.items() if k not in ("MPLRS_LAZY", "MPLRS_THREADS")}
    env.update(VARIANTS[variant])
    cmd = [sys.executable, str(Path(__file__).resolve()), "--samples", str(args.samples),
           "--seed", str(args.seed), "--output", str(args.output),
           "--worker", variant, str(boxes), str(Path(source).resolve())]
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout, env=env)
    if done.returncode:
        raise RuntimeError(f"{variant} with {boxes} boxes exited {done.returncode}:\n{done.stderr}")
    row = json.loads(done.stdout.strip().splitlines()[-1])
    s = row["seconds"]
    print(f"  {variant:14s} build {s['build']:.3f}  draw {s['first_draw']:.3f}  "
          f"redraw {s['redraw']:.3f}  read all {s['read_all']:.3f}  "
          f"changed {s['redraw_changed']:.3f}  peak {row['peak_rss_mib']:.0f} MiB", flush=True)
    return row


def main():
    args = arguments()
    args.output = args.output.resolve()
    if args.worker:
        return worker(args)
    args.output.mkdir(parents=True, exist_ok=True)
    result_file = args.output / "results.json"
    if result_file.exists():
        raise SystemExit(f"{result_file} already exists; select a new --output directory")
    variants = list(args.variants)
    if args.base_source and "base" not in variants:
        variants.append("base")
    if args.mpl and "mpl" not in variants:
        variants.append("mpl")
    payload = {
        "parameters": {k: str(v) if isinstance(v, Path) else v
                       for k, v in vars(args).items() if k != "worker"},
        "python": sys.version, "platform": platform.platform(), "cpus": os.cpu_count(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
        "runs": [], "summary": {},
    }
    ok = True
    for boxes in args.boxes:
        print(f"{boxes} boxes", flush=True)
        active = [v for v in variants if v != "mpl" or boxes <= args.mpl_max_boxes]
        rows = []
        for i in range(args.rounds + 1):  # (round 0 warms the file cache up)
            order = active[i % len(active):] + active[:i % len(active)]
            for variant in order:
                # (Matplotlib takes minutes: once is enough)
                if variant == "mpl" and i != 1:
                    continue
                row = run(args, variant, boxes)
                row["round"] = i
                if i:
                    rows.append(row)
                    payload["runs"].append(row)
                    result_file.write_text(json.dumps(payload, indent=2) + "\n")
        summary = {}
        for variant in active:
            subset = [r for r in rows if r["variant"] == variant]
            summary[variant] = {
                phase: {"median": statistics.median(r["seconds"][phase] for r in subset),
                        "min": min(r["seconds"][phase] for r in subset),
                        "max": max(r["seconds"][phase] for r in subset)}
                for phase in PHASES}
            for key in ("rss_after_draw_mib", "peak_rss_mib"):
                summary[variant][key] = statistics.median(r[key] for r in subset)
        identical = {key: len({r[key] for r in rows}) == 1
                     for key in ("data_sha256", "first_sha256", "unchanged_sha256", "changed_sha256")}
        identical["first_is_unchanged"] = all(r["first_sha256"] == r["unchanged_sha256"] for r in rows)
        counts = {(r["children"], r["lines"], r["patches"]) for r in rows}
        identical["artists"] = counts == {(7 * boxes, 6 * boxes, boxes)}
        ok = ok and all(identical.values())
        payload["summary"][str(boxes)] = {"variants": summary, "identical": identical}
        result_file.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"  identical: {identical}", flush=True)
        reference = "base" if "base" in summary else "eager" if "eager" in summary else None
        if reference and "lazy" in summary:
            for phase in ("build", "first_draw", "build_and_first_draw", "redraw", "save_png",
                          "redraw_one_by_one"):
                a, b = summary[reference][phase]["median"], summary["lazy"][phase]["median"]
                print(f"  {phase:22s} {reference} {a:.3f} s  lazy {b:.3f} s  x {a / b:.2f}", flush=True)
            a, b = summary[reference]["peak_rss_mib"], summary["lazy"]["rss_after_draw_mib"]
            print(f"  memory                 {reference} peak {a:.0f} MiB  lazy after the draws {b:.0f} MiB",
                  flush=True)
    if not ok:
        raise SystemExit("the variants differ")


if __name__ == "__main__":
    main()
