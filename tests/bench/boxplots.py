"""Benchmark 30 groups of 180 boxplots on Matplotlib and matplotlib-rs.

Run from the project root: python tests/bench/boxplots.py
Each implementation runs in a fresh subprocess, sequentially. Data generation
is excluded from plotting timings. PNG saving includes another complete draw.
"""

import argparse
import colorsys
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PHASES = ("import", "build", "first_draw", "build_and_first_draw", "redraw", "save_png")


def positive(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=positive, default=30)
    parser.add_argument("--boxes", type=positive, default=180, help="boxes per group")
    parser.add_argument("--samples", type=positive, default=100, help="samples per box")
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--rounds", type=positive, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--redraws", type=positive, default=2)
    parser.add_argument("--dpi", type=positive, default=100)
    parser.add_argument("--width", type=float, default=36)
    parser.add_argument("--height", type=float, default=12)
    parser.add_argument("--timeout", type=positive, default=300)
    parser.add_argument("--output", type=Path, default=ROOT / "tests/output/boxplots-5400")
    parser.add_argument("--python-source", type=Path, default=ROOT / "python",
                        help="Python package root, e.g. an extracted baseline wheel")
    parser.add_argument("--official-optimization", choices=("none", "statistics", "artists", "all"),
                        default="none", help="experimental optimization of official Matplotlib")
    parser.add_argument("--png-compress-level", type=int, choices=range(10),
                        help="use the same Pillow compression level for all implementations")
    parser.add_argument("--impl", choices=("mpl", "rs"), help=argparse.SUPPRESS)
    parser.add_argument("--image", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.seed < 0 or args.warmups < 0 or args.width <= 0 or args.height <= 0:
        parser.error("seed/warmups must be nonnegative and figure dimensions positive")
    return args


def worker(args):
    import resource

    start = time.perf_counter()
    if args.impl == "rs":
        sys.path.insert(0, str(args.python_source.resolve()))
        import matplotlib_rs
        matplotlib_rs.install()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import __version__ as pillow_version
    if args.impl == "mpl" and args.official_optimization != "none":
        import official_boxplot_fastpath
        official_boxplot_fastpath.install(
            artists=args.official_optimization in ("artists", "all"),
            statistics=args.official_optimization in ("statistics", "all"))
    imported = time.perf_counter()
    matplotlib.rc_file(ROOT / "tests/mplrc/matplotlibrc")

    # One shared seeded workload, with a different location/scale per box.
    rng = np.random.default_rng(args.seed)
    group_means = 2 * np.sin(np.linspace(0, 2 * np.pi, args.groups, endpoint=False))
    locations = group_means[:, None] + rng.normal(0, 0.3, (args.groups, args.boxes))
    scales = rng.uniform(0.5, 1.3, (args.groups, args.boxes))
    data = rng.normal(locations[..., None], scales[..., None],
                      (args.groups, args.boxes, args.samples))
    data_hash = hashlib.sha256(data.tobytes()).hexdigest()
    palette = [colorsys.hsv_to_rgb(i / args.groups, 0.72, 0.78) for i in range(args.groups)]
    gap = max(6, args.boxes // 10)
    centers = [i * (args.boxes + gap) + (args.boxes + 1) / 2 for i in range(args.groups)]

    cpu = time.process_time()
    begin = time.perf_counter()
    fig, ax = plt.subplots(figsize=(args.width, args.height), dpi=args.dpi)
    fig.subplots_adjust(left=0.045, right=0.99, bottom=0.13, top=0.90)
    for i, color in enumerate(palette):
        positions = i * (args.boxes + gap) + np.arange(1, args.boxes + 1)
        artists = ax.boxplot(
            data[i].T, positions=positions, widths=0.65, patch_artist=True,
            manage_ticks=False, showfliers=True, whis=1.5,
            boxprops={"facecolor": color, "edgecolor": color, "linewidth": 0.45},
            whiskerprops={"color": color, "linewidth": 0.45},
            capprops={"color": color, "linewidth": 0.45},
            medianprops={"color": "#252525", "linewidth": 0.45},
            flierprops={"marker": ".", "markersize": 1.1,
                        "markerfacecolor": color, "markeredgecolor": color},
        )
        assert len(artists["boxes"]) == args.boxes
    ax.set_xticks(centers, [f"G{i + 1:02d}" for i in range(args.groups)], fontsize=10)
    ax.set_xlim(-gap, (args.groups - 1) * (args.boxes + gap) + args.boxes + gap)
    ax.set_xlabel("Group (180 boxplots per group)" if args.boxes == 180
                  else f"Group ({args.boxes} boxplots per group)", fontsize=13)
    ax.set_ylabel("Random value (synthetic)", fontsize=13)
    ax.set_title(
        f"Boxplot stress test: {args.groups} groups × {args.boxes} boxes = "
        f"{args.groups * args.boxes:,} boxplots\n"
        f"{args.samples} samples per box · seed {args.seed} · one color per group",
        fontsize=17, pad=18,
    )
    ax.grid(axis="y", color="#dddddd", linewidth=0.5)
    ax.set_axisbelow(True)
    built = time.perf_counter()
    build_cpu = time.process_time() - cpu
    assert len(ax.patches) == args.groups * args.boxes
    paths_ready = sum(line._path is not None and not line._invalidx and not line._invalidy
                      for line in ax.lines)

    draw_begin = time.perf_counter()
    cpu = time.process_time()
    fig.canvas.draw()
    drawn = time.perf_counter()
    draw_cpu = time.process_time() - cpu
    rgba = np.asarray(fig.canvas.buffer_rgba())
    image_hash = hashlib.sha256(rgba.tobytes()).hexdigest()
    redraws, redraw_cpus = [], []
    for _ in range(args.redraws):
        t, cpu = time.perf_counter(), time.process_time()
        fig.canvas.draw()
        redraws.append(time.perf_counter() - t)
        redraw_cpus.append(time.process_time() - cpu)
    t, cpu = time.perf_counter(), time.process_time()
    save_options = ({"pil_kwargs": {"compress_level": args.png_compress_level}}
                    if args.png_compress_level is not None else {})
    fig.savefig(args.image, dpi=args.dpi, **save_options)
    save_time, save_cpu = time.perf_counter() - t, time.process_time() - cpu
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mib = rss / (1024 * 1024 if sys.platform == "darwin" else 1024)
    plt.close(fig)
    print(json.dumps({
        "impl": args.impl, "matplotlib_api_version": matplotlib.__version__,
        "python_source": str(args.python_source.resolve()),
        "numpy_version": np.__version__, "pillow_version": pillow_version,
        "python_version": platform.python_version(), "data_sha256": data_hash,
        "rgba_sha256": image_hash, "image": str(args.image),
        "shape": list(rgba.shape), "box_count": args.groups * args.boxes,
        "line_count": len(ax.lines), "peak_rss_mib": peak_mib,
        "official_optimization": args.official_optimization if args.impl == "mpl" else None,
        "rust_extension_loaded": "matplotlib_rs._mplrs" in sys.modules,
        "line_paths_ready_after_build": int(paths_ready),
        "pre_draw_check_seconds": draw_begin - built,
        "png_bytes": args.image.stat().st_size,
        "optimization_counts": (dict(official_boxplot_fastpath.COUNTS)
                                if args.impl == "mpl" and args.official_optimization != "none" else {}),
        "seconds": {"import": imported - start, "build": built - begin,
                    "first_draw": drawn - draw_begin,
                    "build_and_first_draw": built - begin + drawn - draw_begin,
                    "redraw": statistics.median(redraws), "save_png": save_time},
        "cpu_seconds": {"build": build_cpu, "first_draw": draw_cpu,
                        "build_and_first_draw": build_cpu + draw_cpu,
                        "redraw": statistics.median(redraw_cpus), "save_png": save_cpu},
        "redraw_seconds": redraws,
    }))


def run_one(args, impl, label):
    image = args.output / f"{label}-{impl}.png"
    cmd = [sys.executable, str(Path(__file__).resolve()), "--impl", impl,
           "--image", str(image), "--groups", str(args.groups), "--boxes", str(args.boxes),
           "--samples", str(args.samples), "--seed", str(args.seed),
           "--redraws", str(args.redraws), "--dpi", str(args.dpi),
           "--width", str(args.width), "--height", str(args.height)]
    cmd += ["--python-source", str(args.python_source)]
    cmd += ["--official-optimization", args.official_optimization]
    if args.png_compress_level is not None:
        cmd += ["--png-compress-level", str(args.png_compress_level)]
    print(f"START {label} {impl}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
    (args.output / f"{label}-{impl}.stderr.log").write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(f"{label} {impl} exited {result.returncode}:\n{result.stderr}")
    row = json.loads(result.stdout.strip().splitlines()[-1])
    row["round"] = label
    print(f"END {label} {impl}: build {row['seconds']['build']:.3f}s, "
          f"draw {row['seconds']['first_draw']:.3f}s, "
          f"save PNG {row['seconds']['save_png']:.3f}s, "
          f"peak RSS {row['peak_rss_mib']:.1f} MiB", flush=True)
    return row


def main():
    args = arguments()
    if args.impl:
        return worker(args)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    result_file = args.output / "results.json"
    if result_file.exists():
        raise SystemExit(f"{result_file} already exists; select a new --output directory")
    payload = {"parameters": {k: str(v) if isinstance(v, Path) else v
                              for k, v in vars(args).items() if k not in ("impl", "image")},
               "python": sys.version, "platform": platform.platform(),
               "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                      cwd=ROOT, text=True).strip(),
               "warmups": [], "runs": []}
    for i in range(args.warmups):
        for impl in ("mpl", "rs"):
            payload["warmups"].append(run_one(args, impl, f"warmup-{i + 1:02d}"))
            result_file.write_text(json.dumps(payload, indent=2) + "\n")
    for i in range(args.rounds):
        # Alternate which implementation goes first to reduce ordering effects.
        for impl in (("mpl", "rs") if i % 2 == 0 else ("rs", "mpl")):
            payload["runs"].append(run_one(args, impl, f"round-{i + 1:02d}"))
            result_file.write_text(json.dumps(payload, indent=2) + "\n")
    runs = payload["runs"]
    assert len({r["data_sha256"] for r in runs + payload["warmups"]}) == 1
    assert all(r["box_count"] == args.groups * args.boxes for r in runs)
    summary = {}
    for impl in ("mpl", "rs"):
        subset = [r for r in runs if r["impl"] == impl]
        summary[impl] = {phase: {"median": statistics.median(r["seconds"][phase] for r in subset),
                                 "min": min(r["seconds"][phase] for r in subset),
                                 "max": max(r["seconds"][phase] for r in subset)}
                         for phase in PHASES}
        summary[impl]["peak_rss_mib_median"] = statistics.median(r["peak_rss_mib"] for r in subset)
        shutil.copyfile(subset[0]["image"], args.output / ("matplotlib.png" if impl == "mpl" else "matplotlib-rs.png"))
    payload["summary"] = summary
    payload["speedup_median"] = {p: summary["mpl"][p]["median"] / summary["rs"][p]["median"] for p in PHASES}
    payload["all_canvas_rgba_identical"] = len({r["rgba_sha256"] for r in runs + payload["warmups"]}) == 1

    import numpy as np
    from PIL import Image
    a = np.asarray(Image.open(args.output / "matplotlib.png").convert("RGBA"))
    b = np.asarray(Image.open(args.output / "matplotlib-rs.png").convert("RGBA"))
    assert a.shape == b.shape
    differences = np.any(a != b, axis=2)
    payload["png_comparison"] = {"shape": list(a.shape), "different_pixels": int(differences.sum()),
                                  "max_channel_difference": int(np.abs(a.astype(int) - b.astype(int)).max())}
    result_file.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: payload[k] for k in ("summary", "speedup_median", "all_canvas_rgba_identical", "png_comparison")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
