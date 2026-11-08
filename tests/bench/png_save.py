"""PNG encoding speed and size: Pillow (as matplotlib saves PNG) and matplotlib-rs.

Run from the project root with Pillow and the release extension installed:

    python tests/bench/png_save.py [--output DIR] [--rounds 9] [--base-source DIR]

The RGBA canvases of several figures are rendered once by matplotlib-rs; each
is then encoded, in fresh worker processes, by

- ``pillow``: ``PIL.Image.save`` with the PNG info and dpi matplotlib passes
  (zlib level 6, matplotlib's ``savefig``/``imsave`` path),
- ``rs-6`` / ``rs-1``: the Rust writer at zlib level 6 (the default) and 1,
  on all threads (``MPLRS_THREADS`` unset, at most 8) and on one
  (``MPLRS_THREADS=1``),
- ``base``: with ``--base-source``, the ``_mplrs.encode_png`` of another
  matplotlib-rs (its ``python`` directory, built), e.g. 0.0.4's ``png``-crate
  encoder at the level its ``savefig`` used.

Every file is decoded and must give the canvas back.  The medians of
``--rounds`` encodings (after one warm-up) are printed and written to
``DIR/results.json``.
"""

import argparse
import io
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
METADATA = {"Software": "Matplotlib version3.11.2, https://matplotlib.org/"}


def figures():
    """(name, dpi, draw(plt)) of the benchmarked figures."""
    def grid(plt, rng):
        fig, axs = plt.subplots(2, 2, figsize=(8, 6))
        x = np.linspace(0, 10, 2000)
        axs[0, 0].plot(x, np.sin(x))
        axs[0, 0].set_title("line")
        axs[0, 1].scatter(rng.random(2000), rng.random(2000), c=rng.random(2000), s=5)
        axs[1, 0].imshow(rng.random((100, 100)))
        axs[1, 1].hist(rng.normal(size=10000), bins=50)
        return fig

    def line(plt, rng):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9], marker="o", label="y = x²")
        ax.legend()
        ax.set_title("simple")
        return fig

    def field(plt, rng):
        y, x = np.mgrid[-3:3:600j, -3:3:800j]
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(np.sin(x * 3) * np.cos(y * 2) + 0.1 * x, interpolation="bilinear")
        fig.colorbar(im)
        return fig

    def contourf(plt, rng):
        y, x = np.mgrid[-3:3:300j, -3:3:400j]
        z = np.sin(x * 2) * np.cos(y * 3) + 0.2 * x
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.contourf(x, y, z, 20)
        ax.contour(x, y, z, 20, colors="k", linewidths=0.5)
        return fig

    def hatch(plt, rng):
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, h in enumerate(["/", "\\\\", "x", "o", ".", "*"]):
            ax.bar(i, i + 1, hatch=h, fill=False)
        return fig

    def subplots9(plt, rng):
        fig, axs = plt.subplots(3, 3, figsize=(10, 8))
        for ax in axs.flat:
            ax.plot(rng.normal(size=(50, 3)).cumsum(0))
            ax.set(xlabel="time [s]", ylabel="value", title="a title")
            ax.legend(["a", "b", "c"])
            ax.grid(True)
        fig.tight_layout()
        return fig

    def surface3d(plt, rng):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(projection="3d")
        X, Y = np.meshgrid(np.linspace(-3, 3, 60), np.linspace(-3, 3, 60))
        ax.plot_surface(X, Y, np.sin(np.hypot(X, Y)), cmap="coolwarm")
        return fig

    def photo(plt, rng):
        fig, ax = plt.subplots(figsize=(8, 6))
        img = np.clip(np.cumsum(rng.normal(size=(300, 400, 3)), axis=1) / 40 + 0.5, 0, 1)
        ax.imshow(img, interpolation="bicubic")
        ax.axis("off")
        return fig

    return [("line", 100, line), ("grid", 100, grid), ("grid", 300, grid),
            ("field", 200, field), ("contourf", 150, contourf), ("hatch", 150, hatch),
            ("subplots9", 120, subplots9), ("surface3d", 150, surface3d),
            ("photo", 150, photo)]


def render(out):
    """Write the canvases to *out* (one .npy per figure); return their names."""
    sys.path.insert(0, str(ROOT / "python"))
    import matplotlib_rs.pyplot as plt
    names = []
    for name, dpi, draw in figures():
        with plt.rc_context({"figure.dpi": dpi}):
            fig = draw(plt, np.random.default_rng(0))
        fig.canvas.draw()
        key = f"{name}-{dpi}dpi"
        np.save(out / f"{key}.npy", np.asarray(fig.canvas.buffer_rgba()))
        plt.close(fig)
        names.append(key)
    return names


def worker(args):
    """Encode the canvases with one implementation; print a JSON result."""
    if args.impl == "pillow":
        import PIL.Image
        import PIL.PngImagePlugin

        def encode(rgba, dpi):
            info = PIL.PngImagePlugin.PngInfo()
            for k, v in METADATA.items():
                info.add_text(k, v)
            buf = io.BytesIO()
            PIL.Image.fromarray(rgba).save(buf, format="png", pnginfo=info, dpi=(dpi, dpi))
            return buf.getvalue()
    else:
        sys.path.insert(0, str(Path(args.source) / "python"))
        from matplotlib_rs import _mplrs
        level = args.level
        items = list(METADATA.items())

        def encode(rgba, dpi):
            try:
                return _mplrs.encode_png(rgba, (dpi, dpi), items, level)
            except TypeError:  # matplotlib-rs 0.0.4: (rgba, dpi, meta, compression)
                return _mplrs.encode_png(rgba, float(dpi), items, level)

    import PIL.Image
    results = {}
    for path in sorted(Path(args.data).glob("*.npy")):
        rgba = np.load(path)
        dpi = int(path.stem.rsplit("-", 1)[1][:-3])
        data = encode(rgba, dpi)
        decoded = np.asarray(PIL.Image.open(io.BytesIO(data)).convert("RGBA"))
        assert np.array_equal(decoded, rgba), f"{args.impl}: {path.stem} decodes differently"
        times = []
        for _ in range(args.rounds):
            t = time.perf_counter()
            encode(rgba, dpi)
            times.append(time.perf_counter() - t)
        results[path.stem] = {"ms": 1e3 * statistics.median(times),
                              "min_ms": 1e3 * min(times), "bytes": len(data)}
    print(json.dumps(results))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--output", type=Path, default=ROOT / "tests/output/png-save")
    p.add_argument("--rounds", type=int, default=9)
    p.add_argument("--base-source", type=Path)
    p.add_argument("--base-level", type=int, default=1,
                   help="compression argument of the base encoder (0.0.4 savefig: 1)")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--impl", help=argparse.SUPPRESS)
    p.add_argument("--source", help=argparse.SUPPRESS)
    p.add_argument("--level", type=int, help=argparse.SUPPRESS)
    p.add_argument("--data", help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.worker:
        return worker(args)

    args.output.mkdir(parents=True, exist_ok=True)
    data = args.output / "canvases"
    data.mkdir(exist_ok=True)
    names = render(data)
    threads = os.environ.get("MPLRS_THREADS")
    runs = [("pillow", "pillow", None, {}),
            ("rs-6", "rs", 6, {}), ("rs-6 1 thread", "rs", 6, {"MPLRS_THREADS": "1"}),
            ("rs-1", "rs", 1, {}), ("rs-1 1 thread", "rs", 1, {"MPLRS_THREADS": "1"})]
    if args.base_source:
        runs.append(("base", "base", args.base_level, {}))
    results = {}
    for label, impl, level, env in runs:
        source = args.base_source if impl == "base" else ROOT
        cmd = [sys.executable, __file__, "--worker", "--impl", impl, "--source", str(source),
               "--level", str(level or 0), "--data", str(data), "--rounds", str(args.rounds)]
        out = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, **env})
        if out.returncode:
            sys.exit(f"{label} failed:\n{out.stderr}")
        results[label] = json.loads(out.stdout)

    labels = list(results)
    print(f"{'canvas':20s} {'pixels':>11s} " + " ".join(f"{l:>20s}" for l in labels))
    for name in names:
        h, w = np.load(data / f"{name}.npy", mmap_mode="r").shape[:2]
        cells = [f"{results[l][name]['ms']:7.2f} ms {results[l][name]['bytes'] / 1e3:8.1f} kB"
                 for l in labels]
        print(f"{name:20s} {f'{w}x{h}':>11s} " + " ".join(f"{c:>20s}" for c in cells))
    total = {l: sum(r["ms"] for r in results[l].values()) for l in labels}
    size = {l: sum(r["bytes"] for r in results[l].values()) for l in labels}
    print(f"{'total':32s} " + " ".join(
        f"{f'{total[l]:7.1f} ms {size[l] / 1e3:8.1f} kB':>20s}" for l in labels))
    print("speedup over Pillow: " + ", ".join(
        f"{l} {total['pillow'] / total[l]:.1f}x" for l in labels if l != "pillow"))
    (args.output / "results.json").write_text(json.dumps({
        "rounds": args.rounds, "cpu_count": os.cpu_count(), "MPLRS_THREADS": threads,
        "base_source": str(args.base_source) if args.base_source else None,
        "base_level": args.base_level if args.base_source else None,
        "results": results}, indent=1))


if __name__ == "__main__":
    main()
