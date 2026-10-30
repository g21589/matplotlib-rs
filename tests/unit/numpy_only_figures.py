"""Figures exercising every built-in replacement of a dependency (see
test_numpy_only.py): constrained layout, contours, Delaunay triangulation,
dates, mathtext, images and boxplots, saved in four formats.

Usage: python numpy_only_figures.py outdir
"""
import datetime
from pathlib import Path
import sys

import matplotlib_rs

matplotlib_rs.install()

import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RC = Path(__file__).resolve().parents[1] / "mplrc/matplotlibrc"


def figures():
    rng = np.random.default_rng(20261030)
    fig, axs = plt.subplots(2, 3, figsize=(10, 6), layout="constrained")
    axs[0, 0].boxplot(rng.normal(size=(60, 8)), patch_artist=True, notch=True)
    z = rng.random((12, 15)).cumsum(0)
    cs = axs[0, 1].contourf(z, levels=7)
    axs[0, 1].contour(z, levels=7, colors="k", linewidths=0.5)
    fig.colorbar(cs, ax=axs[0, 1])
    x, y = rng.random(40), rng.random(40)
    axs[0, 2].tripcolor(x, y, np.sin(5 * x) * y)
    axs[0, 2].tricontour(x, y, np.sin(5 * x) * y, levels=5, colors="w")
    days = [datetime.datetime(2026, 1, 1) + datetime.timedelta(days=3 * i) for i in range(40)]
    axs[1, 0].plot(days, np.cumsum(rng.normal(size=40)))
    axs[1, 0].set_title(r"$\sum_i \alpha_i^2 / \sqrt{\beta}$")
    axs[1, 1].imshow(rng.random((20, 25)), interpolation="nearest")
    axs[1, 2].imshow(np.dstack([rng.random((6, 7))] * 3 + [rng.random((6, 7))]))
    axs[1, 2].set_prop_cycle(color=["tab:red", "tab:green"])
    for i in range(2):
        axs[1, 2].plot([0, 6], [i, 5 - i], label=f"line {i}")
    axs[1, 2].legend()
    fig.suptitle("matplotlib_rs with NumPy only")
    yield "overview", fig

    fig = plt.figure(figsize=(6, 4), layout="constrained")
    left, right = fig.subfigures(1, 2, width_ratios=[2, 1])
    left.subplot_mosaic("AB;CC")
    right.subplots(3, 1, sharex=True)
    right.supylabel("supylabel")
    yield "nested_layout", fig


def main(outdir):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    matplotlib.rc_file(RC)
    matplotlib.rcParams["svg.hashsalt"] = "numpy-only"
    for name, fig in figures():
        for ext in ("png", "svg", "pdf", "eps"):
            fig.savefig(out / f"{name}.{ext}", metadata={"CreationDate": None}
                        if ext == "pdf" else None)
        plt.close(fig)
    image = plt.imread(out / "overview.png")
    plt.imsave(out / "imsave.png", image[::4, ::4])
    np.save(out / "imread.npy", image)


if __name__ == "__main__":
    main(sys.argv[1])
