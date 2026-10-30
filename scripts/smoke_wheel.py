"""Verify an installed wheel in a clean environment with NumPy as its only dependency."""
import argparse
import importlib.metadata
import importlib.util
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert importlib.metadata.version("matplotlib-rs") == args.version
    installed = {d.metadata["Name"].lower() for d in importlib.metadata.distributions()}
    extra = installed - {"matplotlib-rs", "numpy", "pip", "setuptools", "wheel"}
    assert not extra, f"Use a clean venv with only the wheel and NumPy installed, not {extra}"
    for name in ("matplotlib", "PIL", "kiwisolver", "contourpy", "fontTools", "cycler",
                 "pyparsing", "dateutil", "packaging"):
        assert importlib.util.find_spec(name) is None, f"{name} is installed"
    import matplotlib_rs
    matplotlib_rs.install()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), layout="constrained")
    axes[0].plot([1, 2, 3], [1, 4, 9], label=r"$y=x^2$")
    axes[0].legend()
    boxes = axes[1].boxplot(rng.normal(size=(100, 24)), patch_artist=True, notch=True,
                            showmeans=True)
    assert len(boxes["boxes"]) == 24
    boxes["boxes"][0].set_facecolor("red")
    boxes["medians"][0].set_linewidth(2)
    axes[2].contourf(rng.random((10, 10)).cumsum(0), levels=5)
    axes[2].tricontour(rng.random(30), rng.random(30), rng.random(30), colors="k")
    for extension in ("png", "svg", "pdf", "eps"):
        dest = args.output / f"smoke.{extension}"
        fig.savefig(dest)
        assert dest.stat().st_size > 100
    image = plt.imread(args.output / "smoke.png")
    assert image.shape == (300, 1000, 4), image.shape
    assert image[..., :3].min() < 1, "blank image"
    plt.close(fig)
    third = {m.split(".")[0] for m in sys.modules} - set(sys.stdlib_module_names)
    third -= {"numpy", "matplotlib_rs", "matplotlib", "__main__", "cython_runtime"}
    assert not {m for m in third if not m.startswith("_")}, third
    print(f"Installed wheel {args.version} with NumPy only: mutable boxplots, constrained "
          f"layout, contours and PNG/SVG/PDF/EPS passed")


if __name__ == "__main__":
    main()
