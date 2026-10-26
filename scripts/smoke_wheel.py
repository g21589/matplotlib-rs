"""Verify an installed wheel in a clean environment without reference Matplotlib."""
import argparse
import importlib.metadata
import importlib.util
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert importlib.metadata.version("matplotlib-rs") == args.version
    assert importlib.util.find_spec("matplotlib") is None, "Use a clean venv without Matplotlib"
    import matplotlib_rs
    matplotlib_rs.install()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    args.output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), layout="constrained")
    axes[0].plot([1, 2, 3], [1, 4, 9], label=r"$y=x^2$")
    axes[0].legend()
    boxes = axes[1].boxplot(np.random.default_rng(42).normal(size=(100, 24)),
                            patch_artist=True, notch=True, showmeans=True)
    assert len(boxes["boxes"]) == 24
    boxes["boxes"][0].set_facecolor("red")
    boxes["medians"][0].set_linewidth(2)
    for extension in ("png", "svg", "pdf", "eps"):
        dest = args.output / f"smoke.{extension}"
        fig.savefig(dest)
        assert dest.stat().st_size > 100
    with Image.open(args.output / "smoke.png") as image:
        assert image.size == (800, 300)
        assert image.convert("RGB").getextrema() != ((255, 255),) * 3
    plt.close(fig)
    print(f"Installed wheel {args.version}: mutable boxplots and PNG/SVG/PDF/EPS passed")


if __name__ == "__main__":
    main()
