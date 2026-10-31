"""The batched Rectangle creation of bar (axes/_bar.py) against the ordinary
per-bar loop and against Matplotlib: canvases, data limits and the state of
every bar, plus independently mutable bars."""
from pathlib import Path
import re
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import matplotlib as mpl
import matplotlib.pyplot as mplt
import matplotlib_rs as rs
import matplotlib_rs.pyplot as rplt
from matplotlib_rs.axes import _bar

RC = Path(__file__).resolve().parents[1] / "mplrc/matplotlibrc"
N = 300


def configurations():
    rng = np.random.default_rng(31)
    h = rng.uniform(1, 10, N)
    s1, s2, s3 = rng.normal(size=(3, N))
    widths, lefts = rng.uniform(0.1, 1, N), rng.uniform(-1, 1, N)
    signed = rng.uniform(-5, 10, N)
    d1, d2, d3 = rng.normal(size=10000), rng.normal(size=(3000, 3)), rng.normal(size=(3000, 2))
    e, logs = rng.exponential(size=5000), rng.uniform(0.1, 1000, N)
    return {
        "basic": lambda ax: ax.bar(np.arange(N), h),
        "signed": lambda ax: ax.bar(np.arange(N) * 0.37 - 20, s1, width=widths, bottom=s2),
        "edge": lambda ax: ax.bar(np.arange(N), h, align="edge", width=-0.5, color="C3",
                                  edgecolor="k", linewidth=0.7),
        "cycled colors": lambda ax: ax.bar(np.arange(N), h, color=["r", "g", "b"],
                                           edgecolor=["k", "y"], alpha=0.5),
        "barh": lambda ax: ax.barh(np.arange(N), signed, left=lefts,
                                   height=0.6, color="tab:green", edgecolor="k", linewidth=0.2),
        "integers": lambda ax: ax.bar(np.arange(N), np.arange(N) % 7, width=1),
        "empty bars": lambda ax: ax.bar(np.arange(N), np.where(np.arange(N) % 3 == 0, 0, 2.5),
                                        width=np.where(np.arange(N) % 5 == 0, 0, 0.9)),
        "hist": lambda ax: ax.hist(d1, bins=500),
        "hist datasets": lambda ax: ax.hist(d2, bins=100),
        "hist stacked": lambda ax: ax.hist(d3, bins=80, stacked=True, alpha=0.6),
        "hist horizontal log": lambda ax: ax.hist(e, bins=200, orientation="horizontal", log=True),
        "log": lambda ax: ax.bar(np.arange(N), logs, log=True),
        "properties": lambda ax: ax.bar(np.arange(N), h, zorder=3, linestyle="--", edgecolor="r",
                                        linewidth=1.5, antialiased=False, hatch="//"),
        "twice": lambda ax: (ax.bar(np.arange(N), h), ax.bar(np.arange(N), h / 2)),
        "per-bar colors": lambda ax: ax.bar(np.arange(N), h, color=rplt.cm.viridis(h / 10)
                                            if ax.__module__.startswith("matplotlib_rs")
                                            else mplt.cm.viridis(h / 10)),
    }


def state(fig, ax):
    """Canvas, limits and a description of every patch (without addresses)."""
    fig.canvas.draw()
    patches = []
    for p in ax.patches:
        skip = {"_h", "_callbacks", "_rs_switch", "_remove_method", "_clippath", "clipbox",
                "stale_callback", "_original_facecolor", "_original_edgecolor"}
        attrs = sorted((k, repr(v)) for k, v in p.__dict__.items() if k not in skip)
        props = sorted((k, repr(v)) for k, v in p._rs_props().items() if k != "clip_path")
        patches.append(re.sub(r" at 0x[0-9a-f]+", "", repr((attrs, props))))
    return (np.asarray(fig.canvas.buffer_rgba()).copy(), ax.dataLim.get_points().tolist(),
            ax.dataLim.minpos.tolist(), ax.viewLim.get_points().tolist(), patches,
            len(ax._children))


class BarBatchTests(unittest.TestCase):
    def render(self, module, pyplot, plot, batched=True):
        module.rc_file(RC)
        old = _bar.MIN_BARS
        _bar.MIN_BARS = old if batched else 10 ** 9
        try:
            fig, ax = pyplot.subplots(figsize=(6, 4), dpi=80)
            plot(ax)
            return fig, ax
        finally:
            _bar.MIN_BARS = old

    def test_same_as_loop_and_matplotlib(self):
        for name, plot in configurations().items():
            with self.subTest(name):
                fig, ax = self.render(rs, rplt, plot)
                batched = state(fig, ax)
                rplt.close(fig)
                fig, ax = self.render(rs, rplt, plot, batched=False)
                loop = state(fig, ax)
                rplt.close(fig)
                np.testing.assert_array_equal(batched[0], loop[0])
                self.assertEqual(batched[1:], loop[1:])
                fig, ax = self.render(mpl, mplt, plot)
                fig.canvas.draw()
                reference = np.asarray(fig.canvas.buffer_rgba()).copy()
                self.assertEqual(ax.dataLim.get_points().tolist(), batched[1])
                self.assertEqual(ax.viewLim.get_points().tolist(), batched[3])
                mplt.close(fig)
                np.testing.assert_array_equal(batched[0], reference)

    def test_patch_transform(self):
        rng = np.random.default_rng(5)
        for x0, y0, w, h in zip(*(rng.normal(size=(4, 500)) * [[1], [1e3], [1e-3], [10]])):
            r = rs.patches.Rectangle((x0, y0), w, h)
            x1, y1 = x0 + w, y0 + h
            np.testing.assert_array_equal(r.get_patch_transform().get_matrix()[:2],
                                          [[x1 - x0, 0, x0], [0, y1 - y0, y0]])

    def test_bars_are_independent(self):
        def mutate(module, pyplot, batched):
            fig, ax = self.render(module, pyplot, lambda ax: ax.bar(np.arange(N), np.ones(N)), batched)
            bars = ax.containers[0]
            bars[3].set_height(4)
            bars[4].set_facecolor("red")
            bars[5].set_xy((10.5, 2))
            bars[6].remove()
            bars[7].sticky_edges.y[:] = []
            bars[8].set_visible(False)
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()
            out = np.asarray(fig.canvas.buffer_rgba()).copy()
            pyplot.close(fig)
            return out, bars

        batched, bars = mutate(rs, rplt, True)
        np.testing.assert_array_equal(batched, mutate(rs, rplt, False)[0])
        np.testing.assert_array_equal(batched, mutate(mpl, mplt, True)[0])
        self.assertEqual(bars[2].get_height(), 1)
        self.assertEqual(bars[9].sticky_edges.y, [0])
        self.assertIsNot(bars[9].sticky_edges.y, bars[10].sticky_edges.y)
        self.assertEqual(len({id(b._callbacks) for b in bars}), N)
        self.assertEqual(len({b._h.ident() for b in bars}), N)

    def test_fallbacks(self):
        # unit data, labels per bar and transforms use the ordinary loop
        fig, axs = rplt.subplots(4)
        calls = []
        original = _bar.rectangles

        def spy(*args, **kwargs):
            result = original(*args, **kwargs)
            calls.append(result is not None)
            return result

        _bar.rectangles = spy
        try:
            axs[0].bar([f"c{i}" for i in range(50)], np.ones(50))
            axs[1].bar(np.arange(50), np.ones(50), label=[f"b{i}" for i in range(50)])
            axs[2].bar(np.arange(50), np.ones(50), transform=axs[2].transAxes)
            axs[3].bar(np.arange(50), np.ones(50))
        finally:
            _bar.rectangles = original
        self.assertEqual(calls, [False, False, False, True])
        rplt.close(fig)


if __name__ == "__main__":
    unittest.main()
