"""Boxplot artists created when they are needed (axes/_base_rs.py `_Pending`):
the canvases of Matplotlib whether the artists exist or not, the artists and
lists of an ordinary boxplot once they are read, and many lines and patches
drawn together (src/mpl/batch.rs)."""
import io
import os
from pathlib import Path
import pickle
import sys
import unittest
import warnings

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
os.environ["SOURCE_DATE_EPOCH"] = "0"

import matplotlib as mpl
import matplotlib.pyplot as mplt
import matplotlib_rs as rs
import matplotlib_rs.pyplot as rplt
from matplotlib_rs.axes import _base_rs

RC = Path(__file__).resolve().parents[1] / "mplrc/matplotlibrc"
N = 120  # boxes: 6 or 7 artists each, drawn together on several threads


def data(n=N, samples=40, seed=5):
    rng = np.random.default_rng(seed)
    return rng.normal(rng.normal(0, 2, n), rng.uniform(0.3, 2, n), (samples, n))


def figure(module, pyplot, **kw):
    module.rc_file(RC)
    return pyplot.subplots(figsize=(8, 4), dpi=100, **kw)


def canvas(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba()).copy()


def pending(ax):
    """The `_Pending` of the bxp calls of *ax* (wherever they are now)."""
    return [a for a in list.__iter__(ax._children) if type(a) is _base_rs._Pending]


def created(ax):
    return sum(m is not None for p in pending(ax) for m in p.made)


class eager:
    """Create the artists in bxp, as before."""

    def __enter__(self):
        os.environ["MPLRS_LAZY"] = "0"

    def __exit__(self, *exc):
        del os.environ["MPLRS_LAZY"]


SCENES = {
    "default": lambda ax, x: ax.boxplot(x),
    "patch": lambda ax, x: ax.boxplot(x, patch_artist=True, widths=0.7),
    "notch": lambda ax, x: ax.boxplot(x, notch=True, patch_artist=True),
    "horizontal": lambda ax, x: ax.boxplot(x, orientation="horizontal", patch_artist=True,
                                           showmeans=True),
    "meanline": lambda ax, x: ax.boxplot(x, showmeans=True, meanline=True, manage_ticks=False),
    "bare": lambda ax, x: ax.boxplot(x, showbox=False, showcaps=False, showfliers=False),
    "styles": lambda ax, x: ax.boxplot(
        x, patch_artist=True, positions=np.arange(x.shape[1]) * 1.5 + 0.25, widths=0.9,
        boxprops=dict(facecolor="C2", edgecolor="k", linewidth=0.4, alpha=0.6),
        whiskerprops=dict(color="C3", linestyle="--", linewidth=0.8),
        capprops=dict(color="C1", linewidth=2, solid_capstyle="round"),
        medianprops=dict(color="k", linewidth=1.5),
        flierprops=dict(marker="x", markersize=3, markeredgecolor="C4")),
    "markers": lambda ax, x: ax.boxplot(
        x, flierprops=dict(marker="D", markerfacecolor="C0", markeredgecolor="none", alpha=0.5),
        showmeans=True, meanprops=dict(marker="^", markerfacecolor="w")),
    "usermedians": lambda ax, x: ax.boxplot(
        x, usermedians=[None, 0.5] * (x.shape[1] // 2), notch=True,
        conf_intervals=[None, (0.2, 0.9)] * (x.shape[1] // 2)),
    "whis": lambda ax, x: ax.boxplot(x, whis=(5, 95), sym="r+"),
    "invisible": lambda ax, x: ax.boxplot(x, whiskerprops=dict(visible=False), patch_artist=True),
    "log": lambda ax, x: (ax.boxplot(np.exp(x / 3), patch_artist=True), ax.set_yscale("log")),
    "limits": lambda ax, x: (ax.boxplot(x, patch_artist=True), ax.set_xlim(20.3, 60.7),
                             ax.set_ylim(-1.5, 2.5)),
    "layers": lambda ax, x: (
        ax.boxplot(x[:, ::2], positions=np.arange(0, x.shape[1], 2), patch_artist=True,
                   widths=2.5, boxprops=dict(facecolor="C0", alpha=0.7), manage_ticks=False),
        ax.plot([0, x.shape[1]], [-3, 3], color="k", linewidth=6, zorder=2),
        ax.boxplot(x[:, 1::2], positions=np.arange(1, x.shape[1], 2), patch_artist=True,
                   widths=2.5, boxprops=dict(facecolor="C1", alpha=0.7), zorder=2.05,
                   manage_ticks=False),
        ax.axhspan(-1, 1, color="y", alpha=0.4, zorder=2.1)),
    "zorder": lambda ax, x: (ax.boxplot(x, patch_artist=True, zorder=0.5, widths=1.2),
                             ax.grid(True, linewidth=2)),
    "not clipped": lambda ax, x: (ax.boxplot(x, whiskerprops=dict(clip_on=False)),
                                  ax.set_ylim(-1, 1)),
    "many calls": lambda ax, x: [
        ax.boxplot(x[:, k:k + 4], positions=np.arange(k, k + 4), patch_artist=True,
                   boxprops=dict(facecolor=f"C{k % 9}"), manage_ticks=False)
        for k in range(0, x.shape[1], 4)],
}


class LazyBoxplotTests(unittest.TestCase):
    def test_canvas_and_limits(self):
        x = data()
        for name, scene in SCENES.items():
            with self.subTest(scene=name):
                fig, ax = figure(mpl, mplt)
                scene(ax, x)
                expected = canvas(fig)
                limits = (ax.get_xlim(), ax.get_ylim(), tuple(ax.dataLim.extents))
                ticks = [t.get_text() for t in ax.get_xticklabels()]
                mplt.close(fig)

                fig, ax = figure(rs, rplt)
                scene(ax, x)
                lazy = name != "not clipped"
                self.assertEqual(bool(ax._children._holders), lazy)
                np.testing.assert_array_equal(canvas(fig), expected)
                self.assertEqual((ax.get_xlim(), ax.get_ylim(), tuple(ax.dataLim.extents)), limits)
                self.assertEqual([t.get_text() for t in ax.get_xticklabels()], ticks)
                # drawing, limits and ticks need no artist
                self.assertEqual(created(ax), 0)
                self.assertTrue(all(not p.exploded for p in pending(ax)))
                # ... and the canvas is the same once they all exist,
                children = ax.get_children()
                self.assertEqual(ax._children._holders, 0)
                self.assertTrue(all(not p.exploded for p in ax._children._packed))
                np.testing.assert_array_equal(canvas(fig), expected)
                # once those of a call are drawn one by one (one of them changed),
                packed = len(ax._children._packed)
                lines = [a for a in children if isinstance(a, rs.lines.Line2D)]
                lines[0].stale = True
                np.testing.assert_array_equal(canvas(fig), expected)
                self.assertEqual(len(ax._children._packed), packed - lazy)
                rplt.close(fig)
                # and when they are created in bxp
                with eager():
                    fig, ax = figure(rs, rplt)
                    scene(ax, x)
                    self.assertEqual(ax._children._holders, 0)
                    np.testing.assert_array_equal(canvas(fig), expected)
                    rplt.close(fig)

    def test_one_thread(self):
        # (the same code draws few artists, or with MPLRS_THREADS=1)
        x = data(30)
        for name in ("patch", "styles", "markers", "layers"):
            with self.subTest(scene=name):
                fig, ax = figure(mpl, mplt)
                SCENES[name](ax, x)
                expected = canvas(fig)
                mplt.close(fig)
                fig, ax = figure(rs, rplt)
                SCENES[name](ax, x)
                np.testing.assert_array_equal(canvas(fig), expected)
                self.assertEqual(created(ax), 0)
                rplt.close(fig)

    def test_what_keeps_artists_pending(self):
        x = data()
        fig, (ax, other) = figure(rs, rplt, ncols=2, sharey=True)
        result = ax.boxplot(x, patch_artist=True)
        other.boxplot(x[:, :50], orientation="horizontal")
        n = x.shape[1]
        self.assertEqual({k: len(v) for k, v in result.items()},
                         dict(whiskers=2 * n, caps=2 * n, boxes=n, medians=n, fliers=n, means=0))
        self.assertEqual(len(ax._children), 7 * n)
        self.assertEqual((len(ax.lines), len(ax.patches), len(ax.artists)), (6 * n, n, 0))
        self.assertTrue(ax.has_data())
        self.assertEqual(len(ax._rs.children()), 1)
        line, = ax.plot([1, n], [0, 1], label="line")
        self.assertEqual(line.get_label(), "line")
        unlabelled, = ax.plot([1, n], [1, 0])
        self.assertEqual(unlabelled.get_label(), f"_child{7 * n + 1}")
        ax.set(title="title", xlabel="x", ylabel="y", xlim=(0, n + 1))
        ax.grid(True)
        legend = ax.legend(loc="upper left")
        self.assertEqual([t.get_text() for t in legend.get_texts()], ["line"])
        ax.relim()
        ax.autoscale_view()
        ax.relim(visible_only=True)
        fig.tight_layout()
        fig.canvas.draw()
        for fmt in ("png", "svg", "pdf"):
            fig.savefig(io.BytesIO(), format=fmt, bbox_inches="tight")
        fig.set_layout_engine("constrained")
        fig.canvas.draw()
        ax.get_tightbbox()
        self.assertEqual(created(ax) + created(other), 0)
        self.assertEqual(len(ax._rs.children()), 4)  # (the boxplot, two lines, the legend)
        # reading a component creates its artists only
        self.assertIsInstance(result["boxes"][3], rs.patches.PathPatch)
        self.assertEqual(created(ax), 1)
        self.assertEqual(len(list(result["caps"])), 2 * n)
        self.assertEqual(created(ax), 3)
        self.assertFalse(pending(ax)[0].exploded)
        rplt.close(fig)

    def test_artists_as_created_in_bxp(self):
        x = data(12, seed=8)

        def describe(result, ax):
            out = []
            for key, artists in result.items():
                self.assertIsInstance(artists, list)
                for a in artists:
                    geometry = (a.get_path().vertices if isinstance(a, rs.patches.Patch)
                                else np.asarray(a.get_xydata()))
                    out.append((key, type(a).__name__, a.get_label(), a.get_zorder(),
                                geometry.tolist(), list(a.sticky_edges.x), list(a.sticky_edges.y),
                                [type(v).__name__ for v in a.sticky_edges.x],
                                a.axes is ax, a.get_figure() is ax.get_figure(),
                                a.get_clip_on(), a.get_transform() is ax.transData,
                                a.get_visible(), a._in_autoscale))
            out.append([type(a).__name__ + ":" + str(a.get_label()) for a in ax.get_children()])
            return out

        for kw in ({}, {"patch_artist": True, "showmeans": True, "label": "boxes"},
                   {"orientation": "horizontal", "notch": True, "tick_labels": list("abcdefghijkl")},
                   {"manage_ticks": False, "showfliers": False}):
            with self.subTest(kw=kw):
                with eager():
                    fig, ax = figure(rs, rplt)
                    expected = describe(ax.boxplot(x, **kw), ax)
                    rplt.close(fig)
                fig, ax = figure(rs, rplt)
                result = ax.boxplot(x, **kw)
                fig.canvas.draw()
                self.assertEqual(describe(result, ax), expected)
                # the lists and the children hold the same artists
                children = ax._children
                for artists in result.values():
                    for a in artists:
                        self.assertIs(children[children.index(a)], a)
                rplt.close(fig)

    def test_lists(self):
        fig, ax = figure(rs, rplt)
        result = ax.boxplot(data(5), patch_artist=True)
        boxes, whiskers = result["boxes"], result["whiskers"]
        self.assertIsInstance(boxes, list)
        self.assertEqual((len(boxes), bool(boxes), len(whiskers)), (5, True, 10))
        self.assertEqual(created(ax), 0)
        self.assertEqual(len(pickle.loads(pickle.dumps(result["means"]))), 0)
        first = boxes[0]
        self.assertIs(boxes[-5], first)
        self.assertEqual(boxes[1:3], [boxes[1], boxes[2]])
        self.assertIn(first, boxes)
        self.assertEqual(boxes.index(boxes[4]), 4)
        self.assertEqual(boxes, list(boxes))
        self.assertEqual(len(boxes + whiskers), 15)
        self.assertEqual(len(result["caps"] + result["fliers"]), 15)  # (neither read before)
        self.assertEqual(len([first] + result["medians"]), 6)
        self.assertEqual(result["medians"], ax.lines[4::6])
        self.assertEqual(ax.lines[4::6], result["medians"])
        self.assertEqual([a is b for a, b in zip(reversed(boxes), boxes[::-1])], [True] * 5)
        self.assertTrue(repr(boxes).startswith("[<matplotlib_rs.patches.PathPatch"))
        # lower and upper whiskers alternate
        self.assertEqual([w.get_ydata()[0] <= w.get_ydata()[1] for w in whiskers],
                         [False, True] * 5)
        extra = boxes.pop()
        boxes.append(extra)
        boxes.sort(key=id)
        self.assertEqual(len(boxes), 5)
        self.assertEqual(len(ax.patches), 5)  # (the list is the caller's)
        rplt.setp(result["medians"], color="C3")
        self.assertEqual({m.get_color() for m in ax.lines[4::6]}, {"C3"})
        rplt.close(fig)

    def test_changes(self):
        x = data()
        colors = [f"C{i % 10}" for i in range(x.shape[1])]

        def scene(module, pyplot, change):
            fig, ax = figure(module, pyplot)
            result = ax.boxplot(x, patch_artist=True)
            before = canvas(fig)
            change(ax, result)
            ax.relim()
            ax.autoscale_view()
            after = canvas(fig)
            state = (ax.get_xlim(), ax.get_ylim(), len(ax.lines), len(ax.patches),
                     [a.get_label() for a in ax.get_legend_handles_labels()[0]])
            return fig, ax, before, after, state

        def recolor(ax, result):
            for box, color in zip(result["boxes"], colors):
                box.set_facecolor(color)

        def remove(ax, result):
            for a in result["fliers"][::2] + result["caps"][1::4]:
                a.remove()

        def restyle(ax, result):
            result["medians"][3].set(linewidth=4, color="k", zorder=5)
            result["whiskers"][10].set_data([11.3, 11.3], [0, 30])
            result["boxes"][7].set_visible(False)
            result["boxes"][9].set_label("nine")

        def children(ax, result):
            ax.lines[5].set_marker("o")
            ax.patches[2].set_hatch("//")
            ax._children[100].set_alpha(0.3)

        def effects(ax, result):  # (drawn by Python from then on)
            result["boxes"][4].set_path_effects([])
            result["caps"][0].set_rasterized(True)
            result["medians"][1].set_animated(True)

        for change in (recolor, remove, restyle, children, effects):
            with self.subTest(change=change.__name__):
                fig, _, before, expected, state = scene(mpl, mplt, change)
                mplt.close(fig)
                fig, ax, lazy_before, after, lazy_state = scene(rs, rplt, change)
                np.testing.assert_array_equal(lazy_before, before)
                np.testing.assert_array_equal(after, expected)
                self.assertEqual(lazy_state, state)
                self.assertEqual(ax._children._packed, [])
                rplt.close(fig)

    def test_labels(self):
        x = data()
        labels = [f"box {i}" if i % 3 else None for i in range(x.shape[1])]
        for kw in ({"label": "all"}, {"label": labels, "patch_artist": True},
                   {"label": labels[:1] + [""] * (len(labels) - 1), "showbox": False},
                   {"medianprops": {"label": "median"}, "patch_artist": True}):
            with self.subTest(kw=kw):
                states = []
                for module, pyplot in ((mpl, mplt), (rs, rplt)):
                    fig, ax = figure(module, pyplot)
                    result = ax.boxplot(x, **kw)
                    if module is rs:
                        self.assertEqual(created(ax), 0)
                    before = canvas(fig)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        legend = ax.legend(ncols=8, fontsize=5, loc="upper center")
                    handles, texts = ax.get_legend_handles_labels()
                    states.append((before, canvas(fig), texts,
                                   [type(h).__name__ for h in handles],
                                   [t.get_text() for t in legend.get_texts()],
                                   [a.get_label() for a in result["medians"] + result["boxes"]]))
                    if module is rs:
                        self.assertEqual(len(ax._children._packed), 1)
                    pyplot.close(fig)
                for a, b in zip(*states):
                    np.testing.assert_array_equal(a, b)

    def test_clear_and_remove(self):
        x = data(8)
        fig, ax = figure(rs, rplt)
        result = ax.boxplot(x, patch_artist=True)
        medians = result["medians"]
        self.assertIs(medians[0].axes, ax)
        ax.clear()
        self.assertEqual((len(ax._children), len(ax.lines)), (0, 0))
        self.assertIsNone(medians[0].axes)
        box = result["boxes"][0]  # (created after the Axes was cleared)
        self.assertIsNone(box.axes)
        self.assertIsNone(box.get_figure())
        box.set_facecolor("r")
        np.testing.assert_array_equal(box.get_path().vertices[0], [0.75, np.percentile(x[:, 0], 25)])
        again = ax.boxplot(x)
        self.assertEqual(len(ax._children), 7 * 8)
        fig.canvas.draw()
        again["medians"][0].remove()
        self.assertEqual((len(ax._children), len(ax._rs.children())), (7 * 8 - 1, 7 * 8 - 1))
        fig.canvas.draw()
        fig.clear()
        self.assertIsNone(again["whiskers"][0].axes)
        rplt.close(fig)

    def test_statistics_stay_arrays_until_read(self):
        from matplotlib_rs import cbook
        x = data(6)
        stats = cbook._boxplot_stats(x, labels=["a", None, "c", "d", "e", "f"])
        self.assertIsInstance(stats, cbook._BoxplotStats)
        self.assertEqual(len(stats), 6)
        self.assertIsNotNone(stats.columns)
        plain = cbook.boxplot_stats(x, labels=["a", None, "c", "d", "e", "f"])
        self.assertIs(type(plain), list)
        self.assertEqual([sorted(s) for s in stats], [sorted(s) for s in plain])
        self.assertIsNone(stats.columns)  # (the dicts may be changed)
        for a, b in zip(stats, plain):
            for key in a:
                np.testing.assert_array_equal(a[key], b[key])
        self.assertEqual(pickle.loads(pickle.dumps(cbook._boxplot_stats(x)))[0]["med"], plain[0]["med"])

    def test_vector_output(self):
        x = data(40)
        rs.rc_file(RC)
        outputs = []
        for lazy in (True, False):
            if not lazy:
                os.environ["MPLRS_LAZY"] = "0"
            try:
                with rs.rc_context({"svg.hashsalt": "lazy", "svg.fonttype": "none"}):
                    fig, ax = rplt.subplots(figsize=(6, 3))
                    ax.boxplot(x, patch_artist=True, notch=True, showmeans=True,
                               flierprops=dict(marker="o", markersize=2, alpha=0.5),
                               medianprops=dict(linestyle=":"))
                    ax.boxplot(x[:, :10] + 4, positions=np.arange(5, 15), widths=2,
                               boxprops=dict(gid="wide", url="https://example.org/"),
                               manage_ticks=False)
                    self.assertEqual(bool(ax._children._holders), lazy)
                    files = []
                    for fmt, metadata in (("svg", {"Date": None}), ("pdf", {"CreationDate": None}),
                                          ("ps", None), ("eps", None)):
                        buffer = io.BytesIO()
                        fig.savefig(buffer, format=fmt, metadata=metadata)
                        files.append(buffer.getvalue())
                    self.assertEqual(created(ax), 0)
                    rplt.close(fig)
                    outputs.append(files)
            finally:
                os.environ.pop("MPLRS_LAZY", None)
        for fmt, a, b in zip(("svg", "pdf", "ps", "eps"), *outputs):
            self.assertGreater(len(a), 20000, fmt)
            self.assertEqual(a, b, fmt)


class ManyArtistsTests(unittest.TestCase):
    """Lines and patches that follow each other are drawn together."""

    def scene(self, ax, cm):
        rng = np.random.default_rng(12)
        t = np.linspace(0, 1, 30)
        styles = ["-", "--", ":", "-.", "none"]
        markers = ["", "o", "s", "x", "D", ".", ",", "+", "|", "^", "d"]
        for i in range(330):
            line, = ax.plot(t + i * 0.01, rng.normal(i * 0.02, 0.3, 30).cumsum() * 0.05,
                            linestyle=styles[i % 5], marker=markers[i % 11], markersize=3 + i % 4,
                            linewidth=0.3 + (i % 6) * 0.5, alpha=1 if i % 3 else 0.5,
                            color=cm(i / 330), zorder=2 + (i % 4) * 0.01,
                            markevery=3 if i % 29 == 0 else None,
                            drawstyle="steps-mid" if i % 17 == 0 else "default",
                            fillstyle="left" if i % 23 == 0 else "full",
                            markerfacecolor="w" if i % 2 else None,
                            antialiased=i % 13 != 0, solid_capstyle="round" if i % 2 else "butt")
            if i % 31 == 0:
                line.set_gapcolor("k")
            if i % 37 == 0:
                line.set_visible(False)
            if i % 41 == 0:
                line.set_sketch_params(1.0, 30.0, 4.0)
            if i % 43 == 0:
                line.set_clip_on(False)
        ax.add_line(ax.lines[5])  # (a child twice)
        for i in range(300):
            x, y = rng.uniform(0, 3.5), rng.uniform(-1, 2)
            ax.fill([x, x + 0.3, x + 0.15], [y, y, y + 0.3], color=cm(i / 300),
                    alpha=0.3 + 0.7 * (i % 2), edgecolor="k" if i % 3 else "none",
                    linestyle="--" if i % 7 == 0 else "-", hatch="xx" if i % 19 == 0 else None,
                    zorder=2 + (i % 3) * 0.01)
        ax.bar(np.arange(280) * 0.012, rng.uniform(0.2, 1, 280), width=0.01, bottom=-2,
               color=cm(rng.uniform(size=280)), edgecolor="k", linewidth=0.3)
        ax.errorbar(np.linspace(0, 3.5, 150), rng.normal(2.5, 0.1, 150), yerr=0.1, fmt="none",
                    capsize=1.5)

    def test_canvas(self):
        for name in ("canvas", "log"):
            with self.subTest(name):
                images = []
                for module, pyplot in ((mpl, mplt), (rs, rplt)):
                    fig, ax = figure(module, pyplot)
                    self.scene(ax, pyplot.cm.viridis)
                    if name == "log":
                        ax.set_xscale("symlog", linthresh=0.5)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        images.append(canvas(fig))
                    pyplot.close(fig)
                np.testing.assert_array_equal(*images)


if __name__ == "__main__":
    unittest.main()
