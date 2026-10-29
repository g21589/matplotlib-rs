"""Correctness checks for the experimental upstream-only Python fast path."""
from contextlib import nullcontext
import copy
import io
import os
from pathlib import Path
import pickle
import sys
import unittest

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np

import official_boxplot_fastpath as fast

RC = Path(__file__).resolve().parents[1] / "mplrc/matplotlibrc"


class OfficialFastpathTests(unittest.TestCase):
    def tearDown(self):
        fast.uninstall()
        plt.close("all")

    def test_statistics_exact(self):
        rng = np.random.default_rng(18)
        inputs = []
        for shape in [(1, 3), (2, 5), (17, 8), (100, 180)]:
            x = rng.normal(size=shape)
            inputs.extend([x, np.asfortranarray(x), x[:, ::-1], np.round(x, 1),
                           x.astype(np.float32), x.astype(np.int64)])
        inputs += [np.zeros((20, 3)), np.ones((20, 3)), np.empty((0, 3)),
                   rng.normal(size=20), [[], [1, 2, 3]],
                   np.ma.array([[1., 2.], [3., 4.]], mask=[[False, True], [False, False]]),
                   np.array([[1., np.nan], [2., 4.], [5., 6.]])]
        for x in inputs:
            for options in [{}, {"whis": 0.}, {"whis": -1.}, {"whis": (5, 95)},
                            {"autorange": True}, {"bootstrap": 10}]:
                with self.subTest(options=options):
                    np.random.seed(2)
                    expected = fast.ORIGINAL_STATS(x, **options)
                    np.random.seed(2)
                    actual = fast.boxplot_stats(x, **options)
                    self.assertEqual(len(expected), len(actual))
                    for a, b in zip(expected, actual):
                        self.assertEqual(a.keys(), b.keys())
                        for key in a:
                            np.testing.assert_array_equal(a[key], b[key], err_msg=key)
        x = np.arange(21, dtype=float).reshape(7, 3)
        for labels in [["a", None, "c"], ("a", "b", "c")]:
            self.assertEqual([s.get("label") for s in fast.ORIGINAL_STATS(x, labels=labels)],
                             [s.get("label") for s in fast.boxplot_stats(x, labels=labels)])
        for labels in [[], ["one"]]:
            with self.assertRaises(ValueError) as expected:
                fast.ORIGINAL_STATS(x, labels=labels)
            with self.assertRaises(ValueError) as actual:
                fast.boxplot_stats(x, labels=labels)
            self.assertEqual(str(expected.exception), str(actual.exception))

    def test_install_modes_and_original_api(self):
        import inspect

        for artists, statistics in [(True, True), (False, True), (True, False), (False, False)]:
            fast.install(artists=artists, statistics=statistics)
            self.assertEqual(inspect.signature(Axes.bxp), inspect.signature(fast.ORIGINAL_BXP))
            self.assertEqual(Axes.bxp is fast.ORIGINAL_BXP, not artists)
            self.assertEqual(mpl.cbook.boxplot_stats is fast.ORIGINAL_STATS, not statistics)
        if mpl.__version__ == "3.8.2":
            self.assertIn("vert", inspect.signature(Axes.boxplot).parameters)
            self.assertNotIn("orientation", inspect.signature(Axes.boxplot).parameters)

    def render(self, optimized, options, mutate=False, cycle=False, scale="linear"):
        fast.uninstall()
        mpl.rcdefaults()
        mpl.rc_file(RC)
        with fast.enabled() if optimized else nullcontext():
            fig, ax = plt.subplots(figsize=(5, 3), dpi=80)
            if cycle:
                from cycler import cycler
                ax.set_prop_cycle(cycler(color=["red", "blue"]) + cycler(linewidth=[1., 2.]))
            ax.set_yscale(scale)
            x = np.random.default_rng(17).normal(size=(30, 4))
            if scale == "log":
                x = np.exp(x)
            result = ax.boxplot(x, **copy.deepcopy(options))
            registry = pickle.loads(pickle.dumps(result["whiskers"][0]._callbacks))
            self.assertEqual(registry._signals, ["pchanged"])
            self.assertEqual(len({id(a) for group in result.values() for a in group}),
                             sum(map(len, result.values())))
            self.assertTrue(all(a._path is not None and not a._invalidx and not a._invalidy
                                for a in ax.lines))
            if mutate:
                first, second = result["medians"][:2]
                changes = []
                cid = first.add_callback(lambda a: changes.append(a))
                second.set_label("second")
                self.assertEqual(changes, [])
                first.set_label("first")
                self.assertEqual(changes, [first])
                first.remove_callback(cid)
                first.set_color("purple")
                second.set_color("green")
                first.set_linewidth(3)
                second.set_data([1.5, 2.5], [3., 3.])
                first.sticky_edges.x.append(-1)
                self.assertNotIn(-1, second.sticky_edges.x)
                self.assertIsNot(first.clipbox, second.clipbox)
                if options.get("patch_artist"):
                    a, b = result["boxes"][:2]
                    a.set_facecolor("orange")
                    a.set_alpha(0.4)
                    self.assertNotEqual(a.get_facecolor(), b.get_facecolor())
                    a.set_path(mpl.path.Path.unit_rectangle())
                    self.assertIsNot(a.get_path(), b.get_path())
                if result["fliers"]:
                    result["fliers"][0].remove()
                ax.relim()
                ax.autoscale_view()
                fig.set_size_inches(6, 3)
                ax.set_position([0.2, 0.2, 0.6, 0.6])
            line, = ax.plot([0, 5], [0.1, 3])
            state = (ax.dataLim.extents.copy(), ax.get_xlim(), ax.get_ylim(), line.get_color(),
                     [a.get_label() for a in ax.lines],
                     [type(a.get_xdata()).__name__ + ":" + str(np.asarray(a.get_xdata()).dtype)
                      + "/" + type(a.get_ydata()).__name__ + ":" + str(np.asarray(a.get_ydata()).dtype)
                      for a in result["whiskers"] + result["medians"]])
            fig.canvas.draw()
            pixels = np.asarray(fig.canvas.buffer_rgba()).copy()
            plt.close(fig)
            return pixels, state

    def test_pixels_and_mutable_artist_semantics(self):
        variants = [{}, {"patch_artist": True}, {"patch_artist": True, "notch": True},
                    {**({"vert": False} if mpl.__version__ == "3.8.2"
                        else {"orientation": "horizontal"}), "patch_artist": True},
                    {"showmeans": True, "meanline": True},
                    {"showbox": False, "showcaps": False, "showfliers": False},
                    {"manage_ticks": False, "widths": [0.2, 0.3, 0.4, 0.5], "capwidths": 0.1},
                    {"patch_artist": True,
                     "labels" if mpl.__version__ == "3.8.2" else "label": ["a", "b", "c", "d"]},
                    {"medianprops": {"color": None}},
                    {"whiskerprops": {"drawstyle": "steps"}}]
        for options in variants:
            for mutate in (False, True):
                with self.subTest(options=options, mutate=mutate):
                    a, sa = self.render(False, options, mutate)
                    b, sb = self.render(True, options, mutate)
                    for old, new in zip(sa, sb):
                        np.testing.assert_array_equal(old, new)
                    np.testing.assert_array_equal(a, b)
        for options in [{}, {"patch_artist": True}]:
            for scale, cycle in [("linear", True), ("log", False)]:
                a, sa = self.render(False, options, cycle=cycle, scale=scale)
                b, sb = self.render(True, options, cycle=cycle, scale=scale)
                for old, new in zip(sa, sb):
                    np.testing.assert_array_equal(old, new)
                np.testing.assert_array_equal(a, b)

    def test_custom_axes_and_transforms_use_fallback(self):
        class CountingAxes(Axes):
            calls = 0

            def plot(self, *args, **kwargs):
                self.calls += 1
                return super().plot(*args, **kwargs)

        fig = plt.figure()
        ax = CountingAxes(fig, [0.1, 0.1, 0.8, 0.8])
        fig.add_axes(ax)
        with fast.enabled():
            ax.boxplot(np.arange(120.).reshape(30, 4), patch_artist=True)
            self.assertEqual(ax.calls, 24)
            self.assertEqual(fast.COUNTS["fast_groups"], 0)
            self.assertEqual(fast.COUNTS["fallback_groups"], 1)
        fig, ax = plt.subplots()
        with fast.enabled():
            ax.boxplot(np.arange(120.).reshape(30, 4),
                       whiskerprops={"transform": ax.transAxes})
            self.assertEqual(fast.COUNTS["fast_groups"], 0)
            self.assertEqual(fast.COUNTS["fallback_groups"], 1)

    def test_vector_formats_and_figure_pickle(self):
        os.environ["SOURCE_DATE_EPOCH"] = "0"
        outputs = []
        for optimized in (False, True):
            fast.uninstall()
            mpl.rc_file(RC)
            mpl.rcParams["svg.hashsalt"] = "official-boxplot-test"
            with fast.enabled() if optimized else nullcontext():
                fig, ax = plt.subplots()
                ax.boxplot(np.random.default_rng(5).normal(size=(60, 7)),
                           patch_artist=True, notch=True, showmeans=True)
                files = []
                for extension in ("png", "svg", "pdf", "eps"):
                    stream = io.BytesIO()
                    fig.savefig(stream, format=extension)
                    files.append(stream.getvalue())
                outputs.append(files)
                restored = pickle.loads(pickle.dumps(fig))
                restored.canvas.draw()
                np.testing.assert_array_equal(np.asarray(fig.canvas.buffer_rgba()),
                                              np.asarray(restored.canvas.buffer_rgba()))
                plt.close("all")
        self.assertEqual(outputs[0], outputs[1])
        self.assertNotIn("matplotlib_rs._mplrs", sys.modules)


if __name__ == "__main__":
    unittest.main()
