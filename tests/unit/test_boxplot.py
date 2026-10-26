"""Statistics equivalence and mutable artist semantics of the fast bxp path."""
import copy
import pickle
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import matplotlib as mpl
import matplotlib.pyplot as mplt
from matplotlib import cbook as mcbook
import matplotlib_rs as rs
import matplotlib_rs.pyplot as rplt
from matplotlib_rs import cbook as rcbook

RC = Path(__file__).resolve().parents[1] / "mplrc/matplotlibrc"


class BoxplotTests(unittest.TestCase):
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
            for opts in [{}, {"whis": 0.}, {"whis": -1.}, {"whis": (5, 95)},
                         {"autorange": True}, {"bootstrap": 10}]:
                with self.subTest(shape=np.shape(x) if isinstance(x, np.ndarray) else "ragged", opts=opts):
                    np.random.seed(2)
                    expected = mcbook.boxplot_stats(x, **opts)
                    np.random.seed(2)
                    actual = rcbook.boxplot_stats(x, **opts)
                    self.assertEqual(len(expected), len(actual))
                    for a, b in zip(expected, actual):
                        self.assertEqual(a.keys(), b.keys())
                        for key in a:
                            np.testing.assert_array_equal(a[key], b[key], err_msg=key)

    def test_labels_and_errors(self):
        x = np.arange(21, dtype=float).reshape(7, 3)
        for labels in [["a", None, "c"], ("a", "b", "c")]:
            a = mcbook.boxplot_stats(x, labels=labels)
            b = rcbook.boxplot_stats(x, labels=labels)
            self.assertEqual([r.get("label") for r in a], [r.get("label") for r in b])
        for labels in [[], ["one"]]:
            with self.assertRaisesRegex(ValueError, "number of labels"):
                rcbook.boxplot_stats(x, labels=labels)

    def render(self, module, pyplot, options, mutate=False, cycle=False):
        module.rc_file(RC)
        fig, ax = pyplot.subplots(figsize=(5, 3), dpi=80)
        if cycle:
            from cycler import cycler
            ax.set_prop_cycle(cycler(color=["red", "blue"]) + cycler(linewidth=[1., 2.]))
        x = np.random.default_rng(17).normal(size=(30, 4))
        result = ax.boxplot(x, **copy.deepcopy(options))
        registry = pickle.loads(pickle.dumps(result["whiskers"][0]._callbacks))
        self.assertEqual(registry._signals, ["pchanged"])
        self.assertEqual(len(result["whiskers"]), 8)
        self.assertEqual(len({id(a) for artists in result.values() for a in artists}),
                         sum(map(len, result.values())))
        if mutate:
            changes = []
            first, second = result["medians"][:2]
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
            self.assertNotEqual(first.get_color(), second.get_color())
            first.sticky_edges.x.append(-1)
            self.assertNotIn(-1, second.sticky_edges.x)
            registry = pickle.loads(pickle.dumps(second._callbacks))
            self.assertEqual(registry._signals, ["pchanged"])
            with self.assertRaises(ValueError):
                registry.process("invalid-signal")
            if options.get("patch_artist"):
                a, b = result["boxes"][:2]
                a.set_facecolor("orange")
                a.set_alpha(0.4)
                self.assertNotEqual(a.get_facecolor(), b.get_facecolor())
            if result["fliers"]:
                result["fliers"][0].remove()
            ax.relim()
            ax.autoscale_view()
        # The next line must see the same property-cycle state after bxp.
        line, = ax.plot([0, 5], [-3, 3])
        state = (list(ax.dataLim.extents), ax.get_xlim(), ax.get_ylim(), line.get_color(),
                 [a.get_label() for a in ax.get_children() if isinstance(a, module.lines.Line2D)])
        state += ([type(a.get_xdata()).__name__ + ":" + str(np.asarray(a.get_xdata()).dtype)
                   + "/" + type(a.get_ydata()).__name__ + ":" + str(np.asarray(a.get_ydata()).dtype)
                   for a in result["whiskers"] + result["medians"]],)
        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba()).copy()
        pyplot.close(fig)
        return image, state

    def test_output_and_independent_mutations(self):
        options = [
            {}, {"patch_artist": True}, {"patch_artist": True, "notch": True},
            {"orientation": "horizontal", "patch_artist": True},
            {"showmeans": True, "meanline": True},
            {"showbox": False, "showcaps": False, "showfliers": False},
            {"manage_ticks": False, "widths": [0.2, 0.3, 0.4, 0.5], "capwidths": 0.1},
            {"patch_artist": True, "label": ["a", "b", "c", "d"]},
            {"medianprops": {"color": None}},
        ]
        for kw in options:
            for mutate in [False, True]:
                with self.subTest(options=kw, mutate=mutate):
                    a, sa = self.render(mpl, mplt, kw, mutate=mutate)
                    b, sb = self.render(rs, rplt, kw, mutate=mutate)
                    for x, y in zip(sa, sb):
                        np.testing.assert_array_equal(x, y)
                    np.testing.assert_array_equal(a, b)
        a, sa = self.render(mpl, mplt, {}, cycle=True)
        b, sb = self.render(rs, rplt, {}, cycle=True)
        self.assertEqual(sa[-3:], sb[-3:])
        np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
