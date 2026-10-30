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
                   for key in ("whiskers", "caps", "medians", "means", "fliers")
                   for a in result[key]],)
        # Children (types, labels and data) in creation order, and the data of
        # every returned component.
        state += ([type(a).__name__ + ":" + str(a.get_label()) for a in ax.get_children()],)
        for key in ("whiskers", "caps", "boxes", "medians", "means", "fliers"):
            for a in result[key]:
                state += (a.get_path().vertices if isinstance(a, module.patches.Patch)
                          else np.asarray(a.get_xydata()),)
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
            {"positions": [1.5, 2.25, 3.5, 4.75], "patch_artist": True},
            {"notch": True, "orientation": "horizontal"},
            {"showmeans": True},
            {"showmeans": True, "meanline": True, "patch_artist": True, "notch": True},
            {"showcaps": False, "widths": 0.3},
            {"whis": (0, 100)},
            {"widths": [0.2, 0.3, 0.4, 0.5], "capwidths": [0.1, 0.2, 0.3, 0.4],
             "patch_artist": True, "orientation": "horizontal"},
            {"positions": np.array([2, 4, 6, 8]), "showmeans": True, "manage_ticks": False},
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
        for x, y in zip(sa[3:], sb[3:]):
            np.testing.assert_array_equal(x, y)
        np.testing.assert_array_equal(a, b)

    def draw_bxp(self, module, pyplot, stats, **kw):
        module.rc_file(RC)
        fig, ax = pyplot.subplots(figsize=(4, 3), dpi=72)
        result = ax.bxp(copy.deepcopy(stats), **kw)
        state = [type(a).__name__ + ":" + str(a.get_label()) for a in ax.get_children()]
        for key, artists in result.items():
            for a in artists:
                if isinstance(a, module.lines.Line2D):
                    x, y = a.get_xdata(), a.get_ydata()
                    state.append((key, np.asarray(x).dtype.str, np.asarray(y).dtype.str,
                                  np.asarray(x).tolist(), np.asarray(y).tolist()))
        state.append(tuple(ax.dataLim.extents))
        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba()).copy()
        pyplot.close(fig)
        return image, state

    def test_bxp_statistics_types(self):
        # Plain floats take the vectorized path; ints, float32 and lists of
        # fliers keep the per-box loop and its dtypes.
        base = [dict(med=2.5, q1=1.5, q3=3.5, whislo=0.5, whishi=4.5, mean=2.75,
                     cilo=2.0, cihi=3.0, fliers=np.array([-1.0, 6.0])),
                dict(med=5.0, q1=4.0, q3=6.0, whislo=3.0, whishi=7.0, mean=5.25,
                     cilo=4.5, cihi=5.5, fliers=np.array([]))]
        ints = [dict(med=2, q1=1, q3=3, whislo=0, whishi=4, mean=2, cilo=2, cihi=3,
                     fliers=[-1, 6]), base[1]]
        f32 = [{k: (np.float32(v) if isinstance(v, float) else v) for k, v in s.items()}
               for s in base]
        for stats in (base, ints, f32):
            for kw in ({}, {"showmeans": True, "shownotches": True},
                       {"positions": [0.5, 3.0], "widths": [0.4, 0.8], "orientation": "horizontal",
                        "patch_artist": True, "showmeans": True, "meanline": True}):
                with self.subTest(stats=type(stats[0]["med"]).__name__, kw=kw):
                    a, sa = self.draw_bxp(mpl, mplt, stats, **kw)
                    b, sb = self.draw_bxp(rs, rplt, stats, **kw)
                    self.assertEqual(sa, sb)
                    np.testing.assert_array_equal(a, b)

    def test_vectorized_artists_are_independent(self):
        fig, ax = rplt.subplots()
        result = ax.boxplot(np.random.default_rng(3).normal(size=(40, 5)), patch_artist=True)
        whiskers = result["whiskers"]
        x0 = whiskers[0].get_xdata()
        x0[:] = 99
        self.assertTrue(np.all(whiskers[1].get_xdata() != 99))
        self.assertTrue(np.all(whiskers[2].get_xdata() == 2))
        boxes = result["boxes"]
        boxes[0].get_path().vertices[0] = (-5, -5)
        self.assertNotEqual(tuple(boxes[1].get_path().vertices[0]), (-5, -5))
        self.assertIsNot(boxes[0].get_path().codes, boxes[1].get_path().codes)
        self.assertEqual(len({id(a._callbacks) for v in result.values() for a in v}),
                         sum(map(len, result.values())))
        # Each artist owns its own native handle, in the order of ax._children.
        native = [a for a in ax._children if a._rs_native_in]
        self.assertEqual(len({a._h.ident() for a in native}), len(native))
        rplt.close(fig)


if __name__ == "__main__":
    unittest.main()
