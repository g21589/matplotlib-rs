"""Shortcuts that must not change any result, against the code they replace
and against Matplotlib:

- the legend's best position from one call into the Rust core, found once
  per draw, with the extents of the legend's boxes computed once per draw;
- the font height metrics `Text._get_layout` keeps per font;
- `GaussianKDE.evaluate` for several points at a time;
- the clip transform artists of an Axes share;
- path collections whose transforms are one array, drawn on several threads;
- lines whose data, path and visible part share one buffer;
- the worker threads after a fork and with drawing threads.
"""
import io
import multiprocessing
import os
from pathlib import Path
import sys
import threading
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import matplotlib as mpl
import matplotlib.pyplot as mplt
import matplotlib_rs as rs
import matplotlib_rs.pyplot as rplt
from matplotlib_rs import _path, legend as rlegend, offsetbox as roffsetbox, text as rtext

RC = Path(__file__).resolve().parents[1] / "mplrc/matplotlibrc"
SIDES = ((rs, rplt), (mpl, mplt))


def canvas(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba()).copy()


def figure(module, pyplot, plot, **kwargs):
    module.rc_file(RC)
    fig, ax = pyplot.subplots(figsize=(6, 4), dpi=80, **kwargs)
    return fig, ax, plot(ax)


# --- legends ---------------------------------------------------------------

def legend_scenes():
    rng = np.random.default_rng(7)
    x = np.linspace(0, 10, 400)
    walk = np.cumsum(rng.normal(size=5000))
    px, py = rng.normal(size=(2, 300))
    dense = rng.uniform(size=(2, 4000))
    heights = rng.uniform(1, 5, 12)
    gaps = np.where(np.arange(400) % 50 < 5, np.nan, np.sin(x))

    def lines(ax):
        ax.plot(x, np.sin(x), label="sin")
        ax.plot(x, np.cos(x), "--", label="cos")
        return ax.legend()

    def walk_and_points(ax):
        ax.plot(walk, label="walk")
        ax.scatter(np.arange(0, 5000, 100), walk[::100], c="k", label="samples")
        return ax.legend(title="Data")

    def points(ax):
        ax.scatter(px, py, s=20, label="a")
        ax.scatter(px + 2, py - 1, s=np.abs(px) * 30, c=py, label="b")
        return ax.legend(ncols=2)

    def bars(ax):
        ax.bar(np.arange(12), heights, label="bars")
        ax.plot(np.arange(12), heights[::-1], "o-", label="line")
        return ax.legend()

    def patches_and_text(ax, patches):
        ax.fill_between(x, np.sin(x), np.sin(x) + 1, alpha=0.4, label="band")
        ax.add_patch(patches.Circle((8, 1.5), 0.6, label="circle"))
        ax.add_patch(patches.Polygon([[1, -1], [3, -0.5], [2, 0.5]], label="polygon"))
        ax.text(5, 1.8, "a note in the plot")
        ax.annotate("peak", (1.5, 2), (3, 2.3), arrowprops={"arrowstyle": "->"})
        return ax.legend()

    def everywhere(ax):
        # (every candidate position covers data)
        ax.plot(dense[0], dense[1], ".", label="everywhere")
        ax.plot([0, 1], [0, 1], label="diagonal")
        return ax.legend()

    def logarithmic(ax):
        ax.loglog(x + 1, np.exp(x), label="exp")
        ax.loglog(x + 1, (x + 1) ** 2, label="square")
        return ax.legend()

    def with_gaps(ax):
        ax.plot(x, gaps, label="gaps")
        ax.plot([], [], label="empty")
        ax.plot([np.nan, np.nan], [0, 1], label="nan")
        return ax.legend()

    def anchored(ax):
        ax.plot(x, np.sin(x), label="sin")
        ax.scatter(x[::20], np.cos(x[::20]), label="cos")
        return ax.legend(loc="best", bbox_to_anchor=(0.2, 0.2, 0.6, 0.6), shadow=True)

    def step_and_errorbar(ax):
        ax.step(x[::10], np.sin(x[::10]), where="mid", label="step")
        ax.errorbar(x[::25], np.cos(x[::25]), yerr=0.2, fmt="s", label="errors")
        ax.axhline(0.5, label="level")
        return ax.legend()

    return {
        "lines": lines, "walk and points": walk_and_points, "points": points, "bars": bars,
        "patches and text": lambda ax: patches_and_text(
            ax, rs.patches if ax.__module__.startswith("matplotlib_rs") else mpl.patches),
        "everywhere": everywhere, "logarithmic": logarithmic, "with gaps": with_gaps,
        "anchored": anchored, "step and errorbar": step_and_errorbar,
    }


class no_badness_call:
    """Make `Legend._find_best_position` use its loop."""

    def __enter__(self):
        self.original = _path.legend_badness

        def unavailable(*args):
            raise TypeError("the loop decides")

        _path.legend_badness = unavailable

    def __exit__(self, *exc):
        _path.legend_badness = self.original


class LegendTests(unittest.TestCase):
    def test_best_position_same_as_loop_and_matplotlib(self):
        for name, plot in legend_scenes().items():
            with self.subTest(name):
                fig, ax, leg = figure(rs, rplt, plot)
                image = canvas(fig)
                extent = leg.get_window_extent().bounds
                renderer = fig.canvas.get_renderer()
                box = leg._legend_box.get_bbox(renderer)
                found = leg._find_best_position(box.width, box.height, renderer)
                with no_badness_call():
                    self.assertEqual(leg._find_best_position(box.width, box.height, renderer), found)
                    np.testing.assert_array_equal(canvas(fig), image)
                    self.assertEqual(leg.get_window_extent().bounds, extent)
                rplt.close(fig)

                fig, ax, leg = figure(mpl, mplt, plot)
                reference = canvas(fig)
                self.assertEqual(leg.get_window_extent().bounds, extent)
                mplt.close(fig)
                np.testing.assert_array_equal(image, reference)

    def test_the_call_is_used(self):
        # (the comparison above would pass if the loop always decided)
        fig, ax, leg = figure(rs, rplt, legend_scenes()["patches and text"])
        calls = []
        original = _path.legend_badness
        _path.legend_badness = lambda *args: calls.append(args) or original(*args)
        try:
            fig.canvas.draw()
        finally:
            _path.legend_badness = original
        self.assertEqual(len(calls), 1)
        boxes, paths, offsets, bboxes = calls[0]
        self.assertEqual(boxes.shape, (10, 4))
        self.assertEqual((len(paths), offsets.shape[1], bboxes.shape[1:]), (3, 2, (2, 2)))
        self.assertGreaterEqual(len(bboxes), 2)
        rplt.close(fig)

    def test_badness_same_as_bbox_methods(self):
        rng = np.random.default_rng(3)
        Bbox, Path_ = rs.transforms.Bbox, rs.path.Path
        for _ in range(40):
            paths = []
            for n in rng.integers(0, 60, rng.integers(0, 5)):
                v = rng.normal(size=(n, 2)) * 3
                if n > 4 and rng.random() < 0.5:
                    v[rng.integers(0, n, 2)] = np.nan
                codes = None
                if n > 6 and rng.random() < 0.5:
                    codes = np.full(n, Path_.LINETO, dtype=np.uint8)
                    codes[[0, n // 2]] = Path_.MOVETO
                paths.append(Path_(v, codes))
            offsets = rng.normal(size=(rng.integers(0, 50), 2)) * 3
            others = [Bbox(np.sort(rng.normal(size=(2, 2)) * 3, axis=0)) for _ in range(rng.integers(0, 6))]
            corners = rng.normal(size=(12, 2, 2)) * 3
            corners[0, 0, 0] = np.nan
            expected = []
            for c in corners:
                box = Bbox(c)
                expected.append(sum(box.count_contains(p.vertices) for p in paths)
                                + box.count_contains(offsets) + box.count_overlaps(others)
                                + sum(p.intersects_bbox(box, filled=False) for p in paths))
            got = _path.legend_badness(
                corners.reshape(-1, 4), paths, offsets,
                np.asarray([np.array(b) for b in others], dtype=float).reshape(-1, 2, 2))
            self.assertEqual(got, [int(e) for e in expected])

    def test_one_search_and_one_layout_per_draw(self):
        fig, ax, leg = figure(rs, rplt, legend_scenes()["lines"])
        fig.canvas.draw()
        searches, layouts = [], []
        data, layout = rlegend.Legend._auto_legend_data, rtext.Text._get_layout
        rlegend.Legend._auto_legend_data = lambda self, r: searches.append(1) or data(self, r)
        rtext.Text._get_layout = lambda self, r: layouts.append(self) or layout(self, r)
        try:
            fig.canvas.draw()
        finally:
            rlegend.Legend._auto_legend_data, rtext.Text._get_layout = data, layout
        self.assertEqual(len(searches), 1)
        legend_texts = {id(t) for t in [*leg.get_texts(), leg.get_title()]}
        per_text = [sum(1 for t in layouts if id(t) == i) for i in legend_texts]
        # (once for the extent of its box and once in Text.draw)
        self.assertTrue(all(n <= 2 for n in per_text), per_text)
        self.assertIsNone(roffsetbox._rs_kept_extents)
        self.assertNotIn("_rs_best_found", leg.__dict__)
        rplt.close(fig)

    def test_changes_between_draws_are_seen(self):
        # nothing is kept from one draw to the next
        def steps(ax):
            line, = ax.plot([0, 1, 2], [0, 1, 4], label="first")
            leg = ax.legend()
            yield
            leg.get_texts()[0].set_text("a much longer label\nin two lines")
            yield
            line.set_data([0, 1, 2], [4, 1, 0])
            yield
            leg.get_texts()[0].set_fontsize(18)
            leg.set_title("title")
            yield
            ax.set_xlim(-5, 3)
            yield

        figs = [figure(m, p, steps) for m, p in SIDES]
        for step in range(5):
            images = []
            for fig, ax, gen in figs:
                next(gen)
                images.append(canvas(fig))
            np.testing.assert_array_equal(images[0], images[1], err_msg=f"step {step}")
        rplt.close(figs[0][0])
        mplt.close(figs[1][0])

    def test_error_in_draw_leaves_nothing_kept(self):
        fig, ax, leg = figure(rs, rplt, legend_scenes()["lines"])
        fig.canvas.draw()
        original = rlegend.Legend._auto_legend_data

        def fail(self, renderer):
            raise RuntimeError("no data")

        rlegend.Legend._auto_legend_data = fail
        try:
            with self.assertRaises(RuntimeError):
                leg.draw(fig.canvas.get_renderer())
        finally:
            rlegend.Legend._auto_legend_data = original
        self.assertIsNone(roffsetbox._rs_kept_extents)
        self.assertNotIn("_rs_best_found", leg.__dict__)
        rplt.close(fig)


# --- text ------------------------------------------------------------------

class FontMetricsTests(unittest.TestCase):
    def fresh(self, prop, dpi):
        rtext._rs_height_metrics_cache.clear()
        return rtext._rs_font_height_metrics(prop, dpi)

    def test_kept_values_are_the_computed_ones(self):
        rs.rc_file(RC)
        FP = rs.font_manager.FontProperties
        props = [FP(), FP(size=20), FP(family="serif"), FP(family="monospace", weight="bold", style="italic"),
                 FP(family=["cmr10", "DejaVu Sans"]), FP(family="STIXGeneral", size=7.5)]
        seen = set()
        for prop in props:
            for dpi in (72, 100.0, 150.5):
                first = self.fresh(prop, dpi)
                self.assertEqual(rtext._rs_font_height_metrics(prop, dpi), first)
                self.assertEqual(rtext._rs_font_height_metrics(prop.copy(), dpi), first)
                self.assertNotIn(None, first)
                seen.add(first)
        # (the DejaVu families share their metrics)
        self.assertGreaterEqual(len(seen), 9)
        # a changed property is another font
        prop = FP(size=10)
        small = rtext._rs_font_height_metrics(prop, 100)
        prop.set_size(30)
        large = rtext._rs_font_height_metrics(prop, 100)
        self.assertEqual(large, self.fresh(prop, 100))
        self.assertAlmostEqual(large[0], 3 * small[0])

    def test_rc_params_and_added_fonts(self):
        rs.rc_file(RC)
        prop = rs.font_manager.FontProperties(family=["sans-serif"])
        default = self.fresh(prop, 100)
        with rs.rc_context({"font.sans-serif": ["cmr10"]}):
            other = rtext._rs_font_height_metrics(prop, 100)
            self.assertEqual(other, self.fresh(prop, 100))
            self.assertNotEqual(other, default)
        self.assertEqual(rtext._rs_font_height_metrics(prop, 100), default)
        # the number of fonts of the font manager is part of what is kept
        before = len(rtext._rs_height_metrics_cache)
        fonts = rs.font_manager.fontManager.ttflist
        fonts.append(fonts[0])
        try:
            self.assertEqual(rtext._rs_font_height_metrics(prop, 100), default)
            self.assertEqual(len(rtext._rs_height_metrics_cache), before + 1)
        finally:
            fonts.pop()

    def test_text_extents_same_as_matplotlib(self):
        def texts(ax):
            ax.set_axis_off()
            out = [ax.text(0.1, 0.8, "plain text"),
                   ax.text(0.1, 0.6, "two\nlines", fontsize=16, family="serif"),
                   ax.text(0.5, 0.4, "rotated", rotation=30, family="monospace", weight="bold"),
                   ax.text(0.5, 0.2, r"math $\alpha^2 + \frac{1}{x}$", fontsize=14),
                   ax.text(0.7, 0.7, "three\nlines\nhere", linespacing=1.6, ha="center", va="top")]
            return out

        figs = [figure(m, p, texts) for m, p in SIDES]
        images = [canvas(fig) for fig, ax, out in figs]
        np.testing.assert_array_equal(images[0], images[1])
        for a, b in zip(figs[0][2], figs[1][2]):
            self.assertEqual(a.get_window_extent().bounds, b.get_window_extent().bounds)
            # (the Python layout, which legends and offset boxes use)
            ra, rb = figs[0][0].canvas.get_renderer(), figs[1][0].canvas.get_renderer()
            la, lb = a._get_layout(ra), b._get_layout(rb)
            self.assertEqual(la[0].bounds, lb[0].bounds)
            self.assertEqual([i[1:] for i in la[1]], [i[1:] for i in lb[1]])
        rplt.close(figs[0][0])
        mplt.close(figs[1][0])


# --- KDE -------------------------------------------------------------------

class KdeTests(unittest.TestCase):
    def test_evaluate_same_bits_as_matplotlib(self):
        rng = np.random.default_rng(11)
        cases = [(150, 100, None), (2000, 100, None), (500, 37, "silverman"), (8192, 100, 0.3),
                 (8193, 64, None), (30000, 50, None), (4097, 3, None), (101, 100, 0.05)]
        for n, m, bw in cases:
            with self.subTest(n=n, m=m, bw=bw):
                data = rng.normal(size=n)
                data[::7] = data[0]
                points = np.linspace(data.min() - 1, data.max() + 1, m)
                points[m // 2] = np.nan
                got = rs.mlab.GaussianKDE(data, bw).evaluate(points)
                expected = mpl.mlab.GaussianKDE(data, bw).evaluate(points)
                self.assertEqual(got.tobytes(), expected.tobytes())

    def test_other_inputs_use_the_loop(self):
        rng = np.random.default_rng(12)
        data = rng.normal(size=(2, 400))
        points = rng.normal(size=(2, 50))
        for d, p in [(data, points), (data[0].astype(np.float32), points[0]),
                     (np.arange(300), np.linspace(0, 300, 20)),
                     (np.ma.masked_invalid(data[0]), points[0])]:
            got = rs.mlab.GaussianKDE(d).evaluate(p)
            expected = mpl.mlab.GaussianKDE(d).evaluate(p)
            self.assertEqual(np.asarray(got).tobytes(), np.asarray(expected).tobytes())

    def test_violin_statistics_and_canvas(self):
        rng = np.random.default_rng(13)
        data = [rng.normal(size=n) * s for n, s in [(60, 1), (500, 2), (3000, 0.5), (12000, 1.5)]]
        for a, b in zip(rs.cbook.violin_stats(data, ("GaussianKDE", "scott")),
                        mpl.cbook.violin_stats(data, ("GaussianKDE", "scott"))):
            self.assertEqual(a["vals"].tobytes(), b["vals"].tobytes())
            self.assertEqual(a["coords"].tobytes(), b["coords"].tobytes())
        figs = [figure(m, p, lambda ax: ax.violinplot(data, showmedians=True)) for m, p in SIDES]
        np.testing.assert_array_equal(canvas(figs[0][0]), canvas(figs[1][0]))
        rplt.close(figs[0][0])
        mplt.close(figs[1][0])


# --- clipping --------------------------------------------------------------

class ClipTransformTests(unittest.TestCase):
    def test_artists_share_the_transform_while_the_patch_is_unchanged(self):
        def steps(ax):
            out = [ax.plot([0, 1], [0, 1])[0], ax.text(0.5, 0.5, "t", clip_on=True)]
            yield out
            ax.patch.set_bounds(0.1, 0.2, 0.5, 0.6)
            out += [ax.plot([0, 1], [1, 0])[0]]
            yield out
            ax.patch.set_angle(15)
            out += [ax.axhline(0.3)]
            yield out
            ax.set_position([0.2, 0.2, 0.5, 0.5])
            out += [ax.plot([0.5, 0.5], [0, 1])[0]]
            yield out

        figs = [figure(m, p, steps) for m, p in SIDES]
        ours = figs[0][1]
        for step in range(4):
            artists, reference = next(figs[0][2]), next(figs[1][2])
            for a, b in zip(artists, reference):
                np.testing.assert_array_equal(a.clipbox.get_points(), b.clipbox.get_points())
            fresh = rs.transforms.TransformedBbox(rs.transforms.Bbox.unit(), ours.patch.get_transform())
            np.testing.assert_array_equal(artists[-1].clipbox.get_points(), fresh.get_points())
            if step == 0:
                # (the Rust core draws an Axes patch with other bounds or an
                # angle differently from Matplotlib: only the clip boxes
                # are compared after that)
                np.testing.assert_array_equal(canvas(figs[0][0]), canvas(figs[1][0]))
        a, b, c, d, e = artists
        self.assertIs(a.clipbox._transform, b.clipbox._transform)
        self.assertEqual(len({id(x.clipbox._transform) for x in (a, c, d)}), 3)
        self.assertIs(d.clipbox._transform, e.clipbox._transform)
        rplt.close(figs[0][0])
        mplt.close(figs[1][0])

    def test_moved_axes_clip_the_same_as_matplotlib(self):
        def moved(ax):
            first = ax.plot([0, 1], [0, 1], lw=6)[0]
            ax.set_position([0.3, 0.25, 0.4, 0.5])
            second = ax.plot([0, 1], [1, 0], lw=6)[0]
            ax.set_xlim(0.2, 0.8)
            return first, second

        figs = [figure(m, p, moved) for m, p in SIDES]
        np.testing.assert_array_equal(canvas(figs[0][0]), canvas(figs[1][0]))
        for a, b in zip(figs[0][2], figs[1][2]):
            np.testing.assert_array_equal(a.clipbox.get_points(), b.clipbox.get_points())
        self.assertIs(figs[0][2][0].clipbox._transform, figs[0][2][1].clipbox._transform)
        rplt.close(figs[0][0])
        mplt.close(figs[1][0])

    def test_other_rectangles_are_not_kept(self):
        fig, ax = rplt.subplots()
        rect = rs.patches.Rectangle((0.2, 0.2), 0.3, 0.3, transform=ax.transAxes)
        ax.add_patch(rect)
        a, b = ax.plot([0, 1])[0], ax.plot([1, 0])[0]
        a.set_clip_path(rect)
        b.set_clip_path(rect)
        self.assertIsNot(a.clipbox._transform, b.clipbox._transform)
        np.testing.assert_array_equal(a.clipbox.get_points(), b.clipbox.get_points())
        rplt.close(fig)


# --- path collections ------------------------------------------------------

def many_markers(ax, n=6000, seed=5):
    rng = np.random.default_rng(seed)
    x, y = rng.normal(size=(2, n))
    return ax.scatter(x, y, s=rng.uniform(1, 60, n), c=rng.uniform(size=n), alpha=0.6,
                      edgecolors="k", linewidths=0.3)


class CollectionTests(unittest.TestCase):
    def test_transform_array_same_as_matrices(self):
        rs.rc_file(RC)
        fig = rplt.figure(figsize=(4, 3), dpi=80)
        renderer = fig.canvas.get_renderer()
        rng = np.random.default_rng(2)
        n = 700
        scales = rng.uniform(2, 9, n)
        matrices = np.zeros((n, 3, 3))
        matrices[:, 0, 0] = matrices[:, 1, 1] = scales
        matrices[:, 2, 2] = 1
        offsets = rng.uniform(0, 1, (n, 2)) * [320, 240]
        colors = rng.uniform(size=(n, 4))
        path = rs.path.Path.unit_circle()
        identity = rs.transforms.IdentityTransform()

        def draw(transforms, paths):
            renderer.clear()
            gc = renderer.new_gc()
            renderer.draw_path_collection(
                gc, identity, paths, transforms, offsets, identity, colors, colors[::-1],
                [0.5], [(0, None)], [True], [None], "screen", hatchcolors=None)
            return np.asarray(renderer.buffer_rgba()).copy()

        array = draw(matrices, [path])
        self.assertGreater(len(np.unique(array.reshape(-1, 4), axis=0)), 100)
        np.testing.assert_array_equal(draw(list(matrices), [path]), array)
        np.testing.assert_array_equal(draw([rs.transforms.Affine2D(m) for m in matrices], [path]), array)
        np.testing.assert_array_equal(draw(matrices.astype(np.float32).astype(float)[:, :, :], [path]),
                                      draw(list(matrices.astype(np.float32).astype(float)), [path]))
        # without paths nothing is drawn
        empty = draw(matrices, [])
        self.assertEqual(len(np.unique(empty.reshape(-1, 4), axis=0)), 1)
        rplt.close(fig)

    def test_threads_give_the_same_image_as_matplotlib(self):
        figs = [figure(m, p, many_markers) for m, p in SIDES]
        np.testing.assert_array_equal(canvas(figs[0][0]), canvas(figs[1][0]))
        rplt.close(figs[0][0])
        mplt.close(figs[1][0])


# --- lines -----------------------------------------------------------------

def line_cases():
    rng = np.random.default_rng(17)
    n = 6000
    x = np.sort(rng.uniform(0, 100, n))
    y = np.cumsum(rng.normal(size=n))
    with_nan_x = x.copy()
    with_nan_x[[5, 900, 901, 4000]] = np.nan
    with_nan_y = y.copy()
    with_nan_y[1000:1100] = np.nan
    cases = {}
    for style in ("default", "steps-pre", "steps-mid", "steps-post"):
        cases[f"zoomed {style}"] = lambda ax, s=style: (ax.plot(x, y, drawstyle=s, lw=0.8)[0], ax.set_xlim(30, 34))[0]
        cases[f"full {style}"] = lambda ax, s=style: ax.plot(x, y, drawstyle=s, lw=0.8)[0]
    cases["zoomed markers"] = lambda ax: (ax.plot(x, y, "o-", ms=3, lw=0.5)[0], ax.set_xlim(40, 41))[0]
    cases["nan in x"] = lambda ax: (ax.plot(with_nan_x, y, "-s", ms=2)[0], ax.set_xlim(10, 16))[0]
    cases["nan in y"] = lambda ax: (ax.plot(x, with_nan_y, drawstyle="steps-post")[0], ax.set_xlim(12, 25))[0]
    shuffled = rng.permutation(x)
    cases["unsorted"] = lambda ax: (ax.plot(shuffled, y, lw=0.3)[0], ax.set_xlim(20, 60))[0]
    cases["markevery"] = lambda ax: (ax.plot(x, y, "-^", markevery=97)[0], ax.set_xlim(20, 60))[0]
    cases["dashed zoomed"] = lambda ax: (ax.plot(x, y, "--")[0], ax.set_xlim(50, 52))[0]
    cases["no clipping"] = lambda ax: (ax.plot(x, y, clip_on=False)[0], ax.set_xlim(45, 55))[0]
    cases["log x"] = lambda ax: (ax.semilogx(x + 1, y)[0], ax.set_xlim(3, 30))[0]
    return cases, (x, y)


class LineTests(unittest.TestCase):
    def test_long_lines_same_as_matplotlib(self):
        cases, (x, y) = line_cases()
        for name, plot in cases.items():
            with self.subTest(name):
                figs = [figure(m, p, plot) for m, p in SIDES]
                np.testing.assert_array_equal(canvas(figs[0][0]), canvas(figs[1][0]))
                # another view, then other data: the kept data is not stale
                for fig, ax, line in figs:
                    lo, hi = ax.get_xlim()
                    ax.set_xlim(lo + 0.37 * (hi - lo), hi + 2.5 * (hi - lo))
                np.testing.assert_array_equal(canvas(figs[0][0]), canvas(figs[1][0]))
                for fig, ax, line in figs:
                    line.set_data(x[::2] + 1.5, -y[::2])
                np.testing.assert_array_equal(canvas(figs[0][0]), canvas(figs[1][0]))
                for fig, ax, line in figs:
                    line.set_ydata(y[::2] * 0.5)
                    line.set_drawstyle("steps-mid")
                np.testing.assert_array_equal(canvas(figs[0][0]), canvas(figs[1][0]))
                np.testing.assert_array_equal(figs[0][2].get_xydata(), figs[1][2].get_xydata())
                rplt.close(figs[0][0])
                mplt.close(figs[1][0])


# --- worker threads --------------------------------------------------------

def _draw_in_child(fig, queue):
    queue.put(canvas(fig).tobytes())


class WorkerTests(unittest.TestCase):
    def test_child_process_draws_after_fork(self):
        if "fork" not in multiprocessing.get_all_start_methods():
            self.skipTest("no fork on this platform")
        import warnings
        fig, ax, _ = figure(rs, rplt, many_markers)
        expected = canvas(fig).tobytes()  # (starts the worker threads)
        context = multiprocessing.get_context("fork")
        queue = context.Queue()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            child = context.Process(target=_draw_in_child, args=(fig, queue))
            child.start()
        try:
            self.assertEqual(queue.get(timeout=120), expected)
        finally:
            child.join(10)
            if child.is_alive():
                child.kill()
        self.assertEqual(child.exitcode, 0)
        self.assertEqual(canvas(fig).tobytes(), expected)
        rplt.close(fig)

    def test_threads_drawing_at_the_same_time(self):
        # (a renderer belongs to the thread that created it: every thread
        # draws figures of its own; PNG compression releases the GIL, so the
        # threads' parallel loops overlap)
        rs.rc_file(RC)
        Figure = rs.figure.Figure

        def save(seed):
            fig = Figure(figsize=(6, 4), dpi=80)
            many_markers(fig.subplots(), 3000, seed)
            out = io.BytesIO()
            fig.savefig(out, format="png")
            return out.getvalue()

        expected = [save(seed) for seed in range(4)]
        results = [[] for _ in expected]

        def work(seed):
            for _ in range(3):
                results[seed].append(save(seed))

        threads = [threading.Thread(target=work, args=(seed,)) for seed in range(len(expected))]
        for t in threads:
            t.start()
        for t in threads:
            t.join(300)
        for seed, r in enumerate(results):
            self.assertEqual(r, [expected[seed]] * 3)


if __name__ == "__main__":
    unittest.main()
