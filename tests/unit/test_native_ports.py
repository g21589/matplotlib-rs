"""The built-in replacements of matplotlib's compiled dependencies.

matplotlib_rs needs only NumPy: constrained layout uses a port of kiwi's
solver, contour the mpl2014 algorithm of ContourPy, Delaunay triangulation the
qhull library matplotlib builds, and PNG files are read and written natively.
Each is compared with the package it replaces when that package is installed
(as in CI); otherwise those comparisons are skipped and only self-checks run.
"""
import io
from pathlib import Path
import random
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import matplotlib_rs as rs
import matplotlib_rs.pyplot as rplt
from matplotlib_rs import _contour, _kiwi, _layoutgrid, _png, _qhull


def optional(name):
    try:
        return __import__(name)
    except ImportError:
        return None


kiwisolver = optional("kiwisolver")
contourpy = optional("contourpy")
PIL = optional("PIL")
if PIL is not None:
    import PIL.Image
    import PIL.PngImagePlugin
try:
    from matplotlib import _qhull as mpl_qhull
except ImportError:
    mpl_qhull = None


def layout_figures():
    rng = np.random.default_rng(0)
    fig, axs = rplt.subplots(2, 3, layout="constrained")
    fig.suptitle("suptitle")
    axs[0, 0].set_ylabel("long y label\nline 2")
    yield fig
    fig, axs = rplt.subplot_mosaic("AAB;CDB", layout="constrained")
    fig.colorbar(axs["B"].imshow(np.eye(3)), ax=axs["B"])
    yield fig
    fig = rplt.figure(layout="constrained")
    sf = fig.subfigures(1, 2, width_ratios=[1, 2])
    sf[0].subplots(3, 1)
    sf[1].subplots(2, 2, sharex=True)
    sf[1].suptitle("x")
    yield fig
    fig, axs = rplt.subplots(3, 3, layout="constrained",
                             gridspec_kw={"width_ratios": [1, 2, 3], "height_ratios": [3, 1, 1]})
    for a in axs.flat:
        a.set_title("title")
        a.set_xlabel("xlabel")
    fig.colorbar(axs[0, 0].pcolormesh(rng.random((4, 4))), ax=axs[:, -1])
    yield fig
    fig = rplt.figure(layout="constrained", figsize=(9, 4))
    gs = fig.add_gridspec(2, 4)
    fig.add_subplot(gs[:, 0])
    fig.add_subplot(gs[0, 1:])
    fig.add_subplot(gs[1, 1])
    fig.add_subplot(gs[1, 2:]).set_ylabel("y")
    fig.supxlabel("supxlabel")
    yield fig
    fig = rplt.figure(layout="constrained")
    top, bottom = fig.subfigures(2, 1)
    top.subplots(1, 4)
    inner = bottom.subfigures(1, 2)
    inner[0].subplots(2, 2)
    inner[1].subplots(1, 1).plot([1, 2], label="line")
    fig.legend(loc="outside right upper")
    yield fig


def layout_positions(kiwi):
    saved = _layoutgrid.kiwi
    _layoutgrid.kiwi = kiwi
    try:
        out = []
        for fig in layout_figures():
            fig.canvas.draw()
            out.append([ax.get_position().bounds for ax in fig.get_axes()])
            rplt.close(fig)
        return out
    finally:
        _layoutgrid.kiwi = saved


class KiwiTests(unittest.TestCase):
    def test_arithmetic(self):
        K = _kiwi
        x, y, z = K.Variable("x"), K.Variable("y"), K.Variable("z")
        e = (x - 2 * y + z / 4 - 3) * 2
        self.assertEqual([(t.variable().name(), t.coefficient()) for t in e.terms()],
                         [("x", 2.0), ("y", -4.0), ("z", 0.5)])
        self.assertEqual(e.constant(), -6.0)
        self.assertEqual((1 - x).constant(), 1.0)
        self.assertEqual(((-x).variable(), (-x).coefficient()), (x, -1.0))
        cn = (x + y >= z) | "strong"
        self.assertEqual((cn.op(), cn.strength()), (">=", K.strength.strong))
        self.assertEqual(K.strength.required, 1001001000.0)
        for bad in (lambda: x < y, lambda: x * y, lambda: x / y, lambda: x != y):
            with self.assertRaises(TypeError):
                bad()
        with self.assertRaises(TypeError):
            hash(x)

    @unittest.skipIf(kiwisolver is None, "kiwisolver is not installed")
    def test_arithmetic_matches_kiwisolver(self):
        def build(K):
            a, b, c = K.Variable("a"), K.Variable("b"), K.Variable("c")
            exprs = [a + b, a - b, 3 * a, a * 0.5 - 1, b / 3, (a + 2 * b - c) / 7,
                     1.5 - (a - b), -(a + 1), (a + b) + (b - c), 2 + a + 1,
                     a + np.float64(0.25), (a - b) * np.int64(3)]
            out = []
            for e in exprs:
                e = e if hasattr(e, "terms") else (e + 0)
                out.append(([(t.variable().name(), t.coefficient()) for t in e.terms()],
                            e.constant()))
            cn = ((a + b + a) == 4) | 5.0
            reduced = sorted((t.variable().name(), t.coefficient())
                             for t in cn.expression().terms())
            out.append((reduced, cn.expression().constant(), cn.op(), cn.strength()))
            return out
        self.assertEqual(build(_kiwi), build(kiwisolver))

    def test_solver_and_errors(self):
        K = _kiwi
        x, y = K.Variable("x"), K.Variable("y")
        s = K.Solver()
        s.addConstraint(x + y == 10)
        s.addConstraint(x >= 2)
        s.addConstraint((x == 8) | "weak")
        s.addEditVariable(y, "strong")
        s.suggestValue(y, 9)
        s.updateVariables()
        self.assertEqual((x.value(), y.value()), (2.0, 8.0))
        cn = x <= 1
        with self.assertRaises(K.UnsatisfiableConstraint) as cm:
            s.addConstraint(cn)
        self.assertIs(cm.exception.constraint, cn)
        with self.assertRaises(K.DuplicateEditVariable):
            s.addEditVariable(y, "strong")
        with self.assertRaises(K.BadRequiredStrength):
            s.addEditVariable(K.Variable(), "required")
        with self.assertRaises(K.UnknownEditVariable):
            s.suggestValue(K.Variable(), 1.0)
        self.assertTrue(s.hasEditVariable(y))
        s.removeEditVariable(y)
        s.updateVariables()
        # As in kiwi, the rejected constraint has already changed the tableau.
        self.assertEqual((x.value(), y.value()), (1.0, 9.0))

    @unittest.skipIf(kiwisolver is None, "kiwisolver is not installed")
    def test_random_systems_match_kiwisolver(self):
        # kiwisolver orders each constraint's variables by memory address, so
        # results can differ in the last bits; layouts are compared exactly below.
        def solve(K, seed):
            rnd = random.Random(seed)
            vs = [K.Variable(f"v{i}") for i in range(rnd.randint(2, 12))]
            s = K.Solver()
            for v in vs:
                s.addConstraint(v >= -100)
                s.addConstraint(v <= 100)
            failures = 0
            for _ in range(rnd.randint(1, 20)):
                a, b = rnd.sample(vs, 2)
                e = rnd.uniform(-3, 3) * a + rnd.choice([1, -1, 0.5]) * b
                cn = {0: e == rnd.uniform(-10, 10), 1: e >= rnd.uniform(-10, 10),
                      2: e <= rnd.uniform(-10, 10)}[rnd.randint(0, 2)]
                try:
                    s.addConstraint(cn | rnd.choice(["required", "strong", "medium", "weak"]))
                except K.UnsatisfiableConstraint:
                    failures += 1
            s.updateVariables()
            return failures, [v.value() for v in vs]
        for seed in range(400):
            with self.subTest(seed=seed):
                fa, va = solve(kiwisolver, seed)
                fb, vb = solve(_kiwi, seed)
                self.assertEqual(fa, fb)
                np.testing.assert_allclose(vb, va, rtol=1e-9, atol=1e-9)

    @unittest.skipIf(kiwisolver is None, "kiwisolver is not installed")
    def test_layouts_match_kiwisolver(self):
        a, b = layout_positions(kiwisolver), layout_positions(_kiwi)
        self.assertEqual(a, b)


def contour_fields():
    rng = np.random.default_rng(1)
    for trial in range(200):
        ny, nx = (int(v) for v in rng.integers(2, 30, 2))
        kind = trial % 4
        if kind == 0:
            z = rng.random((ny, nx))
        elif kind == 1:
            yy, xx = np.mgrid[0:ny, 0:nx]
            z = np.sin(xx / 3.0) * np.cos(yy / 4.0) + 0.1 * rng.random((ny, nx))
        elif kind == 2:
            z = rng.integers(0, 4, (ny, nx)).astype(float)  # ties and saddles
        else:
            z = rng.normal(size=(ny, nx)).cumsum(0).cumsum(1)
        if trial % 3 == 0:
            z = np.ma.array(z, mask=rng.random((ny, nx)) < 0.15)
        if trial % 7 == 0:
            z = np.array(z, dtype=float)
            z[rng.random((ny, nx)) < 0.1] = np.nan
        if trial % 2:
            x, y = np.meshgrid(np.arange(nx) * 0.5, np.linspace(-1, 3, ny))
        else:
            x, y = np.cumsum(rng.random(nx)), np.cumsum(rng.random(ny))
        yield x, y, z, bool(trial % 5), [0, 1, 2, 3, 5][trial % 5]


def contour_outputs(make, x, y, z, corner_mask, chunk):
    g = make(x, y, z, corner_mask, chunk)
    zz = np.ma.masked_invalid(np.ma.asarray(z))
    lo, hi = float(zz.min()), float(zz.max())
    levels = np.linspace(lo, hi, 7)
    out = [g.create_contour(lev) for lev in [*levels, lo - 1, hi + 1, np.nan]]
    out += [g.create_filled_contour(a, b)
            for a, b in [*zip(levels[:-1], levels[1:]), (-np.inf, levels[2]), (levels[3], np.inf)]]
    return out


def port_generator(x, y, z, corner_mask, chunk):
    g = _contour.contour_generator(x, y, z, name="mpl2014",
                                   corner_mask=corner_mask, chunk_size=chunk)
    assert type(g).__module__ == "matplotlib_rs._mplrs"
    return g


class ContourTests(unittest.TestCase):
    def test_simple_contours(self):
        x, y = np.meshgrid(np.arange(4.), np.arange(3.))
        g = port_generator(x, y, x, True, 0)
        vertices, codes = g.create_contour(1.5)
        np.testing.assert_array_equal(vertices[0], [[1.5, 2], [1.5, 1], [1.5, 0]])
        np.testing.assert_array_equal(codes[0], [1, 2, 2])
        vertices, codes = g.create_filled_contour(0.5, 1.5)
        self.assertEqual(codes[0][-1], 79)
        with self.assertRaises(ValueError):
            g.create_filled_contour(2, 1)
        with self.assertRaises(ImportError if contourpy is None else ValueError):
            _contour.contour_generator(x, y, x, name="serial" if contourpy is None else "bogus",
                                       corner_mask=True, chunk_size=0)
        self.assertIs(_layoutgrid.kiwi, _kiwi)

    @unittest.skipIf(contourpy is None, "contourpy is not installed")
    def test_matches_contourpy(self):
        def reference(x, y, z, corner_mask, chunk):
            return contourpy.contour_generator(
                x, y, z, name="mpl2014", corner_mask=corner_mask,
                line_type=contourpy.LineType.SeparateCode,
                fill_type=contourpy.FillType.OuterCode, chunk_size=chunk)
        for i, args in enumerate(contour_fields()):
            a = contour_outputs(reference, *args)
            b = contour_outputs(port_generator, *args)
            for (va, ca), (vb, cb) in zip(a, b):
                with self.subTest(field=i):
                    self.assertEqual(len(va), len(vb))
                    for p, q in zip(va + ca, vb + cb):
                        self.assertEqual(p.dtype, q.dtype)
                        np.testing.assert_array_equal(p, q)


class QhullTests(unittest.TestCase):
    def points(self):
        rng = np.random.default_rng(5)
        for trial in range(150):
            k = int(rng.integers(3, 400))
            kind = trial % 5
            if kind == 0:
                yield rng.random(k), rng.random(k)
            elif kind == 1:
                yield rng.normal(size=k) * 1e3, rng.normal(size=k) * 1e-3
            elif kind == 2:  # cocircular points on a grid
                g = int(np.sqrt(k)) + 2
                xx, yy = np.meshgrid(np.arange(g), np.arange(g))
                yield xx.ravel().astype(float), yy.ravel().astype(float)
            elif kind == 3:  # a circle and its center
                t = rng.random(k) * 2 * np.pi
                yield np.append(np.cos(t), 0.0), np.append(np.sin(t), 0.0)
            else:  # duplicates
                yield (rng.integers(0, 10, k).astype(float), rng.integers(0, 10, k).astype(float))

    def test_triangulation(self):
        tri, nbr = _qhull.delaunay(np.array([0., 1, 0, 1]), np.array([0., 0, 1, 1]), 0)
        self.assertEqual(tri.shape, (2, 3))
        self.assertEqual(nbr.shape, (2, 3))
        self.assertIn("2020.2", _qhull.version())
        with self.assertRaisesRegex(ValueError, "3 unique points"):
            _qhull.delaunay(np.zeros(5), np.zeros(5), 0)
        with self.assertRaisesRegex(RuntimeError, "qhull"):
            _qhull.delaunay(np.arange(5.), np.arange(5.), 0)  # collinear

    @unittest.skipIf(mpl_qhull is None, "matplotlib is not installed")
    def test_matches_matplotlib(self):
        for i, (x, y) in enumerate(self.points()):
            with self.subTest(case=i):
                try:
                    a = mpl_qhull.delaunay(x, y, 0)
                except Exception as e:
                    a = (type(e), str(e))
                try:
                    b = _qhull.delaunay(x, y, 0)
                except Exception as e:
                    b = (type(e), str(e))
                if isinstance(a[0], type):
                    self.assertEqual(a, b)
                    continue
                for p, q in zip(a, b):
                    self.assertEqual(p.dtype, q.dtype)
                    np.testing.assert_array_equal(p, q)


@unittest.skipIf(PIL is None, "Pillow is not installed")
class PngTests(unittest.TestCase):
    def images(self):
        rng = np.random.default_rng(7)
        rgba = rng.integers(0, 256, (13, 17, 4), dtype=np.uint8)
        gray16 = rng.integers(0, 65536, (13, 17), dtype=np.uint16)
        yield "RGBA", PIL.Image.fromarray(rgba), {}
        yield "RGB", PIL.Image.fromarray(rgba[..., :3]), {}
        yield "L", PIL.Image.fromarray(rgba[..., 0]), {}
        yield "LA", PIL.Image.fromarray(rgba[..., :2], "LA"), {}
        yield "1", PIL.Image.fromarray(rgba[..., 0] > 128), {}
        yield "I;16", PIL.Image.fromarray(gray16), {}
        yield "P", PIL.Image.fromarray(rgba[..., :3]).quantize(37), {}
        yield "P-trns", PIL.Image.fromarray(rgba[..., :3]).quantize(5), {"transparency": 2}
        yield "P-4bit", PIL.Image.fromarray(rgba[..., :3]).quantize(12), {"bits": 4}
        yield "P-2bit", PIL.Image.fromarray(rgba[..., :3]).quantize(4), {"bits": 2}
        yield "RGBA-interlaced", PIL.Image.fromarray(rgba), {"interlace": True}
        yield "L-trns", PIL.Image.fromarray(rgba[..., 0]), {"transparency": 7}

    def test_read_matches_pillow(self):
        from matplotlib_rs.image import _pil_png_to_float_array
        for name, im, kw in self.images():
            with self.subTest(mode=name):
                buf = io.BytesIO()
                im.save(buf, format="png", **kw)
                expected = _pil_png_to_float_array(
                    PIL.PngImagePlugin.PngImageFile(io.BytesIO(buf.getvalue())))
                got = _png.read(io.BytesIO(buf.getvalue()))
                self.assertEqual(got.dtype, expected.dtype)
                np.testing.assert_array_equal(got, expected)

    def test_16bit_color(self):
        # Pillow reads 16-bit color samples as their most significant byte.
        import struct
        import zlib
        rng = np.random.default_rng(8)
        a = rng.integers(0, 65536, (5, 6, 3), dtype=np.uint16)
        raw = b"".join(b"\0" + row.astype(">u2").tobytes() for row in a)

        def chunk(kind, data):
            return (struct.pack(">I", len(data)) + kind + data
                    + struct.pack(">I", zlib.crc32(kind + data)))
        png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 6, 5, 16, 2, 0, 0, 0))
               + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
        from matplotlib_rs.image import _pil_png_to_float_array
        expected = _pil_png_to_float_array(PIL.PngImagePlugin.PngImageFile(io.BytesIO(png)))
        np.testing.assert_array_equal(_png.read(io.BytesIO(png)), expected)

    def test_write(self):
        rgba = np.random.default_rng(9).integers(0, 256, (9, 11, 4), dtype=np.uint8)
        data = _png.encode(rgba, dpi=72, text=[("Software", "matplotlib_rs"), ("Title", "t")])
        im = PIL.Image.open(io.BytesIO(data))
        np.testing.assert_array_equal(np.asarray(im), rgba)
        self.assertEqual(im.info["Software"], "matplotlib_rs")
        self.assertAlmostEqual(im.info["dpi"][0], 72, places=1)
        self.assertNotIn("dpi", PIL.Image.open(io.BytesIO(_png.encode(rgba))).info)


if __name__ == "__main__":
    unittest.main()
