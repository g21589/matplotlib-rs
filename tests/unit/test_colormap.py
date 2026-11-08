"""`Colormap._get_rgba_and_mask`'s Rust lookup of large float arrays and the
image path (colormap indices, with `Normalize` fused) give numpy's
results."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from matplotlib_rs import _colormap, colors  # noqa: E402
import matplotlib_rs.pyplot as plt  # noqa: E402


def numpy_lookup(cmap, X, alpha=None, bytes=False):
    saved = _colormap.MIN_SIZE
    _colormap.MIN_SIZE = 1 << 62
    try:
        return cmap._get_rgba_and_mask(X, alpha=alpha, bytes=bytes)
    finally:
        _colormap.MIN_SIZE = saved


def special_values(n, dtype):
    """Values around every colormap boundary and the special floats."""
    k = np.arange(-2, n + 3, dtype=np.float64) / n
    edges = np.concatenate([k, np.nextafter(k, 9), np.nextafter(k, -9)])
    other = np.array([0.0, -0.0, 1.0, np.nextafter(1.0, 0), np.nextafter(1.0, 2), np.nan, np.inf,
                      -np.inf, 1e300, -1e300, 5e-324, -5e-324, 1e-30, 0.5, 2.0, -1.0])
    with np.errstate(over="ignore"):
        return np.concatenate([edges, other]).astype(dtype)


def datasets(rng, n):
    for dtype in (np.float64, np.float32):
        v = special_values(n, dtype)
        yield f"{np.dtype(dtype).name} special", np.resize(v, 70_000)
        a = rng.uniform(-0.3, 1.3, (300, 250)).astype(dtype)
        a[rng.random(a.shape) < 0.02] = np.nan
        yield f"{np.dtype(dtype).name} uniform", a
        yield f"{np.dtype(dtype).name} non-contiguous", a[:, ::2]
        yield f"{np.dtype(dtype).name} 3d", a.reshape(30, 50, 50)
        m = np.ma.masked_where(rng.random(a.shape) < 0.1, a)
        yield f"{np.dtype(dtype).name} masked", m
        yield f"{np.dtype(dtype).name} empty mask", np.ma.masked_array(a, mask=np.zeros(a.shape, bool))
        yield f"{np.dtype(dtype).name} nomask", np.ma.masked_array(a)
        yield f"{np.dtype(dtype).name} all masked", np.ma.masked_array(a, mask=True)


def colormaps(rng):
    yield plt.get_cmap("viridis")
    yield plt.get_cmap("gray")
    yield plt.get_cmap("viridis", 5)
    yield plt.get_cmap("magma").with_extremes(bad=(1, 0, 0, 0.5), under="w", over=(0, 0, 0, 0.3))
    yield colors.ListedColormap(rng.random((3, 4)))
    yield colors.LinearSegmentedColormap.from_list("two", ["r", (0, 0, 1, 0.2)], N=1)


class LookupTests(unittest.TestCase):
    def test_same_as_numpy(self):
        rng = np.random.default_rng(3)
        used = 0
        for cmap in colormaps(rng):
            cmap._ensure_inited()
            for name, X in datasets(rng, cmap.N):
                for byts in (False, True):
                    with self.subTest(cmap.name, data=name, bytes=byts):
                        found = _colormap.get_rgba_and_mask(cmap, X, None, byts)
                        self.assertIsNotNone(found)
                        used += 1
                        rgba, mask = found
                        e_rgba, e_mask = numpy_lookup(cmap, X, bytes=byts)
                        self.assertEqual(rgba.dtype, e_rgba.dtype)
                        self.assertEqual(rgba.shape, e_rgba.shape)
                        np.testing.assert_array_equal(rgba, e_rgba)
                        np.testing.assert_array_equal(mask, e_mask)
                        self.assertEqual(np.shape(mask), np.shape(e_mask))
                        np.testing.assert_array_equal(cmap(X, bytes=byts), e_rgba)
        self.assertGreater(used, 100)

    def test_left_to_numpy(self):
        cmap = plt.get_cmap("viridis")
        a = np.random.default_rng(4).random((200, 200))
        for X, alpha in [(a, 0.5), ((a * 255).astype(np.uint8), None), (a.astype(np.float16), None),
                         (a[:10, :10], None), (a.astype(">f8"), None), (list(a[0]), None)]:
            self.assertIsNone(_colormap.get_rgba_and_mask(cmap, X, alpha, False))
            np.testing.assert_array_equal(cmap(X, alpha=alpha), numpy_lookup(cmap, X, alpha=alpha)[0])


class ImageIndicesTests(unittest.TestCase):
    """`image_rgba` gives lut rows whose lut values are ``image.to_rgba(A)``."""

    def check(self, A, **kwargs):
        fig, ax = plt.subplots()
        im = ax.imshow(A, **kwargs)
        A = im._A
        im.norm.autoscale_None(A)
        expected = im.to_rgba(A)
        got = _colormap.image_rgba(im, A)
        plt.close(fig)
        if isinstance(got, _colormap.Indexed):
            got = got.lut[got.idx]
            indexed = True
        else:
            indexed = False
        self.assertEqual(got.dtype, expected.dtype)
        np.testing.assert_array_equal(got, expected)
        return indexed

    def test_same_rgba(self):
        rng = np.random.default_rng(5)
        z = rng.normal(size=(200, 300))
        z[rng.random(z.shape) < 0.02] = np.nan
        z[rng.random(z.shape) < 0.01] = np.inf
        extremes = plt.get_cmap("viridis").with_extremes(bad="r", under="w", over="k")
        cases = [
            (z, {}, True),
            (z, {"vmin": -1, "vmax": 1.5, "cmap": extremes}, True),
            (np.ma.masked_where(z > 1, z), {"cmap": "coolwarm"}, True),
            (z.astype(np.float32), {}, True),
            (z, {"norm": colors.LogNorm(1e-3, 3)}, True),
            (z, {"norm": colors.Normalize(-1, 1, clip=True)}, True),
            (z, {"norm": colors.TwoSlopeNorm(0.2, -2, 3)}, True),
            (np.full((200, 200), 3.0), {}, True),
            (z, {"vmin": 1e-300, "vmax": 2e-300}, True),
            ((z * 9).round(), {"vmin": -3, "vmax": 3, "cmap": plt.get_cmap("viridis", 6)}, True),
            (np.clip(np.nan_to_num(z * 50), -900, 900).astype(np.int32), {}, True),
            (np.clip(np.nan_to_num(z * 50), -900, 900).astype(np.int16), {"cmap": "tab20"}, True),
            (np.clip(np.nan_to_num(z * 50), 0, 255).astype(np.uint8), {}, True),
            (z, {"alpha": np.ones(z.shape) * 0.5}, False),
            (z, {"alpha": 0.5}, True),
            (z[:50, :50], {}, False),
        ]
        for A, kwargs, indexed in cases:
            with self.subTest(dtype=A.dtype, shape=A.shape, **{k: str(v)[:30] for k, v in kwargs.items()}):
                self.assertEqual(self.check(A, **kwargs), indexed)


if __name__ == "__main__":
    unittest.main()
