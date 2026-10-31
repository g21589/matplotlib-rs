"""`Axes.hist`'s Rust counting (axes/_hist.py) gives numpy.histogram's results."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from matplotlib_rs.axes import _hist


def datasets(rng):
    n = 200_000
    edges = np.linspace(-3, 3, 101)
    yield "normal", rng.normal(size=n)
    yield "on edges", rng.choice(edges, n)
    yield "on edges and neighbours", np.concatenate(
        [edges, np.nextafter(edges, np.inf), np.nextafter(edges, -np.inf)] * 700)
    yield "ties", np.round(rng.normal(size=n), 1)
    yield "integers", rng.integers(-50, 50, n).astype(float)
    yield "constant", np.full(n, 3.25)
    yield "nan and inf", np.where(rng.random(n) < 0.05, np.nan, rng.normal(size=n))
    yield "tiny range", 1 + rng.random(n) * 1e-12
    yield "huge range", rng.normal(size=n) * 1e300
    yield "subnormal", rng.random(n) * 5e-324 * 1000
    yield "non-contiguous", rng.normal(size=(n, 2))[:, 1]
    yield "2d", rng.normal(size=(500, 400))


class HistogramTests(unittest.TestCase):
    def assert_same(self, a, **kwargs):
        try:
            expected = np.histogram(a, **kwargs)
        except Exception as e:
            with self.assertRaises(type(e)):
                _hist.histogram(a, **kwargs)
            return
        actual = _hist.histogram(a, **kwargs)
        np.testing.assert_array_equal(actual[0], expected[0])
        self.assertEqual(actual[0].dtype, expected[0].dtype)
        np.testing.assert_array_equal(actual[1], expected[1])

    def test_same_counts(self):
        rng = np.random.default_rng(7)
        finite = None
        for name, a in datasets(rng):
            finite = a[np.isfinite(a)]
            lo, hi = (finite.min(), finite.max()) if finite.size else (0, 1)
            options = [{"bins": 10}, {"bins": 1}, {"bins": 200}, {"bins": 10_000},
                       {"bins": 7, "range": (lo, hi)}, {"bins": 33, "range": (-1.0, 0.5)},
                       {"bins": 50, "range": (np.float64(lo) - 1, np.float64(hi) / 3)},
                       {"bins": np.sort(rng.normal(size=40))},
                       {"bins": np.array([-np.inf, -1, 0, 0, 0.5, 1, np.inf])},
                       {"bins": np.linspace(lo, hi, 64)},
                       {"bins": "auto"}]
            for opts in options:
                with self.subTest(name, **{k: str(v)[:40] for k, v in opts.items()}):
                    self.assert_same(a, **opts)

    def test_fallbacks(self):
        rng = np.random.default_rng(8)
        a = rng.normal(size=100_000)
        self.assert_same(a, bins=20, weights=rng.random(a.size))
        self.assert_same(a, bins=20, density=True)
        self.assert_same(a.astype(np.float32), bins=20)
        self.assert_same(a, bins=20, range=(0, 1))
        self.assert_same(a, bins=[3, 2, 1])
        self.assert_same(a, bins=0)


if __name__ == "__main__":
    unittest.main()
