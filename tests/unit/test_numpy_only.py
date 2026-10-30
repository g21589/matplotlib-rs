"""matplotlib_rs with NumPy as the only installed package.

A child interpreter runs without site-packages (``python -S``), with a path
that holds NumPy and matplotlib_rs only (or, if NUMPY_ONLY_PYTHON is set, that
interpreter, e.g. of a virtual environment with only NumPy installed).  It must not import any other
third-party package and its figures (numpy_only_figures.py) must be identical
to those of an interpreter with matplotlib's dependencies installed: the same
PNG pixels, the same EPS files, and SVG/PDF files that differ only in how
their raster images are encoded (Pillow is not available to encode them).
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tests/compare"))


def numpy_only_path(tmp):
    """A directory holding only (links to) NumPy."""
    site = Path(tmp) / "site"
    site.mkdir()
    numpy_dir = Path(np.__file__).resolve().parent
    for entry in [numpy_dir, numpy_dir.parent / "numpy.libs"]:
        if entry.exists():
            (site / entry.name).symlink_to(entry)
    return site


NUMPY_ONLY_PYTHON = os.environ.get("NUMPY_ONLY_PYTHON")


def run(code, path, env=None):
    if NUMPY_ONLY_PYTHON:
        cmd, path = [NUMPY_ONLY_PYTHON], [ROOT / "python"]
    else:
        cmd = [sys.executable, "-S"]
    prelude = f"import sys; sys.path[:0] = {[str(p) for p in path]!r}\n"
    return subprocess.run([*cmd, "-c", prelude + textwrap.dedent(code)],
                          capture_output=True, text=True, env={**os.environ, **(env or {})})


try:
    import PIL  # noqa: F401 (to compare the images)
except ImportError:
    PIL = None


@unittest.skipIf(PIL is None, "the comparisons need Pillow in this interpreter")
class NumpyOnlyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = [numpy_only_path(cls.tmp.name), ROOT / "python"]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_no_other_packages(self):
        p = run("""
            import importlib.util
            for name in ("cycler", "dateutil", "pyparsing", "kiwisolver", "contourpy",
                         "fontTools", "PIL", "packaging", "matplotlib"):
                assert importlib.util.find_spec(name) is None, name
            import matplotlib_rs
            matplotlib_rs.install()
            import matplotlib.pyplot as plt
            import matplotlib.dates, matplotlib.contour, matplotlib.tri, matplotlib.mathtext
            from matplotlib_rs import _contour, _kiwi, _layoutgrid, _png
            assert _layoutgrid.kiwi is _kiwi and _contour._contourpy("mpl2014") is None
            assert _png.pil_image() is None
            # Scripts written for matplotlib can import its pure-Python dependencies.
            from cycler import cycler
            import dateutil.parser
            assert cycler.__module__ == "matplotlib_rs._vendor.cycler"
            assert dateutil.parser.parse("2026-10-30").year == 2026
            third = sorted({m.split(".")[0] for m in sys.modules}
                           - set(sys.stdlib_module_names)
                           - {"numpy", "matplotlib_rs", "matplotlib", "__main__",
                              "cycler", "dateutil", "cython_runtime"}  # (numpy's)
                           - {m for m in sys.modules if m.startswith("_")})
            print("third-party:", third)
            assert not third, third
            """, self.path, {"MPLBACKEND": "Agg"})
        self.assertEqual(p.returncode, 0, p.stderr)

    def test_optional_features_need_their_package(self):
        p = run("""
            import io
            import matplotlib_rs.pyplot as plt
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "x")
            for fmt, needs in [("jpg", "Pillow"), ("tiff", "Pillow"), ("webp", "Pillow")]:
                try:
                    fig.savefig(io.BytesIO(), format=fmt)
                except ImportError as e:
                    assert needs in str(e), e
                else:
                    raise AssertionError(fmt)
            try:
                fig.savefig(io.BytesIO(), format="png", pil_kwargs={"optimize": True})
            except ImportError as e:
                assert "Pillow" in str(e)
            else:
                raise AssertionError("pil_kwargs")
            for key in ("pdf.fonttype", "ps.fonttype"):
                with plt.rc_context({key: 42}):
                    try:
                        fig.savefig(io.BytesIO(), format=key[:-9] if key[0] == "p" else "pdf")
                    except ImportError as e:
                        assert "fontTools" in str(e), e
                    else:
                        raise AssertionError(key)
            try:
                ax.contour([[1, 2], [3, 4]], algorithm="serial")
            except ImportError as e:
                assert "ContourPy" in str(e), e
            else:
                raise AssertionError("contour algorithm")
            """, self.path, {"MPLBACKEND": "Agg"})
        self.assertEqual(p.returncode, 0, p.stderr)

    def test_figures_identical_to_full_environment(self):
        script = HERE / "numpy_only_figures.py"
        out_np = Path(self.tmp.name) / "numpy_only"
        out_full = Path(self.tmp.name) / "full"
        env = {"MPLBACKEND": "Agg", "SOURCE_DATE_EPOCH": "0"}
        p = run(f"import runpy; sys.argv = ['x', {str(out_np)!r}]; "
                f"runpy.run_path({str(script)!r}, run_name='__main__')", self.path, env)
        self.assertEqual(p.returncode, 0, p.stderr)
        p = subprocess.run([sys.executable, str(script), str(out_full)], capture_output=True,
                           text=True, env={**os.environ, **env,
                                           "PYTHONPATH": str(ROOT / "python")})
        self.assertEqual(p.returncode, 0, p.stderr)

        import normalize
        from PIL import Image
        child_numpy = run("import numpy; print(numpy.__version__)", self.path).stdout.strip()
        # matplotlib's SVG clip ids hash NumPy reprs, which changed in NumPy 2.
        hashed_ids = child_numpy.split(".")[0] != np.__version__.split(".")[0]
        names = sorted(f.name for f in out_full.iterdir())
        self.assertEqual(names, sorted(f.name for f in out_np.iterdir()))
        for name in names:
            a, b = out_full / name, out_np / name
            with self.subTest(file=name):
                if name.endswith(".png"):
                    np.testing.assert_array_equal(np.asarray(Image.open(a).convert("RGBA")),
                                                  np.asarray(Image.open(b).convert("RGBA")))
                elif name.endswith(".npy"):
                    np.testing.assert_array_equal(np.load(a), np.load(b))
                elif name.endswith(".svg"):
                    self.assertEqual(normalize.svg(a.read_bytes(), hashed_ids),
                                     normalize.svg(b.read_bytes(), hashed_ids))
                elif name.endswith(".pdf"):
                    self.assertEqual(normalize.pdf(a.read_bytes()), normalize.pdf(b.read_bytes()))
                else:
                    self.assertEqual(a.read_bytes(), b.read_bytes())


if __name__ == "__main__":
    unittest.main()
