# matplotlib-rs

**Matplotlib with a Rust rendering core.** Your existing Matplotlib code runs unchanged and produces the same pixels, usually much faster. NumPy is the only dependency.

[![PyPI](https://img.shields.io/pypi/v/matplotlib-rs?label=PyPI)][pypi]
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)][pypi]
[![CI](https://github.com/g21589/matplotlib-rs/actions/workflows/ci.yml/badge.svg)][ci]
[![License: PSF-2.0](https://img.shields.io/badge/license-PSF--2.0-green)][license]

English · [繁體中文][readme-zh]

matplotlib-rs rewrites Matplotlib's rendering pipeline in Rust and keeps the Matplotlib 3.11 Python API on top of it. The Rust code covers Agg rasterization, font loading and text layout, the figure/axes/artist model, image resampling and image file codecs. Scripts written for Matplotlib run as-is, and across the project's test suite their output matches Matplotlib 3.11.2 pixel for pixel.

```python
import matplotlib_rs
matplotlib_rs.install()           # from now on, `import matplotlib` loads matplotlib-rs

import matplotlib.pyplot as plt   # your existing code, unchanged
```

[Installation](#installation) · [Quick start](#quick-start) · [Performance](#performance) · [Compatibility](#compatibility) · [Configuration](#configuration) · [Verification](#verification) · [Development](#development)

## Highlights

- **Drop-in replacement.** It implements the Matplotlib 3.11 API, both `pyplot` and the object-oriented interface. A single call makes `import matplotlib` and `import mpl_toolkits` load matplotlib-rs. You can also run an unmodified script with `python -m matplotlib_rs script.py`.
- **Identical output.** In every test case, PNG output is pixel-identical to Matplotlib 3.11.2, and SVG, PDF and EPS output matches the reference content.
- **Fast.** A bar chart with 20,000 bars is built and drawn 77× faster than with Matplotlib. A typical 2 × 2 figure with legends saves 6–7× faster, and 50,000 boxplots draw over 300× faster. Rasterization, colormapping, resampling, histograms, contours and PNG compression all run on multiple threads.
- **Only NumPy.** You don't need Pillow, kiwisolver or ContourPy. Image files are read and written in Rust. The Rust ports of kiwi and of ContourPy's default algorithm, and the bundled qhull, reproduce the original results bit for bit.
- **Checked before every release.** A release goes out only after the full CI suite passes, every wheel works in a clean environment that has only NumPy, and a performance check passes.

## Installation

```bash
pip install matplotlib-rs
```

| Platform | Support |
| --- | --- |
| Linux x86_64 (glibc ≥ 2.28), CPython 3.10–3.13 | Prebuilt wheels, tested in CI and in every release |
| Other platforms and Python versions | Builds from the source distribution, but hasn't been tested yet |

The only runtime dependency is **NumPy ≥ 1.25**. In a fresh environment, installing adds only `matplotlib-rs` and `numpy`.

Two features need optional packages, which you can add with `pip install "matplotlib-rs[full]"`:

| Feature | Package |
| --- | --- |
| Type 42 font embedding (`pdf.fonttype` or `ps.fonttype` set to `42`) | fontTools |
| The `serial`, `threaded` and `mpl2005` contour algorithms (`mpl2014`, the default, is built in) | ContourPy |

If one of these packages is missing, the feature raises an `ImportError` that explains what to install. Features that rely on external programs, such as usetex, GUI backends and movie writers, still need those programs.

**Building from source** needs a stable Rust toolchain and a C compiler (gcc or clang on Linux, MSVC on Windows). pip uses maturin to compile the Rust extension and the bundled qhull:

```bash
git clone --branch v0.0.7 https://github.com/g21589/matplotlib-rs.git
cd matplotlib-rs
python -m pip install .
```

PyPI and [GitHub Releases][releases] carry the same tested files. Each GitHub release also includes the source distribution, the raw benchmark records and a `SHA256SUMS` file.

## Quick start

```python
import matplotlib_rs

# Call this before anything imports matplotlib or mpl_toolkits.
matplotlib_rs.install()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 9], marker="o", label="y = x²")
ax.set(xlabel="x", ylabel="y", title="matplotlib-rs")
ax.legend()
fig.savefig("example.png", dpi=150)
plt.close(fig)
```

There are three ways to use matplotlib-rs:

| Approach | When to use it |
| --- | --- |
| `matplotlib_rs.install()` and then `import matplotlib` | Existing code and third-party libraries that import `matplotlib` |
| `python -m matplotlib_rs script.py [args]` or `python -m matplotlib_rs -m package.module` | Running unmodified scripts or modules |
| `import matplotlib_rs.pyplot as plt` | New code that names matplotlib-rs directly |

- Use only one implementation in each Python process. `install()` raises `RuntimeError` if `matplotlib` has already been imported.
- After `install()`, `from cycler import cycler` and `import dateutil` load the bundled copies if those packages aren't installed.
- `importlib.metadata.version("matplotlib-rs")` returns the package version. `matplotlib_rs.__version__` returns `3.11.2`, the Matplotlib API version that matplotlib-rs implements, so version checks in existing code keep working.

## Performance

These timings compare matplotlib-rs 0.0.7 with Matplotlib 3.11.2. Every run checks that the two canvases are pixel-identical. Chart timings include creating the figure and drawing it for the first time.

| Workload | Matplotlib 3.11.2 | matplotlib-rs 0.0.7 | Speedup |
| --- | ---: | ---: | ---: |
| 2 × 2 figure, four legends at `loc="best"`, `tight_layout`, save as PNG | 176.2 ms | **26.1 ms** | 6.8× |
| Bar chart, 20,000 bars | 9.355 s | **0.121 s** | 77× |
| Histogram, 2,000,000 values in 10,000 bins | 4.454 s | **0.082 s** | 54× |
| Boxplot, 1,000 boxes × 200 values | 3.519 s | **0.077 s** | 46× |
| `imshow`, 4000 × 4000 image | 1.550 s | **0.232 s** | 6.7× |
| Histogram, 10,000,000 values | 0.265 s | **0.053 s** | 5.0× |
| `hist2d`, 2,000,000 points | 0.519 s | **0.124 s** | 4.2× |
| Scatter, 200,000 points with per-point size and color | 2.286 s | **0.736 s** | 3.1× |
| `contourf`, 1000 × 1000 grid | 1.113 s | **0.422 s** | 2.6× |
| Line, 1,000,000 points | 0.127 s | 0.118 s | 1.1× |
| Pie, 500 wedges | 0.281 s | 0.285 s | 1.0× |

Each value is the median of five fresh processes on a 4-vCPU Intel Xeon VM at 2.10 GHz, running Linux x86_64, CPython 3.13 and NumPy 2.5.3. Your results will depend on your hardware and workload.

The biggest gains come where Matplotlib creates one Python object per element, such as bars, histogram bins and boxes, and from multi-threaded image, colormap, contour and path-collection work. Where Matplotlib already does the work in C++, for example simplifying long lines, the two are about even.

**Large boxplots.** A single `boxplot` call with 50,000 boxes takes 158.5 s to build and draw in Matplotlib and 0.53 s in matplotlib-rs, 301× faster. Peak memory falls from 4,123 MiB to 211 MiB. matplotlib-rs creates the `Line2D` and `PathPatch` artists only when your code reads them, and they behave like ordinary artists. These figures come from 0.0.6, as the median of five runs; Matplotlib was run once.

**PNG encoding.** matplotlib-rs encodes every PNG in Rust on multiple threads, using the same zlib level 6 that Matplotlib gets from Pillow. The resulting file doesn't depend on the number of threads. For nine typical figures, from 640 × 480 to 2400 × 1800:

| Encoder | Total time | Total size |
| --- | ---: | ---: |
| Matplotlib (Pillow, zlib level 6) | 724.7 ms | 3,150 kB |
| matplotlib-rs, default (zlib level 6) | **75.2 ms (9.6×)** | **3,061 kB** |
| matplotlib-rs, `pil_kwargs={"compress_level": 1}` | 23.1 ms (31×) | 4,931 kB |

The reports below give the methodology, per-round data, raw JSON and limitations. They are currently written in Traditional Chinese.

- [24 chart types with large data][bench-basic]
- [Worker threads, shared vertices and legend layout (0.0.7)][bench-threads]
- [Lazy boxplots and parallel drawing (0.0.6)][bench-lazy]
- [PNG encoding][bench-png]

To reproduce the benchmarks, install the reference Matplotlib from `requirements-ci.txt` and build the extension (see [Development](#development)), then run:

```bash
python tests/bench/basic_charts.py --rounds 3
python tests/bench/legends.py --rounds 5
python tests/bench/boxplot_lazy.py --boxes 5400 50000 --rounds 5 --mpl
python tests/bench/png_save.py
```

## Compatibility

### What's covered

The project's test suite checks the following features against Matplotlib:

| Area | Features |
| --- | --- |
| Basic charts | line, scatter, bar/barh, hist/hist2d, hexbin, pie, boxplot, violin, stem, step, stackplot |
| Fields and images | imshow, pcolormesh, contour, triangulations, quiver, streamplot, image resampling |
| Coordinates and layout | log, logit, dates, polar, 3D, subplots, subfigures, mosaic, tight and constrained layout, insets, secondary axes, twin axes |
| Text and styling | annotations, arrows, fonts, mathtext, text bounding boxes, legends, tables, styles, hatching, spines, ticks |
| Rendering and output | clip paths, blitting, animated artists, rasterization, PNG, SVG, PDF and EPS, transparent backgrounds, tight bounding boxes |

This table shows what the tests cover. It doesn't mean that every Matplotlib API, parameter or backend is supported.

### Replaced dependencies

matplotlib-rs provides everything Matplotlib normally installs alongside it:

| Matplotlib dependency | How matplotlib-rs replaces it |
| --- | --- |
| cycler, pyparsing, python-dateutil (with six) | Bundled unmodified in `matplotlib_rs._vendor`. An installed cycler takes precedence. |
| kiwisolver (constrained layout) | Rust port of the kiwi 1.5 solver, with the same order of operations. Layouts are deterministic. |
| ContourPy (contours) | Rust port of the default `mpl2014` algorithm, with bit-identical vertices |
| qhull (Delaunay triangulation) | Compiles in qhull 8.0.2, the version Matplotlib uses, so triangles come out in the same order |
| Pillow (image files, images embedded in PDF and SVG, `PillowWriter`) | Image reading and writing is done in Rust (see [Image file formats](#image-file-formats)) |
| fontTools, packaging | fontTools is needed only for Type 42 fonts. Version parsing is built in. |

matplotlib-rs never imports Pillow, even if it's installed. You can still pass PIL images to functions such as `imshow`.

### Image file formats

`savefig`, `imsave`, `imread`, `PillowWriter` and the images embedded in SVG, PDF and PGF files all go through the Rust core. The defaults match what Matplotlib produces through Pillow.

| Format | Writing | Reading (`imread`) |
| --- | --- | --- |
| PNG | Parallel compression at zlib level 6. Options: `compress_level`, `optimize`, `compress_type`, `pnginfo`, `icc_profile`, `exif` | Bit-identical to Matplotlib (float32 values from 0 to 1) |
| JPEG | Quality 75, 4:2:0 subsampling and JFIF dpi; large images are encoded in parallel segments. Options: `quality`, `subsampling`, `optimize`, `progressive`, `icc_profile`, `exif` | Values within ±8 of libjpeg-turbo, and usually identical. CMYK is converted as Pillow does. |
| TIFF | Uncompressed, like Pillow. `compression` can be LZW, Deflate or PackBits | Grayscale (8- and 16-bit), RGB(A), grayscale with alpha, 1-bit and CMYK images, read as Pillow reads them |
| WebP | Bundled libwebp 1.6.0, lossy quality 80, method 4. **Files are byte-identical to Pillow's.** | libwebp, same as Pillow |
| GIF | A separate palette for each frame. Exact for up to 256 colors, otherwise median cut | First frame as RGBA, same as Pillow |
| AVIF | rav1e, quality 75, speed 6 | Not supported |

`PillowWriter` (`writer="pillow"`) writes GIF, APNG (`.png`) and animated WebP. If you pass a `pil_kwargs` option that isn't supported, you get a `ValueError` that lists the options that are.

Encode and decode times compared with Pillow 13.0.0, in milliseconds (median of 7 runs on 4 vCPUs):

| Image | Format | Pillow write | matplotlib-rs write | Pillow read | matplotlib-rs read |
| --- | --- | ---: | ---: | ---: | ---: |
| 800 × 600 | JPEG | 2.8 | 3.1 | 2.0 | 1.4 |
| 800 × 600 | TIFF | 1.4 | 0.5 | 0.8 | 0.2 |
| 800 × 600 | WebP | 29.3 | 30.3 | 5.3 | 3.9 |
| 800 × 600 | GIF | 7.3 | 9.4 | 1.9 | 2.9 |
| 2400 × 1800 | JPEG | 28.7 | 16.5 | 18.4 | 11.9 |
| 2400 × 1800 | TIFF | 9.8 | 8.0 | 14.7 | 5.3 |
| 2400 × 1800 | WebP | 213.2 | 235.3 | 46.7 | 24.3 |
| 2400 × 1800 | GIF | 66.6 | 85.5 | 11.4 | 17.8 |

### Known limitations

- **Validated scope.** Testing covers non-interactive rendering with Agg and the vector backends on Linux x86_64. GUI backends, interactive use, external TeX and movie tools, and other platforms haven't been tested yet.
- **Official gallery.** In the last full run (0.0.1), the PNG output of 508 of 544 image-producing gallery scripts (93.4%) was identical to Matplotlib's. The other 36 had pixel differences, and no script failed with an error. The details are in the [0.0.1 validation report][release-001].
- **Embedded images.** Images embedded in SVG and PDF files are compressed by matplotlib-rs rather than Pillow. Their bytes differ from Matplotlib's files, but they decode to the same pixels.
- **Image codecs.** AVIF and palette TIFF files can't be read. JPEG and AVIF use different codecs from Pillow, so their output isn't guaranteed to be identical. For images with more than 256 colors, GIF palettes differ from Pillow's, and file sizes differ by 1–25%.
- **Pie charts and violin plots** run at about Matplotlib's speed. Pie charts still create one Python object per wedge, and violin plots spend most of their time in the NumPy density estimate.
- **Idle worker threads.** After the first parallel draw, a process keeps `MPLRS_THREADS - 1` idle worker threads. Forked child processes start their own.

## Configuration

| Environment variable | Default | Effect |
| --- | --- | --- |
| `MPLRS_THREADS` | Number of available cores, up to 8 | Sets the number of threads for parallel work: path collections, large batches of lines and patches, image resampling, colormapping, histograms, contours and PNG compression. `1` turns threading off. Output is the same for any thread count. |
| `MPLRS_LAZY` | `1` | `0` creates all boxplot artists immediately in `boxplot` and `bxp`. |
| `MPLRS_KIWI` | unset | `kiwisolver` uses the installed kiwisolver instead of the built-in solver. |
| `MPLRS_CONTOUR` | unset | `contourpy` uses the installed ContourPy for the `mpl2014` algorithm. |
| `MPLRS_QHULL` | unset | `rust` uses the Rust Delaunay triangulation. It produces the same triangles, possibly in a different order. |

Matplotlib's own settings, such as `MPLBACKEND`, `MPLCONFIGDIR`, `matplotlibrc` and style sheets, work as usual.

## How it works

```
your code
   │   import matplotlib  ──►  matplotlib_rs  (Python layer: the Matplotlib 3.11 API)
   ▼
_mplrs  (Rust core, connected to Python with PyO3 and rust-numpy)
   ├── mpl/       figure, axes and artist model, legends, mathtext, kiwi solver, mpl2014 contours
   ├── agg/       port of Anti-Grain Geometry 2.4 (rasterizer, scanlines, strokes)
   ├── font/      TrueType/OpenType/CFF loading and glyph rasterization, text layout
   ├── backend/   Agg renderer, path converters, image resampling, image codecs, PNG encoder
   └── pool.rs    worker threads that start once and are reused for parallel loops
extern/qhull      qhull 8.0.2, compiled into the extension
```

The Python package is adapted from Matplotlib 3.11, so public classes, signatures, rcParams and behavior follow Matplotlib. Drawing, layout and file output run in the Rust core.

## Verification

Every comparison uses **Matplotlib 3.11.2** as the reference, with the Agg backend and a fixed [`matplotlibrc`][mplrc]. A PNG passes only if every RGBA pixel is identical. SVG, PDF and EPS files are compared by content, with timestamps and the SVG hash salt fixed. The only line ignored is the output filename line in PostScript files. Raster images embedded in SVG and PDF files are compared after decoding.

| Check | Result |
| --- | --- |
| Plotting cases ([`tests/compare`][compare]), on every CI run | 59/59 identical in PNG, SVG, PDF and EPS |
| Large-data chart benchmark | 24/24 chart types pixel-identical |
| Python 3.10 with only NumPy installed, compared with Matplotlib on Python 3.12 | All 59 plotting cases identical. The minimum supported NumPy, 1.25.2, is tested too. |
| Low-level renderer and font tests against Matplotlib's C++ Agg and FreeType | Renderer, path collections, TrueType and CFF glyphs and the `ft2font` API are identical |
| kiwi port compared with kiwisolver | 6 constrained layouts are bit-identical, and 400 random systems agree within 1e-9 |
| `mpl2014` port compared with ContourPy | 3,600 line and fill results from 200 random fields are bit-identical |
| Bundled qhull compared with Matplotlib's `_qhull` | 150 point sets give identical triangles and neighbor arrays |
| Image resampling compared with Matplotlib's `_image.resample` | All 7,880 cases are bit-identical |
| Official Matplotlib gallery (0.0.1) | 508/544 scripts pixel-identical, with no runtime errors |

The [CI/CD documentation][ci-doc] describes how these checks run. The [0.0.1 validation report][release-001] covers the original full validation, including the gallery run and the remaining differences.

## Development

You need a stable Rust toolchain, a C compiler and Python 3.11–3.13. matplotlib-rs itself supports Python 3.10, but the reference Matplotlib 3.11.2 used by the tests needs 3.11 or later. `requirements-ci.txt` pins the reference packages and build tools:

```bash
git clone https://github.com/g21589/matplotlib-rs.git
cd matplotlib-rs
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements-ci.txt

sh scripts/dev_build.sh --locked   # build the extension into python/matplotlib_rs (Linux)
sh scripts/test_rust.sh            # Rust unit tests
bash scripts/ci_test.sh            # the full CI suite: unit, low-level and image comparisons
```

To use the development build directly, set `PYTHONPATH="$PWD/python"`. Test logs and images are written to `tests/output/`. To build distributions:

```bash
maturin build --release --locked --out dist   # wheel
maturin sdist --out dist                      # source distribution
```

| Path | Contents |
| --- | --- |
| `python/matplotlib_rs/` | Python package with the Matplotlib-compatible API |
| `src/` | Rust core |
| `extern/qhull/` | Bundled qhull sources |
| `tests/compare/` | Cases rendered by both implementations and compared in four formats |
| `tests/unit/`, `tests/lowlevel/`, `tests/model/` | Unit tests and low-level comparison tests |
| `tests/bench/` | Benchmarks |
| `tests/gallery/` | Runner for the official Matplotlib gallery |
| `scripts/`, `tools/` | Build, test, release and vendoring helpers |
| `docs/` | Release notes, benchmark reports, CI/CD documentation |

## Releases

Pushing a `vX.Y.Z` tag starts the release workflow:

1. It checks that the tag and the versions in `pyproject.toml`, `Cargo.toml` and `Cargo.lock` match, and that release notes exist for that version.
2. It runs the full CI suite and builds manylinux_2_28 x86_64 wheels for CPython 3.10–3.13, plus a source distribution.
3. It installs each wheel in a clean environment that has only NumPy and renders sample figures in PNG, SVG, PDF and EPS.
4. It benchmarks 5,400 boxplots against `v0.0.1` on the same runner. The release requires identical pixels and a median build time at least 10× faster.
5. It publishes the GitHub release with `SHA256SUMS`. PyPI then receives the same verified files through Trusted Publishing.

See the [CI/CD documentation][ci-doc] for details.

| Version | Highlights |
| --- | --- |
| [0.0.7][rn-007] | Reusable worker threads, shared vertex buffers, faster legend placement and text layout |
| [0.0.6][rn-006] | Boxplot artists created only when read; parallel drawing of large artist batches |
| [0.0.5][rn-005] | All image file reading and writing moved to Rust, with no Pillow; parallel PNG encoder |
| [0.0.4][rn-004] | Large-data speedups for bar, hist, imshow, hist2d, contour, pcolormesh and more |
| [0.0.3][rn-003] | Python 3.10 support; NumPy as the only runtime dependency |
| [0.0.2][rn-002] | Faster creation of many boxplots; GitHub Actions CI/CD |
| [0.0.1][release-001] | First release, with a full validation report |

matplotlib-rs is in early development, and version numbers follow `0.0.x`. The public API follows Matplotlib 3.11.2.

## License

matplotlib-rs is released under the [PSF-2.0 license][license]. It contains Python code and data derived from Matplotlib and keeps Matplotlib's license and copyright notices. Bundled third-party code keeps its own license:

| Component | License |
| --- | --- |
| cycler | [BSD][lic-cycler] |
| pyparsing | [MIT][lic-pyparsing] |
| python-dateutil | [Apache-2.0 / BSD][lic-dateutil] |
| six | [MIT][lic-six] |
| qhull | [Qhull][lic-qhull] |
| libwebp (through libwebp-sys) | [BSD-3-Clause with patent grant][lic-libwebp] |
| Rust port of the kiwi solver | [BSD][lic-kiwi] |
| Rust port of ContourPy's `mpl2014` algorithm | [BSD-3-Clause][lic-contourpy] |
| Rust port of Anti-Grain Geometry 2.4 | AGG license; the notice is kept in `src/agg` |
| jpeg-encoder (JPEG encoding) | Partly derived from Independent JPEG Group code (IJG license) |
| rav1e (AVIF encoding) | BSD-2-Clause with the AOMedia patent license |

matplotlib-rs is an independent project. It isn't affiliated with or endorsed by the Matplotlib development team.

[pypi]: https://pypi.org/project/matplotlib-rs/
[ci]: https://github.com/g21589/matplotlib-rs/actions/workflows/ci.yml
[releases]: https://github.com/g21589/matplotlib-rs/releases
[readme-zh]: https://github.com/g21589/matplotlib-rs/blob/HEAD/README.zh-TW.md
[license]: https://github.com/g21589/matplotlib-rs/blob/HEAD/licenses/LICENSE_MATPLOTLIB
[mplrc]: https://github.com/g21589/matplotlib-rs/blob/HEAD/tests/mplrc/matplotlibrc
[compare]: https://github.com/g21589/matplotlib-rs/tree/HEAD/tests/compare
[ci-doc]: https://github.com/g21589/matplotlib-rs/blob/HEAD/docs/CI.md
[bench-basic]: https://github.com/g21589/matplotlib-rs/blob/HEAD/docs/benchmarks/basic-charts.md
[bench-threads]: https://github.com/g21589/matplotlib-rs/blob/HEAD/docs/benchmarks/threads-legends-lines.md
[bench-lazy]: https://github.com/g21589/matplotlib-rs/blob/HEAD/docs/benchmarks/boxplots-lazy.md
[bench-png]: https://github.com/g21589/matplotlib-rs/blob/HEAD/docs/benchmarks/png-save.md
[release-001]: https://github.com/g21589/matplotlib-rs/blob/HEAD/docs/releases/0.0.1.md
[rn-002]: https://github.com/g21589/matplotlib-rs/blob/HEAD/docs/releases/0.0.2.md
[rn-003]: https://github.com/g21589/matplotlib-rs/blob/HEAD/docs/releases/0.0.3.md
[rn-004]: https://github.com/g21589/matplotlib-rs/blob/HEAD/docs/releases/0.0.4.md
[rn-005]: https://github.com/g21589/matplotlib-rs/blob/HEAD/docs/releases/0.0.5.md
[rn-006]: https://github.com/g21589/matplotlib-rs/blob/HEAD/docs/releases/0.0.6.md
[rn-007]: https://github.com/g21589/matplotlib-rs/blob/HEAD/docs/releases/0.0.7.md
[lic-cycler]: https://github.com/g21589/matplotlib-rs/blob/HEAD/licenses/LICENSE_CYCLER
[lic-pyparsing]: https://github.com/g21589/matplotlib-rs/blob/HEAD/licenses/LICENSE_PYPARSING
[lic-dateutil]: https://github.com/g21589/matplotlib-rs/blob/HEAD/licenses/LICENSE_DATEUTIL
[lic-six]: https://github.com/g21589/matplotlib-rs/blob/HEAD/licenses/LICENSE_SIX
[lic-qhull]: https://github.com/g21589/matplotlib-rs/blob/HEAD/licenses/LICENSE_QHULL
[lic-libwebp]: https://github.com/g21589/matplotlib-rs/blob/HEAD/licenses/LICENSE_LIBWEBP
[lic-kiwi]: https://github.com/g21589/matplotlib-rs/blob/HEAD/licenses/LICENSE_KIWISOLVER
[lic-contourpy]: https://github.com/g21589/matplotlib-rs/blob/HEAD/licenses/LICENSE_CONTOURPY
