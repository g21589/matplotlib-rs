"""Vendor pure-Python modules of matplotlib 3.11 into matplotlib_rs.

Usage: python tools/vendor.py [module.py ...]   (default: all of MODULES)

Imports of ``matplotlib`` are rewritten to ``matplotlib_rs`` and the small
adaptations listed in PATCHES are applied (each patch must match exactly
once, so an upstream change is noticed instead of silently ignored).
Modules not listed here (artist, lines, text, ...) were vendored earlier and
edited in place; they are not touched by this script.
"""
import pathlib
import re
import sys

import matplotlib

SRC = pathlib.Path(matplotlib.__file__).parent
DST = pathlib.Path(__file__).resolve().parent.parent / "python" / "matplotlib_rs"
HEADER = ("# Vendored from matplotlib 3.11 (see licenses/LICENSE_MATPLOTLIB), "
          "adapted for matplotlib_rs.\n")

MODULES = [
    "_afm.py", "_animation_data.py", "_blocking_input.py", "_constrained_layout.py",
    "_internal_utils.py", "_layoutgrid.py", "_mathtext.py", "_mathtext_data.py",
    "_pylab_helpers.py", "_style_helpers.py", "_text_helpers.py", "_tight_bbox.py",
    "_tight_layout.py", "_type1font.py", "animation.py", "backend_bases.py",
    "backend_managers.py", "backend_tools.py", "category.py", "collections.py",
    "colorbar.py", "colorizer.py", "container.py", "contour.py", "dates.py",
    "dviread.py", "figure.py", "gridspec.py", "image.py", "inset.py",
    "layout_engine.py", "legend.py", "legend_handler.py", "mathtext.py",
    "offsetbox.py", "patheffects.py", "pylab.py", "pyplot.py", "quiver.py",
    "sankey.py", "spines.py", "stackplot.py", "streamplot.py", "table.py",
    "texmanager.py", "textpath.py", "typing.py", "units.py", "widgets.py",
    "axes/__init__.py", "axes/_axes.py", "axes/_base.py", "axes/_secondary_axes.py",
    "backends/__init__.py", "backends/registry.py",
    "backends/_backend_pdf_ps.py", "backends/backend_mixed.py", "backends/backend_pdf.py",
    "backends/backend_pgf.py", "backends/backend_ps.py", "backends/backend_svg.py",
    "backends/backend_template.py",
    "projections/__init__.py", "projections/geo.py", "projections/polar.py",
    "style/__init__.py", "style/core.py",
    "tri/__init__.py", "tri/_triangulation.py", "tri/_tricontour.py",
    "tri/_trifinder.py", "tri/_triinterpolate.py", "tri/_tripcolor.py",
    "tri/_triplot.py", "tri/_trirefine.py", "tri/_tritools.py",
    "_axis_py.py",
    "mpl_toolkits/mplot3d/__init__.py", "mpl_toolkits/mplot3d/art3d.py",
    "mpl_toolkits/mplot3d/axes3d.py", "mpl_toolkits/mplot3d/axis3d.py",
    "mpl_toolkits/mplot3d/proj3d.py",
    "mpl_toolkits/axes_grid1/__init__.py", "mpl_toolkits/axes_grid1/anchored_artists.py",
    "mpl_toolkits/axes_grid1/axes_divider.py", "mpl_toolkits/axes_grid1/axes_grid.py",
    "mpl_toolkits/axes_grid1/axes_rgb.py", "mpl_toolkits/axes_grid1/axes_size.py",
    "mpl_toolkits/axes_grid1/inset_locator.py", "mpl_toolkits/axes_grid1/mpl_axes.py",
    "mpl_toolkits/axes_grid1/parasite_axes.py",
    "mpl_toolkits/axisartist/__init__.py", "mpl_toolkits/axisartist/angle_helper.py",
    "mpl_toolkits/axisartist/axes_divider.py", "mpl_toolkits/axisartist/axis_artist.py",
    "mpl_toolkits/axisartist/axisline_style.py", "mpl_toolkits/axisartist/axislines.py",
    "mpl_toolkits/axisartist/floating_axes.py", "mpl_toolkits/axisartist/grid_finder.py",
    "mpl_toolkits/axisartist/grid_helper_curvelinear.py",
    "mpl_toolkits/axisartist/parasite_axes.py",
]

# vendored under another name: {destination: source}
SOURCES = {
    # matplotlib's own Axis (Python ticks), for the projections whose axes
    # are drawn in Python (polar, geo); rectilinear axes use axis.py
    "_axis_py.py": "axis.py",
}

PATCHES = {
    "backend_bases.py": [
        # import time: texmanager pulls in dviread and fontTools; load it on use
        ("from matplotlib_rs.texmanager import TexManager\n", ""),
        ("            self._texmanager = TexManager()",
         "            from matplotlib_rs.texmanager import TexManager\n"
         "            self._texmanager = TexManager()"),
        # ... and the text-to-path converter (mathtext's parser) as well
        ("        self._text2path = text.TextToPath()\n", ""),
        ("    def open_group(self, s, gid=None):",
         "    @property\n"
         "    def _text2path(self):\n"
         "        t = self.__dict__.get('_rs_text2path')\n"
         "        if t is None:\n"
         "            t = self.__dict__['_rs_text2path'] = text.TextToPath()\n"
         "        return t\n\n"
         "    @_text2path.setter\n"
         "    def _text2path(self, value):\n"
         "        self.__dict__['_rs_text2path'] = value\n\n"
         "    def open_group(self, s, gid=None):"),
        # file formats are served by matplotlib_rs' own backends
        ("_default_backends = {", "_default_backends = {  # matplotlib_rs backends"),
        ('if mod.startswith(("matplotlib.", "mpl_toolkits.")):',
         'if mod.startswith(("matplotlib.", "matplotlib_rs.", "mpl_toolkits.")):'),
        ('sys.modules.get("matplotlib.pyplot")', 'sys.modules.get("matplotlib_rs.pyplot")', 2),
        (None, "\n\n_default_backends = {k: v.replace('matplotlib.', 'matplotlib_rs.', 1)\n"
               "                     for k, v in _default_backends.items()}\n"),
    ],
    "axes/_base.py": [
        ("noclip = (_AxesBase, maxis.Axis,",
         "noclip = (_AxesBase, maxis.Axis, *maxis._rs_other_axis_classes,"),
        # Rust-backed state and drawing (see axes/_base_rs.py)
        (None, "\n\nfrom ._base_rs import _patch_axes_base  # noqa: E402\n"
               "_patch_axes_base(_AxesBase)\n"),
    ],
    "category.py": [
        # import time: dateutil's parser is imported when first needed
        ("import dateutil.parser\n", ""),
        ("            try:\n                dateutil.parser.parse(val)",
         "            try:\n                import dateutil.parser\n                dateutil.parser.parse(val)"),
    ],
    "projections/__init__.py": [
        # import time: polar, geo and 3d axes (~15 ms) are loaded when used
        ("""from .geo import AitoffAxes, HammerAxes, LambertAxes, MollweideAxes
from .polar import PolarAxes

try:
    from matplotlib_rs.mpl_toolkits.mplot3d import Axes3D
except Exception:
    import warnings
    warnings.warn("Unable to import Axes3D. This may be due to multiple versions of "
                  "Matplotlib being installed (e.g. as a system package and as a pip "
                  "package). As a result, the 3D projection is not available.")
    Axes3D = None
""", """import importlib

# projection name -> (module, class), imported on first use
_LAZY_PROJECTIONS = {
    "aitoff": ("matplotlib_rs.projections.geo", "AitoffAxes"),
    "hammer": ("matplotlib_rs.projections.geo", "HammerAxes"),
    "lambert": ("matplotlib_rs.projections.geo", "LambertAxes"),
    "mollweide": ("matplotlib_rs.projections.geo", "MollweideAxes"),
    "polar": ("matplotlib_rs.projections.polar", "PolarAxes"),
    "3d": ("matplotlib_rs.mpl_toolkits.mplot3d", "Axes3D"),
}
_LAZY_NAMES = {cls: (mod, cls) for mod, cls in _LAZY_PROJECTIONS.values()}


def __getattr__(name):
    if name in _LAZY_NAMES:
        mod, cls = _LAZY_NAMES[name]
        value = getattr(importlib.import_module(mod), cls)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
"""),
        ("""    def get_projection_class(self, name, _error_cls=KeyError):
        \"\"\"Get a projection class from its *name*.\"\"\"
        return""", """    def get_projection_class(self, name, _error_cls=KeyError):
        \"\"\"Get a projection class from its *name*.\"\"\"
        if name not in self._all_projection_types and name in _LAZY_PROJECTIONS:
            mod, cls = _LAZY_PROJECTIONS[name]
            self.register(getattr(importlib.import_module(mod), cls))
        return"""),
        ("""        return sorted(self._all_projection_types)""",
         """        return sorted({*self._all_projection_types, *_LAZY_PROJECTIONS})"""),
        ("""projection_registry.register(
    axes.Axes,
    PolarAxes,
    AitoffAxes,
    HammerAxes,
    LambertAxes,
    MollweideAxes,
)
if Axes3D is not None:
    projection_registry.register(Axes3D)
else:
    # remove from namespace if not importable
    del Axes3D
""", """projection_registry.register(axes.Axes)
"""),
    ],
    "pyplot.py": [
        ("from matplotlib_rs.projections import PolarAxes\n", ""),
        ("\n    if not isinstance(ax, PolarAxes):",
         "\n    from matplotlib_rs.projections import PolarAxes\n"
         "    if not isinstance(ax, PolarAxes):", 2),
        ("\n        if not isinstance(ax, PolarAxes):",
         "\n        from matplotlib_rs.projections import PolarAxes\n"
         "        if not isinstance(ax, PolarAxes):"),
    ],
    "style/__init__.py": [
        # import time: the style files (~15 ms) are read when first needed
        ("""_base_library = _read_style_directory(_BASE_LIBRARY_PATH)
library = {}
available = []


def reload_library():
    \"\"\"Reload the style library.\"\"\"
    library.clear()""", """_base_library = None


def _load_library():
    global _base_library, library, available
    if _base_library is None:
        _base_library = _read_style_directory(_BASE_LIBRARY_PATH)
        library = {}
        available = []
        reload_library()


def __getattr__(name):
    if name in ("library", "available"):
        _load_library()
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def reload_library():
    \"\"\"Reload the style library.\"\"\"
    if _base_library is None:
        return _load_library()
    library.clear()"""),
        ("\n\nreload_library()\n", "\n"),
        ("""    if isinstance(style, (str, Path)) or hasattr(style, 'keys'):
        # If name is a single str, Path or dict, make it a single element list.""",
         """    _load_library()
    if isinstance(style, (str, Path)) or hasattr(style, 'keys'):
        # If name is a single str, Path or dict, make it a single element list."""),
    ],
    "image.py": [
        # import time: Pillow (~25 ms) is imported on first use
        ("import PIL.Image\nimport PIL.PngImagePlugin\n",
         "import sys\n\n\n"
         "class _LazyPIL:\n"
         "    \"\"\"Pillow, imported when first used.\"\"\"\n\n"
         "    def __getattr__(self, name):\n"
         "        import PIL.Image\n"
         "        import PIL.PngImagePlugin\n"
         "        globals()['PIL'] = PIL\n"
         "        return getattr(PIL, name)\n\n\n"
         "PIL = _LazyPIL()\n"),
        ("        if isinstance(A, PIL.Image.Image):",
         "        if 'PIL.Image' in sys.modules and isinstance(A, PIL.Image.Image):"),
    ],
    "axes/_secondary_axes.py": [
        # the Rust core draws the axes: run SecondaryAxis.draw's preparation
        (None, "\n\ndef _rs_prepare_draw(self, renderer):\n"
               "    self._set_lims()\n"
               "    # this sets the scale in case the parent has set its scale.\n"
               "    self._set_scale()\n"
               "    super(SecondaryAxis, self)._rs_prepare_draw(renderer)\n\n\n"
               "SecondaryAxis._rs_prepare_draw = _rs_prepare_draw\n"),
    ],
    "_axis_py.py": [
        (None, "\n\nfrom . import axis as _rs_axis  # noqa: E402\n"
               "_rs_axis._rs_other_axis_classes.append(Axis)\n"),
    ],
    "mpl_toolkits/mplot3d/axis3d.py": [
        # 3D axes are drawn by matplotlib's Python code (see _axis_py.py)
        ("axis as maxis,", "_axis_py as maxis,"),
    ],
    "projections/polar.py": [
        ("import matplotlib_rs.axis as maxis", "import matplotlib_rs._axis_py as maxis"),
    ],
    "projections/geo.py": [
        ("import matplotlib_rs.axis as maxis", "import matplotlib_rs._axis_py as maxis"),
    ],
    "figure.py": [
        (None, "\n\nfrom ._figure_rs import _patch_figure  # noqa: E402\n"
               "_patch_figure(FigureBase, Figure, SubFigure)\n"),
    ],
    "spines.py": [
        (None, "\n\nfrom ._spines_rs import _patch_spines, _patch_spines_container  # noqa: E402\n"
               "_patch_spines(Spine)\n"
               "_patch_spines_container(Spines, Spine)\n"),
    ],
    "backends/registry.py": [
        ('''        return (backend[9:] if backend.startswith("module://")
                else f"matplotlib.backends.backend_{backend}")''',
         '''        if backend.startswith("module://"):
            return backend[9:]
        # matplotlib_rs renders with its Rust Agg port; interactive and
        # vector backends fall back to it.
        if backend in _RS_BACKENDS:
            return f"{__package__}.backend_{backend}"
        return f"{__package__}.backend_agg"'''),
        ("class BackendFilter(Enum):", '_RS_BACKENDS = {"agg"}\n\n\nclass BackendFilter(Enum):'),
    ],
}


def rewrite_imports(s):
    s = re.sub(r"^(\s*)from mpl_toolkits((?:\.[\w]+)*) import", r"\1from matplotlib_rs.mpl_toolkits\2 import",
               s, flags=re.M)
    s = re.sub(r"^(\s*)import mpl_toolkits((?:\.[\w]+)+) as (\w+)",
               r"\1import matplotlib_rs.mpl_toolkits\2 as \3", s, flags=re.M)
    s = re.sub(r"^(\s*)import matplotlib as mpl\b", r"\1import matplotlib_rs as mpl", s, flags=re.M)
    s = re.sub(r"^(\s*)import matplotlib(\s*(#.*)?)$", r"\1import matplotlib_rs as matplotlib\2", s,
               flags=re.M)
    s = re.sub(r"^(\s*)from matplotlib((?:\.[\w]+)*) import", r"\1from matplotlib_rs\2 import", s,
               flags=re.M)
    s = re.sub(r"^(\s*)import matplotlib((?:\.[\w]+)+) as (\w+)", r"\1import matplotlib_rs\2 as \3", s,
               flags=re.M)
    # bare "import matplotlib.x": also binds the name "matplotlib"
    s = re.sub(r"^(\s*)import matplotlib((?:\.[\w]+)+)(\s*(#.*)?)$",
               r"\1import matplotlib_rs\2; import matplotlib_rs as matplotlib\3", s, flags=re.M)
    return s


def vendor(rel):
    if rel.startswith("mpl_toolkits/"):
        src = (SRC.parent / rel).read_text()
    else:
        src = (SRC / SOURCES.get(rel, rel)).read_text()
    s = HEADER + rewrite_imports(src)
    # renderer groups named after the module keep matplotlib's module name
    srcrel = rel if rel.startswith("mpl_toolkits/") else "matplotlib/" + SOURCES.get(rel, rel)
    modname = srcrel[:-3].replace("/", ".").removesuffix(".__init__")
    s = s.replace("open_group(__name__", f"open_group({modname!r}")
    for old, new, *count in PATCHES.get(rel, []):
        if old is None:
            s += new
            continue
        n = s.count(old)
        if n != (count[0] if count else 1):
            raise SystemExit(f"{rel}: patch matches {n} times:\n{old[:200]}")
        s = s.replace(old, new)
    out = DST / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(s)


def main():
    for rel in sys.argv[1:] or MODULES:
        vendor(rel)
        print("vendored", rel)


if __name__ == "__main__":
    main()
