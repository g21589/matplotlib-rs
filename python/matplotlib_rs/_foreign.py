"""Drawing into renderers other than the Rust Agg one (matplotlib's vector
backends, or any third-party renderer).

The Rust core draws its artists by calling these functions with plain data;
they rebuild the objects matplotlib's artists would pass (graphics context,
`Path`, `Affine2D`, `FontProperties`) and call the renderer.
"""
import contextlib

import numpy as np

from . import _mplrs
from ._enums import CapStyle, JoinStyle
from .font_manager import FontProperties
from .path import Path
from .transforms import Affine2D, Bbox, TransformedPath


def warn_missing_glyphs():
    """Issue the "Glyph ... missing" warnings of text laid out by the Rust core."""
    missing = _mplrs.take_missing_glyphs()
    if missing:
        from ._text_helpers import warn_on_missing_glyph
        for codepoint, fonts in missing:
            warn_on_missing_glyph(codepoint, fonts)


def is_foreign(renderer):
    """Whether *renderer* is not backed by the Rust Agg renderer."""
    return renderer is not None and not isinstance(
        getattr(renderer, "_renderer", None), _mplrs.RendererAgg)


@contextlib.contextmanager
def metrics(renderer):
    """Use *renderer*'s text metrics for layout done by the Rust core."""
    if not is_foreign(renderer):
        yield
        return
    _mplrs.push_foreign(renderer)
    try:
        yield
    finally:
        _mplrs.pop_foreign()


def _path(verts, codes, should_simplify, simplify_threshold):
    p = Path(verts, codes)
    p._should_simplify = should_simplify
    p._simplify_threshold = simplify_threshold
    return p


def _gc(renderer, state, clip_cache):
    (alpha, forced_alpha, antialiased, capstyle, joinstyle, cliprect, clippath,
     dashes, linewidth, rgb, snap, sketch, hatch, url) = state
    gc = renderer.new_gc()
    gc._alpha = alpha
    gc._forced_alpha = forced_alpha
    gc._antialiased = int(antialiased)
    gc._capstyle = CapStyle(capstyle)
    gc._joinstyle = JoinStyle(joinstyle)
    if cliprect is not None:
        bb = clip_cache.get(cliprect)
        if bb is None:
            bb = clip_cache[cliprect] = Bbox.from_extents(*cliprect[1])
        gc._cliprect = bb
    if clippath is not None:
        key, verts, codes, ss, st, matrix = clippath
        tp = clip_cache.get(key)
        if tp is None:
            tp = clip_cache[key] = TransformedPath(_path(verts, codes, ss, st), Affine2D(matrix))
        gc._clippath = tp
    gc._dashes = dashes
    gc._linewidth = linewidth
    gc._rgb = rgb
    gc._snap = snap
    gc._sketch = sketch
    gc._url = url
    if hatch is not None:
        pattern, gc._hatch_color, gc._hatch_linewidth, is_np = hatch
        gc._hatch = np.str_(pattern) if is_np else pattern
    return gc


_FONTPROPS = {}


def _fontprops(t):
    key = (tuple(t[0]), *t[1:])
    fp = _FONTPROPS.get(key)
    if fp is None:
        fam, style, variant, weight, stretch, size, fname, mff = t
        fp = _FONTPROPS[key] = FontProperties(
            family=fam, style=style, variant=variant, weight=weight,
            stretch=stretch, size=size, fname=fname, math_fontfamily=mff)
    return fp


class _MText:
    """What renderers use of the `Text` passed as *mtext*, for texts owned by
    the Rust core (tick labels, ...)."""

    def __init__(self, x, y, ha, va, rotation_mode):
        self._xy = (x, y)
        self._ha, self._va, self._rotation_mode = ha, va, rotation_mode

    def get_transform(self):
        from .transforms import IdentityTransform
        return IdentityTransform()

    def get_unitless_position(self):
        return self._xy

    def get_ha(self):
        return self._ha

    get_horizontalalignment = get_ha

    def get_va(self):
        return self._va

    get_verticalalignment = get_va

    def get_rotation_mode(self):
        return self._rotation_mode

    def get_fontfeatures(self):
        return None

    def get_language(self):
        return None


class Target:
    """The renderer the Rust core forwards to, with per-draw caches."""

    def __init__(self, renderer):
        self.renderer = renderer
        self.clip_cache = {}

    def draw_path(self, gc, path, matrix, face):
        r = self.renderer
        gc = _gc(r, gc, self.clip_cache)
        r.draw_path(gc, _path(*path), Affine2D(matrix), face)
        gc.restore()

    def draw_markers(self, gc, marker, marker_matrix, path, matrix, face):
        r = self.renderer
        gc = _gc(r, gc, self.clip_cache)
        r.draw_markers(gc, _path(*marker), Affine2D(marker_matrix), _path(*path),
                       Affine2D(matrix), face)
        gc.restore()

    def draw_path_collection(self, gc, master, paths, transforms, offsets, offset_matrix,
                             facecolors, edgecolors, linewidths, linestyles, antialiaseds):
        r = self.renderer
        gc = _gc(r, gc, self.clip_cache)
        r.draw_path_collection(
            gc, Affine2D(master), [_path(*p) for p in paths],
            [np.asarray(t) for t in transforms], np.asarray(offsets).reshape(-1, 2),
            Affine2D(offset_matrix), facecolors, edgecolors, linewidths, linestyles,
            antialiaseds, [None], "screen")
        gc.restore()

    def draw_image(self, gc, x, y, im):
        r = self.renderer
        gc = _gc(r, gc, self.clip_cache)
        r.draw_image(gc, x, y, im)
        gc.restore()

    def draw_text(self, gc, x, y, s, prop, angle, ismath, owner, anchor):
        r = self.renderer
        gc = _gc(r, gc, self.clip_cache)
        mtext = owner() if owner is not None else None
        if mtext is None and anchor is not None:
            mtext = _MText(*anchor)
        r.draw_text(gc, x, y, s, _fontprops(prop), angle, ismath=ismath, mtext=mtext)
        gc.restore()

    def text_metrics(self, s, prop, ismath):
        w, h, d = self.renderer.get_text_width_height_descent(s, _fontprops(prop), ismath)
        return float(w), float(h), float(d)

    def open_group(self, name, gid):
        self.renderer.open_group(name, gid=gid)

    def close_group(self, name):
        self.renderer.close_group(name)

    def artist_start(self):
        r = self.renderer
        if r._raster_depth == 0 and r._rasterizing:
            r.stop_rasterizing()
            r._rasterizing = False

    def raster_begin(self):
        r = self.renderer
        if r._raster_depth == 0 and not r._rasterizing:
            r.start_rasterizing()
            r._rasterizing = True
        r._raster_depth += 1
        # (a MixedModeRenderer sets the figure dpi to its raster dpi)
        fig = getattr(r, "figure", None)
        return None if fig is None else float(fig.dpi)

    def raster_end(self):
        r = self.renderer
        r._raster_depth -= 1
        fig = getattr(r, "figure", None)  # (a MixedModeRenderer's)
        if r._rasterizing and fig is not None and fig.suppressComposite:
            # restart rasterizing to prevent merging
            r.stop_rasterizing()
            r.start_rasterizing()

    def flip(self):
        r = self.renderer
        return bool(r.flipy()), float(r.get_canvas_width_height()[1])
