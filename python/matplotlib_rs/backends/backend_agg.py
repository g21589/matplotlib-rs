"""
The Agg backend of matplotlib_rs.

`RendererAgg` keeps the API of matplotlib's Agg renderer; the drawing
primitives run in the Rust port of Agg.  Figures are drawn by the Rust core;
Python artists that the core does not implement are drawn by their own
``draw`` method through a `RendererAgg` that proxies the figure's renderer.

Parts of this file are from matplotlib 3.11 (see licenses/LICENSE_MATPLOTLIB).
"""

from contextlib import contextmanager, nullcontext
import io
import math
import os

import numpy as np
from PIL import features

import matplotlib_rs as mpl
from matplotlib_rs import _api, cbook, _mplrs
from matplotlib_rs.backend_bases import (
    _Backend, FigureCanvasBase, FigureManagerBase, RendererBase)
from matplotlib_rs.font_manager import fontManager as _fontManager, get_font
from matplotlib_rs.ft2font import LoadFlags, RenderMode
from matplotlib_rs.path import Path
from matplotlib_rs.transforms import Bbox, BboxBase


def get_hinting_flag():
    mapping = {
        'default': LoadFlags.DEFAULT,
        'no_autohint': LoadFlags.NO_AUTOHINT,
        'force_autohint': LoadFlags.FORCE_AUTOHINT,
        'no_hinting': LoadFlags.NO_HINTING,
        True: LoadFlags.FORCE_AUTOHINT,
        False: LoadFlags.NO_HINTING,
        'either': LoadFlags.DEFAULT,
        'native': LoadFlags.NO_AUTOHINT,
        'auto': LoadFlags.FORCE_AUTOHINT,
        'none': LoadFlags.NO_HINTING,
    }
    return mapping[mpl.rcParams['text.hinting']]


# ---------------------------------------------------------------------------
# argument conversion for the Rust renderer


def _matrix(transform):
    if transform is None:
        return np.eye(3)
    if hasattr(transform, "get_matrix"):
        return np.ascontiguousarray(transform.get_matrix(), dtype=float)
    return np.ascontiguousarray(transform, dtype=float)


def _vertices(path):
    return np.ascontiguousarray(path.vertices, dtype=float).reshape(-1, 2)


def _codes(path):
    c = path.codes
    return None if c is None else np.ascontiguousarray(c, dtype=np.uint8)


def _gc_dict(gc):
    """The attributes of a GraphicsContext read by matplotlib's Agg."""
    off, dashes = gc.get_dashes()
    d = {
        "linewidth": float(gc._linewidth),
        "alpha": float(gc._alpha),
        "forced_alpha": bool(gc._forced_alpha),
        "rgb": tuple(map(float, gc._rgb)),
        "antialiased": bool(gc._antialiased),
        "capstyle": gc._capstyle.name,
        "joinstyle": gc._joinstyle.name,
        "dashes": (0.0 if off is None else float(off),
                   None if dashes is None else [float(v) for v in dashes]),
        "snap": gc.get_snap(),
    }
    cr = gc._cliprect
    if cr is not None:
        (x0, y0), (x1, y1) = np.asarray(cr, dtype=float) if not isinstance(cr, BboxBase) \
            else cr.get_points()
        d["cliprect"] = (float(x0), float(y0), float(x1), float(y1))
    tpath, tr = gc.get_clip_path()
    if tpath is not None:
        d["clippath"] = (_vertices(tpath), _codes(tpath), _matrix(tr))
    hp = gc.get_hatch_path()
    if hp is not None:
        d["hatch"] = (_vertices(hp), _codes(hp), tuple(map(float, gc.get_hatch_color())),
                      float(gc.get_hatch_linewidth()))
    sk = gc.get_sketch_params()
    if sk is not None:
        d["sketch"] = tuple(map(float, sk))
    return d


def _face(rgbFace):
    if rgbFace is None:
        return None
    return tuple(map(float, rgbFace))


def _fontprops(prop):
    from matplotlib_rs.text import _rs_fontprops
    return _rs_fontprops(prop)


class RendererAgg(RendererBase):
    """
    The renderer handles all the drawing primitives using a graphics
    context instance that controls the colors/styles.
    """

    def __init__(self, width, height, dpi, *, _target=None):
        super().__init__()
        self.dpi = dpi
        self.width = width
        self.height = height
        if _target is None:
            _target = _mplrs.RendererAgg(int(width), int(height), dpi)
        self._renderer = _target
        self._filter_renderers = []
        self._mathtext_parser = None
        self.bbox = Bbox.from_bounds(0, 0, self.width, self.height)

    @classmethod
    def _for_target(cls, target):
        """A renderer drawing on the renderer of a figure draw in progress."""
        return cls(target.width, target.height, target.dpi, _target=target)

    @property
    def mathtext_parser(self):
        if self._mathtext_parser is None:
            from matplotlib_rs.mathtext import MathTextParser
            self._mathtext_parser = MathTextParser('path')
        return self._mathtext_parser

    @property
    def _rs_disabled(self):
        return bool(self.__dict__.get("_rs_disabled_flag")) or self._renderer.disabled

    @contextmanager
    def _draw_disabled(self):
        with super()._draw_disabled(), cbook._setattr_cm(self, _rs_disabled_flag=True):
            yield

    def __getstate__(self):
        return {'width': self.width, 'height': self.height, 'dpi': self.dpi}

    def __setstate__(self, state):
        self.__init__(state['width'], state['height'], state['dpi'])

    # -- primitives ----------------------------------------------------------
    def draw_path(self, gc, path, transform, rgbFace=None):
        # docstring inherited
        nmax = mpl.rcParams['agg.path.chunksize']  # here at least for testing
        npts = path.vertices.shape[0]

        if (npts > nmax > 100 and path.should_simplify and
                rgbFace is None and gc.get_hatch() is None):
            nch = np.ceil(npts / nmax)
            chsize = int(np.ceil(npts / nch))
            i0 = np.arange(0, npts, chsize)
            i1 = np.zeros_like(i0)
            i1[:-1] = i0[1:] - 1
            i1[-1] = npts
            for ii0, ii1 in zip(i0, i1):
                v = path.vertices[ii0:ii1, :]
                c = path.codes
                if c is not None:
                    c = c[ii0:ii1]
                    c[0] = Path.MOVETO  # move to end of last chunk
                p = Path(v, c)
                p.simplify_threshold = path.simplify_threshold
                self._draw_path(gc, p, transform, rgbFace)
        else:
            self._draw_path(gc, path, transform, rgbFace)

    def _draw_path(self, gc, path, transform, rgbFace):
        self._renderer.draw_path(
            _gc_dict(gc), _vertices(path), _codes(path), bool(path.should_simplify),
            float(path.simplify_threshold), _matrix(transform), _face(rgbFace))

    def draw_markers(self, gc, marker_path, marker_trans, path, trans, rgbFace=None):
        # docstring inherited
        self._renderer.draw_markers(
            _gc_dict(gc), _vertices(marker_path), _codes(marker_path), _matrix(marker_trans),
            _vertices(path), _codes(path), _matrix(trans), _face(rgbFace))

    def draw_path_collection(self, gc, master_transform, paths, all_transforms,
                             offsets, offset_trans, facecolors, edgecolors,
                             linewidths, linestyles, antialiaseds, urls,
                             offset_position):
        # docstring inherited
        transforms = np.asarray(
            [_matrix(t) for t in all_transforms] if len(all_transforms) else
            np.zeros((0, 3, 3)), dtype=float).reshape(-1, 3, 3)
        offsets = np.ascontiguousarray(offsets, dtype=float).reshape(-1, 2) \
            if len(offsets) else np.zeros((0, 2))
        fc = np.ascontiguousarray(facecolors, dtype=float).reshape(-1, 4) \
            if len(facecolors) else np.zeros((0, 4))
        ec = np.ascontiguousarray(edgecolors, dtype=float).reshape(-1, 4) \
            if len(edgecolors) else np.zeros((0, 4))
        lw = np.ascontiguousarray(linewidths, dtype=float).ravel()
        dashes = [(0.0 if o is None else float(o),
                   None if d is None else [float(v) for v in d])
                  for o, d in linestyles]
        aa = np.ascontiguousarray(antialiaseds, dtype=np.uint8).ravel()
        self._renderer.draw_path_collection(
            _gc_dict(gc), _matrix(master_transform),
            [(_vertices(p), _codes(p)) for p in paths],
            transforms, offsets, _matrix(offset_trans), fc, ec, lw, dashes, aa)

    def draw_image(self, gc, x, y, im):
        # docstring inherited
        im = np.ascontiguousarray(im, dtype=np.uint8)
        self._renderer.draw_image(_gc_dict(gc), float(x), float(y), im)

    def draw_quad_mesh(self, gc, master_transform, meshWidth, meshHeight,
                       coordinates, offsets, offsetTrans, facecolors,
                       antialiased, edgecolors):
        # matplotlib's generic implementation (via draw_path_collection)
        return super().draw_quad_mesh(
            gc, master_transform, meshWidth, meshHeight, coordinates, offsets,
            offsetTrans, facecolors, antialiased, edgecolors)

    def _draw_text_glyphs_and_boxes(self, gc, x, y, angle, glyphs, boxes):
        self._renderer.draw_glyphs_and_boxes(
            _gc_dict(gc), float(x), float(y), float(angle),
            [(font.fname, int(getattr(font, "face_index", 0)), float(size),
              int(glyph_index), float(slant), float(extend), float(dx), float(dy))
             for font, size, glyph_index, slant, extend, dx, dy in glyphs],
            [tuple(map(float, b)) for b in boxes])

    def draw_mathtext(self, gc, x, y, s, prop, angle):
        """Draw mathtext using :mod:`matplotlib.mathtext`."""
        parse = self.mathtext_parser.parse(
            s, self.dpi, prop, antialiased=gc.get_antialiased())
        self._draw_text_glyphs_and_boxes(
            gc, x, y, angle,
            ((font, size, glyph_index, 0, 1, dx, dy)
             for font, size, _char, glyph_index, dx, dy in parse.glyphs),
            parse.rects)

    def draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
        # docstring inherited
        if ismath == "TeX":
            return self.draw_tex(gc, x, y, s, prop, angle, mtext=mtext)
        if ismath:
            return self.draw_mathtext(gc, x, y, s, prop, angle)
        self._renderer.draw_text(_gc_dict(gc), float(x), float(y), s,
                                 _fontprops(prop), float(angle), False)

    def get_text_width_height_descent(self, s, prop, ismath):
        # docstring inherited
        _api.check_in_list(["TeX", True, False], ismath=ismath)
        if ismath == "TeX":
            return super().get_text_width_height_descent(s, prop, ismath)

        if ismath:
            parse = self.mathtext_parser.parse(s, self.dpi, prop)
            return parse.width, parse.height, parse.depth

        return self._renderer.get_text_width_height_descent(s, _fontprops(prop), False)

    def draw_tex(self, gc, x, y, s, prop, angle, *, mtext=None):
        raise NotImplementedError("matplotlib_rs does not support usetex yet")

    def get_canvas_width_height(self):
        # docstring inherited
        return self.width, self.height

    def _prepare_font(self, font_prop):
        """
        Get the `.FT2Font` for *font_prop*, clear its buffer, and set its size.
        """
        font = get_font(_fontManager._find_fonts_by_props(font_prop))
        font.clear()
        size = font_prop.get_size_in_points()
        font.set_size(size, self.dpi)
        return font

    def points_to_pixels(self, points):
        # docstring inherited
        return points * self.dpi / 72

    def buffer_rgba(self):
        return memoryview(self._renderer.buffer_rgba())

    def tostring_argb(self):
        return np.asarray(self._renderer.buffer_rgba()).take([3, 0, 1, 2], axis=2).tobytes()

    def clear(self):
        self._renderer.clear()

    def option_image_nocomposite(self):
        # docstring inherited

        # It is generally faster to composite each image directly to
        # the Figure, and there's no file size benefit to compositing
        # with the Agg backend
        return True

    def option_scale_image(self):
        # docstring inherited
        return False

    def copy_from_bbox(self, bbox):
        raise NotImplementedError("matplotlib_rs does not support blitting yet")

    def restore_region(self, region, bbox=None, xy=None):
        raise NotImplementedError("matplotlib_rs does not support blitting yet")

    def start_filter(self):
        """
        Start filtering. It simply creates a new canvas (the old one is saved).
        """
        self._filter_renderers.append(self._renderer)
        self._renderer = _mplrs.RendererAgg(int(self.width), int(self.height), self.dpi)

    def stop_filter(self, post_processing):
        """
        Save the current canvas as an image and apply post processing.
        """
        orig_img = np.asarray(self.buffer_rgba())
        slice_y, slice_x = cbook._get_nonzero_slices(orig_img[..., 3])
        cropped_img = orig_img[slice_y, slice_x]

        self._renderer = self._filter_renderers.pop()

        if cropped_img.size:
            img, ox, oy = post_processing(cropped_img / 255, self.dpi)
            gc = self.new_gc()
            if img.dtype.kind == 'f':
                img = np.asarray(img * 255., np.uint8)
            self.draw_image(
                gc, slice_x.start + ox, int(self.height) - slice_y.stop + oy,
                img[::-1])


class FigureCanvasAgg(FigureCanvasBase):
    # docstring inherited

    _lastKey = None  # Overwritten per-instance on the first draw.

    def copy_from_bbox(self, bbox):
        renderer = self.get_renderer()
        return renderer.copy_from_bbox(bbox)

    def restore_region(self, region, bbox=None, xy=None):
        renderer = self.get_renderer()
        return renderer.restore_region(region, bbox, xy)

    def draw(self):
        # docstring inherited
        self.renderer = self.get_renderer()
        self.renderer.clear()
        with (self.toolbar._wait_cursor_for_draw_cm() if self.toolbar
              else nullcontext()):
            self.figure.draw(self.renderer)
            # A GUI class may be need to update a window using this draw, so
            # don't forget to call the superclass.
            super().draw()

    def get_renderer(self):
        w, h = self.get_width_height(physical=True)
        key = w, h, self.figure.dpi
        reuse_renderer = (self._lastKey == key)
        if not reuse_renderer:
            self.renderer = RendererAgg(w, h, self.figure.dpi)
            self._lastKey = key
        return self.renderer

    def tostring_argb(self):
        return self.renderer.tostring_argb()

    def buffer_rgba(self):
        return self.renderer.buffer_rgba()

    def print_raw(self, filename_or_obj, *, metadata=None):
        if metadata is not None:
            raise ValueError("metadata not supported for raw/rgba")
        FigureCanvasAgg.draw(self)
        renderer = self.get_renderer()
        with cbook.open_file_cm(filename_or_obj, "wb") as fh:
            fh.write(renderer.buffer_rgba())

    print_rgba = print_raw

    def _print_pil(self, filename_or_obj, fmt, pil_kwargs, metadata=None):
        FigureCanvasAgg.draw(self)
        mpl.image.imsave(
            filename_or_obj, self.buffer_rgba(), format=fmt, origin="upper",
            dpi=self.figure.dpi, metadata=metadata, pil_kwargs=pil_kwargs)

    def print_png(self, filename_or_obj, *, metadata=None, pil_kwargs=None):
        """
        Write the figure to a PNG file.

        The pixels are encoded by the Rust core unless *pil_kwargs* are given.
        """
        if pil_kwargs is not None:
            return self._print_pil(filename_or_obj, "png", pil_kwargs, metadata)
        FigureCanvasAgg.draw(self)
        meta = {"Software": f"Matplotlib version{mpl.__version__}, https://matplotlib.org/",
                **(metadata or {})}
        items = []
        for k, v in meta.items():
            if v is None:
                continue
            if isinstance(v, bytes):
                v = v.decode("latin-1")
            items.append((str(k), str(v)))
        rgba = np.asarray(self.renderer.buffer_rgba())
        data = _mplrs.encode_png(rgba, float(self.figure.dpi), items, 1)
        if isinstance(filename_or_obj, (str, os.PathLike)):
            with open(filename_or_obj, "wb") as fh:
                fh.write(data)
        else:
            filename_or_obj.write(data)

    def print_to_buffer(self):
        FigureCanvasAgg.draw(self)
        renderer = self.get_renderer()
        return (bytes(renderer.buffer_rgba()),
                (int(renderer.width), int(renderer.height)))

    def print_gif(self, filename_or_obj, *, metadata=None, pil_kwargs=None):
        self._print_pil(filename_or_obj, "gif", pil_kwargs, metadata)

    def print_jpg(self, filename_or_obj, *, metadata=None, pil_kwargs=None):
        # savefig() has already applied savefig.facecolor; we now set it to
        # white to make imsave() blend semi-transparent figures against an
        # assumed white background.
        with mpl.rc_context({"savefig.facecolor": "white"}):
            self._print_pil(filename_or_obj, "jpeg", pil_kwargs, metadata)

    print_jpeg = print_jpg

    def print_tif(self, filename_or_obj, *, metadata=None, pil_kwargs=None):
        self._print_pil(filename_or_obj, "tiff", pil_kwargs, metadata)

    print_tiff = print_tif

    def print_webp(self, filename_or_obj, *, metadata=None, pil_kwargs=None):
        self._print_pil(filename_or_obj, "webp", pil_kwargs, metadata)

    def print_avif(self, filename_or_obj, *, metadata=None, pil_kwargs=None):
        if not features.check("avif"):
            raise RuntimeError(
                "The installed pillow version does not support avif. Full "
                "avif support has been added in pillow 11.3."
            )
        self._print_pil(filename_or_obj, "avif", pil_kwargs, metadata)


@_Backend.export
class _BackendAgg(_Backend):
    backend_version = 'v2.2'
    FigureCanvas = FigureCanvasAgg
    FigureManager = FigureManagerBase
