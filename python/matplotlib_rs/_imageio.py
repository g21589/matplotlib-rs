"""
Image files without Pillow.

matplotlib_rs never imports Pillow: every raster file is read and written by
the Rust core.  PNG (`._png`), JPEG, TIFF, GIF, WebP and AVIF (writing only)
are supported, with the defaults and ``pil_kwargs`` options of Pillow that
matplotlib relies on:

- PNG: zlib level 6; ``compress_level``, ``optimize``, ``compress_type``,
  ``pnginfo`` (any object with Pillow's ``PngInfo.chunks``), ``icc_profile``,
  ``exif``.
- JPEG: quality 75, 4:2:0 subsampling; ``quality``, ``subsampling``,
  ``optimize``, ``progressive``/``progression``, ``icc_profile``, ``exif``.
- TIFF: uncompressed; ``compression`` (``"raw"``, ``"tiff_lzw"``,
  ``"tiff_deflate"``/``"tiff_adobe_deflate"``, ``"packbits"``).
- WebP: libwebp, as Pillow (lossy, quality 80, method 4); ``lossless``,
  ``quality``, ``alpha_quality``, ``method``, ``exact``.
- GIF: a palette per frame (exact up to 256 colors, else median cut); ``loop``.
- AVIF: rav1e, quality 75, speed 6; ``quality``, ``speed``.

``dpi`` and ``format`` are accepted for every format.  Other ``pil_kwargs``
raise a ValueError naming the supported ones.
"""

import io
import os

import numpy as np

from . import _api, _mplrs, _png

#: Canonical names of the writable formats.
_ALIASES = {"jpg": "jpeg", "jpeg": "jpeg", "tif": "tiff", "tiff": "tiff", "png": "png",
            "gif": "gif", "webp": "webp", "avif": "avif"}

_COMMON = {"dpi", "format"}
_OPTIONS = {
    "png": {"optimize", "compress_level", "compress_type", "pnginfo", "icc_profile", "exif"},
    "jpeg": {"quality", "subsampling", "optimize", "progressive", "progression",
             "icc_profile", "exif"},
    "tiff": {"compression"},
    "webp": {"lossless", "quality", "alpha_quality", "method", "exact"},
    "gif": {"loop"},
    "avif": {"quality", "speed"},
}

_TIFF_COMPRESSION = {None: 0, "raw": 0, "none": 0, "tiff_lzw": 1, "lzw": 1,
                     "tiff_deflate": 2, "tiff_adobe_deflate": 2, "deflate": 2,
                     "packbits": 3}
_SUBSAMPLING = {-1: 2, 0: 0, 1: 1, 2: 2, "4:4:4": 0, "4:2:2": 1, "4:2:0": 2}


def canonical_format(fmt):
    """The writer name of *fmt* (e.g. ``"jpg"`` -> ``"jpeg"``), or None."""
    return _ALIASES.get(str(fmt).lower())


def _check_options(fmt, pil_kwargs):
    unknown = set(pil_kwargs) - _OPTIONS[fmt] - _COMMON
    if unknown:
        raise ValueError(
            f"pil_kwargs {sorted(unknown)} are not supported when saving {fmt.upper()} "
            f"images (matplotlib_rs writes them without Pillow); supported: "
            f"{sorted(_OPTIONS[fmt] | _COMMON)}")
    given = pil_kwargs.get("format")
    if given is not None and canonical_format(given) != fmt:
        raise ValueError(f"pil_kwargs format {given!r} does not match {fmt!r}")


def _pnginfo_chunks(info):
    """The chunks Pillow writes for a ``PngInfo``-like *info* (an object with
    a ``chunks`` list of ``(type, data[, after_idat])``), as (type, data,
    position) for ``_mplrs.encode_png``."""
    before_plte = {b"cHRM", b"cLLi", b"mDCv", b"gAMA", b"sBIT", b"sRGB", b"tIME"}
    multiple = {b"sPLT", b"iTXt", b"tEXt", b"zTXt"}
    after_phys = {b"bKGD", b"hIST"}
    out = []
    for chunk in info.chunks:
        cid, data = bytes(chunk[0]), bytes(chunk[1])
        after_idat = len(chunk) == 3 and bool(chunk[2])
        if cid in before_plte:
            before_plte.discard(cid)
            out.append((cid, data, 0))
        elif cid in multiple:
            out.append((cid, data, 0))
        elif cid[1:2].islower():  # private chunk
            out.append((cid, data, 2 if after_idat else 0))
        elif cid in after_phys:
            after_phys.discard(cid)
            out.append((cid, data, 1))
    return out


def _exif_bytes(exif):
    exif = bytes(exif.tobytes() if hasattr(exif, "tobytes") else exif)
    return exif[6:] if exif.startswith(b"Exif\x00\x00") else exif


def _dpi(pil_kwargs):
    dpi = pil_kwargs.get("dpi")
    return (float(dpi[0]), float(dpi[1])) if dpi else None


def encode(rgba, fmt, *, dpi=None, metadata=None, pil_kwargs=None, background=None):
    """
    The file contents of the (h, w, 4) uint8 array *rgba* in format *fmt*
    (as `matplotlib.image.imsave` passes them to Pillow): *dpi* is the default
    of ``pil_kwargs["dpi"]``, *metadata* the PNG text, and *background* the
    RGB color (0-255) JPEG images are composited on.
    """
    fmt = canonical_format(fmt)
    pil_kwargs = dict(pil_kwargs or {})
    _check_options(fmt, pil_kwargs)
    if fmt != "png" and metadata is not None:
        raise ValueError(f"metadata not supported for format {fmt!r}")
    if dpi is not None:
        pil_kwargs.setdefault("dpi", (dpi, dpi))
    rgba = np.ascontiguousarray(rgba, np.uint8)
    if fmt == "png":
        level, strategy, png_dpi = _png.native_options(
            {k: v for k, v in pil_kwargs.items() if k in _png.NATIVE_PIL_KWARGS})
        chunks = []
        icc = pil_kwargs.get("icc_profile")
        if icc:
            import zlib
            chunks.append((b"iCCP", b"ICC Profile\0\0" + zlib.compress(icc), 0))
        if "pnginfo" in pil_kwargs:
            if metadata:
                _api.warn_external("'metadata' is overridden by the 'pnginfo' entry in "
                                   "'pil_kwargs'.")
            text = []
            chunks += [c for c in _pnginfo_chunks(pil_kwargs["pnginfo"])
                       if not (icc and c[0] == b"sRGB")]
        else:
            text = _png.text_chunks({
                "Software": f"Matplotlib version{_mpl_version()}, https://matplotlib.org/",
                **(metadata if metadata is not None else {})})
        if pil_kwargs.get("exif"):
            chunks.append((b"eXIf", _exif_bytes(pil_kwargs["exif"]), 1))
        return _mplrs.encode_png(rgba, png_dpi, text, level, strategy, chunks)
    if fmt == "jpeg":
        quality = pil_kwargs.get("quality", 75)
        if isinstance(quality, str) or not 0 <= quality <= 100:
            raise ValueError(f"JPEG quality must be an integer from 0 to 100, not {quality!r}")
        subsampling = pil_kwargs.get("subsampling", -1)
        if subsampling not in _SUBSAMPLING:
            raise ValueError(f"invalid JPEG subsampling {subsampling!r}")
        dpi = _dpi(pil_kwargs)
        # Composited over *background* (as Pillow's paste) by the Rust core.
        return _mplrs.encode_jpeg(
            rgba, int(max(quality, 1)), _SUBSAMPLING[subsampling],
            bool(pil_kwargs.get("optimize", False)),
            bool(pil_kwargs.get("progressive", pil_kwargs.get("progression", False))),
            tuple(min(int(round(v)), 65535) for v in dpi) if dpi and any(dpi) else None,
            pil_kwargs.get("icc_profile") or None,
            _exif_bytes(pil_kwargs["exif"]) if pil_kwargs.get("exif") else None,
            tuple(background) if background is not None else (255, 255, 255))
    if fmt == "tiff":
        compression = pil_kwargs.get("compression")
        key = compression.lower() if isinstance(compression, str) else compression
        if key not in _TIFF_COMPRESSION:
            raise ValueError(f"unsupported TIFF compression {compression!r}; supported: "
                             f"{sorted(k for k in _TIFF_COMPRESSION if k)}")
        return _mplrs.encode_tiff(rgba, _TIFF_COMPRESSION[key], _dpi(pil_kwargs))
    if fmt == "webp":
        return _mplrs.encode_webp(
            [rgba], bool(pil_kwargs.get("lossless", False)),
            float(pil_kwargs.get("quality", 80)), float(pil_kwargs.get("alpha_quality", 100)),
            int(pil_kwargs.get("method", 4)), bool(pil_kwargs.get("exact", False)))
    if fmt == "gif":
        loop = pil_kwargs.get("loop")
        return _mplrs.encode_gif([rgba], 0, None if loop is None else int(loop))
    if fmt == "avif":
        return _mplrs.encode_avif(rgba, float(pil_kwargs.get("quality", 75)),
                                  int(pil_kwargs.get("speed", 6)))
    raise ValueError(f"cannot write {fmt!r} images")


def _mpl_version():
    import matplotlib_rs
    return matplotlib_rs.__version__


def save(fname, rgba, fmt, **kwargs):
    """Write `encode` (*rgba*, *fmt*, ...) to the path or binary file *fname*."""
    _png.write_bytes(fname, encode(rgba, fmt, **kwargs))


def save_animation(fname, frames, duration, loop=0):
    """
    Write the (h, w, 4) uint8 *frames* as an animated GIF, PNG (APNG) or
    WebP file (by the extension of *fname*), each shown *duration* ms, played
    *loop* times (0: forever), as Pillow's ``save(save_all=True)``.
    """
    ext = os.path.splitext(os.fspath(fname) if isinstance(fname, (str, os.PathLike))
                           else getattr(fname, "name", ""))[1].lower()[1:]
    frames = [np.ascontiguousarray(f, np.uint8) for f in frames]
    opaque = all(f[..., 3].min() == 255 for f in frames)
    if ext == "gif":
        data = _mplrs.encode_gif(frames, int(duration), int(loop))
    elif ext in ("png", "apng"):
        data = _mplrs.encode_apng([f[..., :3].copy() if opaque else f for f in frames],
                                  min(int(duration), 65535), int(loop))
    elif ext == "webp":
        data = _mplrs.encode_webp([f[..., :3].copy() if opaque else f for f in frames],
                                  duration=int(duration), loop_count=int(loop))
    else:
        raise ValueError(f"cannot write {ext!r} animations; matplotlib_rs writes GIF, "
                         f"PNG (APNG) and WebP animations")
    _png.write_bytes(fname, data)


def read(fname, ext):
    """
    Read the image file *fname* (path or binary file) as `matplotlib.image.imread`:
    PNG files (an *ext* of ``"png"`` requires one) as float32 arrays in
    [0, 1], others as uint8 (uint16 for 16-bit gray TIFF) arrays, (h, w) for
    gray, (h, w, 3) for RGB and (h, w, 4) for RGBA images, palette, gray +
    alpha and CMYK images being converted to RGBA as by Pillow.
    """
    if isinstance(fname, (str, bytes, os.PathLike)):
        with open(fname, "rb") as fh:
            data = fh.read()
    else:
        data = fname.read()
    if ext == "png" or _mplrs.sniff_image(data[:16]) == "png":
        return _png.read(io.BytesIO(data))
    _fmt, array = _mplrs.decode_image(data)
    return array
