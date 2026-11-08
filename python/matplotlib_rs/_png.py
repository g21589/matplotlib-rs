"""
PNG reading and writing in the Rust core.

PNG files are always written by the Rust core, whether or not Pillow is
installed: rows are filtered with libpng's (and Pillow's) heuristic and
deflated at zlib level 6 by default, as Pillow does, in strips compressed on
several threads (``MPLRS_THREADS``).  The PNG options of ``pil_kwargs``
(`NATIVE_PIL_KWARGS`) are understood; the others need Pillow.

When Pillow is not installed, PNG files are also decoded by the Rust core.
`read` returns the same arrays as matplotlib's Pillow-based ``imread``
(``_pil_png_to_float_array``): float32 values in [0, 1]; grayscale images are
2D, palette and grayscale+alpha images are expanded to RGBA, and 16-bit color
samples are truncated to their most significant byte as Pillow does.
"""

import os

import numpy as np

from . import _mplrs

#: The zlib level of Pillow's PNG writer (``compress_level=-1``).
DEFAULT_LEVEL = 6

#: The ``pil_kwargs`` of a PNG save that are handled without Pillow, with
#: Pillow's meaning: ``optimize`` (zlib level 9), ``compress_level`` (0-9,
#: -1 for the default), ``compress_type`` (the zlib strategy, 0-4, -1 for the
#: default), ``dpi`` (an (x, y) pair for the pHYs chunk) and ``format``.
NATIVE_PIL_KWARGS = frozenset({"optimize", "compress_level", "compress_type", "dpi", "format"})


def pil_image():
    """``PIL.Image`` if Pillow is installed, else None."""
    try:
        import PIL.Image
    except ImportError:
        return None
    return PIL.Image


def require_pil(what):
    """``PIL.Image``, or an ImportError naming *what* needs Pillow."""
    image = pil_image()
    if image is None:
        raise ImportError(
            f"{what} requires Pillow (pip install pillow); matplotlib_rs reads and "
            f"writes PNG files without it")
    return image


def native_options(pil_kwargs):
    """
    The zlib ``(level, strategy)`` and pHYs ``dpi`` (an (x, y) pair, or None
    if not given) of a PNG save with *pil_kwargs* (a dict or None), or None if
    *pil_kwargs* has options that only Pillow handles.
    """
    pil_kwargs = pil_kwargs or {}
    if not NATIVE_PIL_KWARGS.issuperset(pil_kwargs):
        return None
    if str(pil_kwargs.get("format", "png")).lower() != "png":
        return None
    level = pil_kwargs.get("compress_level", -1)
    strategy = pil_kwargs.get("compress_type", -1)
    if pil_kwargs.get("optimize", False):
        level = 9  # Pillow's ZipEncoder: optimize means Z_BEST_COMPRESSION
    if not isinstance(level, (int, np.integer)) or not -1 <= level <= 9:
        raise ValueError(f"compress_level must be an integer from -1 to 9, not {level!r}")
    if not isinstance(strategy, (int, np.integer)) or not -1 <= strategy <= 4:
        raise ValueError(f"compress_type must be an integer from -1 to 4, not {strategy!r}")
    dpi = pil_kwargs.get("dpi")
    if dpi:
        dpi = (float(dpi[0]), float(dpi[1]))
    else:
        dpi = None
    return (DEFAULT_LEVEL if level == -1 else int(level),
            0 if strategy == -1 else int(strategy), dpi)


def text_chunks(metadata):
    """
    The (keyword, text) pairs of the text chunks Pillow's ``PngInfo.add_text``
    writes for the items of the dict *metadata*, skipping None values; bytes
    are Latin-1.  Keywords must be Latin-1; texts that are not are written as
    UTF-8 iTXt chunks.
    """
    items = []
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(key, bytes):
            key = key.decode("latin-1")
        else:
            key = str(key)
            key.encode("latin-1", "strict")  # raises as PngInfo.add_text does
        if isinstance(value, bytes):
            value = value.decode("latin-1")
        items.append((key, str(value)))
    return items


def _dpi_pair(dpi):
    if dpi is None:
        return None
    if np.ndim(dpi) == 0:
        return None if dpi <= 0 else (float(dpi), float(dpi))
    return (float(dpi[0]), float(dpi[1]))


def encode(rgba, dpi=None, text=(), level=DEFAULT_LEVEL, strategy=0):
    """
    PNG file contents of the (h, w, 4) uint8 array *rgba* (or an (h, w) gray,
    (h, w, 2) gray + alpha or (h, w, 3) RGB array), with a pHYs chunk if *dpi*
    (a number > 0 or an (x, y) pair) is given, the text chunks *text* (pairs
    of str, as from `text_chunks`) and the zlib *level* and *strategy*.
    """
    rgba = np.asarray(rgba, dtype=np.uint8)
    if not (rgba.ndim == 2 or rgba.ndim == 3 and 1 <= rgba.shape[2] <= 4):
        raise ValueError(f"expected an RGBA array, not one of shape {rgba.shape}")
    return _mplrs.encode_png(rgba, _dpi_pair(dpi), [(str(k), str(v)) for k, v in text],
                             level, strategy)


def write_bytes(fname, data):
    """Write *data* to the path or binary file *fname* (as Pillow's save)."""
    if isinstance(fname, (str, bytes, os.PathLike)):
        with open(fname, "wb") as fh:
            fh.write(data)
    else:
        fname.write(data)
        if hasattr(fname, "flush"):
            fname.flush()


def write(fname, rgba, dpi=None, text=(), level=DEFAULT_LEVEL, strategy=0):
    """Write `encode` (*rgba*, *dpi*, *text*, *level*, *strategy*) to a path or
    binary file."""
    write_bytes(fname, encode(rgba, dpi, text, level, strategy))


def _unpack(samples, height, width, channels, depth):
    """The samples of each row as one uint8/uint16 array per channel."""
    if depth == 16:
        a = np.frombuffer(samples, ">u2").reshape(height, width * channels)
        return a.astype(np.uint16)
    stride = (width * channels * depth + 7) // 8
    raw = np.frombuffer(samples, np.uint8).reshape(height, stride)
    if depth == 8:
        return raw[:, :width * channels]
    bits = np.unpackbits(raw, axis=1)[:, :width * channels * depth]
    bits = bits.reshape(height, width * channels, depth)
    weights = (1 << np.arange(depth - 1, -1, -1)).astype(np.uint8)
    return (bits * weights).sum(axis=2, dtype=np.uint8)


def read(fname):
    """Read a PNG file (path or binary file) as a float32 array."""
    if isinstance(fname, (str, os.PathLike)):
        with open(fname, "rb") as fh:
            data = fh.read()
    else:
        data = fname.read()
    width, height, color, depth, samples, palette, trns = _mplrs.decode_png(data)
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color]
    a = _unpack(samples, height, width, channels, depth)
    if color == 0:  # grayscale (any transparency chunk is ignored, as by Pillow)
        if depth == 1:
            return a.astype(np.float32)
        return np.divide(a, 2 ** depth - 1, dtype=np.float32)
    if color == 3:  # palette, expanded to RGBA
        pal = np.frombuffer(palette, np.uint8).reshape(-1, 3)
        lut = np.full((256, 4), 255, np.uint8)
        lut[:len(pal), :3] = pal
        if trns is not None:
            alpha = np.frombuffer(trns, np.uint8)
            lut[:len(alpha), 3] = alpha
        return np.divide(lut[a], 2 ** 8 - 1, dtype=np.float32)
    if depth == 16:  # Pillow keeps the most significant byte of each sample
        a = (a >> 8).astype(np.uint8)
    a = a.reshape(height, width, channels)
    if color == 4:  # grayscale + alpha, expanded to RGBA
        a = np.concatenate([a[..., :1], a[..., :1], a[..., :1], a[..., 1:]], axis=2)
    return np.divide(a, 2 ** 8 - 1, dtype=np.float32)
