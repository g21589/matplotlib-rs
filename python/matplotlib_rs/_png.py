"""
PNG reading and writing without Pillow.

Pillow is optional for matplotlib_rs: when it is not installed, PNG files are
decoded and encoded by the Rust core.  `read` returns the same arrays as
matplotlib's Pillow-based ``imread`` (``_pil_png_to_float_array``): float32
values in [0, 1]; grayscale images are 2D, palette and grayscale+alpha
images are expanded to RGBA, and 16-bit color samples are truncated to their
most significant byte as Pillow does.
"""

import os

import numpy as np

from . import _mplrs


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


def encode(rgba, dpi=0, text=()):
    """PNG file contents of the (h, w, 4) uint8 array *rgba*, with a pHYs
    chunk if *dpi* > 0 and the tEXt chunks *text* (pairs of str)."""
    rgba = np.require(np.asarray(rgba, dtype=np.uint8), requirements="C")
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError(f"expected an RGBA array, not one of shape {rgba.shape}")
    return _mplrs.encode_png(rgba, float(dpi), [(str(k), str(v)) for k, v in text], 6)


def write(fname, rgba, dpi=0, text=()):
    """Write `encode` (*rgba*, *dpi*, *text*) to a path or binary file."""
    data = encode(rgba, dpi, text)
    if isinstance(fname, (str, os.PathLike)):
        with open(fname, "wb") as fh:
            fh.write(data)
    else:
        fname.write(data)


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
