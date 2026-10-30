"""Compare SVG/PDF files independently of how their raster images are encoded.

Without Pillow, matplotlib_rs encodes embedded images itself: the pixels are
the same, but the PNG/deflate bytes (and PDF palette conversion) differ from
Pillow's.  These functions decode every embedded image to its RGB(A) pixels
and decompress every PDF stream, leaving all other content byte for byte, so
that such files can still be compared strictly.
"""
import base64
import hashlib
import io
import re
import zlib

import numpy as np
from PIL import Image


def _digest(a):
    a = np.ascontiguousarray(a)
    return f"{a.shape}:{a.dtype}:{hashlib.sha256(a.tobytes()).hexdigest()}"


def svg(data, hashed_ids=False):
    """SVG text with each inline PNG replaced by a digest of its pixels and
    image ids (derived from the PNG bytes) numbered in order.

    With *hashed_ids*, clip path ids are numbered in order too: matplotlib
    hashes ``str()`` of their NumPy values, which NumPy 2 prints differently.
    """
    text = data.decode("utf-8")
    if hashed_ids:
        clips = {}
        text = re.sub(r"\bp[0-9a-f]{10}\b",
                      lambda m: f"p#{clips.setdefault(m.group(0), len(clips))}", text)

    def png(m):
        raw = base64.b64decode(re.sub(r"\s", "", m.group(1)))
        return "PNG[" + _digest(np.asarray(Image.open(io.BytesIO(raw)).convert("RGBA"))) + "]"

    text = re.sub(r"data:image/png;base64,\s*([A-Za-z0-9+/=\s]+)", png, text)
    ids = {}
    text = re.sub(r'id="(image[0-9a-f]{10})"',
                  lambda m: f'id="image#{ids.setdefault(m.group(1), len(ids))}"', text)
    return text


def _unfilter(raw, width, height, bpp, row_bytes):
    """Undo the PNG row filters of predictor-encoded data."""
    out = np.zeros((height, row_bytes), np.uint8)
    prior = np.zeros(row_bytes, np.int64)
    pos = 0
    for y in range(height):
        f = raw[pos]
        row = np.frombuffer(raw, np.uint8, row_bytes, pos + 1).astype(np.int64)
        pos += 1 + row_bytes
        if f == 0:
            cur = row
        elif f == 2:
            cur = (row + prior) & 0xff
        elif f == 1:
            cur = row.copy()
            for i in range(bpp, row_bytes):
                cur[i] = (cur[i] + cur[i - bpp]) & 0xff
        elif f == 3:
            cur = row.copy()
            for i in range(row_bytes):
                left = cur[i - bpp] if i >= bpp else 0
                cur[i] = (cur[i] + ((left + prior[i]) >> 1)) & 0xff
        elif f == 4:
            cur = row.copy()
            for i in range(row_bytes):
                a = cur[i - bpp] if i >= bpp else 0
                b = prior[i]
                c = prior[i - bpp] if i >= bpp else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pred = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
                cur[i] = (cur[i] + pred) & 0xff
        else:
            raise ValueError(f"bad PNG filter {f}")
        out[y] = cur
        prior = cur
    return out


def _pdf_string(s):
    """Bytes of a PDF literal string (as written by matplotlib)."""
    out = bytearray()
    i = 0
    while i < len(s):
        c = s[i]
        if c == 0x5c:  # backslash
            n = s[i + 1]
            esc = {ord("n"): 10, ord("r"): 13, ord("t"): 9, ord("b"): 8, ord("f"): 12}
            if n in esc:
                out.append(esc[n])
                i += 2
            elif 0x30 <= n <= 0x37:
                j = i + 1
                while j < len(s) and j < i + 4 and 0x30 <= s[j] <= 0x37:
                    j += 1
                out.append(int(s[i + 1:j], 8))
                i = j
            else:
                out.append(n)
                i += 2
        else:
            out.append(c)
            i += 1
    return bytes(out)


def _pdf_image(head, data):
    width = int(re.search(rb"/Width (\d+)", head).group(1))
    height = int(re.search(rb"/Height (\d+)", head).group(1))
    bits = int(re.search(rb"/BitsPerComponent (\d+)", head).group(1))
    palette = None
    m = re.search(rb"/ColorSpace \[ /Indexed /DeviceRGB (\d+)\s*\((.*)\)\s*\]", head, re.S)
    if m:
        channels = 1
        palette = np.frombuffer(_pdf_string(m.group(2)), np.uint8).reshape(-1, 3)
    else:
        cs = re.search(rb"/ColorSpace /(\w+)", head).group(1)
        channels = {b"DeviceGray": 1, b"DeviceRGB": 3}[cs]
    if b"/FlateDecode" in head:
        data = zlib.decompress(data)
    row_bytes = (width * channels * bits + 7) // 8
    if b"/Predictor" in head:
        rows = _unfilter(data, width, height, max(1, channels * bits // 8), row_bytes)
    else:
        rows = np.frombuffer(data, np.uint8)[:height * row_bytes].reshape(height, row_bytes)
    if bits < 8:
        samples = np.unpackbits(rows, axis=1)[:, :width * bits].reshape(height, width, bits)
        samples = (samples * (1 << np.arange(bits - 1, -1, -1))).sum(axis=2).astype(np.uint8)
    else:
        samples = rows.reshape(height, width, channels)
    if palette is not None:
        samples = palette[samples.reshape(height, width)]
    smask = re.search(rb"/SMask (\d+) 0 R", head)
    return ("IMAGE[" + _digest(samples) + (f" smask={smask.group(1).decode()}" if smask else "")
            + "]").encode()


def pdf(data):
    """PDF objects with images decoded, streams decompressed, stream lengths
    and cross-reference offsets dropped."""
    objects = re.findall(rb"(?ms)^(\d+) 0 obj\n(.*?)\nendobj\n", data)
    lengths = set(re.findall(rb"/Length (\d+) 0 R", data))
    out = []
    for num, body in objects:
        if num in lengths:
            continue
        m = re.match(rb"(?s)(<<.*?>>)\nstream\n(.*)\nendstream$", body)
        if m is None:
            out.append(num + b": " + body)
            continue
        head, stream = m.groups()
        head_wo_length = re.sub(rb"\s*/Length \d+ 0 R", b"", head)
        if b"/Subtype /Image" in head:
            out.append(num + b": " + _pdf_image(head, stream))
        else:
            if b"/FlateDecode" in head:
                stream = zlib.decompress(stream)
            out.append(num + b": " + head_wo_length + b"\n" + stream)
    trailer = re.search(rb"(?s)trailer\n(<<.*?>>)", data)
    out.append(trailer.group(1) if trailer else b"")
    return out
