"""Bit-exact comparison of matplotlib_rs._image.resample with matplotlib's."""

import itertools
import sys

import numpy as np

sys.path.insert(0, "python")

from matplotlib import _image as mimage  # noqa: E402
from matplotlib import transforms as mtrans  # noqa: E402
from matplotlib_rs import _image as rimage  # noqa: E402

rng = np.random.default_rng(0)

DTYPES = [np.uint8, np.int8, np.uint16, np.float32, np.float64]
INTERPS = list(range(17))


def make(dtype, shape):
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return rng.integers(info.min, info.max, size=shape, endpoint=True).astype(dtype)
    a = rng.random(shape).astype(dtype)
    return a


def affines():
    yield None
    yield mtrans.Affine2D().scale(3.3, 2.7).translate(1.2, -0.7)
    yield mtrans.Affine2D().scale(0.37, 0.41).translate(0.3, 0.2)
    yield mtrans.Affine2D().scale(-1, 1).translate(9, 0)
    yield mtrans.Affine2D().rotate_deg(17).scale(1.7).translate(4, 1)
    yield mtrans.Affine2D().scale(5.0, 5.0)
    yield mtrans.Affine2D().scale(0.02, 0.03).translate(3, 4)
    yield mtrans.Affine2D().scale(40, 0.5).translate(-100, 2)
    yield mtrans.Affine2D().skew_deg(25, 10).scale(0.3)


class Polar(mtrans.Transform):
    input_dims = output_dims = 2
    is_affine = False

    def transform_non_affine(self, xy):
        x, y = xy[:, 0], xy[:, 1]
        return np.column_stack([x * 1.3 + np.sin(y / 4) * 3, y * 1.1 + np.cos(x / 5) * 2])

    def inverted(self):
        return Polar()


def run(mod, inp, out, t, interp, resample, alpha, norm, radius):
    o = out.copy()
    if mod is mimage:
        interp = mimage._InterpolationType(interp)
    mod.resample(inp, o, t, interp, resample, alpha, norm, radius)
    return o


fails = 0
count = 0
for dtype, ndim in itertools.product(DTYPES, (2, 3)):
    shape = (11, 13) if ndim == 2 else (11, 13, 4)
    inp = make(dtype, shape)
    if ndim == 3 and np.issubdtype(dtype, np.floating):
        # premultiplied input as matplotlib feeds it
        inp[..., :3] *= inp[..., 3:]
    for t in list(affines()) + [Polar()]:
        for interp in INTERPS:
            for resample, alpha, norm, radius in [(False, 1.0, False, 1.0), (True, 1.0, True, 4.0),
                                                  (True, 0.3, False, 2.5), (False, 0.7, True, 1.0)]:
                out_shape = (23, 29) + shape[2:]
                out = np.zeros(out_shape, dtype)
                if dtype == np.float32 and ndim == 2:
                    out[3:7, 5:9] = 0.25
                a = run(mimage, inp, out, t, interp, resample, alpha, norm, radius)
                b = run(rimage, inp, out, t, interp, resample, alpha, norm, radius)
                count += 1
                if not np.array_equal(a, b, equal_nan=True):
                    fails += 1
                    if fails <= 15:
                        d = np.argwhere(a != b)
                        print("MISMATCH", np.dtype(dtype).name, ndim, type(t).__name__,
                              None if t is None or not t.is_affine else t.get_matrix()[:2].ravel().round(3),
                              interp, resample, alpha, norm, radius, len(d), d[:2].tolist(),
                              a[tuple(d[0])], b[tuple(d[0])])
print(f"{count - fails}/{count} identical")

# Large outputs (spans generated on several threads), and matplotlib_rs's
# resample_indexed: resampling lut[idx] without making it.
big_fails = 0
big_count = 0
LUT = rng.random((259, 4))
LUT[:5, 3] = [0, 0.25, 0.5, 1, 1]
LUT_PM = LUT * LUT[:, 3:4]
LUT_PM[:, 3] = LUT[:, 3]
for in_shape, out_shape in [((300, 200), (170, 130)), ((900, 1100), (150, 140)),
                            ((40, 50), (160, 210)), ((1, 700), (130, 140))]:
    idx = rng.integers(0, len(LUT), in_shape).astype(np.uint32)
    rgba = LUT_PM[idx]
    sy, sx = out_shape[0] / in_shape[0], out_shape[1] / in_shape[1]
    for t in [None, mtrans.Affine2D().scale(sx, sy),
              mtrans.Affine2D().scale(-sx * 0.9, sy * 1.1).translate(out_shape[1] * 0.95, -2.5),
              mtrans.Affine2D().rotate_deg(11).scale(sx * 0.8, sy * 0.8).translate(9, 3), Polar()]:
        for interp in [0, 1, 5, 9, 14, 16]:
            for resample, alpha, norm, radius in [(True, 1.0, False, 1.0), (False, 1.0, True, 1.0),
                                                  (True, 0.6, True, 3.0)]:
                out = np.zeros(out_shape + (4,))
                out[5:9, 7:30] = 0.125
                a = run(mimage, rgba, out, t, interp, resample, alpha, norm, radius)
                b = run(rimage, rgba, out, t, interp, resample, alpha, norm, radius)
                c = out.copy()
                rimage.resample_indexed(idx, LUT_PM, c, t, interp, resample, alpha, norm, radius)
                gray_in = rgba[..., 0].copy()
                g = np.zeros(out_shape)
                ga = run(mimage, gray_in, g, t, interp, resample, alpha, norm, radius)
                gb = run(rimage, gray_in, g, t, interp, resample, alpha, norm, radius)
                for name, x, y in [("rgba", a, b), ("indexed", a, c), ("gray", ga, gb)]:
                    big_count += 1
                    if not np.array_equal(x, y, equal_nan=True):
                        big_fails += 1
                        if big_fails <= 15:
                            print("MISMATCH (large)", name, in_shape, out_shape, type(t).__name__,
                                  interp, resample, alpha, norm, radius,
                                  int((x != y).sum()))
print(f"{big_count - big_fails}/{big_count} large identical")
sys.exit(1 if fails or big_fails else 0)
