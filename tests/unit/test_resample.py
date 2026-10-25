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
sys.exit(1 if fails else 0)
