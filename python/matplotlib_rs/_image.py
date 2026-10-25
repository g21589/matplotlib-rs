"""
Replacement for matplotlib's ``_image`` extension module.

The image resampling of the Rust core is not wired up yet; ``resample``
raises until it is.
"""

NEAREST = 0
BILINEAR = 1
BICUBIC = 2
SPLINE16 = 3
SPLINE36 = 4
HANNING = 5
HAMMING = 6
HERMITE = 7
KAISER = 8
QUADRIC = 9
CATROM = 10
GAUSSIAN = 11
BESSEL = 12
MITCHELL = 13
SINC = 14
LANCZOS = 15
BLACKMAN = 16


def resample(input_array, output_array, transform, interpolation=NEAREST,
             resample=False, alpha=1.0, norm=False, radius=1.0):
    raise NotImplementedError("matplotlib_rs: image resampling is not supported yet")
