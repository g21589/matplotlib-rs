"""Transforms and bboxes whose values live in the Rust model.

``ax.transData``, ``ax.transAxes``, ``fig.transFigure``, ``ax.bbox``, ... are
"live" nodes: they always read the current state of the Rust figure, and
they carry a ``_rs_spec`` that tells the Rust renderer which symbolic
transform they stand for, so that artists using them are drawn with the
exact same transform evaluation as matplotlib.
"""
import numpy as np

from .transforms import (
    Affine2D, Affine2DBase, BboxBase, BlendedAffine2D, BlendedGenericTransform,
    BboxTransformTo, CompositeAffine2D, CompositeGenericTransform, IdentityTransform,
    ScaledTranslation, Transform, TransformWrapper)


class LiveAffine(Affine2DBase):
    """An affine transform whose matrix is fetched from Rust on demand."""
    _live = True

    def __init__(self, getter, spec, name=""):
        super().__init__(shorthand_name=name)
        self._getter = getter
        self._rs_spec = spec

    def get_matrix(self):
        return np.array(self._getter(), dtype=float)

    def inverted(self):
        return Affine2D(np.linalg.inv(self.get_matrix()))

    __hash__ = object.__hash__

    def __eq__(self, other):
        if isinstance(other, LiveAffine):
            return self._rs_spec == other._rs_spec and self._getter == other._getter
        return False


class LiveTransData(Transform):
    """``ax.transData`` = transScale + (transLimits + transAxes)."""
    _live = True
    input_dims = output_dims = 2
    has_inverse = True
    is_separable = True

    def __init__(self, ax):
        super().__init__(shorthand_name="transData")
        self._ax = ax
        self._rs_spec = "data"

    @property
    def is_affine(self):
        return (self._ax.xaxis.get_scale() == "linear"
                and self._ax.yaxis.get_scale() == "linear")

    def _scale_transforms(self):
        return (self._ax.xaxis.get_transform(), self._ax.yaxis.get_transform())

    def transform_non_affine(self, values):
        values = np.asanyarray(values, dtype=float)
        if self.is_affine:
            return values
        tx, ty = self._scale_transforms()
        out = np.array(values, dtype=float)
        if values.ndim == 1:
            out = out.reshape(1, -1)
        out[:, 0] = tx.transform_non_affine(out[:, 0])
        out[:, 1] = ty.transform_non_affine(out[:, 1])
        return out.reshape(values.shape)

    def get_affine(self):
        return Affine2D(np.array(self._ax._rs.transform_matrix("data"), dtype=float))

    def get_matrix(self):
        return self.get_affine().get_matrix()

    def transform_path_non_affine(self, path):
        from .path import Path
        x = self.transform_non_affine(path.vertices)
        return Path._fast_from_codes_and_verts(x, path.codes, path)

    def inverted(self):
        aff_inv = self.get_affine().inverted()
        if self.is_affine:
            return aff_inv
        tx, ty = self._scale_transforms()
        from .transforms import blended_transform_factory
        return aff_inv + blended_transform_factory(tx.inverted(), ty.inverted())

    def frozen(self):
        if self.is_affine:
            return self.get_affine()
        return self

    __hash__ = object.__hash__

    def __eq__(self, other):
        return isinstance(other, LiveTransData) and other._ax is self._ax


class LiveBbox(BboxBase):
    """A bbox read from Rust (``ax.bbox``, ``ax.viewLim``, ``fig.bbox``)."""
    _live = True

    def __init__(self, getter, setter=None, name=""):
        super().__init__(shorthand_name=name)
        self._getter = getter
        self._setter = setter

    def get_points(self):
        x0, y0, x1, y1 = self._getter()
        return np.array([[x0, y0], [x1, y1]], dtype=float)

    def _set_points(self, pts):
        if self._setter is None:
            raise AttributeError("this bbox is read-only")
        (x0, y0), (x1, y1) = np.asarray(pts, float)
        self._setter(x0, y0, x1, y1)

    @property
    def intervalx(self):
        return self.get_points()[:, 0]

    @intervalx.setter
    def intervalx(self, interval):
        p = self.get_points()
        p[:, 0] = interval
        self._set_points(p)

    @property
    def intervaly(self):
        return self.get_points()[:, 1]

    @intervaly.setter
    def intervaly(self, interval):
        p = self.get_points()
        p[:, 1] = interval
        self._set_points(p)

    def frozen(self):
        from .transforms import Bbox
        return Bbox(self.get_points().copy())

    __hash__ = object.__hash__


def tspec(t):
    """The Rust transform specification of a Transform object."""
    if t is None:
        return "identity"
    spec = getattr(t, "_rs_spec", None)
    if spec is not None:
        return spec
    if isinstance(t, IdentityTransform):
        return "identity"
    if isinstance(t, TransformWrapper):
        return tspec(t._child)
    if isinstance(t, (CompositeGenericTransform, CompositeAffine2D)):
        return ("then", tspec(t._a), tspec(t._b))
    if isinstance(t, (BlendedGenericTransform, BlendedAffine2D)):
        return ("blend", tspec(t._x), tspec(t._y))
    if isinstance(t, ScaledTranslation):
        if getattr(t._scale_trans, "_rs_spec", None) == "dpi":
            return ("scaled_translation", float(t._t[0]), float(t._t[1]))
    if isinstance(t, BboxTransformTo):
        bs = getattr(t._boxout, "_rs_bbox_spec", None)
        if bs is not None:
            return bs
    if not hasattr(t, "_as_mpl_transform") and t.is_affine:
        m = t.get_matrix()
        return ("affine", [list(map(float, m[0])), list(map(float, m[1]))])
    raise NotImplementedError(f"matplotlib_rs cannot use transform {t!r} yet")
