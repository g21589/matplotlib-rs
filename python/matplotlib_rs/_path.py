"""Rust implementations of matplotlib's ``_path`` extension module."""
from ._mplrs import _path as _impl

affine_transform = _impl.affine_transform
count_bboxes_overlapping_bbox = _impl.count_bboxes_overlapping_bbox
is_sorted_and_has_non_nan = _impl.is_sorted_and_has_non_nan
cleanup_path = _impl.cleanup_path
convert_path_to_polygons = _impl.convert_path_to_polygons
get_path_collection_extents = _impl.get_path_collection_extents


def _missing(name):
    def f(*args, **kwargs):
        raise NotImplementedError(f"_path.{name} is not implemented in matplotlib_rs yet")
    f.__name__ = name
    return f


for _n in ["point_in_path", "points_in_path", "path_in_path", "path_intersects_path",
           "path_intersects_rectangle", "point_in_path_collection", "clip_path_to_rect",
           "convert_to_string"]:
    globals()[_n] = _missing(_n)
