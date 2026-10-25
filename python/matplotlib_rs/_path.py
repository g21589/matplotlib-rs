"""Rust implementations of matplotlib's ``_path`` extension module."""
import numpy as np

from ._mplrs import _path as _impl

affine_transform = _impl.affine_transform
is_sorted_and_has_non_nan = _impl.is_sorted_and_has_non_nan
cleanup_path = _impl.cleanup_path
convert_path_to_polygons = _impl.convert_path_to_polygons
get_path_collection_extents = _impl.get_path_collection_extents
point_in_path = _impl.point_in_path
points_in_path = _impl.points_in_path
path_in_path = _impl.path_in_path
point_in_path_collection = _impl.point_in_path_collection
path_intersects_path = _impl.path_intersects_path
path_intersects_rectangle = _impl.path_intersects_rectangle
clip_path_to_rect = _impl.clip_path_to_rect


def count_bboxes_overlapping_bbox(bbox, bboxes):
    return _impl.count_bboxes_overlapping_bbox(
        np.asarray(bbox, dtype=float), np.asarray(bboxes, dtype=float).reshape(-1, 2, 2))


def convert_to_string(*args, **kwargs):
    raise NotImplementedError("_path.convert_to_string is not implemented in matplotlib_rs yet")
