from . import semantic_kitti
from . import nuScenes
from . import semantic_poss

__all__ = [
    "semantic_kitti",
    "nuScenes",
    "semantic_poss",
    "RangeViewLoader",
    "custom_collate_kpconv_fn",
]


def __getattr__(name):
    if name in ("RangeViewLoader", "custom_collate_kpconv_fn"):
        from .range_view_loader import RangeViewLoader, custom_collate_kpconv_fn

        return {
            "RangeViewLoader": RangeViewLoader,
            "custom_collate_kpconv_fn": custom_collate_kpconv_fn,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
