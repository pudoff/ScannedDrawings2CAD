"""Reject a clipped guard raster before calling the immutable legacy validator."""
from contextlib import contextmanager
import numpy as np

from . import arrow_path_recovery_v1 as legacy


def safe_validator(previous,events):
    def validate(candidate,mask,stroke,text_polygons):
        points=np.asarray(candidate['polygon_px'],float)
        if points.shape==(3,2) and np.isfinite(points).all():
            right,bottom=np.ceil(points.max(axis=0)).astype(int)+1
            if right>mask.shape[1] or bottom>mask.shape[0]:
                event={'reason':'native_guard_raster_clipped','polygon_px':points.tolist(),
                       'guard_edge_px':[int(right),int(bottom)],'source_size_px':list(mask.shape[::-1])}
                events.append(event)
                return False,{'reason':'native_guard_raster_clipped'}
        return previous(candidate,mask,stroke,text_polygons)
    return validate


@contextmanager
def guarded_native_heads():
    previous=legacy.native_head_check;events=[]
    legacy.native_head_check=safe_validator(previous,events)
    try:yield events
    finally:legacy.native_head_check=previous
