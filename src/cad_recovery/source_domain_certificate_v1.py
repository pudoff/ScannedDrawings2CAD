"""Certify observed source coverage separately from bounded primitive fitting."""
from copy import deepcopy
import numpy as np
from skimage.morphology import skeletonize
from .stroke_roles_v1 import checked_gray,foreground
from .geometry_bundle_v1 import validate_geometry
from .grid_role_revision_v1 import sha
from .part_admission_v1 import object_sha


def certify(packet):
    gray=packet['gray'];checked_gray(gray);before=object_sha([packet['source'],packet['geometry']]);pixels=sha(gray)
    if packet.get('input_sha256')!=pixels:raise ValueError('Source packet belongs to another raster')
    source=packet['source'];geometry=packet['geometry']
    if len(source)>4096:raise ValueError('Source inventory bound')
    validate_geometry(gray,source,geometry)
    observed=np.zeros_like(gray);count=0
    for path in source:
        points=np.asarray(path['points_px'],float);count+=len(points)
        if count>2000000:raise ValueError('Source point bound')
        if (points.ndim!=2 or points.shape[1]!=2 or not len(points) or not np.isfinite(points).all()
                or not np.array_equal(points,np.rint(points)) or (points<0).any()
                or (points[:,0]>=gray.shape[1]).any() or (points[:,1]>=gray.shape[0]).any()):
            raise ValueError('Observed source pixel centres required')
        q=points.astype(int)
        if len(q)>1 and (abs(np.diff(q,axis=0))>1).any():raise ValueError('Discontinuous source path')
        observed[q[:,1],q[:,0]]=255
    skeleton=skeletonize(foreground(gray)>0,method='zhang').astype(np.uint8)*255
    complete=np.array_equal(observed,skeleton)
    if before!=object_sha([packet['source'],packet['geometry']]) or pixels!=sha(gray):raise ValueError('Source certifier mutated observations')
    return {'version':'source-domain-certificate/1.0','complete':complete,'source_sha256':pixels,
        'source_and_geometry_sha256':before,'source_mask_sha256':sha(observed),'skeleton_sha256':sha(skeleton),
        'source_paths':len(source),'source_points':count,'source_pixels':int(np.count_nonzero(observed)),
        'missing_skeleton_pixels':int(np.count_nonzero((skeleton>0)&(observed==0))),
        'extra_source_pixels':int(np.count_nonzero((observed>0)&(skeleton==0))),
        'geometric_fit_complete':bool(packet['selection']['extraction_complete']),
        'legacy_complete':packet['complete'],'legacy_graph_search_incomplete':packet['graph']['search_incomplete'],
        'scope':'provided_observed_source_mask_and_bound_geometry_partition_only',
        'foreground_policy':'frozen_foreground_then_zhang_skeleton_not_all_raw_ink',
        'full_CV_semantic_or_physical_completeness':False}
