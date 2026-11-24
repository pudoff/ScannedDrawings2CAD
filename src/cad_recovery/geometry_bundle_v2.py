"""Experimental bundle using exact-compacted QA; the v1 release stays immutable."""
from copy import deepcopy
import cv2
import numpy as np
from .geometry_bundle_v1 import matrix, validate_geometry, rebuild, inspect, POLICY as OLD_POLICY
from .observed_geometry_qa_v2 import geometry_qa
from .part_admission_v1 import object_sha
from .grid_role_revision_v1 import sha
from .stroke_roles_v1 import foreground
from .curve_graph_v1 import width_evidence

POLICY = {**OLD_POLICY, 'version': 'geometry-bundle/2.0', 'QA': 'exact_compacted_topology/2.0',
    'fallback_scope': 'whole observed view; no unverified local promotion'}


def select(gray, source, proposed, *, input_to_native, models, extraction_complete=True):
    m=matrix(input_to_native); validate_geometry(gray,source,proposed)
    before=object_sha([source,proposed]); qa=geometry_qa(source,proposed,m)
    accepted=bool(qa['decision']['accepted'] and qa['decision']['complete'] and extraction_complete)
    revision=object_sha({'pixels':sha(gray),'input_size_px':list(gray.shape[::-1]),'matrix':m.tolist(),
        'source':source,'proposed':proposed,'models':models,'policy':POLICY,'modeled_accepted':accepted})
    prefix='gb2-'+revision
    bound_source=[{**deepcopy(s),'original_id':s['id'],'original_path_id':s['path_id'],
        'id':prefix+'/source/'+s['id'],'path_id':prefix+'/path/'+s['path_id']} for s in source]
    lookup={s['original_path_id']:s for s in bound_source}; selected=[]
    ink=foreground(gray); distance=cv2.distanceTransform(ink,cv2.DIST_L2,5)
    for i,g in enumerate(proposed if accepted else source):
        s=lookup[g['path_id']]; lo,hi=g['source_index_interval'] if accepted else (0,len(g['points_px'])-1)
        c={**deepcopy(g),'id':prefix+'/primitive/'+str(i),'path_id':s['path_id'],
            'source_observation_id':s['id'],'source_index_interval':[lo,hi],
            'support_points':deepcopy(s['points_px'][lo:hi+1]),'classification':'unclassified',
            'semantic_role':'unknown','semantic_ownership_confirmed':False,'physical_geometry_approved':False}
        if not accepted:c['review_status']='exact_source_fallback_observation'
        c['local_width']=width_evidence(c,ink,distance);selected.append(c)
    validate_geometry(gray,bound_source,selected)
    if before!=object_sha([source,proposed]):raise ValueError('Selection changed source geometry')
    return {'revision':revision,'prefix':prefix,'source':bound_source,'geometry':selected,
        'selection':{'mode':'modeled_QA_pass' if accepted else 'exact_source_fallback',
            'candidate_QA':qa,'candidate_accepted':accepted,'extraction_complete':bool(extraction_complete),
            'source_unchanged':True,'fallback_is_CAD_approval':False,'geometry_export_approved':False}}


def build_from_observations(gray, source, proposed, *, input_to_native, models, extraction_complete):
    """Only SHA-checked observations; no inherited graph/roles/heads are inputs."""
    before=sha(gray);m=matrix(input_to_native)
    chosen=select(gray,source,proposed,input_to_native=m,models=models,extraction_complete=extraction_complete)
    dependents=rebuild(gray,chosen['source'],chosen['geometry'],input_to_native=m,prefix=chosen['prefix'])
    raw=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]
    packet={**chosen,**dependents,'raw_source':deepcopy(source),'proposed_geometry':deepcopy(proposed),
        'input_sha256':before,'input_size_px':list(gray.shape[::-1]),'input_to_native':m.tolist(),
        'native_to_input':np.linalg.inv(m).tolist(),'models':deepcopy(models),'policy':deepcopy(POLICY),
        'source_pixels_erased':0,'raw_ink':raw,'preprocessing_excluded_ink':np.where((raw>0)&(foreground(gray)==0),255,0).astype(np.uint8),
        'production_changed':False,'physical_geometry_approved':False}
    packet['inspection']=inspect(gray,packet,m)
    if sha(gray)!=before or not packet['inspection']['valid']:raise ValueError('Invalid rebuilt experimental packet')
    return packet
