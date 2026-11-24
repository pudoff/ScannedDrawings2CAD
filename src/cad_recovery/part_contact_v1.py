"""Source-pixel uncertainty for new, canonically witnessed contour proposals.

Uses no labels, frame masks, native-view role predictions or case identifiers.
The uncertainty radius is a transformed analysis-pixel footprint, not a tuned
physical tolerance or an assertion of semantic truth.
"""
from copy import deepcopy
import math
import cv2
import numpy as np
from PIL import Image
from .part_admission_v1 import analyze as canonical_gate, object_sha
from .grid_role_revision_v1 import sha
from .stroke_roles_v1 import checked_gray
from .supported_role_overlay_v1 import source_ownership

POLICY={'version':'part-contact/1.0','maximum_native_pixels':4194304,
    'maximum_intervals':4096,'maximum_source_points':2000000,
    'source_core_margin':'strictly greater than half diagonal of transformed input pixel',
    'owner_quantization':['round','floor','ceil'],'output':'native_source_pixel_centres',
    'new_part_only':True,'GT_used':False,'native_prediction_transfer':False,
    'physical_geometry_approved':False,'production_admitted':False}
ARMS=('no_part','canonical','source_core','owner_core')


def matrix_for(gray,source_gray,matrix):
    checked_gray(gray);checked_gray(source_gray)
    matrix=np.asarray(matrix,float)
    sx,sy=np.array(source_gray.shape[::-1])/np.array(gray.shape[::-1])
    expected=np.array([[sx,0,(sx-1)/2],[0,sy,(sy-1)/2],[0,0,1.]])
    if matrix.shape!=(3,3) or not np.isfinite(matrix).all() or not np.allclose(matrix,expected,atol=1e-9):
        raise ValueError('Native pixel-centre transform does not match the original raster')
    return matrix


def bind_source(gray,source_gray,matrix,source_pin,*,resize_method='CV_AREA'):
    matrix=matrix_for(gray,source_gray,matrix)
    if set(source_pin)!={'path','sha256'} or len(source_pin['sha256'])!=64:
        raise ValueError('Explicit checked original source pin required')
    if resize_method not in ('CV_AREA','PIL_BOX'):
        raise ValueError('Explicit source resize method required')
    if gray.shape==source_gray.shape:expected=source_gray
    elif resize_method=='CV_AREA':expected=cv2.resize(source_gray,gray.shape[::-1],interpolation=cv2.INTER_AREA)
    else:expected=np.array(Image.fromarray(source_gray).resize(gray.shape[::-1],Image.Resampling.BOX))
    if not np.array_equal(gray,expected):
        raise ValueError('Input pixels differ from their bound original source')
    return {'kind':'checked_original_source_pixels','source_pin':deepcopy(source_pin),
        'source_sha256':sha(source_gray),'input_sha256':sha(gray),'matrix':matrix.tolist(),
        'resize_method':resize_method,'native_role_predictions_used':False,'GT_used':False}


def analyze(gray,graph,raw,previous,candidate,owners,canonical,*,authority,evidence,
            input_to_native,source_gray,source_binding,baseline_native=None):
    old=canonical_gate(gray,graph,raw,previous,candidate,owners,canonical,authority=authority,evidence=evidence)
    # Source pixels must be present and authenticated even when the native view
    # already exists in an experiment. A role map is never a source substitute.
    if source_gray is None or source_binding is None:
        raise ValueError('Bound original source pixels required')
    matrix=matrix_for(gray,source_gray,input_to_native)
    verified=bind_source(gray,source_gray,matrix,source_binding.get('source_pin',{}),
                         resize_method=source_binding.get('resize_method'))
    if source_binding!=verified:
        raise ValueError('Stale or rebound original source evidence')
    size=source_gray.shape[::-1]
    def native(a):return cv2.resize(a,size,interpolation=cv2.INTER_NEAREST)
    previous_n=native(previous);no_part=native(old['no_part']);canonical_n=native(old['canonical'])
    owner_n=native(owners.astype(np.float32)).astype(np.int32)
    new=(canonical_n==1)&(previous_n!=1)
    if baseline_native is not None:
        if baseline_native.shape!=source_gray.shape or baseline_native.dtype!=np.uint8:
            raise ValueError('Aligned native baseline required')
        # External frame refinements may only withdraw a newly proposed grid role.
        delta=baseline_native!=no_part
        valid=(no_part==8)&(previous_n!=8)&(baseline_native==previous_n)
        if np.any(delta&~valid):raise ValueError('Native baseline contains unrelated role changes')
        no_part=baseline_native.copy();canonical_n=no_part.copy();canonical_n[new]=1
    maps={'no_part':no_part,'canonical':canonical_n,'source_core':no_part.copy(),'owner_core':no_part.copy()}
    uncertainty={a:np.zeros(source_gray.shape,np.uint8) for a in ARMS[2:]}
    audit={'canonical_complete':old['complete'],'source_points':0,'owner_variants':[],
        'limits':[],'source_pixels_erased':0,'native_prediction_transfer':False,
        'canonical_decisions_sha256':object_sha(old['decisions'])}
    footprint=.5*math.hypot(matrix[0,0],matrix[1,1])
    if new.any() and old['complete']:
        ink=cv2.threshold(source_gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]
        distance=cv2.distanceTransform(ink,cv2.DIST_L2,5)
        core=new&(ink>0)&(distance>footprint)
        maps['source_core'][core]=1
        stable=core.copy();seen=set()
        for quantization in POLICY['owner_quantization']:
            projected=[]
            for p in graph['proposals']:
                points=np.asarray(p['points_px'],float)
                if points.ndim!=2 or points.shape[1]!=2 or not len(points) or not np.isfinite(points).all():
                    raise ValueError('Finite source interval points required')
                points=points@matrix[:2,:2].T+matrix[:2,2]
                q={'round':np.rint,'floor':np.floor,'ceil':np.ceil}[quantization](points)
                audit['source_points']+=len(points)
                if audit['source_points']>3*POLICY['maximum_source_points']:
                    audit['limits'].append('projected_source_point_budget');break
                projected.append({'id':p['id'],'points_px':q.tolist()})
            if audit['limits']:break
            signature=object_sha(projected)
            if signature in seen:continue
            seen.add(signature)
            ownership,work=source_ownership(ink,projected)
            audit['owner_variants'].append({'quantization':quantization,**work})
            if ownership is None or work.get('incomplete'):
                audit['limits'].append('native_source_ownership_unavailable');break
            stable &= ownership['owner']==owner_n
        if not audit['limits']:maps['owner_core'][stable]=1
    for arm in ARMS[2:]:
        uncertainty[arm][new&(maps[arm]!=1)]=255
    # The source_core band does not need the expensive ownership search. If that
    # search is incomplete, only owner_core abstains; this distinction is explicit.
    if audit['limits']:maps['owner_core'][:]=no_part;uncertainty['owner_core'][new]=255
    return {**maps,'uncertainty':uncertainty,'complete':old['complete'] and not audit['limits'],
        'arm_complete':{'source_core':old['complete'],'owner_core':old['complete'] and not audit['limits']},
        'audit':audit,'input_footprint_half_diagonal_native_px':footprint,
        'source_binding_sha256':object_sha(source_binding),'source_context_used':True,
        'new_part_candidate_native_pixels':int(new.sum()),'output_coordinate_system':POLICY['output'],
        'source_pixels_erased':0,'production_changed':False,'physical_geometry_approved':False}
