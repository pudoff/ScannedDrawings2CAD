"""Per-frame exact source templates with legacy fallback and separate uncertainty."""
import math
import cv2
import numpy as np
from . import native_frame_v1 as previous
from .frame_raster_template_v1 import fit
from .frame_grid_guard_v1 import affine,project
from .grid_role_revision_v1 import sha
from .part_admission_v1 import object_sha

refine=previous.refine
MASKS=('mid','safe','possible','coverage','failed_coverage')
POLICY={'version':'native-frame/2.0','new_frame_roles':False,'semantic_frame_authority':False,
    'source_template':'closed integer-corner stroke; exact uncontaminated profiles and source containment',
    'legacy_fallback':'per frame, preserving unsupported neighbouring hypotheses',
    'native_prediction_transfer':False,'production_admitted':False,'physical_geometry_approved':False}


def source_masks(gray,page,detector,*,context_gray=None,legacy_source=None):
    # The legacy function verifies page bounds, crop identity and source type.
    if legacy_source is None:
        legacy=previous.source_masks(gray,page,detector,context_gray=context_gray)
    else:
        # Only exact source-bound masks from a checked frozen legacy producer.
        previous.checked_gray(gray)
        if (not page['source_verified'] or not detector['complete'] or
            legacy_source['source_gray_sha256']!=sha(gray) or legacy_source['page_sha256']!=object_sha(page) or
            legacy_source['detector_sha256']!=object_sha(detector) or legacy_source['GT_used'] or legacy_source['native_prediction_transfer']):
            raise ValueError('Legacy frame source binding differs')
        if context_gray is not None:
            previous.checked_gray(context_gray)
            box=page['patch_bbox']
            if list(context_gray.shape[::-1])!=page['source_size_px'] or not np.array_equal(gray,context_gray[box[1]:box[3],box[0]:box[2]]):
                raise ValueError('Expanded context differs from original source')
        for name in MASKS:
            a=legacy_source[name]
            if a.shape!=gray.shape or a.dtype!=np.uint8 or sha(a)!=legacy_source['mask_sha256'][name]:
                raise ValueError('Stale legacy frame mask')
        legacy={**legacy_source,**{name:legacy_source[name].copy() for name in MASKS}}
    if not detector['frames'] or not legacy['complete']:
        return {**legacy,'frame_templates':[],'template_search_complete':True,'exact_template_frames':0}
    working=gray if context_gray is None else context_gray
    ink=cv2.threshold(working,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]>0
    matrix=affine(page['overview_to_source'])
    if context_gray is None:matrix=affine(page['source_to_patch'])@matrix
    result={name:np.zeros(gray.shape,np.uint8) for name in MASKS};models=[];complete=True;templates=[]
    box=page['patch_bbox']
    for index,frame in enumerate(detector['frames']):
        sides=[]
        for side in frame['sides']:
            points=np.asarray(side['points_overview'],float);axis=(points[1]-points[0])/np.linalg.norm(points[1]-points[0]);normal=np.array([-axis[1],axis[0]])
            transformed=project(points,matrix);delta=transformed[1]-transformed[0];u=delta/np.linalg.norm(delta);n=np.array([-u[1],u[0]])
            scale=abs(n@matrix[:2,:2]@normal)
            sides.append({'points_native':transformed.tolist(),'width_native_px':side['stroke_width_overview_px']*scale,
                          'error_native_px':side['position_uncertainty_overview_px']*scale})
        template=fit(ink,sides);complete &= template['complete']
        row={k:v for k,v in template.items() if k not in ('mid','safe','possible')}
        row.update(frame_index=index,frame_sha256=object_sha(frame),
            coordinate_system='original_source_pixel_centres' if context_gray is not None else 'ROI_source_pixel_centres')
        models.append(row)
        templates.append(template)
    if not complete:
        return {**legacy,'frame_templates':models,'template_search_complete':False,'exact_template_frames':0}
    # Reuse the union only when no unsupported neighbour can lose its corridor.
    # Mixed fitted/unfitted frames retain the independent per-frame fallback.
    all_fit=all(t['fitted'] for t in templates)
    if not any(t['fitted'] for t in templates):
        return {**legacy,'frame_templates':models,'template_search_complete':True,'exact_template_frames':0}
    if all_fit:
        result['coverage']=legacy['coverage']|legacy['failed_coverage']
    for frame,template in zip(detector['frames'],templates):
        old=({name:np.zeros(gray.shape,np.uint8) for name in MASKS} if all_fit else
             previous.source_masks(gray,page,{**detector,'frames':[frame]},context_gray=context_gray))
        if template['fitted']:
            for name in ('mid','safe','possible'):
                mask=template[name]
                if context_gray is not None:mask=mask[box[1]:box[3],box[0]:box[2]]
                old[name]=mask.astype(np.uint8)*255
            # Only this completely source-fitted frame can clear its own failed
            # legacy corridor; a different unsupported frame keeps its fallback.
            old['coverage'] |= old['failed_coverage']|old['possible']
            old['failed_coverage'][:]=0
        for name in MASKS:result[name] |= old[name]
    return {**legacy,**result,'mask_sha256':{name:sha(result[name]) for name in MASKS},
        'detector_sha256':object_sha(detector),'page_sha256':object_sha(page),
        'frame_templates':models,'template_search_complete':complete,
        'exact_template_frames':sum(m['fitted'] for m in models),
        'GT_used':False,'native_prediction_transfer':False}
