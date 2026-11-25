"""Compose frozen open evidence and source-bound negative guards, without CAD."""
from copy import deepcopy
import cv2
import numpy as np

from .open_section_witness_v5 import analyze as open_witness
from .open_source_continuation_v1 import observe as source_continuation
from .competitive_grid_admission_v2 import analyze as role_context
from .contour_role_competition_v2 import guard as role_guard
from .contour_page_competition_v1 import analyze as frame_guard, _binary, _whole_footprint_safe
from .attached_grid_competition_v1 import analyze as grid_guard
from .frame_connected_context_v1 import detect
from .native_frame_v1 import source_masks as frame_masks
from .page_grid_attachment_v1 import analyze as attach
from .page_grid_source_masks_v1 import build as grid_masks
from .grid_role_revision_v1 import cv_authority, sha
from .part_admission_v1 import object_sha
from .stroke_roles_v1 import checked_gray, foreground

POLICY = {'version':'open-contour-joint/5.0','maximum_pixels':4194304,
          'positive_witness':'open-section-witness/5.0',
          'source_support':'connected_observed_source_walk',
          'maximum_intervals':4096,'required_guards':['role','frame','grid'],
          'native_output':'subset_of_original_V3_admitted_mask',
          'baseline_and_manual_preserved':True,'geometry_modified':False,
          'production_admitted':False,'physical_geometry_approved':False}


def metadata(value):
    return {k:v for k,v in value.items() if not isinstance(v,np.ndarray)}


def negative_subset(before, result):
    _binary(result['mask'],before.shape,'guard output')
    if np.any((result['mask'] > 0) & (before == 0)):
        raise ValueError('Negative guard created positive pixels')
    return result['mask']


def native_footprint(mask, shape):
    """All original cells intersecting any selected input pixel, including edges."""
    yy,xx = np.nonzero(mask); height,width = shape; h,w = mask.shape
    left,top = xx*width//w,yy*height//h
    right,bottom = ((xx+1)*width+w-1)//w,((yy+1)*height+h-1)//h
    delta = np.zeros((height+1,width+1),np.int32)
    for y,x,sign in ((top,left,1),(top,right,-1),(bottom,left,-1),(bottom,right,1)):
        np.add.at(delta,(y,x),sign)
    return delta.cumsum(0).cumsum(1)[:-1,:-1] > 0


def build_page_context(original, source_pin):
    checked_gray(original); pixels = sha(original)
    height,width = original.shape; identity = np.eye(3).tolist()
    page = {'source':deepcopy(source_pin),'kind':'checked_full_original_pixels',
        'source_verified':True,'source_size_px':[width,height],'size_px':[width,height],
        'patch_bbox':[0,0,width,height],'source_to_patch':identity,'patch_to_source':identity,
        'overview_to_source':identity}
    cv2.setRNGSeed(0); detection = detect(original)
    cv2.setRNGSeed(0); observed = attach(original,detection)
    masks = grid_masks(original,observed)
    if detection['complete']:
        framed = frame_masks(original,page,detection)
        framed = {k:v for k,v in framed.items() if not isinstance(v,np.ndarray) or k == 'safe'}
    else:
        framed = None
    if sha(original) != pixels: raise ValueError('Page observations changed original pixels')
    return {'page':page,'detection':detection,'observations':observed,'grid_masks':masks,
        'frame_mask':framed,'source_sha256':pixels,
        'complete':bool(detection['complete'] and observed['complete'] and masks['complete'] and framed is not None and framed['complete']),
        'GT_used':False,'source_immutable':True}


def _guard_on_observed_graph(gray, packet, proposed, *, original, page_context, binding,
                   baseline_roles, protected, authority, positive=None):
    """A negative-only branch usable for actual candidates and fault diagnostics."""
    _binary(proposed,gray.shape,'proposal'); _binary(protected,gray.shape,'protection')
    if baseline_roles.shape != gray.shape or baseline_roles.dtype != np.uint8:
        raise ValueError('Aligned baseline roles required')
    immutable = [sha(a) for a in (gray,original,proposed,baseline_roles,protected)]
    before = object_sha(packet['graph']); candidate = proposed.copy(); stages = {}
    preserve = (proposed > 0) & ((baseline_roles != 255) | (protected > 0))
    candidate[preserve] = 0
    if page_context is None:
        candidate[:] = 0; candidate[preserve] = 255
        return {'mask':candidate,'complete':False,'reason':'page_context_unavailable','stages':{},'source_immutable':True}
    if page_context['source_sha256'] != sha(original) or page_context.get('GT_used') is not False:
        raise ValueError('Page context bound to another original or GT')
    context = role_context(gray,packet['graph'],packet['roles'],packet['heads'],packet['source_owner'],
        packet['ambiguous_owner_mask'],authority=authority)
    negative = {k:v for k,v in context.items() if k not in ('previous','old_candidate','old_owners','candidates','owners')}
    roles = role_guard(gray,packet['graph'],packet['source_owner'],packet['ambiguous_owner_mask'],candidate,negative,positive=positive)
    stages['role'] = metadata(roles); candidate = negative_subset(candidate,roles)
    if page_context['complete']:
        frame = frame_guard(gray,packet['graph'],packet['source_owner'],packet['ambiguous_owner_mask'],candidate,
            native_gray=original,page=page_context['page'],detection=page_context['detection'],
            source_mask=page_context['frame_mask'],input_to_native=binding['matrix'],resize_method=binding['resize_method'])
        stages['frame'] = metadata(frame); candidate = negative_subset(candidate,frame)
        grid = grid_guard(gray,packet['graph'],packet['source_owner'],packet['ambiguous_owner_mask'],candidate,
            source_gray=original,native_gray=original,page=page_context['page'],observations=page_context['observations'],
            source_masks=page_context['grid_masks'],input_to_native=binding['matrix'],resize_method=binding['resize_method'],
            baseline_roles=baseline_roles,protected=protected)
        stages['grid'] = metadata(grid); candidate = negative_subset(candidate,grid)
    complete = bool(context['complete'] and roles['complete'] and page_context['complete'] and
                    all(stages.get(k,{}).get('complete',False) for k in POLICY['required_guards']))
    if not complete: candidate[:] = 0
    candidate[preserve] = 255
    if np.any((candidate > 0) & (proposed == 0)): raise ValueError('Negative guards created positive pixels')
    if before != object_sha(packet['graph']) or immutable != [sha(a) for a in (gray,original,proposed,baseline_roles,protected)]:
        raise ValueError('Joint guards changed input observations')
    return {'mask':candidate,'complete':complete,'reason':None if complete else 'incomplete_required_context',
        'stages':stages,'source_immutable':True,'role_context_sha256':object_sha(negative)}


def analyze(gray, packet, *, original, binding, baseline_native, protected_native,
            page_context=None, producer_pin):
    checked_gray(gray)
    if baseline_native.dtype != np.uint8 or protected_native.dtype != np.uint8 or baseline_native.shape != protected_native.shape:
        raise ValueError('Aligned native baseline/protection required')
    if original is not None and original.shape != baseline_native.shape:
        raise ValueError('Full original and baseline coordinate frames differ')
    _binary(protected_native,baseline_native.shape,'native protection')
    arrays = [gray,baseline_native,protected_native,packet['roles'],packet['source_owner'],packet['ambiguous_owner_mask']]
    if original is not None: arrays.append(original)
    before = [sha(a) for a in arrays]
    graph_before = object_sha([packet['graph'],packet['heads'],binding])
    original_packet = packet
    observed, continuation = source_continuation(gray,packet['graph'])
    packet = {**packet,'graph':observed,'complete':packet.get('complete',True) and continuation['complete']}
    # Protect every input pixel whose native footprint contains an accepted
    # baseline or manual pixel, rather than transferring a nearest label.
    occupied = ((baseline_native != 255) | (protected_native > 0)).astype(np.uint8)
    free = np.where(occupied == 0,255,0).astype(np.uint8)
    protected = np.where(_whole_footprint_safe(free,gray.shape),0,255).astype(np.uint8)
    authority = cv_authority(gray,packet['roles'],producer_pin,protected=protected)
    opened = open_witness(gray,packet['graph'],packet['roles'],packet['source_owner'],packet['ambiguous_owner_mask'],packet['heads'],
        authority=authority,original_gray=original,source_binding=binding)
    raw_native = opened['native_mask'].copy() if opened['native_mask'] is not None else np.zeros_like(baseline_native)
    raw_native[(baseline_native != 255) | (protected_native > 0)] = 0
    candidate = opened['mask']; raw_pixels = int((raw_native > 0).sum())
    guarded = None; native = raw_native.copy()
    if candidate.any():
        if original is None: raise ValueError('Open candidate lacks original pixels')
        guarded = _guard_on_observed_graph(gray,packet,candidate,original=original,page_context=page_context,binding=binding,
            baseline_roles=np.full_like(gray,255),protected=protected,authority=authority,positive=opened)
        removed = (candidate > 0) & (guarded['mask'] == 0)
        native[native_footprint(removed,baseline_native.shape)] = 0
        complete = bool(opened['complete'] and guarded['complete'])
    else:
        complete = opened['complete']
    dependencies_complete = bool(packet.get('complete',True))
    complete = complete and dependencies_complete
    if not complete: native[:] = 0
    full = baseline_native.copy(); full[(native > 0) & (baseline_native == 255) & (protected_native == 0)] = 1
    if before != [sha(a) for a in arrays] or graph_before != object_sha([original_packet['graph'],original_packet['heads'],binding]):
        raise ValueError('Joint open-contour branch changed baseline or graph')
    if np.any((native > 0) & (raw_native == 0)): raise ValueError('Joint output is not a raw-open subset')
    return {'raw_native_mask':raw_native,'native_mask':native,'full_roles':full,
        'raw_view_mask':candidate,'guarded_view_mask':guarded['mask'] if guarded else candidate.copy(),
        'raw_open':{k:v for k,v in opened.items() if k not in ('mask','native_mask')},
        'guards':metadata(guarded) if guarded else None,
        'guard_status':'evaluated' if guarded else 'not_exercised_without_candidates',
        'complete':complete,'raw_open_pixels':raw_pixels,'retained_open_pixels':int((native > 0).sum()),
        'dependencies_complete':dependencies_complete,'source_continuation':continuation,
        'source_immutable':True,'baseline_changed_pixels':int(((full != baseline_native) & (baseline_native != 255)).sum()),
        'protected_changed_pixels':int(((full != baseline_native) & (protected_native > 0)).sum()),
        'policy':deepcopy(POLICY),'geometry_modified':False,'production_changed':False}


def guard_proposal(gray,packet,proposed,**kwargs):
    graph,continuation = source_continuation(gray,packet['graph'])
    result = _guard_on_observed_graph(gray,{**packet,'graph':graph},proposed,**kwargs)
    if not continuation['complete']:
        result['mask'][(kwargs['baseline_roles'] == 255) & (kwargs['protected'] == 0)] = 0
        result['complete'] = False; result['reason'] = 'incomplete_source_continuation'
    result['source_continuation'] = continuation
    return result
