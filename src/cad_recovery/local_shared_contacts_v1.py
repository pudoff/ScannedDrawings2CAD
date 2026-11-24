"""Source-bound corner topology and transverse contacts before frozen page guards."""
from copy import deepcopy
import numpy as np
from .local_contour_witness_v1 import analyze as local_positive
from .contour_contact_witness_v1 import analyze as contact_witness
from .contour_role_contacts_v1 import guard as role_guard
from .negative_source_graph_v1 import build
from .local_negative_context_v1 import check_certificate
from .open_source_continuation_v1 import observe
from .open_contour_joint_v5 import metadata,negative_subset
from .competitive_grid_admission_v2 import analyze as role_context
from .contour_page_competition_v1 import analyze as frame_guard
from .attached_grid_competition_v1 import analyze as grid_guard
from .part_contact_v1 import bind_source
from .part_admission_v1 import object_sha
from .grid_role_revision_v1 import sha

POLICY={'version':'local-shared-contacts/1.0','frame_and_grid_guards_unchanged':True,
    'source_owner_remapped':False,'Web_changed':False,'production_admitted':False,'physical_geometry_approved':False}


def guarded(packet,mask,*,authority,source_pin,page_context,protected,positive,contacts):
    gray=packet['gray'];certificate=None;stages={};graph=packet['graph'];output=np.zeros_like(mask)
    if page_context is None or not page_context['complete']:
        return {'mask':output,'complete':False,'reason':'page_context_unavailable_or_incomplete','stages':{}}
    if page_context['source_sha256']!=sha(gray) or page_context.get('GT_used') is not False:
        raise ValueError('Contact page belongs to another source or GT')
    if graph.get('search_incomplete',True) or 'stroke_px' not in graph:
        graph,certificate=build(packet);check_certificate(packet,graph,certificate)
    graph,continuation=observe(gray,graph)
    context=role_context(gray,graph,packet['roles'],packet['heads'],packet['source_owner'],packet['ambiguous_owner_mask'],authority=authority)
    negative={k:v for k,v in context.items() if k not in ('previous','old_candidate','old_owners','candidates','owners')}
    binding=bind_source(gray,gray,np.eye(3).tolist(),source_pin,resize_method='CV_AREA')
    result=role_guard(gray,graph,packet['source_owner'],packet['ambiguous_owner_mask'],mask,negative,positive=positive,contacts=contacts)
    stages['role']=metadata(result);candidate=negative_subset(mask,result)
    frame=frame_guard(gray,graph,packet['source_owner'],packet['ambiguous_owner_mask'],candidate,
        native_gray=gray,page=page_context['page'],detection=page_context['detection'],source_mask=page_context['frame_mask'],
        input_to_native=binding['matrix'],resize_method=binding['resize_method'])
    stages['frame']=metadata(frame);candidate=negative_subset(candidate,frame)
    grid=grid_guard(gray,graph,packet['source_owner'],packet['ambiguous_owner_mask'],candidate,
        source_gray=gray,native_gray=gray,page=page_context['page'],observations=page_context['observations'],source_masks=page_context['grid_masks'],
        input_to_native=binding['matrix'],resize_method=binding['resize_method'],baseline_roles=packet['roles'],protected=protected.astype(np.uint8)*255)
    stages['grid']=metadata(grid);candidate=negative_subset(candidate,grid)
    complete=bool(continuation['complete'] and context['complete'] and contacts['complete'] and
        all(s['complete'] for s in stages.values()) and (certificate is None or certificate['complete']))
    if complete:output=candidate
    return {'mask':output,'complete':complete,'reason':None if complete else 'incomplete_required_contact_context',
        'stages':stages,'negative_certificate':certificate,'source_continuation':continuation,'role_context_sha256':object_sha(negative)}


def analyze(packet,*,authority,source_pin,page_context,protected=None):
    gray=packet['gray'];protection=np.zeros(gray.shape,bool) if protected is None else protected
    arrays=(gray,packet['roles'],packet['source_owner'],packet['ambiguous_owner_mask'],protection)
    snapshots=[sha(a) for a in arrays];before=object_sha([packet['graph'],packet['geometry'],packet['heads']])
    positive=local_positive(packet,authority=authority,protected=protection);contacts=None;guard=None;final=np.zeros_like(gray)
    complete=positive['complete']
    if positive['mask'].any():
        contacts=contact_witness(packet,positive,protected=protection)
        if contacts['complete']:
            guard=guarded(packet,contacts['mask'],authority=authority,source_pin=source_pin,page_context=page_context,
                protected=protection,positive=positive,contacts=contacts)
            complete=bool(positive['complete'] and guard['complete'])
            if complete:final=guard['mask']
        else:complete=False
    if np.any((final>0)&((positive['mask']==0)|(packet['roles']!=255)|protection)):
        raise ValueError('Shared contacts created unbound pixels or changed baseline/protection')
    if snapshots!=[sha(a) for a in arrays] or before!=object_sha([packet['graph'],packet['geometry'],packet['heads']]):
        raise ValueError('Shared contact composition changed source observations')
    full=packet['roles'].copy();full[final>0]=1
    return {'mask':final,'raw_mask':positive['mask'],'contact_mask':contacts['mask'] if contacts else np.zeros_like(gray),
        'full_roles':full,'complete':complete,'positive':metadata(positive),'contacts':metadata(contacts) if contacts else None,
        'guard':metadata(guard) if guard else None,'raw_positive_pixels':positive['raw_positive_pixels'],
        'guarded_positive_pixels':int(np.count_nonzero(final)),'source_immutable':True,
        'legacy_graph_search_incomplete':packet['graph'].get('search_incomplete'),
        'legacy_packet_complete':packet.get('complete'),'policy':deepcopy(POLICY)}
