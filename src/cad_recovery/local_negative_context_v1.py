"""Complete negative observations for local proposals, without changing Web/CAD.

The positive producer runs here against the original source. Its result is never
replaced by a supplied mask or by the geometry used for negative observations.
"""
from copy import deepcopy
import numpy as np
from .grid_role_revision_v1 import sha
from .local_contour_witness_v1 import analyze as local_positive
from .negative_source_graph_v1 import build
from .open_contour_joint_v5 import guard_proposal, metadata
from .part_admission_v1 import object_sha
from .part_contact_v1 import bind_source

POLICY={'version':'local-negative-context/1.0','positive_producer':'local-contour-witness/1.0',
    'guards_unchanged':True,'legacy_complete_branch_preserved':True,'source_owner_remapped':False,
    'physical_geometry_approved':False,'Web_changed':False,'production_admitted':False}


def check_certificate(packet,graph,certificate):
    """Check the locally computed certificate before feeding the frozen guards."""
    expected={'source_sha256':sha(packet['gray']),'roles_sha256':sha(packet['roles']),
        'legacy_graph_sha256':object_sha(packet['graph']),
        'source_owner_sha256':sha(packet['source_owner']),
        'ambiguous_owner_sha256':sha(packet['ambiguous_owner_mask']),
        'geometry_and_heads_sha256':object_sha([packet['geometry'],packet['heads']]),
        'graph_sha256':object_sha(graph)}
    if any(certificate.get(k)!=v for k,v in expected.items()):
        raise ValueError('Negative context certificate belongs to different observations')
    if certificate['policy']['version']!='negative-source-graph/1.0':
        raise ValueError('Unsupported negative context producer')
    old=packet['graph']['proposals'];new=graph['proposals']
    # Only local direction flags and a measured width may differ. The numerical
    # owner map continues to address exactly the same original source intervals.
    if len(old)!=len(new):raise ValueError('Negative context changed source inventory')
    for a,b in zip(old,new):
        if {k:v for k,v in a.items() if k not in ('linear_context','width_px')}!={k:v for k,v in b.items() if k not in ('linear_context','width_px')}:
            raise ValueError('Negative context changed source intervals')
    if certificate['complete']!= (not graph['search_incomplete']) or not certificate['source_interval_ids_unchanged']:
        raise ValueError('Negative context completeness differs from certificate')


def analyze(packet,*,authority,source_pin,page_context,protected=None):
    gray,roles=packet['gray'],packet['roles']
    protection=np.zeros(gray.shape,bool) if protected is None else protected
    positive=local_positive(packet,authority=authority,protected=protection)
    before=object_sha([packet['graph'],packet['geometry'],packet['heads']])
    snapshots=[sha(a) for a in (gray,roles,packet['source_owner'],packet['ambiguous_owner_mask'],protection)]
    raw=positive['mask'];zero=np.zeros_like(raw);final=zero.copy();legacy=None;guard=None;certificate=None
    branch='not_exercised_without_positive_candidates';complete=positive['complete']
    if positive['complete'] and raw.any():
        binding=bind_source(gray,gray,np.eye(3).tolist(),source_pin,resize_method='CV_AREA')
        kwargs={'original':gray,'page_context':page_context,'binding':binding,'baseline_roles':roles,
            'protected':protection.astype(np.uint8)*255,'authority':authority}
        if 'stroke_px' in packet['graph']:
            legacy=guard_proposal(gray,packet,raw,**kwargs)
        else:
            legacy={'mask':zero.copy(),'complete':False,'reason':'legacy_semantic_context_unavailable','stages':{}}
        if legacy['complete']:
            guard=legacy;branch='unchanged_legacy_complete_context'
        elif page_context is not None and page_context['complete']:
            graph,certificate=build(packet);check_certificate(packet,graph,certificate)
            branch='independent_source_bound_negative_context'
            if certificate['complete']:guard=guard_proposal(gray,{**packet,'graph':graph},raw,**kwargs)
        else:branch='required_page_context_incomplete'
        complete=bool(guard is not None and guard['complete'])
        if complete:final=guard['mask'].copy()
    elif not positive['complete']:branch='local_positive_search_incomplete'
    if np.any((final>0)&(raw==0)):raise ValueError('Negative context created positive pixels')
    if np.any((final>0)&((roles!=255)|protection)):raise ValueError('Negative context changed baseline/protection')
    full=roles.copy();full[final>0]=1
    if before!=object_sha([packet['graph'],packet['geometry'],packet['heads']]) or snapshots!=[sha(a) for a in (gray,roles,packet['source_owner'],packet['ambiguous_owner_mask'],protection)]:
        raise ValueError('Local negative context changed original observations')
    return {'mask':final,'raw_mask':raw,'full_roles':full,'complete':complete,'branch':branch,
        'positive':metadata(positive),'legacy_guard':metadata(legacy) if legacy else None,
        'guard':metadata(guard) if guard else None,'negative_certificate':certificate,
        'raw_positive_pixels':int(np.count_nonzero(raw)),'guarded_positive_pixels':int(np.count_nonzero(final)),
        'legacy_graph_search_incomplete':packet['graph'].get('search_incomplete'),
        'legacy_packet_complete':packet.get('complete'),'source_immutable':True,
        'scope':'local_candidates_on_checked_working_raster','policy':deepcopy(POLICY)}
