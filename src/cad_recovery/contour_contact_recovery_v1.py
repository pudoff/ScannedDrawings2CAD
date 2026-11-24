"""Source-core corner witnesses composed with every accepted stage-1.1 pixel."""
from copy import deepcopy
import numpy as np
from .material_contours_v2 import analyze as baseline
from .local_shared_contacts_v3 import analyze as additional
from .grid_role_revision_v1 import sha
from .part_admission_v1 import object_sha
from .open_contour_joint_v5 import metadata

POLICY={'version':'contour-contact-recovery/1.0','baseline_1_1_preserved':True,
        'new_core_probes_are_witness_only':True,'source_owner_remapped':False,
        'frame_grid_and_direct_negative_veto_preserved':True,'Web_changed':False,
        'physical_geometry_approved':False,'production_admitted':False}


def analyze(packet, *, authority, source_pin, page_context, protected=None):
    protection=np.zeros_like(packet['gray'],bool) if protected is None else protected
    arrays=(packet['gray'],packet['roles'],packet['source_owner'],packet['ambiguous_owner_mask'],protection)
    snapshots=[sha(a) for a in arrays];before=object_sha([packet['geometry'],packet['graph'],packet['heads']])
    kw=dict(authority=authority,source_pin=source_pin,page_context=page_context,protected=protection)
    old=baseline(packet,**kw);extra=additional(packet,**kw)
    mask=old['mask'].copy()
    if extra['complete']:mask|=extra['mask']
    if np.any((old['mask']>0)&(mask==0)) or np.any((mask>0)&((packet['roles']!=255)|protection)):
        raise ValueError('Contact recovery changed baseline/protection')
    if before!=object_sha([packet['geometry'],packet['graph'],packet['heads']]) or snapshots!=[sha(a) for a in arrays]:
        raise ValueError('Contact recovery changed source observations')
    full=packet['roles'].copy();full[mask>0]=1
    return {'mask':mask,'raw_mask':old['raw_mask']|extra['raw_mask'],'full_roles':full,
        'added_mask':np.where((mask>0)&(old['mask']==0),255,0).astype(np.uint8),
        'baseline_mask':old['mask'],'material_raw_mask':extra['raw_mask'],
        'complete':bool(old['complete'] and extra['complete']),
        'baseline':metadata(old),'positive':extra['positive'],'contacts':extra['contacts'],'guard':extra['guard'],
        'raw_positive_pixels':int(np.count_nonzero(old['raw_mask']|extra['raw_mask'])),
        'guarded_positive_pixels':int(np.count_nonzero(mask)),
        'new_material_pixels':int(np.count_nonzero((mask>0)&(old['mask']==0))),
        'source_immutable':True,'policy':deepcopy(POLICY)}
