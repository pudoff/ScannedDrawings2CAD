"""Correct wrong CV baseline proposals while retaining qualified stage-1.1 roles."""
import numpy as np
from .contour_contact_recovery_v1 import analyze as previous
from .contact_negative_overlay_v1 import analyze as negative_overlay
from .open_contour_joint_v5 import metadata


def analyze(packet,*,authority,source_pin,page_context,protected=None):
    protected=np.zeros_like(packet['gray'],bool) if protected is None else protected
    result=previous(packet,authority=authority,source_pin=source_pin,page_context=page_context,protected=protected)
    veto=None;mask=result['mask'].copy()
    if result['complete']:
        positive={**result['positive'],'mask':result['material_raw_mask']}
        veto=negative_overlay(packet,mask,positive=positive,contacts=result['contacts'],protected=protected)
        if veto['complete']:mask=veto['mask']
        else:mask=result['baseline_mask'].copy()
    full=packet['roles'].copy();full[mask>0]=1
    if np.any((mask>0)&((packet['roles']!=255)|protected)):
        raise ValueError('Negative contact correction changed known/manual roles')
    return {**result,'mask':mask,'full_roles':full,
        'added_mask':np.where((mask>0)&(result['baseline_mask']==0),255,0).astype(np.uint8),
        'withdrawn_mask':np.where((result['baseline_mask']>0)&(mask==0),255,0).astype(np.uint8),
        'guarded_positive_pixels':int(np.count_nonzero(mask)),
        'new_material_pixels':int(np.count_nonzero((mask>0)&(result['baseline_mask']==0))),
        'complete':bool(result['complete'] and (veto is None or veto['complete'])),
        'negative_overlay':metadata(veto) if veto else None,
        'policy':{'version':'contour-contact-recovery/2.0','known_roles_and_manual_preserved':True,
            'qualified_1_1_preservation_measured_separately':True,'wrong_CV_baseline_may_be_withdrawn':True,
            'geometry_modified':False,'source_owner_remapped':False,'Web_changed':False,'physical_geometry_approved':False}}
