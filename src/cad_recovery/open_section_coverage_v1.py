"""Add source-resolution open-section evidence to the frozen joint baseline."""
from copy import deepcopy
import numpy as np
from . import open_contour_joint_v3 as previous
from .open_section_source_linear_v1 import observe
from .open_section_source_walk_v1 import checked_original
from .part_contact_v1 import bind_source
from .grid_role_revision_v1 import sha
from .part_admission_v1 import object_sha

POLICY={**previous.POLICY,'version':'open-section-coverage/1.0',
        'additional_source_evidence':'certified_source_linear_full_original',
        'old_native_assignments_preserved':True,'unhatched_contour_supported':False}
guard_proposal=previous.guard_proposal


def analyze(gray,packet,*,original,binding,baseline_native,protected_native,
            page_context=None,producer_pin,source_packet=None):
    result=previous.analyze(gray,packet,original=original,binding=binding,
        baseline_native=baseline_native,protected_native=protected_native,
        page_context=page_context,producer_pin=producer_pin)
    result['policy']=deepcopy(POLICY)
    if original is None:
        result['source_extension']={'status':'full_original_unavailable'};return result
    checked_original(gray,original,binding)
    if source_packet is None:
        if gray.shape==original.shape and np.array_equal(gray,original):source_packet=packet
        else:
            result['source_extension']={'status':'source_bundle_unavailable'};return result
    implicit_native=source_packet is packet and gray.shape==original.shape and np.array_equal(gray,original)
    if ((not implicit_native and source_packet.get('input_sha256')!=sha(original))
            or source_packet['roles'].shape!=original.shape):raise ValueError('Aux source bundle belongs to another original')
    if not source_packet.get('complete',True):
        result['source_extension']={'status':'incomplete_source_bundle'};return result
    immutable=object_sha([source_packet['graph'],source_packet['heads']])
    arrays=[sha(source_packet[k]) for k in ('roles','source_owner','ambiguous_owner_mask')]
    graph,evidence=observe(original,source_packet['graph'])
    native_binding=bind_source(original,original,np.eye(3).tolist(),binding.get('source_pin',{}),resize_method='CV_AREA')
    extra=previous.analyze(original,{**source_packet,'graph':graph},original=original,binding=native_binding,
        baseline_native=baseline_native,protected_native=protected_native,page_context=page_context,
        producer_pin=source_packet.get('files',{}).get('roles',producer_pin))
    result['source_extension']={'status':'evaluated','source_linear':evidence,
        'raw_open':extra['raw_open'],'guards':extra['guards'],'complete':extra['complete'],
        'source_domain_sha256':sha(original),'native_before_guard':extra['raw_open_pixels'],
        'native_after_guard':extra['retained_open_pixels'],'geometry_modified':False}
    result['raw_native_mask']|=extra['raw_native_mask']
    if evidence['complete'] and extra['complete'] and packet.get('complete',True):
        result['native_mask']|=extra['native_mask']
    result['full_roles']=baseline_native.copy()
    free=(baseline_native==255)&(protected_native==0)
    result['full_roles'][(result['native_mask']>0)&free]=1
    result['raw_open_pixels']=int((result['raw_native_mask']>0).sum())
    result['retained_open_pixels']=int((result['native_mask']>0).sum())
    result['complete']=bool(result['complete'] and evidence['complete'] and extra['complete'])
    if result['raw_open_pixels']:result['guard_status']='evaluated'
    if (immutable!=object_sha([source_packet['graph'],source_packet['heads']])
            or arrays!=[sha(source_packet[k]) for k in ('roles','source_owner','ambiguous_owner_mask')]):
        raise ValueError('Source-resolution branch changed original observations')
    return result
