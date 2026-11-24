"""Publish independent straight-anchor source roles beside preserved ARC roles."""
from unittest.mock import patch

import numpy as np

from . import joint_dimension_roles_v6 as joint
from . import native_dimension_roles_v8 as previous
from .native_short_linear_v1 import observe_context,refined_head_domain
from .source_arrow_observation_v1 import source_windows


def propose(original_source,working_image,*,geometry,ocr,roles,source_heads,
            working_to_source,arrows,authority=None,authority_root=None,protected=None):
    result=previous.propose(original_source,working_image,geometry=geometry,ocr=ocr,
        roles=roles,source_heads=source_heads,working_to_source=working_to_source,
        arrows=arrows,authority=authority,authority_root=authority_root,protected=protected)
    context=result['chain_result']['admission']['context']
    heads=refined_head_domain(context,{h['id']:h for h in source_heads['verified_candidates']})
    result['short_linear_roles']={};records=[]
    # Missing manual protection provenance withdraws the additional branch.
    if protected is not None and np.any(protected):
        result['report']['short_linear_hold']='reviewed_protection_requires_source_registration'
        return result
    for c in context['contexts']:
        if 'source_box_px' not in c or not c.get('stroke_observation',{}).get('complete'):continue
        if len(records)>=32:raise ValueError('Short linear context budget')
        gray=source_windows(original_source,result['report']['source_sha256'],
            source_heads['processed_size_px'],[c['source_box_px']])[0]
        value=observe_context(gray,c,heads,geometry,roles,np.asarray(working_to_source),
            source=original_source,source_hash=result['report']['source_sha256'],
            size=source_heads['processed_size_px'])
        records.append(value['report'])
        if value['report']['status']=='source_short_linear_bodies_admitted':
            result['short_linear_roles'][c['id']]=value['roles']
    result['report'].update(source_short_linear_links=records,
        source_short_linear_admitted=sum(r['status']=='source_short_linear_bodies_admitted' for r in records),
        source_short_linear_working_roles_unchanged=True)
    return result


def joint_propose(*args,**kwargs):
    with patch.object(joint,'native_propose',propose):result=joint.propose(*args,**kwargs)
    result['report']['schema_version']='joint-dimension-roles/12.0'
    return result
