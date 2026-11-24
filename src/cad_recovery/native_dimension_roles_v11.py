"""Protect observed OCR ink in additional source-body publications."""
import hashlib

import cv2
import numpy as np

from . import joint_dimension_roles_v6 as joint
from . import native_dimension_roles_v10 as previous
from .source_arrow_observation_v1 import transform
from unittest.mock import patch


def protect(roles,box,ocr,matrix):
    mask=np.zeros_like(roles,dtype='uint8')
    origin=np.asarray(box[:2])
    for line in ocr.get('lines',[]):
        polygon=transform(line['polygon_px'],matrix)-origin
        cv2.fillPoly(mask,[np.rint(polygon).astype('int32')],1)
    result=roles.copy();result[mask>0]=0
    return result,int(np.count_nonzero((roles>0)&(mask>0)))


def propose(*args,**kwargs):
    result=previous.propose(*args,**kwargs)
    for row in result['report'].get('source_short_linear_links',[]):
        if row['id'] not in result['short_linear_roles']:continue
        before=result['short_linear_roles'][row['id']]
        roles,count=protect(before,row['source_box_px'],kwargs['ocr'],np.asarray(kwargs['working_to_source']))
        result['short_linear_roles'][row['id']]=roles
        row.update(ocr_protected_source_body_pixels=count,ocr_reading_used_for_geometry=False,
            source_roles_sha256=hashlib.sha256(roles.tobytes()).hexdigest(),
            body_pixels={str(r):int(np.count_nonzero(roles==r)) for r in [2,3]})
        if not all(row['body_pixels'][str(r)] for r in [2,3]):
            row.update(status='withheld',reasons=['OCR_protection_removed_required_body'])
            del result['short_linear_roles'][row['id']]
    result['report']['source_short_linear_admitted']=sum(
        r['status']=='source_short_linear_bodies_admitted'
        for r in result['report'].get('source_short_linear_links',[]))
    result['report']['source_body_OCR_protection_checked']=True
    return result


def joint_propose(*args,**kwargs):
    with patch.object(joint,'native_propose',propose):result=joint.propose(*args,**kwargs)
    result['report']['schema_version']='joint-dimension-roles/14.0'
    return result
