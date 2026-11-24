"""Whole-source-pixel support for contour proposals; v1 remains immutable."""
import math
import cv2
import numpy as np
from .part_contact_v1 import analyze as previous,bind_source

ARMS=('no_part','canonical','cell_core','boundary_core','owner_boundary')
POLICY={'version':'part-contact/2.0','maximum_native_pixels':4194304,
    'cell_margin':'input_pixel_half_diagonal + source_pixel_half_diagonal',
    'boundary_margin':'input_pixel_half_diagonal + 2*source_pixel_half_diagonal',
    'owner_boundary':'boundary margin plus v1 quantization-stable source owner',
    'output':'native_source_pixel_centres','GT_used':False,'native_prediction_transfer':False,
    'physical_geometry_approved':False,'production_admitted':False}


def analyze(*args,**kwargs):
    old=previous(*args,**kwargs)
    source=kwargs['source_gray']
    r_input=old['input_footprint_half_diagonal_native_px'];r_native=math.sqrt(.5)
    canonical=old['canonical'];baseline=old['no_part'];new=(canonical==1)&(baseline!=1)
    maps={'no_part':baseline.copy(),'canonical':canonical.copy(),
          **{a:baseline.copy() for a in ARMS[2:]}}
    if new.any():
        ink=cv2.threshold(source,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]
        distance=cv2.distanceTransform(ink,cv2.DIST_L2,5)
        supported=new&(old['source_core']==1)
        maps['cell_core'][supported&(distance>r_input+r_native)]=1
        band=supported&(distance>r_input+2*r_native)
        maps['boundary_core'][band]=1
        maps['owner_boundary'][band&(old['owner_core']==1)]=1
    uncertainty={a:np.where(new&(maps[a]!=1),255,0).astype(np.uint8) for a in ARMS[2:]}
    return {**{k:v for k,v in old.items() if k not in ('no_part','canonical','source_core','owner_core','uncertainty','arm_complete')},
        **maps,'uncertainty':uncertainty,
        'arm_complete':{'cell_core':old['arm_complete']['source_core'],
            'boundary_core':old['arm_complete']['source_core'],'owner_boundary':old['arm_complete']['owner_core']},
        'source_pixel_half_diagonal_px':r_native,
        'margins_native_px':{'cell_core':r_input+r_native,'boundary_core':r_input+2*r_native,
                            'owner_boundary':r_input+2*r_native}}
