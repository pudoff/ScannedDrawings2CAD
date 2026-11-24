"""Freeze text/arrow workloads before implementing exact lazy profile queries."""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from cad_recovery.dataset_layout_v1 import pin

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/streamed-head-controls-v1'
FAMILIES=[('label_band','train'),('glyph_grid','train'),
          ('dimension_caption','validation'),('stamp_and_pointer','validation')]


def scenes():
    rng=np.random.default_rng(420683)
    for family,split in FAMILIES:
        for width in (2,4,6):
            for angle in (0,31,90):
                image=np.full((480,480),255,np.uint8);refs=[];theta=np.deg2rad(angle)
                rot=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
                shift=rng.integers(-5,6,2)
                def transform(points):return (np.asarray(points,float)-240)@rot.T+240+shift
                canvas=np.full_like(image,255)
                for j,text in enumerate(('A3 R12 M8','24 16 D7','EDGE R32')):
                    cv2.putText(canvas,text,(95,139+j*56),cv2.FONT_HERSHEY_SIMPLEX,.8,0,width)
                affine=np.c_[rot,np.array([240,240])+shift-rot@np.array([240,240])]
                image=cv2.warpAffine(canvas,affine,(480,480),flags=cv2.INTER_NEAREST,borderValue=255)
                if family!='glyph_grid':
                    tip=np.array([134.,337.]);unit=np.array([-1.,0.]);normal=np.array([0.,1.]);base=tip+(26+width)*np.array([1.,0.])
                    triangle=transform([tip,base+(5+width)*normal,base-(5+width)*normal])
                    cv2.fillConvexPoly(image,np.rint(triangle).astype(np.int32),0)
                    cv2.polylines(image,[np.rint(transform([base,base+[74,0]])).astype(np.int32)],False,0,width)
                    refs.append({'id':'head','tip':transform([tip])[0].tolist(),'direction':(rot@unit).tolist(),'triangle':triangle.tolist()})
                if family in ('dimension_caption','stamp_and_pointer'):
                    cv2.polylines(image,[np.rint(transform([[85,81],[383,81],[383,285],[85,285],[85,81]])).astype(np.int32)],False,0,width)
                yield {'id':f'stream-head-{family}-w{width}-r{angle}','scope':'streamed_head_control',
                    'group_id':'stream-head-'+family,'split':split,'size_px':[480,480],
                    'labels':{'family':family,'width_px':width,'rotation_deg':angle,'arrows':refs,
                              'negative_regions':[],'dense_heads':True}},image
    for k,split in enumerate(('train','train','validation','validation')):
        image=np.full((1000,1600),255,np.uint8);refs=[]
        for row in range(7+k):
            for col in range(9):
                cv2.putText(image,f'A{row} R{13+col}',(25+col*174,55+row*73),
                    cv2.FONT_HERSHEY_SIMPLEX,.55,0,2)
        for j in range(3):
            tip=np.array([217+j*509+k*7,837+(j%2)*71],float);base=tip+[32,0]
            tri=np.array([tip,base+[0,10],base-[0,10]])
            cv2.fillConvexPoly(image,np.rint(tri).astype(np.int32),0)
            cv2.line(image,tuple(base.astype(int)),tuple((base+[91,0]).astype(int)),0,3)
            refs.append({'id':f'head-{j}','tip':tip.tolist(),'direction':[-1.,0.],'triangle':tri.tolist()})
        yield {'id':f'stream-head-text-workload-{k}','scope':'streamed_head_large_control',
            'group_id':f'stream-head-text-workload-{k}','split':split,'size_px':[1600,1000],
            'labels':{'family':f'text_workload_{k}','width_px':3,'rotation_deg':None,
                      'arrows':refs,'negative_regions':[],'dense_heads':True}},image


if __name__=='__main__':
    BASE.mkdir(exist_ok=False);jobs=[]
    for case,gray in scenes():
        folder=BASE/case['id'];folder.mkdir();views=[]
        for mode,scale in [('native_crop',1.),('half_resolution_crop',.5)]:
            image=gray if scale==1 else cv2.resize(gray,tuple(v//2 for v in case['size_px']),interpolation=cv2.INTER_AREA)
            path=folder/(mode+'.png');Image.fromarray(image).save(path)
            views.append({'mode':mode,'gray':pin(ROOT,path.relative_to(ROOT).as_posix()),
                'input_to_native':[[1/scale,0,.5/scale-.5],[0,1/scale,.5/scale-.5],[0,0,1]]})
        jobs.append({'case':case,'views':views,'new_control':True})
    value={'task':'P10.2c.2b.7c.2a.2','generator':pin(ROOT,'scripts/prepare_streamed_head_controls_v1.py'),
        'prepared_before_profile_implementation':True,'seed':420683,'families':FAMILIES,'jobs':jobs,
        'gates':{'lost_previous_true_targets_max':0,'additional_dense_FP_against_previous_max':0,
            'all_source_searches_complete':True,'no_new_ownership_incomplete':True,
            'exact_heads_proposals_context_on_complete_pairs':True,'raw_regions_fully_accounted':True,
            'maximum_profile_samples_per_source':8000000,'maximum_morphology_pixels_per_source':40000000,
            'old_interval':[72,72],'old_width4_half':[12,12],'historical_counts':[[20,21],[13,13]]},
        'training':False,'closed_test_used':False,'production_changed':False,'physical_geometry_approved':False}
    path=BASE/'contract.json'
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps(pin(ROOT,path.relative_to(ROOT).as_posix())))
