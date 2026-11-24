"""Freeze fresh frame/hatch/shaft families before context/axis implementation."""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from cad_recovery.dataset_layout_v1 import pin

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'artifacts/local-head-context-controls-v1'
FAMILIES = [('frame_side_pointer', 'train'), ('bordered_hatch_pointer', 'train'),
    ('continuous_hatch_stain', 'train'), ('framed_glyph_negative', 'train'),
    ('subpixel_short_pointer', 'train'), ('double_frame_pointer', 'validation'),
    ('broken_hatch_boundary_pointer', 'validation'), ('interrupted_hatch_stain', 'validation'),
    ('caption_on_frame_negative', 'validation'), ('attached_short_pointer', 'validation')]


def scenes():
    rng = np.random.default_rng(614289)
    for family, split in FAMILIES:
        for width in (2, 4, 6):
            for angle in (0, 41, 90, 131):
                size = 480; gray = np.full((size,size),255,np.uint8); refs = []
                theta = np.deg2rad(angle)
                rotation = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
                shift = rng.integers(-5,6,2)
                def transform(p): return (np.asarray(p,float)-240)@rotation.T+240+shift
                def line(points,w=width):
                    cv2.polylines(gray,[np.rint(transform(points)).astype(np.int32)],False,0,w)
                def rect(a,b,w=width):
                    x,y=a; X,Y=b; line([[x,y],[X,y],[X,Y],[x,Y],[x,y]],w)
                def arrow(tip,direction,tail):
                    tip=np.asarray(tip,float); u=np.asarray(direction,float); n=np.array([-u[1],u[0]])
                    base=tip-(23+2*width)*u
                    polygon=transform([tip,base+(5+width)*n,base-(5+width)*n])
                    cv2.fillConvexPoly(gray,np.rint(polygon).astype(np.int32),0)
                    line([base,base-tail*u])
                    refs.append({'id':'head','tip':transform([tip])[0].tolist(),
                                 'direction':(rotation@u).tolist(),'triangle':polygon.tolist()})
                positive = family in ('frame_side_pointer','bordered_hatch_pointer','double_frame_pointer',
                                      'broken_hatch_boundary_pointer')
                if positive:
                    rect([149,151],[343,337],2*width+1)
                    if family in ('frame_side_pointer','double_frame_pointer'):
                        rect([47,66],[402,401],width)
                    if family=='double_frame_pointer': rect([64,82],[386,385],width)
                    if 'hatch' in family:
                        for j in range(7):
                            y=177+j*22
                            if family=='broken_hatch_boundary_pointer':
                                line([[163,y],[224,y-10]]);line([[241,y-12],[323,y-26]])
                            else: line([[163,y],[323,y-26]])
                    arrow([149,238],[1,0],56)
                elif family in ('continuous_hatch_stain','interrupted_hatch_stain'):
                    rect([133,127],[354,353],2*width+1)
                    for j in range(7):
                        x=155+j*25
                        if family=='interrupted_hatch_stain':
                            line([[x,147],[x+12,219]]);line([[x+15,236],[x+30,329]])
                        else: line([[x,147],[x+30,329]])
                    polygon=transform([[252,265],[230,222],[255,216]])
                    cv2.fillConvexPoly(gray,np.rint(polygon).astype(np.int32),0)
                elif family in ('framed_glyph_negative','caption_on_frame_negative'):
                    canvas=np.full_like(gray,255)
                    for j,text in enumerate(('A4 R24','M12 32','EDGE 8')):
                        cv2.putText(canvas,text,(151,180+j*54),cv2.FONT_HERSHEY_SIMPLEX,.8,0,width)
                    affine=np.c_[rotation,np.array([240,240])+shift-rotation@np.array([240,240])]
                    gray=cv2.warpAffine(canvas,affine,(size,size),flags=cv2.INTER_NEAREST,borderValue=255)
                    rect([128,139],[353,324],width)
                    if family=='caption_on_frame_negative':line([[128,238],[353,238]])
                else:
                    # Directly rasterized quarter turns and odd/even coordinates
                    # exercise axis proposals without deriving them from labels.
                    tip=[190+(width%3),225+(width%2)]
                    arrow(tip,[-1,0],max(15,3*width))
                    if family=='attached_short_pointer':line([[tip[0],tip[1]-38],[tip[0],tip[1]+63]])
                yield {'case':{'id':f'head-context-{family}-w{width}-r{angle}',
                    'scope':'local_head_context_control','group_id':'head-context-'+family,'split':split,
                    'size_px':[size,size],'labels':{'family':family,'width_px':width,'rotation_deg':angle,
                        'arrows':refs,'negative_regions':[],'dense_heads':True,
                        'authority':'fresh analytic controls; no physical/CAD authority'}},'gray':gray}


if __name__=='__main__':
    BASE.mkdir(exist_ok=False);jobs=[]
    for item in scenes():
        case=item['case']; folder=BASE/case['id'];folder.mkdir();views=[]
        for mode,scale in [('native_crop',1.),('half_resolution_crop',.5)]:
            gray=item['gray'] if scale==1 else cv2.resize(item['gray'],(240,240),interpolation=cv2.INTER_AREA)
            path=folder/(mode+'.png');Image.fromarray(gray).save(path)
            views.append({'mode':mode,'gray':pin(ROOT,path.relative_to(ROOT).as_posix()),
                'input_to_native':[[1/scale,0,.5/scale-.5],[0,1/scale,.5/scale-.5],[0,0,1]]})
        jobs.append({'case':case,'views':views,'new_control':True})
    value={'task':'P10.2c.2b.7c.2a','generator':pin(ROOT,'scripts/prepare_local_head_context_controls_v1.py'),
        'prepared_before_detector_implementation':True,'seed':614289,'families':FAMILIES,'jobs':jobs,
        'gates':{'lost_previous_true_targets_max':0,'additional_dense_FP_against_v1_max':0,
            'additional_negative_ROI_FP_max':0,'new_validation_recall_min_each_scope_scale':.90,
            'all_searches_complete':True,'profile_samples_max_per_source':8000000,
            'morphology_pixels_max_per_source':40000000,'source_and_baseline_unchanged':True,
            'historical_20_of_21_and_13_of_13_preserved':True},
        'training':False,'closed_test_used':False,'physical_geometry_approved':False}
    with (BASE/'contract.json').open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps(pin(ROOT,(BASE/'contract.json').relative_to(ROOT).as_posix())))
