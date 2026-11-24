"""Fresh independent families for own-shaft exclusion and pre-taper axis proposals."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from cad_recovery.dataset_layout_v1 import pin

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/local-head-context-controls-v2'
FAMILIES=[('double_rail_leader','train'),('own_stem_parallel_stain','train'),
          ('stepped_cap_chain','validation'),('broken_multistripe_stain','validation')]

if __name__=='__main__':
    BASE.mkdir(exist_ok=False);jobs=[];rng=np.random.default_rng(703951)
    for family,split in FAMILIES:
        for width in (2,4,6):
            for angle in (0,29,90,151):
                gray=np.full((480,480),255,np.uint8);refs=[];theta=np.deg2rad(angle)
                rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
                shift=rng.integers(-5,6,2)
                def transform(p):return (np.asarray(p,float)-240)@rotation.T+240+shift
                def line(p,w=width):cv2.polylines(gray,[np.rint(transform(p)).astype(np.int32)],False,0,w)
                def arrow(tip,u,tail):
                    tip=np.asarray(tip,float);u=np.asarray(u,float);n=np.array([-u[1],u[0]])
                    base=tip-(25+2*width)*u;tri=transform([tip,base+(5+width)*n,base-(5+width)*n])
                    cv2.fillConvexPoly(gray,np.rint(tri).astype(np.int32),0);line([base,base-tail*u])
                    refs.append({'id':'head','tip':transform([tip])[0].tolist(),'direction':(rotation@u).tolist(),'triangle':tri.tolist()})
                if family=='double_rail_leader':
                    line([[207,158],[376,158],[376,326],[207,326],[207,158]],2*width+1)
                    arrow([207,242],[1,0],91)
                elif family=='stepped_cap_chain':
                    line([[117,135],[268,135],[268,326],[204,326],[204,288],[117,288],[117,135]],2*width+1)
                    line([[268,135],[367,135]]);line([[268,326],[367,326]])
                    arrow([338,135],[0,-1],119)
                else:
                    for j in range(7):
                        y=155+j*27
                        if family=='broken_multistripe_stain':
                            line([[121,y],[213,y-16]]);line([[229,y-19],[365,y-42]])
                        else:line([[121,y],[365,y-42]])
                    tri=transform([[219,229],[255,209],[262,231]])
                    cv2.fillConvexPoly(gray,np.rint(tri).astype(np.int32),0)
                case={'id':f'head-context2-{family}-w{width}-r{angle}','scope':'local_head_context_control_v2',
                    'group_id':'head-context2-'+family,'split':split,'size_px':[480,480],
                    'labels':{'family':family,'width_px':width,'rotation_deg':angle,'arrows':refs,
                              'negative_regions':[],'dense_heads':True,'authority':'fresh procedural second-phase controls'}}
                folder=BASE/case['id'];folder.mkdir();views=[]
                for mode,scale in [('native_crop',1.),('half_resolution_crop',.5)]:
                    image=gray if scale==1 else cv2.resize(gray,(240,240),interpolation=cv2.INTER_AREA)
                    path=folder/(mode+'.png');Image.fromarray(image).save(path)
                    views.append({'mode':mode,'gray':pin(ROOT,path.relative_to(ROOT).as_posix()),
                        'input_to_native':[[1/scale,0,.5/scale-.5],[0,1/scale,.5/scale-.5],[0,0,1]]})
                jobs.append({'case':case,'views':views,'new_control':True})
    gates={'lost_previous_true_targets_max':0,'additional_dense_FP_against_v1_max':0,
           'additional_negative_ROI_FP_max':0,'new_validation_recall_min_each_scope_scale':.90,
           'all_searches_complete':True,'profile_samples_max_per_source':8000000,'morphology_pixels_max_per_source':40000000}
    value={'task':'P10.2c.2b.7c.2a','generator':pin(ROOT,'scripts/prepare_local_head_context_controls_v2.py'),
           'prepared_before_detector_implementation':True,'seed':703951,'families':FAMILIES,'jobs':jobs,'gates':gates}
    with (BASE/'contract.json').open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(pin(ROOT,(BASE/'contract.json').relative_to(ROOT).as_posix())))
