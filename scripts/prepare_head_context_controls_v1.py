"""Prospective short/curved-stem and text counterevidence controls."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from cad_recovery.dataset_layout_v1 import pin
from run_topology_robust_v1 import write

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/head-context-controls-v1'
FAMILIES=[('short_straight','train'),('label_adjacent_short','train'),('arc_pair','train'),('row_letters','train'),
          ('tight_stack','validation'),('back_to_back_arcs','validation'),('closed_curve_text','validation'),('serif_stamp','validation')]


def build():
    BASE.mkdir(exist_ok=False);rng=np.random.default_rng(769241);jobs=[]
    for family,split in FAMILIES:
        for width in (3,5):
            for angle in ((13,67) if split=='train' else (29,119)):
                canvas=np.full((512,512),255,np.uint8);arrows=[];chains=[];theta=np.deg2rad(angle)
                rot=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
                shift=rng.integers(-2,3,2)
                def xy(p):return (np.asarray(p,float)-256)@rot.T+256+shift
                def line(p,pen=width,closed=False):cv2.polylines(canvas,[np.rint(p).astype(np.int32)],closed,0,pen)
                def head(tip,direction):
                    tip=np.asarray(tip,float);d=np.asarray(direction,float);n=np.array([-d[1],d[0]])
                    base=tip-(26+width)*d;triangle=[tip,base+(5+width)*n,base-(5+width)*n]
                    cv2.fillConvexPoly(canvas,np.rint(triangle).astype(np.int32),0)
                    arrows.append({'id':'head-'+str(len(arrows)),'tip':xy([tip])[0].tolist(),
                        'direction':(rot@d).tolist(),'triangle':xy(triangle).tolist()})
                if family in ('row_letters','serif_stamp'):
                    font=cv2.FONT_HERSHEY_SIMPLEX if split=='train' else cv2.FONT_HERSHEY_TRIPLEX|cv2.FONT_ITALIC
                    for text,y in [('Pp R A 47',175),('P 89 4,7',285),('RP 147 AP',395)]:
                        cv2.putText(canvas,text,(70,y),font,1.45,0,width,cv2.LINE_8)
                    for y in (85,215,330,440):line([[35,y],[475,y]],2*width+1)
                elif family in ('short_straight','label_adjacent_short'):
                    for x,sign in ((150,-1),(350,1)):
                        tip=np.array([x,230]);base=tip-[sign*(26+width),0]
                        line([base-[sign*(width+4),0],tip]);head(tip,[sign,0])
                    if family=='label_adjacent_short':
                        cv2.putText(canvas,'P 47 R',(130,305),cv2.FONT_HERSHEY_SIMPLEX,1.0,0,width,cv2.LINE_8)
                elif family in ('arc_pair','back_to_back_arcs'):
                    center=np.array([256,256]);radius=112 if family=='arc_pair' else 95
                    start,end=(-1.05,1.05) if family=='arc_pair' else (2.15,4.10)
                    t=np.linspace(start,end,201);line(center+radius*np.c_[np.cos(t),np.sin(t)])
                    for a,sign in ((start,-1),(end,1)):
                        head(center+radius*np.array([np.cos(a),np.sin(a)]),sign*np.array([-np.sin(a),np.cos(a)]))
                    if family=='back_to_back_arcs':cv2.putText(canvas,'R 50',(270,265),cv2.FONT_HERSHEY_COMPLEX,1.0,0,width,cv2.LINE_8)
                elif family=='closed_curve_text':
                    center=np.array([256,256]);radius=65;t=np.linspace(0,2*np.pi,361)
                    line(center+radius*np.c_[np.cos(t),np.sin(t)],closed=True);head(center+[radius,0],[0,-1])
                    cv2.putText(canvas,'P RP 47',(180,405),cv2.FONT_HERSHEY_COMPLEX,1.0,0,width,cv2.LINE_8)
                else:
                    feet=[[145,405],[365,405]];line([[95,355],[95,405],[430,405],[430,355]],2*width+3)
                    for x in (145,365):line([[x,110],[x,405]])
                    for y in (140,230,320):
                        line([[145,y],[365,y]]);ids=[]
                        for x,sign in ((145,-1),(365,1)):head([x,y],[sign,0]);ids.append(arrows[-1]['id'])
                        chains.append({'id':'dimension-'+str(len(chains)),'head_ids':ids,'shaft':xy([[195,y],[315,y]]).tolist(),
                            'feet_px':xy(feet).tolist(),'head_scope':[0,0,512,512],'topology':'two_extensions',
                            'reference_kinds':['contour','contour'],'whole_part_contour_visible':False})
                matrix=np.c_[rot,256+shift-rot@np.array([256,256])]
                gray=cv2.warpAffine(canvas,matrix,(512,512),flags=cv2.INTER_NEAREST,borderValue=255)
                identity=f'head-context-{family}-w{width}-r{angle}';folder=BASE/identity;folder.mkdir();views=[]
                for mode,scale in [('native_crop',1.),('half_resolution_crop',.5)]:
                    pixels=gray if scale==1 else cv2.resize(gray,(256,256),interpolation=cv2.INTER_AREA)
                    path=folder/(mode+'.png');Image.fromarray(pixels).save(path)
                    views.append({'mode':mode,'gray':pin(ROOT,path.relative_to(ROOT).as_posix()),
                        'input_to_native':[[1/scale,0,.5/scale-.5],[0,1/scale,.5/scale-.5],[0,0,1]]})
                jobs.append({'case':{'id':identity,'scope':'head_context_analytical','group_id':'head-context-'+family,'split':split,
                    'size_px':[512,512],'labels':{'family':family,'arrows':arrows,'chains':chains,
                        'expected_dimension_chains':len(chains)}},'views':views,'new_control':True})
    write(BASE/'contract.json',{'task':'P10.2c.3b.2g.1','seed':769241,'generator':pin(ROOT,'scripts/prepare_head_context_controls_v1.py'),
        'jobs':jobs,'families':FAMILIES,'prepared_before_algorithm':True,'previous_validation_is_development':True,
        'criteria':{'minimum_fresh_TP_retention':.95,'minimum_fresh_correct_heads':8,'no_new_false_heads':True,
            'preserve_all_11_previous_fresh_head_losses':True,'minimum_real_false_heads_removed':3,
            'all_real_correct_heads_and_chains_preserved':True,'no_new_wrong_chains_or_lost_correct_chains':True,
            'raw_heads_source_roles_and_prefix_exact':True,'complete_bounded_repeats':True},
        'budgets':{'child_seconds':60,'RSS_MiB':1024,'output_MiB':64,'files':128,'total_child_seconds':1800},
        'production_changed':False,'physical_geometry_approved':False})
    print(pin(ROOT,'artifacts/head-context-controls-v1/contract.json'),flush=True)


if __name__=='__main__':build()
