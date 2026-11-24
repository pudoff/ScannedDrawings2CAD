"""Fresh construction-level role/head controls before shadow-profile changes."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from cad_recovery.dataset_layout_v1 import pin
from run_topology_robust_v1 import write

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/joint-cv-controls-v1'
FAMILIES=[('short_inside_span','train'),('ellipse_top_body','train'),('two_near_dimensions','train'),('stamp_rules','train'),
    ('outside_short_span','validation'),('angled_wall_short','validation'),('rounded_body_leader','validation'),('parallel_note_rows','validation')]


def build():
    BASE.mkdir(exist_ok=False);rng=np.random.default_rng(837151);jobs=[]
    for family,split in FAMILIES:
        for width in (3,5):
            for angle in ((17,61) if split=='train' else (31,127)):
                gray=np.full((512,512),255,np.uint8);truth=np.full_like(gray,255);negative=np.zeros_like(gray)
                conflict=np.zeros_like(gray,bool);arrows=[];chains=[];theta=np.deg2rad(angle)
                rot=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
                shift=rng.integers(-3,4,2)
                def xy(p):return (np.asarray(p,float)-256)@rot.T+256+shift
                def paint(points,role,pen=width,closed=False,filled=False):
                    mask=np.zeros_like(gray);p=np.rint(xy(points)).astype(np.int32)
                    if filled:cv2.fillConvexPoly(mask,p,255)
                    else:cv2.polylines(mask,[p],closed,255,pen)
                    selected=mask>0;gray[selected]=0
                    if role is None:negative[selected&(truth==255)]=255
                    elif role==255:conflict[:]|=selected
                    else:
                        conflict[:]|=selected&(truth!=255)&(truth!=role);truth[selected&~conflict]=role
                def head(tip,d):
                    tip=np.asarray(tip,float);d=np.asarray(d,float);n=np.array([-d[1],d[0]])
                    base=tip-(24+width)*d;p=[tip,base+(5+width)*n,base-(5+width)*n]
                    paint(p,255,filled=True);identity='head-'+str(len(arrows))
                    arrows.append({'id':identity,'tip':xy([tip])[0].tolist(),'direction':(rot@d).tolist(),'triangle':xy(p).tolist()})
                    return identity
                if family in ('stamp_rules','parallel_note_rows'):
                    for y in (95,195,295,395):paint([[65,y],[447,y]],8,2*width+1)
                    for x in (65,230,447):paint([[x,95],[x,395]],8,2*width+1)
                    font=cv2.FONT_HERSHEY_COMPLEX if split=='train' else cv2.FONT_HERSHEY_TRIPLEX|cv2.FONT_ITALIC
                    for s,y in [('R P 47',160),('AP 89',260),('P 147',360)]:
                        canvas=np.zeros_like(gray);cv2.putText(canvas,s,(95,y),font,1.25,255,width)
                        matrix=np.c_[rot,256+shift-rot@np.array([256,256])]
                        mask=cv2.warpAffine(canvas,matrix,(512,512),flags=cv2.INTER_NEAREST)>0
                        gray[mask]=0;conflict[:]|=mask&(truth!=255)&(truth!=6);truth[mask&~conflict]=6
                    paint([[110,440],[132,447]],None,1)
                    if split=='validation':paint([[345,428],[365,420],[365,438]],None,filled=True)
                else:
                    if family=='ellipse_top_body':
                        t=np.linspace(np.pi,2*np.pi,129);top=np.c_[256+105*np.cos(t),285+35*np.sin(t)]
                        body=np.r_[top,[[361,392],[151,392]]]
                    elif family=='rounded_body_leader':
                        t=np.linspace(-np.pi/2,np.pi/2,129);right=np.c_[305+45*np.cos(t),325+65*np.sin(t)]
                        body=np.r_[[[155,260]],right,[[155,390]]]
                    elif family=='angled_wall_short':body=np.array([[165,275],[347,245],[375,390],[165,390]])
                    else:body=np.array([[155,265],[365,265],[365,390],[155,390]])
                    paint(body,1,2*width+1,True)
                    if family=='rounded_body_leader':
                        tip=np.array([205.,260.]);d=np.array([.6,.8]);base=tip-(24+width)*d
                        paint([base+[-70,-40],base,tip],4);head(tip,d)
                    else:
                        left,right=(218,294) if family!='two_near_dimensions' else (175,335)
                        feet=[[left,265],[right,265]]
                        if family=='ellipse_top_body':feet=[[left,252],[right,252]]
                        if family=='angled_wall_short':feet=[[x,275-(x-165)*30/182] for x in (left,right)]
                        for foot in feet:paint([[foot[0],110],foot],3)
                        for y in ((150,195) if family=='two_near_dimensions' else (150,)):
                            outside=family=='outside_short_span';length=24+width
                            paint([[left-length-width-3 if outside else left,y],[right+length+width+3 if outside else right,y]],2)
                            ids=[head([left,y],[1 if outside else -1,0]),head([right,y],[-1 if outside else 1,0])]
                            chains.append({'id':'dimension-'+str(len(chains)),'head_ids':ids,
                                'shaft':xy([[left+8,y],[right-8,y]]).tolist(),'feet_px':xy(feet).tolist(),
                                'head_scope':[0,0,512,512],'topology':'two_extensions','reference_kinds':['contour','contour'],
                                'whole_part_contour_visible':True})
                    paint([[208,422],[221,427]],None,1);paint([[405,181],[414,186]],None,1)
                truth[conflict]=255;negative[(truth!=255)|(gray==255)]=0
                name=f'joint-cv-{family}-w{width}-r{angle}';folder=BASE/name;folder.mkdir()
                for n,a in [('truth',truth),('negative',negative)]:Image.fromarray(a).save(folder/(n+'.png'))
                views=[]
                for mode,scale in [('native_crop',1.),('half_resolution_crop',.5)]:
                    image=gray if scale==1 else cv2.resize(gray,(256,256),interpolation=cv2.INTER_AREA)
                    path=folder/(mode+'.png');Image.fromarray(image).save(path)
                    views.append({'mode':mode,'gray':pin(ROOT,path.relative_to(ROOT).as_posix()),
                        'input_to_native':[[1/scale,0,.5/scale-.5],[0,1/scale,.5/scale-.5],[0,0,1]]})
                jobs.append({'case':{'id':name,'scope':'joint_cv_analytical','group_id':'joint-cv-'+family,'split':split,
                    'size_px':[512,512],'labels':{'family':family,'arrows':arrows,'chains':chains}},'views':views,'new_control':True,
                    **{n:pin(ROOT,(folder/(n+'.png')).relative_to(ROOT).as_posix()) for n in ('truth','negative')}})
    write(BASE/'contract.json',{'task':'P10.2c.3b.2h','seed':837151,'jobs':jobs,'families':FAMILIES,
        'generator':pin(ROOT,'scripts/prepare_joint_cv_controls_v1.py'),'prepared_before_joint_algorithm':True,
        'previous_validation_is_development':True,'criteria':{'full_role_macro_F1_min':.90,'new_known_role_precision_min':.98,
            'maximum_new_known_wrong_pixels':0,'maximum_new_negative_pixels':0,'minimum_fresh_correct_new_pixels':8,
            'maximum_lost_true_heads_or_chains':0,'maximum_new_false_heads_or_chains':0,
            'old_known_roles_source_and_chain_prefix_exact':True,'width_evidence_explicit_and_source_checked':True,
            'complete_bounded_repeats':True},
        'budgets':{'child_seconds':60,'RSS_MiB':1024,'output_MiB':64,'files':128,'total_child_seconds':1800},
        'production_changed':False,'physical_geometry_approved':False})
    print(pin(ROOT,'artifacts/joint-cv-controls-v1/contract.json'),flush=True)


if __name__=='__main__':build()
