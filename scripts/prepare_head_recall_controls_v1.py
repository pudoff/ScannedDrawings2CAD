"""Prospective split-family sources for short and curved head recall."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from cad_recovery.dataset_layout_v1 import pin
from run_topology_robust_v1 import write

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/head-recall-controls-v1'
FAMILIES=[('tiny_tail_pair','train'),('open_sweep_pair','train'),('ring_tangent','train'),('mechanical_words','train'),
    ('short_oblique','validation'),('s_curve_heads','validation'),('loop_notch_heads','validation'),('stamp_brackets','validation')]


def build():
    BASE.mkdir(exist_ok=False);rng=np.random.default_rng(901756);jobs=[]
    for family,split in FAMILIES:
        for width in (3,5):
            for angle in ((19,73) if split=='train' else (43,133)):
                gray=np.full((512,512),255,np.uint8);refs=[];theta=np.deg2rad(angle)
                rot=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
                shift=rng.integers(-3,4,2)
                def xy(p):return (np.asarray(p,float)-256)@rot.T+256+shift
                def line(p,closed=False):cv2.polylines(gray,[np.rint(p).astype(np.int32)],closed,0,width)
                def head(tip,direction):
                    tip=np.asarray(tip,float);d=np.asarray(direction,float);n=np.array([-d[1],d[0]])
                    base=tip-(25+width)*d;p=[tip,base+(5+width)*n,base-(5+width)*n]
                    cv2.fillConvexPoly(gray,np.rint(p).astype(np.int32),0)
                    refs.append({'id':'head-'+str(len(refs)),'tip':xy([tip])[0].tolist(),'direction':(rot@d).tolist(),'triangle':xy(p).tolist()})
                    return base
                if family in ('tiny_tail_pair','short_oblique'):
                    for y,sign in ((190,1),(320,-1)):
                        d=np.array([sign,0.]) if family=='tiny_tail_pair' else sign*np.array([.8,.6])
                        tip=np.array([250.,y]);base=head(tip,d);line([base-(width+3)*d,tip])
                    if family=='short_oblique':cv2.putText(gray,'R 25',(330,150),cv2.FONT_HERSHEY_SIMPLEX,.8,0,width)
                elif family=='open_sweep_pair':
                    center=np.array([250,265]);radius=102;t=np.linspace(-1.2,.95,241)
                    line(center+radius*np.c_[np.cos(t),np.sin(t)])
                    for a,s in ((t[0],-1),(t[-1],1)):head(center+radius*np.array([np.cos(a),np.sin(a)]),s*np.array([-np.sin(a),np.cos(a)]))
                elif family=='ring_tangent':
                    center=np.array([245,260]);radius=82;t=np.linspace(0,2*np.pi,361)
                    line(center+radius*np.c_[np.cos(t),np.sin(t)],True);head(center+[0,-radius],[1,0])
                elif family=='s_curve_heads':
                    t=np.linspace(0,1,241);p=np.c_[145+220*t,256+65*np.sin(2*np.pi*t)]
                    line(p);d=np.array([220,130*np.pi]);d=d/np.linalg.norm(d)
                    head(p[0],-d);head(p[-1],d)
                elif family=='loop_notch_heads':
                    center=np.array([250,260]);radius=72;t=np.linspace(.4,2*np.pi+.4,361)
                    line(center+radius*np.c_[np.cos(t),np.sin(t)],True)
                    for a in (.4,3.5):head(center+radius*np.array([np.cos(a),np.sin(a)]),np.array([-np.sin(a),np.cos(a)]))
                else:
                    font=cv2.FONT_HERSHEY_COMPLEX if split=='train' else cv2.FONT_HERSHEY_TRIPLEX|cv2.FONT_ITALIC
                    for s,y in [('R P A 47',160),('P 89 R 4,7',280),('RP 147 AP',390)]:
                        cv2.putText(gray,s,(55,y),font,1.35,0,width,cv2.LINE_8)
                    for y in (65,205,325,440):cv2.line(gray,(35,y),(475,y),0,2*width+1)
                    if family=='stamp_brackets':
                        for x in (65,245,415):line([[x,455],[x+20,472],[x,489]])
                        # A detached filled marker is not a head on a shaft.
                        cv2.fillConvexPoly(gray,np.array([[440,245],[405,232],[405,258]],np.int32),0)
                matrix=np.c_[rot,256+shift-rot@np.array([256,256])]
                gray=cv2.warpAffine(gray,matrix,(512,512),flags=cv2.INTER_NEAREST,borderValue=255)
                identity=f'head-recall-{family}-w{width}-r{angle}';folder=BASE/identity;folder.mkdir();views=[]
                for mode,scale in [('native_crop',1.),('half_resolution_crop',.5)]:
                    pixels=gray if scale==1 else cv2.resize(gray,(256,256),interpolation=cv2.INTER_AREA)
                    path=folder/(mode+'.png');Image.fromarray(pixels).save(path)
                    views.append({'mode':mode,'gray':pin(ROOT,path.relative_to(ROOT).as_posix()),
                        'input_to_native':[[1/scale,0,.5/scale-.5],[0,1/scale,.5/scale-.5],[0,0,1]]})
                jobs.append({'case':{'id':identity,'scope':'head_recall_analytical','group_id':'head-recall-'+family,
                    'split':split,'size_px':[512,512],'labels':{'family':family,'arrows':refs,'chains':[]}},
                    'views':views,'new_control':True})
    write(BASE/'contract.json',{'task':'P10.2c.3b.2g.2','seed':901756,'jobs':jobs,'families':FAMILIES,
        'generator':pin(ROOT,'scripts/prepare_head_recall_controls_v1.py'),'prepared_before_algorithm':True,
        'previous_validation_is_development':True,'criteria':{'minimum_fresh_added_true_heads':8,
            'minimum_old_short_curved_added_true_heads':8,'maximum_additional_reviewed_false_heads':0,
            'maximum_lost_previous_true_targets':0,'maximum_new_wrong_or_lost_correct_chains':0,
            'raw_prefix_source_and_roles_exact':True,'complete_bounded_repeats':True},
        'budgets':{'child_seconds':60,'RSS_MiB':1024,'output_MiB':64,'files':128,'total_child_seconds':1800},
        'production_changed':False,'physical_geometry_approved':False})
    print(pin(ROOT,'artifacts/head-recall-controls-v1/contract.json'),flush=True)


if __name__=='__main__':build()
