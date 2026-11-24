"""Freeze new shaft/extension contacts before introducing an ownership guard."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from cad_recovery.dataset_layout_v1 import pin
from run_topology_robust_v1 import write

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/head-junction-controls-v1'
FAMILIES=[('aligned_contact','train'),('offset_tip_contact','train'),('unequal_pens','train'),('detached_extension','train'),
          ('skew_extension','validation'),('terminal_contact','validation'),('bracket_contact','validation'),('intentional_tip_gap','validation')]


def main():
    BASE.mkdir(exist_ok=False);jobs=[];rng=np.random.default_rng(628153)
    for family,split in FAMILIES:
        for width in (3,5,7):
            for angle in ((11,68) if split=='train' else (37,143)):
                gray=np.full((512,512),255,np.uint8);truth=np.full_like(gray,255)
                conflict=np.zeros_like(gray,bool);part=np.zeros_like(gray);negative=np.zeros_like(gray)
                theta=np.deg2rad(angle);rot=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
                shift=rng.integers(-4,5,2)
                def xy(p):return (np.asarray(p,float)-256)@rot.T+256+shift
                def paint(p,cls,pen,closed=False,filled=False):
                    mask=np.zeros_like(gray);q=np.rint(xy(p)).astype(np.int32)
                    if filled:cv2.fillConvexPoly(mask,q,255)
                    else:cv2.polylines(mask,[q],closed,255,pen)
                    selected=mask>0;gray[selected]=0
                    if cls==1:part[selected]=255
                    if cls is None:negative[selected&(truth==255)]=255
                    elif cls==255:conflict[:]|=selected
                    else:
                        conflict[:]|=selected&(truth!=255)&(truth!=cls);truth[selected&~conflict]=cls
                t=np.linspace(0,1,129)
                top=np.c_[140+232*t,242-18*np.sin(np.pi*t)]
                if family=='bracket_contact':top=np.array([[140,242],[195,242],[195,222],[322,222],[322,242],[372,242]])
                paint(np.r_[top,[[372,403],[140,403]]],1,2*width+1,True)
                shaft_pen=max(2,width-2) if family=='unequal_pens' else width
                ext_pen=width+2 if family=='unequal_pens' else width
                left_tip=144 if family=='skew_extension' else 140
                paint([[left_tip,116],[372,116]],2,shaft_pen)
                for x in (140,372):
                    apex=x+(1 if family=='offset_tip_contact' and x==140 else 0)
                    upper=144 if family=='skew_extension' and x==140 else apex
                    lo=116 if family=='terminal_contact' else 94
                    paint([[upper,lo],[x,242]],3,ext_pen)
                arrows=[]
                for x,sign in ((left_tip,-1),(372,1)):
                    tip=np.array([x,116]);base=tip-np.array([sign*(24+width),0])
                    triangle=[tip,base+[0,5+width],base-[0,5+width]];paint(triangle,255,1,filled=True)
                    arrows.append({'id':'head-'+str(len(arrows)),'tip':xy([tip])[0].tolist(),
                                   'direction':(rot@np.array([sign,0])).tolist(),'triangle':xy(triangle).tolist()})
                expected=1
                if family in ('detached_extension','intentional_tip_gap'):
                    y0,y1=(177,195) if family=='detached_extension' else (135,149)
                    cut=np.zeros_like(gray)
                    cv2.fillConvexPoly(cut,np.rint(xy([[124,y0],[156,y0],[156,y1],[124,y1]])).astype(np.int32),255)
                    clear=(cut>0)&(part==0);gray[clear]=255;truth[clear]=255;conflict[clear]=False;expected=0
                paint([[213,174],[226,182]],None,1);paint([[281,433],[295,438]],None,1)
                truth[conflict]=255;negative[truth!=255]=0;negative[gray==255]=0
                name=f'head-junction-{family}-w{width}-r{angle}';folder=BASE/name;folder.mkdir()
                for n,a in [('truth',truth),('negative',negative)]:Image.fromarray(a).save(folder/(n+'.png'))
                views=[]
                for mode,scale in [('native_crop',1.),('half_resolution_crop',.5)]:
                    image=gray if scale==1 else cv2.resize(gray,(256,256),interpolation=cv2.INTER_AREA)
                    p=folder/(mode+'.png');Image.fromarray(image).save(p)
                    views.append({'mode':mode,'gray':pin(ROOT,p.relative_to(ROOT).as_posix()),
                                  'input_to_native':[[1/scale,0,.5/scale-.5],[0,1/scale,.5/scale-.5],[0,0,1]]})
                jobs.append({'case':{'id':name,'scope':'head_junction_analytical_control','group_id':'head-junction-'+family,
                     'split':split,'size_px':[512,512],'labels':{'family':family,'arrows':arrows,'expected_dimension_chains':expected}},
                     'views':views,'new_control':True,**{n:pin(ROOT,(folder/(n+'.png')).relative_to(ROOT).as_posix()) for n in ('truth','negative')}})
    write(BASE/'contract.json',{'task':'P10.2c.3b.2d','generator':pin(ROOT,'scripts/prepare_head_junction_controls_v1.py'),
          'seed':628153,'families':FAMILIES,'jobs':jobs,'prepared_before_ownership_guard':True,
          'criteria':['five previous errors return to unknown','zero new known-wrong/noise roles',
                      'correct additions retained >=80% overall and on new validation','all inherited routes roles unchanged',
                      'heads and chain observations immutable','new validation positive additions','zero false chains on explicit negatives',
                      'full-corpus joint comparison with bounded complete searches and exact repeats'],
          'full_role_macro_F1_min':.90,'production_changed':False,'physical_geometry_approved':False,
          'budgets':{'child_seconds':60,'RSS_MiB':1024,'output_MiB':64,'files':128,'total_child_seconds':1800,'OpenCV_threads':1,'RNG_seed':0}})
    print(pin(ROOT,(BASE/'contract.json').relative_to(ROOT).as_posix()))


if __name__=='__main__':main()
