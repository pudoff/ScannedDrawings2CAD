"""Freeze source-route positives and deliberate white-gap negatives before rules."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from cad_recovery.dataset_layout_v1 import pin
from run_topology_robust_v1 import write

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/source-chain-controls-v1'
FAMILIES=[('sloped_crest','train'),('round_shoulder','train'),('gap_at_shaft_junction','train'),('gap_near_foot','train'),
          ('sinusoidal_crest','validation'),('beveled_wall','validation'),('separated_endpoint','validation'),('missing_extension','validation')]


def main():
    BASE.mkdir(exist_ok=False);jobs=[];rng=np.random.default_rng(934721)
    for family,split in FAMILIES:
        for width in (3,5,7):
            for angle in ((23,76) if split=='train' else (49,118)):
                gray=np.full((512,512),255,np.uint8);truth=np.full_like(gray,255)
                conflict=np.zeros_like(gray,bool);negative=np.zeros_like(gray);part=np.zeros_like(gray)
                a=np.deg2rad(angle);rotation=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
                shift=rng.integers(-4,5,2)
                def transform(p):return (np.asarray(p,float)-256)@rotation.T+256+shift
                def draw(p,cls,pen,closed=False,filled=False):
                    mask=np.zeros_like(gray);q=np.rint(transform(p)).astype(np.int32)
                    if filled:cv2.fillConvexPoly(mask,q,255)
                    else:cv2.polylines(mask,[q],closed,255,pen)
                    selected=mask>0;gray[selected]=0
                    if cls==1:part[selected]=255
                    if cls==255:conflict[:]|=selected
                    elif cls is None:negative[selected&(truth==255)]=255
                    else:
                        conflict[:]|=selected&(truth!=255)&(truth!=cls)
                        truth[selected&~conflict]=cls
                t=np.linspace(0,1,161);x=140+232*t
                if family=='sloped_crest':y=242+14*np.sin(2*np.pi*t)-20*np.sin(np.pi*t)
                elif family=='round_shoulder':y=242-34*np.sin(np.pi*t)**2
                elif family=='sinusoidal_crest':y=242+22*np.sin(3*np.pi*t)
                else:y=242+12*np.sin(np.pi*t)
                top=np.c_[x,y]
                if family=='beveled_wall':top=np.array([[140,242],[185,213],[320,230],[372,242]])
                draw(np.r_[top,[[372,403],[140,403]]],1,2*width+1,True)
                draw([[140,116],[372,116]],2,width)
                for px in (140,372):
                    if family=='missing_extension' and px==372:continue
                    draw([[px,94],[px,242]],3,width)
                arrows=[]
                for px,direction in ((140,-1),(372,1)):
                    tip=np.array([px,116]);base=tip-np.array([direction*(23+width),0])
                    triangle=[tip,base+[0,5+width],base-[0,5+width]]
                    draw(triangle,255,1,filled=True)
                    arrows.append({'id':f'head-{len(arrows)}','tip':transform([tip])[0].tolist(),
                                   'direction':(rotation@np.array([direction,0])).tolist(),'triangle':transform(triangle).tolist()})
                negative_family=family in ('gap_at_shaft_junction','gap_near_foot','separated_endpoint','missing_extension')
                if family in ('gap_at_shaft_junction','gap_near_foot','separated_endpoint'):
                    # Remove only a true transverse piece of the extension. Main
                    # geometry is protected even where the raster pens overlap.
                    yy=(146,156) if family=='gap_at_shaft_junction' else (218,230) if family=='gap_near_foot' else (188,204)
                    cut=np.zeros_like(gray)
                    cv2.fillConvexPoly(cut,np.rint(transform([[126,yy[0]],[154,yy[0]],[154,yy[1]],[126,yy[1]]])).astype(np.int32),255)
                    clear=(cut>0)&(part==0);gray[clear]=255;truth[clear]=255;conflict[clear]=False
                draw([[213,175],[227,181]],None,1);draw([[280,431],[295,437]],None,1)
                truth[conflict]=255;negative[truth!=255]=0;negative[gray==255]=0
                name=f'source-chain-{family}-w{width}-r{angle}';folder=BASE/name;folder.mkdir()
                for n,array in [('truth',truth),('negative',negative)]:Image.fromarray(array).save(folder/(n+'.png'))
                views=[]
                for mode,scale in [('native_crop',1.),('half_resolution_crop',.5)]:
                    image=gray if scale==1 else cv2.resize(gray,(256,256),interpolation=cv2.INTER_AREA)
                    p=folder/(mode+'.png');Image.fromarray(image).save(p)
                    views.append({'mode':mode,'gray':pin(ROOT,p.relative_to(ROOT).as_posix()),
                                  'input_to_native':[[1/scale,0,.5/scale-.5],[0,1/scale,.5/scale-.5],[0,0,1]]})
                jobs.append({'case':{'id':name,'scope':'source_chain_analytical_control','group_id':'source-chain-'+family,'split':split,
                                    'size_px':[512,512],'labels':{'family':family,'arrows':arrows,'expected_dimension_chains':0 if negative_family else 1}},
                             'views':views,'new_control':True,
                             **{n:pin(ROOT,(folder/(n+'.png')).relative_to(ROOT).as_posix()) for n in ('truth','negative')}})
    write(BASE/'contract.json',{'task':'P10.2c.3b.2c','generator':pin(ROOT,'scripts/prepare_source_chain_controls_v1.py'),
          'families':FAMILIES,'jobs':jobs,'seed':934721,'prepared_before_chain_recovery':True,
          'gates':['no false chains on deliberate gaps','positive new validation chain/part gain','no new known-wrong/noise roles',
                   'all inherited heads/roles/source geometry unchanged','old boundary guard and retention preserved','bounded complete searches and exact repeats'],
          'full_classifier_macro_F1_min':.90,'budgets':{'child_seconds':60,'RSS_MiB':1024,'output_MiB':64,'files':128,'total_child_seconds':1800,'OpenCV_threads':1,'RNG_seed':0},
          'production_changed':False,'physical_geometry_approved':False})
    print(pin(ROOT,(BASE/'contract.json').relative_to(ROOT).as_posix()))


if __name__=='__main__':main()
