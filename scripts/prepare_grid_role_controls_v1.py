"""Prospective grids, cross-hatching and part lattices before revision rules."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from cad_recovery.dataset_layout_v1 import pin
from run_topology_robust_v1 import write

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/grid-role-controls-v1'
FAMILIES=[('ruled_title','train'),('equal_pen_table','train'),('cross_hatch_panel','train'),('part_windows','train'),
    ('staggered_title','validation'),('broken_table_rules','validation'),('annotated_cross_hatch','validation'),('labelled_part_cells','validation')]


def build():
    BASE.mkdir(exist_ok=False);rng=np.random.default_rng(418763);jobs=[]
    for family,split in FAMILIES:
        for width in (3,5):
            for angle in ((21,67) if split=='train' else (39,131)):
                gray=np.full((512,512),255,np.uint8);truth=np.full_like(gray,255);conflict=np.zeros_like(gray,bool);negative=np.zeros_like(gray)
                theta=np.deg2rad(angle);rot=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
                shift=rng.integers(-3,4,2)
                def xy(p):return (np.asarray(p,float)-256)@rot.T+256+shift
                def line(p,role,pen=width,closed=False):
                    mask=np.zeros_like(gray);cv2.polylines(mask,[np.rint(xy(p)).astype(np.int32)],closed,255,pen)
                    selected=mask>0;gray[selected]=0
                    if role is None:negative[selected&(truth==255)]=255
                    else:
                        conflict[:]|=selected&(truth!=255)&(truth!=role);truth[selected&~conflict]=role
                def text(s,origin,scale=.95):
                    raw=np.zeros_like(gray);cv2.putText(raw,s,origin,cv2.FONT_HERSHEY_COMPLEX,scale,255,width,cv2.LINE_8)
                    matrix=np.c_[rot,256+shift-rot@np.array([256,256])]
                    selected=cv2.warpAffine(raw,matrix,(512,512),flags=cv2.INTER_NEAREST)>0
                    gray[selected]=0;conflict[:]|=selected&(truth!=255)&(truth!=6);truth[selected&~conflict]=6
                if family in ('ruled_title','equal_pen_table','staggered_title','broken_table_rules'):
                    pen=width if family=='equal_pen_table' else 2*width+1
                    ys=(95,195,295,395) if family!='staggered_title' else (80,185,300,410)
                    xs=(65,235,447)
                    for y in ys:line([[65,y],[447,y]],8,pen)
                    for x in xs:
                        if family=='broken_table_rules' and x==235:
                            line([[x,ys[0]],[x,225]],8,pen);line([[x,243],[x,ys[-1]]],8,pen)
                        else:line([[x,ys[0]],[x,ys[-1]]],8,pen)
                    for s,y in [('R P 47',ys[1]-26),('AP 89',ys[2]-26),('R P 14',ys[3]-26)]:text(s,(90,y),1.05)
                    line([[105,438],[127,445]],None,1)
                elif family in ('cross_hatch_panel','annotated_cross_hatch'):
                    # A real two-direction hatch lattice must not become a table.
                    line([[70,75],[445,75],[445,415],[70,415]],1,2*width+1,True)
                    for x in (110,185,260,335,410):line([[x,95],[x,395]],9)
                    for y in (115,195,275,355):line([[90,y],[425,y]],9)
                    if family=='annotated_cross_hatch':
                        for s,y in [('R P 47',170),('AP 89',250)]:text(s,(125,y),.85)
                    else:text('R 47',(190,458),.9)
                else:
                    # Thick part bars and holes can form a regular lattice too.
                    line([[70,75],[445,75],[445,415],[70,415]],1,2*width+1,True)
                    for x in (195,320):line([[x,75],[x,415]],1,2*width+1)
                    for y in (185,300):line([[70,y],[445,y]],1,2*width+1)
                    for x in (130,255,385):
                        for y in (130,245,355):
                            t=np.linspace(0,2*np.pi,101);line(np.array([x,y])+18*np.c_[np.cos(t),np.sin(t)],1,width,True)
                    if family=='labelled_part_cells':
                        text('A1',(85,160),.65);text('B2',(212,270),.65);text('C3',(335,385),.65)
                truth[conflict]=255;negative[(truth!=255)|(gray==255)]=0
                identity=f'grid-role-{family}-w{width}-r{angle}';folder=BASE/identity;folder.mkdir()
                for n,a in [('truth',truth),('negative',negative)]:Image.fromarray(a).save(folder/(n+'.png'))
                views=[]
                for mode,scale in [('native_crop',1.),('half_resolution_crop',.5)]:
                    image=gray if scale==1 else cv2.resize(gray,(256,256),interpolation=cv2.INTER_AREA)
                    p=folder/(mode+'.png');Image.fromarray(image).save(p)
                    views.append({'mode':mode,'gray':pin(ROOT,p.relative_to(ROOT).as_posix()),
                        'input_to_native':[[1/scale,0,.5/scale-.5],[0,1/scale,.5/scale-.5],[0,0,1]]})
                jobs.append({'case':{'id':identity,'scope':'grid_role_analytical','group_id':'grid-role-'+family,'split':split,
                    'size_px':[512,512],'labels':{'family':family,'arrows':[],'chains':[]}},'views':views,'new_control':True,
                    **{n:pin(ROOT,(folder/(n+'.png')).relative_to(ROOT).as_posix()) for n in ('truth','negative')}})
    write(BASE/'contract.json',{'task':'P10.2c.3b.2h.1','seed':418763,'jobs':jobs,'families':FAMILIES,
        'generator':pin(ROOT,'scripts/prepare_grid_role_controls_v1.py'),'prepared_before_revision_algorithm':True,
        'previous_validation_is_development':True,'criteria':{'minimum_fresh_wrong_roles_removed':32,
            'minimum_real_wrong_roles_removed':1,'maximum_lost_known_correct_roles':0,'maximum_new_wrong_or_noise_roles':0,
            'protected_review_masks_and_raw_maps_exact':True,'revision_only_untrusted_hatching_to_unknown':True,
            'source_geometry_heads_and_canonical_chains_exact':True,'complete_bounded_repeats':True,
            'full_role_macro_F1_min':.90},
        'budgets':{'child_seconds':60,'RSS_MiB':1024,'output_MiB':64,'files':128,'total_child_seconds':1800},
        'production_changed':False,'physical_geometry_approved':False})
    print(pin(ROOT,'artifacts/grid-role-controls-v1/contract.json'),flush=True)


if __name__=='__main__':build()
