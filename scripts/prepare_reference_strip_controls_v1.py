"""Prospective controls for insufficient context on a single cropped support strip."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from cad_recovery.dataset_layout_v1 import pin
from run_topology_robust_v1 import write

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/reference-strip-controls-v1'
FAMILIES=[('plain_clipped_strip','train'),('clipped_detail_leg','train'),
          ('clipped_parallel_border','validation'),('clipped_detail_step','validation')]


def build():
    BASE.mkdir(exist_ok=False); rng=np.random.default_rng(651902); jobs=[]
    for family,split in FAMILIES:
        for width in (3,5):
            for angle in ((17,67) if split=='train' else (37,139)):
                gray=np.full((512,512),255,np.uint8); truth=np.full_like(gray,255); conflict=np.zeros_like(gray,bool)
                theta=np.deg2rad(angle); rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
                offset=rng.integers(-2,3,2)
                def xy(p):return (np.asarray(p,float)-256)@rotation.T+256+offset
                def paint(p,role,pen,filled=False):
                    mask=np.zeros_like(gray); q=np.rint(xy(p)).astype(np.int32)
                    if filled:cv2.fillConvexPoly(mask,q,255)
                    else:cv2.polylines(mask,[q],False,255,pen)
                    use=mask>0;gray[use]=0
                    if role==255:conflict[:]|=use
                    else:
                        conflict[:]|=use&(truth!=255)&(truth!=role);truth[use&~conflict]=role
                negative=family in ('plain_clipped_strip','clipped_parallel_border');pen=2*width+3
                paint([[-400,330],[912,330]],255 if negative else 1,pen)
                if family=='clipped_parallel_border':paint([[145,25],[260,25]],255,pen)
                if family=='clipped_detail_leg':paint([[90,330],[90,205]],1,pen)
                if family=='clipped_detail_step':paint([[430,330],[430,285],[395,285],[395,210]],1,pen)
                paint([[150,130],[360,130]],2,width)
                for x in (150,360):paint([[x,105],[x,330]],3,width)
                arrows=[]
                for x,sign in ((150,-1),(360,1)):
                    tip=np.array([x,130]);base=tip-[sign*(26+width),0]
                    triangle=[tip,base+[0,5+width],base-[0,5+width]];paint(triangle,255,1,True)
                    arrows.append({'id':'head-'+str(len(arrows)),'tip':xy([tip])[0].tolist(),
                                   'direction':(rotation@[sign,0]).tolist(),'triangle':xy(triangle).tolist()})
                truth[conflict]=255;name=f'reference-strip-{family}-w{width}-r{angle}'
                folder=BASE/name;folder.mkdir();Image.fromarray(truth).save(folder/'truth.png');views=[]
                for mode,scale in [('native_crop',1.),('half_resolution_crop',.5)]:
                    image=gray if scale==1 else cv2.resize(gray,(256,256),interpolation=cv2.INTER_AREA)
                    path=folder/(mode+'.png');Image.fromarray(image).save(path)
                    views.append({'mode':mode,'gray':pin(ROOT,path.relative_to(ROOT).as_posix()),
                        'input_to_native':[[1/scale,0,.5/scale-.5],[0,1/scale,.5/scale-.5],[0,0,1]]})
                jobs.append({'case':{'id':name,'scope':'reference_strip_analytical_control','group_id':'reference-strip-'+family,
                    'split':split,'size_px':[512,512],'labels':{'family':family,'arrows':arrows,
                    'feet_px':xy([[150,330],[360,330]]).tolist(),'expected_dimension_chains':int(not negative)}},
                    'views':views,'new_control':True,'truth':pin(ROOT,(folder/'truth.png').relative_to(ROOT).as_posix())})
    write(BASE/'contract.json',{'task':'P10.2c.3b.2f.1','seed':651902,'generator':pin(ROOT,'scripts/prepare_reference_strip_controls_v1.py'),
        'jobs':jobs,'families':FAMILIES,'prepared_before_context_v2':True,
        'negative_meaning':'insufficient visible part context; not a claim that every isolated strip is a frame',
        'previous_validation_is_now_development':True,
        'criteria':{'zero_new_chains_on_explicit_negatives':True,'remove_all_six_previous_false_additions':True,
          'minimum_correct_fresh_validation_chains':4,'minimum_correct_fresh_basis_retention':.8,
          'minimum_correct_initial_validation_retention':.8,'preserve_real_train50_native_and_two_inherited_successes':True,
          'no_new_wrong_usable_positive_chain':True,'unchanged_observations_roles_and_inherited_prefix':True,
          'complete_bounded_search_and_exact_repeats':True},
        'budgets':{'child_wall_seconds':60,'child_memory_bytes':1073741824,'child_output_bytes':67108864,
          'child_output_files':128,'aggregate_child_wall_seconds':1800},'physical_units_approved':False})
    print(pin(ROOT,'artifacts/reference-strip-controls-v1/contract.json'),flush=True)


if __name__=='__main__':build()
