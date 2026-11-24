"""Prospective head localization/own-stem controls before changing inference."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from cad_recovery.dataset_layout_v1 import pin
from run_topology_robust_v1 import write

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/head-refinement-controls-v1'
FAMILIES=[('linear_pair','train'),('neighbor_extensions','train'),('text_glyphs','train'),('coincident_tips','train'),
          ('offset_pair','validation'),('dense_dimensions','validation'),('curved_path','validation'),('title_glyphs','validation')]


def build():
    BASE.mkdir(exist_ok=False);jobs=[];rng=np.random.default_rng(319756)
    for family,split in FAMILIES:
        for pen in (3,5):
            for angle in ((17,53) if split=='train' else (37,121)):
                canvas=np.full((512,512),255,np.uint8);theta=np.deg2rad(angle)
                rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
                offset=rng.integers(-2,3,2);arrows=[];chains=[]
                def xy(p):return (np.asarray(p,float)-256)@rotation.T+256+offset
                def line(points,width):cv2.polylines(canvas,[np.rint(points).astype(np.int32)],False,0,width)
                def head(tip,direction):
                    d=np.asarray(direction,float);normal=np.array([-d[1],d[0]]);tip=np.asarray(tip,float)
                    base=tip-(26+pen)*d;triangle=[tip,base+(5+pen)*normal,base-(5+pen)*normal]
                    cv2.fillConvexPoly(canvas,np.rint(triangle).astype(np.int32),0)
                    arrows.append({'id':'head-'+str(len(arrows)),'tip':xy([tip])[0].tolist(),
                        'direction':(rotation@d).tolist(),'triangle':xy(triangle).tolist()})
                if family in ('text_glyphs','title_glyphs'):
                    font=cv2.FONT_HERSHEY_SIMPLEX if split=='train' else cv2.FONT_HERSHEY_COMPLEX|cv2.FONT_ITALIC
                    for text,y in [('A47X',190),('147 8,7',290),('MT 24.35',380)]:
                        cv2.putText(canvas,text,(80,y),font,1.4,0,pen,cv2.LINE_8)
                    for y in (85,225,330,430):line([[45,y],[465,y]],2*pen+1)
                    line([[45,85],[45,430]],2*pen+1);line([[465,85],[465,430]],2*pen+1)
                elif family=='coincident_tips':
                    line([[60,230],[450,230]],pen);head([255,230],[-1,0]);head([255,230],[1,0])
                elif family=='curved_path':
                    center=np.array([256,256]);radius=130;t=np.linspace(-.9,.9,180)
                    line(center+radius*np.c_[np.cos(t),np.sin(t)],pen)
                    for a,sign in ((-.9,-1),(.9,1)):
                        head(center+radius*np.array([np.cos(a),np.sin(a)]),sign*np.array([-np.sin(a),np.cos(a)]))
                else:
                    ys=(160,255) if family=='dense_dimensions' else (180,)
                    feet=[[145,375],[360,375]]
                    line([[85,300],[85,375],[435,375],[435,300]],2*pen+3)
                    for x in (145,360):line([[x,135],[x,375]],pen)
                    if family=='neighbor_extensions':line([[376,135],[376,395]],pen)
                    if family=='offset_pair':line([[410,135],[410,395]],pen)
                    for y in ys:
                        line([[145,y],[360,y]],pen);ids=[]
                        for x,sign in ((145,-1),(360,1)):
                            head([x,y],[sign,0]);ids.append(arrows[-1]['id'])
                        chains.append({'id':'dimension-'+str(len(chains)),'head_ids':ids,
                            'shaft':xy([[200,y],[305,y]]).tolist(),'feet_px':xy(feet).tolist(),
                            'head_scope':[0,0,512,512],'topology':'two_extensions',
                            'reference_kinds':['contour','contour'],'whole_part_contour_visible':False})
                # Transform the already drawn pixels and separately stored geometry
                # by the same rotation/translation; interpolation is fixed in advance.
                matrix=np.c_[rotation,256+offset-rotation@np.array([256,256])]
                gray=cv2.warpAffine(canvas,matrix,(512,512),flags=cv2.INTER_NEAREST,borderValue=255)
                name=f'head-refinement-{family}-w{pen}-r{angle}';folder=BASE/name;folder.mkdir();views=[]
                for mode,scale in [('native_crop',1.),('half_resolution_crop',.5)]:
                    image=gray if scale==1 else cv2.resize(gray,(256,256),interpolation=cv2.INTER_AREA)
                    path=folder/(mode+'.png');Image.fromarray(image).save(path)
                    views.append({'mode':mode,'gray':pin(ROOT,path.relative_to(ROOT).as_posix()),
                        'input_to_native':[[1/scale,0,.5/scale-.5],[0,1/scale,.5/scale-.5],[0,0,1]]})
                jobs.append({'case':{'id':name,'scope':'head_refinement_analytical','group_id':'head-refinement-'+family,
                    'split':split,'size_px':[512,512],'labels':{'arrows':arrows,'chains':chains,
                    'expected_dimension_chains':len(chains),'family':family}},'views':views,'new_control':True})
    write(BASE/'contract.json',{'task':'P10.2c.3b.2g','jobs':jobs,'families':FAMILIES,'seed':319756,
        'generator':pin(ROOT,'scripts/prepare_head_refinement_controls_v1.py'),'prepared_before_algorithm':True,
        'criteria':{'minimum_fresh_correct_head_retention':.95,'minimum_fresh_correct_heads':8,
          'no_new_wrong_fresh_heads':True,'no_loss_correct_inherited_chains':True,
          'real_50_half_improved_with_fixed_8px_tolerance':True,'title_false_heads_reduced_without_GT_in_model':True,
          'raw_heads_source_roles_and_prefix_immutable':True,'exact_bounded_repeats':True},
        'budgets':{'child_seconds':60,'RSS_MiB':1024,'output_MiB':64,'files':128,'total_child_seconds':1800},
        'physical_geometry_approved':False,'test_used':False})
    print(pin(ROOT,'artifacts/head-refinement-controls-v1/contract.json'),flush=True)


if __name__=='__main__':build()
