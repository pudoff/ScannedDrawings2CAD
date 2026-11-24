"""Procedural role and interference truth prepared before ownership predictions."""
import math
import cv2
import numpy as np
from PIL import Image

from cad_recovery.stroke_roles_v1 import CLASSES

SEED=731865
FAMILIES=[('two_head_dimension','train'),('leader_to_part','train'),('hatched_part','train'),('text_and_frame','train'),
    ('single_head_dimension','validation'),('rotated_chain','validation'),('hatch_fake_tip','validation'),('touching_text_and_fold','validation')]


def scenes():
    rng=np.random.default_rng(SEED);scenes=[]
    for family,split in FAMILIES:
        for variant in range(3):
            # All widths/rotations/layouts of a construction family retain its split.
            shift=rng.integers(-7,8,2);thin=(2,3,4)[variant];thick=(5,7,9)[variant]
            geometry=[];arrows=[];texts=[];noise=[]
            def line(name,a,b,role,width=None):geometry.append({'id':name,'kind':'line','points':[a,b],'role':role,'stroke_width_px':width or thin})
            def loop(name,points,role,width=None):geometry.append({'id':name,'kind':'polyline','points':points,'closed':True,'role':role,'stroke_width_px':width or thick})
            def text(value,origin):texts.append((value,origin))
            def arrow(name,tip,direction):
                p=np.asarray(tip,float);u=np.asarray(direction,float);u/=np.linalg.norm(u);n=np.array([-u[1],u[0]])
                arrows.append({'id':name,'tip':p.tolist(),'direction':u.tolist(),'triangle':[p.tolist(),(p-18*u+6*n).tolist(),(p-18*u-6*n).tolist()]})
            box=[[110,130],[270,130],[270,245],[110,245]]
            loop('part',box,'part_contour')
            if family in ('two_head_dimension','single_head_dimension','rotated_chain'):
                line('shaft',[128,75],[252 if family!='single_head_dimension' else 270,75],'dimension_line')
                line('extension-left',[110,60],[110,130],'extension_line')
                line('extension-right',[270,60],[270,130],'extension_line')
                arrow('head-left',[110,75],[-1,0])
                if family!='single_head_dimension':arrow('head-right',[270,75],[1,0])
                text('160',(165,58))
            elif family=='leader_to_part':
                line('leader',[30,180],[92,180],'leader_line');arrow('leader-head',[110,180],[1,0]);text('NOTE',(20,150))
            elif family in ('hatched_part','hatch_fake_tip'):
                for j in range(6):line(f'hatch-{j}',[125+j*22,145],[140+j*22,225],'hatching')
                text('SECTION',(105,108))
                if family=='hatch_fake_tip':
                    # Explicit known interference: a tapered stain on parallel strokes, not a dimension head.
                    noise.append({'kind':'triangle','points':[[195,200],[187,181],[202,180]]})
                    noise.append({'kind':'line','points':[[195,180],[195,152]]})
            elif family=='text_and_frame':
                loop('frame',[[9,9],[438,9],[438,310],[9,310]],'sheet_frame',thin)
                text('ABC DEF',(100,95));text('PLATE 12',(310,270))
            else:
                text('R40 ABC',(240,165));text('EDGE',(60,132))
                noise.append({'kind':'polyline','points':[[310,50],[285,88],[330,112],[350,90],[399,140]]})
                noise.append({'kind':'arc','center':[340,240],'radius':42,'start':205,'sweep':115})
            angle=(0,90,180)[variant] if family=='rotated_chain' else (0,0,0)[variant]
            def transform(points):
                p=np.asarray(points,float)+shift
                if angle==180:return (np.array([447,319])-p).tolist()
                if angle==90:
                    # Uniform rotation into the same fixed canvas, with safe margins.
                    return (np.c_[319-p[:,1],p[:,0]]*.72+np.array([95,0])).tolist()
                return p.tolist()
            for g in geometry:g['points']=transform(g['points'])
            for a in arrows:
                a['tip']=transform([a['tip']])[0];a['triangle']=transform(a['triangle'])
                u=np.asarray(a['direction'])
                if angle==180:u=-u
                elif angle==90:u=np.array([-u[1],u[0]])
                a['direction']=u.tolist()
            gray=np.full((320,448),255,np.uint8);truth=np.full_like(gray,255);conflict=np.zeros_like(gray,bool)
            def add(mask,cls):
                selected=mask>0;conflict[:]|=selected&(truth!=255)&(truth!=cls);truth[selected&~conflict]=cls;gray[selected]=0
            for g in geometry:
                mask=np.zeros_like(gray);p=np.rint(g['points']).astype(np.int32)
                cv2.polylines(mask,[p],g.get('closed',False),255,g['stroke_width_px'])
                add(mask,CLASSES.index(g['role']))
            for value,origin in texts:
                mask=np.zeros_like(gray);cv2.putText(mask,value,origin,cv2.FONT_HERSHEY_SIMPLEX,.65,255,2,cv2.LINE_8)
                if angle:
                    # Text is transformed with the same uniform map as geometry.
                    a=np.array(transform([[0,0],[1,0],[0,1]]));matrix=np.c_[(a[1]-a[0]),(a[2]-a[0]),a[0]]
                    mask=cv2.warpAffine(mask,matrix,(448,320),flags=cv2.INTER_NEAREST)
                elif np.any(shift):mask=cv2.warpAffine(mask,np.array([[1,0,shift[0]],[0,1,shift[1]]],float),(448,320),flags=cv2.INTER_NEAREST)
                add(mask,6)
            for a in arrows:
                mask=np.zeros_like(gray);cv2.fillConvexPoly(mask,np.rint(a['triangle']).astype(np.int32),255)
                gray[mask>0]=0;conflict|=mask>0
            negative=np.zeros_like(gray)
            for n in noise:
                mask=np.zeros_like(gray)
                if n['kind']=='triangle':cv2.fillConvexPoly(mask,np.rint(transform(n['points'])).astype(np.int32),255)
                elif n['kind'] in ('line','polyline'):cv2.polylines(mask,[np.rint(transform(n['points'])).astype(np.int32)],False,255,thin)
                else:cv2.ellipse(mask,tuple(np.rint(transform([n['center']])[0]).astype(int)),(n['radius'],n['radius']),0,n['start'],n['start']+n['sweep'],255,thin)
                negative[(mask>0)&(truth==255)&~conflict]=255;gray[mask>0]=0
            truth[conflict]=255;negative[truth!=255]=0
            name=f'ownership-control-{family}-v{variant}'
            scenes.append({'case':{'id':name,'scope':'ownership_analytical_control','split':split,'group_id':'ownership-control-'+family,
                'size_px':[448,320],'labels':{'geometry':geometry,'arrows':arrows,'negative_regions':[],'family':family,
                    'authority':'procedural role construction before rasterization','physical_geometry_approved':False}},
                'gray':gray,'truth':truth,'negative':negative})
    return scenes


def prepare(root,base):
    from cad_recovery.dataset_layout_v1 import pin
    jobs=[]
    for scene in scenes():
        case=scene['case'];folder=base/'controls'/case['id'];folder.mkdir(parents=True)
        Image.fromarray(scene['truth']).save(folder/'truth.png');Image.fromarray(scene['negative']).save(folder/'known-negative-ink.png')
        views=[]
        for mode,scale in [('native_crop',1.),('half_resolution_crop',.5)]:
            sub=folder/mode;sub.mkdir()
            gray=scene['gray'] if scale==1 else cv2.resize(scene['gray'],(224,160),interpolation=cv2.INTER_AREA)
            Image.fromarray(gray).save(sub/'gray.png')
            views.append({'mode':mode,'gray':pin(root,(sub/'gray.png').relative_to(root).as_posix()),
                'input_to_native':[[1/scale,0,.5/scale-.5],[0,1/scale,.5/scale-.5],[0,0,1]],'cache':None})
        jobs.append({'case':case,'views':views,'new_control':True,
            'truth':pin(root,(folder/'truth.png').relative_to(root).as_posix()),
            'negative':pin(root,(folder/'known-negative-ink.png').relative_to(root).as_posix())})
    return jobs
