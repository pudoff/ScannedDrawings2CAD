"""Fresh split-family controls before correcting bases or adding a context guard."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from cad_recovery.dataset_layout_v1 import pin
from run_topology_robust_v1 import write

ROOT = Path(__file__).resolve().parents[1]; BASE = ROOT/'artifacts/reference-context-controls-v1'
FAMILIES = [('part_with_hatch','train'), ('double_level_part','train'), ('clipped_frame','train'),
            ('equal_span_hatch','train'), ('ruled_grid','train'), ('stepped_with_thin_hatch','validation'),
            ('curved_open_part','validation'), ('skew_frame','validation'),
            ('headed_parallel_rows','validation'), ('broken_opposing_reference','validation')]


def build():
    BASE.mkdir(exist_ok=False); rng=np.random.default_rng(951843); jobs=[]
    for family,split in FAMILIES:
        for width in (3,5):
            for angle in ((11,69) if split=='train' else (31,127)):
                gray=np.full((640,640),255,np.uint8); truth=np.full_like(gray,255); conflict=np.zeros_like(gray,bool)
                theta=np.deg2rad(angle); rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
                offset=rng.integers(-3,4,2)
                def xy(p): return (np.asarray(p,float)-320)@rotation.T+320+offset
                def paint(p,cls,pen,filled=False):
                    mask=np.zeros_like(gray); q=np.rint(xy(p)).astype(np.int32)
                    if filled: cv2.fillConvexPoly(mask,q,255)
                    else: cv2.polylines(mask,[q],False,255,pen)
                    use=mask>0; gray[use]=0
                    if cls==255: conflict[:]|=use
                    else:
                        conflict[:]|=use&(truth!=255)&(truth!=cls); truth[use&~conflict]=cls
                pen=2*width+3; feet=[[190,345],[450,345]]
                body=[[120,235],[120,345],[530,345],[530,235]]
                negative=family in ('clipped_frame','equal_span_hatch','ruled_grid','skew_frame','headed_parallel_rows','broken_opposing_reference')
                if family=='stepped_with_thin_hatch':
                    body=[[120,235],[120,335],[270,335],[350,375],[530,375],[530,235]]; feet=[[190,335],[450,375]]
                elif family=='curved_open_part':
                    t=np.linspace(0,np.pi,65)
                    body=np.r_[[[120,235],[120,345],[240,345]],np.c_[240+160*t/np.pi,345+22*np.sin(t)],[[530,345],[530,235]]]
                if family in ('clipped_frame','skew_frame'):
                    polygon=[[25,25],[615,25],[615,615],[25,615],[25,25]]
                    if family=='skew_frame': polygon=[[30,55],[620,25],[600,610],[35,625],[30,55]]
                    feet=[[190,615],[450,615]]; paint(polygon,7,pen)
                elif family in ('equal_span_hatch','headed_parallel_rows'):
                    for y in (270,295,320,345,370,395): paint([[120,y],[530,y]],9,pen)
                    paint([[120,270],[120,395]],9,pen)
                    if family=='headed_parallel_rows': paint([[260,320],[283,310],[283,330]],255,1,True)
                elif family=='ruled_grid':
                    for y in (245,280,315,350,385,430): paint([[100,y],[540,y]],8,pen)
                    for x in (100,190,275,360,450,540): paint([[x,245],[x,430]],8,pen)
                    feet=[[190,350],[450,350]]
                else:
                    paint(body,1,pen)
                    if family=='double_level_part': paint([[190,345],[190,285],[330,285],[330,345]],1,pen)
                    if family in ('part_with_hatch','stepped_with_thin_hatch'):
                        for x in range(150,500,36): paint([[x,345],[x+45,285]],9,1)
                paint([[190,145],[450,145]],2,width)
                for foot in feet: paint([[foot[0],120],foot],3,width)
                arrows=[]
                for x,sign in ((190,-1),(450,1)):
                    tip=np.array([x,145]); base=tip-[sign*(28+width),0]
                    triangle=[tip,base+[0,6+width],base-[0,6+width]]; paint(triangle,255,1,True)
                    arrows.append({'id':'head-'+str(len(arrows)),'tip':xy([tip])[0].tolist(),
                                   'direction':(rotation@[sign,0]).tolist(),'triangle':xy(triangle).tolist()})
                if family=='broken_opposing_reference':
                    clear=np.zeros_like(gray); cv2.fillConvexPoly(clear,np.rint(xy([[304,330],[332,330],[332,360],[304,360]])).astype(np.int32),255)
                    gray[clear>0]=255; truth[clear>0]=255; conflict[clear>0]=False
                truth[conflict]=255
                name=f'reference-context-{family}-w{width}-r{angle}'; folder=BASE/name; folder.mkdir()
                Image.fromarray(truth).save(folder/'truth.png'); views=[]
                for mode,scale in [('native_crop',1.),('half_resolution_crop',.5)]:
                    image=gray if scale==1 else cv2.resize(gray,(320,320),interpolation=cv2.INTER_AREA)
                    path=folder/(mode+'.png'); Image.fromarray(image).save(path)
                    views.append({'mode':mode,'gray':pin(ROOT,path.relative_to(ROOT).as_posix()),
                                  'input_to_native':[[1/scale,0,.5/scale-.5],[0,1/scale,.5/scale-.5],[0,0,1]]})
                jobs.append({'case':{'id':name,'scope':'reference_context_analytical_control','group_id':'reference-context-'+family,
                    'split':split,'size_px':[640,640],'labels':{'family':family,'arrows':arrows,'feet_px':xy(feet).tolist(),
                    'expected_dimension_chains':int(not negative)}},'views':views,'new_control':True,
                    'truth':pin(ROOT,(folder/'truth.png').relative_to(ROOT).as_posix())})
    write(BASE/'contract.json',{'task':'P10.2c.3b.2f.1','generator':pin(ROOT,'scripts/prepare_reference_context_controls_v1.py'),
          'seed':951843,'families':FAMILIES,'jobs':jobs,'prepared_before_algorithm_changes':True,
          'arms':['original_v1','basis_v2','context_v2'],'criteria':[
              'basis endpoint/order invariance on synthetic and real source observations',
              'zero false new context chains on all explicit negatives',
              'at least four correct new fresh validation context chains',
              'retain >=80 percent of correct basis-v2 fresh validation additions',
              'preserve real train50 native gain and both old real successes',
              'unchanged source/head/role maps and inherited chain prefix',
              'bounded complete searches and exact repeats'],
          'full_role_macro_F1_min':.90,'production_changed':False,'physical_geometry_approved':False,
          'budgets':{'child_seconds':60,'RSS_MiB':1024,'output_MiB':64,'files':128,'total_child_seconds':1800}})
    print(pin(ROOT,'artifacts/reference-context-controls-v1/contract.json'))


if __name__=='__main__': build()
