"""Qualify only analytical shapes in the isolated scikit-image target directory."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'artifacts/cv-skeleton-runtime-v1/deps'))
import importlib.metadata
import json
import cv2
import numpy as np
from skimage.morphology import skeletonize,medial_axis
from cad_recovery.dataset_layout_v1 import pin,encoded


def topology(mask):
    ink=(mask>0).astype(np.uint8)
    components=cv2.connectedComponents(ink,connectivity=8)[0]-1
    inverse=np.pad(1-ink,1,constant_values=1)
    holes=cv2.connectedComponents(inverse,connectivity=4)[0]-2
    return {'components':components,'holes':holes}


if __name__=='__main__':
    output=ROOT/'artifacts/cv-skeleton-runtime-v1/qualification.json'
    if output.exists():raise FileExistsError(output)
    shapes={}
    bar=np.zeros((80,80),np.uint8);bar[10:70,34:46]=1;shapes['bar']=bar
    ring=np.zeros_like(bar);cv2.circle(ring,(40,40),25,1,7);shapes['ring']=ring
    cross=np.zeros_like(bar);cross[10:70,35:44]=1;cross[35:44,10:70]=1;shapes['cross']=cross
    checks=[]
    for method in ('zhang','medial'):
        for name,ink in shapes.items():
            result=skeletonize(ink,method='zhang') if method=='zhang' else medial_axis(ink,rng=0)
            before,after=topology(ink),topology(result)
            subset=not np.any(result & ~ink.astype(bool))
            checks.append({'method':method,'shape':name,'input_topology':before,'topology':after,
                'subset_of_source':bool(subset),'passed':bool(before==after and subset)})
    packages={p:importlib.metadata.version(p) for p in ('scikit-image','scipy','numpy','opencv-python','Pillow','networkx','imageio','tifffile','lazy-loader','packaging')}
    wheels=[pin(ROOT,p.relative_to(ROOT).as_posix()) for p in sorted((ROOT/'artifacts/cv-skeleton-runtime-v1/wheels').glob('*.whl'))]
    admitted={method:all(c['passed'] for c in checks if c['method']==method) for method in ('zhang','medial')}
    value={'status':'qualified_with_method_exclusion','packages':packages,'wheels':wheels,'checks':checks,
        'method_admission':admitted,'excluded_methods':'Methods failing topology are not run on the comparison corpus',
        'qualification_only':True,'closed_test_used':False,'production_environment_changed':False,
        'code':pin(ROOT,'scripts/qualify_skeleton_runtime_v1.py')}
    with output.open('xb') as stream:stream.write(encoded(value))
    print(json.dumps({'qualification':pin(ROOT,output.relative_to(ROOT).as_posix()),'checks':len(checks),'packages':packages}))
