"""Read-only verification of the full P15.2f.2 contract before later CV runs."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform

from cad_recovery.cv_comparison_protocol_v1 import verify
from cad_recovery.dataset_layout_v1 import checked, read
from cad_recovery.family_catalog_v2 import require

ROOT = Path(__file__).resolve().parents[1]


def verify_contract(root, record):
    contract=read(checked(root,record))
    require(contract['prepared_before_predictions'] is True, 'Not preregistered')
    for key in ('full_page_plans','runtime','metric_definitions','summary'):
        checked(root,contract[key])
    crop=contract['crop_protocol']; result=verify(root,checked(root,crop).parent,crop['sha256'])
    runtime=read(checked(root,contract['runtime']))
    require(platform.python_version()==runtime['python'] and platform.system()==runtime['platform'], 'Runtime changed')
    for package,expected in runtime['packages'].items():
        try:actual=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:actual=None
        require(actual==expected,'Dependency changed: '+package)
    for pin in runtime['baseline_local_dependency_closure']: checked(root,pin)
    plans=read(checked(root,contract['full_page_plans']))['plans']
    for plan in plans:
        checked(root,plan['source']); checked(root,plan['preparation'])
        checked(root,plan['reduced_whole_sheet']['image'])
        require(not plan['screening']['blocked'], 'Blocked full-page plan')
        width,height=plan['size_px']; windows=plan['native_tiled']['windows']
        xs=sorted({(b[0],b[2]) for b in windows}); ys=sorted({(b[1],b[3]) for b in windows})
        require(len(windows)==len(xs)*len(ys) and len(windows)<=128,'Incomplete window grid')
        for intervals,end in ((xs,width),(ys,height)):
            require(intervals[0][0]==0 and intervals[-1][1]==end,'Incomplete source coverage')
            require(all(b[0]<=a[1] for a,b in zip(intervals,intervals[1:])),'Window gap')
        require(all(0<=x<right<=width and 0<=y<bottom<=height and (right-x)*(bottom-y)<=1024**2
                    for x,y,right,bottom in windows),'Unbounded window')
    return {**result,'full_page_plans':len(plans),'runtime_pins_verified':len(runtime['baseline_local_dependency_closure']),
        'parallel_skimage_available':runtime['packages']['scikit-image'] is not None,'inference_executed':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract',default='artifacts/cv-comparison-protocol-v1/contract.json')
    parser.add_argument('--sha256',required=True)
    args=parser.parse_args()
    print(json.dumps(verify_contract(ROOT,{'path':args.contract,'sha256':args.sha256}),ensure_ascii=False))
