"""Owner-map serialization correction; the preregistered CV model is unchanged."""
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from run_grid_role_v1 import ROOT,QUAL,CONTROLS,ARMS,load,truth_for,revision_score
from cad_recovery.dataset_layout_v1 import pin,checked
from cad_recovery.grid_role_revision_v1 import analyze,sha
from cad_recovery.joint_role_metrics_v1 import score
from cad_recovery.stroke_role_metrics_v1 import to_native
from cad_recovery.stroke_ownership_metrics_v1 import ink_ledger
from cad_recovery.stroke_roles_v1 import foreground
from run_topology_robust_v1 import read,write,object_hash

BASE=ROOT/'artifacts/grid-role-profile-v2'
INITIAL={'path':'artifacts/grid-role-profile-v1/plan.json','sha256':'e29720460bc26ab35e5d9fd003e5f9290f06f7948b95087f2c2d762cfd36d5c1'}


def save_owner(path,owner,maximum):
    if owner.dtype.kind not in 'iu' or owner.size==0 or owner.min()<0 or owner.max()>maximum or maximum>65535:
        raise ValueError('Owner IDs must fit the declared source interval domain and uint16')
    Image.fromarray(owner.astype(np.uint16)).save(path)
    return bool(np.array_equal(np.array(Image.open(path)),owner))


def freeze():
    previous=read(checked(ROOT,INITIAL))
    for p in previous['code']:checked(ROOT,p)
    checked(ROOT,read(QUAL/'report.json')['model']);BASE.mkdir(exist_ok=False)
    write(BASE/'plan.json',{**previous,'original_attempt':INITIAL,
        'code':previous['code']+[pin(ROOT,n) for n in ('scripts/run_grid_role_v2.py','tests/test_grid_owner_serialization_v1.py')],
        'serialization_correction':'int32 owner PNG replaced by checked lossless uint16; model and numeric policies unchanged',
        'fresh_validation_predictions_before_freeze':False})
    print(pin(ROOT,'artifacts/grid-role-profile-v2/plan.json'),flush=True)


def child(path):
    cv2.setNumThreads(1);cv2.setRNGSeed(0);request=read(path);plan=read(checked(ROOT,request['plan']))
    for p in plan['code']:checked(ROOT,p)
    j=request['job'];case=j['case'];out=path.parent/'output';out.mkdir();rows=[];native=None
    for v in j['views']:
        folder=out/v['mode'];folder.mkdir()
        gray,graph,roles,heads,links,record,authority,calls=load(j,v,folder/'baseline-source')
        before=object_hash([graph,heads,links]);gray_sha=sha(gray);role_sha=sha(roles);mask_sha=sha(authority['mask'])
        result=analyze(gray,graph,roles,authority=authority)
        if native is None:native=gray.copy()
        truth,negative=truth_for(j,native);metrics={};revisions={};observed={};protected=authority['mask']==2
        initial_exact=True;initial_count=0
        for a in ARMS:
            labels=result[a];image=to_native(labels,case['size_px']);metrics[a]=score(truth,image,negative) if truth is not None else None
            revisions[a]=revision_score(j,truth,negative,roles,labels);delta=labels!=roles
            observed[a]={'only_hatching_to_unknown':bool(np.all((roles[delta]==9)&(labels[delta]==255))),
                'protected_pixels_exact':bool(np.array_equal(labels[protected],roles[protected])),'assigned_new_classes':False}
            Image.fromarray(labels).save(folder/(a+'-roles.png'))
            old=ROOT/'artifacts/grid-role-profile-v1/runs'/(case['id']+'-r0')/'output'/v['mode']/(a+'-roles.png')
            if old.exists():initial_count+=1;initial_exact&=np.array_equal(np.array(Image.open(old)),labels)
        owner_exact=save_owner(folder/'revision-owner.png',result['revision_owner'],len(graph['proposals']))
        Image.fromarray(authority['mask']).save(folder/'authority.png')
        write(folder/'authority.json',{k:z for k,z in authority.items() if k!='mask'});write(folder/'decisions.json',result['decisions'])
        prior=read(checked(ROOT,record['prior_row'])) if record.get('prior_row') else None
        rows.append({**{k:case[k] for k in ('scope','group_id','split')},'case':case['id'],'mode':v['mode'],'repeat':request['repeat'],
            'new_control':j['new_control'],'roles':metrics,'revisions':revisions,'view_checks':observed,
            'source_graph_heads_and_links_exact':before==object_hash([graph,heads,links]),'gray_exact':gray_sha==sha(gray),
            'historical_role_map_exact':role_sha==sha(roles) and np.array_equal(result['baseline'],roles),'authority_mask_exact':mask_sha==sha(authority['mask']),
            'old_role_metrics_exact':not prior or metrics['baseline']==prior['roles']['joint'],'owner_png_exact':owner_exact,
            'initial_partial_train_maps_exact':bool(initial_exact),'initial_partial_train_maps_compared':initial_count,
            'ink_ledgers':{a:ink_ledger(gray,result[a],foreground) for a in ARMS},
            'authority':{'source_sha256':gray_sha,'role_sha256':role_sha,'mask_sha256':mask_sha,'origin':'frozen_cv_inference',
                'protected_pixels':int(protected.sum()),'GT_used_for_authority':False},
            'complete':result['complete'],'audit':result['audit'],'preparation':calls,
            'production_changed':False,'physical_geometry_approved':False,'source_pixels_erased':0})
        write(folder/'result.json',rows[-1])
    write(out/'result.json',{'rows':rows})


def run(sha256):
    from experiment_children_v1 import run as execute
    e=execute(ROOT,BASE,{'path':'artifacts/grid-role-profile-v2/plan.json','sha256':sha256},Path(__file__).resolve())
    rows=[read(ROOT/r['artifact']) for r in read(BASE/'results-index.json')]
    failed=read(ROOT/'artifacts/grid-role-profile-v1/runs/grid-role-ruled_title-w3-r21-r0/process.json')
    e.update(revision_calls=len(rows),preparation_calls={k:sum(r['preparation'][k] for r in rows) for k in rows[0]['preparation']},
        initial_failed_children=1,initial_failed_child_seconds=failed['walltime_seconds'],
        new_OCR_ML_calls=0,new_DXF_exports=0,training=False,closed_test_used=False)
    write(BASE/'execution.json',e);print(e,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','run','child']);p.add_argument('--sha256');p.add_argument('--request',type=Path)
    a=p.parse_args()
    if a.mode=='freeze':freeze()
    elif a.mode=='child':child(a.request)
    elif a.sha256:run(a.sha256)
    else:p.error('--sha256 required')
