"""Full-corpus advanced-head/source-route comparison with a fixed junction guard."""
import argparse
import hashlib
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
from cad_recovery.dataset_layout_v1 import checked,pin
from cad_recovery.source_chain_profile_v2 import analyze,diagnostic_heads
from cad_recovery.head_junction_ownership_v1 import apply,POLICY
from cad_recovery.stroke_role_metrics_v1 import to_native,synthetic_truth,head_scores
from cad_recovery.stroke_ownership_metrics_v1 import score_roles,ink_ledger
from cad_recovery.stroke_roles_v1 import foreground
from head_junction_inputs_v1 import prepare
from run_topology_robust_v1 import read,write,object_hash

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/head-junction-profile-v1'
PREVIOUS={'path':'artifacts/source-chain-release-v2.json','sha256':'8cfaa36354918ab30fcbe06d1380db467a137f8e4001d0d1da304ec873af11e6'}
CONTROLS={'path':'artifacts/head-junction-controls-v1/contract.json','sha256':'d5e9135415a1a4af4649bb9e276bdb0a169dd409e56db6f7c1015cb815a1fd3e'}


def freeze():
    checked(ROOT,PREVIOUS);controls=read(checked(ROOT,CONTROLS));checked(ROOT,controls['generator'])
    previous=read(ROOT/'artifacts/source-chain-profile-v2/plan-final.json')
    index={(r['case'],r['mode']):r for r in read(ROOT/'artifacts/source-chain-profile-v2/results-index.json') if not r['repeat']}
    jobs=[]
    for job in previous['jobs']:
        views=[]
        for view in job['views']:
            folder=(ROOT/index[job['case']['id'],view['mode']]['artifact']).parent
            prior={n:pin(ROOT,(folder/file).relative_to(ROOT).as_posix()) for n,file in [('roles','routes-roles.png'),('row','result.json')]}
            views.append({**view,'prior_routes':prior})
        jobs.append({**job,'case':{**job['case'],'previous_split':job['case']['split'],'split':'development'},'views':views,'new_control':False})
    jobs+=controls['jobs']
    names=['src/cad_recovery/head_junction_ownership_v1.py','scripts/head_junction_inputs_v1.py',
           'scripts/run_head_junction_v1.py','scripts/prepare_head_junction_controls_v1.py',
           'tests/test_head_junction_ownership_v1.py']+[p['path'] for p in previous['code']]
    write(BASE/'plan.json',{'task':'P10.2c.3b.2d','previous':PREVIOUS,'controls':CONTROLS,
          'code':[pin(ROOT,n) for n in dict.fromkeys(names)],'jobs':jobs,'policy':POLICY,'repeats':2,'budgets':controls['budgets'],
          'qualification':pin(ROOT,'artifacts/head-junction-qualification-v1/report.json'),
          'known_errors':pin(ROOT,'artifacts/source-chain-profile-v2/joint-error-diagnosis.json'),
          'new_validation_opened_before_freeze':False,'old_validation_reclassified_as_development':True,
          'all_views_use_same_joint_profile':True,'inherited_roles':'complete frozen routes map from P10.2c.3b.2c',
          'criteria':controls['criteria'],'full_role_macro_F1_min':.90,'production_changed':False,'physical_geometry_approved':False})
    print(pin(ROOT,'artifacts/head-junction-profile-v1/plan.json'),flush=True)


def child(path):
    cv2.setNumThreads(1);cv2.setRNGSeed(0)
    request=read(path);plan=read(checked(ROOT,request['plan']));job=request['job'];case=job['case']
    for record in plan['code']:checked(ROOT,record)
    out=path.parent/'output';out.mkdir();rows=[];truth=negative=native_gray=None
    for view in job['views']:
        gray,source,geometry,inherited,previous,local,heads,extra,calls=prepare(ROOT,view)
        if native_gray is None:native_gray=gray.copy()
        source_before=object_hash([source,geometry,previous,heads,extra]);gray_sha=hashlib.sha256(gray.tobytes()).hexdigest()
        map_sha=hashlib.sha256(inherited.tobytes()).hexdigest()
        started=time.perf_counter();advanced=diagnostic_heads(gray,heads)
        raw=analyze(gray,previous,inherited,local['local']['text_mask'],advanced['candidates'],diagnostic_heads=extra)
        observations=object_hash([advanced,raw['dimension_chains']])
        guarded=apply(gray,previous,inherited,raw,advanced['candidates'])
        seconds=time.perf_counter()-started
        # Every inference stage has finished before any target is opened.
        if truth is None:
            if job.get('truth'):truth=np.array(Image.open(checked(ROOT,job['truth'])))
            elif case['scope']=='analytical_control':truth=synthetic_truth(case,native_gray,np.array(Image.open(checked(ROOT,view['reviewed_mask']))))
        if negative is None and job.get('negative'):negative=np.array(Image.open(checked(ROOT,job['negative'])))
        if truth is None and negative is not None:truth=np.full(tuple(case['size_px'][::-1]),255,np.uint8)
        folder=out/view['mode'];folder.mkdir();scores={};ledgers={};additions={};old=to_native(inherited,case['size_px'])
        for name,labels in [('routes',inherited),('joint',raw['pixel_roles']),('guarded',guarded['pixel_roles'])]:
            Image.fromarray(labels).save(folder/(name+'-roles.png'));native=to_native(labels,case['size_px']);delta=native!=old
            known=delta&(truth>0)&(truth<10) if truth is not None else np.zeros_like(delta)
            scores[name]=score_roles(truth,native,negative) if truth is not None else None
            ledgers[name]=ink_ledger(gray,labels,foreground)
            additions[name]={'correct':int((known&(native==truth)).sum()) if truth is not None else 0,
                             'wrong':int((known&(native!=truth)).sum()) if truth is not None else 0,
                             'unknown_truth':int((delta&(truth==255)).sum()) if truth is not None else int(delta.sum()),
                             'negative':int((delta&(negative>0)&(truth==255)).sum()) if negative is not None else 0}
        prefix_scores_match=True;inherited_scores_match=True;head_metrics=None
        if 'arrows' in case.get('labels',{}):
            head_metrics={name:head_scores(case['labels']['arrows'],[],hh,view['input_to_native'],dense=True)
                          for name,hh in [('legacy',heads),('advanced',advanced['candidates'])]}
        if view.get('prior_routes'):
            prior=read(checked(ROOT,view['prior_routes']['row']))
            inherited_scores_match=scores['routes']==prior['roles']['routes']
            if prior['head_metrics'] is not None:prefix_scores_match=head_metrics['advanced']==prior['head_metrics']['advanced']
        Image.fromarray(guarded['pixel_interval_owner']).save(folder/'pixel-interval-owner.png')
        Image.fromarray(guarded['junction_abstention_mask']).save(folder/'junction-abstention.png')
        for name,labels in [('joint',raw['pixel_roles']),('guarded',guarded['pixel_roles'])]:
            image=np.repeat(gray[:,:,None],3,axis=2)
            for cls,color in [(1,[25,100,230]),(2,[225,110,20]),(3,[160,55,180])]:image[(labels!=inherited)&(labels==cls)]=color
            Image.fromarray(image).save(folder/(name+'-overlay.png'))
        write(folder/'heads.json',advanced);write(folder/'source-intervals.json',previous['proposals'])
        row={k:case[k] for k in ('scope','group_id','split')}
        row.update(case=case['id'],mode=view['mode'],repeat=request['repeat'],new_control=job['new_control'],
             roles=scores,ink_ledgers=ledgers,additions=additions,head_metrics=head_metrics,
             dimension_chains=raw['dimension_chains'],expected_dimension_chains=case.get('labels',{}).get('expected_dimension_chains'),
             details={k:v for k,v in guarded.items() if k not in ('pixel_roles','pixel_interval_owner','junction_abstention_mask')},
             raw_search_incomplete=raw['search_incomplete'],head_search_incomplete=advanced['search_incomplete'],
             inherited_scores_match=inherited_scores_match,previous_advanced_head_scores_match=prefix_scores_match,
             source_observations_unchanged=source_before==object_hash([source,geometry,previous,heads,extra]),
             head_chain_observations_unchanged=observations==object_hash([advanced,guarded['dimension_chains']]),
             input_pixel_sha256=gray_sha,source_pixel_sha256_after=hashlib.sha256(gray.tobytes()).hexdigest(),
             inherited_map_unchanged=map_sha==hashlib.sha256(inherited.tobytes()).hexdigest(),
             head_prefix_unchanged=advanced['prefix_unchanged'],preparation=calls,timing={'joint_and_guard_seconds':seconds},
             production_admitted=False,physical_geometry_approved=False,source_pixels_erased=0)
        write(folder/'result.json',row);rows.append(row)
    write(out/'result.json',{'rows':rows})


def run(sha):
    from experiment_children_v1 import run as execute
    execution=execute(ROOT,BASE,{'path':'artifacts/head-junction-profile-v1/plan.json','sha256':sha},Path(__file__).resolve())
    rows=[read(ROOT/r['artifact']) for r in read(BASE/'results-index.json')]
    execution.update(preparation_calls={k:sum(r['preparation'][k] for r in rows) for k in rows[0]['preparation']},
           advanced_head_calls=len(rows),joint_profile_calls=len(rows),junction_guard_calls=len(rows),
           new_OCR_ML_calls=0,new_DXF_exports=0,training=False,closed_test_used=False,production_changed=False)
    write(BASE/'execution.json',execution)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['freeze','run','child']);parser.add_argument('--sha256');parser.add_argument('--request',type=Path)
    args=parser.parse_args()
    if args.mode=='freeze':freeze()
    elif args.mode=='child':child(args.request)
    elif args.sha256:run(args.sha256)
    else:parser.error('--sha256 required')
