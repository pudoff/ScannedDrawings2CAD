"""Prospective source-chain experiment with an independently scoped head ablation."""
import argparse
import hashlib
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

from cad_recovery.dataset_layout_v1 import checked,pin
from cad_recovery.source_chain_profile_v2 import analyze,diagnostic_heads,head_only_profile
from cad_recovery.source_chain_routes_v1 import POLICY
from cad_recovery.stroke_role_metrics_v1 import to_native,synthetic_truth,head_scores
from cad_recovery.stroke_ownership_metrics_v1 import score_roles,ink_ledger
from cad_recovery.stroke_roles_v1 import foreground
from source_chain_inputs_v1 import prepare
from run_topology_robust_v1 import read,write,object_hash

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/source-chain-profile-v2'
PREVIOUS={'path':'artifacts/boundary-role-release-v2.json','sha256':'c339f9f65065cc09d2854c52e8d9ceedeaced87766840cfb345b912ac7cb3ab3'}
CONTROLS={'path':'artifacts/source-chain-controls-v1/contract.json','sha256':'0d2a7e19c102ba3c5f1f599057c3106261fb9622137de902601a12f0c8d56a46'}


def freeze():
    checked(ROOT,PREVIOUS);initial=read(ROOT/'artifacts/source-chain-profile-v1/plan-final.json')
    names=[p['path'] for p in initial['code']]+['src/cad_recovery/local_role_guard_v3.py',
          'src/cad_recovery/source_chain_profile_v2.py','scripts/run_source_chains_v2.py','tests/test_source_chain_import_fix_v1.py']
    initial.update(code=[pin(ROOT,n) for n in dict.fromkeys(names)],
          import_fix_certificate=pin(ROOT,'artifacts/source-chain-profile-v1/import-fix-certificate.json'),
          initial_prospective_plan=pin(ROOT,'artifacts/source-chain-profile-v1/plan-final.json'),
          new_validation_opened_before_freeze=True,initial_validation_was_prospective=True,
          parameter_or_rule_change=False,reason='Import-only repair; function ASTs identical; completed masks must reproduce')
    write(BASE/'plan-final.json',initial)
    print(pin(ROOT,'artifacts/source-chain-profile-v2/plan-final.json'),flush=True)


def child(path):
    cv2.setNumThreads(1);cv2.setRNGSeed(0);req=read(path);plan=read(checked(ROOT,req['plan']));job=req['job'];case=job['case']
    for p in plan['code']:checked(ROOT,p)
    out=path.parent/'output';out.mkdir();rows=[];truth=negative=native_gray=None
    for view in job['views']:
        gray,source,geometry,inherited,previous,local,heads,extra,calls=prepare(ROOT,view)
        if native_gray is None:native_gray=gray.copy()
        before=object_hash([source,geometry,previous,heads,extra]);sha=hashlib.sha256(gray.tobytes()).hexdigest()
        inherited_sha=hashlib.sha256(inherited.tobytes()).hexdigest()
        started=time.perf_counter();candidate=analyze(gray,previous,inherited,local['local']['text_mask'],heads,diagnostic_heads=extra)
        variants={'previous':inherited,'routes':candidate['pixel_roles']}
        chains={'previous':local['hybrid']['dimension_chains'],'routes':candidate['dimension_chains']}
        newer=head_only=joint=None
        if job['head_ablation']:
            newer=diagnostic_heads(gray,heads)
            head_only=head_only_profile(gray,previous,inherited,local['local']['text_mask'],newer['candidates'],diagnostic_heads=extra)
            joint=analyze(gray,previous,inherited,local['local']['text_mask'],newer['candidates'],diagnostic_heads=extra)
            variants.update(head_only=head_only['pixel_roles'],joint=joint['pixel_roles'])
            chains.update(head_only=head_only['dimension_chains'],joint=joint['dimension_chains'])
        elapsed=time.perf_counter()-started
        # Truth and reference head locations are consumed after every inference arm.
        if truth is None:
            if job.get('truth'):truth=np.array(Image.open(checked(ROOT,job['truth'])))
            elif case['scope']=='analytical_control':truth=synthetic_truth(case,native_gray,np.array(Image.open(checked(ROOT,view['reviewed_mask']))))
        if negative is None and job.get('negative'):negative=np.array(Image.open(checked(ROOT,job['negative'])))
        if truth is None and negative is not None:truth=np.full(tuple(case['size_px'][::-1]),255,np.uint8)
        folder=out/view['mode'];folder.mkdir();scores={};ledgers={};added={}
        old=to_native(inherited,case['size_px'])
        for name,labels in variants.items():
            Image.fromarray(labels).save(folder/(name+'-roles.png'));native=to_native(labels,case['size_px']);delta=native!=old
            known=delta&(truth>0)&(truth<10) if truth is not None else np.zeros_like(delta)
            added[name]={'correct':int((known&(native==truth)).sum()) if truth is not None else 0,
                         'wrong':int((known&(native!=truth)).sum()) if truth is not None else 0,
                         'unknown_truth':int((delta&(truth==255)).sum()) if truth is not None else int(delta.sum()),
                         'negative':int((delta&(negative>0)&(truth==255)).sum()) if negative is not None else 0}
            scores[name]=score_roles(truth,native,negative) if truth is not None else None
            ledgers[name]=ink_ledger(gray,labels,foreground)
        baseline_matches=True
        if view.get('prior_chain_profile'):
            prior=read(checked(ROOT,view['prior_chain_profile']['row']))
            baseline_matches=scores['previous']==prior['roles']['boundary']
        metrics=None
        if newer is not None and 'arrows' in case.get('labels',{}):
            metrics={name:head_scores(case['labels']['arrows'],[],hs,view['input_to_native'],dense=True)
                     for name,hs in [('legacy',heads),('advanced',newer['candidates'])]}
        Image.fromarray(candidate['pixel_interval_owner']).save(folder/'pixel-interval-owner.png')
        overlay=cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB);overlay[candidate['pixel_roles']!=inherited]=[20,100,230]
        Image.fromarray(overlay).save(folder/'routes-overlay.png')
        write(folder/'source-intervals.json',previous['proposals'])
        row={k:case[k] for k in ('scope','group_id','split')}
        row.update(case=case['id'],mode=view['mode'],repeat=req['repeat'],new_control=job['new_control'],head_ablation=job['head_ablation'],
             roles=scores,ink_ledgers=ledgers,additions=added,chains=chains,head_metrics=metrics,
             primary_details={k:v for k,v in candidate.items() if k not in ('pixel_roles','pixel_interval_owner')},
             joint_details={k:v for k,v in joint.items() if k not in ('pixel_roles','pixel_interval_owner')} if joint else None,
             diagnostic_head_analysis=newer['analysis'] if newer else None,
             new_head_prefix_unchanged=newer['prefix_unchanged'] if newer else True,
             head_search_incomplete=newer['search_incomplete'] if newer else False,
             head_only_incomplete=head_only['search_incomplete'] if head_only else False,
             input_pixel_sha256=sha,source_pixel_sha256_after=hashlib.sha256(gray.tobytes()).hexdigest(),
             source_geometry_heads_unchanged=before==object_hash([source,geometry,previous,heads,extra]),
             inherited_map_unchanged=inherited_sha==hashlib.sha256(inherited.tobytes()).hexdigest(),
             inherited_known_roles_unchanged=bool(np.array_equal(candidate['pixel_roles'][inherited!=255],inherited[inherited!=255])),
             previous_scores_match=baseline_matches,expected_dimension_chains=case.get('labels',{}).get('expected_dimension_chains'),
             preparation=calls,timing={'all_inference_arms_seconds':elapsed},
             production_admitted=False,physical_geometry_approved=False,source_pixels_erased=0)
        write(folder/'result.json',row);rows.append(row)
    write(out/'result.json',{'rows':rows})


def run(sha):
    from experiment_children_v1 import run as execute
    result=execute(ROOT,BASE,{'path':'artifacts/source-chain-profile-v2/plan-final.json','sha256':sha},Path(__file__).resolve())
    rows=[read(ROOT/r['artifact']) for r in read(BASE/'results-index.json')]
    result.update(preparation_calls={k:sum(r['preparation'][k] for r in rows) for k in rows[0]['preparation']},
         primary_profile_calls=len(rows),head_ablation_calls=sum(r['head_ablation'] for r in rows),
         additional_joint_profile_calls=sum(r['head_ablation'] for r in rows),
         new_OCR_ML_calls=0,new_DXF_exports=0,training=False,closed_test_used=False,production_changed=False)
    write(BASE/'execution.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['freeze','run','child']);parser.add_argument('--sha256');parser.add_argument('--request',type=Path)
    args=parser.parse_args()
    if args.mode=='freeze':freeze()
    elif args.mode=='child':child(args.request)
    elif args.sha256:run(args.sha256)
    else:parser.error('--sha256 required')
