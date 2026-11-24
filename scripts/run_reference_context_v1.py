"""Separate basis correction from context filtering on identical frozen source inputs."""
import argparse
import hashlib
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from cad_recovery.dataset_layout_v1 import checked,pin
from cad_recovery.real_chain_metrics_v1 import head_metrics,chain_metrics
from cad_recovery.open_reference_chains_v1 import contacts
from cad_recovery.stroke_roles_v1 import foreground
from capture_reference_context_inputs_v1 import CONTROLS,PREVIOUS
from qualify_reference_context_v1 import load,arms
from run_open_references_v1 import inputs as fresh_inputs,control_review,LABELS
from run_topology_robust_v1 import read,write,object_hash

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/reference-context-profile-v1'
BANK=ROOT/'artifacts/reference-context-source-bank-v1'
QA={'path':'artifacts/reference-context-qualification-v1/train-label-qa.json','sha256':'890feba89ae2dd646165e5eee9cab84d1eca4a8baf3fd996355024fc32381c1d'}
ARMS=('original_v1','basis_v2','context_v2')


def freeze():
    checked(ROOT,PREVIOUS);control=read(checked(ROOT,CONTROLS));checked(ROOT,QA)
    previous=read(ROOT/'artifacts/open-reference-profile-v1/plan.json')
    index={(r['case'],r['mode']):r for r in read(ROOT/'artifacts/open-reference-profile-v1/results-index.json') if not r['repeat']}
    jobs=[]
    for job in read(BANK/'previous-index.json')['jobs']:
        views=[]
        for view in job['views']:
            old=(ROOT/index[job['case']['id'],view['mode']]['artifact']).parent
            views.append({**view,'prior_original_chains':pin(ROOT,(old/'new-chains.json').relative_to(ROOT).as_posix()),
                          'prior_result':pin(ROOT,(old/'result.json').relative_to(ROOT).as_posix())})
        jobs.append({**job,'case':{**job['case'],'split':'development'},'views':views,'new_control':False})
    jobs+=read(BANK/'train-index.json')['jobs']
    jobs += [j for j in control['jobs'] if j['case']['split']=='validation']
    names=[p['path'] for p in previous['code']]+['src/cad_recovery/open_reference_chains_v2.py',
           'src/cad_recovery/open_reference_context_v1.py','scripts/run_reference_context_v1.py',
           'scripts/capture_reference_context_inputs_v1.py','scripts/qualify_reference_context_v1.py',
           'scripts/review_reference_context_labels_v1.py','scripts/prepare_reference_context_controls_v1.py',
           'scripts/audit_reference_context_invariance_v1.py','tests/test_reference_context_v1.py']
    write(BASE/'plan.json',{'task':'P10.2c.3b.2f.1','previous':PREVIOUS,'controls':CONTROLS,'labels':LABELS,'label_QA':QA,
          'jobs':jobs,'code':[pin(ROOT,n) for n in dict.fromkeys(names)],'arms':ARMS,'repeats':2,
          'criteria':control['criteria'],'budgets':control['budgets'],
          'qualification':pin(ROOT,'artifacts/reference-context-qualification-v1/report-v2.json'),
          'invariance':pin(ROOT,'artifacts/reference-context-qualification-v1/invariance.json'),
          'fresh_validation_predictions_before_freeze':False,'pixel_role_or_head_models_changed':False,
          'production_changed':False,'physical_geometry_approved':False})
    print(pin(ROOT,'artifacts/reference-context-profile-v1/plan.json'),flush=True)


def get_inputs(view):
    if view.get('bank'):
        gray,graph,base,heads,used=load(view)
        calls={'bank_graph_reads':1,'geometry_extractions':0,'advanced_head_calls':0,'baseline_profile_calls':0}
        return gray,graph,base,heads,used,calls
    gray,graph,base,heads,used,_,prep=fresh_inputs(view)
    return gray,graph,base,heads,used,{'bank_graph_reads':0,**{k:prep[k] for k in ('geometry_extractions','advanced_head_calls','baseline_profile_calls')}}


def child(path):
    cv2.setNumThreads(1);cv2.setRNGSeed(0)
    request=read(path);plan=read(checked(ROOT,request['plan']));job=request['job'];case=job['case']
    for p in plan['code']:checked(ROOT,p)
    out=path.parent/'output';out.mkdir();rows=[]
    for view in job['views']:
        gray,graph,base,heads,used,calls=get_inputs(view)
        snapshot=object_hash([graph,heads,base['dimension_chains']]);pixel_sha=hashlib.sha256(gray.tobytes()).hexdigest()
        role_sha=hashlib.sha256(base['pixel_roles'].tobytes()).hexdigest()
        results=arms(gray,graph,base,used)
        # Only completed model packets are passed to evaluation; no GT enters any arm.
        labels=read(checked(ROOT,plan['labels']));qa=read(checked(ROOT,plan['label_QA']))
        review=next((r for r in labels['records'] if r['id']==case['id']),None)
        if case.get('labels',{}).get('feet_px') is not None:review=control_review(case)
        usable=case['id'] not in qa['cases_with_presence_but_no_unique_foot']
        expected=case.get('labels',{}).get('expected_dimension_chains')
        hm=head_metrics(review,heads,view['input_to_native']) if review else {}
        trace={'contacts':{k:[h['id'] for h in v] for k,v in contacts(graph['tracks'],used,foreground(gray)).items()},'eligible_boundaries':[]}
        strict={};scored={};new_counts={};prefix={};unchanged={};audits={}
        folder=out/view['mode'];folder.mkdir()
        for arm,packet in results.items():
            new_counts[arm]=len(packet['open_reference_chains'])
            prefix[arm]=packet['dimension_chains'][:len(base['dimension_chains'])]==base['dimension_chains']
            unchanged[arm]=role_sha==hashlib.sha256(packet['pixel_roles'].tobytes()).hexdigest()
            metrics={}
            if review:
                for scope,p in [('all',packet),('new_only',{**packet,'dimension_chains':packet['open_reference_chains']})]:
                    metric=chain_metrics(review,heads,view['input_to_native'],p,graph,trace,hm)
                    metrics[scope]={k:metric[k] for k in ('tp','fp','fn','pairs','outside_scope_proposals_unscored','search_incomplete')}
            if expected==0:
                metrics={scope:{'tp':0,'fp':new_counts[arm]+(len(base['dimension_chains']) if scope=='all' else 0),
                         'fn':0,'pairs':[],'outside_scope_proposals_unscored':0,'search_incomplete':False} for scope in ('all','new_only')}
            strict[arm]=metrics
            scored[arm]=metrics if usable else {scope:{'tp':0,'fp':0,'fn':0,'pairs':[],
                       'outside_scope_proposals_unscored':new_counts[arm]+(len(base['dimension_chains']) if scope=='all' else 0),
                       'search_incomplete':False} for scope in ('all','new_only')}
            audits[arm]={'kernel':packet['open_reference_audit'],'junction':packet['open_reference_junction_audit'],
                         'context':packet.get('reference_context_audit'),'incomplete':packet['search_incomplete']}
            write(folder/(arm+'-chains.json'),packet['open_reference_chains'])
        write(folder/'context-decisions.json',results['context_v2']['reference_context_decisions'])
        reproduced=True
        if view.get('prior_original_chains'):
            reproduced=results['original_v1']['open_reference_chains']==read(checked(ROOT,view['prior_original_chains']))
        if not view.get('bank'):
            write(folder/'graph.json',graph);write(folder/'heads.json',heads)
            write(folder/'base.json',{k:v for k,v in base.items() if not isinstance(v,np.ndarray)})
            Image.fromarray(base['pixel_roles']).save(folder/'roles.png')
        row={k:case[k] for k in ('scope','group_id','split')}
        row.update(case=case['id'],mode=view['mode'],repeat=request['repeat'],new_control=job['new_control'],
          original_split=case.get('original_split',case['split']),expected_dimension_chains=expected,
          unique_foot_targets_admitted=usable,strict_chains=strict,chains=scored,heads=hm,new_chains=new_counts,
          previous_chain_count=len(base['dimension_chains']),prefix_unchanged=prefix,pixel_roles_unchanged=unchanged,
          original_v1_reproduced=reproduced,source_observations_unchanged=snapshot==object_hash([graph,heads,base['dimension_chains']]),
          source_pixels_unchanged=pixel_sha==hashlib.sha256(gray.tobytes()).hexdigest(),audits=audits,
          preparation=calls,production_changed=False,physical_geometry_approved=False,source_pixels_erased=0)
        write(folder/'result.json',row);rows.append(row)
    write(out/'result.json',{'rows':rows})


def run(sha):
    from experiment_children_v1 import run as execute
    execution=execute(ROOT,BASE,{'path':'artifacts/reference-context-profile-v1/plan.json','sha256':sha},Path(__file__).resolve())
    rows=[read(ROOT/r['artifact']) for r in read(BASE/'results-index.json')]
    execution.update(preparation_calls={k:sum(r['preparation'][k] for r in rows) for k in rows[0]['preparation']},
                     calls_per_arm={a:len(rows) for a in ARMS},new_OCR_ML_calls=0,new_DXF_exports=0,training=False,closed_test_used=False)
    write(BASE/'execution.json',execution);print(execution,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','run','child']);p.add_argument('--sha256');p.add_argument('--request',type=Path)
    a=p.parse_args()
    if a.mode=='freeze':freeze()
    elif a.mode=='child':child(a.request)
    elif a.sha256:run(a.sha256)
    else:p.error('--sha256 required')
