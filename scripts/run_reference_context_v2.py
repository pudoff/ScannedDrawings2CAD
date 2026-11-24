"""Four-arm prospective context refinement; old validation is development."""
import argparse
import hashlib
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from cad_recovery.dataset_layout_v1 import pin,checked
from cad_recovery.real_chain_metrics_v1 import head_metrics,chain_metrics
from cad_recovery.open_reference_chains_v1 import contacts
from cad_recovery.stroke_roles_v1 import foreground
from run_open_references_v1 import control_review,LABELS
from run_reference_context_v1 import get_inputs,QA
from qualify_reference_context_v2 import arms,CONTROLS
from run_topology_robust_v1 import read,write,object_hash

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/reference-context-profile-v2'
PREVIOUS={'path':'artifacts/reference-context-release-v1.json','sha256':'0912f23a976badd0da1d9946c77a1d16ee0dc660d86ceaab106038015ff4aa53'}
ARMS=('original_v1','basis_v2','context_v1','context_v2')


def freeze():
    checked(ROOT,PREVIOUS);initial=read(ROOT/'artifacts/reference-context-profile-v1/plan.json')
    for p in initial['code']:checked(ROOT,p)
    index={(r['case'],r['mode']):r for r in read(ROOT/'artifacts/reference-context-profile-v1/results-index.json') if not r['repeat']}
    jobs=[]
    for job in initial['jobs']:
        views=[]
        for view in job['views']:
            folder=(ROOT/index[job['case']['id'],view['mode']]['artifact']).parent
            if not view.get('bank'):
                base=read(folder/'base.json')
                view={**view,'bank':{k:pin(ROOT,(folder/file).relative_to(ROOT).as_posix()) for k,file in
                       [('graph','graph.json'),('heads','heads.json'),('base','base.json'),('roles','roles.png')]},
                       'used_head_ids':base['head_ids_used']}
            prior={a:pin(ROOT,(folder/(file+'-chains.json')).relative_to(ROOT).as_posix()) for a,file in
                   [('original_v1','original_v1'),('basis_v2','basis_v2'),('context_v1','context_v2')]}
            views.append({**view,'initial_chains':prior,'initial_result':pin(ROOT,(folder/'result.json').relative_to(ROOT).as_posix())})
        jobs.append({**job,'case':{**job['case'],'split':'development'},'views':views,'new_control':False,
            'initial_fresh_validation':job['new_control'] and job['case']['split']=='validation'})
    jobs+=read(ROOT/'artifacts/reference-context-refinement-qualification-v1/train-index.json')['jobs']
    control=read(checked(ROOT,CONTROLS));jobs += [j for j in control['jobs'] if j['case']['split']=='validation']
    names=[p['path'] for p in initial['code']]+['scripts/run_reference_context_v2.py','scripts/qualify_reference_context_v2.py',
        'scripts/prepare_reference_strip_controls_v1.py','src/cad_recovery/open_reference_context_v2.py','tests/test_reference_context_v2.py']
    write(BASE/'plan.json',{'task':'P10.2c.3b.2f.1','previous':PREVIOUS,'labels':LABELS,'label_QA':QA,
        'controls':CONTROLS,'initial_plan':pin(ROOT,'artifacts/reference-context-profile-v1/plan.json'),
        'qualification':pin(ROOT,'artifacts/reference-context-refinement-qualification-v1/report.json'),
        'invariance':initial['invariance'],'jobs':jobs,'code':[pin(ROOT,n) for n in dict.fromkeys(names)],
        'arms':ARMS,'repeats':2,'criteria':control['criteria'],'budgets':initial['budgets'],
        'initial_validation_is_development':True,'fresh_validation_predictions_before_freeze':False,
        'production_changed':False,'physical_geometry_approved':False})
    print(pin(ROOT,'artifacts/reference-context-profile-v2/plan.json'),flush=True)


def child(path):
    cv2.setNumThreads(1);cv2.setRNGSeed(0);request=read(path);plan=read(checked(ROOT,request['plan']))
    for p in plan['code']:checked(ROOT,p)
    job=request['job'];case=job['case'];out=path.parent/'output';out.mkdir();rows=[]
    for view in job['views']:
        gray,graph,base,heads,used,calls=get_inputs(view)
        before=object_hash([graph,heads,base['dimension_chains']]);pixels=hashlib.sha256(gray.tobytes()).hexdigest()
        roles=hashlib.sha256(base['pixel_roles'].tobytes()).hexdigest();results=arms(gray,graph,base,used)
        # Model inference above takes no targets. Matching below is evaluation only.
        review=next((r for r in read(checked(ROOT,plan['labels']))['records'] if r['id']==case['id']),None)
        if case.get('labels',{}).get('feet_px') is not None:review=control_review(case)
        usable=case['id'] not in read(checked(ROOT,plan['label_QA']))['cases_with_presence_but_no_unique_foot']
        expected=case.get('labels',{}).get('expected_dimension_chains');hm=head_metrics(review,heads,view['input_to_native']) if review else {}
        trace={'contacts':{k:[h['id'] for h in v] for k,v in contacts(graph['tracks'],used,foreground(gray)).items()},'eligible_boundaries':[]}
        strict={};scored={};counts={};prefix={};unchanged={};audits={};reproduced={}
        folder=out/view['mode'];folder.mkdir()
        for arm,packet in results.items():
            counts[arm]=len(packet['open_reference_chains'])
            prefix[arm]=packet['dimension_chains'][:len(base['dimension_chains'])]==base['dimension_chains']
            unchanged[arm]=roles==hashlib.sha256(packet['pixel_roles'].tobytes()).hexdigest();metrics={}
            if review:
                for scope,p in [('all',packet),('new_only',{**packet,'dimension_chains':packet['open_reference_chains']})]:
                    m=chain_metrics(review,heads,view['input_to_native'],p,graph,trace,hm)
                    metrics[scope]={k:m[k] for k in ('tp','fp','fn','pairs','outside_scope_proposals_unscored','search_incomplete')}
            if expected==0:
                metrics={scope:{'tp':0,'fp':counts[arm]+(len(base['dimension_chains']) if scope=='all' else 0),
                    'fn':0,'pairs':[],'outside_scope_proposals_unscored':0,'search_incomplete':False} for scope in ('all','new_only')}
            strict[arm]=metrics
            scored[arm]=metrics if usable else {scope:{'tp':0,'fp':0,'fn':0,'pairs':[],
                'outside_scope_proposals_unscored':counts[arm]+(len(base['dimension_chains']) if scope=='all' else 0),
                'search_incomplete':False} for scope in ('all','new_only')}
            audits[arm]={'kernel':packet['open_reference_audit'],'junction':packet['open_reference_junction_audit'],
                'context':packet.get('reference_context_audit'),'part_structure':packet.get('part_structure_audit'),
                'incomplete':packet['search_incomplete']}
            if arm in view.get('initial_chains',{}):
                reproduced[arm]=packet['open_reference_chains']==read(checked(ROOT,view['initial_chains'][arm]))
            write(folder/(arm+'-chains.json'),packet['open_reference_chains'])
        write(folder/'context-decisions.json',results['context_v2']['reference_context_decisions'])
        write(folder/'part-structure-decisions.json',results['context_v2']['part_structure_decisions'])
        if not view.get('bank'):
            write(folder/'graph.json',graph);write(folder/'heads.json',heads)
            write(folder/'base.json',{k:v for k,v in base.items() if not isinstance(v,np.ndarray)})
            Image.fromarray(base['pixel_roles']).save(folder/'roles.png')
        row={k:case[k] for k in ('scope','group_id','split')}
        row.update(case=case['id'],mode=view['mode'],repeat=request['repeat'],new_control=job['new_control'],
            initial_fresh_validation=job.get('initial_fresh_validation',False),expected_dimension_chains=expected,
            unique_foot_targets_admitted=usable,strict_chains=strict,chains=scored,heads=hm,new_chains=counts,
            previous_chain_count=len(base['dimension_chains']),prefix_unchanged=prefix,pixel_roles_unchanged=unchanged,
            initial_arms_reproduced=reproduced,stage1_reproduced=results['context_v1']['open_reference_chains']==results['context_v2']['stage1_open_reference_chains'],
            source_observations_unchanged=before==object_hash([graph,heads,base['dimension_chains']]),
            source_pixels_unchanged=pixels==hashlib.sha256(gray.tobytes()).hexdigest(),audits=audits,preparation=calls,
            production_changed=False,physical_geometry_approved=False,source_pixels_erased=0)
        write(folder/'result.json',row);rows.append(row)
    write(out/'result.json',{'rows':rows})


def run(sha):
    from experiment_children_v1 import run as execute
    e=execute(ROOT,BASE,{'path':'artifacts/reference-context-profile-v2/plan.json','sha256':sha},Path(__file__).resolve())
    rows=[read(ROOT/r['artifact']) for r in read(BASE/'results-index.json')]
    e.update(preparation_calls={k:sum(r['preparation'][k] for r in rows) for k in rows[0]['preparation']},
        calls_per_arm={a:len(rows) for a in ARMS},new_OCR_ML_calls=0,new_DXF_exports=0,training=False,closed_test_used=False)
    write(BASE/'execution.json',e);print(e,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','run','child']);p.add_argument('--sha256');p.add_argument('--request',type=Path)
    a=p.parse_args()
    if a.mode=='freeze':freeze()
    elif a.mode=='child':child(a.request)
    elif a.sha256:run(a.sha256)
    else:p.error('--sha256 required')
