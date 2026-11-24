"""Target-free four-arm head refinement with frozen source observations."""
import argparse
import hashlib
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from cad_recovery.dataset_layout_v1 import checked,pin
from cad_recovery.head_refinement_v1 import refine
from cad_recovery.real_chain_metrics_v1 import head_metrics,chain_metrics
from cad_recovery.open_reference_chains_v2 import extend
from cad_recovery.open_reference_context_v2 import filter_context
from cad_recovery.open_reference_chains_v1 import contacts
from cad_recovery.stroke_roles_v1 import foreground
from run_reference_context_v1 import get_inputs,LABELS
from run_topology_robust_v1 import read,write,object_hash

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/head-refinement-profile-v1'
QUAL=ROOT/'artifacts/head-refinement-qualification-v1'
PREVIOUS={'path':'artifacts/reference-context-release-v2.json','sha256':'cf59144e5fb1900d88fb248ce4c18672e0cd5669a4dbbff6d3d40592f350bee6'}
CONTROLS={'path':'artifacts/head-refinement-controls-v1/contract.json','sha256':'ff73a600c674a7f505c630e2516e6ba456a687aeecf5f0e91bc4eed4df7ad2f2'}
ARMS=('baseline','localized','stem_only','combined')


def review_for(case):
    labels=case.get('labels',{});arrows=labels.get('arrows')
    if arrows is not None and all(all(k in h for k in ('id','tip','direction')) for h in arrows):
        heads=[{'id':h['id'],'tip_px':h['tip'],'direction':h['direction']} for h in arrows]
        chains=labels.get('chains')
        if chains is None and labels.get('expected_dimension_chains') and labels.get('feet_px') and len(arrows)==2:
            from run_open_references_v1 import control_review
            return control_review(case)
        return {'heads':heads,'head_scopes':[[0,0,*case['size_px']]],'chains':chains or []}
    real=read(checked(ROOT,LABELS))
    if case['id'] in real['fully_reviewed_negative_head_cases']:
        return {'heads':[],'head_scopes':[[0,0,*case['size_px']]],'chains':[]}
    return next((r for r in real['records'] if r['id']==case['id']),None)


def qualify():
    cv2.setNumThreads(1);cv2.setRNGSeed(0);QUAL.mkdir(exist_ok=True);jobs=[];rows=[]
    for job in read(checked(ROOT,CONTROLS))['jobs']:
        if job['case']['split']!='train':continue
        views=[]
        for view in job['views']:
            folder=QUAL/'bank'/job['case']['id']/view['mode']
            if folder.exists():
                bank={k:pin(ROOT,(folder/file).relative_to(ROOT).as_posix()) for k,file in
                    [('graph','graph.json'),('heads','heads.json'),('base','base.json'),('roles','roles.png')]}
                views.append({**view,'bank':bank,'used_head_ids':read(folder/'base.json')['head_ids_used'],
                    'captured_before_interrupted_qualification':True})
                continue
            gray,graph,base,heads,used,calls=get_inputs(view)
            folder.mkdir(parents=True)
            write(folder/'graph.json',graph);write(folder/'heads.json',heads)
            write(folder/'base.json',{k:v for k,v in base.items() if not isinstance(v,np.ndarray)})
            Image.fromarray(base['pixel_roles']).save(folder/'roles.png')
            bank={k:pin(ROOT,(folder/file).relative_to(ROOT).as_posix()) for k,file in
                [('graph','graph.json'),('heads','heads.json'),('base','base.json'),('roles','roles.png')]}
            views.append({**view,'bank':bank,'used_head_ids':[h['id'] for h in used],'capture_preparation':calls})
        jobs.append({**job,'views':views})
    old=read(ROOT/'artifacts/reference-context-profile-v2/plan.json')
    dev=[j for j in old['jobs'] if j['case']['id'] in ('train-3-left','train-3-title','train-2-title','train-2-upper')]
    for job in jobs+dev:
        for view in job['views']:
            gray,graph,base,heads,used,_=get_inputs(view);result=refine(gray,heads,graph['tracks']);review=review_for(job['case'])
            packets={'baseline':heads,**{a:result[a] for a in ARMS[1:]}}
            metrics={a:{k:v for k,v in head_metrics(review,h,view['input_to_native']).items() if k in ('tp','fp','fn')}
                for a,h in packets.items()}
            rows.append({'case':job['case']['id'],'mode':view['mode'],'split':'train' if job in jobs else 'development',
                'heads':metrics,'decisions':result['decisions'],'incomplete':result['search_incomplete']})
    write(QUAL/'train-index.json',{'jobs':jobs,'controls':CONTROLS,'validation_predictions_opened':False})
    write(QUAL/'report.json',{'rows':rows,'validation_predictions_opened':False,
        'model':pin(ROOT,'src/cad_recovery/head_refinement_v1.py')})
    for split in ('train','development'):
        print({'split':split,'metrics':{a:{k:sum(r['heads'][a][k] for r in rows if r['split']==split)
            for k in ('tp','fp','fn')} for a in ARMS}},flush=True)


def freeze():
    checked(ROOT,PREVIOUS);previous=read(ROOT/'artifacts/reference-context-profile-v2/plan.json');control=read(checked(ROOT,CONTROLS))
    for p in previous['code']:checked(ROOT,p)
    checked(ROOT,read(QUAL/'report.json')['model'])
    index={(r['case'],r['mode']):r for r in read(ROOT/'artifacts/reference-context-profile-v2/results-index.json') if not r['repeat']};jobs=[]
    for job in previous['jobs']:
        views=[]
        for view in job['views']:
            folder=(ROOT/index[job['case']['id'],view['mode']]['artifact']).parent
            if not view.get('bank'):
                view={**view,'bank':{k:pin(ROOT,(folder/file).relative_to(ROOT).as_posix()) for k,file in
                    [('graph','graph.json'),('heads','heads.json'),('base','base.json'),('roles','roles.png')]},
                    'used_head_ids':read(folder/'base.json')['head_ids_used']}
            views.append({**view,'previous_context_chains':pin(ROOT,(folder/'context_v2-chains.json').relative_to(ROOT).as_posix())})
        jobs.append({**job,'case':{**job['case'],'split':'development'},'views':views,'new_control':False})
    jobs+=read(QUAL/'train-index.json')['jobs'];jobs+=[j for j in control['jobs'] if j['case']['split']=='validation']
    names=[p['path'] for p in previous['code']]+['scripts/run_head_refinement_v1.py','scripts/prepare_head_refinement_controls_v1.py',
        'src/cad_recovery/head_refinement_v1.py','tests/test_head_refinement_v1.py']
    write(BASE/'plan.json',{'task':'P10.2c.3b.2g','previous':PREVIOUS,'controls':CONTROLS,'labels':LABELS,
        'jobs':jobs,'code':[pin(ROOT,n) for n in dict.fromkeys(names)],'arms':ARMS,'repeats':2,
        'budgets':control['budgets'],'criteria':control['criteria'],'qualification':pin(ROOT,'artifacts/head-refinement-qualification-v1/report.json'),
        'selection_order':['combined','localized','baseline'],'stem_only_is_independent_ablation':True,
        'selection_rule':'localized requires fixed-tolerance real-half gain and no new wrong/chain loss; stem requires title-FP gain and >=95% fresh TP retention; combined requires both',
        'fresh_validation_predictions_before_freeze':False,'production_changed':False,'physical_geometry_approved':False})
    print(pin(ROOT,'artifacts/head-refinement-profile-v1/plan.json'),flush=True)


def child(path):
    cv2.setNumThreads(1);cv2.setRNGSeed(0);request=read(path);plan=read(checked(ROOT,request['plan']))
    for p in plan['code']:checked(ROOT,p)
    job=request['job'];case=job['case'];out=path.parent/'output';out.mkdir();rows=[]
    for view in job['views']:
        gray,graph,base,heads,used,calls=get_inputs(view);snap=object_hash([graph,heads,base['dimension_chains']])
        pixel_sha=hashlib.sha256(gray.tobytes()).hexdigest();roles_sha=hashlib.sha256(base['pixel_roles'].tobytes()).hexdigest()
        result=refine(gray,heads,graph['tracks']);head_views={'baseline':heads,**{a:result[a] for a in ARMS[1:]}}
        allowed={h['id'] for h in used};packets={};binding={};prefix={};immutable={}
        for arm,h in head_views.items():
            effective=[x for x in h if x['id'] in allowed];binding[arm]=effective
            packets[arm]=filter_context(gray,graph,base,extend(gray,graph,base,effective))
            prefix[arm]=packets[arm]['dimension_chains'][:len(base['dimension_chains'])]==base['dimension_chains']
            immutable[arm]=roles_sha==hashlib.sha256(packets[arm]['pixel_roles'].tobytes()).hexdigest()
        # All inference is finished before any target or matching result is used.
        review=review_for(case);hm={};effective_hm={};cm={}
        for arm,h in head_views.items():
            hm[arm]=head_metrics(review,h,view['input_to_native']) if review else {}
            effective_hm[arm]=head_metrics(review,binding[arm],view['input_to_native']) if review else {}
            trace={'contacts':{k:[x['id'] for x in hs] for k,hs in contacts(graph['tracks'],binding[arm],foreground(gray)).items()},'eligible_boundaries':[]}
            cm[arm]=chain_metrics(review,h,view['input_to_native'],packets[arm],graph,trace,hm[arm]) if review and review['chains'] else {}
        folder=out/view['mode'];folder.mkdir()
        for arm,h in head_views.items():
            write(folder/(arm+'-heads.json'),h);write(folder/(arm+'-new-chains.json'),packets[arm]['open_reference_chains'])
        write(folder/'decisions.json',result['decisions'])
        if not view.get('bank'):
            write(folder/'graph.json',graph);write(folder/'heads.json',heads)
            write(folder/'base.json',{k:v for k,v in base.items() if not isinstance(v,np.ndarray)})
            Image.fromarray(base['pixel_roles']).save(folder/'roles.png')
        reproduced=not view.get('previous_context_chains') or packets['baseline']['open_reference_chains']==read(checked(ROOT,view['previous_context_chains']))
        rows.append({**{k:case[k] for k in ('scope','group_id','split')},'case':case['id'],'mode':view['mode'],'repeat':request['repeat'],
            'new_control':job['new_control'],'heads':hm,'binding_heads':effective_hm,'chains':cm,
            'head_counts':{a:len(h) for a,h in head_views.items()},'new_chains':{a:len(p['open_reference_chains']) for a,p in packets.items()},
            'raw_observations_unchanged':heads==result['raw_observations'],'source_observations_unchanged':snap==object_hash([graph,heads,base['dimension_chains']]),
            'input_pixels_unchanged':pixel_sha==hashlib.sha256(gray.tobytes()).hexdigest(),'roles_unchanged':immutable,'prefix_unchanged':prefix,
            'baseline_context_reproduced':reproduced,'refinement_audit':result['audit'],'refinement_incomplete':result['search_incomplete'],
            'chain_search_incomplete':{a:p['search_incomplete'] for a,p in packets.items()},'preparation':calls,
            'production_changed':False,'physical_geometry_approved':False,'source_pixels_erased':0})
        write(folder/'result.json',rows[-1])
    write(out/'result.json',{'rows':rows})


def run(sha):
    from experiment_children_v1 import run as execute
    e=execute(ROOT,BASE,{'path':'artifacts/head-refinement-profile-v1/plan.json','sha256':sha},Path(__file__).resolve())
    rows=[read(ROOT/r['artifact']) for r in read(BASE/'results-index.json')]
    e.update(refinement_calls=len(rows),chain_calls_per_arm={a:len(rows) for a in ARMS},
        preparation_calls={k:sum(r['preparation'][k] for r in rows) for k in rows[0]['preparation']},
        new_OCR_ML_calls=0,new_DXF_exports=0,training=False,closed_test_used=False)
    write(BASE/'execution.json',e);print(e,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['qualify','freeze','run','child']);p.add_argument('--sha256');p.add_argument('--request',type=Path)
    a=p.parse_args()
    if a.mode=='qualify':qualify()
    elif a.mode=='freeze':freeze()
    elif a.mode=='child':child(a.request)
    elif a.sha256:run(a.sha256)
    else:p.error('--sha256 required')
