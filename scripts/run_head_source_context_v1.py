"""Prospective source-path/text-context comparison; no targets reach inference."""
import argparse
import hashlib
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'artifacts/cv-skeleton-runtime-v1/deps'))
import cv2
import numpy as np
from PIL import Image
from cad_recovery.dataset_layout_v1 import checked,pin
from cad_recovery.head_source_context_v1 import analyze
from cad_recovery.head_refinement_v1 import refine
from cad_recovery.real_chain_metrics_v1 import head_metrics,chain_metrics
from cad_recovery.open_reference_chains_v2 import extend
from cad_recovery.open_reference_context_v2 import filter_context
from cad_recovery.open_reference_chains_v1 import contacts
from cad_recovery.stroke_roles_v1 import foreground
from run_reference_context_v1 import get_inputs,LABELS
from run_head_refinement_v1 import review_for
from run_topology_robust_v1 import read,write,object_hash

BASE=ROOT/'artifacts/head-source-context-profile-v1'
QUAL=ROOT/'artifacts/head-source-context-qualification-v1'
PREVIOUS={'path':'artifacts/junction-localization-release-v1.json','sha256':'315bd9e6c9c3505d66a3c130c1af643802add78a195d250d4e8ae9943a72462a'}
CONTROLS={'path':'artifacts/head-context-controls-v1/contract.json','sha256':'28188f786f8d56bc9ede651682320357661bbaca0c0563538742adfcc8040f2b'}
ARMS=('previous','rejected_stem','context_only','combined')


def infer(gray,heads,tracks):
    old=refine(gray,heads,tracks);new=analyze(gray,old['localized'])
    return old,new,{'previous':old['localized'],'rejected_stem':old['combined'],
        'context_only':new['context_only'],'combined':new['combined']}


def capture(job):
    views=[]
    for view in job['views']:
        folder=QUAL/'bank'/job['case']['id']/view['mode']
        gray,graph,base,heads,used,calls=get_inputs(view);folder.mkdir(parents=True,exist_ok=False)
        write(folder/'graph.json',graph);write(folder/'heads.json',heads)
        write(folder/'base.json',{k:v for k,v in base.items() if not isinstance(v,np.ndarray)})
        Image.fromarray(base['pixel_roles']).save(folder/'roles.png')
        bank={k:pin(ROOT,(folder/file).relative_to(ROOT).as_posix()) for k,file in
            [('graph','graph.json'),('heads','heads.json'),('base','base.json'),('roles','roles.png')]}
        views.append({**view,'bank':bank,'used_head_ids':[h['id'] for h in used],'capture_preparation':calls})
    return {**job,'views':views}


def qualify():
    cv2.setNumThreads(1);cv2.setRNGSeed(0);QUAL.mkdir(exist_ok=False)
    jobs=[capture(j) for j in read(checked(ROOT,CONTROLS))['jobs'] if j['case']['split']=='train']
    previous=read(ROOT/'artifacts/head-refinement-profile-v1/plan.json')
    dev=[j for j in previous['jobs'] if j['case']['id'] in ('train-3-left','train-3-title','train-2-title','train-2-upper')
        or j['new_control']]
    rows=[]
    for job in jobs+dev:
        for view in job['views']:
            gray,graph,base,heads,used,_=get_inputs(view);old,new,arms=infer(gray,heads,graph['tracks'])
            review=review_for(job['case'])
            metrics={a:{k:v for k,v in head_metrics(review,h,view['input_to_native']).items() if k in ('tp','fp','fn')}
                for a,h in arms.items()}
            rows.append({'case':job['case']['id'],'mode':view['mode'],'split':'train' if job in jobs else 'development',
                'heads':metrics,'decisions':new['decisions'],'complete':new['complete']})
    write(QUAL/'train-index.json',{'jobs':jobs,'controls':CONTROLS,'validation_predictions_opened':False})
    write(QUAL/'report.json',{'rows':rows,'validation_predictions_opened':False,
        'model':pin(ROOT,'src/cad_recovery/head_source_context_v1.py')})
    for split in ('train','development'):
        print({'split':split,'metrics':{a:{k:sum(r['heads'][a][k] for r in rows if r['split']==split)
            for k in ('tp','fp','fn')} for a in ARMS}},flush=True)


def freeze():
    checked(ROOT,PREVIOUS);previous=read(ROOT/'artifacts/head-refinement-profile-v1/plan.json')
    control=read(checked(ROOT,CONTROLS));qualification=read(QUAL/'report.json');checked(ROOT,qualification['model'])
    for p in previous['code']:checked(ROOT,p)
    runtime=read(ROOT/'artifacts/cv-skeleton-runtime-v1/installed-files.json')
    for name,sha in runtime['installed_static_files'].items():checked(ROOT,{'path':name,'sha256':sha})
    index={(r['case'],r['mode']):r for r in read(ROOT/'artifacts/head-refinement-profile-v1/results-index.json') if not r['repeat']}
    jobs=[]
    for job in previous['jobs']:
        views=[]
        for view in job['views']:
            folder=(ROOT/index[job['case']['id'],view['mode']]['artifact']).parent
            if not view.get('bank'):
                view={**view,'bank':{k:pin(ROOT,(folder/file).relative_to(ROOT).as_posix()) for k,file in
                    [('graph','graph.json'),('heads','heads.json'),('base','base.json'),('roles','roles.png')]},
                    'used_head_ids':read(folder/'base.json')['head_ids_used']}
            views.append({**view,'gold_heads':pin(ROOT,(folder/'localized-heads.json').relative_to(ROOT).as_posix()),
                'gold_chains':pin(ROOT,(folder/'localized-new-chains.json').relative_to(ROOT).as_posix())})
        jobs.append({**job,'case':{**job['case'],'split':'development'},'views':views,'new_control':False,
            'previous_head_validation':job['new_control'] and job['case']['split']=='validation'})
    jobs+=read(QUAL/'train-index.json')['jobs']
    jobs += [j for j in control['jobs'] if j['case']['split']=='validation']
    names=[p['path'] for p in previous['code']]+['scripts/run_head_source_context_v1.py',
        'scripts/prepare_head_context_controls_v1.py','src/cad_recovery/head_source_context_v1.py','tests/test_head_source_context_v1.py']
    BASE.mkdir(exist_ok=False)
    write(BASE/'plan.json',{'task':'P10.2c.3b.2g.1','previous':PREVIOUS,'controls':CONTROLS,'labels':LABELS,
        'jobs':jobs,'code':[pin(ROOT,n) for n in dict.fromkeys(names)],'arms':ARMS,'repeats':2,
        'budgets':control['budgets'],'criteria':control['criteria'],
        'qualification':pin(ROOT,'artifacts/head-source-context-qualification-v1/report.json'),
        'optional_runtime_static_files_verified':len(runtime['installed_static_files']),
        'fresh_validation_predictions_before_freeze':False,'production_changed':False,'physical_geometry_approved':False})
    print(pin(ROOT,'artifacts/head-source-context-profile-v1/plan.json'),flush=True)


def child(path):
    cv2.setNumThreads(1);cv2.setRNGSeed(0);request=read(path);plan=read(checked(ROOT,request['plan']))
    for p in plan['code']:checked(ROOT,p)
    job=request['job'];case=job['case'];out=path.parent/'output';out.mkdir();rows=[]
    for view in job['views']:
        gray,graph,base,heads,used,calls=get_inputs(view);snap=object_hash([graph,heads,base['dimension_chains']])
        pixel_sha=hashlib.sha256(gray.tobytes()).hexdigest();roles_sha=hashlib.sha256(base['pixel_roles'].tobytes()).hexdigest()
        old,new,arms=infer(gray,heads,graph['tracks']);allowed={h['id'] for h in used};packets={};binding={};prefix={};immutable={}
        for arm,h in arms.items():
            effective=[x for x in h if x['id'] in allowed];binding[arm]=effective
            packets[arm]=filter_context(gray,graph,base,extend(gray,graph,base,effective))
            prefix[arm]=packets[arm]['dimension_chains'][:len(base['dimension_chains'])]==base['dimension_chains']
            immutable[arm]=roles_sha==hashlib.sha256(packets[arm]['pixel_roles'].tobytes()).hexdigest()
        review=review_for(case);hm={};effective_hm={};cm={}
        for arm,h in arms.items():
            hm[arm]=head_metrics(review,h,view['input_to_native']) if review else {}
            effective_hm[arm]=head_metrics(review,binding[arm],view['input_to_native']) if review else {}
            trace={'contacts':{k:[x['id'] for x in hs] for k,hs in contacts(graph['tracks'],binding[arm],foreground(gray)).items()},'eligible_boundaries':[]}
            cm[arm]=chain_metrics(review,h,view['input_to_native'],packets[arm],graph,trace,hm[arm]) if review and review['chains'] else {}
        folder=out/view['mode'];folder.mkdir()
        for arm,h in arms.items():
            write(folder/(arm+'-heads.json'),h);write(folder/(arm+'-new-chains.json'),packets[arm]['open_reference_chains'])
        write(folder/'decisions.json',new['decisions'])
        if not view.get('bank'):
            write(folder/'graph.json',graph);write(folder/'heads.json',heads)
            write(folder/'base.json',{k:v for k,v in base.items() if not isinstance(v,np.ndarray)})
            Image.fromarray(base['pixel_roles']).save(folder/'roles.png')
        rows.append({**{k:case[k] for k in ('scope','group_id','split')},'case':case['id'],'mode':view['mode'],'repeat':request['repeat'],
            'new_control':job['new_control'],'previous_head_validation':job.get('previous_head_validation',False),
            'heads':hm,'binding_heads':effective_hm,'chains':cm,'head_counts':{a:len(h) for a,h in arms.items()},
            'new_chains':{a:len(p['open_reference_chains']) for a,p in packets.items()},
            'raw_observations_unchanged':heads==old['raw_observations'] and new['observations']==old['localized'],
            'source_observations_unchanged':snap==object_hash([graph,heads,base['dimension_chains']]),
            'input_pixels_unchanged':pixel_sha==hashlib.sha256(gray.tobytes()).hexdigest(),'roles_unchanged':immutable,'prefix_unchanged':prefix,
            'previous_heads_reproduced':not view.get('gold_heads') or arms['previous']==read(checked(ROOT,view['gold_heads'])),
            'previous_chains_reproduced':not view.get('gold_chains') or packets['previous']['open_reference_chains']==read(checked(ROOT,view['gold_chains'])),
            'context_audit':new['audit'],'context_complete':new['complete'],'refinement_incomplete':old['search_incomplete'],
            'chain_search_incomplete':{a:p['search_incomplete'] for a,p in packets.items()},'preparation':calls,
            'production_changed':False,'physical_geometry_approved':False,'source_pixels_erased':0})
        write(folder/'result.json',rows[-1])
    write(out/'result.json',{'rows':rows})


def run(sha):
    from experiment_children_v1 import run as execute
    e=execute(ROOT,BASE,{'path':'artifacts/head-source-context-profile-v1/plan.json','sha256':sha},Path(__file__).resolve())
    rows=[read(ROOT/r['artifact']) for r in read(BASE/'results-index.json')]
    e.update(context_calls=len(rows),chain_calls_per_arm={a:len(rows) for a in ARMS},
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
