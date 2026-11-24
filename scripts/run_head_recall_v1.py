"""Isolated detector comparison against the immutable head-context release."""
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
from cad_recovery.curved_source_heads_v1 import search
from cad_recovery.head_refinement_v1 import refine
from cad_recovery.head_source_context_v1 import analyze
from cad_recovery.real_chain_metrics_v1 import head_metrics,chain_metrics
from cad_recovery.open_reference_chains_v2 import extend
from cad_recovery.open_reference_context_v2 import filter_context
from cad_recovery.open_reference_chains_v1 import contacts
from cad_recovery.stroke_roles_v1 import foreground
from run_reference_context_v1 import get_inputs,LABELS
from run_head_refinement_v1 import review_for
from run_topology_robust_v1 import read,write,object_hash

BASE=ROOT/'artifacts/head-recall-profile-v1';QUAL=ROOT/'artifacts/head-recall-qualification-v2'
PREVIOUS={'path':'artifacts/head-source-context-release-v1.json','sha256':'891765548b181a55d19c83c33e708fa37a90baf91bc2ca81a068ee7b4764b342'}
CONTROLS={'path':'artifacts/head-recall-controls-v1/contract.json','sha256':'358ec085012f5a9fcc887511ff9077832f03d763799ba46dc2c9ed5a510a491d'}
ARMS=('previous','shape_only','path_only','guarded')


def previous_job(job):
    views=[]
    for v in job['views']:
        folder=ROOT/'artifacts/head-source-context-profile-v1/runs'/(job['case']['id']+'-r0')/'output'/v['mode']
        if not v.get('bank'):
            v={**v,'bank':{k:pin(ROOT,(folder/f).relative_to(ROOT).as_posix()) for k,f in
                [('graph','graph.json'),('heads','heads.json'),('base','base.json'),('roles','roles.png')]},
                'used_head_ids':read(folder/'base.json')['head_ids_used']}
        views.append({**v,'gold_heads':pin(ROOT,(folder/'combined-heads.json').relative_to(ROOT).as_posix()),
            'gold_chains':pin(ROOT,(folder/'combined-new-chains.json').relative_to(ROOT).as_posix())})
    short_curved=job['new_control'] and job['case']['labels']['family'] in (
        'short_straight','label_adjacent_short','arc_pair','back_to_back_arcs','closed_curve_text')
    return {**job,'case':{**job['case'],'split':'development'},'views':views,'new_control':False,
        'previous_short_curved':short_curved}


def infer(gray,heads,tracks):
    localized=refine(gray,heads,tracks);context=analyze(gray,localized['localized'])
    new=search(gray,baseline=context['combined'],prior_observations=localized['localized'])
    return localized,context,new,{'previous':context['combined'],**{a:new[a] for a in ARMS[1:]}}


def qualify():
    cv2.setNumThreads(1);cv2.setRNGSeed(0);QUAL.mkdir(exist_ok=False);jobs=[]
    previous_capture=ROOT/'artifacts/head-recall-qualification-v1/train-index.json'
    if previous_capture.exists():jobs=read(previous_capture)['jobs']
    for job in ([] if jobs else read(checked(ROOT,CONTROLS))['jobs']):
        if job['case']['split']!='train':continue
        views=[]
        for v in job['views']:
            gray,graph,base,heads,used,calls=get_inputs(v);folder=QUAL/'bank'/job['case']['id']/v['mode'];folder.mkdir(parents=True)
            write(folder/'graph.json',graph);write(folder/'heads.json',heads)
            write(folder/'base.json',{k:z for k,z in base.items() if not isinstance(z,np.ndarray)})
            Image.fromarray(base['pixel_roles']).save(folder/'roles.png')
            views.append({**v,'bank':{k:pin(ROOT,(folder/f).relative_to(ROOT).as_posix()) for k,f in
                [('graph','graph.json'),('heads','heads.json'),('base','base.json'),('roles','roles.png')]},
                'used_head_ids':[h['id'] for h in used],'capture_preparation':calls})
        jobs.append({**job,'views':views})
    previous=read(ROOT/'artifacts/head-source-context-profile-v1/plan.json')
    labels=read(checked(ROOT,LABELS));ids={r['id'] for r in labels['records']}|set(labels['fully_reviewed_negative_head_cases'])
    dev=[previous_job(j) for j in previous['jobs'] if j['new_control'] or j['case']['id'] in ids]
    rows=[]
    for job in jobs+dev:
        for v in job['views']:
            gray,graph,base,heads,used,_=get_inputs(v);old,context,new,arms=infer(gray,heads,graph['tracks'])
            review=review_for(job['case']);metrics={a:head_metrics(review,h,v['input_to_native']) for a,h in arms.items()}
            rows.append({'case':job['case']['id'],'mode':v['mode'],'split':'train' if job in jobs else 'development',
                'heads':metrics,'complete':new['complete'],'audit':new['audit'],'added_heads':len(new['guarded'])-len(context['combined'])})
    write(QUAL/'train-index.json',{'jobs':jobs,'validation_predictions_opened':False})
    write(QUAL/'report.json',{'rows':rows,'validation_predictions_opened':False,
        'model':pin(ROOT,'src/cad_recovery/curved_source_heads_v1.py')})
    for split in ('train','development'):
        print({'split':split,'metrics':{a:{k:sum(r['heads'][a][k] for r in rows if r['split']==split) for k in ('tp','fp','fn')} for a in ARMS},
            'incomplete':sum(not r['complete'] for r in rows if r['split']==split)},flush=True)


def freeze():
    checked(ROOT,PREVIOUS);previous=read(ROOT/'artifacts/head-source-context-profile-v1/plan.json')
    control=read(checked(ROOT,CONTROLS));checked(ROOT,read(QUAL/'report.json')['model'])
    for p in previous['code']:checked(ROOT,p)
    runtime=read(ROOT/'artifacts/cv-skeleton-runtime-v1/installed-files.json')
    for name,sha in runtime['installed_static_files'].items():checked(ROOT,{'path':name,'sha256':sha})
    jobs=[previous_job(j) for j in previous['jobs']]+read(QUAL/'train-index.json')['jobs']
    jobs += [j for j in control['jobs'] if j['case']['split']=='validation']
    names=[p['path'] for p in previous['code']]+['scripts/run_head_recall_v1.py','scripts/prepare_head_recall_controls_v1.py',
        'src/cad_recovery/curved_source_heads_v1.py','tests/test_curved_source_heads_v1.py']
    BASE.mkdir(exist_ok=False)
    write(BASE/'plan.json',{'task':'P10.2c.3b.2g.2','previous':PREVIOUS,'controls':CONTROLS,'labels':LABELS,
        'jobs':jobs,'code':[pin(ROOT,n) for n in dict.fromkeys(names)],'arms':ARMS,'repeats':2,
        'budgets':control['budgets'],'criteria':control['criteria'],'qualification':pin(ROOT,'artifacts/head-recall-qualification-v2/report.json'),
        'optional_runtime_static_files_verified':len(runtime['installed_static_files']),
        'fresh_validation_predictions_before_freeze':False,'production_changed':False,'physical_geometry_approved':False})
    print(pin(ROOT,'artifacts/head-recall-profile-v1/plan.json'),flush=True)


def child(path):
    cv2.setNumThreads(1);cv2.setRNGSeed(0);request=read(path);plan=read(checked(ROOT,request['plan']))
    for p in plan['code']:checked(ROOT,p)
    job=request['job'];case=job['case'];out=path.parent/'output';out.mkdir();rows=[]
    for v in job['views']:
        gray,graph,base,heads,used,calls=get_inputs(v);snap=object_hash([graph,heads,base['dimension_chains']])
        pixel_sha=hashlib.sha256(gray.tobytes()).hexdigest();roles_sha=hashlib.sha256(base['pixel_roles'].tobytes()).hexdigest()
        old,context,new,arms=infer(gray,heads,graph['tracks']);original_ids={h['id'] for h in old['localized']}
        allowed={h['id'] for h in used};packets={};binding={};prefix={};immutable={}
        for arm,h in arms.items():
            effective=[x for x in h if x['id'] in allowed or x['id'] not in original_ids];binding[arm]=effective
            packets[arm]=filter_context(gray,graph,base,extend(gray,graph,base,effective))
            prefix[arm]=packets[arm]['dimension_chains'][:len(base['dimension_chains'])]==base['dimension_chains']
            immutable[arm]=roles_sha==hashlib.sha256(packets[arm]['pixel_roles'].tobytes()).hexdigest()
        review=review_for(case);hm={};effective_hm={};cm={}
        for arm,h in arms.items():
            hm[arm]=head_metrics(review,h,v['input_to_native']) if review else {}
            effective_hm[arm]=head_metrics(review,binding[arm],v['input_to_native']) if review else {}
            trace={'contacts':{k:[x['id'] for x in hs] for k,hs in contacts(graph['tracks'],binding[arm],foreground(gray)).items()},'eligible_boundaries':[]}
            cm[arm]=chain_metrics(review,h,v['input_to_native'],packets[arm],graph,trace,hm[arm]) if review and review['chains'] else {}
        folder=out/v['mode'];folder.mkdir()
        for arm,h in arms.items():
            write(folder/(arm+'-heads.json'),h);write(folder/(arm+'-new-chains.json'),packets[arm]['open_reference_chains'])
        write(folder/'proposals.json',new['proposals']);write(folder/'decisions.json',new['decisions'])
        if not v.get('bank'):
            write(folder/'graph.json',graph);write(folder/'heads.json',heads)
            write(folder/'base.json',{k:z for k,z in base.items() if not isinstance(z,np.ndarray)})
            Image.fromarray(base['pixel_roles']).save(folder/'roles.png')
        rows.append({**{k:case[k] for k in ('scope','group_id','split')},'case':case['id'],'mode':v['mode'],'repeat':request['repeat'],
            'new_control':job['new_control'],'previous_short_curved':job.get('previous_short_curved',False),
            'heads':hm,'binding_heads':effective_hm,'chains':cm,'head_counts':{a:len(h) for a,h in arms.items()},
            'new_chains':{a:len(p['open_reference_chains']) for a,p in packets.items()},
            'raw_observations_unchanged':heads==old['raw_observations'] and new['raw_observations']==context['combined'],
            'accepted_prefix_exact':{a:h[:len(context['combined'])]==context['combined'] for a,h in arms.items()},
            'source_observations_unchanged':snap==object_hash([graph,heads,base['dimension_chains']]),
            'input_pixels_unchanged':pixel_sha==hashlib.sha256(gray.tobytes()).hexdigest(),'roles_unchanged':immutable,'prefix_unchanged':prefix,
            'previous_heads_reproduced':not v.get('gold_heads') or arms['previous']==read(checked(ROOT,v['gold_heads'])),
            'previous_chains_reproduced':not v.get('gold_chains') or packets['previous']['open_reference_chains']==read(checked(ROOT,v['gold_chains'])),
            'detector_audit':new['audit'],'detector_complete':new['complete'],'previous_search_complete':context['complete'] and not old['search_incomplete'],
            'chain_search_incomplete':{a:p['search_incomplete'] for a,p in packets.items()},'preparation':calls,
            'production_changed':False,'physical_geometry_approved':False,'source_pixels_erased':0})
        write(folder/'result.json',rows[-1])
    write(out/'result.json',{'rows':rows})


def run(sha):
    from experiment_children_v1 import run as execute
    e=execute(ROOT,BASE,{'path':'artifacts/head-recall-profile-v1/plan.json','sha256':sha},Path(__file__).resolve())
    rows=[read(ROOT/r['artifact']) for r in read(BASE/'results-index.json')]
    e.update(detector_calls=len(rows),chain_calls_per_arm={a:len(rows) for a in ARMS},
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
