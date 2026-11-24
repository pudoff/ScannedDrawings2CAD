"""Frozen-input shadow profile: old-head replay versus normalized new heads."""
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
from cad_recovery.joint_cv_profile_v1 import analyze
from cad_recovery.joint_role_metrics_v1 import score
from cad_recovery.stroke_role_metrics_v1 import to_native,synthetic_truth
from cad_recovery.stroke_ownership_metrics_v1 import ink_ledger
from cad_recovery.stroke_roles_v1 import foreground
from cad_recovery.open_reference_chains_v2 import extend
from cad_recovery.open_reference_context_v2 import filter_context
from cad_recovery.open_reference_chains_v1 import contacts
from cad_recovery.real_chain_metrics_v1 import head_metrics,chain_metrics
from run_head_recall_v1 import infer
from run_reference_context_v1 import get_inputs,LABELS
from run_head_refinement_v1 import review_for
from run_topology_robust_v1 import read,write,object_hash

BASE=ROOT/'artifacts/joint-cv-profile-v1';QUAL=ROOT/'artifacts/joint-cv-qualification-v2'
PREVIOUS={'path':'artifacts/head-recall-release-v1.json','sha256':'acf51c03c06c2476d2807b2894d86851aec7e296d75b536a3ee42cf028979df8'}
CONTROLS={'path':'artifacts/joint-cv-controls-v1/contract.json','sha256':'b23947ab3360f7293ac93c9056d76b70e509aead9440e28b94be067a202c4edd'}
ARMS=('baseline','replay_old','joint')


def old_job(job,role_labels):
    case=job['case'];views=[];witness=role_labels.get(case['id'])
    for v in job['views']:
        folder=ROOT/'artifacts/head-recall-profile-v1/runs'/(case['id']+'-r0')/'output'/v['mode']
        if not v.get('bank'):
            v={**v,'bank':{k:pin(ROOT,(folder/f).relative_to(ROOT).as_posix()) for k,f in
                [('graph','graph.json'),('heads','heads.json'),('base','base.json'),('roles','roles.png')]},
                'used_head_ids':read(folder/'base.json')['head_ids_used']}
        if witness:
            w=next(w for w in witness['views'] if w['mode']==v['mode'])
            if w['gray']['sha256']!=v['gray']['sha256']:raise ValueError('Role label source differs: '+case['id'])
        views.append({**v,'profile_heads':{n:pin(ROOT,(folder/(n+'-heads.json')).relative_to(ROOT).as_posix()) for n in ('previous','guarded')},
            'profile_links':pin(ROOT,(folder/'guarded-new-chains.json').relative_to(ROOT).as_posix()),
            'prior_head_row':pin(ROOT,(folder/'result.json').relative_to(ROOT).as_posix())})
    label_fields={k:witness[k] for k in ('truth','negative') if witness and witness.get(k)}
    return {**job,**label_fields,'case':{**case,'split':'development'},'views':views,'new_control':False}


def inputs(v):
    gray,graph,base,raw,used,calls=get_inputs(v)
    if v.get('profile_heads'):
        previous=read(checked(ROOT,v['profile_heads']['previous']));joint=read(checked(ROOT,v['profile_heads']['guarded']))
        links=read(checked(ROOT,v['profile_links']));head_calls=0
    else:
        localized,context,detector,arms=infer(gray,raw,graph['tracks']);previous=arms['previous'];joint=arms['guarded']
        if not context['complete'] or localized['search_incomplete'] or not detector['complete']:raise ValueError('Incomplete new head preparation')
        original={h['id'] for h in localized['localized']};allowed={h['id'] for h in used}
        effective=[h for h in joint if h['id'] in allowed or h['id'] not in original]
        links_packet=filter_context(gray,graph,base,extend(gray,graph,base,effective))
        if links_packet['search_incomplete']:raise ValueError('Incomplete new chain preparation')
        links=links_packet['open_reference_chains'];head_calls=1
    fullbase={**base,'dimension_chains':base['dimension_chains']+links}
    allowed={h['id'] for h in used}&{h['id'] for h in previous}
    return gray,graph,fullbase,raw,previous,joint,allowed,{**calls,'new_head_calls':head_calls},len(base['dimension_chains'])


def truth_for(job,native_gray):
    if job.get('truth'):truth=np.array(Image.open(checked(ROOT,job['truth'])))
    elif job['case']['scope']=='analytical_control' and job['views'][0].get('reviewed_mask'):
        truth=synthetic_truth(job['case'],native_gray,np.array(Image.open(checked(ROOT,job['views'][0]['reviewed_mask']))))
    else:truth=None
    negative=np.array(Image.open(checked(ROOT,job['negative']))) if job.get('negative') else None
    if truth is None and negative is not None:truth=np.full(tuple(job['case']['size_px'][::-1]),255,np.uint8)
    return truth,negative


def evaluate(job,v):
    gray,graph,base,raw,old,new,allowed,calls,_=inputs(v)
    before=object_hash([graph,raw,old,new,base['dimension_chains']]);pixels=hashlib.sha256(gray.tobytes()).hexdigest()
    known=base['pixel_roles']!=255;oldmap=base['pixel_roles'].copy()
    result=analyze(gray,graph,base,old,new,prior_binding_ids=allowed)
    packets={'baseline':base,'replay_old':result['replay_old'],'joint':result['joint']}
    heads={'baseline':new,'replay_old':old,'joint':result['width_contract']['normalized_heads']}
    return gray,graph,base,heads,packets,result,{'calls':calls,'source_exact':before==object_hash([graph,raw,old,new,base['dimension_chains']]),
        'pixels_exact':pixels==hashlib.sha256(gray.tobytes()).hexdigest(),'inherited_map_exact':np.array_equal(oldmap,base['pixel_roles']),
        'known_roles_exact':{a:bool(np.array_equal(p['pixel_roles'][known],oldmap[known])) for a,p in packets.items()},
        'chain_prefix_exact':{a:p['dimension_chains'][:len(base['dimension_chains'])]==base['dimension_chains'] for a,p in packets.items()}}


def role_rows(job,truth,negative,base,packets):
    metrics={};additions={};native_old=to_native(base['pixel_roles'],job['case']['size_px'])
    for a,p in packets.items():
        native=to_native(p['pixel_roles'],job['case']['size_px']);metrics[a]=score(truth,native,negative) if truth is not None else None
        delta=native!=native_old;valid=(truth>=1)&(truth<=9) if truth is not None else np.zeros_like(delta)
        unknown=(truth==255) if truth is not None else np.ones_like(delta)
        additions[a]={'correct':int((delta&valid&(native==truth)).sum()) if truth is not None else 0,
            'wrong':int((delta&valid&(native!=truth)).sum()) if truth is not None else 0,
            'negative':int((delta&unknown&(negative>0)).sum()) if negative is not None else 0,
            'unknown_truth':int((delta&unknown).sum()),'all_native_pixels':int(delta.sum())}
    return metrics,additions


def qualify():
    cv2.setNumThreads(1);cv2.setRNGSeed(0);QUAL.mkdir(exist_ok=False);jobs=[]
    prior_capture=ROOT/'artifacts/joint-cv-qualification-v1/train-index.json'
    if prior_capture.exists():jobs=read(prior_capture)['jobs']
    for job in ([] if jobs else read(checked(ROOT,CONTROLS))['jobs']):
        if job['case']['split']!='train':continue
        views=[]
        for v in job['views']:
            gray,graph,base,raw,previous,joint,allowed,calls,old_count=inputs(v);folder=QUAL/'bank'/job['case']['id']/v['mode'];folder.mkdir(parents=True)
            write(folder/'graph.json',graph);write(folder/'heads.json',raw)
            # get_inputs expects the unextended base. Source chains are frozen separately.
            original={**base,'dimension_chains':base['dimension_chains'][:old_count]}
            write(folder/'base.json',{k:z for k,z in original.items() if not isinstance(z,np.ndarray)})
            Image.fromarray(original['pixel_roles']).save(folder/'roles.png')
            write(folder/'previous-heads.json',previous);write(folder/'guarded-heads.json',joint)
            write(folder/'links.json',base['dimension_chains'][len(original['dimension_chains']):])
            views.append({**v,'bank':{k:pin(ROOT,(folder/f).relative_to(ROOT).as_posix()) for k,f in
                [('graph','graph.json'),('heads','heads.json'),('base','base.json'),('roles','roles.png')]},
                'used_head_ids':original['head_ids_used'],'profile_heads':{n:pin(ROOT,(folder/(n+'-heads.json')).relative_to(ROOT).as_posix()) for n in ('previous','guarded')},
                'profile_links':pin(ROOT,(folder/'links.json').relative_to(ROOT).as_posix()),'capture_preparation':calls})
        jobs.append({**job,'views':views})
    prior=read(ROOT/'artifacts/head-recall-profile-v1/plan.json');role_index={j['case']['id']:j for j in read(ROOT/'artifacts/head-junction-profile-v1/plan.json')['jobs']}
    labels=read(checked(ROOT,LABELS));ids={r['id'] for r in labels['records']}|set(labels['fully_reviewed_negative_head_cases'])
    ids|={e['case'] for e in read(ROOT/'artifacts/source-chain-profile-v2/joint-error-diagnosis.json')['entries']}
    dev=[old_job(j,role_index) for j in prior['jobs'] if j['case']['id'] in ids]
    rows=[]
    for job in jobs+dev:
        native=None
        for v in job['views']:
            gray,graph,base,heads,packets,result,checks=evaluate(job,v)
            if native is None:native=gray.copy()
            truth,negative=truth_for(job,native);metrics,additions=role_rows(job,truth,negative,base,packets)
            rows.append({'case':job['case']['id'],'mode':v['mode'],'split':'train' if job in jobs else 'development','additions':additions,
                'width_complete':result['width_contract']['complete'],'role_incomplete':{a:p.get('search_incomplete',False) for a,p in packets.items()},'checks':checks})
    write(QUAL/'train-index.json',{'jobs':jobs,'validation_predictions_opened':False})
    write(QUAL/'report.json',{'rows':rows,'validation_predictions_opened':False,'code':[pin(ROOT,n) for n in
        ('src/cad_recovery/head_width_contract_v1.py','src/cad_recovery/joint_cv_profile_v1.py','src/cad_recovery/joint_role_metrics_v1.py')]})
    for split in ('train','development'):
        print({'split':split,'additions':{a:{k:sum(r['additions'][a][k] for r in rows if r['split']==split) for k in ('correct','wrong','negative','unknown_truth')}
            for a in ARMS},'incomplete':sum(any(r['role_incomplete'].values()) for r in rows if r['split']==split)},flush=True)


def freeze():
    checked(ROOT,PREVIOUS);prior=read(ROOT/'artifacts/head-recall-profile-v1/plan.json');control=read(checked(ROOT,CONTROLS))
    for p in prior['code']+read(QUAL/'report.json')['code']:checked(ROOT,p)
    labels={j['case']['id']:j for j in read(ROOT/'artifacts/head-junction-profile-v1/plan.json')['jobs']}
    jobs=[old_job(j,labels) for j in prior['jobs']]+read(QUAL/'train-index.json')['jobs']
    jobs += [j for j in control['jobs'] if j['case']['split']=='validation']
    names=[p['path'] for p in prior['code']]+[p['path'] for p in read(QUAL/'report.json')['code']]+[
        'scripts/run_joint_cv_v1.py','scripts/prepare_joint_cv_controls_v1.py','tests/test_joint_cv_profile_v1.py']
    BASE.mkdir(exist_ok=False)
    write(BASE/'plan.json',{'task':'P10.2c.3b.2h','previous':PREVIOUS,'controls':CONTROLS,'labels':LABELS,
        'jobs':jobs,'code':[pin(ROOT,n) for n in dict.fromkeys(names)],'arms':ARMS,'budgets':control['budgets'],
        'criteria':control['criteria'],'qualification':pin(ROOT,'artifacts/joint-cv-qualification-v2/report.json'),
        'new_validation_predictions_before_freeze':False,'role_metric_includes_false_positive_classes_without_GT_support':True,
        'new_junction_rewiring_used':False,'production_changed':False,'physical_geometry_approved':False})
    print(pin(ROOT,'artifacts/joint-cv-profile-v1/plan.json'),flush=True)


def child(path):
    cv2.setNumThreads(1);cv2.setRNGSeed(0);request=read(path);plan=read(checked(ROOT,request['plan']))
    for p in plan['code']:checked(ROOT,p)
    job=request['job'];case=job['case'];out=path.parent/'output';out.mkdir();rows=[];native=None
    for v in job['views']:
        gray,graph,base,heads,packets,result,checks=evaluate(job,v)
        if native is None:native=gray.copy()
        truth,negative=truth_for(job,native);roles,additions=role_rows(job,truth,negative,base,packets)
        review=review_for(case);hm={a:head_metrics(review,h,v['input_to_native']) if review else {} for a,h in heads.items()};cm={};source_cm={}
        for a,h in heads.items():
            trace={'contacts':{k:[x['id'] for x in hs] for k,hs in contacts(graph['tracks'],h,foreground(gray)).items()},'eligible_boundaries':[]}
            cm[a]=chain_metrics(review,h,v['input_to_native'],packets[a],graph,trace,hm[a]) if review and review['chains'] else {}
            if a!='baseline':source_cm[a]=chain_metrics(review,h,v['input_to_native'],{'dimension_chains':packets[a]['source_chain_hypotheses']},graph,trace,hm[a]) if review and review['chains'] else {}
        folder=out/v['mode'];folder.mkdir()
        for a,p in packets.items():
            Image.fromarray(p['pixel_roles']).save(folder/(a+'-roles.png'));write(folder/(a+'-chains.json'),p['dimension_chains']);write(folder/(a+'-heads.json'),heads[a])
        for a in ARMS[1:]:Image.fromarray(packets[a]['pixel_interval_owner']).save(folder/(a+'-owners.png'))
        write(folder/'width-contract.json',result['width_contract'])
        for a in ARMS[1:]:
            write(folder/(a+'-source-chains.json'),packets[a]['source_chain_hypotheses'])
            write(folder/(a+'-chain-authority.json'),packets[a]['chain_authority_decisions'])
        prior=read(checked(ROOT,v['prior_head_row'])) if v.get('prior_head_row') else None
        rows.append({**{k:case[k] for k in ('scope','group_id','split')},'case':case['id'],'mode':v['mode'],'repeat':request['repeat'],
            'new_control':job['new_control'],'roles':roles,'additions':additions,'heads':hm,'chains':cm,'checks':checks,
            'source_chain_hypothesis_metrics':source_cm,
            'ink_ledgers':{a:ink_ledger(gray,p['pixel_roles'],foreground) for a,p in packets.items()},
            'width_complete':result['width_contract']['complete'],'width_decisions':result['width_contract']['decisions'],
            'role_incomplete':{a:p.get('search_incomplete',False) for a,p in packets.items()},
            'prior_head_metrics_exact':not prior or hm['baseline']==prior['heads']['guarded'],
            'shared_role_evaluation':result['shared_role_evaluation'],'role_evaluations':result['role_evaluations'],
            'raw_known_role_conflicts_withdrawn':{a:packets[a].get('raw_known_role_conflicts_withdrawn',0) for a in ARMS[1:]},
            'chain_authority_withdrawn_pixels':{a:packets[a]['chain_authority_withdrawn_pixels'] for a in ARMS[1:]},
            'new_CAD_primitives_or_roles_approved':0,'source_pixels_erased':0,'production_changed':False,'physical_geometry_approved':False})
        write(folder/'result.json',rows[-1])
        if not v.get('bank'):
            write(folder/'graph.json',graph)
    write(out/'result.json',{'rows':rows})


def run(sha):
    from experiment_children_v1 import run as execute
    e=execute(ROOT,BASE,{'path':'artifacts/joint-cv-profile-v1/plan.json','sha256':sha},Path(__file__).resolve())
    rows=[read(ROOT/r['artifact']) for r in read(BASE/'results-index.json')]
    e.update(role_evaluations=sum(r['role_evaluations'] for r in rows),shared_evaluations=sum(r['shared_role_evaluation'] for r in rows),
        preparation_calls={k:sum(r['checks']['calls'][k] for r in rows) for k in rows[0]['checks']['calls']},
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
