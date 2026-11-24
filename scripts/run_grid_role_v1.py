"""Explicit-authority role withdrawals on frozen shadow CV maps."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'artifacts/cv-skeleton-runtime-v1/deps'))
import cv2
import numpy as np
from PIL import Image
from cad_recovery.dataset_layout_v1 import checked,pin
from cad_recovery.grid_role_revision_v1 import analyze,cv_authority,sha
from cad_recovery.joint_role_metrics_v1 import score
from cad_recovery.stroke_role_metrics_v1 import to_native
from cad_recovery.stroke_ownership_metrics_v1 import ink_ledger
from cad_recovery.stroke_roles_v1 import foreground
from run_joint_cv_v1 import evaluate as prepare_baseline,truth_for
from run_topology_robust_v1 import read,write,object_hash

BASE=ROOT/'artifacts/grid-role-profile-v1';QUAL=ROOT/'artifacts/grid-role-qualification-v1'
PREVIOUS={'path':'artifacts/joint-cv-release-v1.json','sha256':'c80c7c0517f5987b69d8378993909766536d7a73952b475847b38e4ae1b1ba25'}
CONTROLS={'path':'artifacts/grid-role-controls-v1/contract.json','sha256':'d634d14796080f98d662fbb63dd45c40114e66450f9023dafadb41473a822c04'}
ARMS=('baseline','structure_only','guarded')


def old_job(job):
    views=[]
    for v in job['views']:
        folder=ROOT/'artifacts/joint-cv-profile-v1/runs'/(job['case']['id']+'-r0')/'output'/v['mode']
        graph=v['bank']['graph'] if v.get('bank') else pin(ROOT,(folder/'graph.json').relative_to(ROOT).as_posix())
        record={'graph':graph,**{k:pin(ROOT,(folder/f).relative_to(ROOT).as_posix()) for k,f in
            [('roles','joint-roles.png'),('heads','joint-heads.json'),('links','joint-chains.json'),('prior_row','result.json')]}}
        views.append({**v,'structural_inputs':record})
    return {**job,'case':{**job['case'],'split':'development'},'views':views,'new_control':False}


def load(job,v,folder=None):
    calls={k:0 for k in ('graph_reads','geometry_extractions','advanced_head_calls','baseline_profile_calls','new_head_calls','joint_role_evaluations')}
    gray=np.array(Image.open(checked(ROOT,v['gray'])).convert('L'))
    if v.get('structural_inputs'):
        record=v['structural_inputs'];graph=read(checked(ROOT,record['graph']));roles=np.array(Image.open(checked(ROOT,record['roles'])))
        heads=read(checked(ROOT,record['heads']));links=read(checked(ROOT,record['links']));calls['graph_reads']=1
    else:
        gray,graph,base,packets_heads,packets,result,checks=prepare_baseline(job,v)
        roles=packets['joint']['pixel_roles'];heads=packets_heads['joint'];links=packets['joint']['dimension_chains']
        for key in calls:calls[key]=checks['calls'].get(key,0)
        calls['joint_role_evaluations']=result['role_evaluations']
        if not result['width_contract']['complete'] or packets['joint']['search_incomplete']:raise ValueError('Incomplete baseline producer')
        if folder is None:raise ValueError('New baseline needs a frozen source folder')
        folder.mkdir(parents=True,exist_ok=False)
        write(folder/'graph.json',graph);write(folder/'heads.json',heads);write(folder/'links.json',links)
        Image.fromarray(roles).save(folder/'roles.png')
        record={k:pin(ROOT,(folder/f).relative_to(ROOT).as_posix()) for k,f in
            [('graph','graph.json'),('roles','roles.png'),('heads','heads.json'),('links','links.json')]}
    checked(ROOT,record['roles']);authority=cv_authority(gray,roles,record['roles'])
    return gray,graph,roles,heads,links,record,authority,calls


def revision_score(job,truth,negative,old,new):
    old=to_native(old,job['case']['size_px']);new=to_native(new,job['case']['size_px']);delta=new!=old
    valid=(truth>=1)&(truth<=9) if truth is not None else np.zeros_like(delta)
    unknown=truth==255 if truth is not None else np.ones_like(delta)
    return {'withdrawn_pixels':int(delta.sum()),
        'wrong_roles_removed':int((delta&valid&(old!=truth)&(new==255)).sum()) if truth is not None else 0,
        'known_correct_roles_lost':int((delta&valid&(old==truth)).sum()) if truth is not None else 0,
        'new_wrong_roles':int((delta&valid&(new!=255)&(new!=truth)).sum()) if truth is not None else 0,
        'noise_roles_removed':int((delta&unknown&(negative>0)&(new==255)).sum()) if negative is not None else 0,
        'new_noise_roles':int((delta&unknown&(negative>0)&(new!=255)).sum()) if negative is not None else 0,
        'unscored_withdrawals':int((delta&unknown&(~(negative>0) if negative is not None else True)).sum())}


def qualify():
    cv2.setNumThreads(1);cv2.setRNGSeed(0);QUAL.mkdir(exist_ok=False);jobs=[]
    for j in read(checked(ROOT,CONTROLS))['jobs']:
        if j['case']['split']!='train':continue
        views=[]
        for v in j['views']:
            gray,graph,roles,heads,links,record,authority,calls=load(j,v,QUAL/'bank'/j['case']['id']/v['mode'])
            views.append({**v,'structural_inputs':record,'capture_preparation':calls})
        jobs.append({**j,'views':views})
    prior=read(ROOT/'artifacts/joint-cv-profile-v1/plan.json')
    dev=[old_job(j) for j in prior['jobs'] if j['case']['scope']=='partial_real_reviewed_ink' or j['new_control']]
    rows=[]
    for j in jobs+dev:
        native=None
        for v in j['views']:
            gray,graph,roles,heads,links,record,authority,calls=load(j,v);result=analyze(gray,graph,roles,authority=authority)
            if native is None:native=gray.copy()
            truth,negative=truth_for(j,native)
            rows.append({'case':j['case']['id'],'mode':v['mode'],'split':'train' if j in jobs else 'development',
                'revisions':{a:revision_score(j,truth,negative,roles,result[a]) for a in ARMS},
                'complete':result['complete'],'decisions':result['decisions'],'audit':result['audit']})
    write(QUAL/'train-index.json',{'jobs':jobs,'validation_predictions_opened':False})
    write(QUAL/'report.json',{'rows':rows,'validation_predictions_opened':False,
        'model':pin(ROOT,'src/cad_recovery/grid_role_revision_v1.py')})
    for split in ('train','development'):
        print({'split':split,'metrics':{a:{k:sum(r['revisions'][a][k] for r in rows if r['split']==split)
            for k in ('withdrawn_pixels','wrong_roles_removed','known_correct_roles_lost','noise_roles_removed','unscored_withdrawals')} for a in ARMS},
            'incomplete':sum(not r['complete'] for r in rows if r['split']==split)},flush=True)


def freeze():
    checked(ROOT,PREVIOUS);prior=read(ROOT/'artifacts/joint-cv-profile-v1/plan.json');control=read(checked(ROOT,CONTROLS))
    for p in prior['code']:checked(ROOT,p)
    checked(ROOT,read(QUAL/'report.json')['model'])
    jobs=[old_job(j) for j in prior['jobs']]+read(QUAL/'train-index.json')['jobs']
    jobs += [j for j in control['jobs'] if j['case']['split']=='validation']
    names=[p['path'] for p in prior['code']]+['src/cad_recovery/grid_role_revision_v1.py','scripts/run_grid_role_v1.py',
        'scripts/prepare_grid_role_controls_v1.py','tests/test_grid_role_revision_v1.py']
    BASE.mkdir(exist_ok=False)
    write(BASE/'plan.json',{'task':'P10.2c.3b.2h.1','previous':PREVIOUS,'controls':CONTROLS,'jobs':jobs,
        'code':[pin(ROOT,n) for n in dict.fromkeys(names)],'arms':ARMS,'criteria':control['criteria'],'budgets':control['budgets'],
        'qualification':pin(ROOT,'artifacts/grid-role-qualification-v1/report.json'),
        'fresh_validation_predictions_before_freeze':False,'historical_maps_immutable':True,
        'inference_uses_GT_for_authority':False,'production_changed':False,'physical_geometry_approved':False})
    print(pin(ROOT,'artifacts/grid-role-profile-v1/plan.json'),flush=True)


def child(path):
    cv2.setNumThreads(1);cv2.setRNGSeed(0);request=read(path);plan=read(checked(ROOT,request['plan']))
    for p in plan['code']:checked(ROOT,p)
    j=request['job'];case=j['case'];out=path.parent/'output';out.mkdir();rows=[];native=None
    for v in j['views']:
        folder=out/v['mode'];folder.mkdir()
        gray,graph,roles,heads,links,record,authority,calls=load(j,v,folder/'baseline-source')
        before=object_hash([graph,heads,links]);gray_sha=sha(gray);role_sha=sha(roles);mask_sha=sha(authority['mask'])
        result=analyze(gray,graph,roles,authority=authority)
        # Inference and the authority decision finish before target access.
        if native is None:native=gray.copy()
        truth,negative=truth_for(j,native);metrics={};revisions={};observed={};protected=authority['mask']==2
        for a in ARMS:
            labels=result[a];image=to_native(labels,case['size_px']);metrics[a]=score(truth,image,negative) if truth is not None else None
            revisions[a]=revision_score(j,truth,negative,roles,labels);delta=labels!=roles
            observed[a]={'only_hatching_to_unknown':bool(np.all((roles[delta]==9)&(labels[delta]==255))),
                'protected_pixels_exact':bool(np.array_equal(labels[protected],roles[protected])),'assigned_new_classes':False}
            Image.fromarray(labels).save(folder/(a+'-roles.png'))
        Image.fromarray(result['revision_owner']).save(folder/'revision-owner.png');Image.fromarray(authority['mask']).save(folder/'authority.png')
        write(folder/'authority.json',{k:z for k,z in authority.items() if k!='mask'});write(folder/'decisions.json',result['decisions'])
        prior=read(checked(ROOT,record['prior_row'])) if record.get('prior_row') else None
        rows.append({**{k:case[k] for k in ('scope','group_id','split')},'case':case['id'],'mode':v['mode'],'repeat':request['repeat'],
            'new_control':j['new_control'],'roles':metrics,'revisions':revisions,'view_checks':observed,
            'source_graph_heads_and_links_exact':before==object_hash([graph,heads,links]),'gray_exact':gray_sha==sha(gray),
            'historical_role_map_exact':role_sha==sha(roles) and np.array_equal(result['baseline'],roles),'authority_mask_exact':mask_sha==sha(authority['mask']),
            'old_role_metrics_exact':not prior or metrics['baseline']==prior['roles']['joint'],
            'ink_ledgers':{a:ink_ledger(gray,result[a],foreground) for a in ARMS},
            'authority':{'source_sha256':gray_sha,'role_sha256':role_sha,'mask_sha256':mask_sha,'origin':'frozen_cv_inference',
                'protected_pixels':int(protected.sum()),'GT_used_for_authority':False},
            'complete':result['complete'],'audit':result['audit'],'preparation':calls,
            'production_changed':False,'physical_geometry_approved':False,'source_pixels_erased':0})
        write(folder/'result.json',rows[-1])
    write(out/'result.json',{'rows':rows})


def run(sha256):
    from experiment_children_v1 import run as execute
    e=execute(ROOT,BASE,{'path':'artifacts/grid-role-profile-v1/plan.json','sha256':sha256},Path(__file__).resolve())
    rows=[read(ROOT/r['artifact']) for r in read(BASE/'results-index.json')]
    e.update(revision_calls=len(rows),preparation_calls={k:sum(r['preparation'][k] for r in rows) for k in rows[0]['preparation']},
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
