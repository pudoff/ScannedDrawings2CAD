"""Freeze and compare contextual stroke ownership on shared anchored geometry."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/stroke-ownership-v1'
PREVIOUS={'path':'artifacts/anchored-fit-release-v1.json','sha256':'7b87bee56be57ccf6913e1a5109095c6d1e6099a3d3f7bef40fb217664cb79f1'}
CORE={'path':'artifacts/cv-comparison-protocol-v1/contract.json','sha256':'815c8a5ee7f47714066206b2a7a55cbdd94328bb75fdc2ce5383b02ca300d901'}


def read(path):return json.loads(path.read_text('utf-8'))
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def freeze():
    from verify_cv_comparison_contract_v1 import verify_contract
    from cad_recovery.dataset_layout_v1 import checked,pin
    from cad_recovery.stroke_ownership_v1 import POLICY
    from prepare_ownership_controls_v1 import prepare,SEED,FAMILIES
    verify_contract(ROOT,CORE);checked(ROOT,PREVIOUS)
    if BASE.exists():raise FileExistsError(BASE)
    old=read(ROOT/'artifacts/stroke-roles-v1/plan.json');head_rows=read(ROOT/'artifacts/stroke-roles-v1/results.json')
    geometry_rows=read(ROOT/'artifacts/anchored-fit-v1/results-index.json')
    heads={(r['case'],r['mode']):r for r in head_rows if not r['repeat']}
    geometry={(r['case'],r['mode']):r for r in geometry_rows if not r['repeat'] and r['text_mode']=='untouched'}
    jobs=[]
    for old_job in old['jobs']:
        case=old_job['case'];views=[];truth=None
        if case['scope']=='partial_real_reviewed_ink':truth=pin(ROOT,f'artifacts/cv-comparison-inputs-v2/cases/{case["id"]}/parent-labels.png')
        for v in old_job['views']:
            row=geometry[(case['id'],v['mode'])];folder=(ROOT/row['artifact']).parent
            views.append({'mode':v['mode'],'gray':v['gray'],'input_to_native':v['input_to_native'],
                'reviewed_mask':v['reviewed_mask'],'cache':{'geometry_row':pin(ROOT,row['artifact']),
                    'candidates':pin(ROOT,(folder/'candidates.json').relative_to(ROOT).as_posix()),
                    'heads':pin(ROOT,heads[(case['id'],v['mode'])]['artifact'])}})
        negative=pin(ROOT,'artifacts/noise-baseline-v3/noise-baseline-ink.png') if case['scope']=='known_fold_diagnostic' else None
        jobs.append({'case':case,'views':views,'truth':truth,'negative':negative,'new_control':False})
    BASE.mkdir();jobs.extend(prepare(ROOT,BASE))
    names=['src/cad_recovery/stroke_ownership_v1.py','src/cad_recovery/ownership_contours_v1.py',
        'src/cad_recovery/stroke_ownership_metrics_v1.py','scripts/prepare_ownership_controls_v1.py',
        'scripts/run_stroke_ownership_v1.py','tests/test_stroke_ownership_v1.py',
        'src/cad_recovery/stroke_roles_v1.py','src/cad_recovery/stroke_roles_v2.py','src/cad_recovery/native_head_bounds_v1.py',
        'src/cad_recovery/anchored_curve_fit_v1.py','src/cad_recovery/primitive_graph_qa_v1.py','scripts/resume_centerline_comparison_v2.py']
    plan={'task':'P10.2c.3b.1','previous':PREVIOUS,'core':CORE,'code':[pin(ROOT,n) for n in names],
        'policy':POLICY,'jobs':jobs,'reading':old['reading'],'new_controls':{'seed':SEED,'families':FAMILIES,'scenes':24},
        'variants':['width_only','old_context','contour_context','protected_context'],'repeats':2,
        'prepared_before_new_predictions':True,'validation_policy':'family split fixed; no parameters selected on viewed validation',
        'baseline_scope':'old v1 role classifier evaluated on the same anchored geometry and width observations; historical scores are not direct baselines',
        'input_policy':'no reviewed role/text labels enter classifier; oracle text masks only score old synthetic truth',
        'DT_policy':'not re-admitted; novel DT observations are retained and context-gated diagnostically, never supplied to role classifier',
        'new_heads_policy':'safe stroke_roles_v2.analyze in an isolated single child only',
        'gates':{'synthetic_macro_F1_min':.90,'maximum_lost_assigned_heads_after_context_guard':0,
            'maximum_source_or_unknown_erased_pixels':0,'semantic_CAD_promotions':0,
            'full_real_precision':'not_evaluable_from_partial_labels','unassigned_real_ink_is_background':False},
        'budgets':{'child_seconds':60,'RSS_MiB':1024,'output_MiB':64,'files':128,'total_child_seconds':1800,'CPU_threads':1},
        'training':False,'new_OCR_ML_calls':0,'closed_test_used':False,'physical_geometry_approved':False,'production_changed':False}
    write(BASE/'plan.json',plan);print(json.dumps(pin(ROOT,'artifacts/stroke-ownership-v1/plan.json')))


def widths_for(candidates):
    import numpy as np
    rows=[]
    for c in candidates:
        w=c.get('local_width',{});local=w.get('local_observations',[])
        rows.append({'id':c['id'],'width_px':w.get('width_px'),
            'coverage':w.get('coverage',float(np.mean([p['coverage'] for p in local])) if local else 0.),
            'heterogeneous':w.get('heterogeneous',any(p['heterogeneous'] for p in local))})
    return rows


def child(request_path):
    sys.path.insert(0,str(ROOT/'artifacts/cv-skeleton-runtime-v1/deps'))
    import cv2
    import numpy as np
    from PIL import Image
    from skimage.morphology import skeletonize
    from cad_recovery.dataset_layout_v1 import checked
    from cad_recovery.anchored_curve_fit_v1 import extract
    from cad_recovery.stroke_roles_v1 import foreground,stroke_floor,text_components,guard_heads,role_proposals,append_heads
    from cad_recovery.stroke_roles_v2 import analyze as safe_heads
    from cad_recovery.stroke_ownership_v1 import analyze,head_context_gate
    from cad_recovery.stroke_ownership_metrics_v1 import score_roles,ink_ledger
    from cad_recovery.stroke_role_metrics_v1 import synthetic_truth,to_native,head_scores,mask_effect
    from run_stroke_roles_v1 import refs_for
    request=read(request_path);plan=read(checked(ROOT,request['plan']));job=request['job'];case=job['case']
    for p in plan['code']:checked(ROOT,p)
    reading=read(checked(ROOT,plan['reading']));refs,negatives=refs_for(case,reading)
    folder=request_path.parent/'output';folder.mkdir();rows=[];native_gray=None;truth=None;negative=None
    for view in job['views']:
        cv2.setNumThreads(1);cv2.setRNGSeed(0)
        with Image.open(checked(ROOT,view['gray'])) as im:gray=np.array(im)
        before=hashlib.sha256(gray.tobytes()).hexdigest()
        if native_gray is None:native_gray=gray.copy()
        head_guard_events=[];head_seconds=None
        if view['cache']:
            cache=view['cache'];old=read(checked(ROOT,cache['geometry_row']));h=read(checked(ROOT,cache['heads']))
            if old['input_pixel_sha256']!=before or h['input_pixel_sha256']!=before:raise ValueError('Input hash rebound')
            geometry=read(checked(ROOT,cache['candidates']))['anchored'];base_heads=h['head_candidates']['existing_head_completion']
            extra_heads=[a for a in h['head_candidates']['plus_DT'] if a['id'].startswith('DT-added-')]
            head_incomplete=h['base_head_search_incomplete']
        else:
            ink=foreground(gray);skeleton=skeletonize(ink>0,method='zhang').astype(np.uint8)*255
            geometry=extract(ink,skeleton,conditioned=False,input_to_native=view['input_to_native'])['candidates']
            observed=widths_for(geometry);configured_scale=stroke_floor(observed) or 3.
            start=time.perf_counter();h=safe_heads(gray,geometry,observed,stroke_px=configured_scale);head_seconds=time.perf_counter()-start
            base_heads=h['base_heads']['candidates'];combined=append_heads(base_heads,h['DT_proposals']['candidates'])
            extra_heads=[a for a in combined if a['id'].startswith('DT-added-')]
            head_guard_events=h['guard_events'];head_incomplete=h['base_heads']['search_incomplete']
        widths=widths_for(geometry);stroke=stroke_floor(widths);text=text_components(gray,stroke or 3.)
        base_kept,_=guard_heads(base_heads,text['component_mask']);start=time.perf_counter()
        old_width=role_proposals(gray,geometry,widths,[],text['component_mask'],context=False)
        old_context=role_proposals(gray,geometry,widths,base_kept,text['component_mask'],context=True)
        contour=analyze(gray,geometry,widths,base_heads,protected=False)
        protected=analyze(gray,geometry,widths,base_heads,protected=True)
        elapsed=time.perf_counter()-start
        variants={'width_only':old_width,'old_context':old_context,'contour_context':contour,'protected_context':protected}
        if truth is None:
            if job['truth']:
                with Image.open(checked(ROOT,job['truth'])) as im:truth=np.array(im)
            elif case['scope']=='analytical_control':
                with Image.open(checked(ROOT,view['reviewed_mask'])) as im:reviewed=np.array(im)
                truth=synthetic_truth(case,native_gray,reviewed)
        if negative is None and job['negative']:
            with Image.open(checked(ROOT,job['negative'])) as im:negative=np.array(im)
        if truth is None and negative is not None:truth=np.full(tuple(case['size_px'][::-1]),255,np.uint8)
        sub=folder/view['mode'];sub.mkdir();scores={};ledgers={};details={}
        for name,value in variants.items():
            pixel=value['pixel_roles'];Image.fromarray(pixel).save(sub/(name+'-roles.png'))
            scores[name]=score_roles(truth,to_native(pixel,case['size_px']),negative) if truth is not None else None
            ledgers[name]=ink_ledger(gray,pixel,foreground)
            details[name]={k:v for k,v in value.items() if k not in ('pixel_roles','text_mask')}
        Image.fromarray(protected['text_mask']).save(sub/'protected-text-mask.png')
        Image.fromarray(text['component_mask']).save(sub/'old-text-mask.png')
        original_ids={a['id'] for a in protected['head_events']};new_kept=[a for a in base_heads if a['id'] not in original_ids]
        dt_kept,dt_events=head_context_gate(extra_heads,protected['text_mask'],protected['hatch_bundles'],stroke or 3.)
        head_metrics={name:head_scores(refs,negatives,candidates,view['input_to_native'],dense=job['new_control'] or case['scope']=='analytical_control',noise_mask=negative)
            for name,candidates in [('baseline',base_heads),('old_text_guard',base_kept),('protected_guard',new_kept)]}
        text_effect={name:mask_effect(truth,native_gray,to_native(mask,case['size_px'])) for name,mask in
            [('old',text['component_mask']),('protected',protected['text_mask'])]} if truth is not None else None
        row={'case':case['id'],'scope':case['scope'],'split':case['split'],'group_id':case['group_id'],'new_control':job['new_control'],
            'mode':view['mode'],'repeat':request['repeat'],'input_pixel_sha256':before,'input_to_native':view['input_to_native'],
            'roles':scores,'details':details,'ink_ledgers':ledgers,'heads':head_metrics,
            'head_candidates':{'baseline':base_heads,'DT_novel':extra_heads},'DT_context_events':dt_events,
            'DT_novel_unblocked_ids':[a['id'] for a in dt_kept],'DT_admitted_for_roles':False,
            'head_guard_events':head_guard_events,'head_search_incomplete':head_incomplete,'text_mask_effect':text_effect,
            'source_pixel_sha256_after':hashlib.sha256(gray.tobytes()).hexdigest(),
            'timing':{'new_head_search_seconds':head_seconds,'classification_seconds':elapsed},
            'source_pixels_erased':0,'automatic_semantic_CAD_promotions':0,'training':False,'production_admitted':False}
        write(sub/'result.json',row);write(sub/'geometry.json',geometry);rows.append(row)
    write(folder/'result.json',{'rows':rows})


def canonical(row):return {k:v for k,v in row.items() if k not in ('repeat','timing','artifact','runtime')}


def run(expected):
    from verify_cv_comparison_contract_v1 import verify_contract
    from cad_recovery.dataset_layout_v1 import checked,pin
    from cad_recovery.service_profiles_v2.budget import AttemptBudget
    from resume_centerline_comparison_v2 import StableOutputCounter
    plan=read(checked(ROOT,{'path':'artifacts/stroke-ownership-v1/plan.json','sha256':expected}));verify_contract(ROOT,plan['core']);checked(ROOT,plan['previous'])
    index=[];spent=0;prior={};checks=[];started=time.perf_counter()
    for split in ('train','validation','development'):
        for p in plan['code']:checked(ROOT,p)
        write(BASE/(split+'-parameter-gate.json'),{'plan_sha256':expected,'parameter_changes':False})
        for job in [j for j in plan['jobs'] if j['case']['split']==split]:
            for repeat in range(2):
                if spent>=1800:raise RuntimeError('Experiment wall budget exhausted')
                folder=BASE/'runs'/f'{job["case"]["id"]}-r{repeat}';folder.mkdir(parents=True,exist_ok=False)
                write(folder/'request.json',{'job':job,'repeat':repeat,'plan':pin(ROOT,'artifacts/stroke-ownership-v1/plan.json')})
                env={**os.environ,'PYTHONIOENCODING':'utf-8','PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1',
                    'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','XDG_CACHE_HOME':str(ROOT/'artifacts/.cache')}
                with (folder/'child.log').open('xb') as log:
                    p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'child','--request',str(folder/'request.json')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0,start_new_session=os.name!='nt')
                    budget=AttemptBudget(p,folder,StableOutputCounter(),SimpleNamespace(timeout_seconds=min(60,1800-spent),max_output_bytes=64*1024**2,max_output_files=128),memory_MiB=1024)
                    try:
                        while p.poll() is None:budget.check();time.sleep(.05)
                    finally:budget.stop()
                runtime=budget.measurements();spent+=runtime['walltime_seconds'];write(folder/'process.json',runtime)
                if p.returncode!=0 or runtime['budget_error']:raise RuntimeError('Child failed: '+folder.name)
                for row in read(folder/'output/result.json')['rows']:
                    key=(row['case'],row['mode'])
                    if repeat==0:prior[key]=canonical(row)
                    else:checks.append({'case':row['case'],'mode':row['mode'],'identical':prior[key]==canonical(row)})
                    index.append({k:row[k] for k in ('case','scope','split','group_id','new_control','mode','repeat')}|
                        {'artifact':(folder/'output'/row['mode']/'result.json').relative_to(ROOT).as_posix(),'runtime':runtime})
            print(json.dumps({'completed':job['case']['id'],'split':split,'child_wall_seconds':round(spent,2)}),flush=True)
    write(BASE/'results-index.json',index);write(BASE/'repeat-checks.json',checks)
    write(BASE/'execution.json',{'cases':len(plan['jobs']),'children':len(plan['jobs'])*2,'views':len(index),'canonical_views':len(checks),
        'exact_repeat_checks':len(checks),'all_repeat_identical':all(c['identical'] for c in checks),
        'child_wall_seconds':spent,'wall_seconds':time.perf_counter()-started,'fresh_head_and_geometry_views':96,
        'cached_head_and_geometry_views':182,'direct_role_classifier_calls':len(index)*4,'legacy_wrapper_role_calls':96,
        'new_OCR_ML_calls':0,'training':False,'closed_test_used':False,'production_changed':False})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['freeze','run','child']);p.add_argument('--sha256');p.add_argument('--request',type=Path)
    args=p.parse_args()
    if args.mode=='freeze':freeze()
    elif args.mode=='child':child(args.request)
    else:
        if not args.sha256:p.error('--sha256 required')
        run(args.sha256)
