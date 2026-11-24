"""Frozen independent role, DT-head and text-mask experiments on approved crops."""
import time
CPU_START=time.process_time()
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/stroke-roles-v1'
CORE={'path':'artifacts/cv-comparison-protocol-v1/contract.json','sha256':'815c8a5ee7f47714066206b2a7a55cbdd94328bb75fdc2ce5383b02ca300d901'}


def read(path):return json.loads(path.read_text('utf-8'))
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def freeze():
    from verify_cv_comparison_contract_v1 import verify_contract
    from cad_recovery.dataset_layout_v1 import checked,pin
    from cad_recovery.stroke_roles_v1 import POLICY
    verify_contract(ROOT,CORE)
    if BASE.exists():raise FileExistsError(BASE)
    previous={'path':'artifacts/centerline-comparison-release-v1.json','sha256':'5a72ade7a20d181e32e4bbedb8046c7ed6df965046c21d9864aed676dd1c0b36'}
    checked(ROOT,previous)
    cases=read(ROOT/'artifacts/cv-comparison-inputs-v2/effective-cases.json')
    results=read(ROOT/'artifacts/centerline-comparison-v1/results.json')
    lookup={(r['case'],r['variant'],r['mode'],r['text_mode']):r for r in results if not r['failed'] and r['repeat']==0}
    jobs=[]
    for case in cases:
        views=[]
        for v in case['variants']:
            cache={}
            for name,variant,text in [('baseline','baseline','untouched'),('reviewed','baseline','common_reviewed_text_mask'),('zhang','zhang','untouched')]:
                row=lookup[(case['id'],variant,v['mode'],text)]
                cache[name]=pin(ROOT,row['artifact'])
            views.append({**v,'gray':pin(ROOT,'artifacts/cv-comparison-inputs-v2/'+v['gray']),
                'reviewed_mask':pin(ROOT,'artifacts/cv-comparison-inputs-v2/'+v['common_text_mask']),'cache':cache})
        jobs.append({'case':case,'views':views})
    jpeg=ROOT/'artifacts/e2e-acceptance-v1/known-folds-jpeg/automatic'
    jobs.append({'case':{'id':'known-folds-jpeg','split':'development','group_id':'viewed-jpeg-development',
            'scope':'known_fold_diagnostic','size_px':[459,618]},
        'views':[{'mode':'native_crop','size_px':[459,618],'input_to_native':[[1,0,0],[0,1,0],[0,0,1]],
            'gray':pin(ROOT,(jpeg/'page-001-processed.png').relative_to(ROOT).as_posix()),
            'geometry':pin(ROOT,(jpeg/'page-001-geometry.json').relative_to(ROOT).as_posix()),
            'reviewed_mask':None,'cache':{}}]})
    names=['scripts/run_stroke_roles_v1.py','src/cad_recovery/stroke_roles_v1.py',
        'src/cad_recovery/stroke_role_metrics_v1.py','src/cad_recovery/arrow_head_completion_v1.py',
        'src/cad_recovery/arrow_path_recovery_v1.py','src/cad_recovery/arrow_detection.py',
        'src/cad_recovery/local_stroke_evidence_v1.py','src/cad_recovery/geometry_detection.py',
        'scripts/resume_centerline_comparison_v2.py']
    BASE.mkdir()
    plan={'task':'P10.2c.3b','core':CORE,'previous':previous,'cases':pin(ROOT,'artifacts/cv-comparison-inputs-v2/effective-cases.json'),
        'jobs':jobs,'code':[pin(ROOT,p) for p in names],
        'reading':pin(ROOT,'artifacts/cv-comparison-inputs-v2/inherited-stroke-reading.json'),
        'prepared_before_predictions':True,'repeats':2,'parameters_selected_without_validation':True,'role_policy':POLICY,
        'experiments':{'heads':['existing_head_completion','plus_DT','plus_DT_auto_text_guard','plus_DT_reviewed_text_guard'],
            'roles':['width_only','context_base','context_DT','context_zhang'],
            'automatic_text':['untouched','components','bbox'],
            'reviewed_text':['untouched','reviewed_components','reviewed_bbox']},
        'roles_input':'automatic text groups only; reviewed labels never used by role_proposals',
        'oracle_scope':'reviewed text mask/bbox are localization controls only, not automatic methods',
        'text_geometry':['filter existing LINE if >=50% masked samples','fresh baseline extraction after mask'],
        'gates':{'maximum_lost_assigned_heads':0,'maximum_added_false_heads_in_negative_ROI':0,
            'minimum_new_head_targets_for_DT':1,'synthetic_head_precision_min':.98,'synthetic_role_macro_F1_min':.90,
            'maximum_known_nontext_or_unknown_suppressed_pixels':0,'production_admission':False,
            'real_unknowns':'not background; unassigned ink suppression requires review'},
        'budgets':{'child_seconds':60,'RSS_MiB':1024,'output_MiB':64,'files':128,'total_child_seconds':1800,
            'OpenCV_threads':1,'RNG_seed':0},
        'source_roles':'partial assigned ink on 25 real crops; constructed role truth on procedural scenes; aliases scoped heads only',
        'source_pixels_erased':False,'closed_test_used':False,'new_OCR_ML_calls':0,
        'full_page_windows':False,'physical_geometry_approved':False}
    write(BASE/'plan.json',plan);print(json.dumps(pin(ROOT,(BASE/'plan.json').relative_to(ROOT).as_posix())))


def refs_for(case,reading):
    refs=[];negative=[]
    old=next((p for p in reading['patches'] if p['patch_id']==case['id']),None)
    if old:refs.extend(old['arrows']);negative.extend(old['negatives'])
    for a in case.get('arrow_controls',[]):
        if a['state']=='visible_head':refs.append({'id':a['id'],'tip':a['tip_px'],'direction':a['direction']})
        elif a['state']=='negative_region':negative.append(a)
    if case['scope']=='analytical_control':
        refs.extend(case['labels']['arrows']);negative.extend(case['labels']['negative_regions'])
    if 'reading' in case:
        refs.extend(case['reading']['arrows']);negative.extend(case['reading']['negative_regions'])
    return refs,negative


def cached(view,key):
    result=read(ROOT/view['cache'][key]['path'])
    text='common_reviewed_text_mask' if key=='reviewed' else 'untouched'
    return next(r for r in result['rows'] if r['mode']==view['mode'] and r['text_mode']==text)


def child(request_path):
    import cv2
    import numpy as np
    from PIL import Image
    from cad_recovery.dataset_layout_v1 import checked
    from cad_recovery.stroke_roles_v1 import (text_components,dt_heads,append_heads,guard_heads,role_proposals,stroke_floor,foreground)
    from cad_recovery.stroke_role_metrics_v1 import synthetic_truth,to_native,role_scores,mask_effect,head_scores
    from cad_recovery.arrow_head_completion_v1 import search
    from cad_recovery.local_stroke_evidence_v1 import line_width
    from cad_recovery.geometry_detection import detect_geometry
    from cad_recovery.centerline_comparison_v1 import line_scores
    request=read(request_path);job=request['job'];case=job['case'];plan=read(ROOT/request['plan']['path'])
    reading=read(ROOT/plan['reading']['path']);references,negative=refs_for(case,reading)
    folder=request_path.parent/'output';folder.mkdir();rows=[]
    native_gray=None;truth=None
    if case['scope']=='partial_real_reviewed_ink':
        with Image.open(ROOT/f'artifacts/cv-comparison-inputs-v2/cases/{case["id"]}/parent-labels.png') as im:truth=np.array(im)
    for view in job['views']:
        cv2.setNumThreads(1);cv2.setRNGSeed(0)
        with Image.open(checked(ROOT,view['gray'])) as im:gray=np.array(im)
        if native_gray is None:native_gray=gray.copy()
        if view['reviewed_mask'] is not None:
            with Image.open(checked(ROOT,view['reviewed_mask'])) as im:reviewed=np.array(im)
        else:reviewed=np.zeros_like(gray)
        if case['scope']=='known_fold_diagnostic':
            report=read(checked(ROOT,view['geometry']));geometry=report['candidates'];ink=foreground(gray)
            dt=cv2.distanceTransform(ink,cv2.DIST_L2,5)
            widths=[{'id':g['id'],**line_width(g['points_px'],ink,dt)} for g in geometry if g['kind']=='line']
            base={'candidates':geometry,'widths':widths,'stats':{'analysis':report['analysis']}}
        else:
            for record in view['cache'].values():checked(ROOT,record)
            base=cached(view,'baseline');geometry=base['candidates'];widths=base['widths']
            if base['input_pixel_sha256']!=hashlib.sha256(gray.tobytes()).hexdigest():raise ValueError('Cache input rebound')
        stroke=stroke_floor(widths)
        if stroke is None:stroke=max(1,min(24,float(base['stats']['analysis']['estimated_stroke_width_px'])))
        start=time.perf_counter()
        original=search(gray,lines=[g for g in geometry if g['kind']=='line'],arcs=[g for g in geometry if g['kind']=='arc'],stroke_px=stroke)
        additional=dt_heads(gray,stroke);combined=append_heads(original['candidates'],additional['candidates'])
        text=text_components(gray,stroke)
        auto,excluded_auto=guard_heads(combined,text['component_mask'])
        original_auto,_=guard_heads(original['candidates'],text['component_mask'])
        oracle,excluded_oracle=guard_heads(combined,reviewed)
        noise=None
        if case['scope']=='known_fold_diagnostic':
            with Image.open(ROOT/'artifacts/noise-baseline-v3/noise-baseline-ink.png') as im:noise=np.array(im)
        head_variants={'existing_head_completion':original['candidates'],'plus_DT':combined,
            'plus_DT_auto_text_guard':auto,'plus_DT_reviewed_text_guard':oracle}
        head_metrics={k:head_scores(references,negative,v,view['input_to_native'],dense=case['scope']=='analytical_control',noise_mask=noise)
                      for k,v in head_variants.items()}
        sub=folder/view['mode'];sub.mkdir()
        Image.fromarray(text['component_mask']).save(sub/'auto-components.png')
        Image.fromarray(text['bbox_mask']).save(sub/'auto-bbox.png')
        role_configs={'width_only':(base,[],False),'context_base':(base,original_auto,True),'context_DT':(base,auto,True)}
        if case['scope']!='known_fold_diagnostic':role_configs['context_zhang']=(cached(view,'zhang'),auto,True)
        if case['scope']=='analytical_control' and truth is None:truth=synthetic_truth(case,native_gray,reviewed)
        role_metrics={};role_details={}
        for name,(packet,arrows,context) in role_configs.items():
            result=role_proposals(gray,packet['candidates'],packet['widths'],arrows,text['component_mask'],context=context)
            Image.fromarray(result['pixel_roles']).save(sub/(name+'-roles.png'))
            role_metrics[name]=role_scores(truth,to_native(result['pixel_roles'],case['size_px'])) if truth is not None else None
            role_details[name]={k:v for k,v in result.items() if k!='pixel_roles'}
        bbox_native=np.zeros(tuple(case['size_px'][::-1]),np.uint8)
        boxes=[a['box'] for a in case.get('annotations',[]) if a.get('role')=='text' and a.get('kind')=='box']
        if 'reading' in case:boxes.extend(case['reading']['text_regions'])
        if case['scope']=='analytical_control':
            native_mask=next(v for v in case['variants'] if v['mode']=='native_crop')['common_text_mask']
            with Image.open(ROOT/'artifacts/cv-comparison-inputs-v2'/native_mask) as im:full_text=np.array(im)
            yy,xx=np.nonzero(full_text)
            if len(xx):boxes.append([int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)])
        for x,y,r,b in boxes:bbox_native[y:b,x:r]=255
        reviewed_bbox=cv2.resize(bbox_native,gray.shape[::-1],interpolation=cv2.INTER_NEAREST)
        masks={'components':text['component_mask'],'bbox':text['bbox_mask'],'reviewed_components':reviewed,'reviewed_bbox':reviewed_bbox}
        text_metrics={}
        for name,mask in masks.items():
            derived=gray.copy();derived[mask>0]=255
            Image.fromarray(derived).save(sub/(name+'-gray.png'))
            filtered=[]
            for g in geometry:
                if g['kind']!='line':filtered.append(g);continue
                p=np.array(g['points_px']);q=np.rint(p[0]+np.linspace(0,1,64)[:,None]*(p[1]-p[0])).astype(int)
                valid=(q>=0).all(axis=1)&(q[:,0]<gray.shape[1])&(q[:,1]<gray.shape[0])
                if valid.any() and np.mean(mask[q[valid,1],q[valid,0]]>0)<.5:filtered.append(g)
            if name=='reviewed_components' and case['scope']!='known_fold_diagnostic':fresh=cached(view,'reviewed')
            else:
                source=sub/(name+'-gray.png');det=detect_geometry(source,sub/(name+'-fresh'),prefix='roles',search_profile='regional_masked')
                ink=foreground(derived);dt=cv2.distanceTransform(ink,cv2.DIST_L2,5)
                fresh={'candidates':det['candidates'],'widths':[{'id':g['id'],**line_width(g['points_px'],ink,dt)} for g in det['candidates'] if g['kind']=='line']}
            metrics={'filtered_line_count':sum(g['kind']=='line' for g in filtered),
                'fresh_line_count':sum(g['kind']=='line' for g in fresh['candidates']),
                'filter_only':line_scores(case,{'candidates':filtered,'widths':widths},view['input_to_native']),
                'reextract':line_scores(case,fresh,view['input_to_native'])}
            if truth is not None:metrics['pixel_effect']=mask_effect(truth,native_gray,to_native(mask,case['size_px']))
            if case['scope']=='known_fold_diagnostic':
                with Image.open(ROOT/'artifacts/noise-baseline-v3/useful-baseline-ink.png') as im:useful=np.array(im)
                with Image.open(ROOT/'artifacts/noise-baseline-v3/unknown-baseline-ink.png') as im:unknown=np.array(im)
                metrics['fold_preservation']={'protected_suppressed_px':int(((mask>0)&(useful>0)).sum()),
                    'unknown_suppressed_px':int(((mask>0)&(unknown>0)).sum())}
            text_metrics[name]=metrics
        row={'case':case['id'],'split':case['split'],'group_id':case['group_id'],'scope':case['scope'],
            'mode':view['mode'],'repeat':request['repeat'],'input_pixel_sha256':hashlib.sha256(gray.tobytes()).hexdigest(),
            'roles':role_metrics,'role_details':role_details,'heads':head_metrics,'head_candidates':head_variants,
            'DT_diagnostics':{k:v for k,v in additional.items() if k!='candidates'},
            'text_groups':text['groups'],'text_search_incomplete':text['search_incomplete'],'text':text_metrics,
            'base_head_search_incomplete':original['search_incomplete'],'elapsed_seconds':time.perf_counter()-start,
            'source_pixels_erased':False,'new_OCR_ML_calls':0,'production_admitted':False}
        write(sub/'result.json',row);rows.append(row)
    write(folder/'result.json',{'rows':rows,'child_CPU_seconds':time.process_time()-CPU_START})


def run(expected):
    from verify_cv_comparison_contract_v1 import verify_contract
    from cad_recovery.dataset_layout_v1 import checked,pin
    from cad_recovery.service_profiles_v2.budget import AttemptBudget
    from resume_centerline_comparison_v2 import StableOutputCounter
    plan=read(checked(ROOT,{'path':'artifacts/stroke-roles-v1/plan.json','sha256':expected}));verify_contract(ROOT,plan['core'])
    for record in plan['code']:checked(ROOT,record)
    rows=[];spent=0;repeats=[];prior={}
    for split in ('train','validation','development'):
        for record in plan['code']:checked(ROOT,record)
        write(BASE/(split+'-parameter-gate.json'),{'plan_sha256':expected,'changed':False})
        for job in [j for j in plan['jobs'] if j['case']['split']==split]:
            for repeat in range(2):
                if spent>=1800:raise RuntimeError('Total budget exhausted')
                folder=BASE/'runs'/f'{job["case"]["id"]}-r{repeat}';folder.mkdir(parents=True,exist_ok=False)
                write(folder/'request.json',{'job':job,'repeat':repeat,'plan':pin(ROOT,'artifacts/stroke-roles-v1/plan.json')})
                env={**os.environ,'PYTHONIOENCODING':'utf-8','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1',
                    'OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','XDG_CACHE_HOME':str(ROOT/'artifacts/.cache')}
                with (folder/'child.log').open('xb') as log:
                    process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'child','--request',str(folder/'request.json')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0,start_new_session=os.name!='nt')
                    budget=AttemptBudget(process,folder,StableOutputCounter(),SimpleNamespace(timeout_seconds=min(60,1800-spent),max_output_bytes=64*1024**2,max_output_files=128),memory_MiB=1024)
                    try:
                        while process.poll() is None:budget.check();time.sleep(.05)
                    finally:budget.stop()
                runtime=budget.measurements();spent+=runtime['walltime_seconds'];write(folder/'process.json',runtime)
                if process.returncode!=0 or runtime['budget_error']:raise RuntimeError('Child failed: '+folder.name)
                result=read(folder/'output/result.json')
                for row in result['rows']:
                    row['artifact']=(folder/'output'/row['mode']/'result.json').relative_to(ROOT).as_posix();row['runtime']=runtime
                    key=(row['case'],row['mode']);canonical={k:v for k,v in row.items() if k not in ('repeat','elapsed_seconds','artifact','runtime')}
                    if repeat==0:prior[key]=canonical
                    else:repeats.append({'case':row['case'],'mode':row['mode'],'identical':canonical==prior[key]})
                    rows.append(row)
            print(json.dumps({'completed':job['case']['id'],'split':split,'child_wall_seconds':spent}),flush=True)
    write(BASE/'results.json',rows);write(BASE/'repeat-checks.json',repeats)
    write(BASE/'execution.json',{'rows':len(rows),'children':len(plan['jobs'])*2,'child_wall_seconds':spent,
        'exact_repeat_checks':len(repeats),'all_repeat_identical':all(r['identical'] for r in repeats),
        'new_OCR_ML_calls':0,'production_changed':False,'closed_test_used':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['freeze','run','child']);parser.add_argument('--sha256');parser.add_argument('--request',type=Path)
    args=parser.parse_args()
    if args.mode=='freeze':freeze()
    elif args.mode=='child':child(args.request)
    else:
        if not args.sha256:parser.error('--sha256 required')
        run(args.sha256)
