"""Resume only unpublished CV jobs after the v1 temporary-pyc accounting race."""
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
from types import SimpleNamespace

import run_centerline_comparison_v1 as original
from verify_cv_comparison_contract_v1 import verify_contract
from cad_recovery.centerline_comparison_v1 import line_scores
from cad_recovery.dataset_layout_v1 import checked,pin
from cad_recovery.service_profiles_v2.budget import AttemptBudget

ROOT=original.ROOT;BASE=original.BASE
RECOVERY=ROOT/'artifacts/centerline-comparison-recovery-v1'
read=original.read;write=original.write;canonical=original.canonical


class StableOutputCounter:
    @staticmethod
    def output_size(folder):
        total=count=0
        for path in folder.rglob('*'):
            try:info=path.stat()
            except FileNotFoundError:continue
            if stat.S_ISREG(info.st_mode):total+=info.st_size;count+=1
        return total,count


if __name__=='__main__':
    if RECOVERY.exists():raise FileExistsError(RECOVERY)
    plan_pin={'path':'artifacts/centerline-comparison-v1/plan.json',
        'sha256':'c04cbef98b429e9a6726e04990a42c3f9ea889eb60a81b324f9f29e686aabd13'}
    plan=read(checked(ROOT,plan_pin));verify_contract(ROOT,plan['core'])
    for p in plan['code']:checked(ROOT,p)
    if (BASE/'results.json').exists():raise ValueError('Original run already complete')
    preserved={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
        for p in BASE.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    RECOVERY.mkdir()
    write(RECOVERY/'preserved-before-resume.json',preserved)
    cases=read(checked(ROOT,plan['cases']));rows=[];spent=0;existing={}
    for folder in sorted((BASE/'runs').iterdir()):
        if not (folder/'scored.json').exists():continue
        request=read(folder/'request.json');score=read(folder/'scored.json')
        if len(score)!=4 or any(r['failed'] for r in score):raise ValueError('Invalid published partial result')
        existing[(request['case'],request['variant'],request['repeat'])]=score
        spent+=read(folder/'process.json')['walltime_seconds']
    contract={'scope':'watchdog file-count recovery; algorithms and train/validation parameters unchanged',
        'original_plan':plan_pin,'recovery_code':pin(ROOT,'scripts/resume_centerline_comparison_v2.py'),
        'completed_jobs_reused':len(existing),'planned_jobs':270,'missing_jobs':270-len(existing),
        'partial_files':pin(ROOT,(RECOVERY/'preserved-before-resume.json').relative_to(ROOT).as_posix()),
        'cause':'FileNotFoundError on atomic temporary .pyc replacement during stat',
        'env_change':'PYTHONDONTWRITEBYTECODE=1 for remaining children only; no algorithm/runtime versions changed',
        'CV_parameter_changes':False,'data_changes':False}
    write(RECOVERY/'contract.json',contract);new_jobs=0
    for split in ('train','validation'):
        for p in plan['code']:checked(ROOT,p)
        for case in [c for c in cases if c['split']==split]:
            for variant in plan['variants']:
                for repeat in range(2):
                    key=(case['id'],variant,repeat)
                    if key in existing:
                        rows.extend(existing[key]);continue
                    if spent>=1800:raise RuntimeError('Total budget exhausted')
                    folder=RECOVERY/'runs'/f'{case["id"]}-{variant}-r{repeat}';folder.mkdir(parents=True)
                    request={'case':case['id'],'variant':variant,'repeat':repeat,'views':[
                        {**v,'gray':pin(ROOT,'artifacts/cv-comparison-inputs-v2/'+v['gray']),
                         'mask':pin(ROOT,'artifacts/cv-comparison-inputs-v2/'+v['common_text_mask'])} for v in case['variants']]}
                    write(folder/'request.json',request)
                    env={**os.environ,'PYTHONIOENCODING':'utf-8','PYTHONDONTWRITEBYTECODE':'1',
                        'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
                        'XDG_CACHE_HOME':str(ROOT/'artifacts/.cache')}
                    with (folder/'child.log').open('xb') as log:
                        child=subprocess.Popen([sys.executable,str(ROOT/'scripts/run_centerline_comparison_v1.py'),
                            'child','--request',str(folder/'request.json')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0,start_new_session=os.name!='nt')
                        budget=AttemptBudget(child,folder,StableOutputCounter(),SimpleNamespace(timeout_seconds=min(60,1800-spent),
                            max_output_bytes=64*1024**2,max_output_files=200),memory_MiB=1024)
                        try:
                            while child.poll() is None:budget.check();time.sleep(.05)
                        finally:budget.stop()
                    runtime=budget.measurements();spent+=runtime['walltime_seconds'];write(folder/'process.json',runtime)
                    if runtime['budget_error'] or child.returncode!=0:raise RuntimeError('Recovery child failed')
                    path=folder/'output/result.json';result=read(path)
                    for raw in result['rows']:
                        rows.append({'case':case['id'],'scope':case['scope'],'variant':variant,'repeat':repeat,'split':split,
                            'group_id':case['group_id'],'mode':raw['mode'],'text_mode':raw['text_mode'],'failed':False,
                            'extraction_seconds':raw['extraction_seconds'],'total_seconds':raw['total_seconds'],
                            'input_pixel_sha256':raw['input_pixel_sha256'],'foreground_sha256':raw['foreground_sha256'],
                            'input_topology':raw['input_topology'],'skeleton_topology':raw['skeleton_topology'],
                            'widths':raw['widths'],'scores':line_scores(case,raw,raw['input_to_native']),
                            'runtime':runtime,'artifact':path.relative_to(ROOT).as_posix()})
                    write(folder/'scored.json',rows[-4:]);new_jobs+=1
                    print(json.dumps({'recovered_job':folder.name,'new_jobs':new_jobs,'child_wall_seconds':spent}),flush=True)
    repeats=[];previous={};cache={}
    for row in rows:
        name=row['artifact']
        if name not in cache:cache[name]=read(ROOT/name)['rows']
        raw=next(r for r in cache[name] if (r['mode'],r['text_mode'])==(row['mode'],row['text_mode']))
        key=(row['case'],row['variant'],row['mode'],row['text_mode'])
        if row['repeat']==0:previous[key]=canonical(raw)
        else:repeats.append({'case':row['case'],'variant':row['variant'],'mode':row['mode'],
            'text_mode':row['text_mode'],'identical':canonical(raw)==previous[key]})
    for path,sha in preserved.items():checked(ROOT,{'path':path,'sha256':sha})
    write(BASE/'results.json',rows);write(BASE/'repeat-checks.json',repeats)
    gates=[read(BASE/(split+'-parameter-gate.json')) for split in ('train','validation')]
    write(BASE/'execution.json',{'rows':len(rows),'failed_children':0,'child_wall_seconds':spent,
        'repeat_checks':len(repeats),'all_repeat_identical':all(r['identical'] for r in repeats),
        'parameter_gates':gates,'closed_test_used':False,'production_changed':False,'new_OCR_ML_calls':0,
        'interrupted_parent_attempts':1,'reused_completed_jobs':len(existing),'recovery_jobs':new_jobs,
        'recovery_contract':pin(ROOT,(RECOVERY/'contract.json').relative_to(ROOT).as_posix()),
        'unpublished_interrupted_child':'validation-new-2-2-parallel_reference-r1; original payload retained, not scored'})
    print(json.dumps({'completed_rows':len(rows),'repeat_identical':all(r['identical'] for r in repeats)}))
