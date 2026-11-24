"""Preregister, run and summarize bounded P10.2c.3a CV children."""
import time
CPU_START=time.process_time()
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/centerline-comparison-v1'
CORE={'path':'artifacts/cv-comparison-protocol-v1/contract.json',
      'sha256':'815c8a5ee7f47714066206b2a7a55cbdd94328bb75fdc2ce5383b02ca300d901'}


def read(path):return json.loads(path.read_text('utf-8'))
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def freeze():
    from verify_cv_comparison_contract_v1 import verify_contract
    from cad_recovery.dataset_layout_v1 import checked,pin
    verify_contract(ROOT,CORE)
    if BASE.exists():raise FileExistsError(BASE)
    qualification=pin(ROOT,'artifacts/cv-skeleton-runtime-v1/qualification.json');q=read(checked(ROOT,qualification))
    if not q['method_admission']['zhang'] or q['method_admission']['medial']:
        raise ValueError('Unexpected qualification disposition')
    BASE.mkdir()
    names=['scripts/run_centerline_comparison_v1.py','src/cad_recovery/centerline_comparison_v1.py',
        'src/cad_recovery/local_stroke_evidence_v1.py','scripts/qualify_skeleton_runtime_v1.py',
        'src/cad_recovery/service_profiles_v2/budget.py','src/cad_recovery/service_v1/process.py',
        'src/cad_recovery/hard_raster_runs_v1.py']
    plan={'task':'P10.2c.3a','core':CORE,'qualification':qualification,'prepared_before_corpus_predictions':True,
        'code':[pin(ROOT,n) for n in names],'variants':['baseline','zhang','parallel_reference'],
        'excluded_variant':{'medial':'qualification cross creates 1 hole; excluded by preregistered topology gate'},
        'cases':pin(ROOT,'artifacts/cv-comparison-inputs-v2/effective-cases.json'),
        'repeats':2,'input_modes':['native_crop','half_resolution_crop'],
        'text_modes':['untouched','common_reviewed_text_mask'],
        'scope':'line extraction and observational width only; curves/roles/heads/DXF admission not evaluated',
        'full_page_plans':'deferred until a candidate passes crop gates; no full-page quality claim',
        'parameters':{'cv2_rng_seed':0,'medial_axis_rng':0,'CPU_threads':1,
            'width_sample_range':[.12,.88],'width_samples_max':128,'normal_sampling_step_px':.5,
            'normal_half_span_px':32,'line_matching_native_px':2,'line_matching_angle_deg':5,
            'minimum_line_coverage':.8,'real_guide_default_band_px':8},
        'budgets':{'case_seconds':60,'case_RSS_MiB':1024,'case_output_MiB':64,'max_output_files':200,
            'total_conservative_child_wall_seconds':1800},
        'validation_policy':'all candidates fixed now; train first, hash gate again before validation; no parameter search',
        'timing_policy':'shared variants run same detector; parallel reference has different stages/output and is reported separately',
        'new_data_training':False,'closed_test_used':False,'production_changed':False}
    write(BASE/'plan.json',plan);print(json.dumps(pin(ROOT,(BASE/'plan.json').relative_to(ROOT).as_posix())))


def child(request_path):
    sys.path.insert(0,str(ROOT/'artifacts/cv-skeleton-runtime-v1/deps'))
    from skimage.morphology import skeletonize,medial_axis
    import cv2
    import numpy as np
    from PIL import Image
    from cad_recovery.centerline_comparison_v1 import extract
    from cad_recovery.dataset_layout_v1 import checked
    request=read(request_path);folder=request_path.parent/'output';folder.mkdir()
    results=[]
    for v in request['views']:
        with Image.open(checked(ROOT,v['gray'])) as im:gray=np.asarray(im).copy()
        with Image.open(checked(ROOT,v['mask'])) as im:mask=np.asarray(im)>0
        for mode in ('untouched','common_reviewed_text_mask'):
            cv2.setNumThreads(1);cv2.setRNGSeed(0)
            pixels=gray.copy()
            if mode!='untouched':pixels[mask]=255
            if pixels.size>4194304:raise ValueError('Input pixel budget')
            result=extract(ROOT,folder/(v['mode']+'-'+mode),pixels,request['variant'])
            results.append({'mode':v['mode'],'text_mode':mode,'input_to_native':v['input_to_native'],**result})
    write(folder/'result.json',{'rows':results,'child_cpu_seconds':time.process_time()-CPU_START})


class OutputCounter:
    @staticmethod
    def output_size(folder):
        paths=[p for p in folder.rglob('*') if p.is_file()]
        return sum(p.stat().st_size for p in paths),len(paths)


def canonical(row):
    return {k:row[k] for k in ('candidates','widths','input_pixel_sha256','foreground_sha256',
        'skeleton_sha256','input_topology','skeleton_topology')}


def run(expected):
    from verify_cv_comparison_contract_v1 import verify_contract
    from cad_recovery.dataset_layout_v1 import checked,pin
    from cad_recovery.service_profiles_v2.budget import AttemptBudget
    from cad_recovery.centerline_comparison_v1 import line_scores
    checked(ROOT,{'path':(BASE/'plan.json').relative_to(ROOT).as_posix(),'sha256':expected})
    plan=read(BASE/'plan.json');verify_contract(ROOT,plan['core']);q=read(checked(ROOT,plan['qualification']))
    for record in [*plan['code'],*q['wheels']]:checked(ROOT,record)
    cases=read(checked(ROOT,plan['cases']));rows=[];spent=0;previous={};repeat_checks=[];gate=[]
    for split in ('train','validation'):
        for record in plan['code']:checked(ROOT,record)
        gate.append({'split':split,'plan_sha256':pin(ROOT,(BASE/'plan.json').relative_to(ROOT).as_posix())['sha256'],
                     'parameters_changed':False})
        write(BASE/(split+'-parameter-gate.json'),gate[-1])
        for case in [c for c in cases if c['split']==split]:
            for variant in plan['variants']:
                for repeat in range(2):
                    if spent>=1800:raise RuntimeError('Total experiment budget reached')
                    folder=BASE/'runs'/f'{case["id"]}-{variant}-r{repeat}'
                    if folder.exists():raise FileExistsError(folder)
                    folder.mkdir(parents=True)
                    request={'case':case['id'],'variant':variant,'repeat':repeat,'views':[
                        {**v,'gray':pin(ROOT,('artifacts/cv-comparison-inputs-v2/'+v['gray'])),
                            'mask':pin(ROOT,('artifacts/cv-comparison-inputs-v2/'+v['common_text_mask']))}
                        for v in case['variants']]}
                    write(folder/'request.json',request)
                    env={**os.environ,'PYTHONIOENCODING':'utf-8','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1',
                         'MKL_NUM_THREADS':'1','XDG_CACHE_HOME':str(ROOT/'artifacts/.cache')}
                    with (folder/'child.log').open('xb') as log:
                        process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'child','--request',str(folder/'request.json')],
                            cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0,start_new_session=os.name!='nt')
                        settings=SimpleNamespace(timeout_seconds=min(60,1800-spent),max_output_bytes=64*1024**2,max_output_files=200)
                        budget=AttemptBudget(process,folder,OutputCounter(),settings,memory_MiB=1024)
                        try:
                            while process.poll() is None:
                                budget.check();time.sleep(.05)
                        except RuntimeError:pass
                        finally:budget.stop()
                    runtime=budget.measurements();spent+=runtime['walltime_seconds'];write(folder/'process.json',runtime)
                    path=folder/'output/result.json'
                    if process.returncode!=0 or runtime['budget_error'] or not path.exists():
                        rows.append({'case':case['id'],'variant':variant,'repeat':repeat,'split':split,
                            'group_id':case['group_id'],'failed':True,'runtime':runtime})
                        write(BASE/'partial-results.json',rows) if not (BASE/'partial-results.json').exists() else None
                        continue
                    result=read(path)
                    for raw in result['rows']:
                        key=(case['id'],variant,raw['mode'],raw['text_mode'])
                        if repeat==0:previous[key]=canonical(raw)
                        else:repeat_checks.append({'case':case['id'],'variant':variant,'mode':raw['mode'],
                            'text_mode':raw['text_mode'],'identical':canonical(raw)==previous.get(key)})
                        rows.append({'case':case['id'],'scope':case['scope'],'variant':variant,'repeat':repeat,'split':split,
                            'group_id':case['group_id'],'mode':raw['mode'],'text_mode':raw['text_mode'],'failed':False,
                            'extraction_seconds':raw['extraction_seconds'],'total_seconds':raw['total_seconds'],
                            'input_pixel_sha256':raw['input_pixel_sha256'],'foreground_sha256':raw['foreground_sha256'],
                            'input_topology':raw['input_topology'],'skeleton_topology':raw['skeleton_topology'],
                            'widths':raw['widths'],'scores':line_scores(case,raw,raw['input_to_native']),
                            'runtime':runtime,'artifact':(path).relative_to(ROOT).as_posix()})
                    write(folder/'scored.json',rows[-len(result['rows']):])
            print(json.dumps({'completed_case':case['id'],'split':split,'child_wall_seconds':round(spent,2)}),flush=True)
    write(BASE/'results.json',rows);write(BASE/'repeat-checks.json',repeat_checks)
    write(BASE/'execution.json',{'child_wall_seconds':spent,'rows':len(rows),'failed_children':sum(r['failed'] for r in rows),
        'repeat_checks':len(repeat_checks),'all_repeat_identical':all(r['identical'] for r in repeat_checks),
        'parameter_gates':gate,'closed_test_used':False,'production_changed':False,'new_OCR_ML_calls':0})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['freeze','run','child']);p.add_argument('--sha256');p.add_argument('--request',type=Path)
    a=p.parse_args()
    if a.mode=='freeze':freeze()
    elif a.mode=='child':child(a.request)
    else:
        if not a.sha256:p.error('--sha256 required')
        run(a.sha256)
