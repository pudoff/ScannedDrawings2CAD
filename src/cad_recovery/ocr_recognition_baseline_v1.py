"""Fixed validation-only native/ONNX/pipeline baseline and export qualification."""
from copy import deepcopy
from collections import Counter
from datetime import datetime,timezone
from html import escape
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from .dataset_layout_v1 import checked,encoded,inside,pin,read
from .family_catalog_v2 import require
from .ocr_validation_expansion_v1 import verify_release
from .ocr_training_runtime_v1 import validate_runtime_config,tree_files
from .ocr_baseline_v1 import _make_engine,_effective_engine_settings
from .ocr_baseline_v2 import _environment,_check_engine
from .ocr_baseline_v3 import _one,_memory
from .ocr_release_v3 import _package
from .ocr_recognition_core_v1 import PREPROCESS,TRAINING,decode_ctc,EXPORT_BUDGET,compare_export
from .ocr_recognition_metrics_v1 import score_recognition,assembly_diagnostics

PARENT={'path':'artifacts/ocr-validation-expansion-final-v1.json','sha256':'b5126bbe7f463c68bc4e3db5e2401e7b16910d3e35c827b6f9477a5448af26f8'}
RELEASE={'path':'artifacts/ocr-current-release-v2/manifest.json','sha256':'6f04f9239843c89f35d0b5b5a176a8a137a0f4e0baa91fafbfad0639b8d7cea7'}
RUNTIME={'path':'configs/reviews/ocr-training-runtime-v1.json','sha256':'e64749be194240f407485f431ca3e006dc0aa2272875e60ed6554680ec7ac219'}
EXPORT_ENV='artifacts/ocr-recognition-setup-v1/venv'
CODE=['src/cad_recovery/'+n+'.py' for n in ('ocr_recognition_core_v1','ocr_recognition_metrics_v1','ocr_recognition_baseline_v1')]+[
 'scripts/evaluate_native_recognition_v1.py','scripts/export_native_recognition_v1.py','scripts/evaluate_ocr_recognition_v1.py']
PROTOCOL={'task':'P16.2a.4','split':'validation','count':64,'kinds':{'legacy_identifier':16,'name_literal':23,'dimension_literal':25},
 'order':'sample ID ascending; one sample per call; native export then native recognition then ONNX then full pipeline',
 'arms':['native_rec','exported_rec','legacy_rec','legacy_pipeline'],'preprocess':PREPROCESS,
 'export':{'opset':17,'enable_optimize':False,'polygraphy_auto_install':False,'budget':EXPORT_BUDGET,
           'acceptance':'all validation tensors meet explicit budget and every timestep class and greedy text identical; updated synthetic checkpoint retains strict rtol=1e-4/atol=1e-5 gate',
           'superseded_configs':['configs/reviews/ocr-recognition-baseline-v1.json','configs/reviews/ocr-recognition-baseline-v2.json','configs/reviews/ocr-recognition-baseline-v3.json','configs/reviews/ocr-recognition-baseline-v4.json']},
 'test_used':False,'training_started':False,'normalization':'unchanged COMPARISON_POLICY_V1','training_candidate':TRAINING}


def _rows(rows,base):
    return [{**r,'text_kind':r.get('text_kind','legacy_identifier'),
              'image_pin':pin(base,r['image'])} for r in sorted(rows,key=lambda r:r['id']) if r['split']=='validation']


def freeze(workspace,output):
    root=Path(workspace).resolve();checked(root,PARENT)
    _,records,base=verify_release(root,RELEASE);rows=_rows(records,base)
    require(len(rows)==64,'Wrong validation scope')
    for r in rows:r['image_pin']['path']=(base/r['image']).relative_to(root).as_posix()
    runtime=validate_runtime_config(root,RUNTIME);old=read(root/'configs/reviews/ocr-baseline-v4.json')
    require(_environment()==old['expected_environment'],'Inference environment changed')
    names=set(CODE+[PARENT['path'],RUNTIME['path'],'configs/reviews/ocr-baseline-v4.json',
       'configs/ocr-recognition-export-additions-v1.lock','configs/ocr-recognition-export-paddle-v1.lock'])
    names.update(p['path'] for p in runtime['inputs']);names.update(p['path'] for p in old['pins'])
    names.update(tree_files(root,EXPORT_ENV))
    names.update(p.relative_to(root).as_posix() for p in (root/'artifacts/ocr-recognition-setup-v1').glob('install-report*.json'))
    names.update(r['image_pin']['path'] for r in rows)
    export_cases={'pretrained':pin(root,runtime['checkpoint']),
                  'updated_smoke':pin(root,'artifacts/ocr-training-native-smoke-v1/synthetic-step.pdparams')}
    names.update(p['path'] for p in export_cases.values())
    config={'schema_version':'ocr-recognition-baseline-config/1.0','created_before_inference_utc':datetime.now(timezone.utc).isoformat(),
      'parent':PARENT,'release':RELEASE,'runtime_config':RUNTIME,'protocol':PROTOCOL,'rows':rows,
      'samples':[{'id':r['id'],'image':r['image_pin']} for r in rows], 'export_cases':export_cases,
      'pins':[pin(root,n) for n in sorted(names)],'expected_environment':_environment(),
      'native_python':runtime['python'],'export_python':EXPORT_ENV+'/Scripts/python.exe','export_tree':EXPORT_ENV}
    target=inside(root,output);require(not target.exists(),'Choose a new baseline config');target.write_bytes(encoded(config));return pin(root,output)


def authenticate(root,config_pin,*,full=False):
    c=read(checked(root,config_pin))
    require(c['schema_version']=='ocr-recognition-baseline-config/1.0' and c['protocol']==PROTOCOL
        and c['parent']==PARENT and c['release']==RELEASE and c['runtime_config']==RUNTIME,'Changed evaluation protocol')
    require(_environment()==c['expected_environment'],'Inference environment changed')
    require(set(CODE).issubset({p['path'] for p in c['pins']})
            and len(c['pins'])==len({p['path'] for p in c['pins']}),'Incomplete or duplicated evaluation code pins')
    require(len(c['rows'])==64 and all(r['split']=='validation' for r in c['rows'])
            and Counter(r['text_kind'] for r in c['rows'])==PROTOCOL['kinds']
            and c['samples']==[{'id':r['id'],'image':r['image_pin']} for r in c['rows']], 'Changed validation inventory')
    for p in c['pins']:checked(root,p)
    require(set(tree_files(root,EXPORT_ENV))=={p['path'] for p in c['pins'] if p['path'].startswith(EXPORT_ENV+'/')},'Export environment inventory changed')
    validate_runtime_config(root,RUNTIME)
    if full:
        _,rows,base=verify_release(root,RELEASE);expected=_rows(rows,base)
        for r in expected:r['image_pin']['path']=(base/r['image']).relative_to(root).as_posix()
        require(c['rows']==expected and c['samples']==[{'id':r['id'],'image':r['image_pin']} for r in expected],'Validation binding changed')
    return c


def _worker(root,config_pin,target,python,script):
    result=subprocess.run([str(inside(root,python)),str(inside(root,script)),'--config',str(checked(root,config_pin)),
        '--config-sha256',config_pin['sha256'],'--output',str(target)],cwd=root,capture_output=True,check=False,timeout=1200)
    # Native Windows libraries may interleave UTF-16 diagnostics; preserve raw bytes.
    log=target.parent/(target.name+'-process.json')
    log.write_bytes(encoded({'returncode':result.returncode,'stdout_utf8_lossy':result.stdout.decode('utf-8','replace'),
                             'stderr_utf8_lossy':result.stderr.decode('utf-8','replace')}))
    require(result.returncode==0,'Worker failed; inspect '+log.relative_to(root).as_posix())
    return read(target/'result.json')


def summarize(c,arms,equivalence,export_evidence):
    metrics={name:score_recognition(c['rows'],predictions) for name,predictions in arms.items()}
    return {'schema_version':'ocr-recognition-baseline-report/1.0','task':'P16.2a.4','split':'validation',
      'arms':metrics,'export_equivalence':equivalence,'export_evidence':export_evidence,
      'diagnostics':assembly_diagnostics(c['rows'],arms['legacy_pipeline'],arms['legacy_rec']),
      'coverage':{'lines':64,'unique_images':61,'sources':30,'fields':58,'new_unresolved_dimension_fields':1,
                  'old_test_unresolved_fields_preserved':2,'independent_evaluation':False},
      'training_started':False,'test_used':False,'parameter_search':False,'training_candidate':TRAINING,
      'limitations':['Native/ONNX share exact saved tensors; full pipeline uses its own frozen preprocessing.',
        'Paired successes locate a pipeline difference but do not separately identify detection, orientation and preprocessing causes.',
        'Tiny title/dimension subset and conservative groups; no family independence or full-page/CAD accuracy claim.',
        'Deployment export is checked numerically, not inferred from matching text or class count.']}


def run(workspace,config_pin,output):
    root=Path(workspace).resolve();c=authenticate(root,config_pin,full=True)
    target=inside(root,output);require(target.parent==root/'artifacts' and not target.exists(),'Choose a new artifacts output');target.mkdir()
    memory=[{'checkpoint':'before_models',**_memory()}]
    exported=_worker(root,config_pin,target/'export',c['export_python'],'scripts/export_native_recognition_v1.py')
    native=_worker(root,config_pin,target/'native',c['native_python'],'scripts/evaluate_native_recognition_v1.py')
    authenticate(root,config_pin)
    import onnxruntime as ort
    options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1;options.log_severity_level=3
    paths={'exported_rec':target/'export/pretrained.onnx','legacy_rec':root/'artifacts/models/ppocr-v5/eslav_PP-OCRv5_rec_mobile.onnx'}
    sessions={k:ort.InferenceSession(str(p),sess_options=options,providers=['CPUExecutionProvider']) for k,p in paths.items()}
    chars=read(root/'artifacts/ocr-training-preparation-v1/inference-alphabet.json')['characters']
    arms={'native_rec':native['results'],'exported_rec':[],'legacy_rec':[]};equivalence=[];times=[]
    for row in c['rows']:
        identifier=row['id'];x=np.load(target/'native/tensors'/f'{identifier}.npy',allow_pickle=False)[None]
        reference=np.load(target/'native/probabilities'/f'{identifier}.npy',allow_pickle=False)
        for name,session in sessions.items():
            tick=time.perf_counter();prob=session.run(None,{session.get_inputs()[0].name:x})[0][0];elapsed=time.perf_counter()-tick
            text,confidence=decode_ctc(prob,chars);arms[name].append({'id':identifier,'prediction':text,'confidence':confidence,'status':'ok' if text else 'empty'})
            np.save(target/'native/probabilities'/f'{identifier}-{name}.npy',prob,allow_pickle=False)
            times.append({'id':identifier,'arm':name,'seconds':elapsed})
            if name=='exported_rec':equivalence.append({'id':identifier,**compare_export(reference,prob),
                'text_equal':text==next(p['prediction'] for p in native['results'] if p['id']==identifier)})
    old=read(root/'configs/reviews/ocr-baseline-v4.json');engine=_make_engine(root/'artifacts/models/ppocr-v5')
    _check_engine(root,old,engine.provenance,_effective_engine_settings(engine))
    pipeline=[];pipeline_times=[]
    for row in c['rows']:
        result,timing=_one(engine,{**row,'image_path':str(checked(root,row['image_pin']))});pipeline.append(result);pipeline_times.append(timing)
    _check_engine(root,old,engine.provenance,_effective_engine_settings(engine));arms['legacy_pipeline']=pipeline
    memory.append({'checkpoint':'after_models',**_memory()})
    require(all(r['allclose'] and r['text_equal'] for r in equivalence),'Native/export numerical equivalence failed')
    authenticate(root,config_pin,full=True)
    report=summarize(c,arms,equivalence,exported)
    updated_checks=[]
    for count in (1,2):
        native_updated=np.load(target/'native'/f'updated-batch-{count}.npy',allow_pickle=False)
        exported_updated=np.load(target/'export'/f'updated_smoke-batch-{count}.npy',allow_pickle=False)
        require(np.allclose(native_updated,exported_updated,rtol=0.0001,atol=0.00001),'Updated Paddle 3.2.2 state export differs')
        updated_checks.append({'batch':count,'allclose':True,'max_abs':float(np.abs(native_updated-exported_updated).max())})
    report['updated_checkpoint_cross_backend_export']=updated_checks
    (target/'predictions.json').write_bytes(encoded(arms));(target/'report.json').write_bytes(encoded(report))
    (target/'runtime.json').write_bytes(encoded({'onnx_calls':times,'pipeline':pipeline_times,'native':native['runtime'],'memory':memory,
        'scope':'process-lifetime peak includes input authentication; native/export timings separate processes; CPU only, not isolated model RAM'}))
    page=['<!doctype html><html lang="ru"><meta charset="utf-8"><title>OCR: recognizer baseline</title><h1>Validation: распознаватель и конвейер</h1>',
          '<p>64 строки, только validation. Метки не исправлялись по ответам. <a href="report.json">Метрики и ограничения</a></p>']
    for row in c['rows']:
        page.append('<section><h3>'+escape(row['text_kind']+' / '+row['id'])+'</h3><b>'+escape(row['text'])+'</b>')
        for name,values in arms.items():page.append('<p>'+escape(name+': '+next(p['prediction'] for p in values if p['id']==row['id']))+'</p>')
        page.append('</section>')
    (target/'index.html').write_text(''.join(page)+'</html>',encoding='utf-8')
    names=sorted(p.relative_to(target).as_posix() for p in target.rglob('*') if p.is_file())
    (target/'manifest.json').write_bytes(encoded({'schema_version':'ocr-recognition-baseline-manifest/1.0','config':config_pin,'files':[pin(target,n) for n in names]}))
    return report


def verify(workspace,manifest_pin):
    root=Path(workspace).resolve();path,m=_package(root,manifest_pin);c=authenticate(root,m['config'],full=True)
    require(m['schema_version']=='ocr-recognition-baseline-manifest/1.0','Wrong baseline manifest')
    report=read(path.parent/'report.json');arms=read(path.parent/'predictions.json')
    require(set(arms)==set(PROTOCOL['arms']),'Wrong model arms')
    expected=summarize(c,arms,report['export_equivalence'],read(path.parent/'export/result.json'))
    expected['updated_checkpoint_cross_backend_export']=report['updated_checkpoint_cross_backend_export']
    require(report==expected and all(e['allclose'] and e['text_equal'] for e in report['export_equivalence']),'Saved metrics/equivalence changed')
    chars=read(root/'artifacts/ocr-training-preparation-v1/inference-alphabet.json')['characters']
    for row in c['rows']:
        for arm,suffix in [('native_rec',''),('legacy_rec','-legacy_rec'),('exported_rec','-exported_rec')]:
            probability=np.load(path.parent/'native/probabilities'/(row['id']+suffix+'.npy'),allow_pickle=False)
            text,confidence=decode_ctc(probability,chars);p=next(p for p in arms[arm] if p['id']==row['id'])
            require(p['prediction']==text and p['confidence']==confidence,'Saved text differs from CTC probabilities')
        native=np.load(path.parent/'native/probabilities'/(row['id']+'.npy'),allow_pickle=False)
        exported=np.load(path.parent/'native/probabilities'/(row['id']+'-exported_rec.npy'),allow_pickle=False)
        eq=next(e for e in report['export_equivalence'] if e['id']==row['id'])
        recomputed=compare_export(native,exported)
        require(recomputed['allclose'] and all(eq[k]==v for k,v in recomputed.items()),'Saved probability equivalence changed')
    for case in report['updated_checkpoint_cross_backend_export']:
        n=case['batch'];a=np.load(path.parent/'native'/f'updated-batch-{n}.npy',allow_pickle=False)
        b=np.load(path.parent/'export'/f'updated_smoke-batch-{n}.npy',allow_pickle=False)
        require(np.allclose(a,b,rtol=0.0001,atol=0.00001) and case['max_abs']==float(np.abs(a-b).max()),'Updated export evidence changed')
    return report
