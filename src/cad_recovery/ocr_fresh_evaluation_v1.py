"""One fixed OCR candidate on the fresh holdout; offline replay never runs a model."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from html import escape
import hashlib
import json
import math
import os
from pathlib import Path
import time
import numpy as np
from .dataset_layout_v1 import checked, encoded, inside, pin, read
from .family_catalog_v2 import require
from .ocr_fresh_holdout_v1 import (SELECTION, NEW_MARKER, check_admission,
    validate_admission, verify_release)
from .ocr_release_v3 import _package
from .ocr_first_training_v1 import verify_tensors, ort_session, journal_append
from .ocr_recognition_baseline_v1 import _worker
from .ocr_recognition_core_v1 import PREPROCESS, decode_ctc
from .ocr_baseline_metrics_v1 import aggregate_scores_v1, normalize_text_v1, COMPARISON_POLICY_V1
from .ocr_baseline_v3 import _memory

GATE={'path':'configs/reviews/ocr-fresh-holdout-evaluation-v1.json',
      'sha256':'444806356287b789b51b2b48fb2c1e54c6733e211db6c58187cc61d4458508bb'}
PARENT={'path':'artifacts/ocr-fresh-holdout-final-v1.json',
        'sha256':'e0504b819c11cf6fb2822145ea0b65fbd6db5bb4085985197fc4c06f9c7fa4ae'}
EXPERIMENT={'path':'configs/reviews/ocr-frozen-bn-experiment-v1.json',
    'sha256':'32b1c072c9d81f4ccd965d305c2f3fb10d12046858df7e2474f7dbbc3fbc2c44'}
ALPHABET='artifacts/ocr-training-preparation-v1/inference-alphabet.json'
KINDS=('identifier_literal','name_literal','dimension_literal')
CODE=['src/cad_recovery/ocr_fresh_evaluation_v1.py','scripts/evaluate_ocr_fresh_holdout_v1.py',
    'tests/test_ocr_fresh_evaluation_v1.py','src/cad_recovery/ocr_first_training_v1.py',
    'src/cad_recovery/ocr_recognition_baseline_v1.py','src/cad_recovery/ocr_recognition_core_v1.py',
    'src/cad_recovery/ocr_baseline_metrics_v1.py','src/cad_recovery/ocr_baseline_v3.py',
    'scripts/infer_selected_ocr_v1.py','scripts/smoke_ocr_training_native_v1.py',
    'scripts/train_native_recognition_v1.py']
POLICY={'device':'CPUExecutionProvider','intra_op_threads':4,'inter_op_threads':1,'batch_size':1,
    'warmup_model_calls':0,'one_model_call_per_accepted_row':True,'retry_after_error':False,
    'model_selection_allowed':False,'parameter_search':False,'training_allowed':False,
    'old_test_inference_allowed':False,'human_verified':False,'independent_evaluation':False,
    'production_model_updated':False,'coverage_denominator':48,
    'probabilities':'save original softmax output for offline decoding; never rerun test to verify',
    'failure':'atomically claim global marker before first inference; retain marker on any failure; empty/error rows stay in denominator',
    'timing':'per-call session.run time, including failures; wall time additionally includes decode and probability write',
    'memory':'CPU process-lifetime peak including authentication/preparation; not isolated model RAM; no GPU used'}


def score_holdout(rows, predictions, slots):
    require(bool(rows) and bool(slots),'Empty final evaluation scope')
    require(len({r['id'] for r in rows})==len(rows),'Duplicate reference row')
    by={p['id']:p for p in predictions}
    require(len(by)==len(predictions)==len(rows) and set(by)=={r['id'] for r in rows},'Incomplete or duplicated predictions')
    for p in predictions:
        require(isinstance(p['prediction'],str) and p['status'] in ('ok','empty','error')
            and type(p['confidence']) in (int,float) and math.isfinite(p['confidence'])
            and 0<=p['confidence']<=1,'Invalid prediction status/confidence')
        require((p['status']=='ok')==bool(p['prediction']),'Empty/error predictions must be empty strings')
    require(len({s['slot_id'] for s in slots})==len(slots),'Duplicate field slot')
    by_slot=defaultdict(list)
    for row in rows:
        require(row['text_kind'] in KINDS,'Unknown reporting stratum')
        by_slot[row['slot_id']].append(row)
    accepted={s['slot_id'] for s in slots if s['status']=='accepted'}
    require(set(by_slot)==accepted,'Reference rows do not cover accepted slots')
    for s in slots:
        require(s['status'] in ('accepted','unresolved','absent','quarantine'),'Invalid field status')
        if s['status']=='accepted':
            part=sorted(by_slot[s['slot_id']],key=lambda r:r['line_index'])
            require([r['line_index'] for r in part]==list(range(len(s['lines'])))
                and [r['text'] for r in part]==[l['text'] for l in s['lines']],'Physical field lines changed')
        else:require(s['text'] is None and not s['lines'],'Unknown field has an invented target')
    def score(part):
        return aggregate_scores_v1([{'id':r['id'],'source_id':r['source_id'],'reference':r['text'],
            'prediction':by[r['id']]['prediction'],'status':by[r['id']]['status']} for r in part])
    def compact(part):
        value=score(part);return {k:value[k] for k in ('sample_count','status_counts','raw','normalized')}
    def strata(key):return {v:compact([r for r in rows if r[key]==v]) for v in sorted({r[key] for r in rows})}
    kinds=strata('text_kind');sources=strata('source_id');field_results=[]
    for slot in slots:
        part=by_slot.get(slot['slot_id'],[])
        item={'slot_id':slot['slot_id'],'status':slot['status'],'line_count':len(part)}
        for mode in ('raw','normalized'):
            norm=(lambda v:v) if mode=='raw' else normalize_text_v1
            item[mode+'_exact']=all(by[r['id']]['status']=='ok' and norm(by[r['id']]['prediction'])==norm(r['text']) for r in part) if part else None
        field_results.append(item)
    counts={mode:sum(s[mode+'_exact'] is True for s in field_results) for mode in ('raw','normalized')}
    reps={}
    for r in rows:reps.setdefault(r['image_sha256'],r)
    name_roles={}
    for role,placeholder in [('identifier_placeholder',True),('actual_name',False)]:
        part=[r for r in rows if r['field']=='part_name' and r['name_is_identifier_placeholder'] is placeholder]
        if part:name_roles[role]=compact(part)
    return {'overall':score(rows),'by_kind':kinds,'by_source':sources,'by_group':strata('group_id'),
        'by_field':strata('field'),'unique_images':compact(list(reps.values())),
        'name_cell_roles':name_roles,
        'macro_raw_CER_by_kind':sum(v['raw']['cer'] for v in kinds.values())/len(kinds),
        'macro_raw_CER_by_source':sum(v['raw']['cer'] for v in sources.values())/len(sources),
        'field_results':field_results,'coverage':{'scheduled_fields':len(slots),'accepted_fields':len(accepted),
            'unknown_fields':len(slots)-len(accepted),'accepted_lines':len(rows),
            'field_statuses':dict(Counter(s['status'] for s in slots)),'exact_fields':counts,
            'raw_exact_fraction_of_accepted':counts['raw']/len(accepted),
            'raw_verified_fraction_of_scheduled':counts['raw']/len(slots),
            'unknowns':'no invented CER targets; included in scheduled-field denominator'}}


def package(root,target,schema,**extra):
    # A nested preparation manifest is itself evidence in the outer package.
    files=[pin(target,p.relative_to(target).as_posix()) for p in sorted(target.rglob('*'))
        if p.is_file() and p!=target/'manifest.json']
    (target/'manifest.json').write_bytes(encoded({'schema_version':schema,'files':files,**extra}))
    return pin(root,(target/'manifest.json').relative_to(root).as_posix())


def _context(root):
    gate=read(checked(root,GATE));parent=read(checked(root,PARENT))
    require(parent['release']==gate['release'] and parent['gate']==GATE,'Parent proof binds another holdout')
    report,records=verify_release(root,gate['release']);selection=read(checked(root,SELECTION))
    validate_admission(gate,selection,gate['release'],report)
    for p in gate['code']:checked(root,p)
    for k in ('candidate_checkpoint','candidate_onnx','old_test_marker'):checked(root,gate[k])
    qpath,_=_package(root,gate['candidate_qualification']);q=read(qpath.parent/'config.json')
    require(q['checkpoint']==gate['candidate_checkpoint'] and q['mode']=='validation'
        and pin(root,(qpath.parent/'export/selected.onnx').relative_to(root).as_posix())==gate['candidate_onnx'],'Candidate qualification changed')
    base=checked(root,gate['release']).parent;rows=[]
    for record in records:
        image=pin(root,(base/record['image']).relative_to(root).as_posix())
        require(image['sha256']==record['image_sha256'],'Changed fresh image')
        rows.append({**record,'image_pin':image})
    labels=read(base/'labels.json')['slots']
    require(len(rows)==51 and len(labels)==48 and len({r['image_sha256'] for r in rows})==51
        and Counter(r['text_kind'] for r in rows)=={'identifier_literal':14,'name_literal':13,'dimension_literal':24}
        and all(r['split']=='test' for r in rows),'Changed fixed final scope')
    return gate,rows,labels


def prepare_run(workspace, config_path, output):
    root=Path(workspace).resolve();check_admission(root,GATE);gate,rows,_=_context(root)
    experiment=read(checked(root,EXPERIMENT));runtime=read(checked(root,experiment['runtime_config']))
    target=inside(root,config_path);out=inside(root,output)
    require(target.parent==root/'configs/reviews' and not target.exists()
        and out.parent==root/'artifacts' and not out.exists(),'Use new review config and artifacts output')
    pins={p['path']:p for p in experiment['inputs']}
    extras=[GATE,PARENT,EXPERIMENT,SELECTION,experiment['runtime_config'],pin(root,ALPHABET)]
    extras += [pin(root,n) for n in CODE]+gate['code']
    extras += [gate[k] for k in ('candidate_checkpoint','candidate_onnx','candidate_qualification','old_test_marker','release')]
    for m in ('release','candidate_qualification'):
        path,manifest=_package(root,gate[m])
        extras += [pin(root,(path.parent/p['path']).relative_to(root).as_posix()) for p in manifest['files']]
    for p in extras:
        require(p['path'] not in pins or pins[p['path']]==p,'Conflicting frozen input pins')
        pins[p['path']]=p
    for p in pins.values():checked(root,p)
    require(inside(root,runtime['python']).is_file(),'Native preprocessing interpreter unavailable')
    config={'schema_version':'ocr-fresh-evaluation-run/1.0','task':'P16.2f',
        'sealed_before_test_utc':datetime.now(timezone.utc).isoformat(),'gate':GATE,'parent':PARENT,
        'policy':POLICY,'preprocessing':PREPROCESS,'comparison':COMPARISON_POLICY_V1,
        'runtime_config':experiment['runtime_config'],'mode':'prepare_test','output':output,
        'candidate_checkpoint':gate['candidate_checkpoint'],'candidate_onnx':gate['candidate_onnx'],
        'one_shot_marker':NEW_MARKER,'alphabet':pin(root,ALPHABET),'rows':rows,
        'samples':[{'id':r['id'],'image':r['image_pin']} for r in rows],
        'code':[pin(root,n) for n in CODE],'pins':[pins[k] for k in sorted(pins)]}
    with target.open('xb') as stream:stream.write(encoded(config))
    return pin(root,config_path)


def verify_run(root,config_pin):
    c=read(checked(root,config_pin));require(c['schema_version']=='ocr-fresh-evaluation-run/1.0'
        and c['gate']==GATE and c['parent']==PARENT and c['policy']==POLICY
        and c['preprocessing']==PREPROCESS and c['comparison']==COMPARISON_POLICY_V1
        and c['mode']=='prepare_test' and c['one_shot_marker']==NEW_MARKER,'Changed final evaluation policy')
    gate,rows,slots=_context(root)
    require(c['rows']==rows and c['samples']==[{'id':r['id'],'image':r['image_pin']} for r in rows]
        and c['candidate_checkpoint']==gate['candidate_checkpoint'] and c['candidate_onnx']==gate['candidate_onnx'],'Changed candidate or final rows')
    experiment=read(checked(root,EXPERIMENT))
    require(c['runtime_config']==experiment['runtime_config'] and c['alphabet']==pin(root,ALPHABET),'Changed preprocessing runtime or alphabet')
    require([p['path'] for p in c['code']]==CODE,'Missing evaluation code')
    for p in c['code']:checked(root,p)
    require(len({p['path'] for p in c['pins']})==len(c['pins']),'Duplicate final input pins')
    inputs={p['path']:p for p in c['pins']}
    for p in experiment['inputs']+c['code']+[GATE,PARENT,EXPERIMENT,SELECTION,c['runtime_config'],c['alphabet']]:
        require(inputs.get(p['path'])==p,'Incomplete frozen input inventory')
    for p in c['pins']:checked(root,p)
    return c,gate,slots


def reserve_once(root,config_pin,preparation_pin,c):
    marker=inside(root,NEW_MARKER)
    value={'schema_version':'ocr-fresh-test-once/1.0','task':'P16.2f','config':config_pin,
        'gate':GATE,'preparation':preparation_pin,'candidate_onnx':c['candidate_onnx'],
        'expected_rows':51,'policy':POLICY,'started_utc':datetime.now(timezone.utc).isoformat()}
    with marker.open('xb') as stream:
        stream.write(encoded(value));stream.flush();os.fsync(stream.fileno())
    return marker


def call_once(session,input_name,x,identifier,characters,probability_path,journal):
    journal_append(journal,{'stage':'started','id':identifier,'arm':'selected_rec'})
    wall=time.perf_counter();before=time.perf_counter();inference=None;phase='inference'
    try:
        try:outputs=session.run(None,{input_name:x})
        finally:inference=time.perf_counter()-before
        phase='decode';prob=outputs[0][0];text,confidence=decode_ctc(prob,characters)
        result={'id':identifier,'prediction':text,'confidence':confidence,'status':'ok' if text else 'empty'}
        phase='save_probabilities';np.save(probability_path,prob,allow_pickle=False)
    except Exception as error:
        result={'id':identifier,'prediction':'','confidence':0.0,'status':'error',
            'error':{'stage':phase,'type':type(error).__name__,'message':str(error)}}
    journal_append(journal,{'stage':'completed','id':identifier,'arm':'selected_rec','result':result,
        'inference_seconds':inference,'wall_seconds':time.perf_counter()-wall})
    return result


def validate_journal(events,rows,predictions):
    expected=[(stage,r['id'],'selected_rec') for r in rows for stage in ('started','completed')]
    require([(e['stage'],e['id'],e['arm']) for e in events]==expected,'Missing, repeated or reordered model calls')
    require([e['result'] for e in events if e['stage']=='completed']==predictions,'Journal/result disagreement')
    for e in events:
        if e['stage']=='completed':
            require(all(isinstance(e[k],(int,float)) and math.isfinite(e[k]) and e[k]>=0
                for k in ('inference_seconds','wall_seconds')) and e['wall_seconds']>=e['inference_seconds'],'Invalid call timing')


def report_result(rows,predictions,slots,events,runtime):
    validate_journal(events,rows,predictions)
    completed=[e for e in events if e['stage']=='completed'];seconds=[e['inference_seconds'] for e in completed]
    return {'schema_version':'ocr-fresh-evaluation-report/1.0','task':'P16.2f','selected_epoch':0,
        'policy':POLICY,'metrics':score_holdout(rows,predictions,slots),'actual_model_calls':len(completed),
        'successful_output_calls':sum(p['status']!='error' for p in predictions),
        'candidate_selected_before_cohort':True,'checkpoint_selection_changed_after_test':False,
        'fine_tuning_improved_validation':False,'training_steps':0,'old_test_model_calls':0,
        'timing':{'model_call_seconds':sum(seconds),'per_call_mean_seconds':sum(seconds)/len(seconds),
            'per_call_min_seconds':min(seconds),'per_call_max_seconds':max(seconds),
            'call_wall_seconds':sum(e['wall_seconds'] for e in completed)},'runtime':runtime,
        'production_model_updated':False,'human_verified':False,'independent_evaluation':False,
        'limitations':['Recognition of fixed original-pixel ROI lines, not detector, full-sheet OCR or CAD.',
            'Assistant labels, two unknown fields; families/styles may overlap other splits.',
            'Epoch zero was selected before this cohort; this is not a fine-tuning gain.',
            'This holdout is now consumed; no rerun or model/label tuning against its results.']}


def _html(rows,predictions,slots):
    by={p['id']:p for p in predictions};parts=['<!doctype html><html lang="ru"><meta charset="utf-8">',
        '<title>Fresh OCR: one-shot evaluation</title><h1>51 строки / 48 позиций</h1>',
        '<p>Одна модель эпохи 0. Неизвестные цели сохранены. Повторный inference закрыт.</p>']
    for slot in slots:
        parts.append('<hr><h2>'+escape(slot['slot_id'])+'</h2><p>'+escape(slot['status'])+'</p>')
        if slot['status']!='accepted':parts.append('<p>Цель неизвестна; не восстанавливается.</p>')
        for r in [r for r in rows if r['slot_id']==slot['slot_id']]:
            p=by[r['id']];parts.append('<p>Эталон: '+escape(r['text'])+'<br>Ответ: '+escape(p['prediction'])+' / '+escape(p['status'])+'</p>')
            parts.append('<img style="max-width:100%" src="../ocr-fresh-holdout-v1/'+r['image']+'">')
    return ('\n'.join(parts)+'</html>').encode('utf-8')


def evaluate(workspace,config_pin):
    root=Path(workspace).resolve()
    require(not inside(root,NEW_MARKER).exists(),'Fresh final test already claimed; only offline verification is allowed')
    c,gate,slots=verify_run(root,config_pin)
    target=inside(root,c['output']);require(target.parent==root/'artifacts' and not target.exists(),'Choose a new final evaluation output')
    target.mkdir();start=time.perf_counter();runtime=read(checked(root,c['runtime_config']))
    prepared=_worker(root,config_pin,target/'prepared',runtime['python'],'scripts/infer_selected_ocr_v1.py')
    require(prepared['mode']=='prepare_test' and prepared['test_model_calls']==prepared['gradient_steps']==0
        and prepared['state_sha256'] is None and prepared['config_sha256']==config_pin['sha256'],'Preparation ran a model or changed scope')
    verify_tensors(target/'prepared',prepared['results'],c['samples'])
    preparation_pin=package(root,target/'prepared','ocr-fresh-prepared-tensors/1.0',config=config_pin)
    characters=read(checked(root,c['alphabet']))['characters'];memory=[{'stage':'before_session',**_memory()}]
    session_start=time.perf_counter();session=ort_session(checked(root,c['candidate_onnx']));session_seconds=time.perf_counter()-session_start
    require(session.get_providers()==['CPUExecutionProvider'],'Unexpected final inference provider')
    inputs=session.get_inputs();require(len(inputs)==1 and inputs[0].type=='tensor(float)'
        and inputs[0].shape[-3:]==[3,48,320],'Qualified model input contract changed')
    memory.append({'stage':'after_session',**_memory()});(target/'probabilities').mkdir()
    # This is the last admission check. The atomic create below is the global lock.
    require(not inside(root,NEW_MARKER).exists(),'Fresh evaluation already claimed')
    checked(root,gate['old_test_marker']);checked(root,c['candidate_onnx'])
    marker=reserve_once(root,config_pin,preparation_pin,c)
    predictions=[];journal=target/'calls.jsonl'
    for row in c['rows']:
        x=np.load(target/'prepared/tensors'/f"{row['id']}.npy",allow_pickle=False)[None]
        predictions.append(call_once(session,inputs[0].name,x,row['id'],characters,
            target/'probabilities'/f"{row['id']}.npy",journal))
    memory.append({'stage':'after_calls',**_memory()})
    runtime_result={'preparation':{'mode':prepared['mode'],'test_model_calls':0,'memory':prepared['memory']},
        'memory':memory,'session_initialization_seconds':session_seconds,
        'preparation_through_calls_seconds':time.perf_counter()-start,'providers':session.get_providers(),
        'scope':POLICY['memory'],'gpu_used':False}
    events=[json.loads(line) for line in journal.read_text('utf-8').splitlines()]
    report=report_result(c['rows'],predictions,slots,events,runtime_result)
    (target/'predictions.json').write_bytes(encoded(predictions));(target/'report.json').write_bytes(encoded(report))
    (target/'runtime.json').write_bytes(encoded(runtime_result));(target/'one-shot-marker.json').write_bytes(marker.read_bytes())
    (target/'index.html').write_bytes(_html(c['rows'],predictions,slots))
    for p in c['pins']:checked(root,p)
    return package(root,target,'ocr-fresh-evaluation-manifest/1.0',config=config_pin,
        marker=pin(root,NEW_MARKER),preparation=preparation_pin)


def verify_evaluation(workspace,manifest_pin):
    """Authenticate and rescore saved evidence only. No session or worker is constructed."""
    root=Path(workspace).resolve();path,m=_package(root,manifest_pin);target=path.parent
    require(m['schema_version']=='ocr-fresh-evaluation-manifest/1.0','Wrong final evaluation schema')
    c,gate,slots=verify_run(root,m['config']);require(target==inside(root,c['output']),'Evaluation output rebound')
    marker=read(checked(root,m['marker']))
    require(m['marker']['path']==NEW_MARKER and marker['config']==m['config'] and marker['gate']==GATE
        and marker['preparation']==m['preparation'] and marker['candidate_onnx']==c['candidate_onnx']
        and marker['expected_rows']==51 and marker['policy']==POLICY,'Wrong one-shot marker')
    require(checked(root,m['marker']).read_bytes()==(target/'one-shot-marker.json').read_bytes(),'Marker copy changed')
    prepared_path,prepared_manifest=_package(root,m['preparation'])
    require(prepared_path.parent==target/'prepared' and prepared_manifest['config']==m['config'],'Preparation rebound')
    prepared=read(target/'prepared/result.json')
    require(prepared['mode']=='prepare_test' and prepared['test_model_calls']==prepared['gradient_steps']==0
        and prepared['state_sha256'] is None and prepared['config_sha256']==m['config']['sha256'],'Preparation changed')
    verify_tensors(target/'prepared',prepared['results'],c['samples'])
    predictions=read(target/'predictions.json');events=[json.loads(l) for l in (target/'calls.jsonl').read_text('utf-8').splitlines()]
    report=read(target/'report.json');runtime=read(target/'runtime.json')
    require(runtime['preparation']['test_model_calls']==0 and runtime['providers']==['CPUExecutionProvider']
        and runtime['gpu_used'] is False,'Runtime scope changed')
    require(report==report_result(c['rows'],predictions,slots,events,runtime),'Saved metrics/timings do not replay')
    characters=read(checked(root,c['alphabet']))['characters']
    for row,p in zip(c['rows'],predictions,strict=True):
        require(row['id']==p['id'],'Prediction order changed')
        probability=target/'probabilities'/f"{row['id']}.npy"
        if p['status']!='error':
            prob=np.load(probability,allow_pickle=False);text,confidence=decode_ctc(prob,characters)
            require(text==p['prediction'] and confidence==p['confidence'],'Saved probabilities/decoding changed')
    require((target/'index.html').read_bytes()==_html(c['rows'],predictions,slots),'Review page changed')
    checked(root,gate['old_test_marker'])
    return report
