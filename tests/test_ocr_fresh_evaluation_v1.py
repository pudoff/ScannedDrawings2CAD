from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace
import numpy as np
from PIL import Image
import pytest
from cad_recovery import ocr_fresh_evaluation_v1 as module
from cad_recovery.dataset_layout_v1 import encoded,pin,read


def case():
    rows=[]
    for identifier,slot,text,kind,field,index,source,placeholder in [
        ('a','s1-id','A','identifier_literal','drawing_id',0,'s1',False),
        ('b','s1-name','Part','name_literal','part_name',0,'s1',False),
        ('c','s1-name','Name','name_literal','part_name',1,'s1',False),
        ('d','s2-dim','10±0,1','dimension_literal','dimension_1',0,'s2',False),
        ('e','s2-name','A','identifier_literal','part_name',0,'s2',True)]:
        rows.append({'id':identifier,'slot_id':slot,'text':text,'text_kind':kind,'field':field,
            'line_index':index,'source_id':source,'group_id':source+'-group',
            'image_sha256':hashlib.sha256(identifier.encode()).hexdigest(),'image':identifier+'.png',
            'name_is_identifier_placeholder':placeholder})
    slots=[]
    for sid in dict.fromkeys(r['slot_id'] for r in rows):
        part=[r for r in rows if r['slot_id']==sid]
        slots.append({'slot_id':sid,'status':'accepted','text':' '.join(r['text'] for r in part),
            'lines':[{'text':r['text']} for r in part]})
    slots.append({'slot_id':'s2-unknown','status':'unresolved','text':None,'lines':[]})
    predictions=[{'id':r['id'],'prediction':r['text'],'status':'ok','confidence':1.0} for r in rows]
    return rows,predictions,slots


def test_literal_strata_names_multiline_fields_and_unknown_coverage():
    rows,pred,slots=case();value=module.score_holdout(rows,pred,slots)
    assert value['by_kind']['identifier_literal']['sample_count']==2
    assert value['by_kind']['name_literal']['sample_count']==2
    assert value['name_cell_roles']['actual_name']['sample_count']==2
    assert value['name_cell_roles']['identifier_placeholder']['sample_count']==1
    assert value['coverage']['accepted_fields']==4 and value['coverage']['scheduled_fields']==5
    assert value['coverage']['exact_fields']=={'raw':4,'normalized':4}
    assert value['coverage']['raw_verified_fraction_of_scheduled']==0.8
    assert value['field_results'][-1]['raw_exact'] is None


@pytest.mark.parametrize('status',['empty','error'])
def test_bad_outputs_stay_in_line_and_field_denominators(status):
    rows,pred,slots=case();pred[1].update(prediction='',status=status,confidence=0.)
    value=module.score_holdout(rows,pred,slots)
    assert value['overall']['sample_count']==5 and value['overall']['raw']['distance']==4
    assert value['overall']['status_counts'][status]==1
    assert value['coverage']['exact_fields']['raw']==3
    assert value['coverage']['raw_exact_fraction_of_accepted']==0.75
    assert value['coverage']['raw_verified_fraction_of_scheduled']==0.6


def test_frozen_normalization_is_symmetric_without_target_repair():
    rows,pred,slots=case();rows[3]['text']='10-1';slots[2]['lines'][0]['text']='10-1'
    pred[3]['prediction']='10−1';value=module.score_holdout(rows,pred,slots)
    assert value['overall']['raw']['distance']==1 and value['overall']['normalized']['distance']==0
    pred[1]['prediction']='part';value=module.score_holdout(rows,pred,slots)
    assert value['overall']['normalized']['distance']==1


@pytest.mark.parametrize('tamper',['missing','duplicate','wrong_id','wrong_status','error_with_text','empty_ok',
    'nan','confidence_range','confidence_bool','unknown_target','duplicate_slot','missing_slot','line_order','kind','field_status'])
def test_metrics_reject_tampered_results_and_invented_unknowns(tamper):
    rows,pred,slots=case()
    if tamper=='missing':pred.pop()
    elif tamper=='duplicate':pred.append(deepcopy(pred[0]))
    elif tamper=='wrong_id':pred[0]['id']='other'
    elif tamper=='wrong_status':pred[0]['status']='skipped'
    elif tamper=='error_with_text':pred[0]['status']='error'
    elif tamper=='empty_ok':pred[0]['prediction']=''
    elif tamper=='nan':pred[0]['confidence']=float('nan')
    elif tamper=='confidence_range':pred[0]['confidence']=1.1
    elif tamper=='confidence_bool':pred[0]['confidence']=True
    elif tamper=='unknown_target':slots[-1]['text']='guessed'
    elif tamper=='duplicate_slot':slots.append(deepcopy(slots[-1]))
    elif tamper=='missing_slot':slots.pop(0)
    elif tamper=='line_order':rows[1]['line_index']=1
    elif tamper=='kind':rows[0]['text_kind']='legacy_identifier'
    elif tamper=='field_status':slots[-1]['status']='guessed'
    with pytest.raises(ValueError):module.score_holdout(rows,pred,slots)


def test_macro_strata_and_macro_sources_are_not_pooled_CER():
    rows,pred,slots=case();pred[0]['prediction']='AB'
    value=module.score_holdout(rows,pred,slots)
    assert value['macro_raw_CER_by_kind']==pytest.approx((1/2)/3)
    assert value['macro_raw_CER_by_source']==pytest.approx((1/9)/2)
    assert value['overall']['raw']['cer']==pytest.approx(1/16)


def test_global_claim_is_exclusive_across_outputs_and_retained_after_failure(tmp_path):
    (tmp_path/'artifacts').mkdir();c={'candidate_onnx':{'path':'model','sha256':'a'*64}}
    first={'path':'run1','sha256':'b'*64};prep={'path':'prep','sha256':'c'*64}
    marker=module.reserve_once(tmp_path,first,prep,c);before=marker.read_bytes()
    with pytest.raises(FileExistsError):module.reserve_once(tmp_path,{**first,'path':'run2'},prep,c)
    assert marker.read_bytes()==before and read(marker)['config']==first


def test_claimed_cohort_refuses_before_config_model_or_worker_access(tmp_path,monkeypatch):
    p=tmp_path/module.NEW_MARKER;p.parent.mkdir();p.write_bytes(b'partial/failed claim still locks cohort')
    monkeypatch.setattr(module,'verify_run',lambda *a:pytest.fail('claimed cohort must not reach configuration'))
    with pytest.raises(ValueError,match='already claimed'):module.evaluate(tmp_path,{'path':'new-output-config'})
    assert p.read_bytes()==b'partial/failed claim still locks cohort'


@pytest.mark.parametrize('tamper',['gate','parent','policy','preprocessing','comparison','mode','one_shot_marker'])
def test_changed_run_policy_is_rejected_before_reading_dataset_or_model(tmp_path,monkeypatch,tamper):
    config={'schema_version':'ocr-fresh-evaluation-run/1.0','gate':module.GATE,'parent':module.PARENT,
        'policy':module.POLICY,'preprocessing':module.PREPROCESS,'comparison':module.COMPARISON_POLICY_V1,
        'mode':'prepare_test','one_shot_marker':module.NEW_MARKER}
    config=deepcopy(config);config[tamper]='changed';(tmp_path/'run.json').write_bytes(encoded(config))
    monkeypatch.setattr(module,'_context',lambda *a:pytest.fail('bad policy reached source authentication'))
    with pytest.raises(ValueError,match='policy'):module.verify_run(tmp_path,pin(tmp_path,'run.json'))


class Session:
    def __init__(self,fail=False,invalid=False):self.calls=0;self.fail=fail;self.invalid=invalid
    def run(self,*args):
        self.calls+=1
        if self.fail:raise RuntimeError('controlled runtime failure')
        if self.invalid:return [np.array([[[float('nan')]*4]],dtype=np.float32)]
        return [np.array([[[0.,1.,0.,0.],[0.,0.,1.,0.]]],dtype=np.float32)]


@pytest.mark.parametrize('failure',['none','inference','decode','save'])
def test_call_attempts_once_and_journals_failures_without_retry(tmp_path,monkeypatch,failure):
    s=Session(fail=failure=='inference',invalid=failure=='decode')
    if failure=='save':monkeypatch.setattr(module.np,'save',lambda *a,**k:(_ for _ in ()).throw(OSError('disk full')))
    result=module.call_once(s,'x',np.zeros((1,3,48,320),np.float32),'one',['A','B'],tmp_path/'p.npy',tmp_path/'calls.jsonl')
    events=[json.loads(l) for l in (tmp_path/'calls.jsonl').read_text().splitlines()]
    assert s.calls==1 and [e['stage'] for e in events]==['started','completed']
    assert result['status']==('ok' if failure=='none' else 'error')
    module.validate_journal(events,[{'id':'one'}],[result])
    if failure!='none':assert result['prediction']=='' and not (tmp_path/'p.npy').exists()


@pytest.mark.parametrize('tamper',['missing','duplicate','reorder','result','negative','nan','wall'])
def test_journal_rejects_missing_repeated_or_forged_attempts(tamper):
    pred={'id':'one','prediction':'A','status':'ok','confidence':1.0}
    events=[{'stage':'started','id':'one','arm':'selected_rec'},
        {'stage':'completed','id':'one','arm':'selected_rec','result':deepcopy(pred),'inference_seconds':1.,'wall_seconds':2.}]
    if tamper=='missing':events.pop()
    elif tamper=='duplicate':events+=deepcopy(events)
    elif tamper=='reorder':events.reverse()
    elif tamper=='result':events[1]['result']['prediction']='B'
    elif tamper=='negative':events[1]['inference_seconds']=-1.
    elif tamper=='nan':events[1]['inference_seconds']=float('nan')
    elif tamper=='wall':events[1]['wall_seconds']=0.
    with pytest.raises(ValueError):module.validate_journal(events,[{'id':'one'}],[pred])


def test_nested_preparation_manifest_is_authenticated_by_outer_package(tmp_path):
    target=tmp_path/'artifacts/result';(target/'prepared').mkdir(parents=True)
    (target/'prepared/proof.json').write_bytes(b'{}')
    module.package(tmp_path,target/'prepared','prepared/1')
    result=module.package(tmp_path,target,'evaluation/1')
    _,manifest=module._package(tmp_path,result)
    assert 'prepared/manifest.json' in {p['path'] for p in manifest['files']}
    (target/'prepared/manifest.json').write_bytes(b'changed')
    with pytest.raises(ValueError):module._package(tmp_path,result)


@pytest.mark.parametrize('failed_call',[False,True])
def test_end_to_end_fake_session_then_offline_replay_never_constructs_model(tmp_path,monkeypatch,failed_call):
    rows,pred,slots=case();root=tmp_path;(root/'artifacts').mkdir()
    (root/'model.onnx').write_bytes(b'fake-model');(root/'old.json').write_bytes(b'old-marker')
    (root/'runtime.json').write_bytes(encoded({'python':'not-used'}))
    chars=list('APartNme10±,');(root/'alphabet.json').write_bytes(encoded({'characters':chars}))
    (root/'run.json').write_bytes(b'{}');cp=pin(root,'run.json')
    c={'output':'artifacts/result','candidate_onnx':pin(root,'model.onnx'),'runtime_config':pin(root,'runtime.json'),
        'alphabet':pin(root,'alphabet.json'),'rows':rows,'samples':[{'id':r['id']} for r in rows],'pins':[]}
    gate={'old_test_marker':pin(root,'old.json')};monkeypatch.setattr(module,'verify_run',lambda *a:(c,gate,slots))
    def worker(workspace,config,target,*args):
        assert not (root/module.NEW_MARKER).exists()
        target.mkdir();(target/'tensors').mkdir();results=[]
        for i,row in enumerate(rows):
            x=np.full((3,48,320),i,dtype=np.float32);np.save(target/'tensors'/f"{row['id']}.npy",x,allow_pickle=False)
            results.append({'id':row['id'],'preprocessing':{'tensor_sha256':hashlib.sha256(x.tobytes()).hexdigest()}})
        value={'mode':'prepare_test','test_model_calls':0,'gradient_steps':0,'state_sha256':None,
            'config_sha256':config['sha256'],'results':results,'memory':{'peak_bytes':0}}
        (target/'result.json').write_bytes(encoded(value));return value
    class FakeSession:
        def __init__(self):self.calls=0
        def get_inputs(self):return [SimpleNamespace(name='x',type='tensor(float)',shape=[None,3,48,320])]
        def get_providers(self):return ['CPUExecutionProvider']
        def run(self,_,inputs):
            assert (root/module.NEW_MARKER).is_file()
            marker=read(root/module.NEW_MARKER)
            assert marker['config']==cp and marker['candidate_onnx']==c['candidate_onnx']
            index=int(inputs['x'][0,0,0,0]);self.calls+=1
            if failed_call and index==1:raise RuntimeError('controlled failed row')
            text=rows[index]['text'];prob=np.zeros((1,len(text),len(chars)+2),np.float32)
            for i,char in enumerate(text):prob[0,i,chars.index(char)+1]=1.
            return [prob]
    session=FakeSession();monkeypatch.setattr(module,'_worker',worker)
    monkeypatch.setattr(module,'ort_session',lambda *a:session)
    monkeypatch.setattr(module,'_memory',lambda:{'peak_bytes':0})
    manifest=module.evaluate(root,cp);assert session.calls==len(rows)
    marker=(root/module.NEW_MARKER).read_bytes()
    monkeypatch.setattr(module,'_worker',lambda *a:pytest.fail('offline verification invoked worker'))
    monkeypatch.setattr(module,'ort_session',lambda *a:pytest.fail('offline verification constructed model'))
    report=module.verify_evaluation(root,manifest)
    assert report['actual_model_calls']==5 and report['metrics']['overall']['sample_count']==5
    assert report['metrics']['overall']['status_counts'].get('error',0)==int(failed_call)
    with pytest.raises(ValueError,match='already claimed'):module.evaluate(root,cp)
    assert session.calls==5 and (root/module.NEW_MARKER).read_bytes()==marker
    assert (root/'old.json').read_bytes()==b'old-marker'
    (root/'artifacts/result/report.json').write_bytes(b'{}')
    with pytest.raises(ValueError):module.verify_evaluation(root,manifest)
