"""A separate source-fresh, pixel-reviewed OCR holdout. No model imports or calls."""
from collections import Counter
from copy import deepcopy
from html import escape
from math import ceil, floor
from pathlib import Path
import json
import numpy as np
from PIL import Image
from .dataset_layout_v1 import checked, encoded, inside, pin, read
from .dataset_preparation_v1 import crop_source
from .family_catalog_v2 import require
from .ocr_fresh_selection_v1 import POLICY, choose_sources
from .ocr_release_v3 import _package

SELECTION = {'path':'configs/reviews/ocr-fresh-holdout-selection-v1.json',
             'sha256':'b2f34d091be171fd3a11fa352e1b52c854154d6cfd447379c33b0670c8c3a8f0'}
CODE = ['src/cad_recovery/'+n+'.py' for n in (
    'ocr_fresh_holdout_v1','ocr_fresh_selection_v1','dataset_preparation_v1',
    'dataset_layout_v1','family_catalog_v2','ocr_release_v3','raster_regions','png_stream',
    'clear_pairs','label_release_v1')]+['scripts/localize_ocr_fresh_holdout_v1.py',
    'scripts/release_ocr_fresh_holdout_v1.py']
FLAGS = {'OCR_run':False,'DXF_read_for_transcription':False,'training_started':False,
         'human_verified':False,'independent_evaluation':False,'semantic_family_approved':False,
         'training_eligible_for_geometry':False,'production_model_updated':False}
NEW_MARKER = 'artifacts/ocr-fresh-holdout-test-once-v1.json'


def box_ok(box, size):
    require(isinstance(box,list) and len(box)==4 and all(type(x) is int for x in box), 'Invalid crop coordinates')
    x0,y0,x1,y1=box;w,h=size
    require(0<=x0<x1<=w and 0<=y0<y1<=h and (x1-x0)*(y1-y0)<=16777216,'Crop outside original source or too large')


def validate_slots(selected, slots, old_hashes):
    """Bind all scheduled positions, including unresolved ones, before releasing targets."""
    require(len(selected)==12 and len(slots)==48,'Incomplete fixed cohort/field ledger')
    expected=[(i,f) for i in range(1,13) for f in POLICY['fields']]
    require([(s['index'],s['field']) for s in slots]==expected,'Changed slot order/coverage')
    identities=set();hashes=set()
    for slot in slots:
        index=slot['index'];require(type(index) is int,'Invalid source index')
        source=selected[index-1];field=slot['field']
        require(slot['slot_id']==f'fresh-{index:02d}-{field}' and slot['source_path']==source['input']
            and slot['source_sha256']==source['input_sha256'] and slot['group_id']==source['group_id']
            and slot['split']=='test','Rebound slot source/group/split')
        require(type(slot['name_is_identifier_placeholder']) is bool and
            (field=='part_name' or not slot['name_is_identifier_placeholder']),'Invalid identifier placeholder')
        require(slot['status'] in ('accepted','unresolved','absent','quarantine'),'Unknown field disposition')
        context=slot['context']['box_xyxy'];box_ok(context,source['source_size_px'])
        if slot['status']!='accepted':
            require(not slot['lines'] and isinstance(slot['reason'],str) and slot['reason'].strip(), 'Unresolved field has targets or lacks reason')
            continue
        require(bool(slot['lines']),'Accepted field lacks physical lines')
        require(field=='part_name' or len(slot['lines'])==1,'Unexpected multiline non-name field')
        for line in slot['lines']:
            text=line['text'];require(isinstance(text,str) and text and text==text.strip()
                and not any(ord(c)<32 or ord(c)==127 for c in text),'Invalid literal target')
            box=line['box_xyxy'];box_ok(box,source['source_size_px'])
            require(context[0]<=box[0]<box[2]<=context[2] and context[1]<=box[1]<box[3]<=context[3], 'Line outside context')
            require(type(line['rotation_clockwise']) is int and line['rotation_clockwise'] in (0,90,180,270),'Invalid rotation')
            identity=(slot['source_sha256'],tuple(box),line['rotation_clockwise'])
            require(identity not in identities,'Repeated physical line');identities.add(identity)
            sha=line['image']['sha256']
            require(sha not in old_hashes,'Old OCR crop leakage; quarantine explicitly, do not replace source')
            require(sha not in hashes,'Duplicate new crop');hashes.add(sha)
    return slots


def replay_pixels(root, source, item, rotation=0):
    rgb,proof=crop_source(root,source,item['box_xyxy'])
    require(proof==item['pixel_provenance'],'Changed original-pixel provenance')
    expected=np.rot90(rgb,-rotation//90) if rotation else rgb
    with Image.open(checked(root,item['image'])) as im:
        require(im.mode=='RGB' and np.array_equal(np.asarray(im),expected),'Crop differs from original RGB pixels')


def _dependencies(root, config_pin):
    config=read(checked(root,config_pin))
    require(config['schema_version']=='ocr-fresh-holdout-release-config/1.0' and config['selection']==SELECTION,'Wrong fresh release protocol')
    require([p['path'] for p in config['code']]==CODE,'Incomplete release code inventory')
    for p in config['code']:checked(root,p)
    selection=read(checked(root,SELECTION));require(selection['policy']==POLICY,'Selection policy changed')
    for p in selection['code']:checked(root,p)
    for key in ('parent','candidate_checkpoint','candidate_onnx','old_test_marker'):checked(root,selection[key])
    _package(root,selection['candidate_qualification'])
    path,_=_package(root,selection['release']);base=path.parent
    old_rows=[json.loads(line) for split in ('train','validation','test')
        for line in (base/f'{split}.jsonl').read_text(encoding='utf-8').splitlines()]
    selected,audit,exclusions=choose_sources(read(base/'source-dispositions.json'),read(base/'labels.json'),old_rows)
    require(selected==selection['selected_sources'] and audit==selection['source_audit']
        and exclusions==selection['exclusions'],'Fresh source selection does not replay')
    require(len({s['input_sha256'] for s in selected})==12 and len({s['group_id'] for s in selected})==12,'Selected cohort repeats source/group')
    labels=read(checked(root,config['labels']));readings=read(checked(root,config['readings']))
    require(labels['selection']==SELECTION and labels['readings']==config['readings']
        and readings['selection_sha256']==SELECTION['sha256']
        and readings['review_status']=='original_pixel_review_complete_by_implementing_assistant'
        and readings['human_verified'] is False and readings['independent_evaluation'] is False,'Unreviewed/rebound literal readings')
    require(len(readings['sources'])==12 and len(config['navigation'])==12,'Incomplete reading source coverage')
    slots=validate_slots(selected,labels['slots'],{r['image_sha256'] for r in old_rows})
    for i,(spec,source) in enumerate(zip(readings['sources'],selected,strict=True),1):
        require(spec['index']==i and type(spec['index']) is int and [f['field'] for f in spec['fields']]==POLICY['fields'],'Rebound reading positions')
        with Image.open(checked(root,config['navigation'][i-1])) as im:nw,nh=im.size
        w,h=source['source_size_px']
        for f,slot in zip(spec['fields'],slots[(i-1)*4:i*4],strict=True):
            require(slot['status']==f.get('status','accepted') and slot['reason']==f.get('reason')
                and slot['layout']==spec['layout'] and slot['name_is_identifier_placeholder']==f.get('name_is_identifier_placeholder',False),'Reading disposition differs')
            require(isinstance(f['lines'],list) and bool(f['lines']) and
                len(slot['lines'])==(len(f['lines']) if slot['status']=='accepted' else 0),'Reading line inventory differs')
            boxes=[]
            for j,line in enumerate(f['lines']):
                x0,y0,x1,y1=line['box'];box=[floor(x0*w/nw),floor(y0*h/nh),ceil(x1*w/nw),ceil(y1*h/nh)];boxes.append(box)
                box_ok(box,[w,h])
                if slot['status']=='accepted':
                    require(slot['lines'][j]['text']==line['text'] and slot['lines'][j]['box_xyxy']==box
                        and slot['lines'][j]['rotation_clockwise']==line.get('rotation',0),'Reading text/crop differs')
                else:require(line['text'] is None,'Unknown reading contains inferred text')
            context=[max(0,min(b[0] for b in boxes)-40),max(0,min(b[1] for b in boxes)-40),
                     min(w,max(b[2] for b in boxes)+40),min(h,max(b[3] for b in boxes)+40)]
            require(slot['context']['box_xyxy']==context,'Reading context differs')
            replay_pixels(root,source,slot['context'])
            for line in slot['lines']:replay_pixels(root,source,line,line['rotation_clockwise'])
    return config,selection,slots,old_rows


def _payload(root, selection, slots):
    labels=[];rows=[];images={}
    for slot in slots:
        s=deepcopy(slot);source=selection['selected_sources'][s['index']-1]
        context=s['context'];name=f'contexts/{s["slot_id"]}.png'
        images[name]=checked(root,context['image']).read_bytes();context['image']={'path':name,'sha256':context['image']['sha256']}
        s.update(FLAGS);s['text']=' '.join(l['text'] for l in s['lines']) if s['status']=='accepted' else None
        s['native_visual_status']=source['visual_status'];s['excluded_native_target']=source['excluded_native_target']
        s['crop_contains_rule_pixels']='possible; retained without cleanup'
        for j,line in enumerate(s['lines']):
            identifier=s['slot_id']+f'-{j+1}';name=f'images/{identifier}.png'
            images[name]=checked(root,line['image']).read_bytes();line['image']={'path':name,'sha256':line['image']['sha256']}
            kind='dimension_literal' if s['field'].startswith('dimension') else 'identifier_literal' if s['field']=='drawing_id' or s['name_is_identifier_placeholder'] else 'name_literal'
            rows.append({'id':identifier,'slot_id':s['slot_id'],'line_index':j,'source_id':f'fresh-{s["index"]:02d}',
                'source_path':s['source_path'],'source_sha256':s['source_sha256'],'group_id':s['group_id'],
                'source_size_px':source['source_size_px'],'source_box_xyxy':line['box_xyxy'],
                'rotation_cw_degrees':line['rotation_clockwise'],'split':'test','task':'ocr_recognition',
                'field':s['field'],'text':line['text'],'text_kind':kind,'name_is_identifier_placeholder':s['name_is_identifier_placeholder'],
                'image':name,'image_sha256':line['image']['sha256'],'pixel_provenance':line['pixel_provenance'],
                'native_visual_status':source['visual_status'],'excluded_native_target':source['excluded_native_target'],**FLAGS})
        labels.append(s)
    report={'schema_version':'ocr-fresh-holdout-report/1.0','task':'P16.2e','selection':SELECTION,
        'source_count':12,'eligible_sources':selection['exclusions']['eligible'],
        'previous_test_sources_excluded':selection['exclusions']['previously_reviewed_test_sources'],
        'scheduled_slots':48,'slot_statuses':dict(Counter(s['status'] for s in slots)),
        'accepted_physical_lines':len(rows),'text_kinds':dict(Counter(r['text_kind'] for r in rows)),
        'native_visual_statuses':dict(Counter(r['visual_status'] for r in selection['selected_sources'])),
        'accepted_placeholder_fields':sum(s['status']=='accepted' and s['name_is_identifier_placeholder'] for s in slots),
        'scope':'recognition of fixed original-pixel ROI lines; not detector, whole-sheet OCR or CAD quality',
        'coverage':'Always report all 48 slots; unresolved slots have no CER target and must remain in coverage denominator',
        'source_and_group_disjoint':True,'family_independence_proven':False,
        'test_model_calls':0,'old_test_inference_allowed':False,**FLAGS}
    return labels,rows,images,report


def _files(root,selection,slots,config_pin):
    labels,rows,images,report=_payload(root,selection,slots)
    html=['<!doctype html><meta charset="utf-8"><title>Fresh OCR holdout review</title>',
          '<p>Original RGB crops. Assistant review; human_verified=false; no model run. Unknowns retained.</p>']
    for s in labels:
        html.append('<hr><p>'+escape(s['slot_id']+' / '+s['status']+' / '+str(s['text']))+'</p>')
        html.append('<img style="max-width:100%" src="'+s['context']['image']['path']+'">')
    return {**images,'labels.json':encoded({'selection':SELECTION,'slots':labels,**FLAGS}),
        'test.jsonl':b''.join((json.dumps(r,ensure_ascii=True,sort_keys=True,allow_nan=False)+'\n').encode('utf-8') for r in rows),
        'report.json':encoded(report),'source-audit.json':encoded(selection['source_audit']),
        'protocol.json':encoded({'config':config_pin,'policy':POLICY,'candidate_checkpoint':selection['candidate_checkpoint'],
            'candidate_onnx':selection['candidate_onnx'],'old_test_marker':selection['old_test_marker']}),
        'index.html':'\n'.join(html).encode('utf-8')}


def build_release(root,config_pin,output):
    root=Path(root).resolve();_,selection,slots,_=_dependencies(root,config_pin)
    target=inside(root,output);require(target.parent==root/'artifacts' and not target.exists(),'Use a new artifacts release directory')
    files=_files(root,selection,slots,config_pin);target.mkdir()
    for name,data in files.items():
        path=target/name;path.parent.mkdir(exist_ok=True,parents=True);path.write_bytes(data)
    manifest={'schema_version':'ocr-fresh-holdout-release/1.0','config':config_pin,
        'files':[pin(target,n) for n in sorted(files)]}
    (target/'manifest.json').write_bytes(encoded(manifest))
    return pin(root,output+'/manifest.json')


def verify_release(root,release_pin):
    root=Path(root).resolve();path,manifest=_package(root,release_pin)
    require(manifest['schema_version']=='ocr-fresh-holdout-release/1.0','Wrong holdout schema')
    _,selection,slots,_=_dependencies(root,manifest['config'])
    files=_files(root,selection,slots,manifest['config'])
    require({p['path'] for p in manifest['files']}==set(files),'Holdout payload inventory changed')
    for name,data in files.items():require((path.parent/name).read_bytes()==data,'Holdout payload does not replay: '+name)
    return read(path.parent/'report.json'),[json.loads(l) for l in (path.parent/'test.jsonl').read_text(encoding='utf-8').splitlines()]


def validate_admission(gate,selection,release_pin,report):
    require(gate['schema_version']=='ocr-fresh-holdout-evaluation/1.0' and gate['selection']==SELECTION
        and gate['release']==release_pin,'Wrong evaluation cohort')
    for key in ('candidate_checkpoint','candidate_onnx','candidate_qualification','old_test_marker'):
        require(gate[key]==selection[key],'Candidate or old marker changed')
    require(gate['new_one_shot_marker']==NEW_MARKER and gate['old_test_inference_allowed'] is False
        and gate['evaluation_inference_executed'] is False and gate['model_selection_allowed'] is False
        and gate['training_allowed'] is False and gate['independent_evaluation'] is False
        and gate['recognition_only'] is True and gate['one_fixed_candidate'] is True,'Wrong evaluation scope')
    require(gate['expected_rows']==report['accepted_physical_lines'] and gate['scheduled_slots']==report['scheduled_slots']
        and gate['unresolved_slots']==report['slot_statuses'].get('unresolved',0),'Evaluation denominator differs')


def check_admission(root,gate_pin):
    """Read-only admission; never creates/clears a marker or imports an inference runtime."""
    root=Path(root).resolve();gate=read(checked(root,gate_pin));report,_=verify_release(root,gate['release'])
    selection=read(checked(root,SELECTION));validate_admission(gate,selection,gate['release'],report)
    require([p['path'] for p in gate['code']]==CODE,'Changed gate code inventory')
    for p in gate['code']:checked(root,p)
    for key in ('candidate_checkpoint','candidate_onnx','candidate_qualification','old_test_marker'):checked(root,gate[key])
    qpath,_=_package(root,gate['candidate_qualification']);q=read(qpath.parent/'config.json')
    require(q['checkpoint']==gate['candidate_checkpoint'] and q['mode']=='validation','Candidate qualification rebound')
    onnx=pin(root,(qpath.parent/'export/selected.onnx').relative_to(root).as_posix())
    require(onnx==gate['candidate_onnx'],'Qualified ONNX differs')
    require(not inside(root,gate['new_one_shot_marker']).exists(),'Fresh evaluation has already been claimed; no rerun')
    return {'admission_checked':True,'future_fresh_evaluation_ready':True,'test_model_calls':0,
        'old_marker_preserved':True,'fresh_marker_absent':True,'independent_evaluation':False}
