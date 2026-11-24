"""Limited title-field OCR controls under the frozen two-group V3 protocol.

The complete inherited train stream remains byte-identical. Accepted, uncertain
and absent field slots are all accounted for; neither builder nor loader runs a
model. Group reservations are enforced together with all existing train records.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime
from html import escape
from pathlib import Path
import json
import runpy

import numpy as np

from .clear_pairs import contained, digest
from .coallocation_policy_v1 import validate_assignments_v1
from .coallocation_review_v1 import source_regions_v1
from .family_catalog_v2 import stable_digest
from .label_release_v1 import checked_pin
from .ocr_control_line_schema_v2 import ControlReviewV2, FIELDS
from .ocr_label_schema_v1 import make_ocr_transform_v1
from .ocr_release_v3 import load_ocr_training_v3, _package, _pixels, encoded, pin, read, require
from .raster_regions import read_prepared_region

PROTOCOL={'path':'configs/reviews/ocr-control-lines-protocol-v2.json','sha256':'bea086abe02ece838162b2bb840a89e174eb742e723b1ad1ab18bfcbafbdf010'}
SCOPE_PROTOCOL={'path':'configs/reviews/ocr-control-protocol-v3.json','sha256':'3b6d60097cb888af0d2c0625a5f6803ced3d7e86c05a527a2d6478effa253270'}
PARENT={'path':'artifacts/ocr-control-scope-final-v3.json','sha256':'4d15f21f43d6b3828e6e0915419e68bf8219537b21b5db1a0fd26056afffd72a'}
PREVIOUS={'path':'artifacts/ocr-release-v3/manifest.json','sha256':'40459498a211b577e6cdfbca9d8772b4c9b04a9768b2d9d842395995abb87fdd'}
READING={'path':'artifacts/ocr-control-line-reading-v2.json','sha256':'46e855fe767477f6060e21caaf8d1e1a1b8212b109e0a4530a7c6087f18a7033'}
CROPS={'path':'artifacts/ocr-control-line-crop-fix-v2/manifest.json','sha256':'eae02d95fd46913c455437a8767a11df3d28e4a8ce6fcb648aa86ca74a30fbbb'}
SCOPE='limited_title_fields_two_groups_known_constraints_not_external_independence'
CODE=['src/cad_recovery/'+name+'.py' for name in ('ocr_control_release_v2','ocr_control_line_schema_v2',
    'ocr_line_schema_v3','ocr_label_schema_v1','ocr_release_v3','coallocation_policy_v1','coallocation_review_v1',
    'family_catalog_v2','raster_regions','png_stream','clear_pairs','label_release_v1','label_schema_v1',
    'ocr_control_scope_v3','ocr_control_relations_v2','ocr_control_selection_v2','ocr_control_selection_v1',
    'ocr_control_review_v1')]+['scripts/build_ocr_control_scope_v3.py',
    'scripts/build_ocr_control_relations_v2.py','scripts/prepare_ocr_control_v2.py',
    'scripts/release_ocr_control_v2.py','artifacts/prepare-ocr-control-lines-v2.py',
    'artifacts/prepare-ocr-control-line-crops-v2.py','artifacts/refine-control-line-crop-v2.py',
    'artifacts/author-control-line-reading-v2.py']
ASSIGNMENT_FIELDS=('id','source_path','source_sha256','image_sha256','split')


def same(a,b):return stable_digest(a)==stable_digest(b)
def timestamp(value):
    result=datetime.fromisoformat(value)
    require(result.tzinfo is not None,'Missing timezone');return result


def validate_reading(protocol,proposals,reading,relationships):
    """Validate all fixed slots and inherited uncertainty, independent of file I/O."""
    review=ControlReviewV2.model_validate(reading).model_dump(mode='json')
    require(timestamp(review['reviewed_at_utc'])>=timestamp(proposals['created_at_utc'])>=timestamp(protocol['created_before_line_review_utc']),
        'Line evidence predates its frozen protocol or crop proposal')
    require([r['id'] for r in review['sources']]==[r['id'] for r in protocol['sources']], 'Incomplete or reordered control source inventory')
    require(len(proposals['sources'])==len(protocol['sources']) and protocol['fields']==list(FIELDS)
        and protocol['candidate_count']==3*len(protocol['sources']),'Changed fixed field inventory')
    require(review['viewed_contacts']==proposals['contacts'],'Missing reviewed contact images')
    require(review['additional_viewed_images']==[r['after']['image_pin'] for r in proposals.get('crop_amendments',[])],
        'Missing final corrected crop views')
    for source,binding,crop in zip(review['sources'],protocol['sources'],proposals['sources'],strict=True):
        require(source['id']==crop['source_id'] and all(same(source[k],binding[k]) for k in
            ('input_path','input_sha256','preparation','source_size_px')),'Control reading source binding differs')
        require([r['field'] for r in crop['fields']]==list(FIELDS),'Proposal omits a fixed field')
        for line,proposal in zip(source['fields'],crop['fields'],strict=True):
            keys=('id','field','parent_kind','source_box_xyxy','rotation_cw_degrees','raw_crop_pin','image_pin')
            require(all(same(line[k],proposal[k]) for k in keys),'Final crop differs from the reviewed proposal')
            if proposal['source_box_xyxy'] is None:
                require(line['decision']=='absent','A missing field cannot acquire an invented crop or target')
            else:
                require(line['decision']!='absent','A present field cannot be hidden as absent')
                box=line['source_box_xyxy'];parent=binding['parents'][line['parent_kind']]['box_source_xyxy']
                require(parent[0]<=box[0]<box[2]<=parent[2] and parent[1]<=box[1]<box[3]<=parent[3], 'Field outside frozen parent region')
            if line['field'] in binding['inherited_uncertain_fields']:
                require(line['decision']=='unresolved' and line['text'] is None,'Inherited ambiguity cannot silently become an OCR target')
            if line['decision']=='accepted_for_ocr_transcription':
                observed=relationships[binding['id']]
                previous=observed['title_name'] if line['field']=='part_name' else observed['title_designation']
                require(line['field']!='adjacent_reference' and line['text']==previous,
                    'New or inconsistent identity requires a renewed relationship review before release')
        require(any(r['decision']=='accepted_for_ocr_transcription' for r in source['fields']), 'Every reserved source needs at least one accepted line')
    return review


def _base(root):
    protocol=read(checked_pin(root,PROTOCOL));scope=read(checked_pin(root,SCOPE_PROTOCOL));parent=read(checked_pin(root,PARENT))
    require(protocol['schema_version']=='ocr-control-lines-protocol/2.0' and protocol['task']=='P14.1b.7c'
        and protocol['scope_protocol']==SCOPE_PROTOCOL and protocol['parent_final']==PARENT
        and protocol['previous_ocr_manifest']==PREVIOUS and parent['protocol']==SCOPE_PROTOCOL,'Changed canonical control lineage')
    require(all(protocol[k] is False for k in ('OCR_run','training_started','human_verified','native_CAD_opened',
        'independent_evaluation','physical_geometry_approved')),'Invalid control protocol claims')
    checked_pin(root,protocol['extractor'])
    _,scope_manifest=_package(root,scope['scope_manifest'])
    scope_config=read(checked_pin(root,scope_manifest['config']))
    for record in scope_config['code']:checked_pin(root,record)
    recipe=runpy.run_path(str(root/'scripts/build_ocr_control_scope_v3.py'),run_name='control_release_scope')
    base,manifest=recipe['package'](root,scope['scope_manifest'])
    require(same(scope,recipe['frozen_protocol'](root,scope['scope_manifest'],scope['frozen_at_utc'])), 'Scope protocol differs from pinned reservations')
    _,values=recipe['inputs'](root,read(checked_pin(root,manifest['config'])))
    catalog,policy,exclusions,overlay,current_routing=values[:5]
    require(timestamp(protocol['created_before_line_review_utc'])>=timestamp(scope['frozen_at_utc']), 'Line protocol predates scope freeze')
    require(protocol['scope_manifest']==scope['scope_manifest'] and protocol['source_probes']==scope['review_probes']
        and protocol['source_readings']==scope['review_readings'],'Changed relationship evidence lineage')
    probes=values[7];by_id={r['id']:r for r in probes['sources']}
    expected=[]
    for group in scope['reservations']:
        for identifier in group['source_ids']:
            row=by_id[identifier]
            expected.append({**{k:row[k] for k in ('index','id','input_path','input_sha256','group_id','preparation','source_size_px')},
                'reserved_split':group['reserved_split'],
                'parents':{r['name']:{'box_source_xyxy':r['box_source_px'],'image':r['image']} for r in row['regions']},
                'inherited_uncertain_fields':['designation'] if group['reserved_split']=='validation' else ['adjacent_reference']})
    require(same(protocol['sources'],expected) and len(expected)==23 and protocol['candidate_count']==69,'Changed complete reserved-source scope')
    previous=load_ocr_training_v3(root,PREVIOUS);old_dir=checked_pin(root,PREVIOUS).parent
    require(previous['count']==len(current_routing['assignments'])==90,'Complete inherited train registry is required')
    proposals=read(checked_pin(root,CROPS));checked_pin(root,proposals['extractor'])
    original=read(checked_pin(root,proposals['previous_proposals']));checked_pin(root,original['extractor'])
    require(proposals['protocol']==PROTOCOL and proposals['candidate_count']==69,'Changed line-crop protocol')
    reading=read(checked_pin(root,READING))
    require(reading['protocol']==PROTOCOL and reading['crop_proposals']==CROPS,'Changed line reading provenance')
    relationships={r['id']:r for r in values[8]['sources']}
    review=validate_reading(protocol,proposals,reading,relationships)
    for record in [*review['viewed_contacts'],*review['additional_viewed_images']]:checked_pin(root,record)
    for source in review['sources']:
        for line in source['fields']:
            for record in (line['raw_crop_pin'],line['image_pin']):
                if record is not None:checked_pin(root,record)
    return {'protocol':protocol,'scope':scope,'catalog':catalog,'policy':policy,'exclusions':exclusions,'overlay':overlay,
        'current_routing':current_routing,'previous':previous,'old_labels':read(old_dir/'labels.json'),
        'old_registry':read(old_dir/'task-splits.json'),'old_dir':old_dir,'proposals':proposals,'review':review}


def prepare_control_config_v2(workspace,output_path):
    root=Path(workspace).resolve();target=contained(root,output_path)
    require(target.parent==root/'configs/reviews' and not target.exists(),'Use a new control release configuration')
    _base(root)
    config={'schema_version':'ocr-control-release-config/2.0','task':'ocr_recognition',
        'protocol':PROTOCOL,'reading':READING,'crops':CROPS,'code':[pin(root,p) for p in CODE]}
    target.write_bytes(encoded(config));return pin(root,output_path)


def _verify_inputs(root,config_pin):
    config=read(checked_pin(root,config_pin))
    require(set(config)=={'schema_version','task','protocol','reading','crops','code'}
        and config['schema_version']=='ocr-control-release-config/2.0' and config['task']=='ocr_recognition'
        and config['protocol']==PROTOCOL and config['reading']==READING and config['crops']==CROPS,'Changed control release config')
    require(config['code']==[pin(root,p) for p in CODE],'Changed control release implementation')
    return _base(root)


def _documents(root,config_pin,verified,progress=False):
    protocol=verified['protocol'];overlay=verified['overlay'];scope=verified['scope'];review=verified['review']
    bindings={r['id']:r for r in protocol['sources']};groups={r['group_id']:r for r in overlay['groups']}
    catalog=verified['catalog'];policy=verified['policy'];exclusions=verified['exclusions']
    file_by_path={r['path']:r for r in catalog['files']}
    images={};samples=[];unresolved=[];absent=[];ledger=[];evidence=[]
    for index,source in enumerate(review['sources']):
        binding=bindings[source['id']];parents=binding['parents']
        original=source_regions_v1(checked_pin(root,{'path':'dataset/'+source['input_path'],'sha256':source['input_sha256']}),
            source['input_sha256'],{k:r['box_source_xyxy'] for k,r in parents.items()})
        prep_dir=checked_pin(root,source['preparation']).parent;parent_evidence={}
        for kind,parent in parents.items():
            box=parent['box_source_xyxy'];a,provenance=read_prepared_region(prep_dir,box,max_pixels=16_777_216)
            require(np.array_equal(a,original[kind]) and np.array_equal(a,_pixels(root,parent['image'],[box[2]-box[0],box[3]-box[1]])), 'Parent differs from original/prepared pixels')
            image='parents/'+source['id']+'-'+kind+'.png';images[image]=checked_pin(root,parent['image']).read_bytes()
            parent_evidence[kind]={**parent,'output_image':image,'tile_provenance':provenance,'original_pixels_equal':True}
        for line in source['fields']:
            record={**deepcopy(line),'source_id':source['id'],'reserved_split':binding['reserved_split'],'review':READING}
            if line['decision']=='absent':
                record['absence_evidence']=parent_evidence[line['parent_kind']];absent.append(record);ledger.append(record);continue
            box=line['source_box_xyxy'];parent=parents[line['parent_kind']]['box_source_xyxy']
            raw=original[line['parent_kind']][box[1]-parent[1]:box[3]-parent[1],box[0]-parent[0]:box[2]-parent[0]]
            oriented=np.rot90(raw,-(line['rotation_cw_degrees']//90)).copy()
            for key,array in [('raw_crop_pin',raw),('image_pin',oriented)]:
                require(np.array_equal(array,_pixels(root,line[key],[array.shape[1],array.shape[0]])), 'Reviewed field crop or rotation differs from original')
            record.update(make_ocr_transform_v1(box,line['rotation_cw_degrees']))
            accepted=line['decision']=='accepted_for_ocr_transcription'
            image=('images/' if accepted else 'quarantine/')+line['id']+'.png'
            images[image]=checked_pin(root,line['image_pin']).read_bytes();record['output_image']=image
            if accepted:
                yy,xx=np.where(np.any(oriented<200,axis=2));require(len(xx)>0,'Accepted field is blank')
                ink=[int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1]
                margins=[ink[0],ink[1],oriented.shape[1]-ink[2],oriented.shape[0]-ink[3]]
                require(min(margins)>0,'Accepted text touches crop boundary')
                record.update(ink_box_image_xyxy=ink,white_margins_px_ltrb=margins,diagnostic_ink_threshold_exclusive=200)
                samples.append(record)
            else:unresolved.append(record)
            ledger.append(record)
        evidence.append({'source_id':source['id'],'source_sha256':source['input_sha256'],'preparation':source['preparation'],'parents':parent_evidence})
        if progress and (index+1)%5==0:print(json.dumps({'control_sources_replayed':index+1,'total':len(review['sources'])}),flush=True)
    rows={split:[] for split in ('train','validation','test')};assignments=[]
    old_registry=verified['old_registry'];old_assignments={r['id']:r for r in old_registry['assignments']}
    for row in verified['previous']['records']:
        require(row['split']=='train','Inherited sample changed split')
        old=old_assignments[row['id']]
        require(old['source_sha256']==row['source_sha256'] and old['image_sha256']==row['image_sha256'],'Inherited sample binding differs')
        image=contained(verified['old_dir'],row['image']);require(digest(image)==row['image_sha256'],'Inherited image changed')
        images[row['image']]=image.read_bytes();rows['train'].append({k:v for k,v in row.items() if k!='image_path'})
        assignments.append({k:old[k] for k in ASSIGNMENT_FIELDS})
    for sample in samples:
        binding=bindings[sample['source_id']];path=binding['input_path'];split=binding['reserved_split']
        rows[split].append({'id':sample['id'],'source_id':sample['source_id'],'image':sample['output_image'],
            'image_sha256':sample['image_pin']['sha256'],'text':sample['text'],'source_sha256':binding['input_sha256'],
            'group_id':file_by_path[path]['group_id'],'split':split,'task':'ocr_recognition'})
        assignments.append({'id':sample['id'],'source_path':path,'source_sha256':binding['input_sha256'],
            'image_sha256':sample['image_pin']['sha256'],'split':split})
    all_rows=[r for part in rows.values() for r in part]
    require(len({r['id'].casefold() for r in all_rows})==len(all_rows),'Repeated sample id')
    by_hash=defaultdict(list)
    for row in all_rows:by_hash[row['image_sha256']].append(row)
    require(all(len({r['text'] for r in part})==1 for part in by_hash.values()), 'Identical images have contradictory texts')
    commitments=deepcopy(scope['train_commitments'])
    for group in scope['reservations']:
        require(group['member_paths']==groups[group['group_id']]['members'],'Incomplete reserved group')
        commitments.append({'coallocation_group_id':group['group_id'],'member_paths':group['member_paths'],
            'split':group['reserved_split'],'reason':'inherit_frozen_limited_control_group_reservation',
            'semantic_family_approved':False,'independent_evaluation':False})
    require(len({r['coallocation_group_id'] for r in commitments})==len(commitments),'Cross-split group reservation')
    for group in commitments:
        require(group['member_paths']==groups[group['coallocation_group_id']]['members'],'Incomplete whole-group commitment')
    reserve_rows=[{'id':'reservation-'+stable_digest([r['coallocation_group_id'],p])[:32],
        'source_path':p,'source_sha256':file_by_path[p]['sha256'],'split':r['split']} for r in commitments for p in r['member_paths']]
    routing=validate_assignments_v1(catalog,policy,exclusions,overlay,assignments)
    combined=validate_assignments_v1(catalog,policy,exclusions,overlay,assignments+reserve_rows)
    require(len(ledger)==protocol['candidate_count'] and len(samples)+len(unresolved)+len(absent)==len(ledger),'Incomplete field ledger')
    require({r['source_id'] for r in samples}==set(bindings),'Reserved source has no accepted line')
    parent_train=(verified['old_dir']/'train.jsonl').read_bytes()
    require([json.loads(r) for r in parent_train.splitlines()]==rows['train'],'Inherited train JSONL differs')
    by_source={r['id']:r for r in protocol['sources']};dispositions=deepcopy(read(checked_pin(root,scope['scope_manifest']).parent/'dispositions.json'))
    for source in dispositions:
        if source['id'] in by_source:
            source.update(status='released_for_limited_title_ocr_control',split=by_source[source['id']]['reserved_split'],
                annotation_approved_for_task='ocr_recognition',reason='Fixed field ledger reviewed; whole group reserved; global independence unapproved.')
    labels={'schema_version':'ocr-control-labels/2.0','task':'ocr_recognition','scope':SCOPE,'parent_release':PREVIOUS,
        'inherited_labels':deepcopy(verified['old_labels']),'control_source_records':deepcopy(protocol['sources']),
        'control_samples':samples,'unresolved_candidates':unresolved,'absent_fields':absent,'field_ledger':ledger,
        'human_verified':False,'physical_geometry_approved':False,'native_target_approved':False,
        'semantic_family_approved':False,'independent_evaluation':False,'training_started':False}
    registry={'schema_version':'ocr-control-task-splits/2.0','task':'ocr_recognition','scope':SCOPE,
        'parent_registry':deepcopy(old_registry),'scope_protocol':SCOPE_PROTOCOL,'overlay_content_sha256':stable_digest(overlay),
        'assignments':routing['assignments'],'group_commitments':commitments,
        'splits':{k:[r['id'] for r in v] for k,v in rows.items()},'global_split_policy_modified':False,'independent_evaluation':False}
    report={'schema_version':'ocr-control-release/2.0','task':'P14.1b.7c','config':config_pin,'scope':SCOPE,
        'status':'released_for_limited_title_ocr_control','inherited_sample_count':len(rows['train']),
        'control_source_count':len(bindings),'control_sample_count':len(samples),'sample_count':len(all_rows),
        'candidate_count':len(ledger),'unresolved_candidate_count':len(unresolved),'absent_field_count':len(absent),
        'task_split_counts':{k:len(v) for k,v in rows.items()},
        'candidate_decision_counts':dict(Counter(r['decision'] for r in ledger)),
        'control_source_split_counts':dict(Counter(r['reserved_split'] for r in bindings.values())),
        'committed_members_by_split':{split:sum(len(r['member_paths']) for r in commitments if r['split']==split) for split in rows},
        'coallocation_group_count':len(commitments),'committed_member_count':len(reserve_rows),
        'unique_control_image_hash_count':len({r['image_pin']['sha256'] for r in samples}),
        'original_parent_regions_replayed':sum(len(r['parents']) for r in evidence),'original_line_crops_replayed':len(samples)+len(unresolved),
        'inherited_train_rows_images_and_labels_unchanged':True,'full_registry_and_reservations_validated':True,
        'source_modified':False,'global_split_policy_modified':False,'OCR_run':False,'training_started':False,
        'human_verified':False,'semantic_family_approved':False,'independent_evaluation':False,
        'physical_geometry_approved':False,'native_target_approved':False,
        'limitations':['Only title fields of two conservative groups, one per split, are released.',
            'Known uncertain designations and references retain evidence with no target; absent slots remain explicit.',
            'Generic words, drafting style and organization can overlap; unseen external relations remain possible.',
            'Repeated identical line crops within one split are retained per source; sample count is not an independent-group count.',
            'These recognition crops do not approve text detection, dimension semantics, physical geometry or CAD labels.',
            'Validation is for development choices; test inference requires a frozen experiment. No OCR or training was run.']}
    docs={'labels.json':labels,'task-splits.json':registry,'routing.json':routing,'reservation-routing.json':combined,
        'report.json':report,'evidence.json':evidence,'readings.json':review,'dispositions.json':dispositions,
        'overlay.json':overlay,'review-schema.json':ControlReviewV2.model_json_schema()}
    output={name:encoded(value) for name,value in docs.items()};output.update(images);output['train.jsonl']=parent_train
    for split in ('validation','test'):
        output[split+'.jsonl']=''.join(json.dumps(r,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n' for r in rows[split]).encode('utf-8')
    html=['<!doctype html><html lang="ru"><meta charset="utf-8"><title>Контроль строк штампа OCR</title>',
        '<style>body{font:17px system-ui;max-width:1100px;margin:2em auto}img{max-width:100%}section{margin:2em 0}</style>',
        '<h1>OCR: ограниченный контроль строк штампа</h1><p>Две группы, по одной на validation/test. Глобальная независимость не подтверждена. OCR и обучение не запускались.</p>',
        '<p><a href="report.json">Отчёт</a> · <a href="labels.json">Все поля и ограничения</a> · <a href="task-splits.json">Полный реестр</a></p>']
    for split,part in rows.items():
        html.append('<h2>'+escape(split)+'</h2>')
        for row in part:html.append('<section><p>'+escape(row['id'])+' — <b>'+escape(row['text'])+'</b></p><img loading="lazy" src="'+escape(row['image'],quote=True)+'" alt="'+escape(row['text'],quote=True)+'"></section>')
    html.append('<h2>Неопределённые и отсутствующие поля</h2>')
    for row in unresolved+absent:
        html.append('<section><p>'+escape(row['id']+' / '+row['decision']+' — '+row['reason'])+'</p>')
        if 'output_image' in row:html.append('<img loading="lazy" src="'+escape(row['output_image'],quote=True)+'" alt="Без OCR-цели">')
        else:html.append('<details><summary>Область штампа</summary><img loading="lazy" src="'+escape(row['absence_evidence']['output_image'],quote=True)+'" alt="Отсутствующая ячейка"></details>')
        html.append('</section>')
    output['index.html']=(''.join(html)+'</html>\n').encode('utf-8')
    return output,report,rows


def run_control_release_v2(workspace,config_pin,output_dir):
    root=Path(workspace).resolve();output=contained(root,output_dir)
    require(output.parent==root/'artifacts' and not output.exists(),'Use a new direct artifacts directory')
    files,report,_=_documents(root,config_pin,_verify_inputs(root,config_pin),True)
    _verify_inputs(root,config_pin);output.mkdir()
    for name,data in files.items():
        path=contained(output,name);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    manifest={'schema_version':'ocr-control-release-manifest/2.0','task':'ocr_recognition','config':config_pin,
        'files':[pin(output,p) for p in sorted(files)]}
    (output/'manifest.json').write_bytes(encoded(manifest));return report


def verify_ocr_control_v2(workspace,manifest_pin):
    root=Path(workspace).resolve();path,manifest=_package(root,manifest_pin)
    require(set(manifest)=={'schema_version','task','config','files'} and manifest['schema_version']=='ocr-control-release-manifest/2.0'
        and manifest['task']=='ocr_recognition','Wrong control release manifest')
    files,report,rows=_documents(root,manifest['config'],_verify_inputs(root,manifest['config']),True)
    require(manifest['files']==[pin(path.parent,p) for p in sorted(files)],'Control inventory differs from replay')
    require(all((path.parent/p).read_bytes()==data for p,data in files.items()),'Control release differs from original evidence')
    _verify_inputs(root,manifest['config']);return report,rows,path.parent


def load_ocr_control_v2(workspace,manifest_pin,*,split,task='ocr_recognition'):
    require(task=='ocr_recognition' and split in ('train','validation','test'),'Explicit OCR task and split required')
    report,rows,base=verify_ocr_control_v2(workspace,manifest_pin)
    return {'task':task,'split':split,'count':len(rows[split]),'records':[
        {**r,'image_path':str(contained(base,r['image']))} for r in rows[split]],
        'scope':SCOPE,'training_started':False,'independent_evaluation':False,'report':report}
