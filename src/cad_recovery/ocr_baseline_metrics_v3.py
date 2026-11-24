"""Separate control-split metrics, fixed field coverage and duplicate accounting."""
from collections import Counter, defaultdict
from copy import deepcopy

from .family_catalog_v2 import require, stable_digest
from .ocr_baseline_metrics_v1 import aggregate_scores_v1

SPLITS=('validation','test')


def build_control_metadata_v3(labels,registry,rows):
    """Bind every accepted control and every unscored field before inference."""
    samples=labels['control_samples'];ledger=labels['field_ledger']
    require(len({r['id'] for r in ledger})==len(ledger),'Duplicate field slot')
    require(all(r['decision'] in ('accepted_for_ocr_transcription','unresolved','absent') for r in ledger),'Unknown field decision')
    accepted={r['id']:r for r in ledger if r['decision']=='accepted_for_ocr_transcription'}
    require(len({r['id'] for r in samples})==len(samples) and set(accepted)=={r['id'] for r in samples},'Control samples differ from accepted field ledger')
    require(all(stable_digest(r)==stable_digest(accepted[r['id']]) for r in samples),'Accepted sample details differ from field ledger')
    require(len({r['id'] for r in rows})==len(rows) and {r['id'] for r in rows}==set(accepted),'Incomplete control sample inventory')
    assignments={r['id']:r for r in registry['assignments']}
    bindings={r['id']:r for r in labels['control_source_records']}
    require(len(bindings)==len(labels['control_source_records']),'Duplicate source binding')
    strata=[]
    for row in sorted(rows,key=lambda r:(SPLITS.index(r['split']),r['id'])):
        sample=accepted[row['id']];binding=bindings[sample['source_id']];assignment=assignments[row['id']]
        require(row['split']==sample['reserved_split']==binding['reserved_split']==assignment['split']
            and row['split'] in SPLITS and row['source_id']==sample['source_id']
            and row['text']==sample['text'] and row['image_sha256']==sample['image_pin']['sha256']
            and row['source_sha256']==binding['input_sha256'],'Control target, image or split differs')
        strata.append({'id':row['id'],'source_id':row['source_id'],'split':row['split'],'role':sample['field'],
            'coallocation_group_id':assignment['coallocation_group_id'],'source_path':binding['input_path'],
            'image_sha256':row['image_sha256']})
    require({r['source_id'] for r in strata}==set(bindings),'Some reserved source lacks an accepted line')
    coverage={};by_source={}
    for source,binding in bindings.items():
        fields=[r for r in ledger if r['source_id']==source]
        require(len(fields)==3 and {r['field'] for r in fields}=={'designation','part_name','adjacent_reference'},'Incomplete source field ledger')
        require(all(r['reserved_split']==binding['reserved_split'] for r in fields),'Field crosses source reservation')
        require(all(r['text'] is None for r in fields if r['decision']!='accepted_for_ocr_transcription'),'Unscored field contains a target')
        by_source[source]={'split':binding['reserved_split'],'field_slots':len(fields),
            'decisions':dict(sorted(Counter(r['decision'] for r in fields).items()))}
    require(sum(r['field_slots'] for r in by_source.values())==len(ledger),'Unknown source field slots')
    for split in SPLITS:
        fields=[r for r in ledger if r['reserved_split']==split];part=[r for r in strata if r['split']==split]
        require(part,'Every requested split needs accepted samples')
        coverage[split]={'source_count':sum(r['reserved_split']==split for r in bindings.values()),
            'field_slots':len(fields),'decisions':dict(sorted(Counter(r['decision'] for r in fields).items())),
            'by_role':{role:dict(sorted(Counter(r['decision'] for r in fields if r['field']==role).items()))
                       for role in sorted({r['field'] for r in fields})},
            'sample_count':len(part),'unique_image_count':len({r['image_sha256'] for r in part}),
            'coallocation_group_count':len({r['coallocation_group_id'] for r in part})}
    images=defaultdict(list);owners={}
    for row in strata:
        images[row['image_sha256']].append(row['id'])
        for kind,value in [('image',row['image_sha256']),('group',row['coallocation_group_id']),('source',row['source_id'])]:
            require((kind,value) not in owners or owners[kind,value]==row['split'],'Cross-split control leakage')
            owners[kind,value]=row['split']
    representatives=sorted(min(ids) for ids in images.values())
    return {'schema_version':'ocr-control-evaluation-metadata/3.0','strata':strata,
        'coverage_by_split':coverage,'coverage_by_source':dict(sorted(by_source.items())),
        'field_slot_count':len(ledger),'decision_counts':dict(sorted(Counter(r['decision'] for r in ledger).items())),
        'duplicates':[{'image_sha256':key,'ids':sorted(ids)} for key,ids in sorted(images.items()) if len(ids)>1],
        'unique_image_representative_ids':representatives,
        'primary_policy':'all accepted lines retained; uncertain and absent slots reported as coverage, not scored',
        'unique_image_policy':'secondary diagnostic: smallest frozen sample ID for each identical image hash',
        'independent_evaluation':False}


def score_control_v3(predictions,metadata):
    require(isinstance(predictions,list) and len({r['id'] for r in predictions})==len(predictions),'Duplicate prediction')
    strata={r['id']:r for r in metadata['strata']}
    require({r['id'] for r in predictions}==set(strata),'Missing control prediction')
    for row in predictions:
        meta=strata[row['id']]
        require(row['source_id']==meta['source_id'] and row['split']==meta['split'],'Prediction crosses frozen metadata')
    def score(part):
        return aggregate_scores_v1([{k:r[k] for k in ('id','source_id','reference','prediction','status')} for r in part])
    result={}
    unique=set(metadata['unique_image_representative_ids'])
    for split in SPLITS:
        part=[r for r in predictions if r['split']==split]
        result[split]={'overall':score(part),
            'by_role':{role:score([r for r in part if strata[r['id']]['role']==role])
                for role in sorted({strata[r['id']]['role'] for r in part})},
            'by_group':{gid:score([r for r in part if strata[r['id']]['coallocation_group_id']==gid])
                for gid in sorted({strata[r['id']]['coallocation_group_id'] for r in part})},
            'unique_image_diagnostic':score([r for r in part if r['id'] in unique])}
    return {'schema_version':'ocr-control-evaluation-metrics/3.0','by_split':result,
        'coverage_by_split':deepcopy(metadata['coverage_by_split']),
        'coverage_by_source':deepcopy(metadata['coverage_by_source']),
        'duplicates':deepcopy(metadata['duplicates']),'combined_validation_test_score':None,
        'metadata_content_sha256':stable_digest(metadata),'independent_evaluation':False}
