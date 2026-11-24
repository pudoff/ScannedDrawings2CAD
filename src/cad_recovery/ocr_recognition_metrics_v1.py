"""Separate recognition strata and assembly diagnostics, with frozen normalization."""
from collections import defaultdict
from .ocr_baseline_metrics_v1 import aggregate_scores_v1
from .ocr_recognition_core_v1 import KINDS,selection_score


def score_recognition(rows, predictions):
    mapping={p['id']:p for p in predictions}
    if len(mapping)!=len(predictions) or set(mapping)!={r['id'] for r in rows}:raise ValueError('Incomplete/duplicate recognition result')
    def score(part):
        return aggregate_scores_v1([{'id':r['id'],'source_id':r['source_id'],'reference':r['text'],
             'prediction':mapping[r['id']]['prediction'],'status':mapping[r['id']]['status']} for r in part])
    def compact(part):
        s=score(part);return {k:s[k] for k in ('sample_count','status_counts','raw','normalized')}
    reps={}
    for row in sorted(rows,key=lambda r:r['id']):reps.setdefault(row['image_sha256'],row)
    groups=defaultdict(list)
    for row in rows:groups[row.get('slot_id',row['id'])].append(row)
    fields={mode:sum(all(mapping[r['id']]['status']=='ok' and
                (mapping[r['id']]['prediction']==r['text'] if mode=='raw' else
                 _normalized(mapping[r['id']]['prediction'])==_normalized(r['text'])) for r in part)
                for part in groups.values()) for mode in ('raw','normalized')}
    return {'overall':score(rows),'by_kind':{k:compact([r for r in rows if r['text_kind']==k]) for k in KINDS},
        'unique_images':compact(list(reps.values())),
        'by_source':{k:compact([r for r in rows if r['source_id']==k]) for k in sorted({r['source_id'] for r in rows})},
        'by_group':{k:compact([r for r in rows if r['group_id']==k]) for k in sorted({r['group_id'] for r in rows})},
        'by_rule_pixels':{str(k):compact([r for r in rows if r.get('crop_contains_rule_pixels',False)==k]) for k in (False,True) if any(r.get('crop_contains_rule_pixels',False)==k for r in rows)},
        'full_field':{'count':len(groups),'exact':fields},'selection_score':selection_score(rows,predictions)}


def _normalized(text):
    from .ocr_baseline_metrics_v1 import normalize_text_v1
    return normalize_text_v1(text)


def assembly_diagnostics(rows,pipeline,recognition):
    full={p['id']:p for p in pipeline};rec={p['id']:p for p in recognition};cases=[]
    for row in rows:
        p=full[row['id']];q=rec[row['id']];text=row['text']
        fragments=[p['detections'][i]['raw'] for i in p.get('order',[])]
        cases.append({'id':row['id'],'text_kind':row['text_kind'],'detection_count':len(p.get('detections',[])),
            'pipeline_exact':p['prediction']==text and p['status']=='ok',
            'rec_only_exact':q['prediction']==text and q['status']=='ok',
            'only_injected_join_spaces_disagree':len(fragments)>1 and ''.join(fragments)==text and p['prediction']!=text,
            'scope':'paired observations; detector/orientation/preprocessing effects not individually identified'})
    return {'cases':cases,'pipeline_wrong_rec_only_correct':sum(not c['pipeline_exact'] and c['rec_only_exact'] for c in cases),
            'pipeline_correct_rec_only_wrong':sum(c['pipeline_exact'] and not c['rec_only_exact'] for c in cases),
            'injected_join_space_cases':sum(c['only_injected_join_spaces_disagree'] for c in cases)}
