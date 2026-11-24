"""Partial reviewed-field metrics; unknown content is never a negative label."""
from collections import Counter
from copy import deepcopy
from .dataset_layout_v1 import read
from .native_tiled_metrics_v1 import match
from .joint_region_comparison_v1 import correct_relation
from .ocr_baseline_metrics_v1 import normalize_text_v1, score_text_v1
from .notation_recovery_v1 import bounds

ARMS=('original','rec_only','regrouped','stroke_adjusted','diameter','joint')


def area(b):return max(0,b[2]-b[0])*max(0,b[3]-b[1])


def intersection(a,b):
    return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))


def iou(a,b):
    inter=intersection(a,b);return inter/(area(a)+area(b)-inter) if inter else 0.


def sign_stats(signs,search,threshold):
    positives=[s for s in signs if s['kind']=='diameter']
    negatives=[s for s in signs if s['kind']!='diameter']
    proposals=[p for p in search.get('proposals',[]) if p['score']>=threshold]
    candidates=sorted([(-iou(s['bbox'],p['bbox_px']),s['id'],p['id']) for s in positives for p in proposals
                       if iou(s['bbox'],p['bbox_px'])>=.30])
    hits=set();used=set()
    for _,sid,pid in candidates:
        if sid not in hits and pid not in used:hits.add(sid);used.add(pid)
    false=sum(any(iou(s['bbox'],p['bbox_px'])>=.30 for s in negatives) for p in proposals)
    return {'positive_targets':len(positives),'localized':len(hits),'known_non_diameter_targets':len(negatives),
            'known_false_localizations':false,'threshold_passed':len(proposals),
            'unknown_proposals':sum(p['id'] not in used and not any(iou(s['bbox'],p['bbox_px'])>=.30 for s in negatives) for p in proposals),
            'matched_positive_ids':sorted(hits)}


def text_stats(texts,report):
    lines=report.get('lines',[]);assigned=match(texts,lines);by_id={p['id']:p for p in lines};entries=[]
    false_inserts=0;unknown_inserts=0
    for text in texts:
        p=by_id.get(assigned.get(text['id']));raw=p['raw'] if p else ''
        reference=normalize_text_v1(text['text']);prediction=normalize_text_v1(raw)
        entries.append({'id':text['id'],'assigned_id':p['id'] if p else None,'reference':text['text'],
            'prediction':raw,'normalized':score_text_v1(reference,prediction),'raw':score_text_v1(text['text'],raw)})
        if p and p.get('hypothesis') and '⌀' in prediction and '⌀' not in reference:false_inserts+=1
    known=set(assigned.values())
    for p in lines:
        if p.get('hypothesis') and p['id'] not in known:unknown_inserts+=1
    # Baseline detector merges and newly assembled merges remain separate.
    merged=[]
    for p in lines:
        touched=[t['id'] for t in texts if intersection(t['bbox'],bounds(p['polygon_px']))/area(t['bbox'])>=.5]
        if len(touched)>1:merged.append({'prediction_id':p['id'],'text_ids':sorted(touched)})
    n=sum(e['normalized']['reference_characters'] for e in entries)
    return {'targets':len(texts),'assigned':len(assigned),
            'exact':sum(e['normalized']['exact_match'] for e in entries),
            'edit_distance':sum(e['normalized']['distance'] for e in entries),'characters':n,
            'cer':sum(e['normalized']['distance'] for e in entries)/n if n else None,
            'known_false_diameter_insertions':false_inserts,'unknown_diameter_insertions':unknown_inserts,
            'multi_field_boxes':merged,'entries':entries,'assigned_ids':assigned}


def summarize(contexts,labels,signs,results,threshold,links=None):
    by_id={r['id']:r for r in results.get('cases',[])};rows=[]
    for context in contexts:
        row=by_id.get(context['id'],{});texts=[t for t in labels if t['context_id']==context['id']]
        target_signs=[s for s in signs if s['context_id']==context['id']]
        entry={'id':context['id'],'error':row.get('error','missing_result') if not row.get('arms') else row.get('error'),
               'texts':{},'links':{},'signs':sign_stats(target_signs,row.get('search',{}),threshold),
               'model_calls':row.get('detector_calls',0)+row.get('rec_calls',0),
               'detector_calls':row.get('detector_calls',0),'rec_calls':row.get('rec_calls',0),
               'elapsed_seconds':row.get('elapsed_seconds'),
               'grouped_rows':sum(len(g)>1 for g in row.get('groups',{}).get('groups',[])),
               'stroke_applied':sum(d['stroke'].get('applied',False) for d in row.get('derivatives',[])),
               'template_coverage_complete':row.get('search',{}).get('coverage_complete',False),
               'search_correlations':row.get('search',{}).get('correlations',0)+row.get('joint_search',{}).get('correlations',0)}
        for arm in ARMS:
            metrics=text_stats(texts,row.get('arms',{}).get(arm,{}));entry['texts'][arm]=metrics
            if links is None:continue
            associations=row.get('associations',{}).get(arm,{}).get('associations',[])
            assigned=metrics['assigned_ids'];knownlinks=[l for l in links if l['text_id'] in {t['id'] for t in texts}]
            lrows=[]
            for link in knownlinks:
                aa=[a for a in associations if a['source_line_id']==assigned.get(link['text_id'])]
                proposed=[a for a in aa if a['status']=='proposed']
                def relation(a):
                    return {'kind':a['kind'],'native_anchors':[[x+context['box'][0],y+context['box'][1]] for x,y in a['anchors_px']]}
                correct=[a for a in proposed if correct_relation(link,relation(a))]
                alternatives=sum(correct_relation(link,{'kind':a['kind'],
                    'native_anchors':[[x+context['box'][0],y+context['box'][1]] for x,y in alt.get('anchors_px',[])]})
                    for a in aa for alt in a['alternatives'])
                lrows.append({'id':link['id'],'text_id':link['text_id'],'correct':bool(correct),
                    'wrong_proposed':len(proposed)-len(correct),'correct_alternatives':alternatives,
                    'states':[{k:a[k] for k in ('id','kind','status','reasons','evidence')} for a in aa]})
            negative_ids={assigned[t['id']] for t in texts if not t['is_dimension'] and t['id'] in assigned}
            false_negative_controls=[a['id'] for a in associations if a['status']=='proposed' and a['source_line_id'] in negative_ids]
            labelled_ids=set(assigned.values())
            entry['links'][arm]={'targets':len(knownlinks),'correct':sum(l['correct'] for l in lrows),
                'wrong_proposed':sum(l['wrong_proposed'] for l in lrows),'rows':lrows,
                'non_dimension_false_links':false_negative_controls,
                'unknown_proposed_links':[a['id'] for a in associations if a['status']=='proposed' and a['source_line_id'] not in labelled_ids],
                'status_counts':dict(Counter(a['status'] for a in associations))}
        rows.append(entry)
    totals={}
    for arm in ARMS:
        values=[r['texts'][arm] for r in rows];chars=sum(v['characters'] for v in values);edits=sum(v['edit_distance'] for v in values)
        totals[arm]={key:sum(v[key] for v in values) for key in
            ('targets','assigned','exact','known_false_diameter_insertions','unknown_diameter_insertions')}
        totals[arm].update(characters=chars,edit_distance=edits,cer=edits/chars if chars else None,
                          multi_field_boxes=sum(len(v['multi_field_boxes']) for v in values))
        if links is not None:
            ls=[r['links'][arm] for r in rows]
            totals[arm]['links']={key:sum(v[key] for v in ls) for key in ('targets','correct','wrong_proposed')}
            totals[arm]['links'].update(non_dimension_false_links=sum(len(v['non_dimension_false_links']) for v in ls),
                                       unknown_proposed_links=sum(len(v['unknown_proposed_links']) for v in ls))
    sign_totals={key:sum(r['signs'][key] for r in rows) for key in
        ('positive_targets','localized','known_non_diameter_targets','known_false_localizations','threshold_passed','unknown_proposals')}
    return {'contexts':rows,'text_totals':totals,'sign_totals':sign_totals,
            'completion':len(by_id)==len(contexts) and all(not r['error'] for r in rows),
            'model_calls':sum(r['model_calls'] for r in rows),'unknown_content':'unlabelled fields and links are not negative',
            'independent_evaluation':False,'physical_geometry_approved':False,'production_admitted':False}


def canonical(results):
    """Compare actual outputs, excluding only measured time and output locations."""
    value=deepcopy(results)
    def visit(x):
        if isinstance(x,dict):
            for key in list(x):
                if key in ('elapsed_seconds','image','artifacts','artifact_sha256'):del x[key]
                else:visit(x[key])
        elif isinstance(x,list):
            for v in x:visit(v)
    visit(value);return value
