"""Partial relationship metrics and actual rejection witnesses; no inference."""
from copy import deepcopy
import numpy as np
from .native_tiled_metrics_v1 import match
from .joint_region_comparison_v1 import correct_relation
from .notation_recovery_metrics_v1 import text_stats,intersection,area
from .dimension_support_v1 import parse_composite

ARMS=('unchanged','gate_only','consistency_and_gate')

def _negative(polygon,targets):
    p=np.array(polygon);b=[float(p[:,0].min()),float(p[:,1].min()),float(p[:,0].max()),float(p[:,1].max())]
    return any(intersection(b,t['bbox'])/area(b)>=.5 for t in targets)

def evaluate(contexts,texts,links,payloads,results):
    rows=[]
    for c in contexts:
        payload=payloads[c['id']];result=results.get(c['id']);targets=[t for t in texts if t['context_id']==c['id']]
        known=[l for l in links if l['text_id'] in {t['id'] for t in targets}];entry={'id':c['id'],'arms':{}}
        for arm in ARMS:
            ocr=(result['association_ocr_view'] if result and arm=='consistency_and_gate' else payload['ocr'])
            report=result['arms'][arm] if result else {'associations':[]};assigned=match(targets,ocr['lines'])
            by_id={l['id']:l for l in ocr['lines']};relations=[]
            for link in known:
                line=by_id.get(assigned.get(link['text_id']));pp=[a for a in report['associations']
                    if a['status']=='proposed' and a['source_line_id']==assigned.get(link['text_id'])]
                good=[a for a in pp if correct_relation(link,{'kind':a['kind'],
                    'native_anchors':[[x+c['box'][0],y+c['box'][1]] for x,y in a['anchors_px']]})]
                parsed=parse_composite(line['raw']) if line else None
                typed=bool(parsed and parsed['status']=='parsed' and {'nominal':'linear'}.get(parsed['kind'],parsed['kind'])==link['kind'])
                ready=typed and line['confidence']>=.65 and not line['conflicting_line_ids']
                relations.append({'id':link['id'],'text_id':link['text_id'],'assigned_line_id':line['id'] if line else None,
                    'correct':bool(good),'wrong_proposed':len(pp)-len(good),'typed_ready':bool(ready),
                    'source_conflicts':line['conflicting_line_ids'] if line else [],
                    'states':[{k:a[k] for k in ('id','status','kind','reasons','evidence')}
                              for a in report['associations'] if a['source_line_id']==assigned.get(link['text_id'])]})
            false_roles=unknown=0
            for a in report['associations']:
                if a['status']!='proposed':continue
                line=by_id.get(a['source_line_id'])
                if line and _negative(line['polygon_px'],[t for t in targets if not t['is_dimension']]):false_roles+=1
                elif not any(a['source_line_id']==r['assigned_line_id'] for r in relations):unknown+=1
            entry['arms'][arm]={'known_links':len(known),'correct':sum(r['correct'] for r in relations),
                'wrong_proposed':sum(r['wrong_proposed'] for r in relations),'typed_ready':sum(r['typed_ready'] for r in relations),
                'negative_region_false_links':false_roles,'unknown_proposed_links':unknown,'relations':relations,
                'text':text_stats(targets,ocr),'complete':report.get('complete',False),
                'gate':report.get('completeness_gate')}
        if result:
            entry['verified_groups']=result['consistency']['groups'];entry['new_rec_calls']=result['new_rec_calls']
            entry['rejections']=result['consistency']['rejections']
            # Join source IDs to actually observed raw strings after inference.
            observed={l['id']:l['raw'] for l in payload['ocr']['lines']}
            entry['trace_witnesses']={}
            for key,trace in result['traces'].items():
                events=deepcopy(trace['events'])
                for event in events:
                    event['observed_text']=observed.get(event['text_id'])
                    other=event['witness'].get('other')
                    if isinstance(other,dict):event['other_observed_raw']=observed.get(other.get('id'))
                entry['trace_witnesses'][key]={'events':events,'branch_counts':trace['branch_counts'],
                    'call_inventory_complete':trace['call_inventory_complete'],'unsampled_events':trace['unsampled_events']}
        rows.append(entry)
    summary={}
    for arm in ARMS:
        values=[r['arms'][arm] for r in rows]
        summary[arm]={k:sum(v[k] for v in values) for k in ('known_links','correct','wrong_proposed','typed_ready',
            'negative_region_false_links','unknown_proposed_links')}
        summary[arm]['incomplete_contexts']=sum(not v['complete'] for v in values)
    return {'contexts':rows,'summary':summary,'independent_evaluation':False,'physical_geometry_approved':False}

def evaluate_native(contexts,texts,links,reports):
    rows=[]
    for c in contexts:
        result=reports[c['id']];original=result['gated_observations']
        targets=deepcopy([t for t in texts if t['context_id']==c['id']])
        for t in targets:t['bbox']=[t['bbox'][0]+c['box'][0],t['bbox'][1]+c['box'][1],t['bbox'][2]+c['box'][0],t['bbox'][3]+c['box'][1]]
        assigned=match(targets,original['texts']);known=[l for l in links if l['text_id']in {t['id'] for t in targets}]
        outcomes=[]
        for link in known:
            pp=[a for a in original['associations'] if a['status']=='proposed' and a['source_text_id']==assigned.get(link['text_id'])]
            good=[a for a in pp if correct_relation(link,{'kind':a['local_evidence']['kind'],'native_anchors':a['anchors_px']})]
            outcomes.append({'id':link['id'],'correct':bool(good),'wrong_proposed':len(pp)-len(good)})
        complete=result['merged_assessment']['complete']
        rows.append({'id':c['id'],'known_links':len(known),'correct':sum(o['correct'] for o in outcomes),
            'wrong_proposed':sum(o['wrong_proposed'] for o in outcomes),'relations':outcomes,
            'windows':len(result['windows']),'geometry_complete_windows':sum(w['assessment']['stages']['geometry']['state']=='complete' for w in result['windows']),
            'all_stages_complete_windows':sum(w['assessment']['complete'] for w in result['windows']),
            'deferred_retries':original['deferred_retries'],'complete':complete,
            'text':text_stats(targets,{'lines':original['texts']})})
    return {'contexts':rows,'known_links':sum(r['known_links'] for r in rows),'correct':sum(r['correct'] for r in rows),
            'wrong_proposed':sum(r['wrong_proposed'] for r in rows),'complete_contexts':sum(r['complete'] for r in rows)}

def canonical_native(report):
    # Only actual semantic observations are compared; changing evidence/score or
    # completeness cannot be hidden by dropping measured time.
    keys=('texts','geometry','arrows','associations','aliases','text_conflicts','seam_review','deferred_retries',
          'pixel_coverage_complete','ocr_windows_complete','geometry_search_incomplete')
    return {k:report[k] for k in keys}
