"""Partial native relation and evidence metrics, never used by inference."""
import math
import numpy as np
from .native_tiled_metrics_v1 import match
from .joint_region_comparison_v1 import correct_relation
from .notation_recovery_metrics_v1 import text_stats,area,intersection
from .dimension_association_v3 import _on_curve

ARMS=('baseline','orientation','composite','leaders','flanks','all')

def segment_coverage(a,b,geometry,tolerance):
    a,b=np.array(a,dtype=float),np.array(b,dtype=float);length=float(np.linalg.norm(b-a));u=(b-a)/length
    intervals=[]
    for g in geometry:
        if g['kind']!='line':continue
        points=np.array(g['points_px']);delta=points[1]-points[0];norm=float(np.linalg.norm(delta))
        if not norm or abs(float(np.dot(delta/norm,u)))<math.cos(math.radians(12)):continue
        cross=np.abs((points[:,0]-a[0])*u[1]-(points[:,1]-a[1])*u[0])
        if cross.max()>tolerance:continue
        x,y=sorted((points-a)@u);intervals.append((max(0,float(x)),min(length,float(y))))
    covered=0.;end=0.
    for x,y in sorted(intervals):
        if y>x:covered+=max(0.,y-max(end,x));end=max(end,y)
    return covered/length


def geometry_evidence(link,context,payload):
    origin=np.array(context['box'][:2]);geometry=payload['geometry']['candidates'];tol=link['tolerance_px']
    path=np.array(link['polyline'])-origin
    cover=[segment_coverage(a,b,geometry,tol) for a,b in zip(path,path[1:])]
    result={'dimension_or_leader_segment_coverage':cover,'geometry_candidate_count':len(geometry)}
    if link['scope']=='paired_feature_segments':
        result['feature_segment_coverage']=[segment_coverage(np.array(a)-origin,np.array(b)-origin,geometry,tol) for a,b in link['features']]
    elif link['scope']=='observed_circle':
        x,y,r=link['circle'];center=np.array([x,y])-origin
        result['circle_candidates_within_tolerance']=sum(g['kind']=='circle' and
            np.linalg.norm(np.array(g['center_px'])-center)<=tol and abs(g['radius_px']-r)<=tol for g in geometry)
    else:
        contact=np.array(link['contact'])-origin
        result['curve_candidates_at_known_contact']=sum(_on_curve(contact,{
            **g,'center':np.array(g['center_px']),'radius':g['radius_px']},tol)
            for g in geometry if g['kind'] in {'arc','circle'})
    return result


def evaluate(contexts,texts,links,payloads,runs):
    rows=[]
    for context in contexts:
        payload=payloads[context['id']];outputs=runs[context['id']]
        labels=[t for t in texts if t['context_id']==context['id']]
        assigned=match(labels,payload['ocr']['lines']);line_by={l['id']:l for l in payload['ocr']['lines']}
        knownlinks=[l for l in links if l['text_id'] in {t['id'] for t in labels}]
        entry={'id':context['id'],'ocr_unchanged':text_stats(labels,payload['ocr']),
               'geometry_evidence':{l['id']:geometry_evidence(l,context,payload) for l in knownlinks},'arms':{}}
        for arm in ARMS:
            report=payload['baseline'] if arm=='baseline' else outputs[arm]
            assocs=report['associations'];link_rows=[]
            for link in knownlinks:
                matches=[a for a in assocs if a['source_line_id']==assigned.get(link['text_id'])]
                def relation(a):return {'kind':a['kind'],'native_anchors':[[x+context['box'][0],y+context['box'][1]] for x,y in a['anchors_px']]}
                pp=[a for a in matches if a['status']=='proposed'];correct=[a for a in pp if correct_relation(link,relation(a))]
                if arm=='baseline':
                    notation=next((n['parsed'] for n in payload['ocr']['notations'] if n['source_line_id']==assigned.get(link['text_id'])),None)
                else:notation=next((h['parsed'] for h in report['history'] if h['source_line_id']==assigned.get(link['text_id'])),None)
                syntax=(notation is not None and notation['status']=='parsed' and {'nominal':'linear'}.get(notation['kind'],notation['kind'])==link['kind'])
                source=line_by.get(assigned.get(link['text_id']))
                ready=syntax and source['confidence']>=.65 and not source.get('conflicting_line_ids')
                link_rows.append({'id':link['id'],'correct':bool(correct),'wrong_proposed':len(pp)-len(correct),
                    'typed_syntax_correct':bool(syntax),'typed_confident_observation_ready':bool(ready),
                    'association_states':[{k:a[k] for k in ('id','kind','status','reasons','evidence')} for a in matches]})
            negatives=[t for t in labels if not t['is_dimension']];false_roles=[];unknown=[]
            for a in [a for a in assocs if a['status']=='proposed']:
                line=line_by.get(a['source_line_id'])
                if not line:continue
                p=np.array(line['polygon_px']);b=[p[:,0].min(),p[:,1].min(),p[:,0].max(),p[:,1].max()]
                # A fragmented identifier is still a negative dimension region;
                # full-field matching is not required to notice this false link.
                if any(intersection(b,t['bbox'])/area(b)>=.5 for t in negatives):false_roles.append(a['id'])
                elif not any(a['source_line_id']==assigned.get(l['text_id']) for l in knownlinks):unknown.append(a['id'])
            entry['arms'][arm]={'known_links':len(knownlinks),'correct':sum(l['correct'] for l in link_rows),
                'wrong_proposed':sum(l['wrong_proposed'] for l in link_rows),'negative_region_false_links':false_roles,
                'unknown_proposed_links':unknown,'typed_syntax_correct':sum(l['typed_syntax_correct'] for l in link_rows),
                'typed_ready':sum(l['typed_confident_observation_ready'] for l in link_rows),
                'complete':report.get('complete',True),'links':link_rows,
                'new_notation_count':len(report.get('new_notations',[]))}
        rows.append(entry)
    summary={}
    for arm in ARMS:
        values=[r['arms'][arm] for r in rows]
        summary[arm]={k:sum(v[k] for v in values) for k in ('known_links','correct','wrong_proposed','typed_syntax_correct','typed_ready','new_notation_count')}
        summary[arm].update(negative_region_false_links=sum(len(v['negative_region_false_links']) for v in values),
                            unknown_proposed_links=sum(len(v['unknown_proposed_links']) for v in values),
                            incomplete_contexts=sum(not v['complete'] for v in values))
    return {'contexts':rows,'summary':summary,'independent_evaluation':False,'physical_geometry_approved':False,
            'negative_scope':'reviewed non-dimension regions including fragments; unlabelled content is unknown'}
