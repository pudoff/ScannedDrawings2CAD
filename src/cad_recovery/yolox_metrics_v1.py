"""Saved-tensor detector evaluation; all labels remain on the evaluator side."""
from collections import Counter
import math

import numpy as np

from .family_catalog_v2 import require
from .yolox_protocol_v1 import CLASSES,POLICY,geometry,map_edges,nms,operating_matches
from .semantic_region_experiment_v1 import iou,coarse

COARSE=['view_or_detail','title_block','table','notes']


def decode(raw,size):
    require(isinstance(raw,np.ndarray) and raw.dtype==np.float32 and raw.ndim==2 and raw.shape[1]==10
            and 0<len(raw)<=8400 and np.isfinite(raw).all(),'Invalid saved detector tensor')
    require(np.all((raw[:,4:]>=0)&(raw[:,4:]<=1)),'Invalid detector probabilities')
    spec=geometry(size);classes=np.argmax(raw[:,5:],axis=1);scores=raw[:,4]*raw[np.arange(len(raw)),5+classes]
    eligible=[i for i in range(len(raw)) if scores[i]>=POLICY['ap_min_score']]
    def box(i):
        x,y,w,h=map(float,raw[i,:4]);return [x-w/2,y-h/2,x+w/2,y+h/2]
    ordered=sorted(eligible,key=lambda i:(-float(scores[i]),int(classes[i]),box(i),i))[:1000]
    rows=[];discarded=0
    for index in ordered:
        b=box(index)
        if b[0]>=b[2] or b[1]>=b[3]:discarded+=1;continue
        mapped=map_edges(b,spec['model_to_edges'],size)
        if mapped[0]>=mapped[2] or mapped[1]>=mapped[3]:discarded+=1;continue
        rows.append({'id':f'anchor-{index:04d}','role':CLASSES[int(classes[index])],
            'score':float(scores[index]),'box':mapped,'raw_index':index})
    return {'operating':nms(rows),'ap':nms(rows,confidence=.001,maximum=100),
        'audit':{'raw_cells':len(raw),'above_ap_score':len(eligible),'pre_nms_selected':len(ordered),
                 'invalid_or_padding_only':discarded,'valid_pre_nms':len(rows)}}


def native_predictions(rows,lineage):
    result=[];discarded=0
    for row in rows:
        box=map_edges(row['box'],lineage['overview_to_native_edges'],lineage['native_size_px'])
        if box[0]>=box[2] or box[1]>=box[3]:discarded+=1;continue
        result.append({**row,'box':box})
    return result,discarded


def ratio(n,d):return n/d if d else 0.


def detection_metrics(target_pages,predictions,*,typed):
    require(set(predictions)=={t['id'] for t in target_pages},'Missing or invented scored pages')
    roles=CLASSES if typed else COARSE;totals={r:Counter() for r in roles};pages=[];matched_geometry=set()
    for target in target_pages:
        rows=predictions[target['id']]
        require(len(rows)<=24 and len({r['id'] for r in rows})==len(rows),'Prediction count/ID budget')
        for row in rows:
            require(row['role'] in [*CLASSES,'view_or_detail','unknown'],'Invented semantic role')
            map_edges(row['box'],np.eye(3))
        matches=operating_matches(target['regions'],rows,typed=typed)
        for role in roles:
            actual=sum((r['role'] if typed else coarse(r['role']))==role for r in target['regions'])
            predicted=sum((r['role'] if typed else coarse(r['role']))==role for r in rows)
            tp=sum(r['matched'] and (r['role'] if typed else coarse(r['role']))==role for r in matches)
            totals[role].update(tp=tp,fp=predicted-tp,fn=actual-tp,targets=actual,predictions=predicted)
        if not typed:matched_geometry.update(r['label_id'] for r in matches if r['matched'] and r['role'] in ('view','detail'))
        pages.append({'id':target['id'],'matches':matches,
            'abstained_predictions':sum((r['role'] if typed else coarse(r['role'])) not in roles for r in rows),
            'prediction_count':len(rows)})
    per_class={}
    for role,count in totals.items():
        p=ratio(count['tp'],count['tp']+count['fp']);r=ratio(count['tp'],count['tp']+count['fn'])
        per_class[role]={**count,'precision':p,'recall':r,'F1':ratio(2*p*r,p+r)}
    tp=sum(c['tp'] for c in totals.values());fp=sum(c['fp'] for c in totals.values());fn=sum(c['fn'] for c in totals.values())
    p=ratio(tp,tp+fp);r=ratio(tp,tp+fn)
    matched=[m['iou'] for page in pages for m in page['matches'] if m['matched']]
    return {'tp':tp,'fp':fp,'fn':fn,'precision':p,'recall':r,'micro_F1':ratio(2*p*r,p+r),
        'macro_F1':sum(c['F1'] for c in per_class.values())/len(roles),'per_class':per_class,
        'class_recall':{k:v['recall'] for k,v in per_class.items()},'pages':pages,
        'geometric_matched_ids':sorted(matched_geometry),'geometric_TP':len(matched_geometry),
        'mean_matched_IoU':sum(matched)/len(matched) if matched else 0.,
        'abstained_predictions':sum(p['abstained_predictions'] for p in pages)}


def average_precision(target_pages,predictions):
    require(set(predictions)=={t['id'] for t in target_pages},'AP page inventory differs')
    per_class={}
    for role in CLASSES:
        targets={t['id']:[r for r in t['regions'] if r['role']==role] for t in target_pages}
        count=sum(map(len,targets.values()));aps=[]
        rows=sorted([(p['score'],identifier,p) for identifier,values in predictions.items() for p in values if p['role']==role],
                    key=lambda r:(-r[0],r[1],r[2]['box'],r[2]['id']))
        require(all(math.isfinite(s) and 0<=s<=1 for s,_,_ in rows),'Invalid AP score')
        for threshold in [.5+i*.05 for i in range(10)]:
            used={p:set() for p in targets};tp=[]
            for score,page,prediction in rows:
                candidates=[(iou(t['bbox'],prediction['box']),i) for i,t in enumerate(targets[page]) if i not in used[page]]
                best=max(candidates,key=lambda pair:(pair[0],-pair[1]),default=(0.,-1))
                good=best[0]+1e-12>=threshold and best[1]>=0
                if good:used[page].add(best[1])
                tp.append(int(good))
            cum=np.cumsum(tp);precision=cum/np.arange(1,len(tp)+1) if tp else np.array([])
            recall=cum/count if count else np.array([])
            ap=sum(max(precision[recall>=r],default=0.) for r in np.linspace(0,1,101))/101 if count else None
            aps.append(float(ap) if ap is not None else None)
        per_class[role]={'targets':count,'predictions':len(rows),'AP50':aps[0],
            'AP50_95':sum(aps)/10 if count else None,'AP_by_IoU':aps}
    supported=[v for v in per_class.values() if v['targets']]
    return {'per_class':per_class,'mAP50':sum(v['AP50'] for v in supported)/len(supported) if supported else None,
        'mAP50_95':sum(v['AP50_95'] for v in supported)/len(supported) if supported else None}


def context_coverage(target_pages,predictions,typed_metrics):
    matches={m['label_id']:m for p in typed_metrics['pages'] for m in p['matches']};rows=[]
    for page in target_pages:
        by_id={p['id']:p for p in predictions[page['id']]}
        for link in page['detail_links']:
            for target_id,field in [(link['detail_id'],'caption_box'),(link['parent_view_id'],'marker_box')]:
                match=matches[target_id];coverage=0.
                if match['matched']:
                    a=link[field];b=by_id[match['candidate_id']]['box']
                    coverage=(max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1])))/((a[2]-a[0])*(a[3]-a[1]))
                rows.append({'page':page['id'],'detail_id':link['detail_id'],'target_id':target_id,
                    'field':field,'matched':match['matched'],'coverage':coverage,'passed':coverage>=.99})
    return {'contexts':rows,'passed':bool(rows) and all(r['passed'] for r in rows)}


def decision_input(typed,coarse_metrics,baseline_geometry):
    return {'precision':typed['precision'],'recall':typed['recall'],'macro_F1':typed['macro_F1'],
        'class_recall':typed['class_recall'],'coarse_F1':coarse_metrics['macro_F1'],
        'geometric_TP':coarse_metrics['geometric_TP'],
        'baseline_geometric_targets_newly_missed':len(set(baseline_geometry)-set(coarse_metrics['geometric_matched_ids']))}
