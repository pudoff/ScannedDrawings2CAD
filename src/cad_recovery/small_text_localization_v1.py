"""Bounded spatial OCR diagnostics; never select a reading by its target text."""
import copy
import cv2
import numpy as np
from .ocr_baseline_metrics_v1 import score_text_v1,normalize_text_v1


def polygon_iou(a,b):
    a=cv2.convexHull(np.asarray(a,dtype=np.float32));b=cv2.convexHull(np.asarray(b,dtype=np.float32))
    area_a=cv2.contourArea(a);area_b=cv2.contourArea(b)
    intersection=float(cv2.intersectConvexConvex(a,b)[0])
    return intersection/max(1e-9,area_a+area_b-intersection)


def merge_rotations(rows):
    """Stable original-first active view, preserving overlapping raw conflicts."""
    active=[];alternatives=[]
    if len(rows)>512:raise ValueError('Observation budget exceeded')
    for row in rows:
        overlap=[x for x in active if polygon_iou(x['polygon_px'],row['polygon_px'])>=.3]
        if overlap:
            alternatives.append({'observation':copy.deepcopy(row),'active_ids':[x['id'] for x in overlap],
                'kind':'duplicate' if all(x['raw']==row['raw'] for x in overlap) else 'conflict'})
        else:active.append(copy.deepcopy(row))
    return active,alternatives


def assemble_fragments(rows):
    """Conservative horizontal chain only, in the OCR request coordinate frame.

    Ambiguous neighbours, slanted/vertical rows and conflicting boxes stay apart.
    No string-based attachment or target-guided sorting.
    """
    if len(rows)>256:raise ValueError('Grouping budget exceeded')
    boxes=[]
    for row in rows:
        p=np.asarray(row['polygon_px'],float);v=p[1]-p[0]
        box=[*p.min(axis=0),*p.max(axis=0)]
        eligible=abs(v[1])<=.18*max(1,abs(v[0])) and abs(v[0])>abs(v[1])
        boxes.append((box,eligible))
    nexts={};previous={}
    for i,(a,ok) in enumerate(boxes):
        if not ok:continue
        for j,(b,bok) in enumerate(boxes):
            if i==j or not bok:continue
            height=max(1,min(a[3]-a[1],b[3]-b[1]))
            overlap=min(a[3],b[3])-max(a[1],b[1]);gap=b[0]-a[2]
            if .5*height<=overlap and 0<=gap<=1.5*height:
                nexts.setdefault(i,[]).append(j);previous.setdefault(j,[]).append(i)
    edges={i:js[0] for i,js in nexts.items() if len(js)==1 and len(previous.get(js[0],[]))==1}
    incoming=set(edges.values());visited=set();groups=[]
    for start in range(len(rows)):
        if start in incoming:continue
        members=[];index=start
        while index not in visited:
            visited.add(index);members.append(index)
            if index not in edges:break
            index=edges[index]
        selected=[rows[i] for i in members]
        if len(selected)==1:groups.append(copy.deepcopy(selected[0]));continue
        points=np.concatenate([np.asarray(x['polygon_px']) for x in selected]);l,t=points.min(axis=0);r,b=points.max(axis=0)
        groups.append({'id':'+'.join(x['id'] for x in selected),'raw':' '.join(x['raw'] for x in selected),
            'polygon_px':[[l,t],[r,t],[r,b],[l,b]],'confidence':None,
            'member_observations':copy.deepcopy(selected),'status':'group_proposal'})
    return groups


def evaluate_regions(targets,rows):
    """IoU assignment uses geometry alone; unmatched/empty cases remain errors.

    Targets are conservatively annotated regions, not complete text segmentation.
    Precision only applies to predictions centred inside readable target regions.
    """
    known=[t for t in targets if t['state']=='readable']
    pairs=sorted([(polygon_iou(t['polygon_px'],r['polygon_px']),i,j)
                  for i,t in enumerate(known) for j,r in enumerate(rows)],reverse=True)
    assigned={};used=set()
    for iou,i,j in pairs:
        if iou>=.3 and i not in assigned and j not in used:
            assigned[i]=(j,iou);used.add(j)
    scoped=set()
    for j,row in enumerate(rows):
        center=np.asarray(row['polygon_px'],float).mean(axis=0)
        if any(cv2.pointPolygonTest(np.asarray(t['polygon_px'],dtype=np.float32),tuple(center),False)>=0 for t in known):scoped.add(j)
    # Include every matched prediction, even if its centre lies outside a small target.
    scoped|=used
    fields=[]
    for i,t in enumerate(known):
        row=rows[assigned[i][0]] if i in assigned else None
        raw=row['raw'] if row else ''
        fields.append({'id':t['id'],'reference':t['transcription'],'prediction':raw,
            'observation_id':row['id'] if row else None,'iou':assigned[i][1] if row else 0,
            'raw_score':score_text_v1(t['transcription'],raw),
            'normalized_score':score_text_v1(normalize_text_v1(t['transcription']),normalize_text_v1(raw))})
    total=sum(f['raw_score']['reference_characters'] for f in fields)
    return {'readable_regions':len(known),'matched_regions':len(assigned),
        'localization_region_recall':len(assigned)/len(known),
        'scoped_prediction_count':len(scoped),'scoped_region_precision':len(assigned)/len(scoped) if scoped else None,
        'unscored_predictions':len(rows)-len(scoped),'all_observations':len(rows),
        'exact':sum(f['raw_score']['exact_match'] for f in fields),
        'normalized_exact':sum(f['normalized_score']['exact_match'] for f in fields),
        'cer':sum(f['raw_score']['distance'] for f in fields)/total,
        'fields':fields,'scope':'readable annotated regions only; not full-page text AP'}
