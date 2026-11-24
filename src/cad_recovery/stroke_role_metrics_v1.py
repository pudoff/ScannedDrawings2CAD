"""Partial role/arrow scoring; reference arrays never enter proposal functions."""
import cv2
import numpy as np

from .stroke_roles_v1 import CLASSES
from .stroke_curve_control_v1 import arrow_scores


def synthetic_truth(case,gray,text_mask):
    truth=np.full(gray.shape,255,np.uint8);conflicts=np.zeros_like(gray,bool)
    for g in case['labels']['geometry']:
        mask=np.zeros_like(gray);width=g['stroke_width_px']
        if g['kind']=='line':cv2.line(mask,tuple(g['points'][0]),tuple(g['points'][1]),255,width)
        elif g['kind']=='circle':cv2.circle(mask,tuple(g['center']),g['radius'],255,width)
        else:cv2.ellipse(mask,tuple(g['center']),(g['radius'],g['radius']),0,g['start_angle_deg'],g['start_angle_deg']+g['sweep_angle_deg'],255,width)
        selected=(mask>0)&(gray<240);cls=CLASSES.index(g['role'])
        conflicts|=selected&(truth!=255)&(truth!=cls);truth[selected & ~conflicts]=cls
    for a in case['labels']['arrows']:
        mask=np.zeros_like(gray);cv2.fillConvexPoly(mask,np.array(a['triangle'],np.int32),255)
        conflicts|=mask>0
    truth[(text_mask>0)&(gray<240)]=6
    truth[conflicts]=255
    return truth


def to_native(array,size):
    if list(array.shape[::-1])==size:return array
    return cv2.resize(array,tuple(size),interpolation=cv2.INTER_NEAREST)


def role_scores(truth,prediction):
    valid=(truth>=1)&(truth<=9)
    predicted=np.where(prediction==255,10,prediction)
    matrix=np.bincount((truth[valid].astype(int)*11+predicted[valid]).ravel(),minlength=110).reshape(10,11)
    details=[]
    for cls in range(1,10):
        support=int(matrix[cls].sum())
        if not support:continue
        tp=int(matrix[cls,cls]);fp=int(matrix[:,cls].sum()-tp);fn=support-tp
        details.append({'role':CLASSES[cls],'support':support,'tp':tp,'fp':fp,'fn':fn,
            'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/support,
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None})
    total=int(matrix.sum());assigned=int(matrix[:,:10].sum());correct=int(sum(matrix[c,c] for c in range(1,10)))
    return {'confusion':matrix.tolist(),'unknown_column':10,'roles':details,
        'macro_f1':float(np.mean([r['f1'] for r in details])) if details else None,
        'known_pixels':total,'coverage':assigned/total if total else None,
        'selective_accuracy':correct/assigned if assigned else None,'accuracy_with_abstentions':correct/total if total else None,
        'unknown_ground_truth_ignored':True,'full_page_precision':None}


def mask_effect(truth,gray,mask):
    selected=(mask>0)&(gray<240)
    return {'known_text_suppressed_px':int((selected&(truth==6)).sum()),
        'known_nontext_suppressed_px':int((selected&(truth>=1)&(truth<=9)&(truth!=6)).sum()),
        'unassigned_or_unknown_ink_suppressed_px':int((selected&(truth==255)).sum()),
        'source_pixels_erased':0,'mask_is_derivative_only':True}


def head_scores(refs,negatives,candidates,matrix,*,dense=False,noise_mask=None):
    rows=arrow_scores(refs,candidates,matrix)
    matched={r['candidate_id'] for r in rows if r['matched']}
    transformed=[]
    for a in candidates:
        p=np.asarray(matrix)@[*a['tip_px'],1]
        transformed.append((a['id'],p[:2]))
    negative_rows=[]
    for box in negatives:
        x,y,r,b=box['box'];ids=[identifier for identifier,p in transformed if x<=p[0]<r and y<=p[1]<b]
        negative_rows.append({'id':box['id'],'candidate_ids':ids,'false_tips':len(ids)})
    noise=[]
    if noise_mask is not None:
        for identifier,p in transformed:
            x,y=np.rint(p).astype(int)
            if 0<=x<noise_mask.shape[1] and 0<=y<noise_mask.shape[0] and noise_mask[y,x]>0:noise.append(identifier)
    return {'assigned':len(refs),'matched':len(matched),'recall':len(matched)/len(refs) if refs else None,
        'dense_precision':len(matched)/len(candidates) if dense and candidates else None,
        'dense_false_positives':len(candidates)-len(matched) if dense else None,
        'candidate_count':len(candidates),'targets':rows,'negative_regions':negative_rows,
        'false_tips_in_negative_regions':sum(r['false_tips'] for r in negative_rows),
        'false_tips_on_reviewed_noise':noise,'real_full_precision':None}
