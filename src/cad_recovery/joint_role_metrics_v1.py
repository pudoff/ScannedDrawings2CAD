"""Partial metrics include false predicted classes even without positive support."""
import numpy as np
from .stroke_roles_v1 import CLASSES


def score(truth,prediction,negative=None):
    valid=(truth>=1)&(truth<=9);predicted=np.where(prediction==255,10,prediction)
    if truth.shape!=prediction.shape or np.any((predicted<0)|(predicted>10)):raise ValueError('Aligned bounded role maps required')
    matrix=np.bincount((truth[valid].astype(int)*11+predicted[valid]).ravel(),minlength=110).reshape(10,11)
    selected=(truth==255)&(negative>0) if negative is not None else np.zeros_like(valid)
    false=np.bincount(prediction[selected].astype(int),minlength=256)
    return from_counts(matrix,false[1:10])


def from_counts(matrix,negative_counts):
    matrix=np.asarray(matrix,dtype=np.int64);negative_counts=np.asarray(negative_counts,dtype=np.int64);rows=[]
    for cls in range(1,10):
        support=int(matrix[cls].sum());tp=int(matrix[cls,cls]);fp=int(matrix[:,cls].sum()-tp+negative_counts[cls-1]);fn=support-tp
        if not support and not fp:continue
        rows.append({'role':CLASSES[cls],'support':support,'tp':tp,'fp':fp,'fn':fn,
            'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/support if support else None,
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None})
    total=int(matrix.sum());assigned=int(matrix[:,1:10].sum());correct=int(sum(matrix[c,c] for c in range(1,10)))
    return {'confusion':matrix.tolist(),'roles':rows,'macro_f1':float(np.mean([r['f1'] for r in rows])) if rows else None,
        'weighted_f1':sum(r['f1']*r['support'] for r in rows)/total if total else None,
        'known_pixels':total,'known_correct':correct,'unknown_known_pixels':int(matrix[:,10].sum()),
        'coverage':assigned/total if total else None,'selective_accuracy':correct/assigned if assigned else None,
        'accuracy_with_abstentions':correct/total if total else None,'negative_false_assignments_by_role':negative_counts.tolist(),
        'false_role_assignments_on_negative_ink_px':int(negative_counts.sum()),'unknown_ground_truth_ignored':True,
        'absent_GT_classes_with_false_predictions_in_macro':True,'full_page_precision':None}
