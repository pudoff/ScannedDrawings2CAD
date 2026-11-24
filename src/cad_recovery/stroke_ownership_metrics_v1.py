"""Partial-label role metrics with explicit known interference, never unknown-as-background."""
import numpy as np

from .stroke_roles_v1 import CLASSES
from .stroke_role_metrics_v1 import role_scores


def score_roles(truth,predicted,negative=None):
    result=role_scores(truth,predicted)
    negative=np.zeros_like(truth,bool) if negative is None else (negative>0)
    # Only explicitly constructed negative ink counts here; unassigned real ink stays unscored.
    negative=negative & (truth==255)
    counts=np.bincount(predicted[negative].astype(int),minlength=256)
    details=[]
    for old in result['roles']:
        cls=CLASSES.index(old['role']);row={**old};row['negative_false_assignments']=int(counts[cls]);row['fp']+=int(counts[cls])
        tp,fp,fn=row['tp'],row['fp'],row['fn']
        row['precision']=tp/(tp+fp) if tp+fp else None;row['f1']=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None
        details.append(row)
    result['roles']=details;result['macro_f1']=float(np.mean([d['f1'] for d in details])) if details else None
    result['known_negative_ink_px']=int(negative.sum());result['false_role_assignments_on_negative_ink_px']=int(counts[1:10].sum())
    result['negative_false_assignments_by_role']={CLASSES[c]:int(counts[c]) for c in range(1,10)}
    result['unassigned_real_ink_is_negative']=False
    return result


def ink_ledger(gray,roles,foreground):
    ink=foreground(gray)>0;assigned=(roles!=255)&ink;unknown=(roles==255)&ink
    return {'input_ink_px':int(ink.sum()),'assigned_ink_px':int(assigned.sum()),'unknown_ink_px':int(unknown.sum()),
        'exact_partition':bool(np.array_equal(assigned|unknown,ink) and not np.any(assigned&unknown)),
        'source_pixels_erased':0,'physical_geometry_approved':False}
