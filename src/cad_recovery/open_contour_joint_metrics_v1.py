"""Scorer-only role-bit metrics; never imported by the recognition component."""
import numpy as np


def compare(raw, guarded, role_bits, *, gray, baseline, full, protected):
    shape = gray.shape
    if any(a.shape != shape for a in (raw,guarded,role_bits,baseline,full,protected)):
        raise ValueError('Aligned original reference and prediction frames required')
    if role_bits.dtype != np.uint16: raise ValueError('Explicit multi-owned role bits required')
    before,after = raw > 0,guarded > 0
    if np.any(after & ~before): raise ValueError('Guarded output is not a negative subset')
    part = (role_bits & 2) > 0; known_wrong = (role_bits > 0) & ~part
    paper = (role_bits == 0) & (gray == 255); unknown = (role_bits == 0) & ~paper
    removed = before & ~after
    shared = (role_bits > 0) & ((role_bits & (role_bits-1)) != 0)
    def count(mask):
        return {'pixels':int(mask.sum()),'correct_PART':int((mask & part).sum()),
                'known_wrong':int((mask & known_wrong).sum()),'paper':int((mask & paper).sum()),
                'unknown':int((mask & unknown).sum()),'shared_PART':int((mask & part & shared).sum()),
                'exclusive_wrong_by_role':{str(r):int((mask & (role_bits == (1 << r))).sum()) for r in range(2,10)},
                'shared_nonPART':int((mask & known_wrong & shared).sum())}
    return {'raw':count(before),'guarded':count(after),'removed':count(removed),
            'baseline_changed':int(((full != baseline) & (baseline != 255)).sum()),
            'protected_changed':int(((full != baseline) & (protected > 0)).sum())}
