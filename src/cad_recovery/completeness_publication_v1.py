"""Distinguish a producer's baseline claim from a later gate's decisions."""
from copy import deepcopy
from .observation_completeness_v1 import gate as _gate

def clarify(report):
    result=deepcopy(report)
    if 'baseline_proposals_preserved'in result:
        result['producer_baseline_proposals_preserved_claim']=result.pop('baseline_proposals_preserved')
    changed=any('completeness_original_proposal'in a for a in result['associations'])
    result['producer_decisions_changed_by_completeness_gate']=changed
    result['publication_version']='completeness-publication/1.0'
    return result

def gate(*args,**kwargs):return clarify(_gate(*args,**kwargs))
