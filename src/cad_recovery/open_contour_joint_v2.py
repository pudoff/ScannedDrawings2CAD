"""Joint V1 with exact observed open-track continuation, no CAD changes."""
from copy import deepcopy
from . import open_contour_joint_v1 as previous
from .open_source_continuation_v1 import observe,POLICY as OBSERVER_POLICY
from .part_admission_v1 import object_sha

POLICY = {**previous.POLICY,'version':'open-contour-joint/2.0','source_continuation':OBSERVER_POLICY}
PARENT_ANALYZE = previous.analyze
PARENT_GUARD = previous.guard_proposal


def analyze(gray,packet,**kwargs):
    before = object_sha(packet['graph']); graph,observer = observe(gray,packet['graph'])
    result = PARENT_ANALYZE(gray,{**packet,'graph':graph,'complete':packet.get('complete',True) and observer['complete']},**kwargs)
    result.update(policy=deepcopy(POLICY),source_continuation=observer)
    if before != object_sha(packet['graph']): raise ValueError('Joint adapter changed source graph')
    return result


def guard_proposal(gray,packet,proposed,**kwargs):
    graph,observer = observe(gray,packet['graph'])
    result = PARENT_GUARD(gray,{**packet,'graph':graph},proposed,**kwargs)
    if not observer['complete']:
        result['mask'][(kwargs['baseline_roles'] == 255) & (kwargs['protected'] == 0)] = 0
        result['complete'] = False; result['reason'] = 'incomplete_source_continuation'
    result['source_continuation'] = observer
    return result
