"""Replay audits beyond strict typed localization, without model calls."""
from .joint_region_comparison_v1 import intersection,area

def neighbour_captures(crops,targets,factor):
    """A coarse CV abstention is still a geometric routing window.

    This includes view_or_detail, omitted by the strict typed-view report.
    The predeclared >=10% coverage criterion is unchanged.
    """
    result=[]
    views=[t for t in targets if t['role'] in ('view','detail')]
    for c in crops:
        if c['role'] not in ('view','detail','view_or_detail'):continue
        box=[v*factor for v in c['bbox']]
        overlaps=sorted([(intersection(box,t['bbox']),t) for t in views],key=lambda v:(-v[0],v[1]['id']))
        result.append({'crop_id':c['id'],'role':c['role'],
            'neighbour_ids':[t['id'] for overlap,t in overlaps[1:] if overlap/area(t['bbox'])>=.1]})
    return result
