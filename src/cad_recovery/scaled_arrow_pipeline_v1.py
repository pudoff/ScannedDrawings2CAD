"""Two independent arms over the c.12 geometry producer; c.13 stays frozen.

This bounded comparison does not claim integration of a new ownership solver
or deployment to a service profile. The full c.13 reference is retained by
the experiment worker for explicit comparison.
"""
from pathlib import Path
from copy import deepcopy
from .arrow_shaft_support_v1 import observe,association_view
from .dimension_association_v3 import associate_dimensions_v3
from .dimension_support_verified_v1 import propose as mask_propose
from .sampled_chain_verification_v1 import propose as sampled_propose
from .dimension_support_v1 import _demote
from .completeness_publication_v1 import gate


def run(source_image,output,*,geometry,ocr,arrows,verification='sampled'):
    if verification not in ('mask','sampled'):raise ValueError('Unknown head verification')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    support=observe(source_image,geometry=geometry,ocr=ocr,arrows=arrows)
    geo,heads=association_view(geometry,arrows,support)
    baseline=associate_dimensions_v3(Path(source_image),output/'association-view',geometry=geo,ocr=ocr,arrows=heads,record_timing=False)
    producer=sampled_propose if verification=='sampled' else mask_propose
    raw=producer(source_image,geometry=geo,ocr=ocr,arrows=heads,baseline=baseline)
    verified={s['arrow_id'] for s in support['spans']};span_ids={s['id'] for s in support['spans']}
    for row in raw['associations']:
        if row['status']=='proposed' and not set(row['arrow_candidate_ids']).issubset(verified):
            _demote(row,'arrow_owned_shaft_support_unverified')
        for c in [row]+row.get('alternatives',[]):
            if span_ids.intersection(c['measured_candidate_ids']):raise ValueError('Annotation span used as part')
    for face in (raw.get('contour_topology') or {}).get('faces',[]):
        for edge in face['boundaries']:
            if span_ids.intersection(edge['geometry_ids']):raise ValueError('Annotation span used as contour')
    if not support['complete']:
        raw['complete']=False
        for row in raw['associations']:_demote(row,'arrow_shaft_support_search_incomplete')
    result=gate(source_image,geometry=geometry,ocr=ocr,arrows=arrows,proposals=raw)
    return {'algorithm_version':'scaled-arrow-pipeline/1.0','verification':verification,'support':support,
        'raw':raw,'gated':result,'original_geometry_preserved':deepcopy(geometry),
        'production_admitted':False,'physical_geometry_approved':False}
