"""Separate research pipeline retaining c.12 decisions and upstream proof."""
from pathlib import Path
from copy import deepcopy
from .arrow_shaft_pipeline_v1 import run as previous_run
from .arrow_shaft_support_v1 import association_view
from .text_chain_ownership_v1 import resolve
from .completeness_publication_v1 import gate


def run(source_image, output, *, geometry, ocr, arrows):
    output = Path(output);output.mkdir(parents=True, exist_ok=False)
    previous = previous_run(source_image, output/'previous', geometry=geometry, ocr=ocr, arrows=arrows)
    geo_view, arrow_view = association_view(geometry, arrows, previous['support'])
    raw = resolve(source_image, geometry=geo_view, ocr=ocr, arrows=arrow_view,
                  support=previous['support'], baseline=previous['raw'])
    spans = {s['id'] for s in previous['support']['spans']}
    for row in raw['associations']:
        for choice in [row]+row.get('alternatives', []):
            if spans.intersection(choice['measured_candidate_ids']):
                raise ValueError('Annotation span used as measured part geometry')
    for face in (raw['text_chain_ownership'].get('contour_topology') or {}).get('faces', []):
        for edge in face['boundaries']:
            if spans.intersection(edge['geometry_ids']):
                raise ValueError('Annotation span used as part contour')
    published = gate(source_image, geometry=geometry, ocr=ocr, arrows=arrows, proposals=raw)
    return {'algorithm_version':'text-chain-pipeline/1.0','previous':previous,'raw':raw,'gated':published,
            'original_ocr_preserved':deepcopy(ocr),'production_admitted':False,'physical_geometry_approved':False}
