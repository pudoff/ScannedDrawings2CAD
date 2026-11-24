"""Experimental annotation support with unchanged contour/ownership gates."""
from copy import deepcopy
from pathlib import Path

from .arrow_shaft_support_v1 import observe, association_view
from .dimension_association_v3 import associate_dimensions_v3
from .dimension_support_verified_v1 import propose
from .dimension_support_v1 import _demote
from .completeness_publication_v1 import gate


def run(source_image, output, *, geometry, ocr, arrows):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    support = observe(source_image, geometry=geometry, ocr=ocr, arrows=arrows)
    geo_view, arrow_view = association_view(geometry, arrows, support)
    base = associate_dimensions_v3(Path(source_image), output / 'association-view',
                                   geometry=geo_view, ocr=ocr, arrows=arrow_view, record_timing=False)
    raw = propose(source_image, geometry=geo_view, ocr=ocr, arrows=arrow_view, baseline=base)
    span_ids = {s['id'] for s in support['spans']}
    supported_arrows = {s['arrow_id'] for s in support['spans']}
    # The frozen topology excludes all geometry IDs bound to arrow shafts,
    # including these annotation spans. Also enforce the publication boundary.
    for face in (raw.get('contour_topology') or {}).get('faces', []):
        for boundary in face['boundaries']:
            if span_ids.intersection(boundary['geometry_ids']):
                raise ValueError('Annotation support leaked into part contour')
    for row in raw['associations']:
        choices = [row] + row.get('alternatives', [])
        if any(span_ids.intersection(c.get('measured_candidate_ids', [])) for c in choices):
            raise ValueError('Annotation support leaked into measured geometry')
        used = span_ids.intersection(row['dimension_line_candidate_ids'])
        row.setdefault('evidence', {})['annotation_support_ids'] = sorted(used)
        if row['status'] == 'proposed' and not set(row['arrow_candidate_ids']).issubset(supported_arrows):
            _demote(row, 'arrow_owned_shaft_support_unverified')
    if not support['complete']:
        raw['complete'] = False
        for row in raw['associations']:
            _demote(row, 'arrow_shaft_support_search_incomplete')
    # Assess the ORIGINAL producer completeness, never a fabricated complete
    # report for the added support view. Preserve upstream cap information.
    gated = gate(source_image, geometry=geometry, ocr=ocr, arrows=arrows, proposals=raw)
    return {'algorithm_version': 'arrow-shaft-pipeline/1.0', 'support': support,
            'raw': raw, 'gated': gated, 'association_view_geometry': geo_view,
            'original_geometry_preserved': deepcopy(geometry),
            'production_admitted': False, 'physical_geometry_approved': False}
