"""Observe fresh heads on raster-derived CAD primitives, then verify role chains."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from . import cad_object_assembly_v5
from .cad_line_export_v7 import assemble
from .cad_object_merge_index_v1 import merge_curves
from .joint_dimension_roles_v1 import propose
from .primitive_role_guard_v1 import apply as guard_roles
from .scaled_arrow_detection_v2 import detect
from .stamp_role_projection_v1 import guard_export_layers


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')


def analyze(folder, packet, gray, roles, *, frame_rules=(), ocr=None):
    folder = Path(folder)
    folder.mkdir()
    source = folder/'source.png'
    Image.fromarray(gray).save(source)
    common = {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'processed_size_px': list(gray.shape[::-1])}
    with patch.object(cad_object_assembly_v5, '_merge_curves', merge_curves):
        assembly = assemble(gray, packet['geometry'], roles=roles, protected=packet.get('protected'))
    assembly = guard_export_layers(guard_roles(assembly, gray, packet['geometry'], roles), frame_rules)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    radii = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
    candidates = []
    for obj in assembly['objects']:
        if obj['kind'] not in ('line', 'arc', 'circle'):
            continue
        row = deepcopy(obj)
        points = np.asarray(row['support_points'], int)
        row['width_px'] = max(1., float(np.median(radii[points[:, 1], points[:, 0]])*2-1))
        row['uncertainty_px'] = max(.5, float(row.get('fit_max_px', 0.)))
        candidates.append(row)
    geometry = {**common, 'candidates': candidates, 'complete': assembly['search_complete'],
                'ownership_intervals': [{'id': obj['id'], 'support_points': obj['support_points']}
                                        for obj in assembly['objects']],
                'source': 'native_primitives_from_same_raster',
                'unresolved_primitive_count': len(assembly['objects'])-len(candidates)}
    ocr = {**common, 'lines': [], 'notations': [], 'availability': 'not_run'} if ocr is None else deepcopy(ocr)
    arrows = detect(source, geometry=geometry, text_polygons=[row['polygon_px'] for row in ocr['lines']])
    result = propose(source, geometry=geometry, ocr=ocr, arrows=arrows, roles=roles)
    # A partial observation search cannot license an early subset of roles.
    limited = arrows['analysis']['search_limited'] or not assembly['search_complete']
    if limited:
        result['roles'] = roles.copy()
        result['report'].update(complete=False, publication_withheld='upstream_observation_search_incomplete')
    result['report'].update(source=common, fresh_head_detection=True,
        head_candidates=len(arrows['candidates']), typed_candidates=len(candidates),
        unresolved_primitives=geometry['unresolved_primitive_count'],
        arrow_search_complete=not arrows['analysis']['search_limited'],
        primitive_search_complete=assembly['search_complete'],
        published_role_changes=int(np.count_nonzero(result['roles'] != roles)),
        input_roles_preserved=bool(np.all(result['roles'][roles != 255] == roles[roles != 255])),
        production_changed=False, physical_geometry_approved=False)
    for name, value in (('geometry.json', geometry), ('arrows.json', arrows),
                        ('ocr.json', ocr), ('report.json', result['report'])):
        write(folder/name, value)
    Image.fromarray(result['roles']).save(folder/'roles.png')
    return result
