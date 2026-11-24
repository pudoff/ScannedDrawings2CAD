"""Add fresh original-resolution observations to the frozen raster stage."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import dimension_role_stage_v1 as previous
from .joint_dimension_roles_v2 import propose
from .source_arrow_observation_v1 import observe, digest


def analyze(folder, packet, gray, roles, *, original_source, working_to_source,
            frame_rules=(), ocr=None):
    folder = Path(folder)
    source_results = {}

    def with_native_heads(image, *, geometry, ocr, arrows, roles):
        heads = observe(original_source, geometry=geometry, geometry_image=image,
                        geometry_to_source=working_to_source,
                        text_polygons=[row['polygon_px'] for row in ocr['lines']])
        result = propose(image, geometry=geometry, ocr=ocr, arrows=arrows, roles=roles,
            original_source=original_source, working_to_source=working_to_source, source_heads=heads)
        source_results.update(heads=heads, projected=result['projected_arrows'], bridge=result['native_bridge'])
        result['report']['native_observation'] = {
            'original_source_sha256': digest(original_source),
            'working_to_source': np.asarray(working_to_source).tolist(),
            'search_complete': heads['search_complete'],
            'source_candidates': len(heads['candidates']),
            'source_verified_candidates': len(heads['verified_candidates']),
            'source_diagnostic_verified_candidates': len(heads['diagnostic_verified_candidates']),
            'bridge_supported_candidates': result['native_bridge']['report']['supported_heads'],
            'bridge_published_support_candidates': result['native_bridge']['report']['published_support_heads'],
            'bridge_qualification_complete': result['native_bridge']['report']['qualification_complete'],
            'source_limits': deepcopy(heads['analysis']['limits'])}
        return result

    with patch.object(previous, 'propose', with_native_heads):
        result = previous.analyze(folder, packet, gray, roles, frame_rules=frame_rules, ocr=ocr)
    # Original stage documents remain truthful: arrows.json is its working
    # detector output. Native and combined observations have separate files.
    for name, value in (('original-heads.json', source_results['heads']),
                        ('projected-arrows.json', source_results['projected']),
                        ('native-head-bridge.json', source_results['bridge'])):
        previous.write(folder/name, value)
    return result
