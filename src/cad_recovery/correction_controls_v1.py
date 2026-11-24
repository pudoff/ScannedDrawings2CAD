"""Analytically specified correction fixture, independent of solver output."""
from copy import deepcopy
import hashlib
from pathlib import Path

from .graph_v2 import read_graph_v2
from .graph_correction_v1 import canonical_sha256


def circle_control(source_path='examples/correction-control-source-v1.json'):
    source = Path(source_path)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    evidence = {'source_ref': source.as_posix(), 'method': 'derived', 'status': 'verified',
                'review_ref': 'analytic control premises; not customer ground truth',
                'uncertainty': {'bound': 0, 'unit': 'model_mm', 'basis': 'exact analytic construction'}}

    def entity(key, **kwargs):
        return {'id': key, 'view_id': 'main', 'status': 'accepted', 'evidence': deepcopy(evidence), **kwargs}

    graph = read_graph_v2({'schema_version': '0.2',
        'source': {'path': source.as_posix(), 'sha256': source_hash, 'page_index': 0, 'method': 'analytic correction fixture'},
        'page': {'width_mm': 210, 'height_mm': 297, 'kind': 'vector', 'size_source': 'user'},
        'views': [{'id': 'main', 'evidence': evidence, 'coordinate_space': 'model_mm',
                   'model_scale_xy': [1, 1], 'physical_scale_confirmed': True,
                   'scale_evidence_refs': ['analytic_mm_X', 'analytic_mm_Y'],
                   'view_to_sheet': [[1, 0, 40], [0, 1, 40], [0, 0, 1]]}],
        'primitives': [entity('outer', kind='circle', center=[20, 20], radius=10),
                       entity('hole', kind='circle', center=[20, 20], radius=2)],
        'dimensions': [entity('hole-diameter', kind='diameter', nominal=4, raw_text='original synthetic diameter 4',
                              points=[[20, 20], [22, 20]], placement=[32, 25], measured_entity_ids=['hole'])],
        'contours': [entity('outer-contour', edges=[{'primitive_id': 'outer'}], closed=True, role='outer'),
                     entity('hole-contour', edges=[{'primitive_id': 'hole'}], closed=True, role='hole', parent_id='outer-contour')],
        'hatches': [entity('section', outer_contour_id='outer-contour', hole_contour_ids=['hole-contour'])],
        'constraints': [entity('outer-center', kind='fixed', entity_ids=['outer'],
                               parameters={'point_refs': [{'entity_id': 'outer', 'field': 'center'}], 'target_mm': [20, 20]}),
                        entity('hole-center', kind='fixed', entity_ids=['hole'],
                               parameters={'point_refs': [{'entity_id': 'hole', 'field': 'center'}], 'target_mm': [20, 20]}),
                        entity('outer-radius', kind='radius', entity_ids=['outer'], value=10),
                        entity('hole-size', kind='diameter', entity_ids=['hole'], value=4)],
        'provenance': {'scope': 'analytical_control', 'customer_label': False}})
    request = {'parent_graph_sha256': canonical_sha256(graph), 'view_id': 'main', 'constraint_id': 'hole-size',
               'expected_old_value': 4, 'new_value': 6, 'parameter_evidence_ref': source.as_posix()+'#premises',
               'max_point_displacement_mm': 0.01, 'max_radius_change_mm': 2}
    bindings = {'dimensions': [{'dimension_id': 'hole-diameter', 'nominal_policy': 'derive_from_reviewed_geometry',
                               'points': [{'entity_id': 'hole', 'field': 'center'},
                                          {'entity_id': 'hole', 'field': 'circle_point', 'angle_deg': 0}]}]}
    return graph, request, bindings
