from copy import deepcopy
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from cad_recovery.robust_topology_v1 import build, Budget, Unresolved, relation, numeric_envelope
from cad_recovery.robust_graph_qa_v1 import compare_graphs, guard, match_nodes, verified_provenance


def box(name, x, y, size):
    return {'id': name, 'path_id': name, 'kind': 'polyline', 'closed': True,
            'points_px': [[x, y], [x+size, y], [x+size, y+size], [x, y+size]]}


def test_large_translation_preserves_nested_areas_and_depth():
    for offset in (0., 1e6, 9e6):
        g = build([box('outer', offset+.123, offset+.234, 10), box('inner', offset+.133, offset+.244, 9.98)])
        assert g['complete'] and g['nesting_complete']
        assert sorted(f['depth'] for f in g['faces']) == [0, 1]
        assert sorted(f['area_px2'] for f in g['faces']) == pytest.approx([9.98**2, 100], abs=1e-7)


def test_large_coordinate_disjoint_boxes_do_not_become_nested_or_touching():
    geometry = [box('a', 9000000.123, 9000000.234, 1), box('b', 9000001.12303, 9000000.234, 1)]
    g = build(geometry)
    assert g['complete'] and g['nesting_complete'] and g['components'] == 2
    assert [f['depth'] for f in g['faces']] == [0, 0]


def test_contact_remains_unknown_even_for_equal_area_siblings():
    g = build([box('a', 0, 0, 10), box('b', 10, 10, 10)])
    assert g['complete'] and not g['nesting_complete']
    assert not compare_graphs(g, g)['accepted']


def test_connected_nested_non_simple_face_is_not_silently_decomposed():
    geometry = [box('a', 0, 0, 20), box('b', 5, 5, 5),
                {'id': 'bridge', 'kind': 'line', 'points_px': [[0, 0], [5, 5]]}]
    g = build(geometry)
    assert not g['nesting_complete'] and g['unresolved_face_reasons'].get('non_simple_boundary_walk')
    assert not guard(geometry, geometry)['decision']['accepted']


def test_vertex_budget_stays_at_512_without_implicit_coarsening():
    points = [[30*math.cos(t), 30*math.sin(t)] for t in np.linspace(0, 2*math.pi, 521)[:-1]]
    geometry = [{'id': 'p', 'kind': 'polyline', 'points_px': points, 'closed': True}]
    g = build(geometry)
    assert g['complete'] and not g['nesting_complete']
    assert 'Flattened contour vertex budget exceeded' in g['unresolved_face_reasons']


def test_topology_predicate_budget_abstains_without_spending_extra_work():
    b = Budget()
    b.spend(250000)
    with pytest.raises(Unresolved, match='budget'):
        b.spend(1)
    assert b.used == 250000


def test_circle_area_is_analytic_and_nesting_is_certified():
    geometry = [{'id': name, 'path_id': name, 'kind': 'circle', 'center_px': [1000000.123, 1000000.234], 'radius_px': r}
                for name, r in [('outer', 5.), ('hole', 1.25)]]
    g = build(geometry)
    assert g['nesting_complete'] and sorted(f['depth'] for f in g['faces']) == [0, 1]
    assert sorted(f['area_px2'] for f in g['faces']) == pytest.approx([1.25**2*math.pi, 25*math.pi], abs=1e-7)


def test_actual_owner_reconnection_is_rejected_with_equal_global_counts():
    source = [{'id': 'a', 'path_id': 'a', 'kind': 'line', 'points_px': [[0, 0], [10, 0]]},
              {'id': 'b', 'path_id': 'b', 'kind': 'line', 'points_px': [[10, 0], [20, 0]]},
              {'id': 'c', 'path_id': 'c', 'kind': 'line', 'points_px': [[50, 0], [60, 0]]}]
    changed = deepcopy(source)
    changed[1]['path_id'], changed[2]['path_id'] = 'c', 'b'
    result = guard(source, changed)
    assert not result['decision']['accepted'] and result['selected'] == source


def test_near_nodes_match_only_equal_degree_and_owners():
    before = {'structural_nodes': [
        {'id': 'a', 'point_px': [0, 0], 'degree': 3, 'owners': ['a', 'stem']},
        {'id': 'b', 'point_px': [1, 0], 'degree': 3, 'owners': ['b', 'stem']}]}
    after = {'structural_nodes': [
        {'id': 'b2', 'point_px': [1.4, 0], 'degree': 3, 'owners': ['b', 'stem']},
        {'id': 'a2', 'point_px': [.4, 0], 'degree': 3, 'owners': ['a', 'stem']}]}
    matches = match_nodes(before, after)['junction']
    assert not matches['ambiguous'] and not matches['missed']
    assert {(p['before_id'], p['after_id']) for p in matches['matches']} == {('a', 'a2'), ('b', 'b2')}


def test_ambiguous_compatible_matching_is_explicitly_rejected():
    before = {'structural_nodes': [{'id': str(i), 'point_px': [i, 0], 'degree': 1, 'owners': ['same']} for i in (0, 1)]}
    after = {'structural_nodes': [{'id': str(i), 'point_px': [i+.4, 0], 'degree': 1, 'owners': ['same']} for i in (0, 1)]}
    assert match_nodes(before, after)['endpoint']['ambiguous']


def test_exact_coordinate_identity_does_not_depend_on_node_order():
    before = {'structural_nodes': [{'id': str(i), 'point_px': [i, 0], 'degree': 1, 'owners': ['same']} for i in (0, 1)]}
    after = {'structural_nodes': list(reversed(before['structural_nodes']))}
    assert not match_nodes(before, after)['endpoint']['ambiguous']


def provenance_fixture():
    source = [{'id': 'source', 'path_id': 'p', 'kind': 'polyline', 'closed': False, 'points_px': [[0, 0], [20, 0]]}]
    candidate = [{'id': 'new', 'path_id': 'p', 'kind': 'line', 'points_px': [[0, 0], [20, 0]],
                  'source_observation_id': 'source', 'source_index_interval': [0, 1]}]
    evidence = [{'observed_native_px': p, 'estimated_native_px': p, 'incidences': [['p', end]]}
                for p, end in [([0, 0], 0), ([20, 0], -1)]]
    return source, candidate, evidence


def test_source_provenance_is_checked_against_actual_endpoints_not_just_IDs():
    source, candidate, evidence = provenance_fixture()
    valid = verified_provenance(build(source), build(candidate), source, candidate, evidence)
    assert valid['complete']
    candidate[0]['points_px'][0] = [.01, 0]
    invalid = verified_provenance(build(source), build(candidate), source, candidate, evidence)
    assert not invalid['complete'] and invalid['reason'] == 'modeled_endpoint_contradicts_evidence'


def test_forged_source_ID_or_missing_interval_is_not_authority():
    source, candidate, evidence = provenance_fixture()
    candidate[0]['source_observation_id'] = 'forged'
    assert not verified_provenance(build(source), build(candidate), source, candidate, evidence)['complete']


def test_invalid_provenance_cannot_fall_back_to_permissive_matching():
    source, candidate, _ = provenance_fixture()
    result = guard(source, candidate, provenance={'complete': False, 'mapping': {}, 'reason': 'bad'})
    assert not result['decision']['accepted'] and 'invalid_source_provenance' in result['decision']['reasons']


def test_input_coordinates_and_unknown_source_are_unchanged():
    source = [box('b', 9000000.123, 9000000.234, 10)]
    original = deepcopy(source)
    build(source)
    result = guard(source, source)
    assert source == original and result['selected'] == original
    result['selected'][0]['points_px'][0][0] = 0
    assert source == original


def test_registered_train_controls_have_the_declared_decisions():
    path = Path(__file__).resolve().parents[1]/'scripts/prepare_topology_robust_controls_v1.py'
    spec = importlib.util.spec_from_file_location('robust_controls', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for case in module.cases():
        if case['split'] != 'train':
            continue
        graph = build(case['source'])
        assert graph['nesting_complete'] == case['expected']['nesting_complete'], case['id']
        assert guard(case['source'], case['proposal'])['decision']['accepted'] == case['expected']['guard_accepted'], case['id']
