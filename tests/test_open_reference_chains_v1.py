from copy import deepcopy
import json
import cv2
import numpy as np
import pytest
from cad_recovery import open_reference_chains_v1 as model


def fixture(body_width=9, crossing=False, disconnected=False):
    gray = np.full((180, 210), 255, np.uint8); items = []; tracks = []
    def path(identity, a, b, width):
        points = np.rint(np.linspace(a, b, int(max(abs(np.array(b)-a)))+1)).astype(int)
        cv2.polylines(gray, [points], False, 0, width)
        items.append({'id': identity, 'points_px': points.astype(float).tolist(), 'width_px': float(width)})
    path('shaft', [50, 40], [150, 40], 3)
    for x, name in [(50, 'left'), (150, 'right')]:
        if crossing:
            path(name+'-upper', [x, 40], [x, 80], 3); path(name+'-lower', [x, 80], [x, 110], 3)
            ids = [name+'-upper', name+'-lower']
        else:
            path(name, [x, 40], [x, 110], 3); ids = [name]
        tracks.append({'id': name, 'points': [[x, 40], [x, 110]], 'u': [0., 1.], 'length': 70., 'width': 3., 'interval_ids': ids})
    for name, a, b in [('body-left', [20, 110], [50, 110]), ('body-right', [150, 110], [180, 110])]:
        path(name, a, b, body_width)
    if disconnected:
        path('body-mid-a', [50, 110], [90, 110], body_width)
        path('body-mid-b', [110, 110], [150, 110], body_width)
    else: path('body-mid', [50, 110], [150, 110], body_width)
    if crossing:
        path('cross-left', [20, 80], [50, 80], body_width)
        path('cross-mid', [50, 80], [150, 80], body_width)
        path('cross-right', [150, 80], [180, 80], body_width)
    tracks.insert(0, {'id': 'shaft', 'points': [[50, 40], [150, 40]], 'u': [1., 0.], 'length': 100., 'width': 3., 'interval_ids': ['shaft']})
    heads = []
    for identity, x, sign in [('h-left', 50, -1), ('h-right', 150, 1)]:
        polygon = [[x, 40], [x-sign*20, 34], [x-sign*20, 46]]
        cv2.fillConvexPoly(gray, np.array(polygon, np.int32), 0)
        heads.append({'id': identity, 'tip_px': [x, 40], 'polygon_px': polygon, 'direction': [sign, 0.], 'uncertainty_px': 1.})
    previous = {'proposals': items, 'tracks': tracks, 'boundaries': {'boundaries': []}, 'hatch_bundles': [], 'search_incomplete': False}
    packet = {'pixel_roles': np.full(gray.shape, 255, np.uint8), 'dimension_chains': [], 'search_incomplete': False}
    return gray, previous, packet, heads


def test_open_source_component_recovers_two_local_references():
    result = model.extend(*fixture())
    assert len(result['open_reference_chains']) == 1
    chain = result['open_reference_chains'][0]
    assert chain['feet_px'] == [[50., 110.], [150., 110.]]
    assert chain['boundary_id'] is None and chain['evidence'] == 'two_source_connected_open_references'


def test_observed_thin_continuation_skips_incidental_crossing():
    result = model.extend(*fixture(crossing=True))
    assert len(result['open_reference_chains']) == 1
    assert result['open_reference_chains'][0]['feet_px'] == [[50., 110.], [150., 110.]]


def test_equal_pen_ambiguous_reference_is_not_forced():
    assert not model.extend(*fixture(body_width=3))['open_reference_chains']


def test_long_unsplit_reference_is_not_lost_due_to_interval_representation():
    gray = np.full((150, 1024), 255, np.uint8); cv2.line(gray, (20, 100), (990, 100), 0, 9)
    previous = {'proposals': [{'id': 'long', 'points_px': [[float(x), 100.] for x in range(20, 991)], 'width_px': 9.}],
                'tracks': [], 'hatch_bundles': [], 'boundaries': {'boundaries': []}}
    search = model.LocalReferences(gray, previous, [])
    proof = search.support((20., 100.), np.array([1., 0.]), 1, 3., set())
    assert proof and proof['reference_ids'] == ['long'] and proof['support_length_px'] == 970.


def test_separate_body_components_cannot_be_connected_by_the_dimension():
    assert not model.extend(*fixture(disconnected=True))['open_reference_chains']


def test_graph_does_not_cross_an_intentional_white_extension_gap():
    gray, previous, packet, heads = fixture(); old = previous['proposals'][1]
    previous['proposals'][1] = {**old, 'points_px': [[50., y] for y in range(40, 65)]}
    previous['tracks'][1].update(points=[[50, 40], [50, 64]], length=24.)
    gray[66:80, 46:55] = 255
    assert not model.extend(gray, previous, packet, heads)['open_reference_chains']


def test_frame_reference_owners_are_excluded():
    gray, previous, packet, heads = fixture()
    previous['boundaries']['boundaries'] = [{'frame_hypothesis': True, 'owners': ['body-left', 'body-mid', 'body-right']}]
    assert not model.extend(gray, previous, packet, heads)['open_reference_chains']


def test_unheaded_parallel_hatch_bundle_cannot_supply_part_references():
    gray, previous, packet, heads = fixture()
    previous['hatch_bundles'] = [{'interval_ids': ['body-left', 'body-mid', 'body-right']}]
    assert not model.extend(gray, previous, packet, heads)['open_reference_chains']


def test_bundle_with_observed_dimension_is_not_treated_as_hatch_truth():
    gray, previous, packet, heads = fixture()
    previous['hatch_bundles'] = [{'interval_ids': ['shaft', 'body-left', 'body-mid', 'body-right']}]
    assert len(model.extend(gray, previous, packet, heads)['open_reference_chains']) == 1


def test_existing_proposed_grid_roles_do_not_override_source_evidence():
    gray, previous, packet, heads = fixture(); packet['pixel_roles'][:] = 8
    result = model.extend(gray, previous, packet, heads)
    assert len(result['open_reference_chains']) == 1 and np.all(result['pixel_roles'] == 8)


def test_all_observations_and_role_pixels_are_immutable():
    gray, previous, packet, heads = fixture(); snapshot = deepcopy([previous, heads]); original = gray.copy(); labels = packet['pixel_roles'].copy()
    result = model.extend(gray, previous, packet, heads)
    assert [previous, heads] == snapshot and np.array_equal(gray, original)
    assert np.array_equal(result['pixel_roles'], labels) and np.array_equal(packet['pixel_roles'], labels)
    assert result['source_pixels_erased'] == 0 and not result['physical_geometry_approved']


def test_existing_chain_is_preserved_without_duplicate_addition():
    args = fixture(); first = model.extend(*args)['open_reference_chains'][0]
    args[2]['dimension_chains'] = [first]
    result = model.extend(*args)
    assert result['dimension_chains'] == [first] and not result['open_reference_chains']


def test_single_or_same_direction_heads_do_not_create_a_chain():
    args = fixture(); args[-1].pop()
    assert not model.extend(*args)['open_reference_chains']
    args = fixture(); args[-1][1]['direction'] = [-1., 0.]
    assert not model.extend(*args)['open_reference_chains']


def test_work_budget_failure_exposes_incompleteness_and_keeps_baseline(monkeypatch):
    monkeypatch.setitem(model.POLICY, 'maximum_edge_visits', 1)
    result = model.extend(*fixture())
    assert result['search_incomplete'] and result['open_reference_audit']['limits']
    assert not result['open_reference_chains'] and not result['dimension_chains']


def test_source_point_budget_abstains(monkeypatch):
    monkeypatch.setitem(model.POLICY, 'maximum_source_points', 5)
    result = model.extend(*fixture())
    assert result['search_incomplete'] and not result['open_reference_chains']


def test_duplicate_head_and_nonfinite_observation_are_rejected():
    args = fixture(); args[-1][1]['id'] = args[-1][0]['id']
    with pytest.raises(ValueError): model.extend(*args)
    args = fixture(); args[-1][0]['direction'] = [float('nan'), 0.]
    with pytest.raises(ValueError): model.extend(*args)


def test_invalid_source_width_and_coordinate_are_rejected():
    args = fixture(); args[1]['proposals'][0]['width_px'] = 0.
    with pytest.raises(ValueError): model.extend(*args)
    args = fixture(); args[1]['proposals'][0]['points_px'][0] = [-5, 40]
    with pytest.raises(ValueError): model.extend(*args)


def test_saved_proof_is_json_and_references_only_actual_source_intervals():
    args = fixture(); result = model.extend(*args); json.dumps({k: v for k, v in result.items() if k != 'pixel_roles'}, allow_nan=False)
    chain = result['open_reference_chains'][0]; ids = {p['id'] for p in args[1]['proposals']}
    proof = chain['local_reference_evidence']
    assert set(proof['body_route']['interval_ids']) <= ids
    for foot in proof['references']:
        assert set(foot['extension_route_ids']+foot['reference_ids']) <= ids
