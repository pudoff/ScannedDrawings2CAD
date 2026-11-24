from copy import deepcopy
from pathlib import Path
import builtins
import cv2
import numpy as np
import pytest
from cad_recovery.part_admission_v1 import analyze, provenance, POLICY
from cad_recovery.source_region_roles_v1 import analyze as propose
from cad_recovery.grid_role_revision_v1 import cv_authority
from test_source_region_context_v1 import part_fixture


def fixture():
    gray, graph, raw = part_fixture()
    graph['dimension_chains'][0]['head_ids'] = ['head-a', 'head-b']
    graph['dimension_chains'][0]['status'] = 'geometric_hypothesis_only'
    authority = cv_authority(gray, raw, {'path': 'fixture/raw.png', 'sha256': 'a'*64})
    produced = propose(gray, graph, raw, authority=authority)
    previous, candidate, owners = produced['previous'], produced['guarded'], produced['owners']['guarded']
    canonical = deepcopy(graph['dimension_chains'])
    return gray, graph, raw, previous, candidate, owners, canonical


def run(data, protected=None, supplied=True):
    gray, graph, raw, previous, candidate, owners, canonical = data
    authority = cv_authority(gray, raw, {'path': 'fixture/raw.png', 'sha256': 'a'*64}, protected=protected)
    proof = provenance(*data, {'path': 'fixture/candidate.png', 'sha256': 'b'*64}) if supplied else None
    return analyze(*data, authority=authority, evidence=proof)


def test_no_part_reference_reverts_only_new_part_and_keeps_existing_roles():
    data = fixture()
    gray, graph, raw, previous, candidate, owners, canonical = data
    delta = candidate != previous
    assert delta.any()
    result = run(data)
    assert np.array_equal(result['no_part'], previous)
    assert np.any(result['canonical'] == 1) and result['complete']


def test_canonical_membership_alone_is_not_a_semantic_certificate():
    data = fixture()
    data[-1][0]['feet_px'][0][0] += 3
    result = run(data)
    assert np.array_equal(result['canonical'], result['no_part'])
    assert not result['audit']['canonical_context']['canonical_is_semantic_authority']


def test_known_hatching_is_preserved_even_with_a_matching_canonical_boundary():
    data = list(fixture())
    proposed = data[4] != data[3]
    data[3][proposed] = 9
    result = run(data)
    assert np.all(result['canonical'][proposed] == 9)


def repeated_fixture():
    data = list(fixture())
    gray, graph, raw, previous, candidate, owners, canonical = data
    body = graph['proposals'][0]
    body['points_px'] = [[x, 75] for x in range(60, 236)]
    graph['tracks'] = [{'id': 'body-row', 'interval_ids': ['body']}]
    for n, y in enumerate((100, 125, 150)):
        cv2.line(gray, (60, y), (235, y), 0, 7)
        identity = 'row-'+str(n)
        graph['proposals'].append({'id': identity, 'points_px': [[x, y] for x in range(60, 236)]})
        graph['tracks'].append({'id': identity, 'interval_ids': [identity]})
    return data


def test_equal_width_local_parallel_rows_veto_part_even_without_a_hatch_bundle():
    result = run(repeated_fixture())
    assert result['complete'] and np.array_equal(result['canonical'], result['no_part'])
    assert any(d['parallel_repeat_veto'] for d in result['decisions'])


def test_adjacent_bounded_faces_do_not_certify_an_outer_part_boundary():
    data = fixture()
    boundaries = data[1]['boundaries']['boundaries']
    boundaries[0]['orientation'] = 'bounded_face'
    other = deepcopy(boundaries[0]); other['id'] = 'adjacent'; boundaries.append(other)
    result = run(data)
    assert np.array_equal(result['canonical'], result['no_part'])
    assert all(not d['matched_unshared_boundary'] for d in result['decisions'])


def test_frame_boundary_never_admits_part():
    data = fixture()
    data[1]['boundaries']['boundaries'][0]['frame_hypothesis'] = True
    assert np.array_equal(run(data)['canonical'], data[3])


def test_shared_internal_neighbour_does_not_disqualify_an_unshared_outer_owner():
    data = fixture()
    gray, graph, raw, previous, candidate, owners, canonical = data
    cv2.line(gray, (120, 190), (170, 190), 0, 1)
    graph['proposals'].append({'id': 'internal', 'points_px': [[x, 190] for x in range(120, 171)]})
    first = graph['boundaries']['boundaries'][0]
    first['orientation'] = 'bounded_face'; first['owners'].append('internal')
    adjacent = deepcopy(first); adjacent['id'] = 'internal-face'; adjacent['owners'] = ['internal']
    graph['boundaries']['boundaries'].append(adjacent)
    result = run(data)
    assert result['complete'] and (result['canonical'] == 1).any()


def test_nonpart_candidates_are_unchanged_by_the_part_filter():
    data = list(fixture())
    # An existing producer-owned body pixel is used to exercise independent
    # non-part carry-through, not to invent a bbox or a new owner.
    p = np.flatnonzero(data[4] != data[3])[0]
    data[4].flat[p] = 8
    result = run(data)
    assert result['no_part'].flat[p] == result['canonical'].flat[p] == 8


def test_unknown_or_reviewed_authority_never_adds_new_roles():
    data = fixture()
    assert np.array_equal(run(data, supplied=False)['canonical'], data[3])
    assert np.array_equal(run(data, protected=np.full_like(data[0], 255))['canonical'], data[3])


@pytest.mark.parametrize('what', ['gray', 'graph', 'owners', 'canonical'])
def test_source_bound_packet_detects_stale_input(what):
    data = list(fixture())
    proof = provenance(*data, {'path': 'fixture/candidate.png', 'sha256': 'b'*64})
    if what == 'gray': data[0][0, 0] = 250
    if what == 'graph': data[1]['stroke_px'] += 1
    if what == 'owners': data[5].flat[np.flatnonzero(data[5])[0]] = 0
    if what == 'canonical': data[6][0]['feet_px'][0][0] += 1
    authority = cv_authority(data[0], data[2], {'path': 'fixture/raw.png', 'sha256': 'a'*64})
    with pytest.raises(ValueError): analyze(*data, authority=authority, evidence=proof)


def test_incomplete_context_rolls_back_to_safe_abstention(monkeypatch):
    data = repeated_fixture()
    monkeypatch.setitem(POLICY, 'maximum_source_points', 0)
    result = run(data)
    assert not result['complete'] and result['audit']['limits']
    assert np.array_equal(result['canonical'], result['no_part'])


def test_input_objects_are_immutable_and_targets_cannot_be_read(monkeypatch):
    data = fixture(); before = deepcopy(data)
    def forbidden(*args, **kwargs): raise AssertionError('Inference filesystem access')
    monkeypatch.setattr(Path, 'open', forbidden); monkeypatch.setattr(builtins, 'open', forbidden)
    result = run(data)
    for a, b in zip(data, before):
        assert np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b
    assert not result['source_pixels_erased'] and not result['production_admitted']


def test_case_order_does_not_change_single_case_inference():
    one, two = fixture(), repeated_fixture()
    forward = [run(d)['canonical'] for d in (one, two)]
    reverse = [run(d)['canonical'] for d in (two, one)]
    assert np.array_equal(forward[0], reverse[1]) and np.array_equal(forward[1], reverse[0])
