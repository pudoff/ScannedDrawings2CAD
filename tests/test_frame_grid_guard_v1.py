from copy import deepcopy
import builtins
import cv2
import numpy as np
import pytest
from cad_recovery.frame_grid_guard_v1 import analyze, bind_context, POLICY
from cad_recovery.grid_role_revision_v1 import cv_authority, sha


def fixture():
    gray = np.full((100, 120), 255, np.uint8)
    cv2.line(gray, (15, 30), (105, 30), 0, 3)
    graph = {'proposals': [{'id': 'observed', 'points_px': [[x, 30] for x in range(15, 106)]}]}
    raw = np.zeros_like(gray); previous = raw.copy(); base = raw.copy()
    base[gray == 0] = 8
    owners = (gray == 0).astype(np.uint16)
    page = {'source_verified': True, 'source': {'path': 'page.png', 'sha256': 'a'*64},
            'size_px': [600, 600], 'source_size_px': [600, 600], 'overview_to_source': np.eye(3).tolist(),
            'patch_bbox': [200, 300, 320, 400], 'patch_to_source': [[1, 0, 200], [0, 1, 300], [0, 0, 1]],
            'source_to_patch': [[1, 0, -200], [0, 1, -300], [0, 0, 1]]}
    detector = {'size_px': [600, 600], 'source_sha256': 'b'*64, 'complete': True,
                'frames': [{'frame_semantics_confirmed': False, 'sides': [
                    {'points_overview': [[210, 330], [310, 330]], 'stroke_width_overview_px': 3,
                     'position_uncertainty_overview_px': .71, 'source_sample_count': 21}]}]}
    context = bind_context(gray, graph, previous, base, owners, page, detector, 'b'*64, np.eye(3), {'path': 'packet.json', 'sha256': 'c'*64})
    return gray, graph, raw, previous, base, owners, context, page, detector


def run(data, *, context=True, protected=None, authority=True):
    gray, graph, raw, previous, base, owners, packet, _, _ = data
    auth = cv_authority(gray, raw, {'path': 'raw.png', 'sha256': 'd'*64}, protected=protected) if authority else None
    return analyze(gray, graph, raw, previous, base, owners, authority=auth, context=packet if context else None)


def test_page_context_restores_previous_without_assigning_frame_or_erasing_ink():
    data = fixture(); before = deepcopy(data)
    result = run(data)
    assert np.array_equal(result['page_context'], data[3]) and result['complete']
    assert np.array_equal(result['local_edge'], data[4])
    assert result['new_frame_roles'] == result['source_pixels_erased'] == 0
    for i in (0, 2, 3, 4, 5): assert np.array_equal(before[i], data[i])
    assert before[1] == data[1] and before[6] == data[6]


def test_missing_parent_context_never_treats_roi_as_full_page():
    data = fixture()
    assert np.array_equal(run(data, context=False)['page_context'], data[4])


def test_parallel_line_displaced_from_parent_frame_is_preserved():
    data = fixture(); data[6]['sides'][0]['points_page'] = [[210, 350], [310, 350]]
    assert np.array_equal(run(data)['page_context'], data[4])


def test_crossing_owner_direction_is_not_a_frame_side_match():
    data = fixture(); data[6]['sides'][0]['points_page'] = [[260, 300], [260, 400]]
    assert np.array_equal(run(data)['page_context'], data[4])


def test_review_protection_and_unknown_authority_preserve_candidates():
    data = fixture()
    assert np.array_equal(run(data, protected=np.full_like(data[0], 255))['page_context'], data[4])
    assert np.array_equal(run(data, authority=False)['page_context'], data[4])


@pytest.mark.parametrize('field,index', [('input_sha256', 0), ('graph_sha256', 1), ('previous_sha256', 3), ('base_sha256', 4), ('owners_sha256', 5)])
def test_rebound_source_context_is_rejected(field, index):
    data = fixture(); data[6][field] = 'f'*64
    with pytest.raises(ValueError, match='Stale'): run(data)


def test_semantic_or_manual_frame_packet_is_rejected():
    data = fixture(); data[6]['GT_used'] = True
    with pytest.raises(ValueError): run(data)
    data = fixture(); data[8]['frames'][0]['frame_semantics_confirmed'] = True
    with pytest.raises(ValueError): bind_context(data[0], data[1], data[3], data[4], data[5], data[7], data[8], 'b'*64, np.eye(3), {})


@pytest.mark.parametrize('kind', ['page_identity', 'inverse', 'pixel_centres', 'origin'])
def test_wrong_page_or_transform_is_rejected(kind):
    d = fixture(); transform = np.eye(3)
    if kind == 'page_identity': d[8]['source_sha256'] = 'e'*64
    if kind == 'inverse': d[7]['source_to_patch'][0][2] += 1
    if kind == 'pixel_centres': transform[0, 0] = 2
    if kind == 'origin': d[7]['patch_bbox'][0] += 1
    with pytest.raises(ValueError): bind_context(d[0], d[1], d[3], d[4], d[5], d[7], d[8], 'b'*64, transform, {})


def test_incomplete_detector_or_guard_budget_rolls_back(monkeypatch):
    d = fixture(); d[6]['detector_complete'] = False
    r = run(d); assert not r['complete'] and np.array_equal(r['page_context'], d[4])
    d = fixture(); monkeypatch.setitem(POLICY, 'maximum_source_points', 0)
    r = run(d); assert not r['complete'] and np.array_equal(r['page_context'], d[4])


def test_analyzer_reads_no_files_or_targets(monkeypatch):
    d = fixture()
    def blocked(*a, **kw): raise AssertionError('Filesystem read during inference')
    monkeypatch.setattr(builtins, 'open', blocked)
    assert np.array_equal(run(d)['page_context'], d[3])


def test_half_pixel_footprint_is_propagated_to_page_radius():
    d = fixture(); gray = cv2.resize(d[0], (60, 50), interpolation=cv2.INTER_AREA)
    p = cv2.resize(d[3], (60, 50), interpolation=cv2.INTER_NEAREST); b = p.copy(); owners = p.astype(np.uint16)
    half = [[2, 0, .5], [0, 2, .5], [0, 0, 1]]
    ctx = bind_context(gray, d[1], p, b, owners, d[7], d[8], 'b'*64, half, {})
    assert ctx['input_to_page'] == [[2, 0, 200.5], [0, 2, 300.5], [0, 0, 1]]
    assert ctx['sides'][0]['radius_page_px'] == pytest.approx(d[6]['sides'][0]['radius_page_px']+.5)
