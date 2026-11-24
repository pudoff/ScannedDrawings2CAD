from copy import deepcopy
import cv2
import numpy as np
import pytest
from cad_recovery.grid_context_v1 import analyze, source_track_context, merge_source_tracks, POLICY
from cad_recovery.grid_role_revision_v1 import cv_authority


def fragments(gap=0):
    ink = np.zeros((100, 200), np.uint8)
    intervals, tracks = [], []
    for i, (start, end) in enumerate([(10, 90), (91+gap, 180)]):
        cv2.line(ink, (start, 45), (end, 45), 255, 5)
        points = [[x, 45] for x in range(start, end+1)]
        identity = f'interval-{i}'
        intervals.append({'id': identity, 'points_px': points})
        # The source support is horizontal despite a noisy frozen endpoint.
        tracks.append({'id': f'track-{i}', 'points': [[start, 44], [end, 47]],
                       'width': 5., 'interval_ids': [identity]})
    return ink, {'proposals': intervals, 'tracks': tracks}


def authority(gray, roles, protected=None):
    return cv_authority(gray, roles, {'path': 'fixture/roles.png', 'sha256': 'd'*64}, protected=protected)


def table():
    gray = np.full((320, 380), 255, np.uint8)
    intervals, tracks = [], []
    pairs = [*([(30, y), (345, y)] for y in (35, 115, 195, 280)),
             *([(x, 35), (x, 280)] for x in (30, 190, 345))]
    for a, b in pairs:
        cv2.line(gray, a, b, 0, 7)
        points = np.rint(np.linspace(a, b, int(np.linalg.norm(np.subtract(b, a)))+1)).astype(int)
        identity = f'line-{len(intervals)}'
        intervals.append({'id': identity, 'points_px': points.tolist()})
        tracks.append({'id': f'track-{len(tracks)}', 'points': [list(a), list(b)], 'width': 9., 'interval_ids': [identity]})
    for y in (95, 175, 260):
        cv2.putText(gray, 'R P 47', (50, y), cv2.FONT_HERSHEY_COMPLEX, .9, 0, 1)
    return gray, {'proposals': intervals, 'tracks': tracks}, np.where(gray < 128, 9, 255).astype(np.uint8)


def test_fit_uses_observed_interval_samples_and_does_not_modify_graph():
    ink, graph = fragments()
    before = deepcopy(graph)
    result, audit = source_track_context(ink, graph)
    assert graph == before and audit['complete'] and not audit['raw_graph_changed']
    assert np.allclose(np.asarray(result['tracks'][0]['points'])[:, 1], 45)
    assert all(t['interval_ids'] == graph['tracks'][i]['interval_ids'] for i, t in enumerate(result['tracks']))


def test_collinear_fragments_merge_only_with_continuous_source_support():
    ink, graph = fragments()
    fit, audit = source_track_context(ink, graph)
    merged, work = merge_source_tracks(ink, fit, audit)
    assert work['complete'] and len(merged['tracks']) == 1
    assert set(merged['tracks'][0]['interval_ids']) == {'interval-0', 'interval-1'}
    assert work['merges'][0]['source_coverage'] == 1


def test_paper_gap_is_not_bridged_by_collinearity():
    ink, graph = fragments(15)
    fit, audit = source_track_context(ink, graph)
    merged, work = merge_source_tracks(ink, fit, audit)
    assert work['complete'] and len(merged['tracks']) == 2 and not work['merges']


def test_parallel_nearby_strokes_are_not_merged():
    ink, graph = fragments()
    graph['proposals'][1]['points_px'] = [[x, 55] for x in range(91, 181)]
    graph['tracks'][1]['points'] = [[91, 55], [180, 55]]
    cv2.line(ink, (91, 55), (180, 55), 255, 5)
    fit, audit = source_track_context(ink, graph)
    merged, work = merge_source_tracks(ink, fit, audit)
    assert work['complete'] and len(merged['tracks']) == 2


@pytest.mark.parametrize('field', ['missing_interval', 'duplicate_id', 'off_image', 'nan_point', 'bad_width'])
def test_invalid_source_mapping_is_rejected(field):
    ink, graph = fragments()
    if field == 'missing_interval': graph['tracks'][0]['interval_ids'] = ['missing']
    if field == 'duplicate_id': graph['proposals'][1]['id'] = graph['proposals'][0]['id']
    if field == 'off_image': graph['proposals'][0]['points_px'][0][0] = -1
    if field == 'nan_point': graph['proposals'][0]['points_px'][0][0] = float('nan')
    if field == 'bad_width': graph['tracks'][0]['width'] = 0
    with pytest.raises(ValueError): source_track_context(ink, graph)


def test_context_revision_preserves_previous_changes_and_source_owners():
    gray, graph, roles = table()
    result = analyze(gray, graph, roles, authority=authority(gray, roles))
    assert result['complete'] and np.any(result['previous'] != roles)
    for arm in ('source_fit', 'source_merged'):
        assert np.all(result[arm][result['previous'] == 255] == 255)
        delta = result[arm] != roles
        assert np.all(roles[delta] == 9) and np.all(result[arm][delta] == 255)
        assert np.all(result['owners'][arm][delta] > 0)


def test_protected_roles_and_unknown_authority_remain_exact():
    gray, graph, roles = table()
    protected = np.full_like(gray, 255)
    for auth in (None, authority(gray, roles, protected)):
        result = analyze(gray, graph, roles, authority=auth)
        assert all(np.array_equal(result[a], roles) for a in ('previous', 'source_fit', 'source_merged'))


def test_context_budget_failure_retains_previous_guarded_view(monkeypatch):
    gray, graph, roles = table()
    monkeypatch.setitem(POLICY, 'maximum_source_points', 1)
    result = analyze(gray, graph, roles, authority=authority(gray, roles))
    assert not result['complete']
    assert np.array_equal(result['source_fit'], result['previous'])
    assert np.array_equal(result['source_merged'], result['previous'])


def test_merging_budget_failure_is_explicit(monkeypatch):
    ink, graph = fragments()
    fit, audit = source_track_context(ink, graph)
    monkeypatch.setitem(POLICY, 'maximum_merge_pairs', 0)
    result, work = merge_source_tracks(ink, fit, audit)
    assert result is None and not work['complete'] and work['reason'] == 'merge_pair_budget'


def test_single_raster_call_does_not_read_external_evidence_or_mutate_inputs():
    gray, graph, roles = table()
    a = authority(gray, roles)
    before = deepcopy(graph)
    pixels, old, mask = gray.copy(), roles.copy(), a['mask'].copy()
    result = analyze(gray, graph, roles, authority=a)
    assert graph == before and np.array_equal(gray, pixels) and np.array_equal(roles, old) and np.array_equal(a['mask'], mask)
    assert not result['native_or_GT_inputs'] and not result['source_pixels_erased'] and not result['new_positive_roles']
    assert not result['production_admitted'] and not result['physical_geometry_approved']
