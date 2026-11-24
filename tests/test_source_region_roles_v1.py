from copy import deepcopy
from pathlib import Path
import builtins
import cv2
import numpy as np
import pytest
from cad_recovery.source_region_roles_v1 import analyze, POLICY
from cad_recovery.grid_role_revision_v1 import cv_authority


def fixture(rings=False):
    gray = np.full((360, 480), 255, np.uint8)
    proposals, tracks = [], []
    for a, b in [*([(30, y), (450, y)] for y in (35, 130, 225, 325)),
                 *([(x, 35), (x, 325)] for x in (30, 250, 450))]:
        cv2.line(gray, a, b, 0, 9)
        p = np.rint(np.linspace(a, b, int(np.linalg.norm(np.subtract(b, a)))+1)).astype(int)
        identity = f'grid-{len(proposals)}'
        proposals.append({'id': identity, 'points_px': p.tolist()})
        tracks.append({'id': f'track-{len(tracks)}', 'points': [list(a), list(b)], 'width': 11., 'interval_ids': [identity]})
    for y in (105, 200, 295):
        cv2.putText(gray, 'R E P 4', (50, y), cv2.FONT_HERSHEY_COMPLEX, .8, 0, 1)
    if rings:
        for y in (80, 175, 275): cv2.circle(gray, (350, y), 18, 0, 3)
    # Glyph source paths are also ordinary competitors, not privileged masks.
    from skimage.morphology import skeletonize
    ink = (gray < 128).astype(np.uint8)*255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] > 2000: continue
        yy, xx = np.nonzero(skeletonize(labels == i))
        # Singleton paths still compete for their observed pixels.
        for x, y in zip(xx, yy): proposals.append({'id': f'pixel-{len(proposals)}', 'points_px': [[int(x), int(y)]]})
    return gray, {'proposals': proposals, 'tracks': tracks}, np.full_like(gray, 255)


def auth(gray, roles, protected=None):
    return cv_authority(gray, roles, {'path': 'fixture/roles.png', 'sha256': 'c'*64}, protected=protected)


def test_positive_grid_assignments_need_source_owners_and_leave_paper_unknown():
    gray, graph, roles = fixture()
    result = analyze(gray, graph, roles, authority=auth(gray, roles))
    delta = result['guarded'] != result['previous']
    assert result['complete'] and delta.any() and (result['guarded'] == 8).any()
    assert np.all(result['owners']['guarded'][delta] > 0)
    assert np.all(result['guarded'][gray == 255] == 255)
    assert not any(d['bbox_filled'] for d in result['decisions'])


def test_repeated_round_components_veto_grid_interpretation_but_do_not_paint_circles():
    gray, graph, roles = fixture(True)
    result = analyze(gray, graph, roles, authority=auth(gray, roles))
    assert (result['structure'] == 8).any() and not (result['guarded'] == 8).any()
    assert any(d['table_veto_repeated_circles'] for d in result['decisions'])
    assert not (result['guarded'] == 1).any()


def test_geometry_without_glyph_context_cannot_be_a_table():
    gray, graph, roles = fixture()
    gray[50:120, 40:230] = 255
    gray[145:215, 40:230] = 255
    gray[240:315, 40:230] = 255
    graph['proposals'] = [p for p in graph['proposals'] if p['id'].startswith('grid-')]
    result = analyze(gray, graph, roles, authority=auth(gray, roles))
    assert np.array_equal(result['guarded'], roles)


def test_manual_protection_and_unknown_origin_block_positive_assignments():
    gray, graph, roles = fixture()
    for authority in (None, auth(gray, roles, np.full_like(gray, 255))):
        result = analyze(gray, graph, roles, authority=authority)
        assert np.array_equal(result['guarded'], roles)


def test_non_hatch_known_roles_cannot_be_overwritten_by_grid_context():
    gray, graph, roles = fixture()
    roles[gray < 128] = 2
    result = analyze(gray, graph, roles, authority=auth(gray, roles))
    assert np.array_equal(result['guarded'], roles) and np.array_equal(result['structure'], roles)


def test_authority_hash_mismatch_rejected_before_assignments():
    gray, graph, roles = fixture()
    a = auth(gray, roles)
    a['mask'][0, 0] = 2
    with pytest.raises(ValueError): analyze(gray, graph, roles, authority=a)


def test_budget_failure_returns_previous_and_clears_new_owners(monkeypatch):
    gray, graph, roles = fixture()
    monkeypatch.setitem(POLICY, 'maximum_grids', 0)
    result = analyze(gray, graph, roles, authority=auth(gray, roles))
    assert not result['complete']
    assert np.array_equal(result['guarded'], result['previous']) and not result['owners']['guarded'].any()


def test_inputs_remain_exact_and_model_cannot_read_targets(monkeypatch):
    gray, graph, roles = fixture()
    before, pixels, old = deepcopy(graph), gray.copy(), roles.copy()
    a = auth(gray, roles)
    def forbidden(*args, **kwargs): raise AssertionError('Unexpected inference filesystem access')
    monkeypatch.setattr(Path, 'open', forbidden)
    monkeypatch.setattr(builtins, 'open', forbidden)
    result = analyze(gray, graph, roles, authority=a)
    assert graph == before and np.array_equal(gray, pixels) and np.array_equal(roles, old)
    assert not result['source_pixels_erased'] and not result['physical_geometry_approved']
