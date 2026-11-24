from copy import deepcopy
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from cad_recovery import supported_role_overlay_v1 as overlay
from cad_recovery.local_role_guard_v1 import analyze as local_guard
from test_local_role_guard_v1 import prior_fixture


def fixture(*, leader=False, broken=False):
    gray, previous, _, text, heads = prior_fixture(leader=leader, broken=broken)
    baseline = np.full_like(gray, 255)
    local = local_guard(gray, previous, baseline, text, heads)
    return gray, previous, baseline, local, heads


def test_positive_chain_adds_only_dimension_and_extension_roles():
    gray, previous, baseline, local, heads = fixture()
    result = overlay.analyze(gray, previous, baseline, local, heads=heads)
    assert result['added_pixels'] > 0
    assert set(np.unique(result['pixel_roles'])) <= {2, 3, 255}
    assert np.all(result['pixel_interval_owner'][result['pixel_roles'] != 255] > 0)
    assert result['head_ids_used'] == local['hybrid']['head_ids_used']


def test_all_known_baseline_roles_preserved_not_just_axis_and_grid():
    gray, previous, baseline, local, heads = fixture()
    baseline[90, 160:169] = np.arange(1, 10, dtype=np.uint8)
    result = overlay.analyze(gray, previous, baseline, local, heads=heads)
    assert np.array_equal(result['pixel_roles'][90, 160:169], np.arange(1, 10))
    assert result['baseline_known_roles_unchanged']


def test_broad_hatch_boundary_and_leader_hypotheses_do_not_authorize_new_pixels():
    gray, previous, baseline, local, heads = fixture(leader=True)
    assert local['hybrid']['leader_hypotheses']
    result = overlay.analyze(gray, previous, baseline, local, heads=heads)
    assert not result['added_pixels'] and np.all(result['pixel_roles'] == 255)
    assert result['leader_hypotheses'] == local['hybrid']['leader_hypotheses']


def test_broken_extensions_stay_unconfirmed():
    gray, previous, baseline, local, heads = fixture(broken=True)
    result = overlay.analyze(gray, previous, baseline, local, heads=heads)
    assert not result['dimension_chains'] and not result['added_pixels']


def test_unknown_source_paths_compete_and_disconnected_ink_cannot_inherit_role():
    ink = np.zeros((40, 50), np.uint8)
    cv2.line(ink, (5, 10), (44, 10), 255, 3)
    cv2.line(ink, (5, 18), (44, 18), 255, 1)
    intervals = [{'id': 'known', 'points_px': [[5, 10], [44, 10]]},
                 {'id': 'unknown', 'points_px': [[5, 18], [44, 18]]}]
    result, audit = overlay.source_ownership(ink, intervals)
    assert not audit['incomplete'] and result['owner'][18, 20] == 2
    one, _ = overlay.source_ownership(ink, intervals[:1])
    assert one['owner'][18, 20] == 0


def test_shared_source_seed_abstains_at_junction():
    ink = np.zeros((30, 30), np.uint8)
    cv2.line(ink, (5, 15), (25, 15), 255, 1); cv2.line(ink, (15, 5), (15, 25), 255, 1)
    data = [{'points_px': [[5, 15], [25, 15]]}, {'points_px': [[15, 5], [15, 25]]}]
    result, audit = overlay.source_ownership(ink, data)
    assert result['owner'][15, 15] == -1 and audit['shared_seed_pixels'] == 1


@pytest.mark.parametrize('key', ['maximum_source_points', 'maximum_interval_raster_pixels'])
def test_ownership_budget_falls_back_to_baseline_with_explicit_incomplete(key):
    gray, previous, baseline, local, heads = fixture()
    with patch.dict(overlay.POLICY, {key: 0}):
        result = overlay.analyze(gray, previous, baseline, local, heads=heads)
    assert result['search_incomplete'] and np.array_equal(result['pixel_roles'], baseline)
    assert result['source_ownership_audit']['incomplete']


def test_source_intervals_and_all_observations_unchanged():
    gray, previous, baseline, local, heads = fixture()
    before = deepcopy(previous); pixels = gray.copy(); head_copy = deepcopy(heads)
    result = overlay.analyze(gray, previous, baseline, local, heads=heads)
    assert before == previous and head_copy == heads and np.array_equal(pixels, gray)
    for p, q in zip(previous['proposals'], result['proposals']):
        for key in ('id', 'path_id', 'source_index_interval', 'points_px'): assert p[key] == q[key]
    assert result['dimension_chains'] == local['hybrid']['dimension_chains']
    assert result['head_events'] == local['hybrid']['head_events'] and not result['DT_admitted_for_roles']


def test_no_text_erasure_CAD_or_physical_authority():
    gray, previous, baseline, local, heads = fixture()
    result = overlay.analyze(gray, previous, baseline, local, heads=heads)
    assert np.array_equal(result['text_mask'], local['hybrid']['text_mask'])
    assert result['source_pixels_erased'] == result['semantic_CAD_promotions'] == 0
    assert not result['production_admitted'] and not result['physical_geometry_approved']


def test_invalid_source_coordinates_rejected_before_raster_allocation():
    with pytest.raises(ValueError):
        overlay.source_ownership(np.ones((4, 4), np.uint8), [{'points_px': [[1, 1], [float('nan'), 2]]}])
