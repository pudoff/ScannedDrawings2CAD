from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image
import pytest

from cad_recovery import local_axis_heads_v2 as detector
from cad_recovery import local_axis_ownership_v1 as guard
from cad_recovery.local_head_axis_support_v2 import follow_tail
from cad_recovery.streamed_head_profiles_v1 import Profiles
from cad_recovery.streamed_local_heads_v1 import search as previous_search

ROOT = Path(__file__).resolve().parents[1]


def source(case, mode=0):
    plan = json.loads((ROOT/'artifacts/streamed-heads-v1/plan.json').read_text('utf-8'))
    job = next(j for j in plan['jobs'] if j['case']['id'] == case)
    return np.array(Image.open(ROOT/job['views'][mode]['gray']['path']).convert('L'))


def test_straight_support_is_measured_to_unchanged_width_relative_requirement():
    mask = np.zeros((80, 100), np.uint8)
    cv2.line(mask, (10, 40), (80, 40), 255, 4)
    profiles = Profiles(mask)
    result, reason = follow_tail(np.array([60., 40.]), np.array([1., 0.]), 18., profiles)
    assert reason is None and result['extent_px'] >= max(6, 2*result['width_px'])
    assert result['extent_px'] <= 40 and 2 <= len(result['points_px']) <= 41
    assert profiles.samples == sum(profiles.stage_samples.values()) > 0


def test_a_broken_source_cannot_be_bridged_to_a_distant_stroke():
    mask = np.zeros((80, 100), np.uint8)
    cv2.line(mask, (10, 40), (48, 40), 255, 2)
    cv2.line(mask, (57, 40), (65, 40), 255, 2)
    result, reason = follow_tail(np.array([60., 40.]), np.array([1., 0.]), 12., Profiles(mask))
    assert result is None and reason == 'own_source_path_too_short'


def test_path_queries_share_and_respect_existing_profile_sample_budget():
    mask = np.full((80, 100), 255, np.uint8)
    profiles = Profiles(mask, maximum_samples=10)
    result, reason = follow_tail(np.array([60., 40.]), np.array([1., 0.]), 12., profiles)
    assert result is None and reason == 'profile_sample_budget'
    assert profiles.incomplete and profiles.samples <= 10


def test_wide_solid_regions_do_not_count_as_a_thin_own_shaft():
    mask = np.full((80, 100), 255, np.uint8)
    result, reason = follow_tail(np.array([60., 40.]), np.array([1., 0.]), 12., Profiles(mask))
    assert result is None and reason == 'no_own_source_path'


@pytest.mark.parametrize('case,mode', [
    ('head-support-shared_short_pair-w4-r90', 1),
    ('head-support-single_hook_pointer-w4-r147', 1),
    ('head-support-parallel_loop_noise-w6-r0', 1),
])
def test_original_source_hypotheses_and_pixels_are_preserved(case, mode):
    gray = source(case, mode)
    before = gray.copy()
    previous = previous_search(gray)
    result = detector.search(gray)
    assert np.array_equal(gray, before)
    assert result['proposals'][:len(previous['proposals'])] == previous['proposals']
    assert result['candidates'][:len(previous['candidates'])] == previous['candidates']
    assert not result['search_incomplete'] and result['analysis']['local_axis_complete']
    assert result['analysis']['profile_samples'] == sum(result['analysis']['profile_stage_samples'].values()) <= 8000000
    json.dumps(result, allow_nan=False)


def test_curved_support_is_evidence_and_never_claims_a_straight_extent():
    result = detector.search(source('head-support-single_arc_pointer-w4-r0', 1))
    novel = [h for h in result['added'] if 'shaft_support_path' in h['evidence']]
    assert novel
    for head in novel:
        e = head['evidence']
        assert not e['straight_shaft_extent_claimed']
        path = e['shaft_support_path']
        points = np.asarray(path['points_px'])
        assert path['extent_px'] == pytest.approx(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        assert e['base_to_shaft_ratio'] >= 2.5 and e['source_checked']
        assert e['native_head_check']['reason'] == 'native_filled_triangle_and_clear_flanks'


def test_guard_independently_reobserves_curved_support_and_preserves_raw_packet():
    gray = source('head-support-single_arc_pointer-w4-r0', 1)
    raw = detector.search(gray)
    before = deepcopy(raw)
    result = guard.apply(gray, raw)
    assert raw == before and not result['ownership_incomplete']
    assert any(d.get('witness', {}).get('method') == 'independent_source_reobservation_of_curved_support_path'
               for d in result['ownership_decisions'])
    assert result['analysis']['ownership']['source_samples'] > 0


def test_guard_does_not_accept_a_path_whose_source_has_been_removed():
    gray = source('head-support-single_arc_pointer-w4-r0', 1)
    raw = detector.search(gray)
    head = next(h for h in raw['added'] if 'shaft_support_path' in h['evidence'])
    audit = {'source_samples': 0, 'limits': set()}
    assert guard._witness(np.zeros_like(gray), head, audit) is None


def test_curved_support_inside_an_opposite_head_is_still_excluded():
    # Use the previous ownership regression, replacing only witness geometry.
    import test_head_support_ownership_v1 as fixture
    gray, owner, ghost = fixture.ghost_scene()
    ghost['evidence']['shaft_support_path'] = {
        'points_px': [[92.+i, 90.] for i in range(11)], 'width_px': 1.}
    result = guard.apply(gray, fixture.packet([owner], [ghost]), baseline=[owner])
    assert result['candidates'] == [owner]
    assert result['ownership_exclusions'][0]['candidate_id'] == 'ghost'
    assert result['proposals'] == [ghost]


@pytest.mark.parametrize('key', ['maximum_source_samples', 'maximum_pair_checks',
                               'maximum_owner_validation_pixels', 'maximum_containment_points'])
def test_guard_budget_failure_preserves_reviewable_observations(key):
    import test_head_support_ownership_v1 as fixture
    gray, owner, ghost = fixture.ghost_scene()
    ghost['evidence']['shaft_support_path'] = {
        'points_px': [[92.+i, 90.] for i in range(11)], 'width_px': 1.}
    with patch.dict(guard.POLICY, {key: 0}):
        result = guard.apply(gray, fixture.packet([owner], [ghost]), baseline=[owner])
    assert result['ownership_incomplete'] and result['search_incomplete']
    assert result['added'] == [ghost] and not result['ownership_exclusions']


def test_guarded_entrypoint_uses_the_new_detector():
    gray = source('head-support-single_arc_pointer-w4-r0', 1)
    assert detector.guarded_search(gray) == guard.apply(gray, detector.search(gray))


def test_empty_source_is_complete_without_invented_heads():
    result = detector.guarded_search(np.full((128, 128), 255, np.uint8))
    assert not result['search_incomplete'] and not result['candidates']
    assert result['analysis']['local_axis_complete']


@pytest.mark.parametrize('gray', [np.zeros((0, 0), np.uint8), np.zeros((10, 10), float), np.zeros((10, 10, 3), np.uint8)])
def test_invalid_source_is_rejected(gray):
    with pytest.raises(ValueError):
        detector.search(gray)
