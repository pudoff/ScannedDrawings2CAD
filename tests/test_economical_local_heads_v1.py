from copy import deepcopy
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from cad_recovery import economical_local_heads_v1 as detector
from cad_recovery.economical_head_profiles_v1 import Profiles
from cad_recovery.head_parallel_context_v1 import check
from test_sparse_local_heads_v1 import mask


def test_progressive_samples_equal_eager_after_body_and_reuse_exact_rows():
    ink, _, _ = mask(); lazy, eager = Profiles(ink), Profiles(ink, eager=True)
    args = np.array([66., 80.]), np.array([1., 0.]), 13., 8.
    a, b = lazy.measure(*args, 'coarse'), eager.measure(*args, 'coarse')
    assert lazy.samples < eager.samples
    assert np.array_equal(a[1][a[3]], b[1][a[3]])
    lazy.measure(*args, 'tail'); a = lazy.measure(*args, 'body')
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2])
    previous = lazy.samples; lazy.measure(*args, 'body')
    assert lazy.samples == previous == eager.samples and lazy.hits > 0


@pytest.mark.parametrize('width', [1, 2])
@pytest.mark.parametrize('angle', [0, 90])
def test_short_head_has_consistent_source_support_contrast(width, angle):
    ink, tip, direction = mask(width, angle)
    result = detector.search(255-ink)
    assert len(result['added']) == 1 and not result['search_incomplete']
    h = result['added'][0]; e = h['evidence']
    assert np.linalg.norm(np.array(h['tip_px'])-tip) < 3 and np.dot(h['direction'], direction) > .99
    assert np.isclose(e['base_support_width_px'], e['base_width_px']+1)
    assert np.isclose(e['base_to_shaft_ratio'], e['base_support_width_px']/e['local_shaft_width_px'])
    assert e['base_to_shaft_ratio'] >= 2.5 and e['native_head_check']['triangle_ink'] >= .85
    assert e['native_head_check']['flank_ink'] <= .35


def test_eager_and_lazy_heads_and_context_match_when_complete():
    ink, _, _ = mask()
    a, b = detector.search(255-ink), detector.search(255-ink, eager_reference=True)
    assert a['candidates'] == b['candidates'] and a['proposals'] == b['proposals']
    assert a['context_exclusions'] == b['context_exclusions']
    assert not a['search_incomplete'] and not b['search_incomplete']
    assert a['analysis']['profile_samples'] < b['analysis']['profile_samples']


def test_rectangle_rejection_does_not_spend_full_profiles():
    gray = np.full((160, 160), 255, np.uint8); cv2.rectangle(gray, (35, 45), (125, 90), 0, -1)
    a, b = detector.search(gray), detector.search(gray, eager_reference=True)
    assert not a['added'] and not b['added']
    assert a['analysis']['profile_samples'] <= b['analysis']['profile_samples']
    assert a['analysis']['profile_stage_samples']['body'] == 0
    assert a['analysis']['coverage_complete']


def test_parallel_field_abstains_but_retains_candidate_evidence():
    ink = np.zeros((180, 180), np.uint8)
    for x in (42, 61, 80, 99, 118): cv2.line(ink, (x, 32), (x+22, 150), 255, 2)
    polygon = [[94, 112], [77, 85], [92, 82]]
    cv2.fillConvexPoly(ink, np.array(polygon, np.int32), 255)
    u = np.array(polygon[0])-np.mean(polygon[1:], 0); u = u/np.linalg.norm(u)
    head = {'polygon_px': polygon, 'direction': u.tolist(), 'evidence': {'local_shaft_width_px': 3}}
    audit = {'context_pixels': 0, 'context_segments': 0}
    result = check(ink, head, audit)
    assert not result['available'] and not result['incomplete']
    assert result['reason'] == 'ambiguous_parallel_ink_field' and result['support']['parallel_offsets'] >= 3


def test_head_beside_hatch_is_not_rejected_by_a_distant_parallel_field():
    ink = np.zeros((180, 180), np.uint8)
    for x in (90, 105, 120, 135): cv2.line(ink, (x, 30), (x+15, 145), 255, 2)
    polygon = [[80, 80], [57, 72], [57, 88]]
    cv2.fillConvexPoly(ink, np.array(polygon, np.int32), 255); cv2.line(ink, (25, 80), (57, 80), 255, 2)
    head = {'polygon_px': polygon, 'direction': [1., 0.], 'evidence': {'local_shaft_width_px': 3}}
    result = check(ink, head, {'context_pixels': 0, 'context_segments': 0})
    assert result['available'] and not result['incomplete']


def test_context_caps_are_global_and_explicit():
    ink, _, _ = mask(); head = {'polygon_px': [[80,80],[67,76],[67,84]], 'direction': [1,0], 'evidence': {'local_shaft_width_px': 1}}
    audit = {'context_pixels': 100, 'context_segments': 0}
    result = check(ink, head, audit, maximum_pixels=100)
    assert result['incomplete'] and result['reason'] == 'context_pixel_budget' and audit['context_pixels'] == 100


def test_all_raw_regions_are_accounted_for_and_caps_have_no_window_reset():
    ink, _, _ = mask(); r = detector.search(255-ink); a = r['analysis']
    assert a['coverage_complete'] and a['raw_regions_generated'] == a['raw_regions_processed'] == sum(a['region_status'].values())
    assert sum(a['profile_stage_samples'].values()) == a['profile_samples'] <= 8000000
    with patch.dict(detector.POLICY, {'maximum_profile_samples': 0}): blocked = detector.search(255-ink)
    assert blocked['search_incomplete'] and 'profile_sample_budget' in blocked['analysis']['limits']
    assert not blocked['added']


def test_baseline_ids_source_pixels_and_text_exclusions_preserved():
    ink, _, _ = mask(); gray = 255-ink
    original = detector.search(gray)['candidates']; previous = deepcopy(original)
    a = detector.search(gray, baseline=original); b = detector.search(gray, baseline=original)
    assert a == b and original == previous and a['candidates'][:len(original)] == previous
    assert np.array_equal(gray, 255-ink)
    blocked = detector.search(gray, text_polygons=[[[40,60],[100,60],[100,100],[40,100]]])
    assert not blocked['added'] and a['CAD_promotions'] == 0 and not a['production_admitted']


def test_unsupported_triangle_and_clipped_head_do_not_gain_authority():
    ink, _, _ = mask(shaft=False)
    assert not detector.search(255-ink)['added']
    ink, _, _ = mask(); assert not detector.search((255-ink)[78:])['added']


def test_half_raster_contrast_compares_support_widths_in_the_same_domain():
    image = np.full((320, 320), 255, np.uint8)
    cv2.fillConvexPoly(image, np.rint([[160,160],[186,152.8],[186,167.2]]).astype(np.int32), 0)
    cv2.line(image, (186,160), (250,160), 0, 4)
    gray = cv2.resize(image, (160,160), interpolation=cv2.INTER_AREA)
    result = detector.search(gray)
    assert len(result['added']) == 1
    e = result['added'][0]['evidence']
    assert e['base_width_px']/e['local_shaft_width_px'] < 2.5
    assert e['base_to_shaft_ratio'] >= 2.5 and e['contrast_domain'] == 'source_pixel_support_for_both_operands'
