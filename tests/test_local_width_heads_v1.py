"""Source-backed local heads without fitted LINE inputs or annotation oracles."""
from copy import deepcopy
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from cad_recovery import local_width_heads_v1 as detector


def drawing(width=2, angle=23, shaft=True):
    gray = np.full((256, 256), 255, np.uint8)
    radians = np.deg2rad(angle); u = np.array([np.cos(radians), np.sin(radians)])
    n = np.array([-u[1], u[0]]); tip = np.array([128., 128.]); base = tip-30*u
    p = np.array([tip, base+9*n, base-9*n])
    cv2.fillConvexPoly(gray, np.rint(p).astype(np.int32), 0)
    if shaft:
        cv2.line(gray, tuple(np.rint(base).astype(int)), tuple(np.rint(base-60*u).astype(int)), 0, width)
    return gray, tip, u


@pytest.mark.parametrize('width', [1, 2, 4, 6])
@pytest.mark.parametrize('angle', [0, 23, 90, 137])
def test_local_source_head_needs_no_LINE_fitting_and_has_correct_direction(width, angle):
    gray, tip, direction = drawing(width, angle)
    before = gray.copy()
    with patch('ezdxf.readfile', side_effect=AssertionError('No DXF oracle')):
        result = detector.search(gray)
    assert len(result['added']) == 1
    head = result['added'][0]
    assert np.linalg.norm(np.array(head['tip_px'])-tip) < 4
    assert np.dot(head['direction'], direction) > np.cos(np.deg2rad(10))
    assert head['evidence']['source_checked'] and head['evidence']['base_to_shaft_ratio'] >= 2.5
    assert head['evidence']['native_head_check']['flank_ink'] <= .35
    assert np.array_equal(gray, before) and not result['search_incomplete']


def test_unconnected_triangle_has_no_own_shaft():
    gray, _, _ = drawing(shaft=False)
    assert not detector.search(gray)['added']


def test_blank_rectangle_and_crossing_are_not_arrows():
    gray = np.full((256, 256), 255, np.uint8)
    cv2.rectangle(gray, (25, 25), (220, 78), 0, 3)
    cv2.line(gray, (50, 130), (210, 130), 0, 4)
    cv2.line(gray, (120, 95), (120, 220), 0, 4)
    assert not detector.search(gray)['added']


def test_saved_text_polygon_is_an_exclusion_not_a_source_of_heads():
    gray, _, _ = drawing()
    result = detector.search(gray, text_polygons=[[[70, 70], [160, 70], [160, 170], [70, 170]]])
    assert not result['added']


def test_original_baseline_ids_and_observations_are_preserved_exactly():
    gray, _, _ = drawing()
    heads = detector.search(gray)['candidates']; heads[0]['id'] = 'original-id'
    previous = deepcopy(heads)
    result = detector.search(gray, baseline=heads)
    assert result['candidates'] == previous == heads and not result['added']


@pytest.mark.parametrize('key,reason', [('maximum_proposals', 'proposal_budget'),
    ('maximum_profile_samples', 'profile_sample_budget'), ('maximum_morphology_pixels', 'morphology_pixel_budget'),
    ('maximum_components_per_scale', 'component_budget')])
def test_resource_limits_are_explicit_and_never_accepted_as_complete(key, reason):
    gray, _, _ = drawing()
    with patch.dict(detector.POLICY, {key: 0}):
        result = detector.search(gray)
    assert result['search_incomplete'] and reason in result['analysis']['limits']
    assert not result['added']


def test_clipped_head_does_not_use_missing_pixels_as_background():
    gray, _, _ = drawing(angle=0)
    gray = gray[124:]
    result = detector.search(gray)
    assert not result['added']


def test_uncalibrated_review_only_and_repeated_bytes():
    gray, _, _ = drawing()
    a, b = detector.search(gray), detector.search(gray)
    assert a == b and a['CAD_promotions'] == 0
    assert not a['production_admitted'] and not a['physical_geometry_approved']
    assert not a['policy']['DT_head_proposals_used'] and not a['policy']['source_erasure_allowed']


@pytest.mark.parametrize('gray', [np.zeros((8, 8, 3), np.uint8), np.zeros((8, 8), float), np.zeros((0, 8), np.uint8)])
def test_invalid_input_rejected(gray):
    with pytest.raises(ValueError): detector.search(gray)
