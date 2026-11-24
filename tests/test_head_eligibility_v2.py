from copy import deepcopy
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pytest

from cad_recovery.head_eligibility_v2 import apply, POLICY

ROOT = Path(__file__).resolve().parents[1]


def existing(case, mode):
    plan = json.loads((ROOT/'artifacts/local-axis-heads-v2/plan.json').read_text('utf-8'))
    index = json.loads((ROOT/'artifacts/local-axis-heads-v2/results-index.json').read_text('utf-8'))
    job = next(j for j in plan['jobs'] if j['case']['id'] == case)
    view = next(v for v in job['views'] if v['mode'] == mode)
    item = next(r for r in index if r['case'] == case and r['mode'] == mode and not r['repeat'])
    packet = json.loads((ROOT/item['artifact']).read_text('utf-8'))['local_axis_guarded']
    return np.array(Image.open(ROOT/view['gray']['path']).convert('L')), packet['candidates']


@pytest.mark.parametrize('case,mode,reason', [
    ('head-support-parallel_loop_noise-w6-r0', 'half_resolution_crop', 'enclosed_source_cavity_inside_filled_head'),
    ('head-support-parallel_loop_noise-w6-r90', 'half_resolution_crop', 'enclosed_source_cavity_inside_filled_head'),
    *[(f'curved-head-triangles_without_tail-w6-r{angle}', 'native_crop',
       'compact_filled_triangle_without_independent_tail') for angle in (23, 68, 119, 151)],
])
def test_six_viewed_false_heads_are_excluded_only_by_sidecar(case, mode, reason):
    gray, heads = existing(case, mode)
    source, before = gray.copy(), deepcopy(heads)
    result = apply(gray, heads)
    assert not result['search_incomplete'] and not result['diagnostic_candidate_ids']
    assert len(result['decisions']) == len(heads) == 1
    decision = result['decisions'][0]
    assert decision['reason'] == reason and decision['status'] == 'ineligible'
    assert decision['observation_retained'] and not decision['eligible_for_CAD']
    assert heads == before and np.array_equal(source, gray)


def test_raster_staircase_is_checked_by_boundary_and_interior_not_area_ratio():
    gray, heads = existing('curved-head-triangles_without_tail-w6-r23', 'native_crop')
    decision = apply(gray, heads)['decisions'][0]
    component = decision['evidence']['component']
    assert component['solidity_for_diagnosis_only'] < .95
    assert component['maximum_convex_defect_px'] <= 2**.5
    assert component['enclosing_triangle_interior_fill'] >= .98
    assert decision['status'] == 'ineligible'


@pytest.mark.parametrize('case,mode', [
    ('head-support-shared_short_pair-w4-r90', 'half_resolution_crop'),
    ('head-support-single_arc_pointer-w4-r0', 'half_resolution_crop'),
    ('curved-head-inflected_tail-w6-r23', 'native_crop'),
    ('curved-head-inflected_tail-w6-r151', 'half_resolution_crop'),
])
def test_newly_recovered_source_path_heads_are_not_lost(case, mode):
    gray, heads = existing(case, mode)
    result = apply(gray, heads)
    assert result['diagnostic_candidate_ids'] == [h['id'] for h in heads]
    assert all(d['status'] == 'review' and not d['eligible_for_CAD'] for d in result['decisions'])


def example_head(points):
    polygon = np.asarray(points, float)
    delta = polygon[0]-polygon[1:].mean(0)
    return {'id': 'test-head', 'polygon_px': polygon.tolist(), 'direction': (delta/np.linalg.norm(delta)).tolist(),
            'tip_px': polygon[0].tolist(), 'evidence': {}, 'confidence': .55}


def test_a_cavity_open_to_background_does_not_prove_a_closed_loop():
    gray, heads = existing('head-support-parallel_loop_noise-w6-r0', 'half_resolution_crop')
    assert apply(gray, heads)['decisions'][0]['reason'] == 'enclosed_source_cavity_inside_filled_head'
    cv2.line(gray, (120, 0), (120, 107), 255, 1)
    result = apply(gray, heads)
    assert result['diagnostic_candidate_ids'] == [heads[0]['id']]
    assert not result['decisions'][0]['evidence']['counterevidence']


def test_border_clipped_component_cannot_prove_a_complete_isolated_triangle():
    ink = np.zeros((100, 100), np.uint8)
    head = example_head([[50, 50], [0, 35], [0, 65]])
    cv2.fillConvexPoly(ink, np.array(head['polygon_px'], np.int32), 255)
    result = apply(255-ink, [head])
    assert result['diagnostic_candidate_ids'] == [head['id']]


def test_budget_exhaustion_rolls_back_exclusions_and_exposes_incompleteness():
    gray, heads = existing('head-support-parallel_loop_noise-w6-r0', 'half_resolution_crop')
    result = apply(gray, heads, inherited_morphology_pixels=40000000)
    assert result['search_incomplete'] and result['diagnostic_candidate_ids'] == [h['id'] for h in heads]
    assert result['analysis']['topology_pass_pixels'] == 0
    assert result['decisions'][0]['reason'] == 'eligibility_budget_unchecked'
    assert result['analysis']['combined_morphology_pixels'] == 40000000


def test_geometry_too_large_for_local_proof_remains_reviewable():
    gray = np.full((512, 512), 255, np.uint8)
    head = example_head([[450, 256], [40, 200], [40, 312]])
    result = apply(gray, [head])
    assert result['diagnostic_candidate_ids'] == [head['id']]
    assert result['decisions'][0]['reason'] == 'local_topology_window_unavailable'
    assert not result['search_incomplete'] and result['analysis']['topology_pass_pixels'] == 0


def test_empty_observation_set_is_complete_and_cannot_certify_CAD():
    result = apply(np.full((80, 80), 255, np.uint8), [])
    assert not result['decisions'] and not result['search_incomplete']
    assert result['automatic_CAD_promotions'] == 0 and not result['production_admitted']


def test_duplicate_ids_and_nonfinite_geometry_are_rejected():
    gray = np.full((80, 80), 255, np.uint8)
    head = example_head([[60, 40], [20, 30], [20, 50]])
    with pytest.raises(ValueError):
        apply(gray, [head, head])
    head['polygon_px'][0][0] = float('nan')
    with pytest.raises(ValueError):
        apply(gray, [head])


@pytest.mark.parametrize('gray', [np.zeros((0, 0), np.uint8), np.zeros((10, 10), float), np.zeros((10, 10, 3), np.uint8)])
def test_invalid_source_is_rejected(gray):
    with pytest.raises(ValueError):
        apply(gray, [])
