from copy import deepcopy
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest

from cad_recovery import contact_role_overlay_v2 as model
from cad_recovery.supported_role_overlay_v1 import analyze as old_analyze
from cad_recovery.stroke_role_metrics_v1 import to_native
from test_supported_role_overlay_v1 import fixture

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from contact_role_inputs_v1 import cached_jobs, prepare


def test_positive_additions_remain_nonempty_and_are_a_subset():
    gray, previous, baseline, local, heads = fixture()
    old = old_analyze(gray, previous, baseline, local, heads=heads)
    new = model.analyze(gray, previous, baseline, local, heads=heads)
    assert new['added_pixels'] > 0 and new['added_pixels'] <= old['added_pixels']
    assert np.all((new['pixel_roles'] == baseline) | (new['pixel_roles'] == old['pixel_roles']))
    assert new['additions_are_subset_of_previous_overlay']
    assert new['dimension_chains'] == old['dimension_chains']


@pytest.mark.parametrize('mode,wrong', [('native_crop', 8), ('half_resolution_crop', 10)])
def test_known_part_contact_pixels_return_to_unknown_without_label_input(mode, wrong):
    job = next(j for j in cached_jobs(ROOT) if j['case']['id'] == 'train-2-upper')
    view = next(v for v in job['views'] if v['mode'] == mode)
    gray, source, geometry, baseline, text, previous, local, heads, extra, prep = prepare(ROOT, view)
    result = model.analyze(gray, previous, baseline, local, heads=heads, diagnostic_heads=extra)
    review = json.loads((ROOT/'artifacts/supported-role-overlay-v1/contact-review.json').read_text('utf-8'))
    entry = next(e for e in review['entries'] if e['mode'] == mode)
    assert len(entry['pixels']) == wrong
    native = to_native(result['pixel_roles'], job['case']['size_px'])
    for pixel in entry['pixels']:
        x, y = pixel['native_px']
        assert native[y, x] == 255


def test_all_known_baseline_classes_and_text_are_immutable():
    gray, previous, baseline, local, heads = fixture()
    baseline[90, 160:169] = np.arange(1, 10, dtype=np.uint8)
    source = gray.copy()
    previous_copy, head_copy = deepcopy(previous), deepcopy(heads)
    result = model.analyze(gray, previous, baseline, local, heads=heads)
    assert np.array_equal(result['pixel_roles'][90, 160:169], np.arange(1, 10))
    assert np.array_equal(result['text_mask'], local['hybrid']['text_mask'])
    assert np.array_equal(gray, source) and previous == previous_copy and heads == head_copy
    assert result['head_ids_used'] == local['hybrid']['head_ids_used']
    assert result['head_events'] == local['hybrid']['head_events']


def test_source_ids_indices_and_geometry_survive_abstention():
    gray, previous, baseline, local, heads = fixture()
    result = model.analyze(gray, previous, baseline, local, heads=heads)
    for a, b in zip(previous['proposals'], result['proposals']):
        for key in ('id', 'path_id', 'source_index_interval', 'points_px'):
            assert a[key] == b[key]
    assert all(not p['semantic_ownership_confirmed'] for p in result['proposals'])


def test_missing_contact_boundary_abstains_without_inventing_part_roles():
    gray, previous, baseline, local, heads = fixture()
    previous['boundaries'] = {'boundaries': []}
    result = model.analyze(gray, previous, baseline, local, heads=heads)
    assert not np.any((result['pixel_roles'] == 3) & (baseline == 255))
    assert not np.any(result['pixel_roles'] == 1)
    assert not result['search_incomplete']


def test_competing_boundary_support_blocks_tangent_pixels_far_from_nominal_foot():
    context = {'_competitor_points': np.array([[1., -20.], [2., -21.]]),
               '_competitor_radii': np.array([4., 4.])}
    audit = {'competitor_pair_checks': 0, 'limits': set()}
    keep = model.outside_competitor_support(np.array([[0., -20.], [20., -20.]]), context, .5, audit)
    assert keep.tolist() == [False, True]
    assert audit['competitor_pair_checks'] > 0 and not audit['limits']


def test_competitor_band_scales_with_observed_stroke_width():
    points = np.array([[0., 5.], [0., 8.], [20., 0.]])
    outcomes = []
    for radius in (4., 8.):
        context = {'_competitor_points': np.array([[0., 0.]]), '_competitor_radii': np.array([radius])}
        outcomes.append(model.outside_competitor_support(points, context, .5, {'competitor_pair_checks': 0, 'limits': set()}))
    assert outcomes[0].tolist() == [False, True, True]
    assert outcomes[1].tolist() == [False, False, True]


@pytest.mark.parametrize('key', ['maximum_combined_source_points', 'maximum_combined_raster_pixels', 'maximum_competitor_pairs'])
def test_every_extra_work_budget_exposes_incompleteness_and_preserves_baseline(key):
    gray, previous, baseline, local, heads = fixture()
    if key == 'maximum_competitor_pairs':
        context = {'_competitor_points': np.array([[0., 0.]]), '_competitor_radii': np.array([8.])}
        audit = {'competitor_pair_checks': 0, 'limits': set()}
        with patch.dict(model.POLICY, {key: 0}):
            keep = model.outside_competitor_support(np.array([[0., 0.]]), context, .5, audit)
        assert not keep.any() and 'joint_competitor_pair_budget' in audit['limits']
    else:
        with patch.dict(model.POLICY, {key: 0}):
            result = model.analyze(gray, previous, baseline, local, heads=heads)
        assert result['search_incomplete'] and result['contact_guard_audit']['limits']
        assert np.array_equal(result['pixel_roles'], baseline) and not result['pixel_interval_owner'].any()


def test_unknown_junction_contexts_do_not_delete_head_or_chain_observations():
    gray, previous, baseline, local, heads = fixture()
    extra = deepcopy(local['hybrid']['dimension_chains'][0])
    extra['feet_px'][0][0] += 2
    local['hybrid']['dimension_chains'].append(extra)
    result = model.analyze(gray, previous, baseline, local, heads=heads)
    assert result['dimension_chains'] == local['hybrid']['dimension_chains']
    assert result['head_ids_used'] == local['hybrid']['head_ids_used']
    assert any(p.get('contact_guard', {}).get('reason') == 'ambiguous_or_unobserved_contact_context'
               for p in result['interval_admission'])


@pytest.mark.parametrize('kwargs', [{'leader': True}, {'broken': True}])
def test_unconfirmed_chain_never_gains_contact_authority(kwargs):
    gray, previous, baseline, local, heads = fixture(**kwargs)
    result = model.analyze(gray, previous, baseline, local, heads=heads)
    assert not result['dimension_chains'] and result['added_pixels'] == 0


def test_no_erasure_physical_or_CAD_admission_and_serializable_audit():
    gray, previous, baseline, local, heads = fixture()
    result = model.analyze(gray, previous, baseline, local, heads=heads)
    assert result['source_pixels_erased'] == result['semantic_CAD_promotions'] == 0
    assert not result['production_admitted'] and not result['physical_geometry_approved']
    audit = result['contact_guard_audit']
    assert audit['combined_source_points'] <= 2000000 and audit['combined_raster_pixels'] <= 40000000
    assert audit['competitor_pair_checks'] <= 2000000
    json.dumps({k:v for k,v in result.items() if not isinstance(v,np.ndarray)}, allow_nan=False)
