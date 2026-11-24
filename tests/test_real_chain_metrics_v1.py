from copy import deepcopy
import numpy as np
import pytest

from cad_recovery.real_chain_metrics_v1 import assignment, head_metrics, chain_metrics, transformed_heads

IDENTITY = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def h(identity, x, y=20, direction=(1., 0.)):
    return {'id': identity, 'tip_px': [x, y], 'direction': list(direction)}


def review(heads, boxes=None):
    return {'heads': heads, 'head_scopes': boxes or [[0, 0, 100, 100]], 'chains': []}


def test_matching_maximizes_cardinality_before_distance():
    result = assignment(2, 2, {(0, 0): 0., (0, 1): 1., (1, 0): .5})
    assert set(map(tuple, result['pairs'])) == {(0, 1), (1, 0)}


def test_matching_minimizes_total_error_after_rerouting():
    result = assignment(2, 2, {(0, 0): 0., (0, 1): 1., (1, 0): .1, (1, 1): 10.})
    assert set(map(tuple, result['pairs'])) == {(0, 1), (1, 0)}


def test_budget_failure_never_returns_a_partial_success():
    result = assignment(2, 2, {(0, 0): 0., (1, 1): 0.}, maximum_visits=1)
    assert result['incomplete'] and not result['pairs']


def test_a_candidate_cannot_satisfy_two_targets():
    result = head_metrics(review([h('a', 20), h('b', 21)]), [h('single', 20)], IDENTITY)
    assert (result['tp'], result['fn'], result['fp']) == (1, 1, 0)


def test_coincident_opposite_heads_require_distinct_directed_observations():
    refs = [h('a', 20), h('b', 20, direction=(-1., 0.))]
    result = head_metrics(review(refs), [h('right', 20), h('left', 20, direction=(-1., 0.))], IDENTITY)
    assert (result['tp'], result['fn'], result['fp']) == (2, 0, 0)
    bad = head_metrics(review(refs), [h('one', 20), h('wrong', 20)], IDENTITY)
    assert (bad['tp'], bad['fn'], bad['fp']) == (1, 1, 1)


def test_half_pixel_centre_translation_and_signed_direction():
    matrix = [[2, 0, .5], [0, 2, .5], [0, 0, 1]]
    candidate = h('candidate', 10, 20)
    result = head_metrics(review([h('target', 20.5, 40.5)]), [candidate], matrix)
    assert result['tp'] == 1 and result['pairs'][0]['tip_error_native_px'] == 0.


def test_only_explicit_review_scopes_count_false_heads():
    result = head_metrics(review([], [[0, 0, 10, 10]]), [h('inside', 5, 5), h('outside', 20)], IDENTITY)
    assert result['fp'] == 1 and result['outside_scope_candidates_unscored'] == 1
    assert result['full_page_precision'] is None


@pytest.mark.parametrize('candidate', [h('bad', float('nan')), h('bad', 20, direction=(0., 0.))])
def test_invalid_observations_are_rejected(candidate):
    with pytest.raises(ValueError): transformed_heads([candidate], IDENTITY)


def test_reference_outside_review_scope_is_rejected():
    with pytest.raises(ValueError): head_metrics(review([h('outside', 20)], [[0, 0, 10, 10]]), [], IDENTITY)


def chain_fixture():
    heads = [h('left', 20, direction=(-1., 0.)), h('right', 80)]
    target = {'id': 'dimension', 'head_ids': ['left', 'right'], 'shaft': [[30, 20], [70, 20]],
              'feet_px': [[20, 60], [80, 60]], 'head_scope': [10, 10, 90, 30],
              'topology': 'two_extensions', 'reference_kinds': ['contour', 'contour'], 'whole_part_contour_visible': False}
    r = review(heads); r['chains'] = [target]
    previous = {'tracks': [{'id': 'shaft', 'points': [[20, 20], [80, 20]]}]}
    packet = {'dimension_chains': [{'head_ids': ['left', 'right'], 'shaft_track': 'shaft', 'feet_px': [[80, 60], [20, 60]]}]}
    trace = {'contacts': {'shaft': ['left', 'right']}, 'eligible_boundaries': []}
    metrics = head_metrics(r, heads, IDENTITY)
    return r, heads, IDENTITY, packet, previous, trace, metrics


def test_chain_requires_geometry_and_all_feet_not_just_matching_heads():
    args = chain_fixture(); good = chain_metrics(*args)
    assert (good['tp'], good['fp'], good['fn']) == (1, 0, 0)
    args[3]['dimension_chains'][0]['feet_px'][0] = [80, 90]
    bad = chain_metrics(*args)
    assert (bad['tp'], bad['fp'], bad['fn']) == (0, 1, 1)


def test_duplicate_chain_is_a_scoped_false_proposal():
    args = chain_fixture(); args[3]['dimension_chains'] *= 2
    result = chain_metrics(*args)
    assert (result['tp'], result['fp'], result['fn']) == (1, 1, 0)


def test_missing_head_and_missing_common_shaft_are_separate_diagnostics():
    args = chain_fixture(); args[3]['dimension_chains'] = []
    args[-1]['pairs'].pop()
    assert chain_metrics(*args)['diagnostics'][0]['first_unmet_observation'] == 'missing_detected_head'
    args = chain_fixture(); args[3]['dimension_chains'] = []; args[5]['contacts'] = {}
    assert chain_metrics(*args)['diagnostics'][0]['first_unmet_observation'] == 'no_common_head_to_shaft_contact'


def test_unreviewed_chain_proposals_are_not_automatically_false():
    args = chain_fixture(); args[0]['chains'] = []
    result = chain_metrics(*args)
    assert result['fp'] == 0 and result['outside_scope_proposals_unscored'] == 1
    assert result['precision'] is None and result['full_page_precision'] is None


def test_diagnostics_never_modify_source_observations_or_targets():
    args = chain_fixture(); before = deepcopy(args)
    chain_metrics(*args)
    assert args == before


def test_matching_tolerance_is_in_native_coordinates():
    result = head_metrics(review([h('target', 20)]), [h('too_far', 28.01)], IDENTITY)
    assert result['tp'] == 0 and result['fp'] == 1 and result['fn'] == 1
