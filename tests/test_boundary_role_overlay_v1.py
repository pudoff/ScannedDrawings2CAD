"""Source ownership, role immutability and explicit abstention for part proposals."""
from copy import deepcopy

import cv2
import numpy as np
import pytest

from cad_recovery.boundary_role_overlay_v1 import analyze, boundary_witnesses


def packet():
    shape = (180, 200); ink = np.zeros(shape, np.uint8)
    body = [[30, 80], [170, 80], [170, 150], [30, 150], [30, 80]]
    cv2.polylines(ink, [np.array(body, np.int32)], False, 255, 7)
    contour = ink.copy()
    definitions = [('body', body), ('shaft', [[30, 35], [170, 35]]),
                   ('left', [[30, 35], [30, 80]]), ('right', [[170, 35], [170, 80]])]
    intervals = []
    for key, points in definitions:
        cv2.polylines(ink, [np.array(points, np.int32)], False, 255, 3 if key != 'body' else 7)
        intervals.append({'id': key, 'path_id': key, 'source_index_interval': [0, len(points)-1],
                          'points_px': points, 'role': 'unknown'})
    previous = {'proposals': intervals, 'search_incomplete': False, 'stroke_px': 3.,
                'boundaries': {'boundaries': [{'id': 'loop', 'owners': ['body'], 'polygon': body[:-1],
                    'orientation': 'bounded_face', 'frame_hypothesis': False}], 'search_incomplete': False}}
    chain = {'boundary_id': 'loop', 'shaft_interval_ids': ['shaft'], 'extension_interval_ids': [['left'], ['right']],
             'feet_px': [[30, 80], [170, 80]]}
    evidence = np.full(shape, 255, np.uint8); evidence[contour > 0] = 1
    text = np.zeros(shape, np.uint8)
    local = {'local': {'pixel_roles': evidence, 'text_mask': text, 'search_incomplete': False},
             'hybrid': {'dimension_chains': [chain], 'search_incomplete': False}}
    baseline = np.full(shape, 255, np.uint8)
    return 255-ink, previous, baseline, local


def test_implicit_closed_boundary_gains_part_pixels():
    args = packet(); result = analyze(*args)
    assert result['added_pixels'] > 100 and not result['search_incomplete']
    assert set(np.unique(result['pixel_roles'])) <= {1, 255}
    assert sum(r['added_pixels'] for r in result['interval_admission']) == result['added_pixels']
    assert not result['semantic_ownership_confirmed'] and not result['production_admitted']


@pytest.mark.parametrize('reason', ['frame', 'no_chain', 'missing_owner', 'shaft_is_boundary', 'feet_missing',
                                   'feet_coincident', 'feet_away', 'incomplete', 'no_local_evidence', 'no_orientation'])
def test_ambiguous_or_unsupported_boundary_abstains(reason):
    gray, prev, base, local = packet(); b = prev['boundaries']['boundaries'][0]; c = local['hybrid']['dimension_chains'][0]
    if reason == 'frame': b['frame_hypothesis'] = True
    elif reason == 'no_chain': local['hybrid']['dimension_chains'] = []
    elif reason == 'missing_owner': b['owners'] = ['missing']
    elif reason == 'shaft_is_boundary': b['owners'] = ['body', 'shaft']
    elif reason == 'feet_missing': c['feet_px'] = []
    elif reason == 'feet_coincident': c['feet_px'] = [[30, 80], [31, 80]]
    elif reason == 'feet_away': c['feet_px'] = [[30, 5], [170, 5]]
    elif reason == 'incomplete': prev['search_incomplete'] = True
    elif reason == 'no_local_evidence': local['local']['pixel_roles'][:] = 255
    elif reason == 'no_orientation': b.pop('orientation')
    result = analyze(gray, prev, base, local)
    assert result['added_pixels'] == 0 and np.array_equal(result['pixel_roles'], base)


@pytest.mark.parametrize('role', [1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_existing_roles_never_change(role):
    gray, prev, base, local = packet(); base[80, 50:100] = role
    result = analyze(gray, prev, base, local)
    assert np.all(result['pixel_roles'][80, 50:100] == role)
    assert result['inherited_known_roles_unchanged']


def test_text_and_heads_are_protected():
    gray, prev, base, local = packet(); local['local']['text_mask'][76:85, 40:70] = 255
    head = {'polygon_px': [[90, 80], [110, 76], [110, 84]]}
    result = analyze(gray, prev, base, local, heads=[head])
    assert np.all(result['pixel_roles'][80, 40:70] == 255)
    assert result['pixel_roles'][80, 100] == 255


def test_pixels_geometry_chains_and_roles_are_not_mutated():
    gray, prev, base, local = packet(); a, b, c, d = gray.copy(), deepcopy(prev), base.copy(), deepcopy(local)
    analyze(gray, prev, base, local)
    assert np.array_equal(gray, a) and prev == b and np.array_equal(base, c)
    assert local['hybrid'] == d['hybrid'] and np.array_equal(local['local']['pixel_roles'], d['local']['pixel_roles'])


def test_all_additions_have_source_ink_and_owned_interval():
    gray, prev, base, local = packet(); result = analyze(gray, prev, base, local)
    added = result['pixel_roles'] != base
    assert not (added & (gray == 255)).any()
    assert np.all(result['pixel_interval_owner'][added] == 1)


def test_other_boundary_membership_is_ambiguous():
    gray, prev, base, local = packet()
    b = deepcopy(prev['boundaries']['boundaries'][0]); b['id'] = 'other'; prev['boundaries']['boundaries'].append(b)
    c = deepcopy(local['hybrid']['dimension_chains'][0]); c['boundary_id'] = 'other'; local['hybrid']['dimension_chains'].append(c)
    assert analyze(gray, prev, base, local)['added_pixels'] == 0


def test_search_budget_abstains_without_truncating_source():
    gray, prev, base, local = packet(); prev['boundaries']['boundaries'] *= 513
    result = analyze(gray, prev, base, local)
    assert result['search_incomplete'] and np.array_equal(result['pixel_roles'], base)


def test_duplicate_interval_identity_is_rejected():
    gray, prev, base, local = packet(); prev['proposals'].append(deepcopy(prev['proposals'][0]))
    with pytest.raises(ValueError, match='Duplicate'): analyze(gray, prev, base, local)


def test_foreign_seed_interval_competes_for_source_ownership():
    gray, prev, base, local = packet()
    prev['proposals'].append({'id': 'unknown-competitor', 'path_id': 'foreign', 'source_index_interval': [0, 1],
                             'points_px': [[60, 80], [120, 80]], 'role': 'unknown'})
    result = analyze(gray, prev, base, local)
    assert np.all(result['pixel_roles'][80, 60:121] == 255)


@pytest.mark.parametrize('damage', ['outside', 'NaN', 'bad_role'])
def test_malformed_inputs_fail(damage):
    gray, prev, base, local = packet()
    if damage == 'outside': prev['proposals'][0]['points_px'][0] = [-1, 80]
    elif damage == 'NaN': prev['proposals'][0]['points_px'][0] = [float('nan'), 80]
    else: base[1, 1] = 50
    with pytest.raises(ValueError): analyze(gray, prev, base, local)
