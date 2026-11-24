from copy import deepcopy
import json
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from cad_recovery import head_support_ownership_v1 as guard


def head(identity, tip, unit, length=26., half_width=8., shaft=2., band=0., extent=10.):
    tip, unit = np.array(tip, float), np.array(unit, float); normal = np.array([-unit[1], unit[0]])
    base = tip-length*unit
    return {'id': identity, 'tip_px': tip.tolist(), 'direction': unit.tolist(),
        'polygon_px': np.array([tip, base+half_width*normal, base-half_width*normal]).tolist(),
        'classification': 'filled_arrow_candidate', 'confidence': .55, 'uncertainty_px': 1.,
        'evidence': {'source_checked': True, 'local_shaft_width_px': shaft,
                     'shaft_observed_extent_px': extent, 'profile_band_px': [band, band+length]}}


def packet(baseline, added, *, incomplete=False):
    return {'candidates': deepcopy(baseline+added), 'added': deepcopy(added), 'proposals': deepcopy(added),
        'context_exclusions': [], 'search_incomplete': incomplete, 'production_admitted': False,
        'physical_geometry_approved': False, 'CAD_promotions': 0,
        'analysis': {'limits': ['profile_sample_budget'] if incomplete else [],
                     'coverage_complete': not incomplete, 'accepted_additions': len(added)}}


def draw_head(ink, value): cv2.fillConvexPoly(ink, np.rint(value['polygon_px']).astype(np.int32), 255)


def ghost_scene():
    ink = np.zeros((180, 220), np.uint8)
    owner = head('owner', [80, 90], [-1, 0], length=38, half_width=11, shaft=2)
    draw_head(ink, owner); cv2.line(ink, (118,90), (180,90), 255, 2)
    ghost = head('ghost', [120,90], [1,0], length=10, half_width=3, shaft=1, band=18, extent=10)
    return 255-ink, owner, ghost


def test_source_owned_reverse_candidate_abstains_without_erasing_observation():
    gray, owner, ghost = ghost_scene(); raw = packet([owner], [ghost]); saved = deepcopy(raw); source = gray.copy()
    result = guard.apply(gray, raw, baseline=[owner])
    assert result['candidates'] == [owner] and not result['added']
    decision = result['ownership_exclusions'][0]
    assert decision['candidate_id'] == 'ghost' and decision['relations'][0]['owner_id'] == 'owner'
    assert decision['relations'][0]['owned_fraction'] == 1 and decision['observation_retained']
    assert result['proposals'] == [ghost] and raw == saved and np.array_equal(gray, source)
    json.dumps(result)


@pytest.mark.parametrize('angle', [0, 90, 180, 270])
def test_close_real_opposed_heads_keep_their_independent_shared_shaft(angle):
    ink = np.zeros((180,180),np.uint8)
    left = head('left',[52,90],[-1,0],length=20,half_width=6,extent=10)
    right = head('right',[104,90],[1,0],length=20,half_width=6,extent=10)
    draw_head(ink,left);draw_head(ink,right);cv2.line(ink,(72,90),(84,90),255,2)
    rotation = np.array([[np.cos(np.deg2rad(angle)),-np.sin(np.deg2rad(angle))],
                         [np.sin(np.deg2rad(angle)),np.cos(np.deg2rad(angle))]])
    matrix = np.c_[rotation,np.array([90,90])-rotation@np.array([90,90])]
    image=cv2.warpAffine(ink,matrix,(180,180),flags=cv2.INTER_NEAREST)
    for h in (left,right):
        h['polygon_px']=((np.array(h['polygon_px'])-90)@rotation.T+90).tolist()
        h['direction']=(rotation@h['direction']).tolist();h['tip_px']=h['polygon_px'][0]
    r=guard.apply(255-image,packet([left],[right]),baseline=[left])
    assert r['candidates']==[left,right] and not r['ownership_exclusions']


def test_nearby_same_direction_head_does_not_gain_an_owner_by_distance():
    gray, owner, ghost = ghost_scene();ghost['direction']=[-1.,0.]
    r=guard.apply(gray,packet([owner],[ghost]),baseline=[owner])
    assert r['added']==[ghost] and not r['ownership_exclusions']


def test_missing_or_background_owner_evidence_cannot_exclude_a_new_observation():
    gray, owner, ghost = ghost_scene();owner['evidence']['source_checked']=False
    assert guard.apply(gray,packet([owner],[ghost]),baseline=[owner])['added']==[ghost]
    owner['evidence']['source_checked']=True
    assert guard.apply(np.full_like(gray,255),packet([owner],[ghost]),baseline=[owner])['added']==[ghost]


def test_proposals_with_unknown_shaft_metadata_are_retained_for_review():
    gray, owner, ghost = ghost_scene();del ghost['evidence']['profile_band_px']
    r=guard.apply(gray,packet([owner],[ghost]),baseline=[owner])
    assert r['added']==[ghost] and r['ownership_decisions'][0]['reason']=='no_reobserved_witness_retained'


def test_baseline_is_never_subject_to_the_new_rejection():
    gray, owner, ghost = ghost_scene()
    assert guard.apply(gray,packet([owner,ghost],[]),baseline=[owner,ghost])['candidates']==[owner,ghost]


@pytest.mark.parametrize('key,limit', [('maximum_source_samples',0),('maximum_pair_checks',0),
    ('maximum_owner_validation_pixels',0),('maximum_containment_points',0)])
def test_every_guard_budget_is_explicit_and_preserves_unchecked_observations(key,limit):
    gray, owner, ghost=ghost_scene()
    with patch.dict(guard.POLICY,{key:limit}):r=guard.apply(gray,packet([owner],[ghost]),baseline=[owner])
    assert r['ownership_incomplete'] and r['search_incomplete'] and r['added']==[ghost]
    assert r['analysis']['ownership']['limits'] and not r['ownership_exclusions']


def test_inherited_incomplete_search_is_not_promoted_by_successful_guard():
    gray, owner, ghost=ghost_scene();r=guard.apply(gray,packet([owner],[ghost],incomplete=True),baseline=[owner])
    assert r['search_incomplete'] and r['source_search_incomplete'] and not r['ownership_incomplete']
    assert 'profile_sample_budget' in r['analysis']['limits'] and not r['analysis']['coverage_complete']


def test_new_ownership_is_order_independent():
    gray,owner,ghost=ghost_scene()
    owner['evidence'].update(profile_band_px=[28,66],shaft_observed_extent_px=6)
    a=guard.apply(gray,packet([], [owner,ghost]));b=guard.apply(gray,packet([], [ghost,owner]))
    assert {h['id'] for h in a['candidates']}=={h['id'] for h in b['candidates']}
    assert {d['candidate_id']:d['excluded'] for d in a['ownership_decisions']}=={
        d['candidate_id']:d['excluded'] for d in b['ownership_decisions']}


def test_mutual_ownership_cycle_cannot_silently_delete_both_observations():
    relations={'a':[{'owner_id':'b'}],'b':[{'owner_id':'a'}]}
    excluded,unresolved=guard._resolve(relations,set())
    assert not excluded and unresolved=={'a','b'}


def test_rejected_hypothesis_cannot_reserve_source_support_for_another_rejection():
    relations={'a':[{'owner_id':'b'}],'b':[{'owner_id':'baseline'}]}
    excluded,unresolved=guard._resolve(relations,{'baseline'})
    assert set(excluded)=={'b'} and not unresolved


def test_fresh_pipeline_matches_guard_on_the_same_frozen_source_detector():
    from cad_recovery.local_head_context_v2 import search as raw_search
    gray,owner,_=ghost_scene()
    raw=raw_search(gray,baseline=[owner])
    assert guard.search(gray,baseline=[owner])==guard.apply(gray,raw,baseline=[owner])
