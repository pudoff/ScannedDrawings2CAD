from copy import deepcopy
import cv2
import numpy as np
import pytest
from cad_recovery.curved_source_heads_v1 import search,POLICY


def drawing(tail=True):
    gray=np.full((180,240),255,np.uint8)
    cv2.fillConvexPoly(gray,np.array([[150,90],[120,81],[120,99]],np.int32),0)
    if tail:cv2.line(gray,(112,90),(150,90),0,3)
    return gray


def test_short_stem_is_detected_without_Hough_or_fitted_lines():
    r=search(drawing());assert r['complete'] and len(r['guarded'])==1
    h=r['guarded'][0];assert np.linalg.norm(np.array(h['tip_px'])-[150,90])<4
    assert h['evidence']['source_path']['source_only'] and not h['evidence']['source_path']['semantic_arrow_confirmed']


def test_detached_filled_triangle_is_not_a_new_binding_head():
    r=search(drawing(False));assert r['complete'] and r['proposals'] and not r['guarded'] and not r['path_only']


def test_prior_observation_prefix_and_source_pixels_are_exact():
    gray=drawing();h=search(gray)['guarded'][0];h['id']='prior';before=deepcopy(h);pixels=gray.copy()
    r=search(gray,baseline=[h]);assert r['guarded']==r['path_only']==r['shape_only']==[before]
    assert r['raw_observations']==[before] and h==before and np.array_equal(gray,pixels)


def test_previously_excluded_observation_is_not_reintroduced_under_a_new_id():
    gray=drawing();h=search(gray)['guarded'][0];h['id']='excluded-before'
    r=search(gray,prior_observations=[h]);assert not r['proposals'] and not r['guarded']


def test_curved_tail_is_followed_in_source_coordinates():
    gray=drawing(False);cv2.polylines(gray,[np.array([[120,90],[105,90],[90,100],[75,125],[50,145]])],False,0,3)
    r=search(gray);assert len(r['guarded'])==1
    path=r['guarded'][0]['evidence']['source_path'];assert path['supported']
    assert len({p[1] for p in path['path_px']})>5 and all(gray[y,x]==0 for x,y in path['path_px'])


def test_only_a_loop_does_not_suffice_for_a_filled_triangle():
    gray=np.full((200,240),255,np.uint8);cv2.circle(gray,(120,100),45,0,3)
    r=search(gray);assert r['complete'] and not r['guarded']


def test_budget_failure_does_not_promote_partial_hypotheses(monkeypatch):
    monkeypatch.setitem(POLICY,'maximum_profile_samples',0);r=search(drawing())
    assert not r['complete'] and not r['guarded'] and r['audit']['limits']


def test_no_CAD_physical_or_service_approval_is_given():
    r=search(drawing());assert not r['production_admitted'] and not r['physical_geometry_approved'] and not r['source_pixels_erased']


def test_duplicate_baseline_ids_are_rejected():
    gray=drawing();h=search(gray)['guarded'][0]
    with pytest.raises(ValueError):search(gray,baseline=[h,h])
