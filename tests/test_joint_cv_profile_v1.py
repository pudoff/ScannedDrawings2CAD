from copy import deepcopy
import cv2
import numpy as np
import pytest
from cad_recovery.head_width_contract_v1 import normalize,POLICY
from cad_recovery.joint_cv_profile_v1 import analyze,chain_authority
from cad_recovery.joint_role_metrics_v1 import score


def fixture():
    gray=np.full((100,160),255,np.uint8);gray[49:52,20:81]=0
    cv2.fillConvexPoly(gray,np.array([[110,50],[80,42],[80,58]],np.int32),0)
    h={'id':'new','tip_px':[110.,50.],'polygon_px':[[110.,50.],[80.,42.],[80.,58.]],'direction':[1.,0.],
        'uncertainty_px':2.,'evidence':{'method':'curved-source-heads/1.0','local_shaft_width_px':9.,
        'observed_path_width_px':999.,'source_path':{'supported':True,'source_only':True,'path_px':[[30,50],[31,50],[32,50]]}}}
    return gray,h


def test_width_is_measured_again_in_source_and_not_copied_from_bootstrap():
    gray,h=fixture();r=normalize(gray,[h]);e=r['normalized_heads'][0]['evidence']
    assert e['initial_shaft_width_px']==9 and e['local_shaft_width_px']==e['observed_path_width_px']==3
    assert e['width_contract']['measurement']['samples']==3 and r['role_heads']==r['normalized_heads']


def test_closed_route_without_samples_has_no_observed_width_or_binding_admission():
    gray,h=fixture();h['evidence']['source_path']={'supported':True,'source_only':True,'closed_source_routes':1}
    r=normalize(gray,[h]);e=r['normalized_heads'][0]['evidence']
    assert e['observed_path_width_px'] is None and e['local_shaft_width_px'] is None and not r['role_heads']
    assert r['raw_observations']==[h] and e['width_contract']['status']=='unavailable_no_source_samples'


@pytest.mark.parametrize('points',[[[30,20],[31,20]],[[30,50],[35,50]],[[999,50]],[[30.5,50]]])
def test_white_disconnected_outside_and_subpixel_samples_cannot_supply_width(points):
    gray,h=fixture();h['evidence']['source_path']['path_px']=points;r=normalize(gray,[h])
    assert not r['role_heads'] and r['normalized_heads'][0]['evidence']['observed_path_width_px'] is None


def test_raw_geometry_uncertainty_source_pixels_and_inherited_contract_remain_exact():
    gray,h=fixture();before=deepcopy(h);pixels=gray.copy();r=normalize(gray,[h])
    for k in ('id','tip_px','polygon_px','direction','uncertainty_px'):assert r['normalized_heads'][0][k]==h[k]
    assert h==before and np.array_equal(gray,pixels) and normalize(gray,[h],immutable_ids=['new'])['normalized_heads']==[h]


def test_budget_failure_withdraws_all_novel_binding_and_is_explicit(monkeypatch):
    gray,h=fixture();monkeypatch.setitem(POLICY,'maximum_path_samples',0);r=normalize(gray,[h])
    assert not r['complete'] and not r['role_heads'] and r['audit']['limits']


def test_absent_positive_class_with_false_predictions_is_in_macro_F1():
    truth=np.array([[1,1]],np.uint8);prediction=np.array([[1,2]],np.uint8);r=score(truth,prediction)
    assert len(r['roles'])==2 and r['roles'][1]['support']==0 and r['roles'][1]['fp']==1
    assert r['macro_f1']==pytest.approx(1/3)


def test_unknown_truth_is_unscored_until_explicit_negative_ink_is_supplied():
    t=np.array([[1,255]],np.uint8);p=np.array([[1,2]],np.uint8)
    assert score(t,p)['macro_f1']==1
    r=score(t,p,np.array([[0,255]],np.uint8));assert r['macro_f1']==.5 and r['false_role_assignments_on_negative_ink_px']==1


def test_shared_role_evaluation_preserves_empty_graph_source_and_chain_prefix():
    gray=np.full((80,80),255,np.uint8);roles=np.full_like(gray,255)
    graph={'proposals':[],'tracks':[],'hatch_bundles':[],'boundaries':{'boundaries':[],'search_incomplete':False},
        'stroke_px':2.,'search_incomplete':False}
    chain={'shaft_track':'prior','head_ids':['old'],'shaft_interval_ids':['old-source'],'extension_interval_ids':[]}
    base={'pixel_roles':roles,'dimension_chains':[chain]};before=deepcopy(graph);r=analyze(gray,graph,base,[],[])
    assert r['shared_role_evaluation'] and r['role_evaluations']==1
    assert r['joint']['dimension_chains']==[chain] and np.array_equal(r['joint']['pixel_roles'],roles) and graph==before
    assert not r['physical_geometry_approved'] and not r['production_admitted']


def chain_fixture():
    graph={'proposals':[{'id':s} for s in ('shaft','left','right','body')],
        'boundaries':{'boundaries':[{'id':'part','owners':['body']}]},'stroke_px':3.}
    chain={'shaft_track':'track','head_ids':['a','b'],'shaft_interval_ids':['shaft'],
        'extension_interval_ids':[['left'],['right']],'boundary_id':'part','feet_px':[[0.,10.],[20.,10.]]}
    return graph,chain


def test_compatible_chain_supplies_source_role_authority_even_with_reversed_foot_order():
    graph,c=chain_fixture();w=deepcopy(c);w['feet_px'].reverse()
    admitted,decisions,owners=chain_authority(graph,[w],[c])
    assert admitted==[c] and owners==[1,2,3,4] and decisions[0]['eligible_for_role_claims']


def test_good_looking_source_chain_without_canonical_witness_grants_no_roles():
    graph,c=chain_fixture();admitted,ds,owners=chain_authority(graph,[],[c])
    assert not admitted and not owners and ds[0]['canonical_witnesses']==0


def test_conflicting_reference_feet_are_diagnostic_and_do_not_grant_role_authority():
    graph,c=chain_fixture();w=deepcopy(c);w['feet_px']=[[0.,40.],[20.,40.]]
    admitted,ds,owners=chain_authority(graph,[w],[c])
    assert not admitted and not owners and ds[0]['maximum_foot_displacement_px']==30


def test_source_claim_shared_with_unconfirmed_chain_abstains_even_if_other_chain_matches():
    graph,c=chain_fixture();other=deepcopy(c);other['head_ids']=['c','d']
    admitted,ds,owners=chain_authority(graph,[c],[c,other])
    assert admitted==[c] and not owners and not ds[1]['eligible_for_role_claims']
