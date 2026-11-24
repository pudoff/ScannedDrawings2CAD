from copy import deepcopy
import json
import cv2
import numpy as np
import pytest
from cad_recovery.head_junction_ownership_v1 import apply


def fixture(pen=3):
    gray=np.full((128,160),255,np.uint8)
    cv2.line(gray,(30,40),(130,40),0,pen);cv2.line(gray,(30,15),(30,95),0,3)
    head={'id':'h','polygon_px':[[30,40],[48,34],[48,46]],'direction':[-1.,0.],'uncertainty_px':1.5}
    cv2.fillConvexPoly(gray,np.array(head['polygon_px'],np.int32),0)
    def interval(key,p):return {'id':key,'points_px':p,'path_id':key,'source_index_interval':[0,len(p)-1]}
    previous={'proposals':[interval('shaft',[[x,40] for x in range(52,131)]),interval('ext',[[30,y] for y in range(15,96)])],
              'tracks':[{'id':'t','points':[[52,40],[130,40]]}]}
    inherited=np.full(gray.shape,255,np.uint8);labels=inherited.copy();labels[15:96,30]=3
    owner=np.zeros(gray.shape,np.uint16);owner[15:96,30]=2
    packet={'pixel_roles':labels,'pixel_interval_owner':owner,'dimension_chains':[{'shaft_track':'t','shaft_interval_ids':['shaft'],
             'extension_interval_ids':[['ext']],'head_ids':['h']}], 'search_incomplete':False,'source_pixels_erased':0}
    return gray,previous,inherited,packet,[head]


def test_withdraws_conflicting_extension_only_near_the_head():
    args=fixture();r=apply(*args)
    assert r['pixel_roles'][40,30]==255 and r['pixel_roles'][70,30]==3
    assert 0<r['junction_ownership_audit']['removed_pixels']<20
    assert r['new_role_changes_are_withdrawals'] and not r['search_incomplete']


def test_all_inherited_roles_and_head_observations_are_immutable():
    gray,previous,inherited,packet,heads=fixture();inherited[40,30]=packet['pixel_roles'][40,30]=3
    snap=deepcopy(previous);old_heads=deepcopy(heads);old_pixels=gray.copy();old_map=packet['pixel_roles'].copy()
    r=apply(gray,previous,inherited,packet,heads)
    assert r['pixel_roles'][40,30]==3 and np.array_equal(packet['pixel_roles'],old_map)
    assert previous==snap and heads==old_heads and np.array_equal(gray,old_pixels)
    assert r['dimension_chains']==packet['dimension_chains']


def test_shaft_radius_follows_observed_pen_width():
    a=apply(*fixture(3));b=apply(*fixture(7))
    ar=a['junction_decisions'][0]['contexts'][0]['observed_shaft_radius_px']
    br=b['junction_decisions'][0]['contexts'][0]['observed_shaft_radius_px']
    assert br>ar and b['junction_ownership_audit']['removed_pixels']>=a['junction_ownership_audit']['removed_pixels']


def test_uncertainty_follows_observed_head_metadata():
    args=fixture();a=apply(*args);args[-1][0]['uncertainty_px']=4.
    b=apply(*args)
    assert b['junction_ownership_audit']['removed_pixels']>a['junction_ownership_audit']['removed_pixels']


def test_other_new_classes_are_not_changed():
    gray,p,b,packet,h=fixture();packet['pixel_roles'][40,80]=2;packet['pixel_interval_owner'][40,80]=1
    r=apply(gray,p,b,packet,h);assert r['pixel_roles'][40,80]==2


def test_missing_chain_context_abstains_without_inventing_a_role():
    gray,p,b,packet,h=fixture();packet['dimension_chains']=[]
    r=apply(gray,p,b,packet,h)
    assert r['added_pixels']==0 and r['junction_ownership_audit']['removed_pixels']==81


def test_no_changed_pixels_has_an_empty_finite_audit():
    gray,p,b,packet,h=fixture();packet['pixel_roles'][:]=b;packet['pixel_interval_owner'][:]=0
    r=apply(gray,p,b,packet,h)
    assert r['added_pixels']==0 and r['junction_ownership_audit']['polygon_checks']==0


def test_budget_failure_reverts_to_inherited_and_exposes_incompleteness():
    gray,p,b,packet,h=fixture();packet['contact_audit']={'combined_raster_pixels':40000000}
    r=apply(gray,p,b,packet,h)
    assert r['search_incomplete'] and np.array_equal(r['pixel_roles'],b)
    assert r['junction_ownership_audit']['limits']


@pytest.mark.parametrize('damage',['no_owner','bad_map','duplicate_head','nan_head','off_source','changed_known'])
def test_malformed_contracts_do_not_receive_silent_approval(damage):
    gray,p,b,packet,h=fixture()
    if damage=='no_owner':packet['pixel_interval_owner'][40,30]=0
    elif damage=='bad_map':packet['pixel_roles']=packet['pixel_roles'][:20]
    elif damage=='duplicate_head':h.append(deepcopy(h[0]))
    elif damage=='nan_head':h[0]['uncertainty_px']=float('nan')
    elif damage=='off_source':p['proposals'][0]['points_px'][0]=[-1,40]
    elif damage=='changed_known':b[40,30]=2
    with pytest.raises(ValueError):apply(gray,p,b,packet,h)


def test_audit_is_json_and_never_approves_physical_geometry():
    r=apply(*fixture());json.dumps(r['junction_ownership_audit'],allow_nan=False);json.dumps(r['junction_decisions'],allow_nan=False)
    assert not r['physical_geometry_approved'] and not r['production_admitted']
