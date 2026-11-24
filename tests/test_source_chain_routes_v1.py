from copy import deepcopy
import cv2
import numpy as np
import pytest

from cad_recovery.source_chain_routes_v1 import Routes
from cad_recovery.local_chain_evidence_v2 import verify
from cad_recovery.local_chain_evidence_v1 import verify as legacy


def segment(key,a,b):
    a,b=np.asarray(a,float),np.asarray(b,float);n=max(1,int(np.ceil(np.linalg.norm(b-a))))
    return {'id':key,'points_px':(a+np.linspace(0,1,n+1)[:,None]*(b-a)).tolist()}


def junction(gap=False):
    ink=np.zeros((64,64),np.uint8)
    cv2.line(ink,(12,12),(12,32),255,3);cv2.line(ink,(12,32),(48,32),255,3)
    head={'id':'h','tip_px':[12,32],'polygon_px':[[12,32],[20,27],[20,37]],'direction':[-1,0],'uncertainty_px':0}
    cv2.fillConvexPoly(ink,np.array(head['polygon_px'],np.int32),255)
    if gap:ink[26:29,8:16]=0
    items=[segment('shaft',[22,32],[48,32]),segment('ext',[12,12],[12,25]),segment('stub',[12,25],[12,32])]
    shaft={'width':3.,'interval_ids':['shaft']};ext={'width':3.,'interval_ids':['ext'],'points':[[12,12],[12,25]]}
    return ink,items,shaft,ext,[head]


def test_junction_requires_actual_connected_ink():
    ink,items,shaft,ext,heads=junction();routes=Routes(items,ink)
    hit,evidence=routes.intersection([[12,32],[48,32]],ext,2.,shaft,heads)
    assert hit is not None and evidence['method']=='connected_source_ink_in_local_junction'
    assert ink[round(hit[1]),round(hit[0])]>0


def test_white_gap_near_head_is_not_filled():
    ink,items,shaft,ext,heads=junction(True);routes=Routes(items,ink)
    hit,evidence=routes.intersection([[12,32],[48,32]],ext,2.,shaft,heads)
    assert hit is None and evidence is None


def foot_packet(gap=False):
    ink=np.zeros((64,64),np.uint8)
    for a,b in [([20,4],[20,25]),([20,25],[20,31]),([20,31],[50,31])]:cv2.line(ink,tuple(a),tuple(b),255,3)
    if gap:ink[27:29,17:24]=0
    items=[segment('ext',[20,4],[20,25]),segment('bridge',[20,25],[20,31]),segment('part',[20,31],[50,31])]
    ext={'points':[[20,4],[20,25]],'interval_ids':['ext'],'u':[0.,1.],'width':3.}
    return ink,items,ext,{'owners':['part']}


def test_foot_can_follow_a_short_observed_source_route():
    ink,items,ext,boundary=foot_packet();r=Routes(items,ink)
    found=r.feet(ext,boundary,np.array([20,4]),set())
    assert len(found)==1 and found[0][1]==['bridge'] and np.array_equal(found[0][0],[20,31])


def test_foot_cannot_jump_a_white_gap():
    ink,items,ext,boundary=foot_packet(True)
    assert not Routes(items,ink).feet(ext,boundary,np.array([20,4]),set())


def test_route_cannot_walk_an_unrelated_shaft():
    ink,items,ext,boundary=foot_packet()
    assert not Routes(items,ink).feet(ext,boundary,np.array([20,4]),{'bridge'})


def test_route_length_is_bounded_by_observed_width():
    ink,items,ext,boundary=foot_packet();ext['width']=1.
    assert not Routes(items,ink).feet(ext,boundary,np.array([20,4]),set())


def test_source_inputs_are_not_mutated():
    ink,items,ext,boundary=foot_packet();old=deepcopy(items);raster=ink.copy()
    Routes(items,ink).feet(ext,boundary,np.array([20,4]),set())
    assert items==old and np.array_equal(ink,raster)


def test_foreign_or_duplicate_ids_are_rejected():
    ink,items,ext,boundary=foot_packet()
    with pytest.raises(ValueError):Routes(items+[items[0]],ink)


def test_off_source_interval_does_not_become_an_edge():
    ink,items,ext,boundary=foot_packet();items[1]['points_px'][3]=[25,28]
    assert not Routes(items,ink).feet(ext,boundary,np.array([20,4]),set())


def test_preserves_legacy_chains_without_forcing_two_new_heads():
    # An empty well-formed input remains an exact no-chain fallback.
    ink=np.zeros((32,32),np.uint8)
    a=legacy([],[],[],[],ink);b=verify([],[],[],[],ink)
    assert b['chains']==a['chains'] and b['leaders']==a['leaders'] and not b['incomplete']
