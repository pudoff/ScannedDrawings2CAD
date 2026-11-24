from copy import deepcopy
import cv2
import numpy as np
import pytest
from cad_recovery.head_refinement_v1 import refine,POLICY


def fixture():
    gray=np.full((180,240),255,np.uint8)
    cv2.line(gray,(5,80),(80,80),0,3)
    cv2.fillConvexPoly(gray,np.array([[80,80],[40,70],[40,90]],np.int32),0)
    for x in (80,100):cv2.line(gray,(x,25),(x,140),0,3)
    head={'id':'raw-1','tip_px':[96.,80.],'direction':[1.,0.],
        'polygon_px':[[96.,80.],[40.,70.],[40.,90.]],'uncertainty_px':5.,'confidence':.7,
        'evidence':{'path_kind':'line','local_shaft_width_px':3.}}
    tracks=[{'id':str(x),'points':[[x,25],[x,140]]} for x in (80,100)]
    return gray,[head],tracks


def test_two_nearby_junctions_are_resolved_by_actual_base_to_tip_ink():
    args=fixture();r=refine(*args)
    assert r['localized'][0]['tip_px']==[80.,80.]
    d=r['decisions'][0]['localization']
    assert d['status']=='unique_source_connected_junction'
    assert any(x['track_id']=='100' and x['reason']=='interrupted_base_to_junction_ink' for x in d['rejected'])


def test_raw_heads_pixels_tracks_and_ids_remain_exact():
    gray,heads,tracks=fixture();original=deepcopy([heads,tracks]);pixels=gray.copy();r=refine(gray,heads,tracks)
    assert [heads,tracks]==original and r['raw_observations']==heads and np.array_equal(gray,pixels)
    assert r['localized'][0]['id']==heads[0]['id'] and not r['new_head_observations']


def test_own_stem_is_independent_of_tip_localization():
    args=fixture();r=refine(*args)
    assert r['stem_only'][0]['tip_px']==[96.,80.] and r['combined'][0]['tip_px']==[80.,80.]


def test_triangle_without_independent_tail_stays_raw_but_not_in_binding_view():
    gray,heads,tracks=fixture();gray[:]=255
    cv2.fillConvexPoly(gray,np.array([[96,80],[40,70],[40,90]],np.int32),0)
    r=refine(gray,heads,[])
    assert len(r['raw_observations'])==1 and not r['stem_only']
    assert not r['decisions'][0]['semantic_text_class_confirmed']


def test_clipped_tail_is_unavailable_evidence_and_is_preserved():
    gray,heads,_=fixture();heads[0]['polygon_px']=[[35.,80.],[2.,70.],[2.,90.]];heads[0]['tip_px']=[35.,80.]
    r=refine(gray,heads,[])
    assert len(r['stem_only'])==1 and r['decisions'][0]['stem']['reason']=='stem_window_clipped_unavailable'


def test_arc_observations_are_preserved_for_curved_stem_analysis():
    gray,heads,tracks=fixture();heads[0]['evidence']['path_kind']='arc';r=refine(gray,heads,tracks)
    assert r['localized']==r['combined']==heads


def test_coincident_opposed_raw_tips_are_not_merged():
    gray,heads,_=fixture();other=deepcopy(heads[0]);other['id']='opposed';other['direction']=[-1.,0.]
    other['polygon_px']=[[96.,80.],[152.,70.],[152.,90.]]
    r=refine(gray,heads+[other],[])
    assert len(r['localized'])==2 and {h['id'] for h in r['raw_observations']}=={'raw-1','opposed'}


def test_track_endpoint_order_does_not_change_the_localized_tip():
    args=fixture();before=refine(*args)
    args[2].reverse()
    for t in args[2]:t['points'].reverse()
    assert refine(*args)['localized']==before['localized']


def test_budget_exhaustion_reports_incomplete_and_reverts_all_views(monkeypatch):
    args=fixture();monkeypatch.setitem(POLICY,'maximum_track_checks',0);r=refine(*args)
    assert r['search_incomplete'] and r['localized']==r['combined']==args[1]


@pytest.mark.parametrize('bad',[[[0,0],[float('nan'),0]],[[0,0,1],[1,1,1]]])
def test_invalid_track_geometry_is_rejected(bad):
    gray,heads,_=fixture()
    with pytest.raises(ValueError):refine(gray,heads,[{'id':'bad','points':bad}])
