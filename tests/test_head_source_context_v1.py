from copy import deepcopy
import cv2
import numpy as np
import pytest
from cad_recovery.head_source_context_v1 import analyze,POLICY


def sample(kind='short'):
    gray=np.full((240,320),255,np.uint8)
    polygon=np.array([[150,100],[120,90],[120,110]],np.int32)
    cv2.fillConvexPoly(gray,polygon,0)
    if kind=='short':cv2.line(gray,(112,100),(150,100),0,3)
    if kind=='long':cv2.line(gray,(20,100),(150,100),0,3)
    if kind=='curved':cv2.polylines(gray,[np.array([[120,100],[100,100],[80,110],[60,130],[50,160]])],False,0,3)
    head={'id':'raw','tip_px':[150.,100.],'direction':[1.,0.],'polygon_px':polygon.astype(float).tolist(),
        'evidence':{'path_kind':'line','local_shaft_width_px':3.}}
    return gray,[head]


@pytest.mark.parametrize('kind',['short','long','curved','absent'])
def test_short_curved_and_unproven_stems_remain_available(kind):
    gray,heads=sample(kind);r=analyze(gray,heads)
    assert r['combined']==heads and r['complete']
    assert not r['decisions'][0]['semantic_text_recognized']


def test_actual_curved_source_path_does_not_require_arc_metadata():
    gray,heads=sample('curved');r=analyze(gray,heads);path=r['decisions'][0]['source_path']
    assert path['supported'] and path['status']=='observed_curved_or_long_stem'
    assert len({p[1] for p in path['path_px']})>5
    assert all(gray[y,x]==0 for x,y in path['path_px'])


def test_short_support_uses_real_skeleton_endpoint():
    r=analyze(*sample('short'));p=r['decisions'][0]['source_path']
    assert p['supported'] and p['status']=='observed_short_terminal_stem'


def test_immutability_and_no_CAD_or_semantic_admission():
    gray,heads=sample('curved');before=deepcopy(heads);pixels=gray.copy();r=analyze(gray,heads)
    assert heads==before==r['observations']==r['combined'] and np.array_equal(gray,pixels)
    assert not r['production_admitted'] and not r['physical_geometry_approved'] and r['source_pixels_erased']==0
    assert not r['decisions'][0]['eligible_for_CAD']


def test_explicit_arc_label_alone_does_not_create_source_proof():
    gray,heads=sample('absent');heads[0]['evidence']['path_kind']='arc';r=analyze(gray,heads)
    assert not r['decisions'][0]['source_path']['supported'] and r['combined']==heads


def test_budget_failure_reverts_entire_binding_view(monkeypatch):
    gray,heads=sample();monkeypatch.setitem(POLICY,'maximum_roi_pixels',0);r=analyze(gray,heads)
    assert not r['complete'] and r['audit']['limits'] and r['combined']==r['context_only']==heads


def test_empty_input_has_complete_uniform_audit():
    r=analyze(np.full((40,40),255,np.uint8),[])
    assert r['complete'] and not r['combined'] and not r['production_admitted'] and not r['source_pixels_erased']


def test_duplicate_ids_and_nonfinite_or_nonunit_geometry_rejected():
    gray,heads=sample()
    with pytest.raises(ValueError):analyze(gray,heads*2)
    for key,value in [('direction',[2.,0.]),('tip_px',[float('nan'),0.]),('polygon_px',[[0,0],[1,1]])]:
        bad=deepcopy(heads);bad[0][key]=value
        with pytest.raises(ValueError):analyze(gray,bad)


def test_border_component_is_unavailable_for_text_exclusion():
    gray,heads=sample('long');cv2.line(gray,(0,100),(120,100),0,3);r=analyze(gray,heads)
    assert not r['decisions'][0]['component']['complete'] and r['combined']==heads
