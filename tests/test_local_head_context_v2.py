from copy import deepcopy
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from cad_recovery import local_head_context_v2 as detector
from cad_recovery.local_head_parallel_field_v2 import check, source_widths
from test_sparse_local_heads_v1 import mask


@pytest.mark.parametrize('angle',[0,90,180,270])
@pytest.mark.parametrize('width',[2,4,6])
def test_area_downsampled_arrow_survives_quarter_turns(width,angle):
    image=np.full((320,320),255,np.uint8)
    base_x=130+2*width
    cv2.fillConvexPoly(image,np.array([[100,160],[base_x,150-width],[base_x,170+width]],np.int32),0)
    cv2.line(image,(base_x,160),(220,160),0,width)
    image=np.rot90(image,angle//90).copy()
    gray=cv2.resize(image,(160,160),interpolation=cv2.INTER_AREA)
    result=detector.search(gray)
    assert len(result['added'])==1 and not result['search_incomplete']
    evidence=result['added'][0]['evidence']
    assert evidence['shaft_observed_extent_px']>=max(6.,2*evidence['local_shaft_width_px'])
    assert evidence['native_head_check']['triangle_ink']>=.85
    assert evidence['native_head_check']['flank_ink']<=.35
    if 'axis_recovery' in evidence:
        assert evidence['axis_recovery']['full_shaft_checks_unchanged']


def test_actual_source_width_is_measured_across_edge_not_its_distance_value():
    ink=np.zeros((100,100),np.uint8);cv2.line(ink,(10,50),(90,50),255,9)
    audit={'context_pixels':0}
    width=source_widths(ink,np.array([12.,45.]),np.array([88.,45.]),3.,audit,100000)
    assert width>4.5 and audit['context_pixels']>0


def test_unrelated_frame_edges_do_not_form_a_hatch_field():
    ink=np.zeros((220,220),np.uint8)
    for x in [20,120,130]:cv2.line(ink,(x,20),(x,200),255,2)
    polygon=[[119,110],[92,102],[92,118]]
    cv2.fillConvexPoly(ink,np.array(polygon,np.int32),255);cv2.line(ink,(55,110),(92,110),255,2)
    head={'polygon_px':polygon,'direction':[1.,0.],'evidence':{'local_shaft_width_px':3}}
    result=check(ink,head,{'context_pixels':0,'context_segments':0})
    assert result['available'] and not result['incomplete']


def test_disjoint_parallel_fragments_need_common_longitudinal_support():
    ink=np.zeros((220,220),np.uint8)
    for x,y in [(45,15),(85,110),(125,15)]:cv2.line(ink,(x,y),(x,y+65),255,2)
    head={'polygon_px':[[92,90],[67,82],[67,98]],'direction':[1.,0.],'evidence':{'local_shaft_width_px':3}}
    cv2.fillConvexPoly(ink,np.array(head['polygon_px'],np.int32),255)
    result=check(ink,head,{'context_pixels':0,'context_segments':0})
    assert result['available'] and not result['incomplete']


def test_local_regular_stain_remains_an_explicit_abstention():
    ink=np.zeros((180,180),np.uint8)
    for x in [42,61,80,99,118]:cv2.line(ink,(x,32),(x+22,150),255,2)
    polygon=[[94,112],[77,85],[92,82]]
    cv2.fillConvexPoly(ink,np.array(polygon,np.int32),255)
    u=np.array(polygon[0])-np.mean(polygon[1:],0);u/=np.linalg.norm(u)
    result=check(ink,{'polygon_px':polygon,'direction':u.tolist(),'evidence':{'local_shaft_width_px':3}},
                 {'context_pixels':0,'context_segments':0})
    assert not result['available'] and not result['incomplete']
    assert result['support']['co_supported_at_head_station']
    gaps=result['support']['spacing_px'];assert max(gaps)<=2*min(gaps)+3


def test_a_head_occluded_stripe_does_not_turn_hatch_continuation_into_own_tail():
    ink=np.zeros((180,180),np.uint8)
    for x in [42,61,99,118]:cv2.line(ink,(x,32),(x+22,150),255,2)
    polygon=[[94,112],[77,85],[92,82]]
    cv2.fillConvexPoly(ink,np.array(polygon,np.int32),255)
    u=np.array(polygon[0])-np.mean(polygon[1:],0);u/=np.linalg.norm(u)
    result=check(ink,{'polygon_px':polygon,'direction':u.tolist(),'evidence':{'local_shaft_width_px':3}},
                 {'context_pixels':0,'context_segments':0})
    assert not result['available'] and not result['incomplete']
    assert result['support']['tail_protection_requires_transverse_exit']


def test_profile_caps_are_global_including_axis_retries():
    ink,_,_=mask()
    with patch.dict(detector.POLICY,{'maximum_profile_samples':0}):result=detector.search(255-ink)
    assert result['search_incomplete'] and not result['added']
    assert result['analysis']['profile_samples']==0


def test_context_caps_account_for_width_queries_and_explicit_abort():
    ink=np.ones((80,80),np.uint8)*255;audit={'context_pixels':99}
    assert source_widths(ink,np.array([10.,40.]),np.array([70.,40.]),3,audit,100) is None
    assert audit['context_pixels']==99


def test_eager_source_proposals_and_baseline_are_preserved():
    ink,_,_=mask();gray=255-ink
    a=detector.search(gray);b=detector.search(gray,eager_reference=True)
    assert a['candidates']==b['candidates'] and a['proposals']==b['proposals']
    assert a['context_exclusions']==b['context_exclusions']
    before=deepcopy(a['candidates']);saved=gray.copy()
    repeat=detector.search(gray,baseline=a['candidates'])
    assert repeat['candidates'][:len(before)]==before==a['candidates']
    assert np.array_equal(gray,saved)
    assert a['analysis']['raw_regions_generated']==a['analysis']['raw_regions_processed']


def test_disconnected_or_clipped_head_has_no_new_authority():
    ink,_,_=mask(shaft=False);assert not detector.search(255-ink)['added']
    ink,_,_=mask();assert not detector.search((255-ink)[78:])['added']
    assert detector.search(255-ink)['CAD_promotions']==0


def test_own_shaft_is_not_independent_evidence_of_a_repeated_field():
    ink=np.zeros((220,220),np.uint8)
    cv2.line(ink,(60,70),(175,70),255,2);cv2.line(ink,(60,150),(175,150),255,2)
    cv2.line(ink,(20,110),(76,110),255,2)
    polygon=[[105,110],[76,101],[76,119]]
    cv2.fillConvexPoly(ink,np.array(polygon,np.int32),255)
    audit={'context_pixels':0,'context_segments':0}
    r=check(ink,{'polygon_px':polygon,'direction':[1.,0.],'evidence':{'local_shaft_width_px':3}},audit)
    assert r['available'] and not r['incomplete']
    assert audit.get('context_own_shaft_segments_excluded',0)>0
