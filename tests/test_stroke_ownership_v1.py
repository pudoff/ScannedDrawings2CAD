from copy import deepcopy
import cv2
import numpy as np
import pytest

from cad_recovery.stroke_ownership_v1 import analyze,strict_text,hatch_bundles,line_objects
from cad_recovery.ownership_contours_v1 import boundaries


def line(name,a,b,width=2):return {'id':name,'path_id':name,'kind':'line','points_px':[a,b],'test_width':width}
def part():return {'id':'part','path_id':'part','kind':'polyline','points_px':[[100,120],[240,120],[240,230],[100,230],[100,120]],'closed':True,'test_width':7}
def head(name,tip,direction):
    tip=np.array(tip,float);u=np.array(direction,float);u/=np.linalg.norm(u);normal=np.array([-u[1],u[0]])
    return {'id':name,'tip_px':tip.tolist(),'direction':u.tolist(),'polygon_px':[tip.tolist(),(tip-12*u+4*normal).tolist(),(tip-12*u-4*normal).tolist()]}
def render(geometry,heads=()):
    gray=np.full((320,448),255,np.uint8)
    for c in geometry:
        cv2.polylines(gray,[np.array(c['points_px'],np.int32)],c.get('closed',False),0,c.get('test_width',2))
    for h in heads:cv2.fillConvexPoly(gray,np.rint(h['polygon_px']).astype(np.int32),0)
    return gray
def widths(g):return [{'id':c['id'],'width_px':c['test_width'],'coverage':1.,'heterogeneous':False} for c in g]
def roles(result):return {p['id']:p['role'] for p in result['proposals']}


@pytest.mark.parametrize('two_heads',[False,True])
def test_one_or_two_heads_with_two_extensions_define_a_dimension(two_heads):
    g=[part(),line('shaft',[112,70],[228 if two_heads else 240,70]),line('left',[100,60],[100,120]),line('right',[240,60],[240,120])]
    h=[head('left-head',[100,70],[-1,0])]
    if two_heads:h.append(head('right-head',[240,70],[1,0]))
    result=analyze(render(g,h),g,widths(g),h)
    assert roles(result)['shaft']=='dimension_line'
    assert roles(result)['left']==roles(result)['right']=='extension_line'
    assert roles(result)['part']=='part_contour' and result['dimension_chains']
    assert not result['production_admitted'] and result['source_pixels_erased']==0


def test_single_head_on_part_with_external_back_is_a_leader():
    g=[part(),line('leader',[30,170],[88,170])];h=[head('h',[100,170],[1,0])]
    result=analyze(render(g,h),g,widths(g),h)
    assert roles(result)['leader']=='leader_line' and not result['dimension_chains']


def test_one_head_with_no_contour_or_extensions_is_unknown():
    g=[line('shaft',[30,70],[88,70])];h=[head('h',[100,70],[1,0])]
    result=analyze(render(g,h),g,widths(g),h)
    assert roles(result)['shaft']=='unknown'


def test_thickness_alone_never_proves_an_open_part_contour():
    g=[line('thick',[30,80],[300,80],7),line('thin',[30,250],[300,250],2)]
    assert roles(analyze(render(g),g,widths(g),[]))['thick']=='unknown'


def test_regular_hatch_bundle_blocks_head_association_but_retains_observation():
    g=[part()]+[line(f'h{i}',[120+i*12,140],[150+i*12,205]) for i in range(5)]
    h=[head('false-DT',[165,180],[0,1])];result=analyze(render(g,h),g,widths(g),h)
    assert result['hatch_bundles'] and result['head_observations_retained']==1
    assert result['head_events'][0]['reason']=='head_in_parallel_stroke_bundle'
    assert result['head_events'][0]['observation_retained'] and 'false-DT' not in result['head_ids_used']


def test_parallel_bars_do_not_become_a_text_group():
    g=[line(f'bar{i}',[40+i*14,50],[50+i*14,80],2) for i in range(5)]
    text=strict_text(render(g),g,2)
    assert not text['component_mask'].any()
    assert any(r['reason']=='parallel_stroke_group' for r in text['rejected'])


def test_closed_round_glyphs_or_holes_remain_ambiguous():
    gray=np.full((180,260),255,np.uint8)
    for x in (50,90,130):cv2.circle(gray,(x,80),8,0,2)
    text=strict_text(gray,[],2)
    assert not text['component_mask'].any() and text['ambiguity_mask'].any()


def test_ordinary_text_is_proposed_only_on_actual_component_ink():
    gray=np.full((180,320),255,np.uint8);cv2.putText(gray,'ABC DEF',(30,90),cv2.FONT_HERSHEY_SIMPLEX,1.,0,2)
    text=strict_text(gray,[],2)
    assert text['component_mask'].any() and not np.any((text['component_mask']>0)&(gray==255))


def test_conflicts_and_unknowns_partition_ink_without_changes():
    g=[part(),line('unowned',[280,20],[400,200])];gray=render(g);original=gray.copy();original_g=deepcopy(g)
    result=analyze(gray,g,widths(g),[])
    assert np.array_equal(gray,original) and g==original_g
    assert result['exact_ink_partition'] and result['assigned_ink_px']+result['unknown_ink_px']==result['input_ink_px']
    assert all(not p['semantic_ownership_confirmed'] for p in result['proposals'])


def test_native_heads_and_geometry_are_not_mutated():
    g=[part(),line('leader',[30,170],[88,170])];h=[head('h',[100,170],[1,0])];h[0]['direction']=np.array([2.,0.])
    before=h[0]['direction'].copy();analyze(render(g,h),g,widths(g),h)
    assert np.array_equal(before,h[0]['direction'])


def test_missing_width_scale_is_explicit_incomplete():
    g=[line('x',[20,20],[100,20])];result=analyze(render(g),g,[],[])
    assert result['search_incomplete'] and result['reason']=='missing_stroke_scale'
    assert np.all(result['pixel_roles']==255)


def test_invalid_head_direction_is_rejected():
    g=[part()];h=[head('h',[100,120],[1,0])];h[0]['direction']=[0,0]
    with pytest.raises(ValueError,match='Finite'):analyze(render(g),g,widths(g),h)


def test_boundary_hypothesis_keeps_open_gap_open():
    g=[part()];closed=boundaries(g,(320,448),2)
    opened=deepcopy(g);opened[0]['closed']=False;opened[0]['points_px'][-1]=[100,121]
    result=boundaries(opened,(320,448),2)
    assert closed['boundaries'] and not result['boundaries']


def test_frame_is_not_automatically_a_part():
    g=[{'id':'frame','path_id':'frame','kind':'polyline','points_px':[[5,5],[442,5],[442,314],[5,314],[5,5]],'closed':True,'test_width':2}]
    r=analyze(render(g),g,widths(g),[])
    assert roles(r)['frame']=='sheet_frame'


def test_text_grid_is_not_promoted_to_part_contours():
    g=[line(f'h{i}',[300,230+i*30],[420,230+i*30]) for i in range(3)]
    g += [line(f'v{i}',[300+i*60,230],[300+i*60,290]) for i in range(3)]
    gray=render(g);cv2.putText(gray,'ABC',(310,250),cv2.FONT_HERSHEY_SIMPLEX,.35,0,1)
    r=analyze(gray,g,widths(g),[])
    assert r['title_grids'] and all(p['role']=='title_grid' for p in r['proposals'])


def test_known_negative_ink_is_penalized_but_unassigned_real_ink_is_not():
    from cad_recovery.stroke_ownership_metrics_v1 import score_roles
    truth=np.full((10,10),255,np.uint8);truth[2,2]=1
    predicted=np.full_like(truth,255);predicted[2,2]=1;predicted[5,5]=1
    unassigned=score_roles(truth,predicted)
    negative=np.zeros_like(truth);negative[5,5]=255
    explicit=score_roles(truth,predicted,negative)
    assert unassigned['roles'][0]['precision']==1 and explicit['roles'][0]['precision']==.5
    assert explicit['false_role_assignments_on_negative_ink_px']==1 and not explicit['unassigned_real_ink_is_negative']


def test_empty_ink_is_complete_and_missing_scale_keeps_source_IDs():
    g=[line('x',[20,20],[100,20])]
    empty=analyze(np.full((320,448),255,np.uint8),[],[],[])
    assert not empty['search_incomplete'] and empty['exact_ink_partition']
    r=analyze(render(g),g,[],[])
    assert r['proposals'][0]['id']=='x' and r['proposals'][0]['role']=='unknown'


def test_new_control_split_is_fixed_without_using_predictions():
    import importlib.util
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'scripts/prepare_ownership_controls_v1.py'
    spec=importlib.util.spec_from_file_location('ownership_controls',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    a,b=module.scenes(),module.scenes()
    assert len(a)==24 and all(np.array_equal(x['gray'],y['gray']) and x['case']==y['case'] for x,y in zip(a,b))
    train={s['case']['group_id'] for s in a if s['case']['split']=='train'};val={s['case']['group_id'] for s in a if s['case']['split']=='validation'}
    assert len(train)==len(val)==4 and not train&val
    assert all(not np.any((s['negative']>0)&(s['truth']!=255)) for s in a)
