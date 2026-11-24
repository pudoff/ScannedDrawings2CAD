import ast
from copy import deepcopy
import itertools
import json
from pathlib import Path
import cv2
import numpy as np
import pytest
from cad_recovery.open_reference_chains_v1 import extend as old_extend
from cad_recovery.open_reference_chains_v2 import extend, POLICY as BASIS_POLICY
from cad_recovery.open_reference_context_v1 import filter_context, POLICY
from test_open_reference_chains_v1 import fixture

ROOT=Path(__file__).resolve().parents[1]


def add_path(args, identity, a, b, pen=9, track=True):
    gray, previous, _, _ = args
    p=np.rint(np.linspace(a,b,int(max(abs(np.asarray(b)-a)))+1)).astype(int)
    cv2.polylines(gray,[p],False,0,pen)
    previous['proposals'].append({'id':identity,'points_px':p.astype(float).tolist(),'width_px':float(pen)})
    if track:
        delta=np.asarray(b,float)-a; length=float(np.linalg.norm(delta))
        previous['tracks'].append({'id':identity,'points':[a,b],'u':(delta/length).tolist(),'length':length,'width':float(pen),'interval_ids':[identity]})


def context_result(args):
    gray,previous,base,heads=args
    corrected=extend(*args)
    return corrected,filter_context(gray,previous,base,corrected)


def body_tracks(args):
    args[1]['tracks'].append({'id':'body','points':[[20,110],[180,110]],'u':[1.,0.],'length':160.,'width':9.,
                              'interval_ids':['body-left','body-mid','body-right']})


def test_basis_fix_preserves_old_kernel_methods_and_numeric_policies():
    paths=[ROOT/'src/cad_recovery'/('open_reference_chains_v'+str(i)+'.py') for i in (1,2)]
    trees=[ast.parse(p.read_text(encoding='utf-8')) for p in paths]
    for name in ('LocalReferences','contacts'):
        nodes=[next(n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==name) for tree in trees]
        assert ast.dump(nodes[0],include_attributes=False)==ast.dump(nodes[1],include_attributes=False)
    from cad_recovery.open_reference_chains_v1 import POLICY as original
    assert {k:v for k,v in original.items() if k!='version'}=={k:v for k,v in BASIS_POLICY.items() if k!='version'}


@pytest.mark.parametrize('bits',list(itertools.product((False,True),repeat=3)))
def test_every_shaft_and_extension_basis_reversal_is_exactly_invariant(bits):
    args=fixture(); expected=extend(*args)['open_reference_chains']
    for track,flip in zip(args[1]['tracks'],bits):
        if flip: track['points']=track['points'][::-1]; track['u']=[-v for v in track['u']]
    assert extend(*args)['open_reference_chains']==expected


def test_source_track_head_order_and_source_path_direction_are_invariant():
    args=fixture(); expected=extend(*args)['open_reference_chains']
    args[1]['tracks'].reverse(); args[1]['proposals'].reverse(); args[-1].reverse()
    for item in args[1]['proposals']: item['points_px'].reverse()
    assert extend(*args)['open_reference_chains']==expected


def test_identifier_renaming_does_not_change_the_chosen_geometry():
    args=fixture(); before=extend(*args)['open_reference_chains']
    mapping={p['id']:'new-'+str(100-i) for i,p in enumerate(args[1]['proposals'])}
    for item in args[1]['proposals']: item['id']=mapping[item['id']]
    for t in args[1]['tracks']: t['id']='renamed-'+t['id']; t['interval_ids']=[mapping[i] for i in t['interval_ids']]
    for h in args[-1]: h['id']='renamed-'+h['id']
    after=extend(*args)['open_reference_chains']
    assert len(after)==len(before)==1 and after[0]['feet_px']==before[0]['feet_px']


def test_quarter_turn_of_pixels_paths_tracks_and_heads_preserves_geometry():
    args=list(fixture());before=extend(*args)['open_reference_chains']; height=args[0].shape[0]
    def point(p):return [height-1-p[1],p[0]]
    args[0]=np.rot90(args[0],k=-1).copy();args[2]['pixel_roles']=np.rot90(args[2]['pixel_roles'],k=-1).copy()
    for item in args[1]['proposals']:item['points_px']=[point(p) for p in item['points_px']]
    for t in args[1]['tracks']:t['points']=[point(p) for p in t['points']];t['u']=[-t['u'][1],t['u'][0]]
    for h in args[-1]:h['tip_px']=point(h['tip_px']);h['polygon_px']=[point(p) for p in h['polygon_px']];h['direction']=[-h['direction'][1],h['direction'][0]]
    after=extend(*args)['open_reference_chains']
    assert len(after)==len(before)==1
    expected=sorted(zip(before[0]['head_ids'],[point(p) for p in before[0]['feet_px']]))
    assert sorted(zip(after[0]['head_ids'],after[0]['feet_px']))==expected


def test_opposite_basis_regression_is_fixed_without_moving_observations():
    args=fixture(); right=next(t for t in args[1]['tracks'] if t['id']=='right')
    right['points'].reverse(); right['u']=[0.,-1.]; snapshot=deepcopy(args[1:2]+args[3:])
    assert not old_extend(*args)['open_reference_chains'] and len(extend(*args)['open_reference_chains'])==1
    assert args[1:2]+args[3:]==snapshot


def test_ordinary_open_body_reference_is_retained():
    args=fixture(); body_tracks(args); a,b=context_result(args)
    assert len(a['open_reference_chains'])==len(b['open_reference_chains'])==1
    assert b['reference_context_decisions'][0]['accepted']


def test_connected_wide_frame_crossing_shaft_is_rejected():
    args=fixture(); body_tracks(args)
    add_path(args,'left-frame',[20,110],[20,10]);add_path(args,'upper-frame',[20,10],[180,10]);add_path(args,'right-frame',[180,10],[180,110])
    a,b=context_result(args)
    assert len(a['open_reference_chains'])==1 and not b['open_reference_chains']
    assert 'wide_component_crosses_dimension_shaft' in b['reference_context_decisions'][0]['reasons']


def test_disconnected_clipped_opposing_frame_strokes_are_rejected():
    args=fixture()
    args[1]['proposals']=[p for p in args[1]['proposals'] if p['id'] not in ('body-left','body-right')]
    add_path(args,'body-left',[0,110],[50,110],track=False);add_path(args,'body-right',[150,110],[209,110],track=False)
    args[1]['tracks'].append({'id':'body','points':[[0,110],[209,110]],'u':[1.,0.],'length':209.,'width':9.,'interval_ids':['body-left','body-mid','body-right']})
    add_path(args,'opposite-frame',[0,10],[209,10])
    a,b=context_result(args)
    assert len(a['open_reference_chains'])==1 and not b['open_reference_chains']
    assert 'clipped_opposing_frame_strokes' in b['reference_context_decisions'][0]['reasons']


def test_equal_span_thick_parallel_rows_are_rejected_without_head_exclusion_shortcut():
    args=fixture();body_tracks(args)
    for y in (80,95,125,140): add_path(args,'stripe-'+str(y),[20,y],[180,y])
    for y0,y1 in ((80,95),(95,110),(110,125),(125,140)): add_path(args,'spine-'+str(y0),[20,y0],[20,y1],track=False)
    a,b=context_result(args)
    assert len(a['open_reference_chains'])==1 and not b['open_reference_chains']
    decision=b['reference_context_decisions'][0]
    assert 'equal_span_periodic_wide_component' in decision['reasons']
    assert decision['periodic_groups'][0]['offset_count']>=4


def test_thin_hatching_does_not_turn_a_valid_body_into_thick_periodic_rows():
    args=fixture();body_tracks(args)
    for y in (80,95,125,140): add_path(args,'thin-'+str(y),[20,y],[180,y],pen=1)
    for y0,y1 in ((80,95),(95,110),(110,125),(125,140)): add_path(args,'thin-spine-'+str(y0),[20,y0],[20,y1],pen=1,track=False)
    a,b=context_result(args)
    assert len(a['open_reference_chains'])==len(b['open_reference_chains'])==1


def test_inherited_chain_maps_and_observations_survive_context_rejection():
    args=fixture();body_tracks(args);args[2]['dimension_chains']=[{'shaft_track':'prior','head_ids':['old-head']}]
    add_path(args,'left-frame',[20,110],[20,10]);add_path(args,'upper-frame',[20,10],[180,10])
    original_gray=args[0].copy(); original_roles=args[2]['pixel_roles'].copy(); snapshot=deepcopy([args[1],args[3]])
    _,packet=context_result(args)
    assert packet['dimension_chains']==args[2]['dimension_chains']
    assert [args[1],args[3]]==snapshot and np.array_equal(args[0],original_gray) and np.array_equal(packet['pixel_roles'],original_roles)


def test_context_budget_failure_keeps_only_baseline_and_is_explicit(monkeypatch):
    args=fixture(); corrected=extend(*args);monkeypatch.setitem(POLICY,'maximum_source_points',5)
    result=filter_context(args[0],args[1],args[2],corrected)
    assert result['search_incomplete'] and result['reference_context_audit']['limits'] and not result['open_reference_chains']


def test_empty_candidate_packet_needs_no_context_work():
    args=fixture();args[3].pop();a,b=context_result(args)
    assert not a['open_reference_chains'] and b['reference_context_audit']['source_points']==0


def test_context_cannot_silently_change_inherited_pixel_roles():
    args=fixture();corrected=extend(*args);corrected={**corrected,'pixel_roles':corrected['pixel_roles'].copy()};corrected['pixel_roles'][0,0]=1
    with pytest.raises(ValueError):filter_context(args[0],args[1],args[2],corrected)


def test_context_evidence_is_serializable_and_never_gives_physical_approval():
    args=fixture();_,packet=context_result(args)
    json.dumps({k:v for k,v in packet.items() if k!='pixel_roles'},allow_nan=False)
    assert not packet['physical_geometry_approved'] and not packet['production_admitted'] and packet['source_pixels_erased']==0
