from copy import deepcopy
import numpy as np
import pytest
from cad_recovery.open_reference_chains_v2 import extend
from cad_recovery.open_reference_context_v2 import filter_context,POLICY
from test_open_reference_chains_v1 import fixture
from test_reference_context_v1 import add_path,body_tracks


def clipped_args():
    args=list(fixture())
    args[1]['proposals']=[p for p in args[1]['proposals'] if p['id'] not in ('body-left','body-right')]
    add_path(args,'body-left',[0,110],[50,110],track=False)
    add_path(args,'body-right',[150,110],[209,110],track=False)
    args[1]['tracks'].append({'id':'body','points':[[0,110],[209,110]],'u':[1.,0.],
        'length':209.,'width':9.,'interval_ids':['body-left','body-mid','body-right']})
    return args


def run(args):
    return filter_context(args[0],args[1],args[2],extend(*args))


def test_cropped_single_strip_abstains_without_claiming_frame_semantics():
    r=run(clipped_args())
    assert len(r['stage1_open_reference_chains'])==1 and not r['open_reference_chains']
    assert r['part_structure_decisions'][0]['reason']=='clipped_reference_strip_without_transverse_part_witness'


def test_visible_transverse_part_wall_preserves_cropped_reference():
    args=clipped_args();add_path(args,'part-wall',[0,110],[0,70])
    r=run(args)
    assert len(r['stage1_open_reference_chains'])==len(r['open_reference_chains'])==1
    assert r['part_structure_decisions'][0]['part_structure']['transverse_part_witnesses']


def test_nonclipped_open_reference_does_not_need_an_extra_wall():
    args=fixture();body_tracks(args)
    assert len(run(args)['open_reference_chains'])==1


@pytest.mark.parametrize('length,pen',[(5,9),(40,1)])
def test_short_or_thin_marks_do_not_supply_transverse_part_context(length,pen):
    args=clipped_args();add_path(args,'mark',[0,110],[0,110-length],pen=pen)
    assert not run(args)['open_reference_chains']


def test_reversed_tracks_and_source_paths_leave_missing_context_decision_unchanged():
    args=clipped_args();before=run(args)
    for track in args[1]['tracks']:track['points'].reverse();track['u']=[-v for v in track['u']]
    for proposal in args[1]['proposals']:proposal['points_px'].reverse()
    args[1]['tracks'].reverse();args[1]['proposals'].reverse();args[-1].reverse()
    after=run(args)
    assert before['open_reference_chains']==after['open_reference_chains']==[]
    assert len(before['stage1_open_reference_chains'])==len(after['stage1_open_reference_chains'])==1


def test_veto_preserves_inherited_chains_pixels_and_input_observations():
    args=clipped_args();args[2]['dimension_chains']=[{'shaft_track':'inherited'}]
    source=deepcopy([args[1],args[-1]]);gray=args[0].copy();roles=args[2]['pixel_roles'].copy()
    r=run(args)
    assert r['dimension_chains']==[{'shaft_track':'inherited'}] and np.array_equal(r['pixel_roles'],roles)
    assert np.array_equal(args[0],gray) and [args[1],args[-1]]==source


def test_extra_stage_has_explicit_bounded_failure(monkeypatch):
    args=clipped_args();monkeypatch.setitem(POLICY,'maximum_track_checks',0)
    r=run(args)
    assert r['search_incomplete'] and r['part_structure_audit']['limits']==['track_checks']
    assert not r['open_reference_chains']
