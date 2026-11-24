from copy import deepcopy
import cv2
import numpy as np
import pytest
from cad_recovery.grid_role_revision_v1 import analyze,cv_authority,POLICY


def fixture(text=True):
    gray=np.full((320,380),255,np.uint8);items=[];tracks=[]
    for a,b in [*([(30,y),(345,y)] for y in (35,115,195,280)),*([(x,35),(x,280)] for x in (30,190,345))]:
        a=np.asarray(a);b=np.asarray(b);cv2.line(gray,tuple(a),tuple(b),0,7)
        p=np.rint(np.linspace(a,b,int(np.linalg.norm(b-a))+1)).astype(int);identity='line-'+str(len(items))
        items.append({'id':identity,'points_px':p.tolist(),'width_px':9.})
        tracks.append({'id':'track-'+str(len(tracks)),'points':[a.tolist(),b.tolist()],'width':9.,'interval_ids':[identity]})
    if text:
        for y in (95,175,260):cv2.putText(gray,'R P 47',(50,y),cv2.FONT_HERSHEY_COMPLEX,.9,0,1,cv2.LINE_8)
    roles=np.where(gray<128,9,255).astype(np.uint8)
    return gray,{'proposals':items,'tracks':tracks},roles


def authority(gray,roles,protected=None):
    return cv_authority(gray,roles,{'path':'fixture/cv-role-map.png','sha256':'a'*64},protected=protected)


def test_two_axis_geometry_without_text_does_not_revoke_genuine_hatching():
    gray,graph,roles=fixture(False);r=analyze(gray,graph,roles,authority=authority(gray,roles))
    assert r['complete'] and np.any(r['structure_only']!=roles) and np.array_equal(r['guarded'],roles)


def test_grid_with_source_glyph_context_revises_only_untrusted_hatching_to_unknown():
    gray,graph,roles=fixture();r=analyze(gray,graph,roles,authority=authority(gray,roles))
    delta=r['guarded']!=roles
    assert r['complete'] and delta.any() and np.all(roles[delta]==9) and np.all(r['guarded'][delta]==255)
    assert np.all(r['revision_owner'][delta]>0) and all(not d['assigned_title_grid'] for d in r['decisions'])


def test_unknown_provenance_cannot_revoke_even_when_the_shape_is_convincing():
    gray,graph,roles=fixture();r=analyze(gray,graph,roles)
    assert np.array_equal(r['guarded'],roles) and np.array_equal(r['structure_only'],roles)


def test_reviewed_pixels_are_immutable_and_not_implicitly_heuristic():
    gray,graph,roles=fixture();protected=np.full_like(gray,255);r=analyze(gray,graph,roles,authority=authority(gray,roles,protected))
    assert np.array_equal(r['guarded'],roles)
    a=authority(gray,roles);a['producer_kind']='manual_review'
    assert np.array_equal(analyze(gray,graph,roles,authority=a)['guarded'],roles)


def test_bound_source_role_and_mask_hashes_detect_changes():
    gray,graph,roles=fixture();a=authority(gray,roles);bad=roles.copy();bad[0,0]=9
    with pytest.raises(ValueError):analyze(gray,graph,bad,authority=a)
    a['mask'][0,0]=2
    with pytest.raises(ValueError):analyze(gray,graph,roles,authority=a)


def test_budget_failure_is_explicit_and_reverts_every_candidate_view(monkeypatch):
    gray,graph,roles=fixture();monkeypatch.setitem(POLICY,'maximum_pairs',0);r=analyze(gray,graph,roles,authority=authority(gray,roles))
    assert not r['complete'] and r['audit']['limits']
    assert np.array_equal(r['guarded'],roles) and np.array_equal(r['structure_only'],roles)


def test_source_graph_raw_maps_and_protection_are_not_modified():
    gray,graph,roles=fixture();pixels=gray.copy();old=roles.copy();before=deepcopy(graph);a=authority(gray,roles);mask=a['mask'].copy()
    r=analyze(gray,graph,roles,authority=a)
    assert np.array_equal(gray,pixels) and np.array_equal(roles,old) and graph==before and np.array_equal(a['mask'],mask)
    assert not r['source_pixels_erased'] and not r['production_admitted']


def test_a_single_frame_without_internal_cells_is_not_a_grid():
    gray,graph,roles=fixture();graph['tracks']=graph['tracks'][:2]
    r=analyze(gray,graph,roles,authority=authority(gray,roles))
    assert r['complete'] and not r['decisions'] and np.array_equal(r['guarded'],roles)
