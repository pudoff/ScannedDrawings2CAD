from copy import deepcopy
import builtins
import cv2
import numpy as np
import pytest
from cad_recovery.page_frame_context_v2 import detect, straight_component, corner_grid
from cad_recovery.stroke_roles_v1 import foreground
from cad_recovery.frame_grid_guard_v2 import bind_context, analyze, POLICY
from cad_recovery.grid_role_revision_v1 import cv_authority
from test_page_frame_context_v1 import page
from test_frame_grid_guard_v1 import fixture


@pytest.mark.parametrize('width', [13, 17, 23])
def test_rule_width_is_observed_instead_of_capped_in_pixels(width):
    assert straight_component([10, 10, 300, width, 300*width], True)
    assert straight_component([10, 10, width, 300, 300*width], False)
    assert not straight_component([10, 10, 300, width, 300], True)


@pytest.mark.parametrize('turns', [0, 1, 2, 3])
def test_legacy_frames_and_source_pixels_are_preserved(turns):
    a = np.ascontiguousarray(np.rot90(page(), turns)); before = a.copy(); r = detect(a)
    assert r['complete'] and r['frames'] and r['legacy_frames_retained']
    assert np.array_equal(a, before) and all(not f['frame_semantics_confirmed'] for f in r['frames'])


def test_thick_shallow_corner_rules_have_a_source_supported_fallback():
    a = np.full((900, 1000), 255, np.uint8)
    cv2.rectangle(a, (70, 350), (930, 830), 0, 11)
    cv2.circle(a, (280, 500), 55, 0, 11)
    for y in (635, 700, 765): cv2.line(a, (500, y), (930, y), 0, 11)
    for x in (500, 700): cv2.line(a, (x, 635), (x, 830), 0, 11)
    r = detect(a)
    assert r['complete'] and r['frames']


@pytest.mark.parametrize('width', [7, 15])
def test_grid_location_uses_stroke_centre_instead_of_top_bbox_edge(width):
    a = np.full((700, 1000), 255, np.uint8)
    cv2.rectangle(a, (100, 100), (900, 555), 0, width)
    for y in (352, 415, 485): cv2.line(a, (500, y), (900, y), 0, width)
    for x in (500, 700): cv2.line(a, (x, 352), (x, 555), 0, width)
    quad = np.array([[100, 100], [900, 100], [900, 555], [100, 555]], float)
    assert corner_grid(foreground(a), quad, {'warp_pixels': 0, 'limits': []})


def data():
    d = fixture()
    context = bind_context(d[0], d[1], d[3], d[4], d[5], d[7], d[8], 'b'*64, np.eye(3), {})
    return d, context


def run(d, context):
    authority = cv_authority(d[0], d[2], {'path': 'raw.png', 'sha256': 'd'*64})
    return analyze(*d[:6], authority=authority, context=context)


def test_context_records_width_error_and_pixel_footprint_separately():
    d, ctx = data(); s = ctx['sides'][0]
    assert s['stroke_width_page_px'] == 3 and s['position_uncertainty_page_px'] == .71
    assert s['input_normal_footprint_page_px'] == .5
    r = run(d, ctx); assert r['complete'] and np.array_equal(r['coarse'], r['uncertainty'])


def test_transverse_source_stroke_marks_uncertainty_without_erasing_ink():
    d = fixture(); cv2.line(d[0], (60, 10), (60, 60), 0, 3)
    d[1]['proposals'].append({'id': 'crossing', 'points_px': [[60, y] for y in range(10, 61)]})
    ctx = bind_context(d[0], d[1], d[3], d[4], d[5], d[7], d[8], 'b'*64, np.eye(3), {})
    before = deepcopy(d); r = run(d, ctx)
    assert r['complete'] and r['uncertainty_mask'].any()
    assert np.all(r['uncertainty'][r['uncertainty_mask'] > 0] == d[4][r['uncertainty_mask'] > 0])
    assert r['source_pixels_erased'] == r['new_frame_roles'] == 0
    assert np.array_equal(d[0], before[0]) and d[1] == before[1]


def test_budget_exhaustion_is_explicit_and_returns_base(monkeypatch):
    d, ctx = data(); monkeypatch.setitem(POLICY, 'maximum_owner_side_pairs', 0)
    r = run(d, ctx)
    assert not r['complete'] and np.array_equal(r['uncertainty'], d[4])


def test_models_read_no_files_or_gt(monkeypatch):
    a = page(); d, ctx = data()
    def blocked(*a, **kw): raise AssertionError('Filesystem access during inference')
    monkeypatch.setattr(builtins, 'open', blocked)
    assert detect(a)['frames'] and run(d, ctx)['complete']


def test_mixed_two_by_two_cell_cannot_satisfy_both_hard_role_constraints():
    truth = np.array([[7, 8], [255, 255]], np.uint8)
    for role in (8, 255):
        output = np.full_like(truth, role)
        assert np.any((truth == 7) & (output == 8)) or np.any((truth == 8) & (output != 8))
