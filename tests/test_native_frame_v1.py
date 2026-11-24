from copy import deepcopy
import builtins
import cv2
import numpy as np
import pytest
from cad_recovery.native_frame_v1 import source_masks, refine, digital_edge, POLICY
from cad_recovery.grid_role_revision_v1 import cv_authority, sha


def fixture():
    gray = np.full((200, 200), 255, np.uint8); frame = np.zeros_like(gray); grid = frame.copy()
    cv2.rectangle(frame, (25, 25), (175, 175), 255, 5)
    for y in (115, 135, 155): cv2.line(grid, (95, y), (175, y), 255, 5)
    for x in (95, 135): cv2.line(grid, (x, 115), (x, 175), 255, 5)
    gray[(frame | grid) > 0] = 0
    page = {'source_verified': True, 'patch_bbox': [0, 0, 200, 200], 'overview_to_source': np.eye(3).tolist(),
            'source_to_patch': np.eye(3).tolist(), 'kind': 'synthetic_full_page_pixels'}
    corners = np.array([[25, 25], [175, 25], [175, 175], [25, 175]], float)
    detector = {'complete': True, 'frames': [{'frame_semantics_confirmed': False, 'sides': [
        {'points_overview': [a.tolist(), b.tolist()], 'stroke_width_overview_px': 7., 'position_uncertainty_overview_px': .71}
        for a, b in zip(corners, np.roll(corners, -1, axis=0))]}]}
    source = source_masks(gray, page, detector)
    return gray, frame > 0, grid > 0, page, detector, source


def data(gray, scale=1):
    size = (gray.shape[1]//scale, gray.shape[0]//scale)
    small = gray if scale == 1 else cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    old = np.full_like(small, 255); raw = old.copy(); base = old.copy(); base[small < 255] = 8
    owners = (base == 8).astype(np.uint16)
    graph = {'proposals': [{'id': 'source-stroke', 'points_px': [[30, 30], [50, 30]]}]}
    auth = cv_authority(small, raw, {'path': 'raw.png', 'sha256': 'd'*64})
    matrix = [[scale, 0, .5*scale-.5], [0, scale, .5*scale-.5], [0, 0, 1]]
    return small, graph, raw, old, base, base.copy(), owners, auth, matrix


def run(d, gray=None, source=None):
    return refine(*d[:7], authority=d[7], input_to_native=d[8], source_gray=gray, source=source)


def test_digital_edge_feasibility_predicts_observed_integer_steps_exactly():
    x = np.arange(100); y = np.floor(.37*x+18.2).astype(int)
    model = digital_edge(x, y, x)
    assert model['source_fit_exact'] and np.array_equal(model['mid'], y)
    assert np.all(model['minimum'] <= y) and np.all(model['maximum'] >= y)


def test_non_linear_edge_is_rejected_instead_of_widening_its_error_tolerance():
    x = np.arange(100); y = np.floor(.37*x+18.2+3*np.sin(x/8)).astype(int)
    assert digital_edge(x, y, x) is None


def test_native_mask_preserves_table_at_shared_frame_contacts():
    gray, frame, grid, page, detector, source = fixture()
    assert source['complete'] and sum(r['fit'] for r in source['decisions']) == 4
    assert not np.any((source['mid'] > 0) & grid & ~frame)
    result = run(data(gray), gray, source)
    assert np.all(result['native_mid'][frame & ~grid] == 255)
    assert np.all(result['native_mid'][grid & ~frame] == 8)


def test_expanded_context_is_bound_to_the_same_original_pixels():
    gray, _, _, page, detector, _ = fixture()
    page['source_size_px'] = [200, 200]
    a = source_masks(gray, page, detector, context_gray=gray)
    assert a['bounded_whole_source_context_used'] and a['context_gray_sha256'] == sha(gray)
    other = gray.copy(); other[80, 80] = 0
    with pytest.raises(ValueError): source_masks(gray, page, detector, context_gray=other)


def test_half_output_can_split_one_input_cell_using_original_source_pixels():
    gray, frame, grid, _, _, source = fixture(); d = data(gray, 2); result = run(d, gray, source)
    assert result['native_mid'].shape == gray.shape
    assert np.all(result['native_mid'][grid & ~frame] == 8)
    assert not np.any((result['native_mid'] == 8) & frame & ~grid)
    original_coarse = cv2.resize(d[5], (200, 200), interpolation=cv2.INTER_NEAREST)
    assert np.any((original_coarse == 8) & frame & ~grid)
    assert result['output_coordinate_system'] == 'native_source_pixel_centres'


def test_inputs_graph_and_source_masks_are_immutable():
    gray, _, _, _, _, source = fixture(); d = data(gray, 2); before = deepcopy((gray, source, d))
    run(d, gray, source)
    assert np.array_equal(gray, before[0]) and d[1] == before[2][1]
    for n in ('mid', 'safe', 'possible', 'coverage'): assert np.array_equal(source[n], before[1][n])
    for i in (0, 2, 3, 4, 5, 6): assert np.array_equal(d[i], before[2][i])


def test_manual_authority_blocks_all_new_native_revisions():
    gray, _, _, _, _, source = fixture(); d = list(data(gray, 2))
    d[7] = cv_authority(d[0], d[2], {'path': 'raw.png', 'sha256': 'd'*64}, protected=np.full_like(d[0], 255))
    r = run(d, gray, source); assert np.array_equal(r['native_mid'], cv2.resize(d[4], (200, 200), interpolation=cv2.INTER_NEAREST))


@pytest.mark.parametrize('kind', ['gray', 'mask', 'input', 'transform', 'GT'])
def test_stale_or_non_source_native_context_is_rejected(kind):
    gray, _, _, _, _, source = fixture(); d = list(data(gray, 2))
    if kind == 'gray': gray[0, 0] = 250
    if kind == 'mask': source['mid'][0, 0] = 255
    if kind == 'input':
        d[0][0, 0] = 250
        d[7] = cv_authority(d[0], d[2], {'path': 'raw.png', 'sha256': 'd'*64})
    if kind == 'transform': d[8][0][2] += 1
    if kind == 'GT': source['GT_used'] = True
    with pytest.raises(ValueError): run(d, gray, source)


def test_missing_context_is_explicit_and_does_not_copy_another_prediction():
    gray, *_ = fixture(); d = data(gray, 2); r = run(d)
    assert not r['source_context_used'] and r['uncertainty_mask'].any()
    assert np.array_equal(r['native_mid'], cv2.resize(d[5], (200, 200), interpolation=cv2.INTER_NEAREST))


def test_budget_failure_discards_all_partial_source_masks(monkeypatch):
    gray, _, _, page, detector, _ = fixture(); monkeypatch.setitem(POLICY, 'maximum_source_samples', 0)
    source = source_masks(gray, page, detector)
    assert not source['complete'] and source['audit']['limits']
    assert not any(source[k].any() for k in ('mid', 'safe', 'possible', 'coverage'))


def test_manual_frame_targets_cannot_become_source_model_inputs():
    gray, _, _, page, detector, _ = fixture(); detector['frames'][0]['frame_semantics_confirmed'] = True
    with pytest.raises(ValueError): source_masks(gray, page, detector)


def test_model_reads_no_files_or_annotation_targets(monkeypatch):
    gray, _, _, page, detector, _ = fixture(); d = data(gray)
    def blocked(*a, **kw): raise AssertionError('Filesystem access during inference')
    monkeypatch.setattr(builtins, 'open', blocked)
    source = source_masks(gray, page, detector)
    assert run(d, gray, source)['complete']
