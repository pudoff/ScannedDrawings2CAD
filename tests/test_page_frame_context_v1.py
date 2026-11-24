import builtins
import cv2
import numpy as np
import pytest
from cad_recovery.page_frame_context_v1 import detect, POLICY


def page():
    a = np.full((800, 800), 255, np.uint8)
    cv2.rectangle(a, (70, 80), (730, 730), 0, 5)
    cv2.circle(a, (270, 300), 75, 0, 5)
    for y in (535, 600, 665): cv2.line(a, (400, y), (730, y), 0, 5)
    for x in (400, 535): cv2.line(a, (x, 535), (x, 730), 0, 5)
    return a


@pytest.mark.parametrize('turns', [0, 1, 2, 3])
def test_full_page_detects_observed_frame_at_quarter_turns(turns):
    a = np.ascontiguousarray(np.rot90(page(), turns)); before = a.copy(); r = detect(a)
    assert r['complete'] and r['frames'] and np.array_equal(a, before)
    assert all(not f['frame_semantics_confirmed'] and len(f['sides']) == 4 for f in r['frames'])
    assert all(s['source_sample_count'] >= 18 and s['stroke_width_overview_px'] > 0
               for f in r['frames'] for s in f['sides'])


def test_blank_padding_does_not_require_frame_to_fill_canvas():
    a = np.full((1800, 1200), 255, np.uint8); a[950:1750, 180:980] = page()
    assert detect(a)['frames']


def test_closed_part_rectangle_without_corner_grid_is_not_a_frame():
    a = page(); a[510:720, 380:720] = 255
    assert not detect(a)['frames']


def test_standalone_table_without_separate_drawing_is_not_a_frame():
    a = page(); a[200:400, 170:370] = 255
    assert not detect(a)['frames']


def test_empty_image_has_no_frame_and_budget_exhaustion_abstains(monkeypatch):
    assert detect(np.full((100, 100), 255, np.uint8))['frames'] == []
    monkeypatch.setitem(POLICY, 'maximum_warp_pixels', 0)
    r = detect(page()); assert not r['complete'] and not r['frames']


def test_detector_is_pixel_only_and_reads_no_annotation_files(monkeypatch):
    a = page()
    def blocked(*a, **kw): raise AssertionError('Filesystem read during inference')
    monkeypatch.setattr(builtins, 'open', blocked)
    assert detect(a)['frames']
