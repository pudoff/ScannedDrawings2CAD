"""Shared, label-free observation preparation for eligibility experiments."""
import hashlib
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from cad_recovery.dataset_layout_v1 import checked
from run_local_width_heads_v1 import read


def prepare(root, view):
    root = Path(root)
    gray = np.array(Image.open(checked(root, view['gray'])).convert('L'))
    if view.get('comparison_cache'):
        saved = read(checked(root, view['comparison_cache']))
        if hashlib.sha256(gray.tobytes()).hexdigest() != saved['input_pixel_sha256']:
            raise ValueError('Mixed source raster')
        return gray, saved['local_axis_guarded'], None
    deps = str(root/'artifacts/cv-skeleton-runtime-v1/deps')
    if deps not in sys.path:
        sys.path.insert(0, deps)
    from skimage.morphology import skeletonize
    from cad_recovery.local_axis_heads_v2 import search
    from cad_recovery.local_axis_ownership_v1 import apply
    from cad_recovery.local_width_heads_v1 import search as width_search
    from cad_recovery.baseline_anchor_windows_v1 import extract
    from cad_recovery.arrow_head_completion_v1 import search as completion
    from cad_recovery.native_head_bounds_v1 import guarded_native_heads
    from cad_recovery.stroke_roles_v1 import foreground, stroke_floor
    from run_stroke_ownership_v1 import widths_for
    ink = foreground(gray)
    skeleton = skeletonize(ink > 0, method='zhang').astype(np.uint8)*255
    geometry_packet = extract(ink, skeleton, conditioned=False, input_to_native=view['input_to_native'])
    geometry = geometry_packet['candidates']
    with guarded_native_heads():
        original = completion(gray, lines=[g for g in geometry if g['kind'] == 'line'],
            arcs=[g for g in geometry if g['kind'] == 'arc'],
            stroke_px=stroke_floor(widths_for(geometry)) or 3.)['candidates']
    width = width_search(gray, baseline=original)
    raw = search(gray, baseline=width['candidates'])
    packet = apply(gray, raw, baseline=width['candidates'])
    prep = {'original': original, 'width_candidates': width['candidates'],
            'geometry_windows': len(geometry_packet['analysis'].get('windows', [])),
            'geometry_search_incomplete': geometry_packet['analysis']['search_incomplete'],
            'raw_search_incomplete': raw['search_incomplete']}
    return gray, packet, prep
