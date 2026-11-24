"""Frozen role producers shared by contact qualification and comparison."""
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from cad_recovery.dataset_layout_v1 import checked
from run_supported_role_overlay_v1 import read


def prepare(root, view):
    root = Path(root)
    gray = np.array(Image.open(checked(root, view['gray'])).convert('L'))
    prep = {'geometry_extractions': 0, 'head_DT_skeleton_calls': 0,
            'interval_calls': 0, 'local_guard_calls': 0, 'legacy_role_calls': 0}
    if view.get('cache'):
        cache = view['cache']
        row = read(checked(root, cache['row']))
        source, geometry = read(checked(root, cache['source'])), read(checked(root, cache['geometry']))
        baseline = np.array(Image.open(checked(root, cache['baseline_roles'])))
        text = np.array(Image.open(checked(root, cache['text_mask'])))
        previous = row['previous_interval_details']
        heads, extra = row['head_candidates']['baseline'], row['head_candidates']['DT_novel']
        local = {name: {**row['details'][key], 'pixel_roles': np.array(Image.open(checked(root, cache[map_key]))), 'text_mask': text}
            for name, key, map_key in [('local', 'local_verified', 'local_roles'), ('hybrid', 'hybrid', 'hybrid_roles')]}
        return gray, source, geometry, baseline, text, previous, local, heads, extra, prep
    from cad_recovery.local_role_guard_v1 import analyze as local_analyze
    from cad_recovery.interval_roles_v1 import analyze as interval_analyze
    if view.get('comparison'):
        cache = view['comparison']
        row = read(checked(root, cache['row']))
        source, geometry = read(checked(root, cache['source'])), read(checked(root, cache['geometry']))
        baseline = np.array(Image.open(checked(root, cache['baseline_roles'])))
        heads, extra = row['head_candidates']['baseline'], row['head_candidates']['DT_novel']
    else:
        deps = str(root/'artifacts/cv-skeleton-runtime-v1/deps')
        if deps not in sys.path:
            sys.path.insert(0, deps)
        from skimage.morphology import skeletonize
        from cad_recovery.joint_shape_v1 import extract
        from cad_recovery.anchored_curve_fit_v1 import extract as anchored_extract
        from cad_recovery.stroke_roles_v2 import analyze as safe_heads
        from cad_recovery.stroke_roles_v1 import foreground, stroke_floor, text_components, guard_heads, role_proposals, append_heads
        from run_stroke_ownership_v1 import widths_for
        ink = foreground(gray)
        skeleton = skeletonize(ink > 0, method='zhang').astype(np.uint8)*255
        f = extract(ink, skeleton, input_to_native=view['input_to_native'], move_nodes=False)
        source, geometry = f['source'], f['candidates']
        anchored = anchored_extract(ink, skeleton, conditioned=False, input_to_native=view['input_to_native'])['candidates']
        widths = widths_for(anchored)
        h = safe_heads(gray, anchored, widths, stroke_px=stroke_floor(widths) or 3.)
        heads = h['base_heads']['candidates']
        extra = [c for c in append_heads(heads, h['DT_proposals']['candidates']) if c['id'].startswith('DT-added-')]
        widths = widths_for(geometry)
        t = text_components(gray, stroke_floor(widths) or 3.)
        kept, _ = guard_heads(heads, t['component_mask'])
        baseline = role_proposals(gray, geometry, widths, kept, t['component_mask'], context=True)['pixel_roles']
        prep.update(geometry_extractions=2, head_DT_skeleton_calls=1, legacy_role_calls=2)
    raw = interval_analyze(gray, geometry, source, heads, input_to_native=view['input_to_native'], diagnostic_heads=extra)
    previous = {k: v for k, v in raw.items() if k not in ('pixel_roles', 'text_mask')}
    text = raw['text_mask']
    local = local_analyze(gray, previous, baseline, text, heads, diagnostic_heads=extra)
    prep.update(interval_calls=1, local_guard_calls=1)
    return gray, source, geometry, baseline, text, previous, local, heads, extra, prep


def cached_jobs(root):
    from cad_recovery.dataset_layout_v1 import pin
    root = Path(root)
    old = read(root/'artifacts/supported-role-overlay-v1/plan.json')
    index = {(r['case'], r['mode']): r for r in read(root/'artifacts/supported-role-overlay-v1/results-index.json') if not r['repeat']}
    jobs = []
    for job in old['jobs']:
        views = []
        for view in job['views']:
            item = index[job['case']['id'], view['mode']]
            folder = (root/item['artifact']).parent
            comparison = {name: pin(root, (folder/file).relative_to(root).as_posix()) for name, file in
                [('row', 'result.json'), ('source', 'source.json'), ('geometry', 'geometry.json'),
                 ('baseline_roles', 'old_context-roles.png'), ('old_overlay_roles', 'supported_overlay-roles.png'),
                 ('old_owner', 'pixel-interval-owner.png'), ('text_mask', 'text-mask.png')]}
            views.append({**view, 'comparison': comparison})
        jobs.append({**job, 'views': views, 'new_control': False})
    return jobs
