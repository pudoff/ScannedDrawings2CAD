"""Keep coarse boxes, partial pixel targets and interval supervision separate."""
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from run_grid_role_v1 import ROOT, load, truth_for
from run_topology_robust_v1 import read, write
from cad_recovery.dataset_layout_v1 import pin, checked
from cad_recovery.stroke_roles_v1 import CLASSES

BASE = ROOT/'artifacts/source-region-data-audit-v1'


def run():
    cv2.setNumThreads(1)
    BASE.mkdir(exist_ok=True)
    if any(BASE.iterdir()):
        raise ValueError('Audit output must be empty')
    release = read(ROOT/'artifacts/grid-context-release-v1.json')
    protected = release['prior_protected_pins'] | release['files']
    names = ['artifacts/layout-region-labels-v1/records.json', 'artifacts/layout-region-labels-v1/admission.json',
             'artifacts/pixel-role-expansion-v3/records.json', 'artifacts/pixel-role-expansion-v3/admission.json']
    pixel_manifest = read(ROOT/'artifacts/pixel-role-expansion-v3/manifest.json')
    for name in names:
        expected = protected.get(name)
        if expected is None:
            relative = Path(name).name
            expected = next(p['sha256'] for p in pixel_manifest['files'] if p['path'] == relative)
        checked(ROOT, {'path': name, 'sha256': expected})
    layout, layout_admission, patches, pixel_admission = [read(ROOT/n) for n in names]
    jobs = [j for j in read(ROOT/'artifacts/grid-context-profile-v1/plan.json')['jobs']
            if j['case']['scope'] == 'partial_real_reviewed_ink']
    by_patch = {p['id']: p for p in patches}
    # Matching only source SHA, never names or directory substrings.
    by_sha = {}
    for row in layout:
        by_sha.setdefault(row['source']['sha256'], []).append(row)
    rows, targets = [], []
    for j in jobs:
        patch = by_patch[j['case']['id']]
        source = patch['source']
        sha = source['sha256'] if isinstance(source, dict) else patch.get('source_sha256')
        matched = by_sha.get(sha, [])
        native = None
        for v in j['views']:
            gray, graph, _, _, _, _, _, _ = load(j, v)
            if native is None:
                native = gray.copy()
            truth, _ = truth_for(j, native)
            target = truth if truth.shape == gray.shape else cv2.resize(truth, gray.shape[::-1], interpolation=cv2.INTER_NEAREST)
            counts = Counter()
            admitted = 0
            for interval in graph['proposals']:
                points = np.rint(interval['points_px']).astype(int)
                values = target[points[:, 1], points[:, 0]]
                known = (values >= 1) & (values <= 9)
                hist = np.bincount(values[known], minlength=10)
                winner = int(hist.argmax()) if known.any() else None
                coverage = float(known.mean())
                agreement = float(hist[winner]/known.sum()) if known.any() else 0.
                usable = known.sum() >= 8 and coverage >= .8 and agreement >= .98
                if usable:
                    admitted += 1
                    counts[CLASSES[winner]] += 1
                    targets.append({'case': j['case']['id'], 'mode': v['mode'], 'interval_id': interval['id'],
                                    'source_index_interval': interval.get('source_index_interval'),
                                    'role': winner, 'sample_count': len(points), 'known_samples': int(known.sum()),
                                    'coverage': coverage, 'agreement': agreement,
                                    'original_split': patch['split'], 'group_id': patch['group_id'],
                                    'train_usable': patch['split'] == 'train',
                                    'partial_supervision_only': True, 'used_for_inference': False})
            rows.append({'case': j['case']['id'], 'mode': v['mode'], 'original_split': patch['split'],
                         'group_id': patch['group_id'], 'coarse_source_matches': [r['id'] for r in matched],
                         'source_intervals': len(graph['proposals']), 'partial_interval_targets': admitted,
                         'by_role': dict(counts), 'coarse_boxes_are_not_pixel_targets': True})
    write(BASE/'interval-targets.json', {'rows': targets, 'validation_never_train': True,
                                      'inference_uses_targets': False, 'coordinate_or_physical_supervision': False})
    write(BASE/'report.json', {'inputs': [pin(ROOT, n) for n in names],
          'pixel_manifest': pin(ROOT, 'artifacts/pixel-role-expansion-v3/manifest.json'), 'layout_boxes': len(layout),
          'layout_by_split_role': {s: dict(Counter(r['role'] for r in layout if r['split'] == s)) for s in ('train', 'validation')},
          'partial_pixel_patches': len(patches), 'partial_by_split': dict(Counter(p['split'] for p in patches)),
          'layout_admission': layout_admission, 'pixel_admission': pixel_admission,
          'rows': rows, 'interval_target_counts': dict(Counter(t['original_split'] for t in targets)),
          'coarse_boxes_inferred_as_dense_labels': False, 'GT_used_for_inference': False,
          'new_real_labels_created': False, 'fresh_synthetic_controls_needed': True,
          'independent_real_evaluation': False, 'closed_test_used': False, 'physical_geometry_approved': False})
    print({'layout_boxes': len(layout), 'patches': len(patches),
           'interval_targets': dict(Counter(t['original_split'] for t in targets))}, flush=True)


if __name__ == '__main__':
    run()
