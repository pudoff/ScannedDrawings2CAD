"""P16.1e: common partial baseline, cached neural outputs and missing CV only."""
from collections import Counter
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image

from .architecture_data_readiness_v1 import Package, verify as verify_readiness
from .dataset_layout_v1 import checked, encoded, inside, pin, read
from .family_catalog_v2 import require
from .hard_raster_runs_v1 import execute
from .ocr_baseline_metrics_v1 import aggregate_scores_v1
from . import segmentation_acceptance_v1 as segmentation

# The readiness manifest pin is obtained from the immutable release below.
RELEASE = {'path': 'artifacts/architecture-data-readiness-release-v1.json',
           'sha256': 'cfa2eca38a15fdcdfcf2028fef9a92ea553a2849340d13d1a02a76a697413127'}
PARENTS = {
    'unet': ('artifacts/segmentation-revision-v2/manifest.json', '9e5b8ec7a6bc72407b940092032e5c80704eac1e3a3baf4494afd8cd8f827491'),
    'cv': ('artifacts/segmentation-revision-assessment-v2/manifest.json', 'c50b6f6defaa7f63c44af40c3336665698131497fa18042c630b6a47eeb929e4'),
    'ocr': ('artifacts/ocr-recognition-baseline-v5/manifest.json', '2b6f1b764e5cb7ef61d5e94df1b62636a02e36a07f6d84ff24caaa5ab67e1bf7'),
    'glyph': ('artifacts/engineering-glyph-classifier-v1/result/manifest.json', 'c9ee1b1aa2ef96a56d322926cccc99d27fffdd39728754ebe515744f84b851a4'),
    'components': ('artifacts/component-signs-v1/result/manifest.json', 'fba2809f095911d3b9039f819e5be00028151f03d86aea0b579584965f799c52'),
}
VARIANTS = ['original', 'control-gray', 'balanced-contrast-transpose', 'balanced-contrast-resize']
POLICY = {'task': 'P16.1e', 'primary': 'balanced-contrast-resize', 'CV_variants': VARIANTS,
    'new_work': '12 missing native ROIs x original/three cached guided variants; geometry only',
    'neural_inference': False, 'training': False, 'parameter_search': False,
    'closed_test_used': False, 'physical_geometry_approved': False, 'production_admitted': False,
    'independent_evaluation': False, 'validation': 'diagnostic_only_no_model_selection',
    'CV_profile': 'regional_masked', 'CV_threads': 4, 'CV_seed': 20261110,
    'max_child_seconds': 900, 'max_host_MiB': 3072,
    'arrow_scope': '13 reviewed ROIs only; other ROIs unmeasured, not zero error',
    'OCR_scope': '64 diagnostic validation lines; 105 allowed train lines reserved, not rerun',
    'glyph_scope': 'frozen isolated-glyph diagnostics; not a VLM or OCR deployment decision',
    'metrics': segmentation.POLICY,
}
OWN_CODE = ['src/cad_recovery/architecture_baseline_v1.py', 'scripts/run_architecture_baseline_v1.py']


def readiness_pin(root):
    release = read(checked(root, RELEASE))
    name = 'artifacts/architecture-data-readiness-v3/manifest.json'
    return {'path': name, 'sha256': release['files'][name]}


def code_pins(root):
    # Preserve exactly the existing CV implementation and its dependencies.
    previous = read(root / 'configs/reviews/segmentation-revision-assessment-v2.json')['code']
    names = list(dict.fromkeys([p['path'] for p in previous] + OWN_CODE +
        ['src/cad_recovery/architecture_data_readiness_v1.py', 'src/cad_recovery/ocr_baseline_metrics_v1.py',
         'src/cad_recovery/service_v1/process.py']))
    for record in previous: checked(root, record)
    return [pin(root, name) for name in names]


def task_grid(rows):
    return [{'id': r['id'], 'variant': v} for r in rows if not r['saved_cv_variants'] for v in VARIANTS]


def freeze(root, destination):
    require(not destination.exists(), 'Choose a new protocol')
    ready_pin = readiness_pin(root); ready = Package(root, ready_pin)
    verify_readiness(root, ready.manifest['config'], ready.folder)
    rows = ready.json('roi-matrix.json')['rows']
    jobs = task_grid(rows)
    require(len(jobs) == 48, 'Changed missing CV task grid')
    parents = {k: {'path': p, 'sha256': s} for k, (p, s) in PARENTS.items()}
    for p in parents.values(): checked(root, p)
    config = {'schema_version': 'architecture-baseline-config/1.0', 'policy': POLICY,
        'readiness': ready_pin, 'parents': parents, 'code': code_pins(root), 'new_CV_jobs': jobs,
        'cached_CV_jobs': 52, 'frozen_before_new_CV': True}
    destination.write_bytes(encoded(config))
    return pin(root, destination.relative_to(root).as_posix())


def context(root, config_pin):
    config = read(checked(root, config_pin))
    require(config['schema_version'] == 'architecture-baseline-config/1.0' and config['policy'] == POLICY,
            'Changed baseline policy')
    require(config['readiness'] == readiness_pin(root) and config['code'] == code_pins(root), 'Changed data/code')
    require(config['parents'] == {k: {'path': p, 'sha256': s} for k, (p, s) in PARENTS.items()}, 'Changed cached releases')
    ready = Package(root, config['readiness']); rows = ready.json('roi-matrix.json')['rows']
    admission = ready.json('admission.json')
    require(admission['partial_CV_UNet_SegFormer_comparison_admitted'] is True and
            admission['independent_evaluation'] is False, 'Invalid data admission')
    require(config['new_CV_jobs'] == task_grid(rows) and len(task_grid(rows)) == 48, 'Changed assigned work')
    # All rows were screened by the frozen, replay-verified parent before freeze.
    boundaries = ready.json('report.json')['boundary']
    require(len(boundaries) == len(rows) and not any(r['blocked'] for r in boundaries), 'Closed ROI')
    for row in rows:
        require(row['split'] in ('train', 'validation'), 'Invalid split')
        checked(root, row['image']); checked(root, row['labels'])
    packages = {k: Package(root, p) for k, p in config['parents'].items()}
    return config, ready, rows, packages


def array(path):
    with Image.open(path) as image: return np.asarray(image).copy()


def new_cv(root, config_pin, destination):
    """Only the assigned missing geometry work. No arrow targets are invented."""
    from .connector_guard_v1 import detect_geometry
    config, _, rows, packages = context(root, config_pin)
    require(not destination.exists(), 'Choose fresh CV output')
    destination.mkdir(parents=True)
    cv2.setNumThreads(POLICY['CV_threads']); cv2.setRNGSeed(POLICY['CV_seed'])
    by_id = {r['id']: r for r in rows}; outputs = []
    for job in config['new_CV_jobs']:
        row = by_id[job['id']]; name, variant = row['id'], job['variant']
        source = checked(root, row['image']); target = array(checked(root, row['labels']))
        folder = destination / variant / name; folder.mkdir(parents=True)
        prediction_pin = None
        if variant != 'original':
            prediction_path = packages['unet'].path(variant + '/result/' + name + '.png')
            prediction = array(prediction_path)
            require(prediction.shape == target.shape and np.isin(prediction, range(10)).all(), 'Invalid saved argmax')
            rgb = array(source); source = folder / 'guided.png'
            Image.fromarray(segmentation.guided_rgb(rgb, prediction)).save(source)
            prediction_pin = pin(root, prediction_path.relative_to(root).as_posix())
        start = time.perf_counter()
        geometry = detect_geometry(source, folder / 'geometry', search_profile=POLICY['CV_profile'])
        elapsed = time.perf_counter() - start
        geometry.pop('timings', None)
        geometry_path = folder / 'geometry.json'; geometry_path.write_bytes(encoded(geometry))
        outputs.append({**job, 'image': row['image'], 'labels': row['labels'], 'prediction': prediction_pin,
            'geometry': pin(root, geometry_path.relative_to(root).as_posix()), 'CV_seconds': elapsed})
        print(name, variant, len(geometry['candidates']), 'primitives', round(elapsed, 3), 's', flush=True)
    (destination / 'results.json').write_bytes(encoded(outputs))


def supported_summary(evaluations):
    if not evaluations: return None
    result = segmentation.combine_evaluations(evaluations)
    # The inherited macro averages all classes, including background. Make the
    # actual support explicit and provide a separate foreground-only macro.
    values = [r['IoU_within_review'] for r in result['classes'][1:]
              if r['target_pixels'] > 0 and r['IoU_within_review'] is not None]
    result['foreground_macro_IoU_supported_only'] = float(np.mean(values)) if values else None
    result['supported_foreground_classes'] = [r['class_id'] for r in result['classes'][1:] if r['target_pixels'] > 0]
    return result


def cv_aggregate(rows):
    values = [x for r in rows for x in r['primitive_coverage'].values()]
    covered = sum(x['covered_ink_pixels'] for x in values)
    assigned = sum(x['assigned_ink_pixels'] for x in values)
    arrows = [r for r in rows if r.get('arrow_targets_measured')]
    return {'ROIs': len(rows), 'assigned_ink': assigned, 'covered_ink': covered,
        'assigned_ink_coverage': covered / assigned if assigned else None,
        'arrow_ROIs': len(arrows), 'arrow_assigned': sum(r['assigned'] for r in arrows) if arrows else None,
        'arrow_matched': sum(r['matched'] for r in arrows) if arrows else None,
        'false_tips_in_reviewed_negatives': sum(r['false_tips'] for r in arrows) if arrows else None,
        'whole_ROI_false_positive_precision': None}


def score_lines(rows, predictions):
    require(len({r['id'] for r in predictions}) == len(predictions), 'Duplicate OCR predictions')
    lookup = {r['id']: r for r in predictions}
    require(set(lookup) == {r['id'] for r in rows}, 'Missing/extra assigned OCR predictions')
    scored = []
    for row in rows:
        pred = lookup[row['id']]; text = pred['prediction']; status = pred['status']
        require(status in ('ok', 'empty', 'error') and isinstance(text, str), 'Invalid cached OCR')
        scored.append({'id': row['id'], 'source_id': row['group_id'], 'reference': row['literal'],
                       'prediction': text, 'status': status})
    result = aggregate_scores_v1(scored)
    return {k: v for k, v in result.items() if k != 'samples'}


def ocr_baseline(root, ready, packages):
    inventory = ready.json('vlm-inventory.json')['items']
    rows = [r for r in inventory if r['corpus'] == 'ocr' and r['split'] == 'validation' and not r['blocked']]
    require(len(rows) == 64 and all(r['exact_target'] for r in rows), 'Changed OCR validation grid')
    config = read(checked(root, packages['ocr'].manifest['config']))
    samples = {r['id']: r['image'] for r in config['samples']}
    require(samples == {r['id']: r['image'] for r in rows}, 'Cached OCR pixels differ')
    labels = {r['id']: r for r in config['rows']}
    for row in rows:
        require(labels[row['id']]['text'] == row['literal'] and labels[row['id']]['source_sha256'] == row['source']['sha256'],
                'Cached OCR text/source differ')
        checked(root, row['image'])
    predictions = packages['ocr'].json('predictions.json')
    metrics = {arm: score_lines(rows, p) for arm, p in predictions.items()}
    return {'scope': '64 fixed diagnostic validation line crops, no train inference or test access',
        'rows': rows, 'arms': metrics, 'cached_predictions': pin(root, packages['ocr'].path('predictions.json').relative_to(root).as_posix()),
        'raw_predictions_repaired': False, 'new_OCR_calls': 0}


def sign_baseline(root, ready, packages):
    inventory = ready.json('vlm-inventory.json')['items']
    targets = [r for r in inventory if r['corpus'] == 'glyph' and not r['blocked']]
    cached = packages['glyph'].json('predictions.json')
    predictions = {r['id']: r for r in cached['glyphs']}
    require(set(predictions) == {r['id'] for r in targets}, 'Changed glyph diagnostic grid')
    results = []
    for row in targets:
        pred = predictions[row['id']]
        require(pred['sha256'] == row['image']['sha256'] and pred['target'] == row['class'] and pred['split'] == row['split'],
                'Cached glyph identity/target mismatch')
        results.append({'id': row['id'], 'split': row['split'], 'group_id': row['group_id'], 'image': row['image'],
            'target': row['class'], 'literal': row['literal'], 'scope': row['scope'],
            'ocr_class': pred['ocr_prediction'], 'classifier_class': pred['prediction'],
            'raw_OCR': pred['ocr_raw'], 'known': row['class'] is not None})
    summary = {}
    for split in ('train', 'validation', 'development', 'synthetic'):
        selected = [r for r in results if r['split'] == split]
        known = [r for r in selected if r['known']]; unknown = [r for r in selected if not r['known']]
        summary[split] = {'known': len(known), 'unknown': len(unknown),
            'OCR_class_correct': sum(r['ocr_class'] == r['target'] for r in known),
            'classifier_class_correct': sum(r['classifier_class'] == r['target'] for r in known),
            'unknown_classifier_abstained': sum(r['classifier_class'] is None for r in unknown),
            'confusion': dict(Counter(str(r['target']) + ' -> ' + str(r['classifier_class']) for r in known))}
    component_results = packages['components'].json('results.json')
    real = [r for r in component_results if r['suite'] == 'real']
    component_targets = [r for r in inventory if r['corpus'] == 'components']
    expected_ids = {r['context_id'] for r in component_targets}
    require({r['id'] for r in real} == expected_ids and len(real) == 20, 'Changed real component controls')
    modes = {}
    for mode in sorted({r['mode'] for r in real}):
        selected = [r for r in real if r['mode'] == mode]
        modes[mode] = {k: sum(r['metrics'][k] for r in selected) for k in
            ('known_component_targets', 'literal_exact_known_components', 'expected_role_edges',
             'correct_role_edges', 'false_role_edges', 'unresolved_component_targets', 'localized_unresolved_components')}
    return {'glyph_rows': results, 'isolated_glyph_class_metrics': summary, 'real_component_modes': modes,
        'real_component_results': real, 'known_component_targets': component_targets,
        'training_admitted': False, 'independent_evaluation': False,
        'warning': 'Isolated glyph boxes and auto component crops have different tasks; counts are not combined.'}


def build(root, config_pin, output):
    config, ready, rows, packages = context(root, config_pin)
    raw = read(output / 'new-cv/results.json')
    require([{k: r[k] for k in ('id', 'variant')} for r in raw] == config['new_CV_jobs'], 'Lost or added CV jobs')
    lookup = {(r['id'], r['variant']): r for r in raw}
    saved = {r['id']: r for r in packages['cv'].json('report.json')['CV']}
    pixel, downstream = [], []
    for row in rows:
        target = array(checked(root, row['labels']))
        for variant in VARIANTS[1:] + ['balanced-contrast-resize-repeat']:
            path = packages['unet'].path(variant + '/result/' + row['id'] + '.png')
            prediction = array(path)
            require(prediction.shape == target.shape and np.isin(prediction, range(10)).all(), 'Invalid prediction grid')
            pixel.append({'id': row['id'], 'split': row['split'], 'stratum': row['stratum'], 'variant': variant,
                          'prediction': pin(root, path.relative_to(root).as_posix()), **segmentation.evaluate(target, prediction)})
        for variant in VARIANTS:
            if row['id'] in saved:
                score = {**saved[row['id']]['scores'][variant], 'arrow_targets_measured': True}
                provenance = {'kind': 'cached', 'package': config['parents']['cv']}
            else:
                record = lookup[row['id'], variant]
                require(record['image'] == row['image'] and record['labels'] == row['labels'], 'Changed CV pixels/targets')
                geometry = read(checked(root, record['geometry']))
                score = {'primitive_coverage': segmentation.primitive_coverage(target, geometry),
                         'primitive_candidates': len(geometry['candidates']), 'arrow_targets_measured': False,
                         'assigned': None, 'matched': None, 'false_tips': None}
                provenance = {'kind': 'new_missing_geometry_only', 'geometry': record['geometry']}
            downstream.append({'id': row['id'], 'split': row['split'], 'stratum': row['stratum'],
                               'variant': variant, 'provenance': provenance, **score})
    summaries = []
    for split in ('train', 'validation'):
        for stratum in ('all', 'monochrome', 'colored'):
            for variant in VARIANTS:
                select = lambda r: r['split'] == split and r['variant'] == variant and (stratum == 'all' or r['stratum'] == stratum)
                px = [r for r in pixel if select(r)]; cv = [r for r in downstream if select(r)]
                summaries.append({'split': split, 'stratum': stratum, 'variant': variant,
                    'pixel_roles': supported_summary(px), 'CV': cv_aggregate(cv) if cv else None})
    ocr = ocr_baseline(root, ready, packages); signs = sign_baseline(root, ready, packages)
    resources = {'new_CV': read(output / 'runtime.json'),
        'new_CV_task_times': [{k: r[k] for k in ('id', 'variant', 'CV_seconds')} for r in raw],
        'cached_UNet_training_runtime': [pin(root, packages['unet'].path(v + '/runtime.json').relative_to(root).as_posix()) for v in VARIANTS[1:]],
        'cached_OCR_runtime': pin(root, packages['ocr'].path('runtime.json').relative_to(root).as_posix()),
        'limits': ['New CV time/memory is one guarded CPU batch, not isolated per-model memory.',
                   'Cached U-Net runtime includes training: not an inference speed comparison.',
                   'No comparable end-to-end GPU/VLM speed or memory result; qualify each future model separately.']}
    report = {'schema_version': 'architecture-baseline-report/1.0', 'policy': POLICY,
        'ROI_count': len(rows), 'pixel_prediction_evaluations': len(pixel),
        'CV_task_count': len(downstream), 'cached_CV_tasks': 52, 'new_CV_tasks': len(raw),
        'arrow_ROI_count': 13, 'OCR_validation_lines': 64, 'new_neural_calls': 0,
        'summaries': summaries, 'physical_geometry_approved': False, 'production_admitted': False,
        'independent_evaluation': False, 'next_task': 'P16.3b.1',
        'remaining_limits': ['Partial labels only; no full view/page precision or physical CAD accuracy.',
            'Monochrome train has no sheet_frame; leader train support is one group; validation hatch support is one group.',
            '105 allowed train OCR lines are reserved, not scored anew. 16 closed historical train lines remain excluded.',
            'Geometry arrows measured only on 13 ROIs; auto angular links remain 0/2 on two known train contexts.',
            'Glyph/component/line crops overlap; synthetic and viewed development are separate, not independent validation.',
            'The existing U-Net/classifier remain diagnostic; no replacement of the production profile.']}
    return {'report.json': report, 'pixel-evaluations.json': pixel, 'downstream-evaluations.json': downstream,
            'ocr-baseline.json': ocr, 'sign-baseline.json': signs, 'resources.json': resources}


def run(root, config_pin, destination):
    context(root, config_pin)
    require(not destination.exists(), 'Choose fresh baseline output')
    destination.mkdir(parents=True)
    command = [sys.executable, str(root / OWN_CODE[1]), 'child', '--config', config_pin['path'],
               '--output', destination.relative_to(root).as_posix()]
    runtime = execute(root, command, destination / 'process', POLICY['max_child_seconds'], POLICY['max_host_MiB'])
    (destination / 'runtime.json').write_bytes(encoded(runtime))
    require(runtime['budget_pass'], 'CV child failed/budget exceeded; preserve output and inspect logs')
    results = build(root, config_pin, destination)
    for name, value in results.items(): (destination / name).write_bytes(encoded(value))
    files = [p.relative_to(destination).as_posix() for p in destination.rglob('*') if p.is_file()]
    manifest = {'schema_version': 'architecture-baseline-manifest/1.0', 'config': config_pin,
                'files': [pin(destination, name) for name in sorted(files)]}
    (destination / 'manifest.json').write_bytes(encoded(manifest))
    return results['report.json']


def verify(root, config_pin, destination):
    manifest = read(destination / 'manifest.json')
    require(manifest['schema_version'] == 'architecture-baseline-manifest/1.0' and manifest['config'] == config_pin,
            'Changed baseline manifest')
    for record in manifest['files']: checked(destination, record)
    expected = build(root, config_pin, destination)
    for name, value in expected.items():
        require((destination / name).read_bytes() == encoded(value), 'Recomputed baseline differs: ' + name)
    return {'verified': True, 'recomputed_files': len(expected), 'new_neural_calls': 0}
