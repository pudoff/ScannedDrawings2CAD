"""Frozen B0 comparison: two train children, common partial metrics, no deployment."""
from html import escape
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

from .architecture_data_readiness_v1 import Package, Boundary
from .architecture_baseline_v1 import supported_summary, cv_aggregate
from .dataset_layout_v1 import checked, encoded, pin, read
from .family_catalog_v2 import require
from .segformer_protocol_v1 import environment, validate_qualification, GPU_PYTHON, EXTRA
from . import segmentation_acceptance_v1 as metrics

PREVIOUS = {'path': 'artifacts/segformer-protocol-release-v1.json',
            'sha256': '9e7867927c2f6bba833cc57f16b93bb3d1fa053b67a64473ba75b7a084e80325'}
EXPERIMENT = {'path': 'configs/reviews/segformer-experiment-v1.json',
              'sha256': '638180ddbac84db32f7fd5f1d10d4555f664dff7707852762cb4e0f5cdec9231'}
CODE = ['src/cad_recovery/segformer_training_v1.py', 'src/cad_recovery/segformer_comparison_v1.py',
        'scripts/train_segformer_v1.py', 'scripts/run_segformer_comparison_v1.py']


def code_pins(root):
    old = read(root / 'configs/reviews/segmentation-revision-assessment-v2.json')['code']
    qualified = read(root / 'configs/reviews/segformer-probe-v2.json')['code']
    for record in old + qualified: checked(root, record)
    names = [r['path'] for r in old + qualified] + CODE + [
        'src/cad_recovery/architecture_baseline_v1.py', 'src/cad_recovery/architecture_data_readiness_v1.py',
        'src/cad_recovery/gpu_comparison_v1.py', 'scripts/review_segmentation_acceptance_v1.py']
    return [pin(root, n) for n in dict.fromkeys(names)]


def inputs(root):
    release = read(checked(root, PREVIOUS))
    protocol = read(checked(root, EXPERIMENT))
    require(environment(root) == protocol['environment'], 'Changed qualified environment')
    qualification = read(checked(root, protocol['qualification']))
    validate_qualification(read(checked(root, protocol['qualification_config'])), qualification,
                           read(checked(root, protocol['qualification_runtime'])))
    signatures = {**release['prior_protected_pins'], **release['files']}
    name = 'artifacts/architecture-data-readiness-v3/manifest.json'
    ready = Package(root, {'path': name, 'sha256': signatures[name]})
    ready_config = read(checked(root, ready.manifest['config']))
    groups, sources, registry1, registry2, fresh = [read(checked(root, p)) for p in ready_config['metadata']]
    boundary = Boundary(groups['groups'], sources, [registry1, registry2],
        [{'source_path': s['input'], 'source_sha256': s['input_sha256']} for s in fresh['selected_sources']])
    rows = ready.json('roi-matrix.json')['rows']
    require(len(rows) == 25 and len({r['id'] for r in rows}) == 25, 'Changed common ROI set')
    screened = [boundary.check(r['source'], r['split'], r['id']) for r in rows]
    require(not any(r['blocked'] for r in screened), 'Closed input source/group')
    require(all(s['group_id'] == r['group_id'] for s, r in zip(screened, rows)), 'Changed ROI groups')
    # Boundary is complete before the first image/mask is opened.
    seen = set()
    for row in rows:
        if row['source']['sha256'] not in seen:
            checked(root, row['source']); seen.add(row['source']['sha256'])
        checked(root, row['image']); checked(root, row['labels'])
    return protocol, rows, ready, Package(root, protocol['baseline_manifest'])


def assignment(rows, ids):
    training = [{'id': r['id'], 'split': r['split'], 'group_id': r['group_id'], 'size_px': r['size_px'],
                 'image': r['image'], 'labels': r['labels']} for r in rows if r['split'] == 'train']
    require([r['id'] for r in training] == ids and len(training) == 15, 'Changed train membership/order')
    inference = [{k: r[k] for k in ('id', 'image', 'size_px')} for r in rows]
    require(not ({r['group_id'] for r in rows if r['split'] == 'train'} &
                 {r['group_id'] for r in rows if r['split'] == 'validation'}), 'Cross-split groups')
    return training, inference


def freeze(root, output):
    require(not output.exists(), 'Choose a fresh run config')
    protocol, rows, ready, _ = inputs(root)
    training, inference = assignment(rows, protocol['training_patch_ids'])
    value = {'schema_version': 'segformer-comparison-config/1.0', 'task': 'P16.3b.2',
        'previous_release': PREVIOUS, 'experiment': EXPERIMENT, 'readiness': ready.record,
        'training': training, 'inference': inference, 'code': code_pins(root),
        'dependency_directory': EXTRA, 'CV': {'seconds': 900, 'host_MiB': 6144, 'seed': 20261110, 'threads': 4},
        'frozen_before_dataset_training': True, 'closed_test_used': False, 'production_admitted': False}
    output.write_bytes(encoded(value)); return pin(root, output.relative_to(root).as_posix())


def context(root, record):
    config = read(checked(root, record))
    require(config['schema_version'] == 'segformer-comparison-config/1.0' and config['experiment'] == EXPERIMENT
            and config['previous_release'] == PREVIOUS and config['dependency_directory'] == EXTRA, 'Changed experiment identity')
    require(config['code'] == code_pins(root), 'Changed runner/producer code')
    protocol, rows, ready, baseline = inputs(root)
    training, inference = assignment(rows, protocol['training_patch_ids'])
    require(config['training'] == training and config['inference'] == inference and config['readiness'] == ready.record,
            'Changed pixels, split or inference target isolation')
    require(config['CV'] == {'seconds': 900, 'host_MiB': 6144, 'seed': 20261110, 'threads': 4}, 'Changed CV budget')
    return config, protocol, rows, baseline


def image_array(path):
    with Image.open(path) as image: return np.asarray(image).copy()


def downstream(root, record, output):
    from .connector_guard_v1 import detect_geometry
    from . import open_arc_search_v1 as arcs
    from . import arrow_head_completion_v1 as arrows
    sys.path.insert(0, str(root / 'scripts'))
    from review_segmentation_acceptance_v1 import cv_score
    config, protocol, rows, _ = context(root, record)
    masks = Package(root, protocol['labels'])
    controls = {r['id']: r for r in masks.json('records.json') if 'arrow_controls' in r['review']}
    require(len(controls) == 13, 'Changed arrow target scope')
    cv2.setNumThreads(config['CV']['threads']); cv2.setRNGSeed(config['CV']['seed'])
    destination = output / 'CV'; require(not destination.exists(), 'Choose fresh downstream output')
    destination.mkdir(); results = []
    for row in rows:
        name = row['id']; folder = destination / name; folder.mkdir()
        rgb = image_array(checked(root, row['image'])); target = image_array(checked(root, row['labels']))
        prediction_path = output / protocol['primary'] / 'result' / (name + '.png')
        prediction = image_array(prediction_path)
        source = folder / 'guided.png'; Image.fromarray(metrics.guided_rgb(rgb, prediction)).save(source)
        geometry = detect_geometry(source, folder / 'geometry', search_profile=protocol['comparison']['CV_profile'])
        geometry.pop('timings', None)
        (folder / 'geometry.json').write_bytes(encoded(geometry))
        if name in controls:
            stroke = geometry['analysis']['estimated_stroke_width_px'] / geometry['analysis']['scale']
            gray = np.asarray(Image.fromarray(rgb).convert('L'))
            query = np.asarray(Image.open(source).convert('L'))
            arc = arcs.search(query, stroke_px=stroke, lines=geometry['candidates'])
            heads = arrows.search(gray, lines=geometry['candidates'], arcs=arc['candidates'], stroke_px=stroke)
            (folder / 'arcs.json').write_bytes(encoded(arc)); (folder / 'heads.json').write_bytes(encoded(heads))
            score = {**cv_score(controls[name], target, geometry, heads), 'arrow_targets_measured': True}
        else:
            score = {'primitive_coverage': metrics.primitive_coverage(target, geometry),
                'primitive_candidates': len(geometry['candidates']), 'arrow_targets_measured': False,
                'assigned': None, 'matched': None, 'false_tips': None}
        results.append({'id': name, 'split': row['split'], 'stratum': row['stratum'], 'group_id': row['group_id'], **score})
        print('CV', name, score['primitive_candidates'], 'primitives', score['matched'], 'arrow matches', flush=True)
    (destination / 'results.json').write_bytes(encoded(results))


def gates(protocol, validation, candidate, original, repeat_equal, resource_pass):
    thresholds = protocol['necessary_partial_gates']
    require(len(candidate) == len(original) == 25 and {r['id'] for r in candidate} == {r['id'] for r in original}, 'Incomplete common CV grid')
    old = {r['id']: r for r in original}
    arrow_rows = [r for r in candidate if r['arrow_targets_measured']]
    require(len(arrow_rows) == 13 and all(old[r['id']]['arrow_targets_measured'] for r in arrow_rows), 'Changed arrow comparison scope')
    coverage = lambda r: sum(x['covered_ink_pixels'] for x in r['primitive_coverage'].values())
    values = validation['classes'][1:]
    return {'each_foreground_IoU': len(values) == 9 and all(r['target_pixels'] > 0 and r['IoU_within_review'] is not None and
                r['IoU_within_review'] >= thresholds['each_foreground_IoU_min'] for r in values),
        'background': validation['background_false_foreground_fraction'] is not None and
                validation['background_false_foreground_fraction'] <= thresholds['background_false_foreground_max'],
        'thin_axis': validation['thin_axis_connected_80_fraction'] is not None and
                validation['thin_axis_connected_80_fraction'] >= thresholds['thin_axis_connected_80_min'],
        'per_ROI_CV_ink': all(coverage(r) >= coverage(old[r['id']]) for r in candidate),
        'per_ROI_arrows_and_negatives': all(r['matched'] >= old[r['id']]['matched'] and r['false_tips'] <= old[r['id']]['false_tips'] for r in arrow_rows),
        'independent_repeat': repeat_equal, 'resources': resource_pass}


def resource_check(protocol, output):
    p = protocol['resources']; values = []
    for variant in protocol['variants']:
        runtime = read(output / variant / 'runtime.json'); result = read(output / variant / 'result/result.json')
        require(result['updates'] == protocol['updates'] and len(result['losses']) == protocol['updates']
                and np.isfinite(result['losses']).all(), 'Lost training updates')
        gpu = runtime['gpu_memory']
        values.append(runtime['budget_pass'] and not gpu['errors'] and gpu['peak_MiB'] is not None and gpu['peak_MiB'] <= p['device_MiB']
            and result['allocated_peak_MiB'] <= p['allocated_MiB'] and result['reserved_peak_MiB'] <= p['reserved_MiB'])
    runtimes = [read(output / v / 'runtime.json') for v in protocol['variants']]
    cv_runtime = read(output / 'CV-runtime.json')
    return all(values) and sum(r['walltime_seconds'] for r in runtimes) <= p['total_train_child_seconds'] and cv_runtime['budget_pass']


def validate_trained_result(config, record, protocol, result):
    require(result['run_protocol'] == record and result['experiment'] == EXPERIMENT
            and result['variant'] in protocol['variants'], 'Model result belongs to another run')
    require(result['training_ids'] == protocol['training_patch_ids'] and result['updates'] == protocol['updates']
            and len(result['losses']) == protocol['updates'] and np.isfinite(result['losses']).all(), 'Incomplete/changed training ledger')
    require(result['validation_targets_read_by_model'] is False and result['checkpoint_selected_before_validation'] is True
            and result['closed_test_used'] is False and result['production_admitted'] is False, 'Changed training authority')
    expected = {r['id']: r['size_px'] for r in config['inference']}
    actual = result['inference']
    require(len(actual) == len(expected) and {r['id'] for r in actual} == set(expected), 'Lost native inference assignment')
    require(all(r['size_px'] == expected[r['id']] and r['coverage_min'] > 0 for r in actual), 'Changed/uncovered native pixel grid')


def summarize(root, record, output):
    config, protocol, rows, baseline = context(root, record)
    base_pixels = baseline.json('pixel-evaluations.json'); base_cv = baseline.json('downstream-evaluations.json')
    saved_cv = read(output / 'CV/results.json')
    require(len(saved_cv) == 25 and {r['id'] for r in saved_cv} == {r['id'] for r in rows}, 'Lost downstream ROI')
    sys.path.insert(0, str(root / 'scripts'))
    from review_segmentation_acceptance_v1 import cv_score
    masks = Package(root, protocol['labels']); annotations = {r['id']: r for r in masks.json('records.json')}
    recomputed_cv = []; evaluations = []
    primary, repeat = protocol['variants']; repeats_equal = True
    a = read(output / primary / 'result/result.json'); b = read(output / repeat / 'result/result.json')
    for result in (a, b): validate_trained_result(config, record, protocol, result)
    repeats_equal &= a['final_parameters_sha256'] == b['final_parameters_sha256'] and a['losses'] == b['losses']
    require(a['initial_parameters_sha256'] == b['initial_parameters_sha256'], 'Changed independent initialization')
    for row in rows:
        name = row['id']; target = image_array(checked(root, row['labels']))
        prediction = image_array(output / primary / 'result' / (name + '.png'))
        other = image_array(output / repeat / 'result' / (name + '.png'))
        repeats_equal &= np.array_equal(prediction, other)
        value = metrics.evaluate(target, prediction)
        evaluations.append({'id': name, 'split': row['split'], 'stratum': row['stratum'], 'group_id': row['group_id'], **value})
        folder = output / 'CV' / name; geometry = read(folder / 'geometry.json')
        if 'arrow_controls' in annotations[name]['review']:
            score = {**cv_score(annotations[name], target, geometry, read(folder / 'heads.json')), 'arrow_targets_measured': True}
        else:
            score = {'primitive_coverage': metrics.primitive_coverage(target, geometry),
                'primitive_candidates': len(geometry['candidates']), 'arrow_targets_measured': False,
                'assigned': None, 'matched': None, 'false_tips': None}
        current = {'id': name, 'split': row['split'], 'stratum': row['stratum'], 'group_id': row['group_id'], **score}
        require(current == next(r for r in saved_cv if r['id'] == name), 'Saved downstream metric differs')
        recomputed_cv.append(current)
    studies = [('original', [], [r for r in base_cv if r['variant'] == 'original']),
        ('unet-balanced-contrast-resize', [r for r in base_pixels if r['variant'] == 'balanced-contrast-resize'],
         [r for r in base_cv if r['variant'] == 'balanced-contrast-resize']),
        ('segformer-mit-b0', evaluations, recomputed_cv)]
    summaries = []
    for split in ('train', 'validation'):
        for stratum in ('all', 'monochrome', 'colored'):
            for label, pixels, cv_rows in studies:
                select = lambda r: r['split'] == split and (stratum == 'all' or r['stratum'] == stratum)
                px, cv_rows = [r for r in pixels if select(r)], [r for r in cv_rows if select(r)]
                summaries.append({'split': split, 'stratum': stratum, 'variant': label,
                    'pixel_roles': supported_summary(px), 'CV': cv_aggregate(cv_rows) if cv_rows else None})
    by_group = [{'group_id': group, 'split': next(r['split'] for r in rows if r['group_id'] == group),
                 'pixel_roles': supported_summary([r for r in evaluations if r['group_id'] == group]),
                 'CV': cv_aggregate([r for r in recomputed_cv if r['group_id'] == group])}
                for group in sorted({r['group_id'] for r in rows})]
    validation = supported_summary([r for r in evaluations if r['split'] == 'validation'])
    checks = gates(protocol, validation, recomputed_cv, studies[0][2], repeats_equal, resource_check(protocol, output))
    report = {'schema_version': 'segformer-comparison-report/1.0', 'experiment': EXPERIMENT, 'run_protocol': record,
        'summaries': summaries, 'by_group': by_group, 'necessary_partial_gates': checks,
        'necessary_partial_gates_pass': all(checks.values()),
        'decision': 'partial_gates_pass_research_only' if all(checks.values()) else 'reject_candidate',
        'repeat_parameters_equal': a['final_parameters_sha256'] == b['final_parameters_sha256'],
        'repeat_losses_equal': a['losses'] == b['losses'], 'repeat_predictions_equal': bool(repeats_equal),
        'dataset_training_started': True, 'dataset_train_runs': 2, 'updates_per_run': protocol['updates'],
        'pixel_ROIs': 25, 'CV_geometry_ROIs': 25, 'arrow_ROIs': 13, 'UNet_training_repeated': False,
        'validation_model_selection': False, 'closed_test_used': False, 'production_admitted': False,
        'physical_geometry_approved': False, 'independent_evaluation': False,
        'limitations': ['Partial assistant-reviewed islands/background only; no dense view/page truth.',
            'Monochrome train lacks frame; train leader and validation hatch each have one group.',
            'Color/brightness leakage and conservative source grouping limit transfer claims.',
            'Different pretrained encoder/decoder/normalization: pipeline comparison, not isolated causal architecture effect.',
            'Coverage of assigned ink is not CAD primitive accuracy or dimensional correctness.',
            'NVIDIA weights remain research/evaluation only; no commercial service admission.',
            'OCR, signs, VLM and physical reconstruction were not changed by this segmentation experiment.']}
    resources = {v: {'runtime': read(output / v / 'runtime.json'), 'model': {k: read(output / v / 'result/result.json')[k]
                    for k in ('training_seconds', 'allocated_peak_MiB', 'reserved_peak_MiB', 'inference')}} for v in protocol['variants']}
    resources['CV'] = read(output / 'CV-runtime.json')
    return {'report.json': report, 'pixel-evaluations.json': evaluations, 'downstream-evaluations.json': recomputed_cv, 'resources.json': resources}


def preview(root, protocol, rows, output, report):
    palette = np.array([[255,255,255],[230,60,60],[40,170,230],[220,160,0],[180,65,200],
                        [0,170,120],[25,25,25],[120,70,20],[100,120,180],[220,100,140]], np.uint8)
    folder = output / 'previews'; folder.mkdir()
    cards = []
    for row in rows:
        rgb = image_array(checked(root, row['image']))
        prediction = image_array(output / protocol['primary'] / 'result' / (row['id'] + '.png'))
        overlay = (rgb.astype(float) * .45 + palette[prediction] * .55).astype(np.uint8)
        panel = Image.new('RGB', (rgb.shape[1] * 2, rgb.shape[0]), 'white')
        panel.paste(Image.fromarray(rgb), (0, 0)); panel.paste(Image.fromarray(overlay), (rgb.shape[1], 0))
        panel.thumbnail((1400, 700)); panel.save(folder / (row['id'] + '.png'))
        cards.append('<figure><figcaption>' + escape(row['id'] + ' / ' + row['split']) + '</figcaption><img src="previews/' + escape(row['id']) + '.png"></figure>')
    html = '<!doctype html><meta charset="utf-8"><title>SegFormer partial comparison</title><style>body{font:16px system-ui;margin:30px;background:#eee}img{max-width:100%}figure{background:white;padding:12px}</style>'
    html += '<h1>P16.3b.2: ' + escape(report['decision']) + '</h1><p>Original RGB (left), predicted role overlay (right). Preview only; no dense or physical CAD acceptance.</p>'
    html += '<p><a href="report.json">Metrics and gates</a></p>' + ''.join(cards)
    (output / 'index.html').write_text(html, encoding='utf-8')


def run(root, record, output):
    from .hard_raster_runs_v1 import execute
    from .gpu_comparison_v1 import DeviceMemorySampler
    config, protocol, rows, _ = context(root, record)
    require(not output.exists(), 'Choose a fresh comparison directory'); output.mkdir(parents=True)
    for variant in protocol['variants']:
        with DeviceMemorySampler() as sampler:
            runtime = execute(root, [str(root / GPU_PYTHON), '-u', str(root / 'scripts/train_segformer_v1.py'),
                '--config', record['path'], '--sha256', record['sha256'], '--variant', variant,
                '--output', (output / variant / 'result').relative_to(root).as_posix()], output / variant / 'process',
                protocol['resources']['seconds_per_child'], protocol['resources']['host_MiB'])
        runtime['gpu_memory'] = sampler.report(); (output / variant / 'runtime.json').write_bytes(encoded(runtime))
        require(runtime['budget_pass'], 'Training child failed; inspect preserved logs before continuing')
        print('saved independent train run', variant, flush=True)
    runtime = execute(root, [sys.executable, str(root / 'scripts/run_segformer_comparison_v1.py'), 'cv',
        '--config', record['path'], '--output', output.relative_to(root).as_posix()], output / 'CV-process',
        config['CV']['seconds'], config['CV']['host_MiB'])
    (output / 'CV-runtime.json').write_bytes(encoded(runtime)); require(runtime['budget_pass'], 'Downstream child failed')
    values = summarize(root, record, output)
    for name, value in values.items(): (output / name).write_bytes(encoded(value))
    preview(root, protocol, rows, output, values['report.json'])
    files = sorted(p.relative_to(output).as_posix() for p in output.rglob('*') if p.is_file())
    (output / 'manifest.json').write_bytes(encoded({'schema_version': 'segformer-comparison-manifest/1.0', 'config': record,
        'files': [pin(output, p) for p in files]}))
    return values['report.json']


def verify(root, record, output):
    manifest = read(output / 'manifest.json')
    require(manifest['schema_version'] == 'segformer-comparison-manifest/1.0' and manifest['config'] == record, 'Changed comparison manifest')
    for row in manifest['files']: checked(output, row)
    values = summarize(root, record, output)
    for name, value in values.items(): require((output / name).read_bytes() == encoded(value), 'Recomputed report differs: ' + name)
    return {'verified': True, 'recomputed_files': len(values), 'training_repeated': False, 'CV_repeated': False}
