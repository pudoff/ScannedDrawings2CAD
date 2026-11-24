"""Frozen single-factor OCR comparison; preparation never runs model inference."""
from collections import Counter
from copy import deepcopy
import importlib.metadata
from pathlib import Path
import re
import shutil
import string

from .dataset_layout_v1 import checked, encoded, pin, read
from .architecture_data_readiness_v1 import Boundary, Package
from .ocr_baseline_metrics_v1 import COMPARISON_POLICY_V1

PREVIOUS = {'path': 'artifacts/dimension-chain-audit-release-v1.json',
            'sha256': '89b0b2fff5272f22456a09a25abaa692445f5dcc7c0b59c72090576809baa238'}
ADAPTER = {'path': 'configs/reviews/vlm-adapter-v2.json',
           'sha256': '66cfcb8f73955fa4b7ea452521513dd49a1506e5950166f232e3a3996967aa4b'}
ASSETS = 'artifacts/ocr-model-comparison-assets-v1'
POLICY = {
    'task': 'P16.2i.1', 'seed': 20261111, 'training_updates': 0, 'repeats': 2,
    'selection': 'all admitted OCR and glyph literals from fixed VLM v2 inventory; no predictions used',
    'primary': {'corpus': 'ocr', 'split': 'validation', 'count': 64},
    'cohorts': {'ocr/train': 105, 'ocr/validation': 64, 'glyph/train': 75,
                'glyph/validation': 50, 'glyph/development': 4, 'glyph/synthetic': 15},
    'primary_relative_raw_CER_reduction_min': .10,
    'primary_raw_exact_must_not_decrease': True,
    'maximum_correct_to_wrong_primary': 0,
    'maximum_primary_per_source_distance_regressions': 0,
    'maximum_correct_to_wrong_known_validation_glyphs': 0,
    'known_validation_glyph_exact_must_not_decrease': True,
    'maximum_lost_or_false_downstream_links': 0,
    'raw_is_primary': True, 'normalization': COMPARISON_POLICY_V1,
    'unknown': 'null is unscored; nonempty reading is review risk, not a proven false reading',
    'supported_negative_glyphs': ['circle_geometry', 'crossed_geometry'],
    'maximum_new_engineering_sign_predictions_on_supported_negatives': 0,
    'critical_codepoints': list('Ø⌀∅°±-−×φФ'),
    'critical_natural_minimum': {'per_symbol_known_instances': 2, 'per_symbol_source_groups': 2},
    'missing_critical_stratum': 'not_evaluable; prevents broad engineering-symbol admission',
    'missing_detection_boxes': 'no localization precision/recall/AP; literal yield only',
    'missing_downstream_labels': 'no service admission until a separate complete downstream regression',
    'failure_prediction': '', 'drop_failed_or_empty_cases': False,
    'selection_after_validation': 'forbidden; no threshold, checkpoint or alphabet changes',
    'provider': 'CPUExecutionProvider', 'intra_op_threads': 4, 'inter_op_threads': 1,
    'precision': 'FP32', 'gpu_comparison': 'not assigned; speed qualification separate',
    'cold_runs_per_arm': 2, 'warm_repeats_per_case': 3,
    'limits': {'case_pixels': 25_000_000, 'all_case_pixels': 30_000_000,
               'case_seconds': 15, 'worker_seconds': 600, 'worker_memory_MiB': 2048,
               'cold_init_seconds': 60, 'median_latency_ratio_max': 3.0},
    'timing': 'isolated sequential arm workers; no tests or other model workers in parallel',
    'memory': 'OS process-tree lifetime peaks, 50ms polling; not simultaneous RSS',
    'independent_evaluation': False, 'closed_test_used': False,
    'physical_geometry_approved': False, 'production_admitted': False,
    'yolo_enabled': False, 'vlm_enabled': False, 'retry_or_template_or_stroke_changes': False,
}
ARMS = {
    'R0': {'task': 'rec_only', 'recognizer': 'eslav_mobile', 'detector': None, 'orientation': 'fixed_input'},
    'R1': {'task': 'rec_only', 'recognizer': 'cyrillic_mobile', 'detector': None, 'orientation': 'fixed_input'},
    'D0': {'task': 'roi_pipeline', 'recognizer': 'eslav_mobile', 'detector': 'mobile', 'orientation': 'baseline_cls'},
    'D1': {'task': 'roi_pipeline', 'recognizer': 'eslav_mobile', 'detector': 'server', 'orientation': 'baseline_cls'},
}
BACKEND = {
    'Global.text_score': 0.0, 'Global.max_side_len': 2000,
    'Det.limit_type': 'max', 'Det.limit_side_len': 2000,
    'Det.thresh': .3, 'Det.box_thresh': .5, 'Det.unclip_ratio': 1.6,
    'Rec.rec_batch_num': 6, 'Rec.rec_img_shape': [3, 48, 320],
    'Cls.cls_image_shape': [3, 80, 160], 'Cls.cls_thresh': .9,
    'dict': 'embedded ONNX character; baseline greedy CTC with blank and space',
    'assembly': 'RapidOCR returned line order; join raw strings using one ASCII space; no sorting by target',
    'rec_input': 'same RGB image bytes converted to BGR; no cls/deskew/padding changes in R0/R1',
    'postprocessing': 'no character replacement in predictions; symmetric normalized metrics only',
}
RUNTIME_FILES = ['config.yaml', 'main.py', 'ch_ppocr_rec/main.py', 'ch_ppocr_rec/utils.py',
                 'ch_ppocr_cls/main.py', 'ch_ppocr_det/main.py', 'inference_engine/onnxruntime/main.py']
CODE = ['src/cad_recovery/ocr_model_comparison_protocol_v1.py',
        'scripts/prepare_ocr_model_comparison_v1.py', 'scripts/fetch_ocr_comparison_assets_v1.py']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_request(request):
    require(set(request) == {'id', 'image', 'size_px'}, 'Request includes labels or metadata')
    require(isinstance(request['id'], str) and re.fullmatch(r'sample-[0-9]{4}', request['id']), 'Opaque request ID required')
    width, height = request['size_px']
    require(all(type(v) is int and v > 0 for v in (width, height)) and
            width*height <= POLICY['limits']['case_pixels'], 'Invalid request dimensions')
    require(set(request['image']) == {'path', 'sha256'}, 'Image requires a pin')
    require(Path(request['image']['path']).name == request['id']+'.png', 'Opaque image filename required')


def validate_arms(arms):
    require(arms == ARMS, 'Confounded or changed comparison arms')


def protected(root):
    previous = read(checked(root, PREVIOUS))
    result = previous['prior_protected_pins'] | previous['files'] | {PREVIOUS['path']: PREVIOUS['sha256']}
    for name, sha in result.items():
        checked(root, {'path': name, 'sha256': sha})
    return result


def qualify_models(root):
    """Inspect ONNX IO/metadata via CPU sessions; never call session.run."""
    import onnxruntime as ort
    baseline = read(root/'src/cad_recovery/data/ocr-models-v1.json')
    alternatives = read(root/ASSETS/'models.json')['files']
    values = {m['role']: {'path': 'artifacts/models/ppocr-v5/'+m['filename'], 'sha256': m['sha256']}
              for m in baseline['models']}
    values = {'eslav_mobile': values['rec'], 'mobile': values['det'], 'baseline_cls': values['cls'],
              'cyrillic_mobile': {k: alternatives[0][k] for k in ('path', 'sha256')},
              'server': {k: alternatives[1][k] for k in ('path', 'sha256')}}
    require(importlib.metadata.version('rapidocr') == baseline['rapidocr_version'] and
            ort.__version__ == baseline['onnxruntime_version'], 'Runtime drift')
    core = string.ascii_letters + string.digits + ''.join(chr(c) for c in range(0x410, 0x450)) + 'Ёё'
    output = {}
    for name, record in values.items():
        path = checked(root, record)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
        def tensors(items):
            return [{'name': item.name, 'shape': item.shape, 'type': item.type} for item in items]
        entry = {'model': record, 'bytes': path.stat().st_size, 'providers': session.get_providers(),
                 'inputs': tensors(session.get_inputs()), 'outputs': tensors(session.get_outputs()),
                 'inference_calls': 0}
        characters = session.get_modelmeta().custom_metadata_map.get('character')
        if name in ('eslav_mobile', 'cyrillic_mobile'):
            require(characters is not None, 'Recognizer has no embedded dictionary')
            keys = characters.splitlines()
            require(len(keys) == len(set(keys)), 'Duplicate dictionary tokens')
            missing = sorted(set(core)-set(keys))
            require(not missing, 'Missing basic Cyrillic/Latin/digits')
            require(entry['outputs'][0]['shape'][-1] == len(keys)+2, 'CTC output/dictionary mismatch')
            entry.update(character_count=len(keys), dictionary=keys, missing_core_characters=missing,
                engineering_support={c: c in keys for c in POLICY['critical_codepoints']},
                dictionary_semantics='517/850 tokens plus CTC blank and space; dictionary presence is not accuracy')
        output[name] = entry
        del session
    return {'models': output, 'sessions_constructed': len(output), 'inference_calls': 0,
            'runtime': {name: importlib.metadata.version(name) for name in ('rapidocr', 'onnxruntime', 'numpy', 'opencv-python')},
            'scope': 'graph load, IO and dictionary only; no tensor execution or resource admission'}


def eligible_catalog(root):
    adapter = read(checked(root, ADAPTER))
    ready = Package(root, adapter['parents']['readiness'])
    config = read(checked(root, ready.manifest['config']))
    groups, sources, registry1, registry2, fresh = [read(checked(root, p)) for p in config['metadata']]
    boundary = Boundary(groups['groups'], sources, [registry1, registry2],
        [{'source_path': s['input'], 'source_sha256': s['input_sha256']} for s in fresh['selected_sources']])
    all_cases = read(checked(root, adapter['catalog']))
    targets = {r['id']: r for r in read(checked(root, adapter['targets']))}
    selected = sorted((r for r in all_cases if r.get('corpus') in ('ocr', 'glyph')), key=lambda r: r['id'])
    # Complete ancestry screening before opening or copying any image bytes.
    for row in selected:
        require(not row['blocked'] and row['id'] in targets, 'Blocked/missing literal')
        split, provenance = row['split'], row['provenance']
        if split in ('train', 'validation'):
            source = provenance.get('source')
            screening = (boundary.check(source, split, row['id']) if source else
                boundary.by_source_hash(provenance['source_sha256'], split, row['id']))
            require(not screening['blocked'] and screening['group_id'] == row['group_id'], 'Closed/reassigned ancestry')
        elif split == 'development':
            require(provenance['source_sha256'] == 'd3baaca90795c9ccb85e3bcfa71245bb2182996c74132bc410e3ce2653e4ca14', 'Unknown development source')
        else:
            require(split == 'synthetic' and row['corpus'] == 'glyph', 'Unknown synthetic corpus')
    counts = Counter(r['corpus']+'/'+r['split'] for r in selected)
    require(dict(counts) == POLICY['cohorts'], 'Comparison cohort changed')
    return selected, targets, {'adapter': ADAPTER, 'catalog': adapter['catalog'], 'targets': adapter['targets'],
                               'readiness': adapter['parents']['readiness']}


def target_record(identifier, row, target, fields):
    return {'id': identifier, 'literal': target['literal'], 'exact_target': target['exact_target'],
            'corpus': row['corpus'], 'split': row['split'], 'group_id': row['group_id'],
            'field': fields.get(row['upstream_id'], 'glyph' if row['corpus'] == 'glyph' else 'train_text'),
            'class': target.get('class'), 'independent_evaluation': False}


def coverage_record(requests, targets):
    primary = [r for r in targets if r['corpus'] == 'ocr' and r['split'] == 'validation']
    require(len(primary) == 64 and all(r['exact_target'] for r in primary), 'Missing primary literals')
    support = {}
    for corpus in ('ocr', 'glyph'):
        cohort = [r for r in targets if r['corpus'] == corpus and r['split'] == 'validation' and r['exact_target']]
        support[corpus] = {c: {'raw_instances': sum(t['literal'].count(c) for t in cohort),
            'source_groups': len({t['group_id'] for t in cohort if c in t['literal']})}
            for c in POLICY['critical_codepoints']}
    return {'primary_fields': dict(Counter(r['field'] for r in primary)),
            'cohorts': dict(Counter(r['corpus']+'/'+r['split'] for r in targets)),
            'unknown_literals': sum(not r['exact_target'] for r in targets),
            'total_pixels': sum(r['size_px'][0]*r['size_px'][1] for r in requests),
            'detection_box_metrics_evaluable': False, 'downstream_links_evaluable': False,
            'critical_natural_validation': support}


def freeze(root, destination, folder):
    root, destination, folder = root.resolve(), destination.resolve(), folder.resolve()
    require(destination.is_relative_to(root/'configs') and folder.is_relative_to(root/'artifacts'),
            'Protocol/data must stay in workspace configs/artifacts')
    require(not destination.exists() and not folder.exists(), 'Use new protocol/data paths')
    pins = protected(root)
    qualification = qualify_models(root)
    selected, labels, parents = eligible_catalog(root)
    fields = {r['id']: r['field'] for r in read(root/'artifacts/architecture-baseline-v1/ocr-baseline.json')['rows']}
    requests, targets, lineage = [], [], []
    folder.mkdir(parents=True)
    (folder/'images').mkdir()
    for index, row in enumerate(selected, 1):
        identifier = f'sample-{index:04d}'
        original = checked(root, row['image'])
        path = folder/'images'/(identifier+'.png')
        shutil.copyfile(original, path)
        request = {'id': identifier, 'image': pin(root, path.relative_to(root).as_posix()), 'size_px': row['size_px']}
        validate_request(request)
        target = labels[row['id']]
        targets.append(target_record(identifier, row, target, fields))
        lineage.append({'id': identifier, 'upstream_id': row['id'], 'image': row['image'],
                        'provenance': row['provenance'], 'split': row['split'], 'group_id': row['group_id']})
        requests.append(request)
    total_pixels = sum(r['size_px'][0]*r['size_px'][1] for r in requests)
    require(total_pixels <= POLICY['limits']['all_case_pixels'], 'Total image budget exceeded')
    for name, payload in [('requests.json', requests), ('targets.json', targets), ('lineage.json', lineage),
                          ('model-qualification.json', qualification)]:
        (folder/name).write_bytes(encoded(payload))
    coverage = coverage_record(requests, targets)
    value = {'schema_version': 'ocr-model-comparison-protocol/1.0', 'policy': POLICY,
        'arms': ARMS, 'backend': BACKEND, 'previous': PREVIOUS, 'previous_pins': pins,
        'parents': parents, 'data': {n: pin(root, (folder/(n+'.json')).relative_to(root).as_posix())
                                    for n in ('requests', 'targets', 'lineage', 'model-qualification')},
        'assets': [pin(root, p.relative_to(root).as_posix()) for p in sorted((root/ASSETS).iterdir()) if p.is_file()],
        'baseline_model_manifest': pin(root, 'src/cad_recovery/data/ocr-models-v1.json'),
        'baseline_engine': pin(root, 'src/cad_recovery/ocr_engine.py'),
        'runtime_implementation': [pin(root, '.venv/Lib/site-packages/rapidocr/'+name) for name in RUNTIME_FILES],
        'assignments': {arm: [t['id'] for t in targets if arm in ('R0', 'R1') or t['corpus'] == 'ocr'] for arm in ARMS},
        'code': [pin(root, p) for p in CODE], 'coverage': coverage,
        'customer_inference_calls': 0, 'candidate_quality_evaluated': False}
    destination.write_bytes(encoded(value))
    return verify(root, pin(root, destination.relative_to(root).as_posix()))


def verify(root, protocol):
    value = read(checked(root, protocol))
    require(value['schema_version'] == 'ocr-model-comparison-protocol/1.0' and value['policy'] == POLICY and
            value['backend'] == BACKEND, 'Protocol policy drift')
    validate_arms(value['arms'])
    require(value['previous'] == PREVIOUS, 'Previous release changed')
    for name, sha in value['previous_pins'].items():
        checked(root, {'path': name, 'sha256': sha})
    for record in [*value['assets'], *value['code'], value['baseline_model_manifest'], value['baseline_engine'],
                   *value['runtime_implementation'], *value['data'].values()]:
        checked(root, record)
    requests = read(checked(root, value['data']['requests']))
    targets = read(checked(root, value['data']['targets']))
    lineage = read(checked(root, value['data']['lineage']))
    expected, labels, parents = eligible_catalog(root)
    require(value['parents'] == parents and len(requests) == len(targets) == len(lineage) == len(expected) == 313,
            'Incomplete comparison assignment')
    fields = {r['id']: r['field'] for r in read(root/'artifacts/architecture-baseline-v1/ocr-baseline.json')['rows']}
    for index, (request, target, origin, upstream) in enumerate(zip(requests, targets, lineage, expected), 1):
        validate_request(request)
        require(request['id'] == target['id'] == origin['id'] == f'sample-{index:04d}' and
                origin['upstream_id'] == upstream['id'], 'Shuffled case identity')
        require(origin['image'] == upstream['image'] and origin['provenance'] == upstream['provenance'], 'Changed image ancestry')
        require(request['image']['sha256'] == origin['image']['sha256'] and request['size_px'] == upstream['size_px'], 'Changed request pixels')
        require(target == target_record(request['id'], upstream, labels[upstream['id']], fields), 'Changed targets or split')
        require(origin['split'] == upstream['split'] and origin['group_id'] == upstream['group_id'], 'Changed lineage group')
        checked(root, request['image'])
    require(value['coverage'] == coverage_record(requests, targets), 'Coverage counters changed')
    require(value['assignments'] == {arm: [t['id'] for t in targets if arm in ('R0', 'R1') or t['corpus'] == 'ocr']
                                    for arm in ARMS}, 'Changed arm assignments')
    return {'protocol': protocol, 'requests_verified': len(requests), 'quality_calls': 0,
            'coverage': deepcopy(value['coverage']), 'candidate_quality_evaluated': False}
