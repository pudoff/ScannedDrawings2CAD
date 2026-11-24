"""Authenticated fragment catalog and synthetic-only VLM admission protocol."""
from collections import Counter
import hashlib
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont

from .architecture_data_readiness_v1 import Package, Boundary
from .dataset_layout_v1 import checked, encoded, inside, pin, read
from .vlm_adapter_v1 import POLICY, SYSTEM, RESPONSE_SCHEMA, validate_case, require, authorize_requests
from .segformer_protocol_v1 import environment, EXTRA, GPU_PYTHON

PREVIOUS = {'path': 'artifacts/segformer-comparison-release-v1.json',
            'sha256': '256d70c4586ad83f1678161a68638537a65235a90f70366bff0a3a8a81dea0a7'}
ASSETS = 'artifacts/vlm-assets-v1'
CODE = ['src/cad_recovery/vlm_adapter_v1.py', 'src/cad_recovery/vlm_runtime_v1.py',
        'src/cad_recovery/vlm_protocol_v1.py', 'scripts/vlm_worker_v1.py', 'scripts/prepare_vlm_v1.py',
        'scripts/download_vlm_assets_v1.py', 'src/cad_recovery/gpu_inference_v1.py',
        'src/cad_recovery/hard_raster_runs_v1.py', 'src/cad_recovery/gpu_comparison_v1.py',
        'src/cad_recovery/service_v1/process.py']


def assets(root):
    values = read(root / ASSETS / 'downloads.json')
    require(values['revision'] == POLICY['revision'] and values['model_id'] == POLICY['model_id'], 'Changed VLM revision')
    files = []
    for value in values['files']:
        record = {'path': value['path'], 'sha256': value['sha256']}; checked(root / ASSETS, record); files.append(record)
    require(next(r['sha256'] for r in files if r['path'] == 'model.safetensors') == POLICY['weights_sha256'], 'Wrong VLM weights')
    return {'directory': ASSETS, 'files': files, 'downloads': pin(root, ASSETS + '/downloads.json')}


def synthetic(root, folder):
    folder.mkdir(parents=True, exist_ok=False)
    font_path = Path('C:/Windows/Fonts/arial.ttf'); font = ImageFont.truetype(str(font_path), 52)
    source_font = {'filename': font_path.name, 'sha256': hashlib.sha256(font_path.read_bytes()).hexdigest()}
    cases = []; targets = []
    specs = [('probe-printed', (256, 128), '45°'), ('probe-title', (384, 128), 'Втулка'),
             ('probe-blank', (160, 128), None), ('probe-candidates', (512, 256), None),
             ('probe-largest', (1024, 1024), None)]
    for identifier, size, text in specs:
        image = Image.new('RGB', size, 'white'); draw = ImageDraw.Draw(image); candidates = []
        if text: draw.text((24, 24), text, font=font, fill='black')
        if identifier == 'probe-candidates':
            for i in range(8):
                y = 18 + 28 * i; draw.line((24, y, 488, y), fill='black', width=3)
                candidates.append({'id': 'line-' + str(i), 'kind': 'line', 'box_px': [24, y-2, 489, y+3]})
        if identifier == 'probe-largest':
            for y in range(24, 1000, 96): draw.text((24, y), 'A O 50 45° ±2', font=font, fill='black')
        path = folder / (identifier + '.png'); image.save(path)
        case = {'id': identifier, 'task': 'candidate' if candidates else 'literal', 'field_kind': 'text',
            'image': pin(root, path.relative_to(root).as_posix()), 'size_px': list(size), 'split': 'synthetic',
            'group_id': 'procedural-vlm-capacity', 'blocked': False, 'candidates': candidates,
            'provenance': {'kind': 'synthetic_capacity_only', 'source_font': source_font,
                'coordinate_scope': 'fragment_pixels_only', 'physical_geometry_approved': False}}
        validate_case(case); cases.append(case)
        targets.append({'id': identifier, 'literal': text, 'exact_target': text is not None, 'capacity_only': True})
    return cases, targets


def catalog(root):
    release = read(checked(root, PREVIOUS)); signatures = {**release['prior_protected_pins'], **release['files']}
    ready_name = 'artifacts/architecture-data-readiness-v3/manifest.json'
    ready = Package(root, {'path': ready_name, 'sha256': signatures[ready_name]})
    rc = read(checked(root, ready.manifest['config']))
    groups, sources, registry1, registry2, fresh = [read(checked(root, p)) for p in rc['metadata']]
    boundary = Boundary(groups['groups'], sources, [registry1, registry2],
        [{'source_path': s['input'], 'source_sha256': s['input_sha256']} for s in fresh['selected_sources']])
    items = ready.json('vlm-inventory.json')['items']; cases = []; targets = []; exclusions = []
    producer = Package(root, {'path': 'artifacts/component-signs-v1/result/manifest.json',
        'sha256': 'fba2809f095911d3b9039f819e5be00028151f03d86aea0b579584965f799c52'})
    for row in sorted(items, key=lambda r: (r['corpus'], r['id'])):
        identifier = 'case-' + row['corpus'] + '-' + hashlib.sha256(row['id'].encode()).hexdigest()[:16]
        if row['blocked']:
            exclusions.append({'id': identifier, 'upstream_id': row['id'], 'reasons': row['reasons']}); continue
        split = row['split']
        if split in ('train', 'validation'):
            result = boundary.check(row['source'], split, identifier) if 'source' in row else boundary.by_source_hash(row['source_sha256'], split, identifier)
            require(not result['blocked'] and result['group_id'] == row['group_id'], 'Closed/reassigned VLM source')
        elif split == 'development':
            sha = row.get('source_sha256', row.get('source', {}).get('sha256'))
            require(sha == 'd3baaca90795c9ccb85e3bcfa71245bb2182996c74132bc410e3ce2653e4ca14', 'Unknown viewed source')
        else: require(split == 'synthetic' and row['corpus'] == 'glyph', 'Unknown synthetic source')
        # Screen ancestry before opening even the image header.
        with Image.open(checked(root, row['image'])) as image: size = list(image.size)
        case = {'id': identifier, 'upstream_id': row['id'], 'corpus': row['corpus'],
            'task': 'candidate' if row['corpus'] == 'angular' else 'literal',
            'field_kind': 'glyph' if row['corpus'] == 'glyph' else 'text', 'split': split,
            'group_id': row['group_id'], 'blocked': False, 'image': row['image'], 'size_px': size, 'candidates': [],
            'provenance': {'source': row.get('source'), 'source_sha256': row.get('source_sha256'),
                'corpus': row['corpus'], 'upstream_id': row['id'], 'coordinate_scope': 'fragment_pixels_only',
                'context_to_source': row.get('context_to_source'), 'source_box': row.get('source_box'),
                'physical_geometry_approved': False}}
        if row['corpus'] == 'angular':
            name = 'ownership-' + row['id'] + '/page-001-angular-associations.json'
            value = producer.json(name); require(value['source_sha256'] == row['image']['sha256'], 'Producer/frame mismatch')
            mapping = []
            for association in value['associations']:
                for index, alternative in enumerate(association.get('alternatives', [])):
                    if len(mapping) == POLICY['maximum_candidates']: break
                    points = alternative.get('points_px', [])
                    if len(points) != 3: continue
                    left = max(0, min(p[0] for p in points)); top = max(0, min(p[1] for p in points))
                    right = min(size[0], max(p[0] for p in points) + 1); bottom = min(size[1], max(p[1] for p in points) + 1)
                    if left >= right or top >= bottom: continue
                    alias = association['id'] + '-alt-' + str(index)
                    case['candidates'].append({'id': alias, 'kind': 'dimension_link', 'box_px': [left, top, right, bottom]})
                    mapping.append({'id': alias, 'association_id': association['id'], 'alternative_index': index,
                        'producer': pin(root, producer.path(name).relative_to(root).as_posix()),
                        'measured_candidate_ids': alternative.get('measured_candidate_ids', []),
                        'arc_candidate_id': alternative.get('dimension_arc_candidate_id'),
                        'arrow_candidate_ids': alternative.get('arrow_candidate_ids', [])})
            case['provenance']['candidate_mapping'] = mapping
        try: validate_case(case)
        except ValueError as error:
            exclusions.append({'id': identifier, 'upstream_id': row['id'], 'reasons': [str(error)]}); continue
        cases.append(case)
        targets.append({'id': identifier, 'upstream_id': row['id'], 'corpus': row['corpus'],
            'split': split, 'literal': row.get('literal'), 'exact_target': row['exact_target'],
            'scope': row['scope'], 'class': row.get('class'), 'expected_points_px': row.get('expected_points_px'),
            'source_label_record': row.get('label_record'), 'independent_evaluation': False})
    require(len([r for r in exclusions if 'closed_source_path' in r['reasons']]) == 16, 'Changed historical closure')
    baseline_name = 'artifacts/architecture-baseline-v1/manifest.json'
    return cases, targets, exclusions, {'readiness': ready.record,
        'baseline': {'path': baseline_name, 'sha256': signatures[baseline_name]}}


def freeze(root, output, folder):
    require(not output.exists() and not folder.exists(), 'Choose fresh VLM protocol/catalog paths')
    model_assets = assets(root); cases, targets, exclusions, parents = catalog(root)
    probes, probe_targets = synthetic(root, folder / 'synthetic')
    cases += probes; targets += probe_targets
    for name, value in [('cases.json', cases), ('targets.json', targets), ('exclusions.json', exclusions)]:
        (folder / name).write_bytes(encoded(value))
    env = environment(root)
    qwen_files = ['models/qwen3_vl/modeling_qwen3_vl.py', 'models/qwen3_vl/configuration_qwen3_vl.py',
        'models/qwen3_vl/processing_qwen3_vl.py', 'models/qwen2_vl/image_processing_qwen2_vl_fast.py', 'generation/utils.py']
    value = {'schema_version': 'vlm-protocol/1.0', 'task': 'P16.4a', 'policy': POLICY, 'system_prompt': SYSTEM,
        'response_schema': RESPONSE_SCHEMA,
        'previous_release': PREVIOUS, 'parents': parents, 'assets': model_assets, 'environment': env,
        'dependency_directory': EXTRA, 'implementation': [pin(root, EXTRA + '/transformers/' + n) for n in qwen_files],
        'code': [pin(root, n) for n in CODE], 'catalog': pin(root, (folder / 'cases.json').relative_to(root).as_posix()),
        'targets': pin(root, (folder / 'targets.json').relative_to(root).as_posix()),
        'exclusions': pin(root, (folder / 'exclusions.json').relative_to(root).as_posix()),
        'customer_VLM_calls_started': False, 'training_admitted': False, 'production_admitted': False,
        'graph_acceptance': False, 'physical_geometry_approved': False}
    output.write_bytes(encoded(value))
    (folder / 'response-schema.json').write_bytes(encoded(RESPONSE_SCHEMA))
    request = {'schema_version': 'vlm-requests/1.0', 'scope': 'synthetic_capacity_only',
        'protocol': pin(root, output.relative_to(root).as_posix()),
        'requests': [{'case_id': r['id'], 'trigger': {'reason': 'capacity_probe'}} for r in probes]}
    (folder / 'qualification-request.json').write_bytes(encoded(request))
    return pin(root, output.relative_to(root).as_posix())


def context(root, record):
    value = read(checked(root, record))
    require(value['schema_version'] == 'vlm-protocol/1.0' and value['policy'] == POLICY
            and value['system_prompt'] == SYSTEM and value['response_schema'] == RESPONSE_SCHEMA
            and value['previous_release'] == PREVIOUS, 'Changed VLM protocol')
    require(value['code'] == [pin(root, n) for n in CODE] and value['environment'] == environment(root), 'Changed VLM code/environment')
    for row in value['implementation']: checked(root, row)
    for key in ('catalog', 'targets', 'exclusions'): checked(root, value[key])
    cases = read(checked(root, value['catalog']))
    for case in cases: validate_case(case)
    return value, cases


def execute_requests(root, record, request, output):
    from .hard_raster_runs_v1 import execute
    from .gpu_comparison_v1 import DeviceMemorySampler
    config, cases = context(root, record)
    batch = read(checked(root, request))
    admission = read(checked(root, batch['admission'])) if 'admission' in batch else None
    authorize_requests({c['id']: c for c in cases}, batch, record, admission)
    require(not output.exists(), 'Choose new VLM job output'); output.mkdir(parents=True)
    with DeviceMemorySampler() as sampler:
        runtime = execute(root, [str(root / GPU_PYTHON), '-u', str(root / 'scripts/vlm_worker_v1.py'),
            '--config', record['path'], '--sha256', record['sha256'], '--request', request['path'],
            '--request-sha256', request['sha256'], '--output', (output / 'result').relative_to(root).as_posix()],
            output / 'process', POLICY['child_seconds'], POLICY['host_MiB'])
    runtime['gpu_memory'] = sampler.report(); (output / 'runtime.json').write_bytes(encoded(runtime))
    require(runtime['budget_pass'], 'VLM child failed; retain logs and inspect before continuing')
    gpu = runtime['gpu_memory']
    require(not gpu['errors'] and gpu['peak_MiB'] is not None and gpu['peak_MiB'] <= POLICY['device_MiB'], 'VLM whole-device memory ceiling')
    return read(output / 'result/result.json')


def qualify(root, record, request, output):
    config, cases = context(root, record); assigned = read(checked(root, request))['requests']
    by_id = {c['id']: c for c in cases}
    require(len(assigned) == 5 and all(by_id[r['case_id']]['split'] == 'synthetic' for r in assigned), 'Capacity must not use customer crops')
    require(not output.exists(), 'Choose a new qualification directory'); output.mkdir(parents=True)
    results = [execute_requests(root, record, request, output / ('repeat-' + str(i))) for i in range(2)]
    for result in results:
        require(result['generation'] == {'do_sample': False, 'num_beams': 1, 'mode': 'greedy_search',
                'max_new_tokens': POLICY['maximum_new_tokens']}, 'Model defaults overrode greedy generation')
        require(result['model_calls'] == 5 and len(result['results']) == 5 and
                {r['case_id'] for r in result['results']} == {r['case_id'] for r in assigned}, 'Incomplete qualification requests')
        require(all(r['runtime']['seconds'] <= POLICY['request_seconds'] and
                    r['runtime']['input_tokens'] <= POLICY['maximum_input_tokens'] and
                    r['runtime']['image_tokens'] <= POLICY['maximum_image_tokens'] and
                    r['runtime']['new_tokens'] <= POLICY['maximum_new_tokens'] for r in result['results']),
                'VLM per-request budget exceeded')
    signatures = [[(r['case_id'], r['raw'], r['status'], r.get('runtime', {}).get('token_ids')) for r in v['results']] for v in results]
    require(signatures[0] == signatures[1], 'VLM greedy repeat differs')
    valid = [r for r in results[0]['results'] if r['reason'] in ('valid_proposal', 'model_abstention')]
    require(valid, 'Model produced no schema-valid synthetic response')
    report = {'schema_version': 'vlm-admission/1.0', 'task': 'P16.4a', 'protocol': record, 'qualified': True,
        'model_id': POLICY['model_id'], 'revision': POLICY['revision'], 'synthetic_cases': len(assigned),
        'fresh_model_runs': 2, 'greedy_repeat_equal': True, 'valid_response_count': len(valid),
        'customer_VLM_calls': 0, 'training_executed': False, 'graph_acceptance': False,
        'physical_geometry_approved': False, 'production_admitted': False, 'next_task': 'P16.4b',
        'catalog_counts': dict(Counter(c.get('corpus', 'capacity') + '/' + c['split'] for c in cases)),
        'note': 'Synthetic compatibility only; no OCR superiority, candidate correctness or CAD acceptance claim.'}
    (output / 'admission.json').write_bytes(encoded(report)); return report
