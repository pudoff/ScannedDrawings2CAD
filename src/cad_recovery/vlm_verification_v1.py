"""Replay VLM contracts/catalogs without loading weights or generating tokens."""
from collections import Counter

from .dataset_layout_v1 import checked, encoded, pin, read
from .vlm_adapter_v1 import POLICY, envelope, authorize_requests, require
from .vlm_protocol_v1 import context, catalog


def seal(root, record, output):
    require(not (output / 'manifest.json').exists(), 'Qualification already sealed')
    context(root, record)
    files = sorted(p.relative_to(output).as_posix() for p in output.rglob('*') if p.is_file())
    value = {'schema_version': 'vlm-qualification-manifest/1.0', 'protocol': record,
             'files': [pin(output, p) for p in files]}
    (output / 'manifest.json').write_bytes(encoded(value))


def verify(root, record, output):
    config, cases = context(root, record); cases_by_id = {r['id']: r for r in cases}
    require(len(cases_by_id) == len(cases), 'Duplicate VLM catalog identity')
    manifest = read(output / 'manifest.json')
    require(manifest['schema_version'] == 'vlm-qualification-manifest/1.0' and manifest['protocol'] == record, 'Changed qualification identity')
    require(len({r['path'] for r in manifest['files']}) == len(manifest['files']), 'Duplicate qualification file')
    for value in manifest['files']: checked(output, value)
    for value in config['assets']['files']: checked(root / config['assets']['directory'], value)
    actual, targets, exclusions, parents = catalog(root)
    require(actual == [r for r in cases if 'corpus' in r] and config['parents'] == parents, 'Catalog source/split/producer drift')
    saved_targets = read(checked(root, config['targets']))
    require(targets == [r for r in saved_targets if 'corpus' in r] and
            exclusions == read(checked(root, config['exclusions'])), 'Changed literal/unknown/exclusion targets')
    for case in cases: checked(root, case['image'])
    results = []; requests = None; checked_replies = 0
    for i in range(2):
        folder = output / ('repeat-' + str(i))
        result = read(folder / 'result/result.json'); runtime = read(folder / 'runtime.json')
        request = read(checked(root, result['request']))
        authorize_requests(cases_by_id, request, record)
        require(request['scope'] == 'synthetic_capacity_only', 'Qualification used customer crops')
        assigned = request['requests']; requests = assigned
        require([r['case_id'] for r in result['results']] == [r['case_id'] for r in assigned]
                and result['model_calls'] == len(assigned) == 5, 'Lost qualification requests')
        require(result['generation'] == {'do_sample': False, 'num_beams': 1, 'mode': 'greedy_search', 'max_new_tokens': 192}, 'Generation mode drift')
        require(result['dtype'] == 'torch.bfloat16' and result['training_executed'] is False
                and result['network_disabled'] is True, 'Changed model execution scope')
        require(runtime['budget_pass'] and not runtime['gpu_memory']['errors']
                and runtime['gpu_memory']['peak_MiB'] <= POLICY['device_MiB']
                and result['allocated_peak_MiB'] <= POLICY['allocated_MiB']
                and result['reserved_peak_MiB'] <= POLICY['reserved_MiB'], 'VLM resource admission drift')
        for row in result['results']:
            expected = envelope(cases_by_id[row['case_id']], row['raw'])
            require({k: v for k, v in row.items() if k != 'runtime'} == expected, 'Reply/provenance/review flags changed')
            stats = row['runtime']
            require(stats['seconds'] <= POLICY['request_seconds'] and stats['input_tokens'] <= POLICY['maximum_input_tokens']
                    and stats['image_tokens'] <= POLICY['maximum_image_tokens']
                    and len(stats['token_ids']) == stats['new_tokens'] <= POLICY['maximum_new_tokens'], 'Reply token/time budget drift')
            checked_replies += 1
        results.append(result)
    compare = lambda r: [(v['case_id'], v['raw'], v['status'], v['runtime']['token_ids']) for v in r['results']]
    require(compare(results[0]) == compare(results[1]), 'Independent VLM repeat differs')
    count = sum(v['reason'] in ('valid_proposal', 'model_abstention') for v in results[0]['results'])
    expected = {'schema_version': 'vlm-admission/1.0', 'task': 'P16.4a', 'protocol': record, 'qualified': True,
        'model_id': POLICY['model_id'], 'revision': POLICY['revision'], 'synthetic_cases': len(requests),
        'fresh_model_runs': 2, 'greedy_repeat_equal': True, 'valid_response_count': count,
        'customer_VLM_calls': 0, 'training_executed': False, 'graph_acceptance': False,
        'physical_geometry_approved': False, 'production_admitted': False, 'next_task': 'P16.4b',
        'catalog_counts': dict(Counter(c.get('corpus', 'capacity') + '/' + c['split'] for c in cases)),
        'note': 'Synthetic compatibility only; no OCR superiority, candidate correctness or CAD acceptance claim.'}
    require((output / 'admission.json').read_bytes() == encoded(expected), 'Recomputed admission differs')
    return {'verified': True, 'catalog_cases': len(cases), 'replayed_replies': checked_replies,
            'recomputed_admission_equal': True, 'model_loaded': False, 'new_model_calls': 0}
