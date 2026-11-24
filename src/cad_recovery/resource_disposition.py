"""Review incomplete native resources and exclude their pinned training targets.

No source, external asset, OLE application or renderer is modified or invoked.
An exclusion is a dataset decision, not a claim that the reference is repaired.
"""
from collections import Counter
from copy import deepcopy
import html
import json
import os
from pathlib import Path

import ezdxf

from .clear_pairs import contained, digest, read_json
from .pair_update_v3_validation import _native_resource_instances
from .resource_native_inventory import inspect_native_resource
from .training_target_policy import build_exclusion_manifest

MODULES = {'resource_disposition.py', 'resource_native_inventory.py', 'training_target_policy.py',
           'clear_pairs.py', 'pair_update_v3_validation.py', 'pair_update_validation.py'}
DEPENDENCIES = {'parent_handoff', 'resource_queue', 'parent_run', 'dataset_manifest'}
DECISION = 'exclude_incomplete_native_target'
REENTRY = ['establish_original_resource_provenance', 'verify_complete_resource_rendering',
           'publish_new_pinned_reference_revision', 'repeat_pair_and_object_review']


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _write(path, data):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(data, stream, ensure_ascii=True, indent=2, allow_nan=False)
        stream.write('\n')


def _index(rows, key):
    result = {row[key]: row for row in rows}
    _require(len(result) == len(rows), 'Duplicate inventory identity: '+key)
    return result


def _check_dataset(dataset, records):
    indexed = _index(records, 'path')
    _require(0 < len(records) <= 50000, 'Invalid registered dataset size')
    actual = {p.relative_to(dataset).as_posix() for p in dataset.rglob('*') if p.is_file()}
    _require(actual == set(indexed), 'Dataset file inventory changed; register a new intake first')
    for name, record in indexed.items():
        path = contained(dataset, name)
        _require(path.stat().st_size == record['size_bytes'] and digest(path) == record['sha256'],
                 'Registered source bytes changed: '+name)


def _resource_coverage(document, observations):
    """Match independent native reachability without inventing callback paths."""
    remaining = _native_resource_instances(document)
    remaining = [row for row in remaining if row['entity_type'] != 'IMAGE'
                 or int(document.entitydb[row['native_handle']].dxf.flags) & 1]
    matched = []
    for observation in observations:
        _require(observation.get('resource_required') is True and observation.get('loaded') is False,
                 'Expected a required unresolved resource observation')
        identity = ('entity_type', 'native_handle', 'native_owner_handle')
        choices = [row for row in remaining if all(row[key] == observation[key] for key in identity)]
        path = observation['insert_path']
        if path is None:
            _require(observation.get('observation_kind') == 'skipped_before_resource_callback'
                     and len(choices) == 1, 'Skipped resource has ambiguous native reachability')
        else:
            choices = [row for row in choices if row['insert_path'] == path]
        _require(bool(choices), 'Resource observation differs from native reachability')
        selected = choices[0]
        remaining.remove(selected)
        matched.append(selected)
    _require(not remaining, 'Resource observation inventory omits visible native instances')
    return matched


def run_resource_disposition(workspace, config_path, output):
    root, config_path, output = (Path(p).resolve() for p in (workspace, config_path, output))
    _require(config_path.is_relative_to(root), 'Configuration must be inside workspace')
    _require(output.is_relative_to(root/'artifacts') and output != root/'artifacts',
             'Output must be a new artifacts subdirectory')
    if output.exists():
        raise FileExistsError('Preserve existing results; choose a new output directory')
    config_sha = digest(config_path)
    config = read_json(config_path)
    _require(config.get('schema_version') == '1.0'
             and config.get('purpose') == 'incomplete_native_target_disposition'
             and config.get('unrestored_reference_action') == DECISION
             and config.get('training_eligible') is False and config.get('split_modified') is False,
             'Expected an explicit exclusion policy without label or split approval')
    _require(set(config.get('implementation', {})) == MODULES, 'Incomplete implementation pins')
    dependencies = config.get('dependencies', {})
    _require(DEPENDENCIES <= set(dependencies), 'Missing disposition dependencies')
    inputs = [(config_path, config_sha)]

    def pinned(record):
        path = contained(root, record['path'])
        _require(digest(path) == record['sha256'], 'Pinned file changed: '+record['path'])
        inputs.append((path, record['sha256']))
        return path

    paths = {name: pinned(record) for name, record in dependencies.items()}
    for name, expected in config['implementation'].items():
        path = Path(__file__).parent/name
        _require(digest(path) == expected, 'Disposition implementation changed')
        inputs.append((path, expected))
    for record in config.get('library_dependencies', {}).values():
        pinned(record)
    for name, expected in config.get('entrypoints', {}).items():
        pinned({'path': name, 'sha256': expected})
    dataset = contained(root, config['dataset'])
    _require(not output.is_relative_to(dataset)
             and not any(output.is_relative_to(paths[name].parent) for name in DEPENDENCIES),
             'Output must not extend sealed input artifact directories')
    canonical_report = contained(root, config['canonical_report_path'])
    _require(canonical_report.name == 'report.json'
             and canonical_report.parent != root/'artifacts'
             and canonical_report.is_relative_to(root/'artifacts')
             and not canonical_report.is_relative_to(dataset)
             and not any(canonical_report.is_relative_to(paths[name].parent) for name in DEPENDENCIES),
             'Canonical report must be in a separate artifacts directory')
    _require(output == canonical_report.parent or not output.is_relative_to(canonical_report.parent),
             'Repeat output must not extend the canonical report directory')
    if output != canonical_report.parent:
        _require(canonical_report.is_file(), 'Create the canonical report before a repeated review')
    parent, queue, run, inventory = (read_json(paths[name]) for name in
        ('parent_handoff', 'resource_queue', 'parent_run', 'dataset_manifest'))
    _require(parent.get('training_eligible') is False and parent.get('split_modified') is False,
             'Parent handoff unexpectedly approved training or splits')
    _require(queue.get('catalog_report_sha256') == dependencies['parent_handoff']['sha256']
             and queue.get('training_eligible') is False and queue.get('split_modified') is False,
             'Resource queue is not the unapproved parent handoff queue')
    _require(run.get('complete') is True and run.get('status') == 'complete'
             and run.get('training_eligible') is False, 'Expected a complete unapproved parent run')
    entries = _index(parent['entries'], 'qualified_pair_id')
    _index(parent['entries'], 'input_path')
    queued = _index(queue['items'], 'qualified_pair_id')
    incomplete = {key: row for key, row in entries.items() if row.get('reference_complete') is False}
    _require(queued == incomplete and queued, 'Queue must exactly cover all incomplete parent references')
    pairs = _index(run['pairs'], 'input_path')
    registered = _index(inventory['files'], 'path')
    expected = config['expected_counts']
    _require(len(entries) == expected['parent_entries'] and len(queued) == expected['source_cases']
             and len(registered) == expected['source_files'], 'Pinned source inventory counts differ')
    _check_dataset(dataset, inventory['files'])
    cases = []
    total_resources = 0
    for row in queue['items']:
        _require(row['handoff_status'] == 'incomplete_reference_quarantine'
                 and row['training_eligible'] is False and row['annotation_approved'] is False
                 and row['split'] is None, 'Resource disposition cannot bypass parent quarantine')
        pair_path = pinned(row['pair_report'])
        pair = read_json(pair_path)
        _require(pair == pairs.get(row['input_path']) and pair.get('reference_complete') is False
                 and pair.get('annotation_status') == 'not_approved' and pair.get('training_eligible') is False,
                 'Queued pair differs from unapproved frozen run')
        reference_path = pinned(row['reference'])
        expected_reference = contained(paths['parent_run'].parent, pair['artifacts']['reference'])
        _require(reference_path == expected_reference and pair_path == reference_path.parent/'pair.json'
                 and digest(reference_path) == pair['artifact_sha256']['reference'], 'Reference location or hash differs')
        reference = read_json(reference_path)
        for side in ('input', 'target'):
            name, sha = row[side+'_path'], row[side+'_sha256']
            _require(pair[side+'_path'] == name and pair[side+'_sha256'] == sha
                     and registered[name]['sha256'] == sha, 'Queued source differs from registered bytes')
        _require(reference.get('source_sha256') == row['target_sha256']
                 and reference.get('reference_complete') is False
                 and reference.get('resource_skip_audit_complete') is True
                 and reference['resource_observations'] == row['resource_observations'],
                 'Incomplete reference resource evidence changed')
        observations = [r for r in row['resource_observations']
                        if r.get('resource_required') is True and r.get('loaded') is False]
        _require(observations and len(observations) == row['missing_resource_count'],
                 'Incomplete reference resource counts differ')
        source = contained(dataset, row['target_path'])
        _require(0 < source.stat().st_size <= 50*1024**2, 'Native source exceeds inspection byte limit')
        document = ezdxf.readfile(source)
        _require(len(document.entitydb) <= 200000, 'Native source exceeds inspection entity limit')
        instances = _resource_coverage(document, observations)
        resources = []
        for observation, instance in zip(observations, instances):
            _require(observation['source_sha256'] == row['target_sha256'], 'Resource source identity differs')
            facts = inspect_native_resource(document, observation, inventory['files'])
            _require(facts['restored'] is False and facts['resource_rendering_verified'] is False
                     and facts['source_modified'] is False, 'Inspection cannot approve a restoration')
            resources.append({'observation': deepcopy(observation), 'native_instance': instance, 'native': facts})
        cases.append({'case_id': row['qualified_pair_id'],
            **{key: deepcopy(row[key]) for key in ('input_path', 'input_sha256', 'target_path', 'target_sha256',
                                                   'pair_report', 'reference')},
            'resources': resources, 'decision': DECISION,
            'reason_codes': sorted({r['native']['reason'] for r in resources}),
            'scope': 'native_dxf_revision_and_derived_labels',
            'input_retained': True, 'native_target_retained': True, 'reference_complete': False,
            'restored': False, 'reentry_requires': list(REENTRY),
            'training_eligible': False, 'annotation_approved': False, 'split': None})
        total_resources += len(resources)
    _require(total_resources == expected['resource_instances'], 'Required resource instance coverage differs')
    _check_dataset(dataset, inventory['files'])
    for path, expected_sha in inputs:
        _require(digest(path) == expected_sha, 'Pinned input changed during resource review')
    report = {'schema_version': '1.0', 'date': config['date'], 'status': 'completed_target_exclusions',
        'purpose': 'incomplete_native_target_disposition_not_resource_repair',
        'config': {'path': config_path.relative_to(root).as_posix(), 'sha256': config_sha},
        'dependencies': dependencies, 'implementation': config['implementation'], 'cases': cases,
        'summary': {'reviewed_sources': len(cases), 'required_resource_instances': total_resources,
                    'excluded_target_hashes': len({r['target_sha256'] for r in cases}), 'restored_references': 0,
                    'resource_types': dict(Counter(r['native']['resource_kind'] for c in cases for r in c['resources'])),
                    'resource_reasons': dict(Counter(r['native']['reason'] for c in cases for r in c['resources'])),
                    'retained_source_files': len(registered), 'other_parent_entries_not_approved': len(entries)-len(cases)},
        'source_modified': False, 'split_modified': False, 'training_eligible': False,
        'annotation_approved': False, 'limitations': [
            'Exclusion resolves the training-target decision, not resource completeness or original drawing quality.',
            'Filename matches do not prove external image provenance; no external path is opened or rebound.',
            'Embedded OLE bytes are retained evidence, not a verified rendering or extracted presentation.',
            'Inputs and native targets remain on disk; derived labels from excluded native revisions are blocked.',
            'Targets absent from the exclusion list still need independent annotation and split approval.',
            'Restoration requires a new sourced reference revision and new complete render, pair and object review.']}
    output.mkdir(parents=True, exist_ok=False)
    _write(output/'report.json', report)
    report_sha = digest(output/'report.json')
    _require(digest(canonical_report) == report_sha, 'Repeated report differs from its canonical evidence')
    report_pin = {'path': canonical_report.relative_to(root).as_posix(), 'sha256': report_sha}
    exclusions = build_exclusion_manifest(report_pin, cases)
    _write(output/'excluded-targets.json', exclusions)
    _write(output/'reentry-queue.json', {'schema_version': '1.0', 'decision_report': report_pin,
        'purpose': 'optional_future_restoration_of_excluded_targets', 'training_eligible': False,
        'items': [{key: deepcopy(row[key]) for key in ('case_id', 'target_path', 'target_sha256', 'reference',
                  'reason_codes', 'reentry_requires')} for row in cases]})
    rows = []
    for case in cases:
        link = Path(os.path.relpath(root/case['reference']['path'], output)).as_posix()
        values = (case['case_id'], case['input_path'], case['target_sha256'], ', '.join(case['reason_codes']))
        rows.append('<tr>'+''.join('<td>'+html.escape(value)+'</td>' for value in values)
                    +'<td><a href="'+html.escape(link, quote=True)+'">Reference JSON</a></td></tr>')
    (output/'index.html').write_text('<!doctype html><html lang="ru"><meta charset="utf-8">'
        '<title>Решения по неполным эталонам</title><style>body{font:14px system-ui;margin:24px}'
        'table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:6px}</style>'
        '<h1>Неполные DXF исключены из обучающих целей</h1>'
        '<p>Исходные PNG и DXF сохранены. Ресурсы не восстановлены; остальные цели не утверждены.</p>'
        '<p><a href="report.json">Решения и нативные наблюдения</a> · '
        '<a href="excluded-targets.json">Исключения по SHA-256</a> · '
        '<a href="reentry-queue.json">Условия будущего восстановления</a></p>'
        '<table><tr><th>ID</th><th>PNG</th><th>Версия DXF</th><th>Причины</th><th>Происхождение</th></tr>'
        +''.join(rows)+'</table></html>', encoding='utf-8')
    _write(output/'manifest.json', {'schema_version': '1.0', 'artifact_sha256': {
        p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()}})
    return report
