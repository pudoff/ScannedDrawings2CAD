"""Register dataset revisions while keeping previous evidence and split boundaries."""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import html
from pathlib import Path

from .clear_pairs import contained, digest, read_json
from .object_review_case import _pinned, _write_json
from .png_stream import MAX_FILE_BYTES, MAX_SIDE, MAX_SOURCE_PIXELS


def _index(records, key):
    if not isinstance(records, list) or len(records) > 100_000:
        raise ValueError('Invalid revision inventory')
    result = {}
    for record in records:
        name = record.get(key)
        if not isinstance(name, str) or not name or name in result:
            raise ValueError('Duplicate or invalid inventory identity')
        result[name] = record
    return result


def _identity(item):
    target = item.get('candidate_target')
    return None if target is None else (target['path'], target['sha256'])


def _header_support(item):
    h = item['png_ihdr']
    reasons = []
    if (h['bit_depth'], h['color_type'], h['interlace_method']) != (8, 2, 0):
        reasons.append('unsupported_pixel_encoding')
    if (max(h['width_px'], h['height_px']) > MAX_SIDE or h['pixel_count'] > MAX_SOURCE_PIXELS):
        reasons.append('source_dimensions_exceed_decoder_limits')
    if item['input_size_bytes'] > MAX_FILE_BYTES:
        reasons.append('source_bytes_exceed_decoder_limit')
    return {'status': 'header_compatible' if not reasons else 'requires_decoder_review',
            'reasons': reasons, 'full_stream_integrity_verified': False,
            'working_buffer_budget_checked': False}


def register_dataset_update(workspace: Path, config_path: Path, output: Path) -> dict:
    root, config_path, output = Path(workspace).resolve(), Path(config_path).resolve(), Path(output).resolve()
    if not config_path.is_relative_to(root) or not output.is_relative_to(root) or output == root:
        raise ValueError('Revision paths must remain within workspace')
    if output.exists():
        raise FileExistsError('Choose a new dataset revision directory')
    config = read_json(config_path)
    if config.get('schema_version') != '1.0':
        raise ValueError('Unknown dataset revision schema')
    dataset = contained(root, config['dataset'])
    if output.is_relative_to(dataset):
        raise ValueError('Revision output must be outside the source dataset')
    keys = ('previous_manifest', 'current_manifest', 'previous_intake', 'current_intake',
            'old_preparation', 'old_pairs', 'family_catalog', 'old_split')
    documents, paths = {}, {}
    for key in keys:
        dep = config['dependencies'][key]
        paths[key] = contained(root, dep['path'])
        _pinned(paths[key], dep['sha256'])
        documents[key] = read_json(paths[key])
    old, current = (_index(documents[k]['files'], 'path') for k in ('previous_manifest', 'current_manifest'))
    actual = {p.relative_to(dataset).as_posix() for p in dataset.rglob('*') if p.is_file() or p.is_symlink()}
    if actual != set(current):
        raise ValueError('Dataset file set differs from current manifest')
    for name, entry in current.items():
        path = contained(dataset, name)
        _pinned(path, entry['sha256'])
        if path.stat().st_size != entry['size_bytes']:
            raise ValueError('Dataset file size differs from current manifest')
    previous, latest = (_index(documents[k]['items'], 'input_path') for k in ('previous_intake', 'current_intake'))
    for key, entries, manifest_key, manifest in [('previous_intake', previous, 'previous_manifest', old),
                                               ('current_intake', latest, 'current_manifest', current)]:
        snapshot = documents[key]
        if snapshot.get('purpose') != 'training_intake_not_split' or snapshot.get('training_eligible') is not False:
            raise ValueError('Expected an unapproved training intake')
        for dependency, parent_key in [('manifest', manifest_key), ('family_catalog', 'family_catalog'), ('old_split', 'old_split')]:
            if snapshot['dependencies'][dependency]['sha256'] != config['dependencies'][parent_key]['sha256']:
                raise ValueError('Intake dependency differs from revision dependency')
        expected_pngs = {n for n in manifest if n.startswith('clear_sum_dataset/') and Path(n).suffix.lower() == '.png'}
        if set(entries) != expected_pngs:
            raise ValueError('Intake does not cover its complete PNG inventory')
        for name, item in entries.items():
            if item['input_sha256'] != manifest[name]['sha256'] or item['input_size_bytes'] != manifest[name]['size_bytes']:
                raise ValueError('Intake source differs from its manifest')
            target = item.get('candidate_target')
            if target and (target['path'] not in manifest or target['sha256'] != manifest[target['path']]['sha256']):
                raise ValueError('Intake target differs from its manifest')
    changes = {
        'added': sorted(set(current)-set(old)), 'removed': sorted(set(old)-set(current)),
        'modified': sorted(n for n in set(current) & set(old) if current[n]['sha256'] != old[n]['sha256']),
        'unchanged': sorted(n for n in set(current) & set(old) if current[n]['sha256'] == old[n]['sha256']),
    }
    batch = documents['old_preparation']
    if batch.get('status') != 'complete' or batch['intake_sha256'] != config['dependencies']['previous_intake']['sha256']:
        raise ValueError('Old preparation is incomplete or belongs to another intake')
    prepared = _index(batch['items'], 'input_path')
    if set(prepared) != set(previous):
        raise ValueError('Old preparation inventory differs from previous intake')
    pairs = _index(documents['old_pairs']['pairs'], 'input_path')
    if (documents['old_pairs'].get('complete') is not True
            or documents['old_pairs']['intake_sha256'] != config['dependencies']['previous_intake']['sha256']
            or documents['old_pairs']['preparation_batch_sha256'] != config['dependencies']['old_preparation']['sha256']
            or set(pairs) != set(previous)):
        raise ValueError('Old pairs do not pin the preparation history')
    for name, pair in pairs.items():
        if pair['input_sha256'] != previous[name]['input_sha256']:
            raise ValueError('Previous pair source differs from intake')
        if pair.get('target_path') is not None:
            target = old.get(pair['target_path'])
            if target is None or target['sha256'] != pair.get('target_sha256'):
                raise ValueError('Previous pair target differs from previous manifest')
    memberships, families, old_hashes = defaultdict(list), defaultdict(list), defaultdict(list)
    for split, records in documents['old_split']['splits'].items():
        for record in records:
            for role in ('input', 'target'):
                if record.get(role+'_sha256'):
                    memberships[record[role+'_sha256']].append({'split': split, 'matched_role': role,
                        'input_path': record['input_path'], 'target_path': record['target_path']})
    for record in documents['family_catalog']['families']:
        families[record['target_sha256']].append(record)
    for name, record in old.items():
        old_hashes[record['sha256']].append(name)
    queue, delta, review_items, reused_files = [], [], [], 0
    for name, item in sorted(latest.items()):
        prior = previous.get(name)
        raster_unchanged = prior is not None and prior['input_sha256'] == item['input_sha256']
        pair_unchanged = raster_unchanged and _identity(prior) == _identity(item)
        prior_pair = pairs.get(name)
        if prior_pair and prior_pair.get('target_path'):
            current_target = current.get(prior_pair['target_path'])
            pair_unchanged = pair_unchanged and current_target is not None and current_target['sha256'] == prior_pair['target_sha256']
        reference = None
        if raster_unchanged:
            batch_item, pair = prepared[name], pairs[name]
            if (batch_item['status'] != 'prepared' or batch_item['input_sha256'] != item['input_sha256']
                    or pair['input_sha256'] != item['input_sha256']):
                raise ValueError('Previous preparation source identity differs')
            prep = contained(paths['old_preparation'].parent, batch_item['preparation_report'])
            pin = pair['source_provenance']['preparation_report_sha256']
            _pinned(prep, pin)
            metadata = read_json(prep)
            if metadata.get('status') != 'prepared' or metadata.get('source_integrity_validated') is not True or metadata['source_sha256'] != item['input_sha256']:
                raise ValueError('Previous preparation metadata is not valid for reuse')
            artifacts = [metadata['overview'], metadata['ink_overview'], *metadata['tiles']]
            for artifact in artifacts:
                _pinned(contained(prep.parent, artifact['path']), artifact['sha256'])
            reused_files += len(artifacts)
            reference = {'path': prep.relative_to(root).as_posix(), 'sha256': pin,
                         'batch_report_sha256': config['dependencies']['old_preparation']['sha256'],
                         'original_output_directory': batch_item['output_directory'],
                         'artifact_files_rehashed': len(artifacts)}
        else:
            delta.append(deepcopy(item))
        if not pair_unchanged:
            review_items.append(deepcopy(item))
        target = item.get('candidate_target')
        target_basis = 'current_intake_candidate'
        if target is None and raster_unchanged and prior_pair.get('target_path') in current:
            target = {'path': prior_pair['target_path'], 'sha256': current[prior_pair['target_path']]['sha256']}
            target_basis = 'previous_pair_target_path_current_hash_candidate_only'
        hits = []
        for role, sha in [('input', item['input_sha256']), ('target', target['sha256'] if target else None)]:
            hits.extend({**hit, 'matched_current_role': role} for hit in memberships.get(sha, []))
        family = families.get(target['sha256'], []) if target else []
        queue.append({'input_path': name, 'input_sha256': item['input_sha256'],
            'candidate_target': None if target is None else {'path': target['path'], 'sha256': target['sha256']},
            'target_basis': target_basis,
            'raster_action': 'reuse_preparation' if raster_unchanged else 'prepare',
            'pair_review_action': 'preserve_prior_review' if pair_unchanged else 'review_required',
            'previous_preparation': reference,
            'previous_pair_visual_status': pairs[name].get('visual_status') if pair_unchanged else None,
            'old_split_memberships': hits, 'family_catalog_entries': deepcopy(family),
            'family_review_status': 'known_target_hash' if family else 'unreviewed',
            'train_only_constraint_inherited': any(f.get('train_only') is True for f in family),
            'prior_identical_input_paths': sorted(old_hashes[item['input_sha256']]),
            'prior_identical_target_paths': sorted(old_hashes[target['sha256']]) if target else [],
            'preparation_header_check': _header_support(item), 'split': None,
            'training_eligible': False, 'annotation_approved': False})
    summary = {key+'_files': len(value) for key, value in changes.items()}
    summary.update(current_files=len(current), previous_files=len(old), current_png_count=len(latest),
                   new_png_count=sum(Path(n).suffix.lower() == '.png' for n in changes['added']),
                   new_dxf_count=sum(Path(n).suffix.lower() == '.dxf' for n in changes['added']),
                   reusable_rasters=sum(q['raster_action'] == 'reuse_preparation' for q in queue),
                   pending_rasters=len(delta), pairs_requiring_review=len(review_items),
                   pending_header_statuses=dict(Counter(q['preparation_header_check']['status'] for q in queue if q['raster_action'] == 'prepare')),
                   pending_known_families=sum(q['family_review_status'] == 'known_target_hash' for q in queue if q['pair_review_action'] == 'review_required'),
                   pending_old_split_hits=sum(bool(q['old_split_memberships']) for q in queue if q['pair_review_action'] == 'review_required'),
                   training_eligible_count=0)
    report = {'schema_version': '1.0', 'date': config['date'], 'status': 'registered',
              'config_sha256': digest(config_path), 'dependencies': config['dependencies'],
              'changes': changes, 'summary': summary,
              'verification': {'current_source_files_rehashed': len(current), 'reused_raster_artifacts_rehashed': reused_files},
              'code_sha256': {n: digest(Path(__file__).with_name(n)) for n in ['dataset_update.py', 'png_stream.py']},
              'training_eligible': False, 'split_modified': False, 'source_modified': False,
              'limitations': ['Filename candidates remain unverified. Byte identity is not semantic family independence.',
                             'Old preparation paths and IDs are preserved; new runs use a separate directory.',
                             'Header compatibility does not certify full PNG integrity or preparation buffer limits.',
                             'A changed target needs pair review even when the unchanged raster is reusable.',
                             'No training split or annotation approval is assigned.']}
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output/'report.json', report)
    _write_json(output/'config.json', config)
    _write_json(output/'preparation-queue.json', {'schema_version': '1.0', 'items': queue, 'summary': summary})
    for filename, items, scope in [('delta-intake.json', delta, 'rasters_requiring_preparation'),
                                   ('pair-review-intake.json', review_items, 'pairs_requiring_review')]:
        snapshot = {'schema_version': '1.0', 'purpose': 'training_intake_not_split',
                    'status': 'needs_preparation_and_pair_verification', 'training_eligible': False,
                    'dataset_root': config['dataset'], 'intake_root': config['dataset']+'/clear_sum_dataset',
                    'dependencies': deepcopy(documents['current_intake']['dependencies']),
                    'parent_intake': config['dependencies']['current_intake'], 'selection_scope': scope,
                    'items': items, 'unmatched_dxf_targets': deepcopy(documents['current_intake'].get('unmatched_dxf_targets', [])),
                    'unmatched_target_scope': 'Complete parent unmatched target pool retained for explicit alias review.',
                    'summary': {'png_count': len(items), 'manifest_pair_candidate_count': sum(i.get('candidate_target') is not None for i in items),
                                'training_eligible_count': 0},
                    'policies': report['limitations']}
        _write_json(output/filename, snapshot)
    rows = ''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in
            [q['input_path'], q['raster_action'], q['pair_review_action'], q['family_review_status'],
             q['train_only_constraint_inherited'], len(q['old_split_memberships'])])+'</tr>' for q in queue)
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Обновление датасета</title>'
        '<style>body{font:16px sans-serif;margin:2em}td,th{border:1px solid #bbb;padding:.4em}</style>'
        '<h1>Новая версия корпуса</h1><p>Кандидаты пар, происхождение и очередь подготовки. '
        'Предыдущие результаты сохраняют собственные пути. Новые обучающие метки и split не утверждены.</p>'
        '<a href="report.json">Сводка</a> · <a href="preparation-queue.json">Полная очередь</a> · '
        '<a href="delta-intake.json">Новые растры</a> · <a href="pair-review-intake.json">Пары для проверки</a>'
        '<table><tr><th>Растр</th><th>Подготовка</th><th>Проверка пары</th><th>Семейство</th><th>Train-only</th><th>Связи со split v1</th></tr>'+rows+'</table>', encoding='utf-8')
    artifacts = {p.name: digest(p) for p in output.iterdir() if p.is_file()}
    _write_json(output/'manifest.json', {'schema_version': '1.0', 'status': 'complete', 'artifact_sha256': artifacts})
    return report
