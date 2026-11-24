"""Reviewed text from six fixed raster sources, admitted only to development train.

The inherited V2 samples remain unchanged. A full old-plus-new registry enforces
the frozen additive coallocation policy; its train commitments include every
sibling member, without approving semantic families, CAD or evaluation data.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime
from html import escape
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .clear_pairs import contained, digest
from .coallocation_policy_v1 import build_coallocation_v1, validate_assignments_v1
from .coallocation_review_v1 import source_regions_v1
from .family_catalog_v2 import stable_digest
from .label_release_v1 import checked_pin
from .ocr_label_schema_v1 import make_ocr_transform_v1
from .ocr_line_schema_v3 import ReviewV3
from .ocr_release_v2 import load_ocr_training_v2
from .raster_regions import read_prepared_region

PROTOCOL_PIN = {'path': 'configs/reviews/ocr-lines-protocol-v3.json',
                'sha256': '7e07c19a587c6d937e770614f2bc74ac4c2d6372f90f036989b23f6aa13027a8'}
READINGS = ['artifacts/ocr-lines-'+letter+'-v3/review.json' for letter in ('a', 'b')]
DEPENDENCIES = {'parent_final', 'coallocation_manifest', 'overlay', 'previous_ocr_manifest',
                'control_manifest', 'control_review', 'catalog', 'policy', 'exclusions', 'prepared_catalog'}
CODE = ['src/cad_recovery/'+name+'.py' for name in (
    'ocr_release_v3', 'ocr_line_schema_v3', 'ocr_label_schema_v1', 'ocr_release_v2',
    'coallocation_policy_v1', 'coallocation_review_v1', 'family_catalog_v2',
    'ocr_lineage_v1', 'raster_regions', 'png_stream', 'clear_pairs', 'label_release_v1')]+['scripts/release_ocr_v3.py']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n').encode('utf-8')


def pin(root, path):
    return {'path': path, 'sha256': digest(contained(root, path))}


def _package(root, record):
    path = checked_pin(root, record)
    package = read(path)
    names = [p['path'] for p in package['files']]
    require(len(names) == len({name.casefold() for name in names}) and 'manifest.json' not in {name.casefold() for name in names}, 'Duplicate or recursive input manifest')
    require({p.relative_to(path.parent).as_posix() for p in path.parent.rglob('*') if p.is_file()} == set(names)|{'manifest.json'}, 'Input package inventory differs')
    for item in package['files']:
        checked_pin(path.parent, item)
    return path, package


def _base(workspace):
    root = Path(workspace).resolve()
    protocol = read(checked_pin(root, PROTOCOL_PIN))
    require(protocol['schema_version'] == 'ocr-lines-protocol/3.0' and protocol['task'] == 'P14.1b.6'
            and protocol['new_release_split'] == 'train' and protocol['previous_ocr_sample_count'] == 32
            and all(protocol[key] is False for key in ('human_verified', 'OCR_run', 'native_CAD_opened',
                'training_started', 'independent_evaluation', 'semantic_family_approved', 'physical_geometry_approved')),
            'Changed OCR line protocol')
    require(set(protocol['dependencies']) == DEPENDENCIES, 'Changed protocol dependencies')
    values = {key: read(checked_pin(root, value)) for key, value in protocol['dependencies'].items()}
    for name in ('coallocation_manifest', 'control_manifest'):
        _package(root, protocol['dependencies'][name])
    require(protocol['dependencies']['coallocation_manifest'] in values['parent_final']['manifests'], 'Parent proof identifies a different coallocation release')
    overlay = values['overlay']
    rebuilt = build_coallocation_v1(values['catalog'], values['policy'], values['exclusions'], overlay['constraints'])
    require(stable_digest(rebuilt) == stable_digest(overlay), 'Overlay differs from frozen policy')
    for constraint in overlay['constraints']:
        for record in constraint['evidence']:
            checked_pin(root, record)
    prepared = {row['input_path']: row for row in values['prepared_catalog']['entries']}
    control = values['control_review']['sources']
    require(len(protocol['sources']) == len(control) == 6, 'Exactly six preregistered sources required')
    expected = []
    for row in control:
        binding = prepared[row['input_path']]
        expected.append({'id': row['id'], 'input_path': row['input_path'], 'input_sha256': binding['input_sha256'],
            'preparation': binding['preparation'], 'source_size_px': row['source_size_px'],
            'parent_regions': {kind: {'image': row['regions'][kind], 'source_box_xyxy': row[kind+'_box_source_px']}
                               for kind in ('title', 'context')}, 'previous_proposed_roles': row['proposed_annotation_roles']})
    require(protocol['sources'] == expected, 'Protocol source or parent region scope changed')
    previous_pin = protocol['dependencies']['previous_ocr_manifest']
    previous = load_ocr_training_v2(root, previous_pin)
    labels = read(checked_pin(root, previous_pin).parent/'labels.json')
    require(previous['count'] == len(previous['records']) == len(labels['samples']) == 32, 'Parent OCR inventory differs')
    return protocol, values, previous, labels


def prepare_ocr_config_v3(workspace, output_path):
    root = Path(workspace).resolve()
    output = contained(root, output_path)
    require(output.parent == root/'configs/reviews' and not output.exists(), 'Use a new OCR config')
    _base(root)
    config = {'schema_version': 'ocr-release-config/3.0', 'task': 'ocr_recognition', 'protocol': PROTOCOL_PIN,
              'readings': [pin(root, path) for path in READINGS], 'code': [pin(root, path) for path in CODE]}
    output.write_bytes(encoded(config))
    return pin(root, output.relative_to(root).as_posix())


def _verify_inputs(workspace, config_pin):
    root = Path(workspace).resolve()
    config = read(checked_pin(root, config_pin))
    require(set(config) == {'schema_version', 'task', 'protocol', 'readings', 'code'}
            and config['schema_version'] == 'ocr-release-config/3.0' and config['task'] == 'ocr_recognition'
            and config['protocol'] == PROTOCOL_PIN, 'Invalid OCR release config')
    require(len(config['code']) == len(CODE) and {p['path'] for p in config['code']} == set(CODE), 'Missing release code pins')
    for record in config['code']:
        checked_pin(root, record)
    require(len(config['readings']) == len(READINGS) and [p['path'] for p in config['readings']] == READINGS, 'Changed reading inventory or order')
    protocol, values, previous, labels = _base(root)
    bindings = {row['id']: row for row in protocol['sources']}
    files = {row['path']: row for row in values['catalog']['files']}
    sources, reviews, identities, annotation_owners = [], [], set(), {}
    for batch, record in enumerate(config['readings']):
        review = ReviewV3.model_validate(read(checked_pin(root, record))).model_dump(mode='json')
        require(review['protocol'] == PROTOCOL_PIN and datetime.fromisoformat(review['reviewed_at_utc']) >=
                datetime.fromisoformat(protocol['created_before_new_views_utc']), 'Review precedes or changes protocol')
        expected = protocol['sources'][batch*3:(batch+1)*3]
        require(len(review['sources']) == len(expected) and {s['id'] for s in review['sources']} == {s['id'] for s in expected}, 'Every fixed source needs its reviewed outcome')
        for row in review['sources']:
            binding = bindings[row['id']]
            require(all(row[key] == binding[key] for key in ('input_path', 'input_sha256', 'preparation', 'source_size_px')), 'Source binding differs')
            require(row['input_path'] in files and files[row['input_path']]['kind'] == 'image'
                    and files[row['input_path']]['sha256'] == row['input_sha256'], 'Only frozen source images may supply text')
            checked_pin(root, {'path': 'dataset/'+row['input_path'], 'sha256': row['input_sha256']})
            preparation = read(checked_pin(root, row['preparation']))
            require(preparation['source_sha256'] == row['input_sha256'] and preparation['source_size_px'] == row['source_size_px']
                    and 'rgba_conversion' not in preparation and 'container_repair' not in preparation, 'Unexpected source preparation')
            for parent in binding['parent_regions'].values():
                checked_pin(root, parent['image'])
            for line in row['lines']:
                require(line['id'].casefold() not in identities, 'Duplicate or case-aliased line across reviews')
                identities.add(line['id'].casefold())
                owner = (row['id'], line['parent_kind'])
                require(line['annotation_id'] not in annotation_owners or annotation_owners[line['annotation_id']] == owner, 'Reused annotation across sources or regions')
                annotation_owners[line['annotation_id']] = owner
                box, parent = line['source_box_xyxy'], binding['parent_regions'][line['parent_kind']]['source_box_xyxy']
                require(parent[0] <= box[0] < box[2] <= parent[2] and parent[1] <= box[1] < box[3] <= parent[3], 'Line lies outside its frozen parent region')
                for key in ('raw_crop_pin', 'image_pin'):
                    if line[key] is not None:
                        checked_pin(root, line[key])
            sources.append({**row, 'review': record})
        reviews.append({'pin': record, 'data': review})
    require(len(sources) == len(bindings) and {s['id'] for s in sources} == set(bindings), 'Source review coverage differs')
    old_ids = {row['id'].casefold() for row in labels['samples']}
    require(len(old_ids) == len(labels['samples']) and not old_ids.intersection(identities), 'New line collides with inherited sample')
    return {'config': config, 'protocol': protocol, 'values': values, 'previous': previous, 'old_labels': labels,
            'sources': sorted(sources, key=lambda r: r['id']), 'reviews': reviews}


def _pixels(root, record, size):
    with Image.open(checked_pin(root, record)) as image:
        require(image.format == 'PNG' and image.mode == 'RGB' and list(image.size) == size
                and getattr(image, 'n_frames', 1) == 1, 'Pinned image mode or dimensions differ')
        return np.array(image)


def _documents(workspace, config_pin, verified):
    root = Path(workspace).resolve()
    protocol, values = verified['protocol'], verified['values']
    bindings = {row['id']: row for row in protocol['sources']}
    overlay = values['overlay']
    groups = {row['group_id']: row for row in overlay['groups']}
    files = {row['path']: row for row in values['catalog']['files']}
    images, new_samples, unresolved, evidence = {}, [], [], []
    for row in verified['sources']:
        binding = bindings[row['id']]
        boxes = {kind: parent['source_box_xyxy'] for kind, parent in binding['parent_regions'].items()}
        raw_parents = source_regions_v1(contained(root, 'dataset/'+row['input_path']), row['input_sha256'], boxes)
        prep = checked_pin(root, row['preparation']).parent
        parent_evidence = {}
        for kind, box in boxes.items():
            pixels, provenance = read_prepared_region(prep, box)
            size = [box[2]-box[0], box[3]-box[1]]
            require(np.array_equal(pixels, raw_parents[kind]) and np.array_equal(pixels,
                    _pixels(root, binding['parent_regions'][kind]['image'], size)), 'Parent original/prepared/viewed pixels differ')
            parent_evidence[kind] = {'image': binding['parent_regions'][kind]['image'], 'source_box_xyxy': box,
                'provenance': provenance, 'original_png_stream_replay_equal': True, 'source_original_rehashed': True}
        for line in row['lines']:
            box = line['source_box_xyxy']; parent = boxes[line['parent_kind']]
            raw = raw_parents[line['parent_kind']][box[1]-parent[1]:box[3]-parent[1], box[0]-parent[0]:box[2]-parent[0]]
            transform = make_ocr_transform_v1(box, line['rotation_cw_degrees'])
            oriented = np.rot90(raw, -(line['rotation_cw_degrees']//90)).copy()
            for key, pixels in [('raw_crop_pin', raw), ('image_pin', oriented)]:
                if line[key] is not None:
                    require(np.array_equal(pixels, _pixels(root, line[key], [pixels.shape[1], pixels.shape[0]])), 'Reviewed line pixels or rotation differ')
            if line['decision'] == 'unresolved':
                unresolved.append({**line, 'source_id': row['id'], 'review': row['review']})
                continue
            yy, xx = np.where(np.any(oriented < 200, axis=2))
            require(len(xx) > 0, 'Accepted OCR line is blank')
            ink = [int(xx.min()), int(yy.min()), int(xx.max())+1, int(yy.max())+1]
            margins = [ink[0], ink[1], oriented.shape[1]-ink[2], oriented.shape[0]-ink[3]]
            require(min(margins) > 0, 'Accepted OCR text touches crop boundary')
            sample = {**line, 'source_id': row['id'], 'review': row['review'], **transform,
                      'white_margins_px_ltrb': margins, 'ink_box_image_xyxy': ink,
                      'diagnostic_ink_threshold_exclusive': 200, 'physical_geometry_approved': False,
                      'semantic_family_approved': False, 'independent_evaluation': False, 'human_verified': False}
            new_samples.append(sample)
            images['images/'+line['id']+'.png'] = checked_pin(root, line['image_pin']).read_bytes()
        evidence.append({'source_id': row['id'], 'source': {'path': 'dataset/'+row['input_path'], 'sha256': row['input_sha256']},
                         'preparation': row['preparation'], 'parents': parent_evidence})
    new_samples.sort(key=lambda row: row['id']); unresolved.sort(key=lambda row: row['id'])
    old = verified['old_labels']
    sources = {row['id']: row for row in old['source_records']}
    require(len(sources) == len(old['source_records']), 'Repeated inherited source identity')
    require(not set(sources).intersection(row['id'] for row in verified['sources']), 'New source collides with inherited identity')
    train, assignments = [], []
    for row in verified['previous']['records']:
        require(row['source_id'] in sources and row['split'] == 'train' and row['task'] == 'ocr_recognition', 'Changed inherited source, task or split')
        source = sources[row['source_id']]['source']
        require(source['path'].startswith('dataset/') and source['sha256'] == row['source_sha256'], 'Changed inherited source binding')
        checked_pin(root, source)
        copied = {key: value for key, value in row.items() if key != 'image_path'}
        image_path = contained(checked_pin(root, protocol['dependencies']['previous_ocr_manifest']).parent, row['image'])
        require(digest(image_path) == row['image_sha256'], 'Inherited OCR image differs')
        images[row['image']] = image_path.read_bytes()
        train.append(copied)
        assignments.append({'id': row['id'], 'source_path': source['path'].removeprefix('dataset/'),
                            'source_sha256': source['sha256'], 'image_sha256': row['image_sha256'], 'split': 'train'})
    require(len(train) == len(old['samples']) == verified['previous']['count'] == 32, 'Do not drop inherited OCR records')
    for sample in new_samples:
        source = bindings[sample['source_id']]
        path = source['input_path']
        train.append({'id': sample['id'], 'source_id': sample['source_id'], 'image': 'images/'+sample['id']+'.png',
            'image_sha256': sample['image_pin']['sha256'], 'text': sample['text'], 'source_sha256': source['input_sha256'],
            'group_id': files[path]['group_id'], 'split': 'train', 'task': 'ocr_recognition'})
        assignments.append({'id': sample['id'], 'source_path': path, 'source_sha256': source['input_sha256'],
                            'image_sha256': sample['image_pin']['sha256'], 'split': 'train'})
    require(len({row['id'].casefold() for row in train}) == len(train), 'Repeated or case-aliased OCR sample identity')
    by_hash = defaultdict(list)
    for row in train:
        by_hash[row['image_sha256']].append(row)
    require(all(len({r['text'] for r in rows}) == 1 for rows in by_hash.values()), 'Identical images have conflicting OCR text')
    routing = validate_assignments_v1(values['catalog'], values['policy'], values['exclusions'], overlay, assignments)
    touched = sorted({row['coallocation_group_id'] for row in routing['assignments']})
    commitments = [{'coallocation_group_id': identifier, 'split': 'train', 'member_paths': groups[identifier]['members'],
                    'reason': 'development_OCR_use_requires_all_related_members_to_remain_out_of_independent_holdouts',
                    'semantic_family_approved': False, 'independent_evaluation': False} for identifier in touched]
    new_sources = [{key: row[key] for key in ('id', 'input_path', 'input_sha256', 'preparation', 'source_size_px', 'review', 'scope_notes')}
                   | {'base_group_id': files[row['input_path']]['group_id'],
                      'coallocation_group_id': overlay['file_to_group'][row['input_path']],
                      'semantic_family_approved': False, 'independent_evaluation': False} for row in verified['sources']]
    labels = {'schema_version': 'ocr-labels/3.0', 'task': 'ocr_recognition', 'release_scope': 'reviewed_text_for_development_train_only',
        'protocol': PROTOCOL_PIN, 'parent_release': protocol['dependencies']['previous_ocr_manifest'],
        'inherited_source_records': deepcopy(old['source_records']), 'inherited_annotations': deepcopy(old['annotations']),
        'inherited_samples': deepcopy(old['samples']), 'inherited_unresolved_annotations': deepcopy(old['unresolved_annotations']),
        'new_source_records': new_sources, 'new_samples': new_samples, 'unresolved_candidates': unresolved,
        'training_eligible_for_task': True, 'training_started': False, 'human_verified': False,
        'physical_geometry_approved': False, 'physical_scale_confirmed': False, 'native_target_approved': False,
        'semantic_family_approved': False, 'independent_evaluation': False}
    registry = {'schema_version': 'ocr-task-splits/3.0', 'task': 'ocr_recognition',
        'scope': 'complete_inherited_32_plus_all_accepted_six_source_lines',
        'overlay': protocol['dependencies']['overlay'], 'inherited_sample_ids': [r['id'] for r in verified['previous']['records']],
        'assignments': routing['assignments'], 'train_commitments': commitments,
        'splits': {'train': [row['id'] for row in train], 'validation': [], 'test': []},
        'global_split_policy_modified': False, 'independent_evaluation': False}
    per_source = [{'id': row['id'], 'candidate_count': len(row['lines']),
                   'accepted': sum(line['decision'] == 'accepted_for_ocr_transcription' for line in row['lines']),
                   'unresolved': sum(line['decision'] == 'unresolved' for line in row['lines'])} for row in verified['sources']]
    report = {'schema_version': 'ocr-release/3.0', 'task': 'P14.1b.6', 'config': config_pin,
        'status': 'released_for_development_ocr_only', 'source_count': len(sources)+len(new_sources),
        'new_source_count': len(new_sources), 'parent_region_count': sum(len(row['parents']) for row in evidence),
        'candidate_count': len(new_samples)+len(unresolved), 'new_sample_count': len(new_samples),
        'unresolved_candidate_count': len(unresolved), 'inherited_sample_count': len(verified['previous']['records']),
        'sample_count': len(train), 'unique_image_hashes': len(by_hash), 'task_split_counts': routing['counts_by_split'],
        'coallocation_group_count': len(touched), 'committed_member_count': sum(len(row['member_paths']) for row in commitments),
        'source_outcomes': per_source, 'new_line_rotations': dict(sorted(Counter(str(row['rotation_cw_degrees']) for row in new_samples).items())),
        'inherited_samples_and_images_unchanged': True, 'source_modified': False, 'global_split_policy_modified': False,
        'ocr_training_eligible': True, 'training_started': False, 'OCR_run': False, 'augmentation_created': False,
        'human_verified': False, 'physical_geometry_approved': False, 'native_target_approved': False,
        'semantic_family_approved': False, 'independent_evaluation': False,
        'limitations': ['Assistant-reviewed text uses explicit glyph conventions; original font code points are unknown.',
            'Unresolved candidates retain context and reasons but have no OCR target.',
            'Crops are recognition inputs, not approved detection boxes or physical dimension labels.',
            'The entire source and its coallocation siblings are committed to development train for future releases.',
            'External family relationships remain unresolved; these data cannot measure independent generalization.',
            'Future releases must check all existing and proposed assignments together and inherit train commitments.']}
    documents = {'labels.json': labels, 'routing.json': routing, 'task-splits.json': registry, 'report.json': report,
                 'evidence.json': evidence, 'readings.json': verified['reviews'], 'review-schema.json': ReviewV3.model_json_schema()}
    result = {**{name: encoded(value) for name, value in documents.items()}, **images}
    for split, rows in [('train', train), ('validation', []), ('test', [])]:
        result[split+'.jsonl'] = ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)+'\n' for row in rows).encode('utf-8')
    html = ['<!doctype html><html lang="ru"><meta charset="utf-8"><title>OCR — выпуск 3</title>',
        '<style>body{font:17px system-ui;max-width:1100px;margin:2em auto}section{margin:2em 0}img{max-width:100%}</style>',
        '<h1>OCR: восемь источников</h1><p>Только данные разработки train. Прежние 32 образца сохранены; обучение не запускалось.</p>',
        '<p><a href="labels.json">Метки и неопределённости</a> · <a href="report.json">Отчёт</a> · <a href="task-splits.json">Общий реестр</a></p>']
    for row in train:
        html.append('<section><p>'+escape(row['source_id']+' / '+row['id'])+' — <b>'+escape(row['text'])+'</b></p><img loading="lazy" src="'+escape(row['image'], quote=True)+'" alt="'+escape(row['text'], quote=True)+'"></section>')
    for row in unresolved:
        html.append('<section><p>Неопределённо: '+escape(row['source_id']+' / '+row['id'])+' — '+escape(row['reason'])+'</p></section>')
    result['index.html'] = (''.join(html)+'</html>\n').encode('utf-8')
    return result, report, train


def run_ocr_release_v3(workspace, config_pin, output_dir):
    root = Path(workspace).resolve()
    output = contained(root, output_dir)
    require(output.parent == root/'artifacts' and not output.exists(), 'Use a new direct artifacts directory')
    files, report, _ = _documents(root, config_pin, _verify_inputs(root, config_pin))
    _verify_inputs(root, config_pin)
    output.mkdir()
    for name, data in files.items():
        path = contained(output, name); path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
    manifest = {'schema_version': 'ocr-release-manifest/3.0', 'task': 'ocr_recognition', 'config': config_pin,
                'files': [pin(output, name) for name in sorted(files)]}
    (output/'manifest.json').write_bytes(encoded(manifest))
    return report


def load_ocr_training_v3(workspace, manifest_pin, *, task='ocr_recognition', split='train'):
    require(task == 'ocr_recognition' and split == 'train', 'Only development OCR train is released')
    root = Path(workspace).resolve()
    path, manifest = _package(root, manifest_pin)
    require(set(manifest) == {'schema_version', 'task', 'config', 'files'} and
            manifest['schema_version'] == 'ocr-release-manifest/3.0' and manifest['task'] == task, 'Wrong OCR manifest')
    files, report, train = _documents(root, manifest['config'], _verify_inputs(root, manifest['config']))
    require(len(manifest['files']) == len(files) and {r['path'] for r in manifest['files']} == set(files), 'Release inventory differs')
    for record in manifest['files']:
        require(checked_pin(path.parent, record).read_bytes() == files[record['path']], 'Release differs from authenticated recipe')
    _verify_inputs(root, manifest['config'])
    return {'task': task, 'split': split, 'count': len(train), 'records': [{**row, 'image_path': str(contained(path.parent, row['image']))} for row in train],
            'training_started': False, 'physical_geometry_approved': False, 'independent_evaluation': False, 'report': report}
