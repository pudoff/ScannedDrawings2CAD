"""Current-layout pixel preparation, vector-PDF inputs and split-aware loading."""
from collections import Counter
from pathlib import Path
import math

import numpy as np
import pymupdf
from PIL import Image

from .dataset_layout_v1 import checked, encoded, inside, pin, read
from .dataset_content_review_v1 import verify_review
from .family_catalog_v2 import require
from .large_raster import prepare_large_png
from .raster_regions import read_prepared_region

CODE = ['src/cad_recovery/'+n+'.py' for n in ('dataset_preparation_v1', 'dataset_layout_v1',
        'dataset_content_review_v1', 'dataset_content_v1', 'family_catalog_v2', 'large_raster',
        'png_stream', 'raster_regions')]+['scripts/prepare_dataset_layout_v1.py']
PARAMETERS = {'tile_size': 2048, 'overlap': 128, 'overview_max_side': 2400, 'strip_rows': 128}


def select_controls(entries, count=8):
    require(type(count) is int and count > 0, 'Invalid fixed selection count')
    selected = {}
    for split in ('validation', 'test'):
        rows = [r for r in entries if r['declared_split'] == split]
        require(len(rows) >= count and len({r['input'] for r in rows}) == len(rows), 'Incomplete control inventory')
        selected[split] = [{k: r[k] for k in ('input', 'input_sha256', 'declared_split')}
                           for r in sorted(rows, key=lambda r: (r['input_sha256'], r['input']))[:count]]
    return selected


def prepare_config(workspace, output):
    root = Path(workspace).resolve()
    review = pin(root, 'artifacts/dataset-content-review-v1/manifest.json')
    verify_review(root, review)
    config = {'schema_version': 'current-preparation-config/1.0', 'task': 'P08.9c',
              'review': review, 'parent_final': pin(root, 'artifacts/dataset-content-final-v1.json'),
              'handoff': pin(root, 'artifacts/pair-review-handoff-v5/report.json'),
              'previous_ocr': pin(root, 'artifacts/ocr-control-release-v2/manifest.json'),
              'inventory': pin(root, 'artifacts/dataset-layout-v1/inventory.json'),
              'parameters': PARAMETERS, 'code': [pin(root, n) for n in CODE],
              'control_selection': select_controls(read(root/'artifacts/dataset-content-review-v1/catalog.json')['entries']),
              'label_protocol': {'fields': ['drawing_id', 'part_name'], 'replace_selected_sources': False,
                                 'reading': 'literal original PNG pixels; no OCR or DXF text targets',
                                 'uncertain_text': None, 'initial_control_scope': 'first eight SHA-ranked sources per declared split',
                                 'training_eligible_for_geometry': False, 'independent_evaluation': False}}
    path = inside(root, output)
    require(not path.exists(), 'Config already exists')
    path.write_bytes(encoded(config))
    return pin(root, output)


def inputs(root, config_pin):
    config = read(checked(root, config_pin))
    require(config['schema_version'] == 'current-preparation-config/1.0' and config['parameters'] == PARAMETERS
            and config['code'] == [pin(root, n) for n in CODE], 'Preparation code or protocol changed')
    checked(root, config['review']); checked(root, config['parent_final'])
    review_dir = inside(root, config['review']['path']).parent
    manifest = read(review_dir/'manifest.json')
    for p in manifest['files']:
        checked(review_dir, p)
    catalog = read(review_dir/'catalog.json')
    require(config['control_selection'] == select_controls(catalog['entries']), 'Control selection changed')
    handoff = read(checked(root, config['handoff']))
    checked(root, config['previous_ocr'])
    inventory = read(checked(root, config['inventory']))['files']
    require({p.relative_to(root/'dataset').as_posix() for p in (root/'dataset').rglob('*') if p.is_file()}
            == {r['path'] for r in inventory}, 'Dataset inventory changed')
    for row in inventory:
        checked(root/'dataset', {'path': row['path'], 'sha256': row['sha256']})
    return config, catalog, handoff


def preparation_assets(root, report_pin, source_sha):
    path = checked(root, report_pin)
    report = read(path)
    require(report['status'] == 'prepared' and report['source_integrity_validated'] is True
            and report['source_sha256'] == source_sha and report['source_modified'] is False, 'Preparation source identity differs')
    require(len(report['tiles']) == report['tile_count'] > 0, 'Incomplete tile inventory')
    assets = [report['overview'], report['ink_overview'], *report['tiles']]
    seen, pins = set(), []
    for asset in assets:
        require(asset['path'] not in seen, 'Repeated prepared asset')
        seen.add(asset['path'])
        record = {'path': (path.parent/asset['path']).relative_to(root).as_posix(), 'sha256': asset['sha256']}
        asset_path = checked(root, record)
        require(asset_path.stat().st_size == asset['size_bytes'], 'Prepared asset size changed')
        pins.append(record)
    return report, pins


def prepare_current(workspace, config_pin, output):
    root = Path(workspace).resolve()
    config, catalog, handoff = inputs(root, config_pin)
    target = inside(root, output)
    require(target.parent == root/'artifacts' and not target.exists(), 'Choose a new direct artifacts directory')
    target.mkdir()
    old = {(r['input_sha256'], r['target_sha256']): r for r in handoff['entries']}
    selected = {r['input'] for rows in config['control_selection'].values() for r in rows}
    group_for_path = {p: g['id'] for g in catalog['relationships']['groups'] for p in g['members']}
    records = []
    jobs = sorted(catalog['entries'], key=lambda r: (r['input'] not in selected, r['bundle'], r['input']))
    for i, pair in enumerate(jobs):
        record = {k: pair[k] for k in ('input', 'input_sha256', 'target', 'target_sha256', 'declared_split', 'bundle',
                                       'excluded_native_target', 'visual_status', 'status')}
        record.update(group_id=group_for_path[pair['input']], training_eligible_for_geometry=False,
                      family_approved=False, independent_evaluation=False)
        source = root/'dataset'/pair['input']
        if pair['bundle'] == 'train_png':
            previous = old.get((pair['input_sha256'], pair['target_sha256']))
            require(previous is not None, 'Train preparation cannot be rebound by filename')
            prep_pin = previous['preparation']
            report, assets = preparation_assets(root, prep_pin, pair['input_sha256'])
            record.update(source_kind='png', preparation=prep_pin, prepared_assets=assets,
                          source_size_px=report['source_size_px'], prepared_status='reused_exact_source_pixels',
                          tile_count=report['tile_count'], source_png_conformant=report.get('source_png_conformant', True))
        elif pair['bundle'] in ('validation_png', 'test_png'):
            directory = target/'rasters'/('source-'+pair['input_sha256'][:20])
            report = prepare_large_png(source, directory, expected_sha256=pair['input_sha256'], **PARAMETERS)
            prep_pin = pin(root, (directory/'preparation.json').relative_to(root).as_posix())
            _, assets = preparation_assets(root, prep_pin, pair['input_sha256'])
            record.update(source_kind='png', preparation=prep_pin, prepared_assets=assets,
                          source_size_px=report['source_size_px'], prepared_status='prepared_original_pixels',
                          tile_count=report['tile_count'], source_png_conformant=True)
        else:
            usable = pair.get('pdf_status') == 'ok' and pair.get('pdf_page_classes') and all(c in ('vector', 'mixed') for c in pair['pdf_page_classes'])
            if usable:
                with pymupdf.open(source) as document:
                    pages = []
                    for page in document:
                        scale = 1600/max(page.rect.width, page.rect.height)
                        pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
                        preview = target/'pdf'/('page-'+pair['input_sha256'][:20]+f'-{page.number+1}.png')
                        preview.parent.mkdir(exist_ok=True)
                        pix.save(preview)
                        pages.append({'number': page.number+1, 'rect_pt': list(page.rect), 'rotation': page.rotation,
                                      'preview': pin(root, preview.relative_to(root).as_posix()),
                                      'preview_size_px': [pix.width, pix.height], 'preview_for_navigation_only': True})
                record.update(source_kind='pdf', prepared_status='direct_vector_pdf_with_preview', pages=pages,
                              source_is_fully_vector=all(c == 'vector' for c in pair['pdf_page_classes']))
            else:
                record.update(source_kind='pdf', prepared_status='excluded_invalid_or_empty_pdf', pages=[])
        records.append(record)
        (target/f'record-{i+1:04d}.json').write_bytes(encoded(record))
        if (i+1) % 10 == 0 or i+1 == len(jobs):
            print(encoded({'prepared_inputs': i+1, 'total': len(jobs), 'last': record['prepared_status']}).decode('ascii'), flush=True)
    summary = summarize(records, config)
    (target/'report.json').write_bytes(encoded(summary))
    inputs(root, config_pin)
    manifest = {'schema_version': 'current-preparation/1.0', 'config': config_pin,
                'records': [pin(target, f'record-{i+1:04d}.json') for i in range(len(records))],
                'report': pin(target, 'report.json')}
    (target/'manifest.json').write_bytes(encoded(manifest))
    return summary


def summarize(records, config):
    return {'source_count': len(records), 'by_preparation_status': dict(Counter(r['prepared_status'] for r in records)),
               'tiles_by_split': {s: sum(r.get('tile_count', 0) for r in records if r['declared_split'] == s) for s in ('train', 'validation', 'test')},
               'control_selection': config['control_selection'], 'training_started': False, 'independent_evaluation': False}


def verify_preparation(workspace, manifest_pin, *, all_assets=True):
    root = Path(workspace).resolve()
    path = checked(root, manifest_pin)
    manifest = read(path)
    require(manifest['schema_version'] == 'current-preparation/1.0', 'Unknown prepared dataset')
    config, catalog, _ = inputs(root, manifest['config'])
    records = [read(checked(path.parent, p)) for p in manifest['records']]
    expected = {r['input']: r for r in catalog['entries']}
    require(len(records) == len(expected) and {r['input'] for r in records} == set(expected), 'Incomplete prepared source inventory')
    groups = {p: g['id'] for g in catalog['relationships']['groups'] for p in g['members']}
    for record in records:
        pair = expected[record['input']]
        require(all(record[k] == pair[k] for k in ('input_sha256', 'target', 'target_sha256', 'declared_split', 'bundle', 'excluded_native_target', 'visual_status', 'status'))
                and record['group_id'] == groups[record['input']], 'Prepared source has changed split, pair or quarantine')
        require(record['training_eligible_for_geometry'] is False and record['family_approved'] is False
                and record['independent_evaluation'] is False, 'Preparation cannot approve labels or independence')
        if record['source_kind'] == 'png':
            require(pair['bundle'] != 'train_pdf' and record['prepared_status'] ==
                    ('reused_exact_source_pixels' if pair['bundle'] == 'train_png' else 'prepared_original_pixels'), 'Raster preparation status changed')
            report = read(checked(root, record['preparation']))
            require(report['source_sha256'] == record['input_sha256'] and report['source_integrity_validated'] is True
                    and report['status'] == 'prepared' and report['tile_count'] == record['tile_count']
                    and report['source_size_px'] == record['source_size_px'], 'Incomplete or rebound raster')
            if all_assets:
                _, assets = preparation_assets(root, record['preparation'], record['input_sha256'])
                require(assets == record['prepared_assets'], 'Prepared asset inventory changed')
        else:
            require(record['source_kind'] == 'pdf' and pair['bundle'] == 'train_pdf', 'Source kind changed')
            usable = pair.get('pdf_status') == 'ok' and bool(pair.get('pdf_page_classes')) and all(c in ('vector', 'mixed') for c in pair.get('pdf_page_classes', []))
            require(record['prepared_status'] == ('direct_vector_pdf_with_preview' if usable else 'excluded_invalid_or_empty_pdf'), 'PDF quarantine changed')
            require(bool(record['pages']) == bool(usable), 'PDF pages differ from eligibility')
            for page in record['pages']:
                checked(root, page['preview'])
    require(read(checked(path.parent, manifest['report'])) == summarize(records, config), 'Preparation summary differs from records')
    return records, config


def load_sources(workspace, manifest_pin, *, split, source_kind=None):
    require(split in ('train', 'validation', 'test') and source_kind in (None, 'png', 'pdf'), 'Explicit split and known source kind required')
    root = Path(workspace).resolve()
    records, _ = verify_preparation(root, manifest_pin, all_assets=False)
    return [{**r, 'source_path': str(inside(root/'dataset', r['input']))} for r in records
            if r['declared_split'] == split and (source_kind is None or r['source_kind'] == source_kind)
            and r['prepared_status'] != 'excluded_invalid_or_empty_pdf']


def crop_source(workspace, record, box):
    root = Path(workspace).resolve()
    require(record['source_kind'] == 'png', 'Exact tiled RGB crop requires PNG source')
    checked(root/'dataset', {'path': record['input'], 'sha256': record['input_sha256']})
    prep = checked(root, record['preparation'])
    require(read(prep)['source_sha256'] == record['input_sha256'], 'Crop source differs from preparation')
    return read_prepared_region(prep.parent, box, max_pixels=16777216)
