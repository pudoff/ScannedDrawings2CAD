"""Reproduce a bounded, explicitly transcribed variant review without labels.

Reading the source image is an assistant review assertion. This runner checks
its exact pixels/provenance, native CAD observations and conservative decisions;
it does not claim automatic OCR, physical scale or unrestricted identity proof.
"""
from collections import Counter
from copy import deepcopy
import html
import json
import os
from pathlib import Path
import re

import ezdxf
from PIL import Image

from .clear_pairs import contained, digest, read_json
from .raster_regions import read_prepared_region
from .variant_decision import evaluate_variant_candidates
from .variant_native import extract_variant_facts, parse_variant_text

MODULES = {'variant_review.py', 'variant_decision.py', 'variant_native.py',
           'raster_regions.py', 'clear_pairs.py'}
TEXT_FIELDS = {'designation', 'displayed_length', 'mass_g', 'strength_class', 'declared_scale'}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _identifier(value):
    _require(isinstance(value, str) and re.fullmatch(r'[a-z][a-z0-9_-]{0,63}', value),
             'Expected a portable review identifier')
    return value


def _pin(root, record):
    _require(isinstance(record, dict) and set(record) >= {'path', 'sha256'}, 'Missing file pin')
    path = contained(root, record['path'])
    _require(digest(path) == record['sha256'], 'Pinned file changed: '+record['path'])
    return path


def _write(path, data):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=True, indent=2, allow_nan=False)
        stream.write('\n')


def source_facts_from_readings(readings, regions):
    """Normalize explicit transcriptions; pixel reading itself is not automated."""
    _require(isinstance(readings, dict) and set(readings) <= TEXT_FIELDS | {'shaft_break'},
             'Unknown source reading field')
    facts, qualifications = {}, []
    for field, reading in readings.items():
        _require(isinstance(reading, dict) and reading.get('region') in regions,
                 'Every reading must refer to an inspected region')
        if field == 'shaft_break':
            _require(type(reading.get('observed_value')) is bool
                     and isinstance(reading.get('description'), str) and reading['description'].strip(),
                     'Shaft-break reading needs a boolean and explicit observation')
            facts[field] = reading['observed_value']
        else:
            normalized = parse_variant_text(field, reading.get('observed_text'))
            if field == 'declared_scale':
                qualifications.append({'declared_scale': normalized['declared_scale'],
                                       'physical_scale_confirmed': False})
            else:
                facts.update(normalized)
    return facts, qualifications


def run_variant_review(workspace, config_path, output):
    root, config_path, output = (Path(p).resolve() for p in (workspace, config_path, output))
    _require(config_path.is_relative_to(root), 'Review configuration must be in workspace')
    _require(output.is_relative_to(root/'artifacts') and output != root/'artifacts',
             'Review output must be a new artifacts subdirectory')
    if output.exists():
        raise FileExistsError('Review output exists; preserve it and choose a new version')
    config_sha = digest(config_path)
    config = read_json(config_path)
    _require(config.get('schema_version') == '1.0'
             and config.get('purpose') == 'explicit_bolt_variant_review'
             and config.get('reviewer') == 'assistant', 'Expected explicit assistant variant review')
    _require(config.get('training_eligible') is False and config.get('split_modified') is False,
             'Variant review cannot approve training or split assignments')
    _require(set(config.get('implementation', {})) == MODULES, 'Incomplete implementation pins')
    dependencies = config.get('dependencies', {})
    _require({'run', 'handoff', 'prepared_batch', 'source_crops'} <= set(dependencies),
             'Missing parent review dependencies')
    inputs = []

    def pinned(record):
        path = _pin(root, record)
        inputs.append((path, record['sha256']))
        return path

    paths = {name: pinned(record) for name, record in dependencies.items()}
    for name, expected in config['implementation'].items():
        path = contained(Path(__file__).parent, name)
        _require(digest(path) == expected, 'Variant review implementation changed')
        inputs.append((path, expected))
    for record in config.get('library_dependencies', {}).values():
        pinned(record)
    for name, expected in config.get('entrypoints', {}).items():
        pinned({'path': name, 'sha256': expected})
    run, handoff, batch, preview = (read_json(paths[name]) for name in ('run', 'handoff', 'prepared_batch', 'source_crops'))
    _require(run.get('status') == 'complete' and run.get('complete') is True,
             'Expected a complete parent pair run')
    pairs = {row['input_path']: row for row in run['pairs']}
    _require(len(pairs) == len(run['pairs']), 'Duplicate parent pair identity')
    catalog = {row['input_path']: row for row in handoff['entries']}
    _require(len(catalog) == len(handoff['entries']), 'Duplicate handoff identity')
    _require(batch.get('status') == 'complete', 'Expected complete source preparation')
    preparations = {row['input_path']: row for row in batch['items']}
    _require(len(preparations) == len(batch['items']), 'Duplicate prepared source identity')
    _require(preview.get('purpose') == 'source_crops_before_review_transcription'
             and preview.get('run') == dependencies['run']
             and preview.get('training_eligible') is False, 'Inspected crop record differs from parent run')
    preview_rows = {row['id']: row for row in preview['cases']}
    _require(len(preview_rows) == len(preview['cases']), 'Duplicate inspected source crop case')
    dataset = contained(root, config['dataset'])
    _require(not output.is_relative_to(dataset)
             and not any(output.is_relative_to(paths[name].parent)
                         for name in ('run', 'handoff', 'prepared_batch', 'source_crops')),
             'Review output must not extend a sealed input artifact directory')

    def parent_pair(record):
        path = pinned(record)
        pair = read_json(path)
        _require(pair == pairs.get(pair.get('input_path')), 'Pair record differs from pinned parent run')
        reference_path = contained(paths['run'].parent, pair['artifacts']['reference'])
        expected = (reference_path.parent/'pair.json').resolve()
        _require(path == expected, 'Pair report is outside its parent run location')
        _require(catalog[pair['input_path']]['pair_report'] == record,
                 'Pair record differs from parent handoff')
        _require(pair.get('training_eligible') is False and pair.get('annotation_status') == 'not_approved',
                 'Parent pair has unexpected label approval')
        for side in ('input', 'target'):
            native = contained(dataset, pair[side+'_path'])
            _require(digest(native) == pair[side+'_sha256'], 'Original source differs from parent pair')
            inputs.append((native, pair[side+'_sha256']))
        return pair

    candidate_specs = config.get('candidates')
    _require(isinstance(candidate_specs, list) and 2 <= len(candidate_specs) <= 40, 'Invalid candidate inventory')
    candidates, target_hashes = {}, set()
    for item in candidate_specs:
        identifier = _identifier(item['id'])
        _require(identifier not in candidates, 'Repeated candidate identifier')
        pair = parent_pair(item['pair_report'])
        _require(pair['target_sha256'] not in target_hashes, 'Repeated native candidate target identity')
        target_hashes.add(pair['target_sha256'])
        source = contained(dataset, pair['target_path'])
        _require(0 < source.stat().st_size <= 50*1024**2, 'CAD candidate exceeds its byte limit')
        document = ezdxf.readfile(source)
        _require(len(document.entitydb) <= 200000, 'CAD candidate exceeds its native entity limit')
        native = extract_variant_facts(document, item['native_spec'])
        reference = contained(paths['run'].parent, pair['artifacts']['reference'])
        target_png = contained(paths['run'].parent, pair['artifacts']['target'])
        for kind, path in (('reference', reference), ('target', target_png)):
            _require(digest(path) == pair['artifact_sha256'][kind], 'Parent CAD reference artifact changed')
            inputs.append((path, pair['artifact_sha256'][kind]))
        candidates[identifier] = {'id': identifier, 'target_path': pair['target_path'],
            'target_sha256': pair['target_sha256'], 'pair_report': item['pair_report'],
            'reference_complete': pair['reference_complete'], 'parent_visual_status': pair['visual_status'],
            'target_render': {'path': target_png.relative_to(root).as_posix(), 'sha256': digest(target_png)},
            'native': native, 'training_eligible': False, 'annotation_approved': False}
    case_specs = config.get('cases')
    _require(isinstance(case_specs, list) and 1 <= len(case_specs) <= 20, 'Invalid review case inventory')
    rows, seen_cases, used_candidates = [], set(), set()
    pending_regions = []
    for case in case_specs:
        identifier = _identifier(case['id'])
        _require(identifier not in seen_cases, 'Repeated source case identifier')
        seen_cases.add(identifier)
        pair = parent_pair(case['pair_report'])
        _require(case['input_path'] == pair['input_path'], 'Reviewed source identity differs')
        registered = preparations[pair['input_path']]
        page = contained(paths['prepared_batch'].parent, pair['source_provenance']['preparation_report'])
        page_sha = pair['source_provenance']['preparation_report_sha256']
        _require(registered['preparation_report_sha256'] == page_sha and digest(page) == page_sha
                 and registered['input_sha256'] == pair['input_sha256'], 'Prepared source provenance differs')
        inputs.append((page, page_sha))
        prior = preview_rows[identifier]
        _require(prior['input_path'] == pair['input_path'] and prior['input_sha256'] == pair['input_sha256']
                 and prior['preparation'] == {'path': page.relative_to(root).as_posix(), 'sha256': page_sha},
                 'Inspected crop provenance does not identify this source preparation')
        region_specs = case['regions']
        regions = {_identifier(row['id']): deepcopy(row) for row in region_specs}
        _require(1 <= len(regions) <= 20 and len(regions) == len(region_specs), 'Invalid source region inventory')
        preview_regions = {r['id']: r for r in prior['regions']}
        for name, region in regions.items():
            inspected = preview_regions[name]
            _require(region['source_box_px'] == inspected['source_box_px']
                     and region['reviewed_crop'] == {k: inspected[k] for k in ('path', 'sha256')},
                     'Source region differs from the pixels inspected by the reviewer')
            pinned(region['reviewed_crop'])
            region['output_png'] = identifier+'/'+name+'.png'
            pending_regions.append((region, page, pair['input_sha256']))
        facts, qualifications = source_facts_from_readings(case['source_readings'], regions)
        choices = case['candidate_ids']
        _require(isinstance(choices, list) and 2 <= len(choices) <= 20 and len(set(choices)) == len(choices)
                 and set(choices) <= set(candidates), 'Invalid candidate set for reviewed source')
        used_candidates.update(choices)
        decision = evaluate_variant_candidates(facts, [
            {'id': name, 'facts': candidates[name]['native']['facts'],
             'reference_complete': candidates[name]['reference_complete'],
             'native_conflicts': candidates[name]['native']['native_conflicts']} for name in choices])
        rows.append({'id': identifier, 'input_path': pair['input_path'], 'input_sha256': pair['input_sha256'],
            'pair_report': case['pair_report'], 'parent_qualified_pair_id': catalog[pair['input_path']]['qualified_pair_id'],
            'source_size_px': pair['source_provenance']['source_size_px'],
            'preparation': {'path': page.relative_to(root).as_posix(), 'sha256': page_sha},
            'source_readings': deepcopy(case['source_readings']), 'source_facts': facts,
            'source_reading_scope': 'Explicit assistant transcription and shape observation from pinned RGB regions; not automated OCR.',
            'source_pixel_readings_verified_automatically': False, 'source_qualifications': qualifications,
            'regions': list(regions.values()), 'candidate_ids': choices, 'decision': decision,
            'training_eligible': False, 'annotation_approved': False, 'split': None})
    _require(used_candidates == set(candidates), 'Unused candidate in review inventory')
    output.mkdir(parents=True, exist_ok=False)
    for region, page, source_sha in pending_regions:
        pixels, provenance = read_prepared_region(page.parent, region['source_box_px'])
        _require(provenance['source_sha256'] == source_sha, 'Cropped pixels identify a different source')
        target = contained(output, region['output_png'])
        target.parent.mkdir(exist_ok=True)
        Image.fromarray(pixels).save(target)
        _require(digest(target) == region['reviewed_crop']['sha256'], 'Reproduced pixels differ from inspected region')
        region.update(sha256=digest(target), provenance=provenance)
    for path, expected in inputs:
        _require(digest(path) == expected, 'Review input changed during processing')
    _require(digest(config_path) == config_sha, 'Review configuration changed during processing')
    report = {'schema_version': '1.0', 'date': config['date'], 'status': 'completed_explicit_review',
        'purpose': 'variant_disambiguation_within_reviewed_candidates_not_training_labels',
        'config': {'path': config_path.relative_to(root).as_posix(), 'sha256': config_sha},
        'dependencies': dependencies, 'implementation': config['implementation'],
        'cases': rows, 'candidate_reviews': list(candidates.values()),
        'summary': {'cases': len(rows), 'candidate_targets': len(candidates), 'source_regions': len(pending_regions),
                    'case_status': dict(Counter(row['decision']['status'] for row in rows)),
                    'candidate_status': dict(Counter(c['status'] for row in rows for c in row['decision']['candidates'])),
                    'native_conflict_targets': sum(bool(c['native']['native_conflicts']) for c in candidates.values())},
        'source_modified': False, 'split_modified': False, 'training_eligible': False,
        'limitations': ['Explicit source readings are assistant review assertions, not automatically verified text labels.',
                       'A choice is supported only within the explicitly reviewed candidate set.',
                       'Native unitless measurements and scale captions do not establish physical scale or endpoint accuracy.',
                       'Rejected alternative identity does not reject that target for its own different PNG.',
                       'Global visual thresholds, prior evidence and resource quarantine are preserved.']}
    _write(output/'report.json', report)
    parts = ['<!doctype html><html lang="ru"><meta charset="utf-8"><title>Различение вариантов болтов</title>',
             '<style>body{font:16px system-ui;margin:24px}img{max-width:100%;max-height:600px}td,th{padding:8px;border:1px solid #aaa}table{border-collapse:collapse}</style>',
             '<h1>Различение вариантов болтов</h1><p>Явное чтение фрагментов и проверка выбранных объектов DXF. Обучающие метки не утверждены.</p>']
    for row in rows:
        parts.append('<h2>'+html.escape(row['input_path'])+'</h2><p>'+html.escape(row['decision']['status'])+'</p>')
        parts.append('<table><tr><th>Кандидат</th><th>Решение</th><th>Противоречия с PNG</th><th>Внутренние конфликты DXF</th><th>Эталон</th></tr>')
        for candidate in row['decision']['candidates']:
            reviewed = candidates[candidate['id']]
            link = Path(os.path.relpath(root/reviewed['target_render']['path'], output)).as_posix()
            parts.append('<tr><td>'+html.escape(candidate['id'])+'</td><td>'+html.escape(candidate['status'])
                         +'</td><td>'+html.escape(json.dumps(candidate['contradictions'], ensure_ascii=False))
                         +'</td><td>'+html.escape(json.dumps(candidate['native_conflicts'], ensure_ascii=False))
                         +'</td><td><a href="'+html.escape(link, quote=True)+'">PNG</a></td></tr>')
        parts.append('</table>')
        for region in row['regions']:
            parts.append('<h3>'+html.escape(region['id'])+'</h3><img src="'+html.escape(region['output_png'], quote=True)+'">')
    parts.append('<p><a href="report.json">Полный журнал наблюдений и происхождения</a></p></html>')
    (output/'index.html').write_text(''.join(parts), encoding='utf-8')
    _write(output/'manifest.json', {'schema_version': '1.0', 'artifact_sha256': {
        path.relative_to(output).as_posix(): digest(path) for path in sorted(output.rglob('*')) if path.is_file()}})
    return report
