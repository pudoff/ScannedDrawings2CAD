"""Versioned pair review with explicit incomplete references and lossless JSON.

The original comparison thresholds, normalization and alignment functions are
shared unchanged. Incomplete resource renders never qualify as verified pairs.
"""
from collections import Counter
import json
from pathlib import Path
import platform
import time

import cv2
import numpy as np
from PIL import Image

from .clear_pairs import (digest, read_json, contained, propose_candidates,
    load_prepared_overview, normalize_raster, compose_alignment, project_dimension_evidence)
from .pairing import compare_renders, THRESHOLDS, METHOD, _save_overlay


def _write(path, data):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=True, indent=2, allow_nan=False)
        stream.write('\n')


def verify_clear_pairs_v2(dataset, intake_path, prepared_dir, output, *, max_side=1600, limit=None):
    from .dxf_reference_v4 import render_dxf_reference
    from .dxf_semantics import audit_dxf_semantics

    started = time.perf_counter()
    dataset, prepared_dir, output = Path(dataset).resolve(), Path(prepared_dir).resolve(), Path(output).resolve()
    if output.is_relative_to(dataset) or output.is_relative_to(prepared_dir):
        raise ValueError('Review output must be outside dataset and prepared sources')
    if output.exists():
        raise FileExistsError('Review output already exists; choose a new directory')
    if isinstance(max_side, bool) or not isinstance(max_side, int) or not 320 <= max_side <= 2400:
        raise ValueError('max_side must be an integer in 320..2400')
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError('limit must be a positive integer')
    intake = read_json(intake_path)
    prepared_report = prepared_dir / 'report.json'
    prepared = read_json(prepared_report)
    if intake.get('purpose') != 'training_intake_not_split' or intake.get('schema_version') != '1.0':
        raise ValueError('Expected training intake version 1')
    if prepared.get('status') != 'complete' or prepared.get('intake_sha256') != digest(intake_path):
        raise ValueError('Preparation must be complete and match this intake')
    if not 1 <= len(intake['items']) <= 10000:
        raise ValueError('Expected 1..10000 registered inputs')
    prepared_items = {item['input_path']: item for item in prepared['items']}
    names = [item['input_path'] for item in intake['items']]
    if len(set(names)) != len(names) or len(prepared_items) != len(prepared['items']) or set(names) != set(prepared_items):
        raise ValueError('Intake and prepared input identities differ or repeat')
    candidates = propose_candidates(intake)
    uses = Counter(target['path'] for candidate in candidates for target in candidate['targets'])
    output.mkdir(parents=True, exist_ok=False)
    report = {'schema_version': '1.0', 'purpose': 'clear_raster_pair_and_semantic_review',
              'status': 'running', 'intake_sha256': digest(intake_path),
              'preparation_batch_sha256': digest(prepared_report), 'candidate_count': len(candidates),
              'review_version': 'v2-explicit-reference-completeness',
              'comparison_method': METHOD, 'thresholds': dict(THRESHOLDS), 'max_side': max_side,
              'training_eligible': False, 'split_modified': False, 'pairs': [],
              'runtime': {'python': platform.python_version(), 'opencv': cv2.__version__, 'numpy': np.__version__},
              'limitations': ['Grid overlap tests are not semantic view segmentation.',
                              'A verified visual match does not verify individual dimensions, layers or geometry.',
                              'No training labels or split assignments are issued.',
                              'Missing IMAGE/OLE resources produce explicitly incomplete diagnostic references and cannot verify a pair.',
                              'JSON escapes preserve native problematic code points without correcting their text.']}
    report['implementation'] = {name: digest(Path(__file__).with_name(name)) for name in
                                ('clear_pairs.py', 'clear_pairs_v2.py', 'dxf_reference.py', 'dxf_reference_v4.py',
                                 'mtext_zero_height.py', 'reference_resources.py', 'dxf_semantics.py', 'pairing.py')}

    def checkpoint(final=False):
        report['processed_count'] = len(report['pairs'])
        report['complete'] = report['processed_count'] == report['candidate_count']
        report['status'] = ('complete' if report['complete'] else 'partial') if final else 'running'
        report['summary'] = {'visual_status': dict(Counter(p['visual_status'] for p in report['pairs'])),
                             'pair_status': dict(Counter(p['status'] for p in report['pairs'])),
                             'errors': sum(bool(p.get('errors')) for p in report['pairs']),
                             'reference_status': dict(Counter(p.get('reference_status', 'unavailable') for p in report['pairs'])),
                             'raw_visual_status': dict(Counter(p.get('raw_visual_status', 'unavailable') for p in report['pairs']))}
        report['elapsed_seconds'] = round(time.perf_counter()-started, 4)
        temporary = output / '.report-writing.json'
        temporary.write_text(json.dumps(report, ensure_ascii=True, indent=2, allow_nan=False), encoding='utf-8')
        temporary.replace(output / 'report.json')

    cv2.setNumThreads(1)
    checkpoint()
    for index, candidate in enumerate(candidates[:limit]):
        tick = time.perf_counter()
        item = candidate['input']
        pair = {'input_path': item['input_path'], 'input_sha256': item['input_sha256'],
                'candidate_method': candidate['method'], 'status': 'quarantine', 'visual_status': 'quarantine',
                'training_eligible': False, 'annotation_status': 'not_approved', 'errors': [], 'artifacts': {}}
        directory = output / f'pair-{index+1:04d}'
        directory.mkdir()
        try:
            source = contained(dataset, item['input_path'])
            if digest(source) != item['input_sha256']:
                raise ValueError('Source PNG differs from intake')
            targets = candidate['targets']
            if len(targets) != 1 or uses[targets[0]['path']] != 1:
                raise ValueError('Target candidate is missing, ambiguous or reused')
            target_item = targets[0]
            target = contained(dataset, target_item['path'])
            pair.update(target_path=target_item['path'], target_sha256=target_item['sha256'])
            if digest(target) != target_item['sha256']:
                raise ValueError('Source DXF differs from intake')
            prepared_item = prepared_items[item['input_path']]
            if digest(contained(prepared_dir, prepared_item['preparation_report'])) != prepared_item['preparation_report_sha256']:
                raise ValueError('Prepared report changed from the pinned batch')
            gray, provenance = load_prepared_overview(prepared_dir, prepared_item, item)
            source_pixels, normalization = normalize_raster(gray, max_side)
            pair.update(source_provenance=provenance, source_normalization=normalization)
            Image.fromarray(source_pixels).save(directory / 'source.png')
            pair['artifacts']['source'] = f'{directory.name}/source.png'
            # Audit semantics independently of rendering success.
            semantics = None
            try:
                semantics = audit_dxf_semantics(target)
                _write(directory / 'semantics.json', semantics)
                pair['artifacts']['semantics'] = f'{directory.name}/semantics.json'
                pair['semantic_review_status'] = semantics.get('status', 'needs_review')
            except Exception as error:
                pair['errors'].append({'stage': 'semantics', 'type': type(error).__name__, 'message': str(error)})
            reference_pixels, reference = render_dxf_reference(target, max_side=max_side,
                allow_incomplete_reference=True, resource_root=dataset)
            _write(directory / 'reference.json', reference)
            Image.fromarray(reference_pixels).save(directory / 'target.png')
            pair['artifacts'].update(reference=f'{directory.name}/reference.json', target=f'{directory.name}/target.png')
            evidence = compare_renders(source_pixels, reference_pixels)
            pair['raw_visual_status'] = evidence['status']
            pair['reference_complete'] = reference['reference_complete']
            pair['reference_status'] = 'complete' if reference['reference_complete'] else 'incomplete_diagnostic'
            if not reference['reference_complete']:
                pair['diagnostic_evidence'] = evidence
                evidence = {**evidence, 'status': 'quarantine', 'diagnostic_only': True,
                            'reasons': [*evidence.get('reasons', []), 'incomplete_reference_resources']}
            pair['evidence'] = evidence
            pair['visual_status'] = evidence['status']
            pair['status'] = 'review_required' if evidence['status'] == 'verified' else 'quarantine'
            pair['alignment'] = compose_alignment(provenance, normalization, reference, evidence)
            if pair['alignment']:
                if semantics is not None:
                    anchors = project_dimension_evidence(semantics, pair['alignment'], provenance['source_size_px'])
                    anchors['input_sha256'] = item['input_sha256']
                    _write(directory / 'dimension-anchor-review.json', anchors)
                    pair['artifacts']['dimension_anchors'] = f'{directory.name}/dimension-anchor-review.json'
                _save_overlay(source_pixels, reference_pixels, evidence, directory / 'overlay.png')
                pair['artifacts']['overlay'] = f'{directory.name}/overlay.png'
            if digest(source) != item['input_sha256'] or digest(target) != target_item['sha256']:
                raise ValueError('Source changed while reviewing pair')
        except Exception as error:
            pair['status'] = pair['visual_status'] = 'quarantine'
            pair['alignment'] = None
            pair['reference_status'] = 'unavailable_due_to_execution_error'
            pair['errors'].append({'stage': 'pair', 'type': type(error).__name__, 'message': str(error)})
        pair['artifact_sha256'] = {name: digest(output / relative) for name, relative in pair['artifacts'].items()}
        pair['elapsed_seconds'] = round(time.perf_counter()-tick, 4)
        _write(directory / 'pair.json', pair)
        report['pairs'].append(pair)
        checkpoint()
        print(f'{index+1}/{len(candidates)} {pair["visual_status"]} errors={len(pair["errors"])}', flush=True)
    if any(digest(Path(__file__).with_name(name)) != expected for name, expected in report['implementation'].items()):
        raise ValueError('Pair review implementation changed during execution')
    checkpoint(final=True)
    return report
