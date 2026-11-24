"""Review prepared customer rasters against separate CAD references.

Visual correspondence, coordinate registration and semantic label approval are
separate conclusions. This workflow never supplies the DXF to the converter.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import platform
import time
import unicodedata

import cv2
import numpy as np
from PIL import Image

from .pairing import compare_renders, THRESHOLDS, METHOD, _save_overlay


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    path = Path(path)
    if path.stat().st_size > 50 * 1024**2:
        raise ValueError('JSON exceeds 50 MiB')
    return json.loads(path.read_text(encoding='utf-8'))


def contained(root, relative):
    if not isinstance(relative, str) or not relative or '\\' in relative:
        raise ValueError('Expected a portable relative artifact path')
    parts = PurePosixPath(relative)
    if parts.is_absolute() or '..' in parts.parts or ':' in relative:
        raise ValueError('Path must stay inside its declared root')
    result = (root / relative).resolve()
    if result == root or not result.is_relative_to(root):
        raise ValueError('Path escapes its declared root')
    return result


def normalize_raster(gray, max_side=1600):
    """Crop blank margins and resize; retain center-coordinate transforms."""
    if gray.ndim != 2 or gray.dtype != np.uint8 or gray.size > 16_777_216:
        raise ValueError('Expected bounded uint8 grayscale overview')
    y, x = np.where(gray < 210)
    if len(x) < 100:
        raise ValueError('Insufficient drawing content')
    left, top = max(0, int(x.min())-3), max(0, int(y.min())-3)
    right, bottom = min(gray.shape[1], int(x.max())+4), min(gray.shape[0], int(y.max())+4)
    crop = gray[top:bottom, left:right]
    scale = max_side / max(crop.shape)
    width, height = max(1, round(crop.shape[1]*scale)), max(1, round(crop.shape[0]*scale))
    sx, sy = width/crop.shape[1], height/crop.shape[0]
    matrix = np.array([[sx, 0., sx*(.5-left)-.5], [0., sy, sy*(.5-top)-.5], [0., 0., 1.]])
    pixels = cv2.resize(crop, (width, height), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    return pixels, {'input_size_px': list(gray.shape[::-1]), 'crop_box_px': [left, top, right, bottom],
                    'normalized_size_px': [width, height], 'input_to_normalized': matrix.tolist(),
                    'normalized_to_input': np.linalg.inv(matrix).tolist()}


def _candidate_key(path, *, raster=False):
    stem = unicodedata.normalize('NFKC', PurePosixPath(path).stem).casefold().strip()
    if raster and stem.endswith('pdf'):
        stem = stem[:-3].rstrip()
    return ' '.join(stem.replace('_', ' ').split())


def propose_candidates(intake):
    """Name aliases propose candidates only; ambiguity never picks a winner."""
    unmatched = intake.get('unmatched_dxf_targets', [])
    result = []
    for item in intake['items']:
        target = item.get('candidate_target')
        method = 'intake_unique_normalized_name'
        candidates = [target] if target else [value for value in unmatched
            if _candidate_key(value['path']) == _candidate_key(item['input_path'], raster=True)]
        if not target:
            method = 'unique_unmatched_alias_spaces_underscores_or_pdf_suffix'
        result.append({'input': item, 'targets': candidates, 'method': method,
                       'name_is_verification': False})
    return result


def load_prepared_overview(root, batch_item, intake_item):
    page_path = contained(root, batch_item['preparation_report'])
    page = read_json(page_path)
    if batch_item['status'] != 'prepared' or page['status'] != 'prepared' or not page['source_integrity_validated']:
        raise ValueError('Prepared source is not fully validated')
    if page['source_sha256'] != intake_item['input_sha256'] or batch_item['input_sha256'] != intake_item['input_sha256']:
        raise ValueError('Prepared source hash differs from intake')
    size = [intake_item['png_ihdr']['width_px'], intake_item['png_ihdr']['height_px']]
    if page['source_size_px'] != size:
        raise ValueError('Prepared dimensions differ from intake')
    overview = page['overview']
    path = contained(page_path.parent, overview['path'])
    if digest(path) != overview['sha256']:
        raise ValueError('Prepared overview hash changed')
    factor = overview['scale_factor']
    if isinstance(factor, bool) or not isinstance(factor, int) or factor < 1:
        raise ValueError('Invalid overview scale factor')
    expected = np.array([[factor, 0., (factor-1)/2], [0., factor, (factor-1)/2], [0., 0., 1.]])
    if not np.allclose(overview['overview_to_source'], expected, atol=1e-12, rtol=0):
        raise ValueError('Inconsistent overview coordinate provenance')
    if not np.allclose(overview['source_to_overview'], np.linalg.inv(expected), atol=1e-12, rtol=0):
        raise ValueError('Inconsistent inverse overview transform')
    with Image.open(path) as picture:
        if picture.width * picture.height > 16_777_216 or list(picture.size) != [math.ceil(v/factor) for v in size]:
            raise ValueError('Invalid prepared overview size')
        gray = np.asarray(picture.convert('L'))
    provenance = {'preparation_report': batch_item['preparation_report'],
                  'preparation_report_sha256': digest(page_path), 'overview_sha256': digest(path),
                  'overview_to_source': expected.tolist(), 'source_size_px': size}
    return gray, provenance


def compose_alignment(source_provenance, normalization, reference, evidence):
    affine = evidence.get('affine_target_to_source')
    if affine is None:
        return None
    registration = np.vstack([np.asarray(affine, dtype=float), [0., 0., 1.]])
    matrix = (np.array(source_provenance['overview_to_source']) @
              np.array(normalization['normalized_to_input']) @ registration @
              np.array(reference['cad_to_pixel']))
    if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-15:
        raise ValueError('Invalid composed registration')
    normalized_to_source = (np.array(source_provenance['overview_to_source']) @
                            np.array(normalization['normalized_to_input']))
    tolerance = evidence.get('tolerance_pixels', 0) * float(np.linalg.norm(normalized_to_source[:2, :2], ord=2))
    return {'status': 'visual_alignment_candidate' if evidence['status'] == 'verified' else 'diagnostic_only',
            'cad_to_source_pixel': matrix.tolist(), 'source_pixel_to_cad': np.linalg.inv(matrix).tolist(),
            'cad_coordinate_system': 'DXF modelspace WCS XY in declared drawing units, not inferred millimeters',
            'raster_coordinate_system': 'original PNG pixel centers; top-left origin; X right, Y down',
            'comparison_tolerance_in_source_pixels': tolerance,
            'tolerance_is_accuracy_bound': False, 'annotation_transform_approved': False,
            'physical_scale_confirmed': False,
            'limitations': ['Global similarity registration is an estimate, not a per-view or per-object approval.',
                            'Comparison tolerance is not a measured maximum registration error.']}


def _write(path, data):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)


def project_dimension_evidence(semantics, alignment, source_size):
    """Expose tentative anchor locations for review, never an accepted label."""
    matrix = np.asarray(alignment['cad_to_source_pixel'], dtype=float)
    width, height = source_size
    dimensions = []
    for dimension in semantics['dimensions']:
        projected = {}
        for name, point in dimension.get('definition_points_modelspace', {}).items():
            if not isinstance(point, list) or len(point) != 3 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in point):
                projected[name] = {'status': 'invalid_or_unavailable_point'}
                continue
            x, y, homogeneous = matrix @ [point[0], point[1], 1.]
            if not np.isfinite([x, y, homogeneous]).all() or abs(homogeneous) < 1e-15:
                projected[name] = {'status': 'invalid_projection'}
                continue
            x, y = float(x/homogeneous), float(y/homogeneous)
            projected[name] = {'source_pixel_center_xy': [x, y], 'cad_z': point[2],
                               'inside_source_pixel_area': -.5 <= x < width-.5 and -.5 <= y < height-.5,
                               'planar_xy_only': True, 'approved': False}
        dimensions.append({'handle': dimension['handle'], 'insert_path': dimension.get('insert_path', []),
                           'entity_type': dimension.get('entity_type'), 'kind': dimension.get('kind'),
                           'native_measurement': dimension.get('measurement_raw'),
                           'text_override': dimension.get('text_override'),
                           'stored_display_texts': dimension.get('stored_display_texts', []),
                           'numeric_consistency': dimension.get('display_comparison'),
                           'points': projected, 'training_eligible': False,
                           'review_reasons': dimension.get('reasons', [])})
    return {'schema_version': '1.0', 'purpose': 'tentative_dimension_anchor_review',
            'target_sha256': semantics['source_sha256'], 'source_size_px': source_size,
            'registration_status': alignment['status'], 'dimensions': dimensions,
            'training_eligible': False, 'coordinate_system': 'original PNG pixel centers; X right, Y down',
            'limitations': ['A projected CAD definition point is not a confirmed measured-part attachment.',
                            'Global registration uncertainty and nonplanar coordinates require individual review.']}


def verify_clear_pairs(dataset, intake_path, prepared_dir, output, *, max_side=1600, limit=None):
    from .dxf_reference import render_dxf_reference
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
              'comparison_method': METHOD, 'thresholds': dict(THRESHOLDS), 'max_side': max_side,
              'training_eligible': False, 'split_modified': False, 'pairs': [],
              'runtime': {'python': platform.python_version(), 'opencv': cv2.__version__, 'numpy': np.__version__},
              'limitations': ['Grid overlap tests are not semantic view segmentation.',
                              'A verified visual match does not verify individual dimensions, layers or geometry.',
                              'No training labels or split assignments are issued.']}
    report['implementation'] = {name: digest(Path(__file__).with_name(name)) for name in
                                ('clear_pairs.py', 'dxf_reference.py', 'dxf_semantics.py', 'pairing.py')}

    def checkpoint(final=False):
        report['processed_count'] = len(report['pairs'])
        report['complete'] = report['processed_count'] == report['candidate_count']
        report['status'] = ('complete' if report['complete'] else 'partial') if final else 'running'
        report['summary'] = {'visual_status': dict(Counter(p['visual_status'] for p in report['pairs'])),
                             'pair_status': dict(Counter(p['status'] for p in report['pairs'])),
                             'errors': sum(bool(p.get('errors')) for p in report['pairs'])}
        report['elapsed_seconds'] = round(time.perf_counter()-started, 4)
        temporary = output / '.report-writing.json'
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
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
            gray, provenance = load_prepared_overview(prepared_dir, prepared_items[item['input_path']], item)
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
            reference_pixels, reference = render_dxf_reference(target, max_side=max_side)
            _write(directory / 'reference.json', reference)
            Image.fromarray(reference_pixels).save(directory / 'target.png')
            pair['artifacts'].update(reference=f'{directory.name}/reference.json', target=f'{directory.name}/target.png')
            evidence = compare_renders(source_pixels, reference_pixels)
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
            pair['errors'].append({'stage': 'pair', 'type': type(error).__name__, 'message': str(error)})
        pair['artifact_sha256'] = {name: digest(output / relative) for name, relative in pair['artifacts'].items()}
        pair['elapsed_seconds'] = round(time.perf_counter()-tick, 4)
        _write(directory / 'pair.json', pair)
        report['pairs'].append(pair)
        checkpoint()
        print(f'{index+1}/{len(candidates)} {pair["visual_status"]} errors={len(pair["errors"])}', flush=True)
    checkpoint(final=True)
    return report
