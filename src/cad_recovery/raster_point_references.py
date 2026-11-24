"""Validate sealed raster-only observations without reading CAD or localizers.

Integrity and coordinate conventions are checked here; the author's separation
from CAD is an attestation, not something that file hashes can prove.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re

import numpy as np
from PIL import Image

from .clear_pairs import contained, digest
from .raster_regions import read_prepared_region

MAX_JSON_BYTES = 8 * 1024**2
MAX_SOURCES = 100
MAX_POINTS = 1000
MAX_ASSETS = 4000
MAX_CROP_PIXELS = 4_194_304
MAX_TOTAL_CROP_PIXELS = 32_000_000


def _json(path):
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError('Raster reference JSON exceeds size budget')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    def reject_constant(value):
        raise ValueError('Nonfinite JSON number')
    result = json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=unique,
                        parse_constant=reject_constant)
    if not isinstance(result, dict):
        raise ValueError('Expected raster reference JSON object')
    return result


def _pin(path, expected):
    if not isinstance(expected, str) or re.fullmatch('[0-9a-f]{64}', expected) is None:
        raise ValueError('Invalid SHA-256 pin')
    if digest(path) != expected:
        raise ValueError(f'Raster reference hash mismatch: {path.name}')


def _numbers(value, count, *, integers=False):
    if (not isinstance(value, list) or len(value) != count
            or any(isinstance(x, bool) or not isinstance(x, (int, float))
                   or not math.isfinite(x) or (integers and not isinstance(x, int)) for x in value)):
        raise ValueError('Invalid finite coordinate array')
    return value


def _identity(value, seen):
    if not isinstance(value, str) or not value or len(value) > 200 or value in seen:
        raise ValueError('Duplicate or invalid reference identity')
    seen.add(value)


def validate_raster_references(workspace: Path, reference_path: Path, seal_path: Path, *,
                              expected_reference_sha256: str, expected_seal_sha256: str) -> dict:
    """Rehash sources and reconstruct bounded crops; never decode original PNGs.

    The caller supplies both independent pins. Successful validation does not
    approve annotations or establish statistical accuracy of the intervals.
    """
    root = Path(workspace).resolve()
    reference_path, seal_path = Path(reference_path).resolve(), Path(seal_path).resolve()
    if any(path == root or not path.is_relative_to(root) for path in (reference_path, seal_path)):
        raise ValueError('Reference paths must stay within workspace')
    _pin(reference_path, expected_reference_sha256)
    _pin(seal_path, expected_seal_sha256)
    reference, seal = _json(reference_path), _json(seal_path)
    if reference.get('schema_version') != '1.0' or seal.get('schema_version') != '1.0':
        raise ValueError('Unknown raster reference schema')
    if (contained(root, seal['reference_path']) != reference_path
            or seal['reference_sha256'] != expected_reference_sha256
            or seal.get('status') != 'sealed_before_CAD_association'):
        raise ValueError('Seal does not identify the raster reference snapshot')
    if (reference.get('reviewed_without_CAD_or_localizer_outputs') is not True
            or reference.get('independent_of_cad') is not True
            or reference.get('coordinate_system') != 'Original PNG pixel centers; X right, Y down; intervals inclusive.'):
        raise ValueError('Unsupported reference independence or coordinate declaration')
    for record in (reference, seal):
        if record.get('annotation_approved') is not False or record.get('training_eligible') is not False:
            raise ValueError('This workflow requires unapproved reference observations')
    source_records, points = reference['sources'], reference['points']
    if (not isinstance(source_records, list) or not 1 <= len(source_records) <= MAX_SOURCES
            or not isinstance(points, list) or not 1 <= len(points) <= MAX_POINTS
            or seal['source_count'] != len(source_records) or seal['point_count'] != len(points)):
        raise ValueError('Invalid reference inventory or size budget')
    base = seal_path.parent
    assets = seal['artifact_sha256']
    if not isinstance(assets, dict) or not 1 <= len(assets) <= MAX_ASSETS:
        raise ValueError('Invalid sealed asset inventory')
    asset_paths = {}
    for relative, expected in assets.items():
        path = contained(base, relative)
        if path in asset_paths:
            raise ValueError('Duplicate sealed asset path')
        _pin(path, expected)
        asset_paths[path] = expected
    _pin(base/'build_references.py', seal['builder_sha256'])
    source_ids, source_paths, source_hashes, sources = set(), set(), set(), {}
    for source in source_records:
        _identity(source['id'], source_ids)
        path = contained(root/'dataset', source['input_path'])
        if path in source_paths or source['source_sha256'] in source_hashes:
            raise ValueError('Duplicate raster source')
        source_paths.add(path)
        source_hashes.add(source['source_sha256'])
        _pin(path, source['source_sha256'])
        size = _numbers(source['source_size_px'], 2, integers=True)
        if min(size) < 1:
            raise ValueError('Invalid source image size')
        prepared = contained(root, source['prepared_dir'])
        prep_path = prepared/'preparation.json'
        _pin(prep_path, source['preparation_report_sha256'])
        preparation = _json(prep_path)
        if (preparation['source_sha256'] != source['source_sha256']
                or preparation['source_size_px'] != size):
            raise ValueError('Preparation source identity differs from reference')
        sources[source['id']] = source
    point_ids, used_sources, total_pixels = set(), set(), 0
    staged = []
    for point in points:
        _identity(point['id'], point_ids)
        if point['source_id'] not in sources:
            raise ValueError('Unknown raster source identity')
        source = sources[point['source_id']]
        used_sources.add(point['source_id'])
        if any(point[key] != source[key] for key in ('input_path', 'source_sha256')):
            raise ValueError('Point source identity differs from source inventory')
        if point.get('annotation_approved') is not False or point.get('training_eligible') is not False:
            raise ValueError('Reference point must remain unapproved')
        x, y = _numbers(point['center_source_xy'], 2)
        x0, y0, x1, y1 = _numbers(point['interval_source_xyxy'], 4)
        width, height = source['source_size_px']
        if not (0 <= x0 <= x <= x1 < width and 0 <= y0 <= y <= y1 < height and x0 < x1 and y0 < y1):
            raise ValueError('Reference interval is empty, outside image or excludes center')
        evidence = point['evidence']
        bx0, by0, bx1, by1 = _numbers(evidence['source_box_px'], 4, integers=True)
        if not (0 <= bx0 <= x0 <= x1 < bx1 <= width and 0 <= by0 <= y0 <= y1 < by1 <= height):
            raise ValueError('Reference interval must be inside its half-open crop')
        count = (bx1-bx0)*(by1-by0)
        total_pixels += count
        if count > MAX_CROP_PIXELS or total_pixels > MAX_TOTAL_CROP_PIXELS:
            raise ValueError('Reference crop pixel budget exceeded')
        raw_path = contained(root, evidence['raw_crop_path'])
        overlay_path = contained(root, evidence['interval_overlay_path'])
        if raw_path not in asset_paths or overlay_path not in asset_paths:
            raise ValueError('Reference crop and overlay must both be sealed')
        if evidence['raw_crop_sha256'] != asset_paths[raw_path]:
            raise ValueError('Raw crop pin differs from seal')
        staged.append((point, source, raw_path, (bx1-bx0, by1-by0)))
    if used_sources != source_ids:
        raise ValueError('Unused source in reference inventory')
    for point, source, raw_path, crop_size in staged:
        evidence = point['evidence']
        rgb, provenance = read_prepared_region(contained(root, source['prepared_dir']),
                                               evidence['source_box_px'], max_pixels=MAX_CROP_PIXELS)
        if provenance != evidence['provenance']:
            raise ValueError('Raw crop provenance differs from reconstructed source')
        with Image.open(raw_path) as picture:
            if picture.size != crop_size or picture.mode != 'RGB' or picture.format != 'PNG':
                raise ValueError('Unexpected raw crop dimensions or encoding')
            if not np.array_equal(np.asarray(picture), rgb):
                raise ValueError('Raw crop pixels differ from prepared source')
    return {'references': reference, 'verification': {
        'reference_sha256': expected_reference_sha256, 'seal_sha256': expected_seal_sha256,
        'source_files_rehashed': len(sources), 'preparation_reports_rehashed': len(sources),
        'sealed_assets_rehashed': len(assets), 'builder_rehashed': True,
        'exact_crops_reconstructed': len(points), 'crop_pixels_checked': total_pixels,
        'original_png_decoded': False, 'cad_or_localizer_accessed': False,
        'independence_attestation_verified_as_history': False,
        'interval_accuracy_validated': False, 'annotation_approved': False,
        'training_eligible': False}}
