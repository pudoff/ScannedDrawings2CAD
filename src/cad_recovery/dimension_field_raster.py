"""Verify a sealed, raster-only dimension review without opening native drawings.

The pins are caller-supplied ``{'path': workspace_relative_path, 'sha256': hex}``
objects. Verification establishes file integrity, exact tile-derived pixels and
referential consistency. Authored readings, support windows and independence
attestations remain claims; they are neither expert labels nor training approval.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re

import numpy as np
from PIL import Image

from . import raster_regions

MAX_JSON_BYTES = 16 * 1024**2
MAX_FILE_BYTES = 52 * 1024**2
MAX_ROI_BYTES = 32 * 1024**2
MAX_REGION_PIXELS = 4_000_000
MAX_TOTAL_REGION_PIXELS = 256_000_000
MAX_SOURCES = 64
MAX_REGIONS = 2048
MAX_ITEMS_PER_SOURCE = 1024
INDEPENDENCE_KEYS = ('native_cad_opened', 'native_measurements_seen', 'alignment_seen',
                     'cad_projection_seen', 'localizer_results_seen')


def _mapping(value, label):
    if not isinstance(value, dict):
        raise ValueError(f'{label} must be an object')
    return value


def _inventory(value, label, maximum, *, nonempty=False):
    if not isinstance(value, list) or not int(nonempty) <= len(value) <= maximum:
        raise ValueError(f'{label} exceeds its inventory bounds or is not a list')
    return value


def _identifier(value, label):
    if not isinstance(value, str) or re.fullmatch(r'[a-z][a-z0-9_-]{0,95}', value) is None:
        raise ValueError(f'Invalid {label} ID')
    return value


def _sha(value, label):
    if not isinstance(value, str) or re.fullmatch(r'[0-9a-f]{64}', value) is None:
        raise ValueError(f'Invalid {label} SHA-256')
    return value


def _contained(root, relative, label):
    if (not isinstance(relative, str) or not relative or len(relative) > 1024
            or '\\' in relative or ':' in relative or '\x00' in relative
            or any(part in ('', '.', '..') for part in relative.split('/'))
            or PurePosixPath(relative).is_absolute()):
        raise ValueError(f'{label} path must be a portable contained relative path')
    path = (root / relative).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError(f'{label} path escapes its owner root')
    return path


def _bytes(path, limit, label):
    if not path.is_file() or not 0 < path.stat().st_size <= limit:
        raise ValueError(f'{label} file is missing, empty or exceeds byte limit')
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    if not raw or len(raw) > limit:
        raise ValueError(f'{label} exceeds byte limit while reading')
    return raw


def _file_sha(path, label, limit=MAX_FILE_BYTES):
    if not path.is_file() or not 0 < path.stat().st_size <= limit:
        raise ValueError(f'{label} file is missing, empty or exceeds byte limit')
    digest, count = hashlib.sha256(), 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(65536), b''):
            count += len(block)
            if count > limit:
                raise ValueError(f'{label} exceeds byte limit while hashing')
            digest.update(block)
    if not count:
        raise ValueError(f'{label} file is empty')
    return digest.hexdigest()


def _unique_json_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f'Nonfinite JSON constant: {value}')


def _json(raw, label):
    try:
        value = json.loads(raw.decode('utf-8'), object_pairs_hook=_unique_json_pairs,
                           parse_constant=_invalid_constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError(f'Invalid {label} JSON') from error
    return _mapping(value, label)


def _pinned_json(root, pin, label):
    pin = _mapping(pin, label + ' pin')
    path = _contained(root, pin.get('path'), label)
    expected = _sha(pin.get('sha256'), label)
    raw = _bytes(path, MAX_JSON_BYTES, label)
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f'{label} hash differs from caller pin')
    return path, _json(raw, label)


def _same(left, right):
    # JSON equality here deliberately distinguishes bool/int and int/float.
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
        right, sort_keys=True, allow_nan=False)


def _box(value, size, label):
    if (not isinstance(value, list) or len(value) != 4
            or any(type(item) is not int for item in value)):
        raise ValueError(f'{label} must be four integer source bounds')
    x0, y0, x1, y1 = value
    if not 0 <= x0 < x1 <= size[0] or not 0 <= y0 < y1 <= size[1]:
        raise ValueError(f'{label} lies outside source bounds')
    return (x1-x0)*(y1-y0)


def _context_ids(item, regions, label, *, singular=False):
    values = [item.get('context_region_id')] if singular else item.get('context_region_ids')
    values = _inventory(values, label + ' contexts', MAX_ITEMS_PER_SOURCE, nonempty=True)
    seen = set()
    for value in values:
        _identifier(value, label + ' context')
        if value not in regions or value in seen:
            raise ValueError(f'{label} has a missing, duplicate or foreign context ID')
        seen.add(value)


def _indexed(items, label, maximum, *, nonempty=False):
    result = {}
    for item in _inventory(items, label, maximum, nonempty=nonempty):
        item = _mapping(item, label + ' item')
        identifier = _identifier(item.get('id'), label)
        if identifier in result:
            raise ValueError(f'Duplicate {label} ID: {identifier}')
        result[identifier] = item
    return result


def _annotations(source, size, regions):
    readings = _indexed(source.get('dimension_readings'), 'reading', MAX_ITEMS_PER_SOURCE)
    for identifier, reading in readings.items():
        _box(reading.get('label_source_box_px'), size, 'reading label ROI')
        _context_ids(reading, regions, 'reading')
        label_id = identifier + '-label'
        if (label_id not in regions or regions[label_id].get('role') != 'dimension_label'
                or not _same(reading['label_source_box_px'], regions[label_id].get('source_box_px'))):
            raise ValueError('Reading label ROI does not exactly match its owned label crop')
        support = reading.get('author_support')
        if support is not None:
            support = _mapping(support, 'author support')
            for box in _inventory(support.get('source_windows_px'), 'support windows', 64, nonempty=True):
                _box(box, size, 'support window')
    views = _indexed(source.get('views'), 'view', MAX_ITEMS_PER_SOURCE)
    for view in views.values():
        _box(view.get('source_box_px'), size, 'view ROI')
        if 'label_source_box_px' in view:
            _box(view['label_source_box_px'], size, 'view label ROI')
        _context_ids(view, regions, 'view')
        if 'direction_marker_region_id' in view and view['direction_marker_region_id'] not in regions:
            raise ValueError('View direction marker has a missing or foreign context ID')
        if view.get('pixel_to_physical_scale_validated', False) is not False:
            raise ValueError('Raster review cannot validate physical scale')
    title = _mapping(source.get('title_readings'), 'title readings')
    _context_ids(title, regions, 'title', singular=True)
    if 'scale_source_box_px' in title:
        _box(title['scale_source_box_px'], size, 'title scale ROI')
    ambiguous = _indexed(source.get('ambiguous_points'), 'ambiguous point', MAX_ITEMS_PER_SOURCE)
    for item in ambiguous.values():
        _context_ids(item, regions, 'ambiguous point')
        if 'source_box_px' in item:
            _box(item['source_box_px'], size, 'ambiguous point ROI')
    curves = _indexed(source.get('curve_and_thin_line_observations'), 'curve/thin observation', MAX_ITEMS_PER_SOURCE)
    for item in curves.values():
        _box(item.get('source_box_px'), size, 'curve/thin ROI')
        _context_ids(item, regions, 'curve/thin observation')
    return readings, len(views), len(ambiguous), len(curves)


def verify_field_raster_evidence(workspace, report_pin, seal_pin):
    """Return pinned report data and bounded verification counts, or raise ValueError.

    IDs below each source are source-scoped: a ``title`` region may exist for
    several drawings, while duplicate (source ID, region/reading ID) is rejected.
    Neither saved builder nor any other executable from the package is executed.
    Original PNGs are streamed for hashing only; decoding is limited to ROIs and
    preparation tiles through ``read_prepared_region(max_pixels=4_000_000)``.
    """
    workspace = Path(workspace).resolve()
    if not workspace.is_dir():
        raise ValueError('Workspace must be an existing directory')
    report_path, report = _pinned_json(workspace, report_pin, 'report')
    seal_path, seal = _pinned_json(workspace, seal_pin, 'seal')
    package = report_path.parent
    if seal_path.parent != package or seal_path == report_path:
        raise ValueError('Seal must be a separate file in the report package')
    required = {report_path.name, 'builder.py', 'raster-review-config.json',
                'snapshots/builder.py', 'snapshots/raster-review-config.json'}
    if set(seal) != required:
        raise ValueError('Seal must list the report and all four builder/config snapshot files')
    manifest_paths = set()
    for relative, expected in seal.items():
        path = _contained(package, relative, 'snapshot')
        if path in manifest_paths:
            raise ValueError('Snapshot paths alias one another')
        manifest_paths.add(path)
        if _file_sha(path, 'snapshot') != _sha(expected, 'snapshot'):
            raise ValueError('Snapshot hash differs from sealed manifest')
    if seal[report_path.name] != report_pin['sha256']:
        raise ValueError('Report caller pin and seal disagree')
    for name in ('builder.py', 'raster-review-config.json'):
        if seal[name] != seal['snapshots/' + name]:
            raise ValueError('Builder/config differs from frozen snapshot')
    config = _json(_bytes(package / 'raster-review-config.json', MAX_JSON_BYTES, 'config'), 'config')
    if not _same(config, {key: value for key, value in report.items() if key not in ('regions', 'build')}):
        raise ValueError('Report annotations differ from frozen config')
    if report.get('schema_version') != '1.0':
        raise ValueError('Unsupported raster evidence schema')
    independence = _mapping(report.get('independence'), 'independence attestations')
    if any(independence.get(key) is not False for key in INDEPENDENCE_KEYS):
        raise ValueError('Raster-only independence attestations must explicitly be false')
    build = _mapping(report.get('build'), 'build')
    if (build.get('builder_sha256') != seal['builder.py']
            or build.get('config_sha256') != seal['raster-review-config.json']
            or build.get('reader_sha256') != _file_sha(Path(raster_regions.__file__), 'reader')
            or type(build.get('max_region_pixels')) is not int
            or build['max_region_pixels'] != MAX_REGION_PIXELS):
        raise ValueError('Build hashes or region budget disagree with verified inputs')
    for flag in ('original_png_decode_performed', 'roi_resampling_performed',
                 'roi_enhancement_performed', 'roi_rotation_performed'):
        if build.get(flag) is not False:
            raise ValueError('Build must explicitly attest unchanged raster samples')

    sources = _indexed(report.get('sources'), 'source', MAX_SOURCES, nonempty=True)
    regions_by_source = {identifier: {} for identifier in sources}
    roi_paths, total_pixels = set(), 0
    region_inventory = _inventory(report.get('regions'), 'region', MAX_REGIONS, nonempty=True)
    for region in region_inventory:
        region = _mapping(region, 'region')
        sid = _identifier(region.get('source_id'), 'region owner')
        rid = _identifier(region.get('id'), 'region')
        if sid not in sources:
            raise ValueError('Region has an unknown source owner')
        if rid in regions_by_source[sid]:
            raise ValueError('Duplicate region ID within source')
        relative = region.get('path')
        if relative != f'{sid}/{rid}.png':
            raise ValueError('Region path does not belong to its source/region IDs')
        path = _contained(package, relative, 'region')
        source_root = _contained(package, sid, 'region owner')
        if not path.is_relative_to(source_root) or path in roi_paths:
            raise ValueError('Region path escapes or aliases source ownership')
        roi_paths.add(path)
        regions_by_source[sid][rid] = region

    originals, preparations, source_sizes = {}, {}, {}
    counts = dict(source_count=len(sources), region_count=len(region_inventory),
                  dimension_reading_count=0, view_count=0, ambiguous_point_count=0,
                  curve_thin_observation_count=0, dimension_readings_by_source={})
    for sid, source in sources.items():
        original = _contained(workspace, source.get('source_path'), 'original PNG')
        preparation = _contained(workspace, source.get('preparation_path'), 'preparation')
        prep_path = _contained(preparation, 'preparation.json', 'preparation report')
        if (original.suffix.lower() != '.png' or original in originals
                or prep_path in preparations or original in roi_paths):
            raise ValueError('Original/preparation ownership is duplicated or invalid')
        source_sha = _sha(source.get('source_sha256'), 'original')
        if _file_sha(original, 'original before') != source_sha:
            raise ValueError('Original PNG hash differs before reconstruction')
        originals[original] = source_sha
        local = regions_by_source[sid]
        if not local:
            raise ValueError('Source has no raster regions')
        prep_pins = set()
        for region in local.values():
            provenance = _mapping(region.get('provenance'), 'region provenance')
            prep_pins.add(_sha(provenance.get('preparation_report_sha256'), 'preparation report'))
        if len(prep_pins) != 1:
            raise ValueError('Source regions disagree on preparation report pin')
        prep_raw = _bytes(prep_path, raster_regions.MAX_REPORT_BYTES, 'preparation report')
        prep_sha = hashlib.sha256(prep_raw).hexdigest()
        if prep_sha not in prep_pins:
            raise ValueError('Preparation report hash differs from region provenance pin')
        prep = _json(prep_raw, 'preparation report')
        size = prep.get('source_size_px')
        if (not isinstance(size, list) or len(size) != 2
                or any(type(value) is not int or not 1 <= value <= 32768 for value in size)
                or size[0]*size[1] > 300_000_000 or prep.get('source_sha256') != source_sha):
            raise ValueError('Preparation source identity or bounds differ')
        preparations[prep_path] = prep_sha
        source_sizes[sid] = size
        for region in local.values():
            pixels = _box(region.get('source_box_px'), size, 'region ROI')
            if pixels > MAX_REGION_PIXELS:
                raise ValueError('Region exceeds pixel budget')
            total_pixels += pixels
            if total_pixels > MAX_TOTAL_REGION_PIXELS:
                raise ValueError('Region inventory exceeds aggregate pixel budget')
        declared = _indexed(source.get('regions'), 'declared region', MAX_ITEMS_PER_SOURCE, nonempty=True)
        for rid, declaration in declared.items():
            if rid not in local or any(not _same(value, local[rid].get(key)) for key, value in declaration.items()):
                raise ValueError('Declared region differs from its owned raster record')
        readings, view_count, ambiguous_count, curve_count = _annotations(source, size, local)
        label_ids = {identifier + '-label' for identifier in readings}
        if set(declared) & label_ids or set(local) != set(declared) | label_ids:
            raise ValueError('Owned regions must exactly cover declarations and reading labels')
        counts['dimension_reading_count'] += len(readings)
        counts['dimension_readings_by_source'][sid] = len(readings)
        counts['view_count'] += view_count
        counts['ambiguous_point_count'] += ambiguous_count
        counts['curve_thin_observation_count'] += curve_count

    # Decode one bounded ROI at a time, compare its exact samples, then release it.
    for region in region_inventory:
        sid, box = region['source_id'], region['source_box_px']
        raw = _bytes(_contained(package, region['path'], 'region'), MAX_ROI_BYTES, 'region PNG')
        if hashlib.sha256(raw).hexdigest() != _sha(region.get('png_sha256'), 'region PNG'):
            raise ValueError('Region PNG hash differs from report')
        try:
            with Image.open(io.BytesIO(raw)) as picture:
                if (picture.format != 'PNG' or picture.mode != 'RGB'
                        or picture.size != (box[2]-box[0], box[3]-box[1])
                        or getattr(picture, 'n_frames', 1) != 1):
                    raise ValueError('Region PNG format or dimensions disagree')
                picture.verify()
            with Image.open(io.BytesIO(raw)) as picture:
                samples = np.asarray(picture).copy()
        except (OSError, SyntaxError) as error:
            raise ValueError('Region PNG integrity check failed') from error
        if hashlib.sha256(samples.tobytes()).hexdigest() != _sha(region.get('rgb_samples_sha256'), 'region RGB'):
            raise ValueError('Region RGB hash differs from report')
        preparation = _contained(workspace, sources[sid]['preparation_path'], 'preparation')
        rebuilt, provenance = raster_regions.read_prepared_region(preparation, box, max_pixels=MAX_REGION_PIXELS)
        if not np.array_equal(samples, rebuilt):
            raise ValueError('Region samples differ from reconstructed preparation pixels')
        if not _same(provenance, region['provenance']):
            raise ValueError('Region provenance differs from exact reconstruction provenance')
        if region.get('original_sha256_reverified') is not True:
            raise ValueError('Region is missing original rehash attestation')
        del raw, samples, rebuilt, provenance
    for original, expected in originals.items():
        if _file_sha(original, 'original after') != expected:
            raise ValueError('Original PNG changed during reconstruction')
    for prep_path, expected in preparations.items():
        if _file_sha(prep_path, 'preparation after', raster_regions.MAX_REPORT_BYTES) != expected:
            raise ValueError('Preparation report changed during reconstruction')
    counts.update(snapshot_file_count=len(seal), preparation_report_count=len(preparations),
                  region_pixels_total=total_pixels, max_region_pixels=MAX_REGION_PIXELS,
                  originals_hashed_before_after=True, all_region_rgb_reconstructed=True,
                  all_region_provenance_exact=True, original_pngs_decoded=False,
                  independence_attestations=independence,
                  independence_history_proven=False, expert_review_approved=False,
                  training_approved=False, physical_accuracy_verified=False,
                  limitations=[
                      'Independence metadata is an author attestation, not proof of access history.',
                      'Readings, view interpretations and support windows remain author assertions; no expert or training approval.',
                      'Printed scale and dimensions do not establish physical accuracy or pixel-to-physical calibration.',
                      'Original PNGs are hashed only; reconstruction verifies the pinned preparation tiles and their recorded provenance.',
                  ])
    return {'data': report, 'verification': counts}
