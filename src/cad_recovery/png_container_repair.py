"""Explicit, pinned preservation of a complete PNG prefix and excluded trailer.

This is a case-specific container repair, never automatic recovery of arbitrary
corrupt PNGs. Original bytes are retained and never described as a valid PNG.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import re
import time

from PIL import Image

from .opaque_rgba import MAX_FILE_BYTES, _validate_png, prepare_opaque_rgba

METHOD = 'retain_pinned_complete_png_prefix'
INTEGRITY_SCOPE = 'original_bytes_pinned_and_partitioned; PNG_structure_validation_applies_to_canonical_prefix'
IDENTITY = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _file_sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def _case(value):
    if not isinstance(value, dict) or value.get('method') != METHOD:
        raise ValueError('An explicit pinned PNG prefix repair case is required')
    try:
        serialized = json.dumps(value, allow_nan=False, ensure_ascii=False)
    except (ValueError, TypeError) as error:
        raise ValueError('Repair case must contain finite JSON data') from error
    if len(serialized.encode('utf-8')) > 16384:
        raise ValueError('Repair case exceeds its metadata budget')
    for name in ('source_sha256', 'prefix_sha256', 'trailer_sha256'):
        pin = value.get(name)
        if not isinstance(pin, str) or re.fullmatch('[0-9a-f]{64}', pin) is None:
            raise ValueError('Repair case requires three valid SHA-256 pins')
    for name in ('source_size_bytes', 'prefix_size_bytes', 'trailer_size_bytes'):
        count = value.get(name)
        if isinstance(count, bool) or not isinstance(count, int) or not 0 < count <= MAX_FILE_BYTES:
            raise ValueError('Repair case byte counts exceed supported limits')
    if (value['prefix_size_bytes'] < 45
            or value['source_size_bytes'] != value['prefix_size_bytes']+value['trailer_size_bytes']):
        raise ValueError('Repair case must partition source exactly into PNG prefix and nonempty trailer')
    return deepcopy(value)


def prepare_repaired_opaque_rgba(source: Path, output_dir: Path, *, repair_case: dict,
                                 tile_size=2048, overlap=128, overview_max_side=2400,
                                 strip_rows=128) -> dict:
    """Prepare an allowlisted nonconformant source through its exact valid prefix.

    All byte identities and the reason for PNG rejection are checked before
    creating output. Canonical PNG and excluded bytes are separately archived.
    """
    started = time.perf_counter()
    source, requested_output = Path(source).resolve(), Path(output_dir)
    if requested_output.exists() or requested_output.is_symlink():
        raise FileExistsError('Repair output already exists; choose a new directory')
    output = requested_output.resolve()
    if output.exists() or source == output or source.is_relative_to(output):
        raise ValueError('Repair output must not contain the original source')
    case = _case(repair_case)
    size = source.stat().st_size
    if size != case['source_size_bytes'] or not 0 < size <= MAX_FILE_BYTES:
        raise ValueError('Original byte count differs from pinned repair case')
    with source.open('rb') as stream:
        original = stream.read(MAX_FILE_BYTES+1)
    if len(original) != size or _sha(original) != case['source_sha256']:
        raise ValueError('Original source differs from pinned repair case')
    prefix = original[:case['prefix_size_bytes']]
    trailer = original[case['prefix_size_bytes']:]
    if (_sha(prefix) != case['prefix_sha256'] or len(trailer) != case['trailer_size_bytes']
            or _sha(trailer) != case['trailer_sha256']):
        raise ValueError('PNG prefix or excluded trailer differs from pinned partition')
    minimal, metadata = _validate_png(prefix)
    try:
        _validate_png(original)
    except ValueError as error:
        if str(error) != 'Unexpected data after PNG IEND':
            raise ValueError('Original rejection is not exclusively an appended PNG trailer') from error
    else:
        raise ValueError('Repair requires a nonconformant original with an appended trailer')
    # Confirm the full-pixel condition before writing a derived repair. The
    # existing adapter repeats this check when preparing the canonical file.
    with Image.open(io.BytesIO(minimal)) as picture:
        picture.load()
        if picture.mode != 'RGBA' or getattr(picture, 'n_frames', 1) != 1:
            raise ValueError('Repair prefix must be a static RGBA image')
        with picture.getchannel('A') as alpha:
            if alpha.getextrema() != (255, 255):
                raise ValueError('Repair prefix must have alpha=255 for every pixel')
    if _file_sha(source) != case['source_sha256']:
        raise ValueError('Original source changed before repair output')
    output.mkdir(parents=True, exist_ok=False)
    try:
        canonical, excluded = output/'canonical-rgba.png', output/'excluded-trailer.bin'
        with canonical.open('xb') as stream:
            stream.write(prefix)
        with excluded.open('xb') as stream:
            stream.write(trailer)
        if _file_sha(canonical) != case['prefix_sha256'] or _file_sha(excluded) != case['trailer_sha256']:
            raise ValueError('Written repair partition differs from its pins')
        del original, prefix, trailer, minimal
        child_dir = output/'rgba-prepared'
        child = prepare_opaque_rgba(canonical, child_dir, expected_sha256=case['prefix_sha256'],
            tile_size=tile_size, overlap=overlap, overview_max_side=overview_max_side,
            strip_rows=strip_rows)
        if child.get('status') != 'prepared' or child.get('source_integrity_validated') is not True:
            raise ValueError('Canonical RGBA preparation did not fully validate')
        if _file_sha(source) != case['source_sha256']:
            raise ValueError('Original source changed during repair preparation')
        report = deepcopy(child)
        for artifact in [report['overview'], report['ink_overview'], *report['tiles']]:
            artifact['path'] = 'rgba-prepared/'+artifact['path']
        conversion = report['rgba_conversion']
        for key in ('rgb_derivative', 'rgb_preparation_report'):
            conversion[key]['path'] = 'rgba-prepared/'+conversion[key]['path']
        conversion['source_sha256'] = case['source_sha256']
        conversion['decoded_source_sha256'] = case['prefix_sha256']
        report['decoder'].update(source_sha256=case['source_sha256'], source_size_bytes=size,
            canonical_png_sha256=case['prefix_sha256'], canonical_size_bytes=case['prefix_size_bytes'],
            source_png_conformant=False, integrity_scope=INTEGRITY_SCOPE)
        repair = {**case,
            'canonical_png': {'path': canonical.name, 'sha256': case['prefix_sha256'],
                'size_bytes': case['prefix_size_bytes'], 'size_px': [metadata['width_px'], metadata['height_px']]},
            'excluded_trailer': {'path': excluded.name, 'sha256': case['trailer_sha256'],
                'size_bytes': case['trailer_size_bytes']},
            'rgba_preparation_report': {'path': 'rgba-prepared/preparation.json',
                'sha256': _file_sha(child_dir/'preparation.json')},
            'original_png_conformant': False, 'canonical_png_integrity_validated': True,
            'excluded_bytes_preserved': True, 'source_to_canonical': deepcopy(IDENTITY)}
        report.update(source=source.name, source_sha256=case['source_sha256'],
            source_png_conformant=False, source_integrity_validated=True,
            source_integrity_scope=INTEGRITY_SCOPE, container_repair=repair,
            source_modified=False, elapsed_seconds=round(time.perf_counter()-started, 4),
            coverage='all RGB samples of the pinned canonical PNG prefix; original trailer archived verbatim')
        report['limitations'] = [
            'The original file is NOT a conformant PNG: bytes follow its complete IEND.',
            'PNG structural integrity validation applies only to the explicitly pinned canonical prefix.',
            'Excluded trailer bytes are archived verbatim, not silently deleted or interpreted as image content.',
            *child['limitations']]
        _write_json(output/'preparation.json', report)
        return report
    except Exception as error:
        if not (output/'preparation.json').exists():
            _write_json(output/'preparation.json', {'schema_version': '1.0', 'status': 'failed',
                'source': source.name, 'source_sha256': case['source_sha256'],
                'source_png_conformant': False, 'source_integrity_validated': False,
                'source_modified': False, 'error': {'type': type(error).__name__, 'message': str(error)},
                'limitations': ['Repair partition and child preparation are provisional; no successful wrapper exists.']})
        raise
