"""Read exact original-resolution regions from validated preparation tiles.

The original large PNG is never opened. Integrity checks cover the preparation
metadata and tiles intersecting the requested region; the original source hash
is inherited provenance, not a fresh verification of that source file.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re

import numpy as np
from PIL import Image

MAX_REPORT_BYTES = 8 * 1024**2
MAX_TILE_BYTES = 32 * 1024**2
MAX_TILES = 2048
MAX_TILE_SIDE = 2048
MAX_REGION_PIXELS = 16_777_216
MAX_WORKING_BYTES = 256 * 1024**2


def _integer(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return value


def _sha(value, name):
    if not isinstance(value, str) or re.fullmatch(r'[0-9a-f]{64}', value) is None:
        raise ValueError(f'{name} must be a lowercase SHA-256 digest')
    return value


def _contained(root, name):
    if not isinstance(name, str) or not name or '\\' in name or ':' in name:
        raise ValueError('Tile path must be a portable relative path')
    parts = PurePosixPath(name)
    if parts.is_absolute() or '..' in parts.parts:
        raise ValueError('Tile path escapes preparation directory')
    result = (root / name).resolve()
    if result == root or not result.is_relative_to(root):
        raise ValueError('Tile path escapes preparation directory')
    return result


def _bounded_bytes(path, limit, name):
    if not path.is_file() or not 0 < path.stat().st_size <= limit:
        raise ValueError(f'{name} is missing, empty or exceeds its byte limit')
    with path.open('rb') as stream:
        data = stream.read(limit + 1)
    if not data or len(data) > limit:
        raise ValueError(f'{name} exceeds its byte limit while reading')
    return data


def _box(value, width, height, name):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f'{name} must contain four integer bounds')
    x0, y0, x1, y1 = [_integer(v, name, 0, max(width, height)) for v in value]
    if not 0 <= x0 < x1 <= width or not 0 <= y0 < y1 <= height:
        raise ValueError(f'{name} must be nonempty and within source bounds')
    return x0, y0, x1, y1


def _translation(x, y):
    return [[1., 0., x], [0., 1., y], [0., 0., 1.]]


def _check_matrix(value, expected):
    try:
        matrix = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError('Invalid tile coordinate transform') from error
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or not np.array_equal(matrix, expected):
        raise ValueError('Tile coordinate transform disagrees with source bounds')


def read_prepared_region(preparation_dir: Path, box_xyxy: list[int], *,
                         max_pixels: int = 4_194_304) -> tuple[np.ndarray, dict]:
    """Return exact RGB samples and provenance for a half-open source-pixel box.

    Only intersecting tile files are read. All relevant file hashes, PNG CRCs,
    dimensions, translations and shared pixels are checked before returning.
    A gap or contradiction causes rejection, never a partly filled result.
    ``max_pixels`` can tighten or raise the default up to the hard 16-Mpixel
    limit; an additional conservative working-buffer estimate is enforced.
    """
    max_pixels = _integer(max_pixels, 'max_pixels', 1, MAX_REGION_PIXELS)
    root = Path(preparation_dir).resolve()
    raw_report = _bounded_bytes(_contained(root, 'preparation.json'), MAX_REPORT_BYTES, 'Preparation report')
    report = json.loads(raw_report.decode('utf-8'))
    if (not isinstance(report, dict) or report.get('schema_version') != '1.0'
            or report.get('status') != 'prepared' or report.get('source_integrity_validated') is not True
            or report.get('source_modified') is not False):
        raise ValueError('Preparation must be complete and source integrity validated')
    source_sha = _sha(report.get('source_sha256'), 'Source hash')
    size = report.get('source_size_px')
    if not isinstance(size, list) or len(size) != 2:
        raise ValueError('Invalid source dimensions')
    width, height = [_integer(v, 'source dimension', 1, 32768) for v in size]
    if width * height > 300_000_000:
        raise ValueError('Source dimensions exceed preparation limits')
    x0, y0, x1, y1 = _box(box_xyxy, width, height, 'Region')
    pixels_count = (x1-x0)*(y1-y0)
    if pixels_count > max_pixels:
        raise ValueError('Region exceeds max_pixels budget')
    tiles = report.get('tiles')
    if not isinstance(tiles, list) or not 1 <= len(tiles) <= MAX_TILES:
        raise ValueError('Invalid preparation tile inventory')
    count = _integer(report.get('tile_count'), 'tile_count', 1, MAX_TILES)
    if count != len(tiles):
        raise ValueError('Preparation tile count disagrees with inventory')
    selected, paths = [], set()
    for tile in tiles:
        if not isinstance(tile, dict):
            raise ValueError('Invalid tile record')
        tx0, ty0, tx1, ty1 = _box(tile.get('source_box_px'), width, height, 'Tile')
        tw, th = tx1-tx0, ty1-ty0
        if max(tw, th) > MAX_TILE_SIDE or tile.get('size_px') != [tw, th]:
            raise ValueError('Tile dimensions disagree with source bounds or exceed limit')
        if tile.get('pixel_encoding') != 'RGB8' or tile.get('source_samples_preserved') is not True:
            raise ValueError('Tile must preserve original RGB8 samples')
        _check_matrix(tile.get('tile_to_source'), _translation(tx0, ty0))
        _check_matrix(tile.get('source_to_tile'), _translation(-tx0, -ty0))
        path = _contained(root, tile.get('path'))
        if path in paths or path.suffix.casefold() != '.png':
            raise ValueError('Duplicate or non-PNG tile path')
        paths.add(path)
        _sha(tile.get('sha256'), 'Tile hash')
        _integer(tile.get('size_bytes'), 'tile size_bytes', 1, MAX_TILE_BYTES)
        ix0, iy0, ix1, iy1 = max(x0, tx0), max(y0, ty0), min(x1, tx1), min(y1, ty1)
        if ix0 < ix1 and iy0 < iy1:
            selected.append((tile, path, (ix0, iy0, ix1, iy1)))
    if not selected:
        raise ValueError('No tiles cover requested region')
    largest_tile = max(t['size_px'][0]*t['size_px'][1] for t, _, _ in selected)
    largest_file = max(t['size_bytes'] for t, _, _ in selected)
    estimate = pixels_count*10 + largest_tile*12 + largest_file*2 + len(raw_report)*2
    if estimate > MAX_WORKING_BYTES:
        raise ValueError('Region exceeds the working-buffer budget')
    result = np.empty((y1-y0, x1-x0, 3), dtype=np.uint8)
    covered = np.zeros((y1-y0, x1-x0), dtype=bool)
    used, overlap_comparisons = [], 0
    for tile, path, (ix0, iy0, ix1, iy1) in selected:
        raw = _bounded_bytes(path, MAX_TILE_BYTES, 'Tile PNG')
        if len(raw) != tile['size_bytes'] or hashlib.sha256(raw).hexdigest() != tile['sha256']:
            raise ValueError('Tile PNG hash or byte size differs from preparation')
        with Image.open(io.BytesIO(raw)) as picture:
            if (picture.format != 'PNG' or picture.mode != 'RGB' or list(picture.size) != tile['size_px']
                    or getattr(picture, 'n_frames', 1) != 1):
                raise ValueError('Tile PNG format or dimensions disagree with preparation')
            picture.verify()
        with Image.open(io.BytesIO(raw)) as picture:
            tile_pixels = np.array(picture)
        tx0, ty0, _, _ = tile['source_box_px']
        samples = tile_pixels[iy0-ty0:iy1-ty0, ix0-tx0:ix1-tx0]
        area = result[iy0-y0:iy1-y0, ix0-x0:ix1-x0]
        seen = covered[iy0-y0:iy1-y0, ix0-x0:ix1-x0]
        overlap_comparisons += int(seen.sum())
        if seen.any() and not np.array_equal(area[seen], samples[seen]):
            raise ValueError('Overlapping tiles contain contradictory RGB samples')
        area[:] = samples
        seen[:] = True
        used.append({'path': tile['path'], 'sha256': tile['sha256'], 'source_box_px': tile['source_box_px'],
                     'size_px': tile['size_px'], 'intersection_source_box_px': [ix0, iy0, ix1, iy1]})
        # Do not retain a previous full tile through its sliced samples while
        # decoding the next one. Only the requested result region accumulates.
        del raw, tile_pixels, samples, area, seen
    if not covered.all():
        raise ValueError('Tile coverage has gaps inside requested region')
    provenance = {
        'schema_version': '1.0', 'source_sha256': source_sha, 'source_size_px': size,
        'source_box_px': [x0, y0, x1, y1], 'region_size_px': [x1-x0, y1-y0],
        'preparation_report_sha256': hashlib.sha256(raw_report).hexdigest(),
        'used_tiles': used, 'used_tile_count': len(used), 'overlap_pixel_comparisons': overlap_comparisons,
        'region_to_source': _translation(x0, y0), 'source_to_region': _translation(-x0, -y0),
        'coordinate_system': 'pixel centers; top-left origin; X right, Y down; half-open source box',
        'samples_unchanged_from_used_tiles': True, 'complete_region_coverage': True,
        'source_original_rehashed': False, 'physical_scale_confirmed': False,
        'resource_estimates': {'working_buffer_bytes': estimate, 'total_process_memory_measured': False},
        'limits': {'max_pixels': max_pixels, 'hard_max_region_pixels': MAX_REGION_PIXELS,
                   'max_working_buffer_bytes': MAX_WORKING_BYTES},
        'integrity_scope': 'Preparation metadata and intersecting tile files only; original PNG is not opened.',
    }
    return result, provenance
