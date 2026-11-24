"""Independently validate a complete preparation batch with bounded image memory.

All original hashes, produced PNGs, coordinate grids and shared tile samples are
checked. Large RGB originals are not decoded again. Opaque RGBA originals are
small and are compared pixel for pixel with their derivative and every tile.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import io
import math
from pathlib import Path
import struct
import zlib

import numpy as np
from PIL import Image

from .clear_pairs import contained, digest, read_json

MAX_ITEMS = 10_000
MAX_TILES = 2048
MAX_TILE_SIDE = 2048
MAX_OVERVIEW_SIDE = 4096
MAX_RGBA_PIXELS = 8_000_000
MAX_PNG_BYTES = 50 * 1024**2
REPAIR_KEYS = {'method', 'source_sha256', 'source_size_bytes', 'prefix_size_bytes',
               'prefix_sha256', 'trailer_size_bytes', 'trailer_sha256'}
REPAIR_METHOD = 'retain_pinned_complete_png_prefix'
REPAIR_SCOPE = 'original_bytes_pinned_and_partitioned; PNG_structure_validation_applies_to_canonical_prefix'


def _integer(value, low, high, name):
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f'Invalid {name}')
    return value


def _size(value, max_side=32768, max_pixels=300_000_000):
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError('Invalid image dimensions')
    width, height = [_integer(n, 1, max_side, 'image dimension') for n in value]
    if width*height > max_pixels:
        raise ValueError('Image dimensions exceed validation budget')
    return width, height


def _sha(path, expected):
    if not isinstance(expected, str) or len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
        raise ValueError('Invalid SHA-256 pin')
    if digest(path) != expected:
        raise ValueError(f'Artifact SHA-256 mismatch: {path.name}')


def _matrix(actual, expected):
    try:
        array = np.asarray(actual, dtype=float)
    except (ValueError, TypeError) as error:
        raise ValueError('Invalid coordinate transform') from error
    if array.shape != (3, 3) or not np.isfinite(array).all() or not np.array_equal(array, expected):
        raise ValueError('Coordinate transform disagrees with pixel bounds')


def _translation(x, y):
    return [[1., 0., x], [0., 1., y], [0., 0., 1.]]


def _index(items):
    if not isinstance(items, list) or not 1 <= len(items) <= MAX_ITEMS:
        raise ValueError('Invalid preparation inventory')
    result = {}
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get('input_path'), str) or item['input_path'] in result:
            raise ValueError('Duplicate or invalid preparation input')
        result[item['input_path']] = item
    return result


def _header(path):
    with path.open('rb') as stream:
        header = stream.read(33)
    if len(header) != 33 or header[:16] != b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR':
        raise ValueError('Invalid original PNG header')
    if zlib.crc32(header[12:29]) & 0xFFFFFFFF != struct.unpack('>I', header[29:33])[0]:
        raise ValueError('Original PNG IHDR CRC mismatch')
    return struct.unpack('>IIBBBBB', header[16:29])


def _static_chunks(raw):
    position, count, seen_data, ended_data = 8, 0, False, False
    while position < len(raw):
        count += 1
        if count > 10000 or position+12 > len(raw):
            raise ValueError('PNG chunk inventory exceeds bounds')
        length = struct.unpack('>I', raw[position:position+4])[0]
        kind = raw[position+4:position+8]
        end = position+12+length
        if end > len(raw) or zlib.crc32(raw[position+4:end-4]) & 0xFFFFFFFF != struct.unpack('>I', raw[end-4:end])[0]:
            raise ValueError('PNG chunk length or CRC mismatch')
        if (count == 1 and (kind != b'IHDR' or length != 13)) or (count > 1 and kind == b'IHDR'):
            raise ValueError('PNG has an invalid or repeated IHDR')
        if kind in (b'acTL', b'fcTL', b'fdAT', b'eXIf', b'tRNS'):
            raise ValueError('Animation, EXIF orientation or transparency metadata is unsupported')
        if kind not in (b'IHDR', b'PLTE', b'IDAT', b'IEND') and kind[:1].isupper():
            raise ValueError('Unknown critical PNG chunk')
        if kind == b'IDAT':
            if ended_data:
                raise ValueError('PNG IDAT chunks must be consecutive')
            seen_data = True
        elif seen_data:
            ended_data = True
        if kind == b'IEND':
            if length != 0 or not seen_data or end != len(raw):
                raise ValueError('PNG IEND has invalid size, missing data or trailing bytes')
            return
        position = end
    raise ValueError('PNG lacks a final IEND')


def _read_png(path, record, mode, *, max_side=MAX_TILE_SIDE, max_pixels=MAX_TILE_SIDE**2):
    width, height = _size(record['size_px'], max_side, max_pixels)
    expected_size = _integer(record['size_bytes'], 1, MAX_PNG_BYTES, 'PNG byte count')
    if path.stat().st_size != expected_size:
        raise ValueError('PNG byte count differs from preparation')
    with path.open('rb') as stream:
        raw = stream.read(MAX_PNG_BYTES+1)
    if len(raw) != expected_size or hashlib.sha256(raw).hexdigest() != record['sha256']:
        raise ValueError('PNG hash differs from preparation')
    if (raw[:16] != b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR'
            or len(raw) < 45 or raw[-12:] != b'\x00\x00\x00\x00IEND\xaeB`\x82'):
        raise ValueError('PNG structure has an invalid header or trailing data')
    header = struct.unpack('>IIBBBBB', raw[16:29])
    if header != (width, height, 8, {'L': 0, 'RGB': 2, 'RGBA': 6}[mode], 0, 0, 0):
        raise ValueError('PNG encoding is not the declared static noninterlaced 8-bit mode')
    _static_chunks(raw)
    # Pillow verify validates PNG chunk CRCs; reopening decodes the bounded image.
    with Image.open(io.BytesIO(raw)) as picture:
        if (picture.format != 'PNG' or picture.mode != mode or picture.size != (width, height)
                or getattr(picture, 'n_frames', 1) != 1):
            raise ValueError('PNG format, dimensions or frame count disagree')
        picture.verify()
    with Image.open(io.BytesIO(raw)) as picture:
        pixels = np.array(picture)
    return pixels


def _repair_cases(batch):
    records = batch.get('container_repair_cases', [])
    if not isinstance(records, list) or len(records) > MAX_ITEMS:
        raise ValueError('Invalid explicit container repair inventory')
    result = {}
    for record in records:
        if not isinstance(record, dict) or set(record) != REPAIR_KEYS or record['method'] != REPAIR_METHOD:
            raise ValueError('Invalid explicit container repair case')
        for name in ('source_sha256', 'prefix_sha256', 'trailer_sha256'):
            value = record[name]
            if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('Invalid container repair hash')
        source_size = _integer(record['source_size_bytes'], 1, MAX_PNG_BYTES, 'repair source bytes')
        prefix_size = _integer(record['prefix_size_bytes'], 45, MAX_PNG_BYTES, 'repair prefix bytes')
        trailer_size = _integer(record['trailer_size_bytes'], 1, MAX_PNG_BYTES, 'repair trailer bytes')
        if prefix_size+trailer_size != source_size or record['source_sha256'] in result:
            raise ValueError('Duplicate or inconsistent container repair case')
        result[record['source_sha256']] = record
    return result


def _container_repair(source, item, report, root, allowed):
    repair = report.get('container_repair')
    if not isinstance(repair, dict) or not allowed or {k: repair.get(k) for k in REPAIR_KEYS} != allowed:
        raise ValueError('Container repair is unlisted or differs from its explicit batch case')
    if (report.get('source_png_conformant') is not False or report.get('source_integrity_scope') != REPAIR_SCOPE
            or repair.get('original_png_conformant') is not False
            or repair.get('canonical_png_integrity_validated') is not True
            or repair.get('excluded_bytes_preserved') is not True):
        raise ValueError('Repair must distinguish original bytes from canonical PNG integrity')
    _matrix(repair.get('source_to_canonical'), np.eye(3))
    if item['input_size_bytes'] != allowed['source_size_bytes']:
        raise ValueError('Repair byte partition differs from registered original')
    with source.open('rb') as stream:
        original = stream.read(MAX_PNG_BYTES+1)
    if len(original) != allowed['source_size_bytes'] or hashlib.sha256(original).hexdigest() != allowed['source_sha256']:
        raise ValueError('Repair original bytes changed')
    canonical, trailer = repair['canonical_png'], repair['excluded_trailer']
    if (canonical['sha256'] != allowed['prefix_sha256'] or canonical['size_bytes'] != allowed['prefix_size_bytes']
            or canonical['size_px'] != report['source_size_px']
            or trailer['sha256'] != allowed['trailer_sha256'] or trailer['size_bytes'] != allowed['trailer_size_bytes']):
        raise ValueError('Repair artifacts differ from explicit prefix/trailer pins')
    canonical_path, trailer_path = contained(root, canonical['path']), contained(root, trailer['path'])
    if canonical_path == trailer_path or source in (canonical_path, trailer_path):
        raise ValueError('Repair artifacts must be distinct from the original and each other')
    for path, record, expected in [(canonical_path, canonical, original[:allowed['prefix_size_bytes']]),
                                    (trailer_path, trailer, original[allowed['prefix_size_bytes']:])]:
        if path.stat().st_size != record['size_bytes']:
            raise ValueError('Repair artifact size changed')
        with path.open('rb') as stream:
            raw = stream.read(MAX_PNG_BYTES+1)
        if raw != expected or hashlib.sha256(raw).hexdigest() != record['sha256']:
            raise ValueError('Repair original is not the pinned canonical prefix plus preserved trailer')
    child_record = repair['rgba_preparation_report']
    child_path = contained(root, child_record['path'])
    _sha(child_path, child_record['sha256'])
    child = read_json(child_path)
    if (child.get('status') != 'prepared' or child.get('source_integrity_validated') is not True
            or child.get('source_modified') is not False or child.get('source_sha256') != canonical['sha256']
            or 'container_repair' in child):
        raise ValueError('Repair child must validate only the canonical PNG')
    for key in ('source_size_px', 'tile_count', 'parameters', 'overview_padding_px',
                'coordinate_system', 'source_box_convention'):
        if report.get(key) != child.get(key):
            raise ValueError('Repair wrapper metadata differs from canonical preparation')
    if len(child['tiles']) != len(report['tiles']):
        raise ValueError('Repair wrapper lost canonical preparation tiles')
    for native, wrapped in zip([child['overview'], child['ink_overview'], *child['tiles']],
                               [report['overview'], report['ink_overview'], *report['tiles']]):
        if (contained(child_path.parent, native['path']) != contained(root, wrapped['path'])
                or {k: v for k, v in native.items() if k != 'path'} != {k: v for k, v in wrapped.items() if k != 'path'}):
            raise ValueError('Repair wrapper artifact differs from canonical preparation')
    conversion, original_conversion = report['rgba_conversion'], child['rgba_conversion']
    if (conversion.get('source_sha256') != allowed['source_sha256']
            or conversion.get('decoded_source_sha256') != canonical['sha256']):
        raise ValueError('Repair RGB conversion source identity differs')
    for key in ('method', 'rgb_samples_preserved', 'alpha_extrema', 'source_to_derivative'):
        if conversion.get(key) != original_conversion.get(key):
            raise ValueError('Repair RGB conversion differs from canonical preparation')
    for key in ('rgb_derivative', 'rgb_preparation_report'):
        native, wrapped = original_conversion[key], conversion[key]
        if (contained(child_path.parent, native['path']) != contained(root, wrapped['path'])
                or {k: v for k, v in native.items() if k != 'path'} != {k: v for k, v in wrapped.items() if k != 'path'}):
            raise ValueError('Repair conversion artifact differs from canonical preparation')
    expected_decoder = dict(child['decoder'])
    expected_decoder.update(source_sha256=allowed['source_sha256'], source_size_bytes=allowed['source_size_bytes'],
                            canonical_png_sha256=canonical['sha256'], canonical_size_bytes=canonical['size_bytes'],
                            source_png_conformant=False, integrity_scope=REPAIR_SCOPE)
    if report['decoder'] != expected_decoder:
        raise ValueError('Repair decoder scope differs from canonical preparation')
    return canonical_path, canonical, {'source_sha256': allowed['source_sha256'],
        'canonical_png_sha256': canonical['sha256'], 'excluded_trailer_sha256': trailer['sha256'],
        'excluded_trailer_size_bytes': trailer['size_bytes'], 'original_png_conformant': False,
        'original_bytes_equal_canonical_plus_preserved_trailer': True}


def _rgba_pixels(source, source_record, report, prep_root, *, canonical=None):
    conversion = report.get('rgba_conversion')
    if (not isinstance(conversion, dict) or conversion.get('source_sha256') != report['source_sha256']
            or conversion.get('alpha_extrema') != [255, 255]
            or conversion.get('rgb_samples_preserved') is not True
            or conversion.get('method') != 'discard_proven_opaque_alpha_only'):
        raise ValueError('Missing or invalid opaque RGBA provenance')
    _matrix(conversion.get('source_to_derivative'), np.eye(3))
    if canonical is None and 'decoded_source_sha256' in conversion:
        raise ValueError('Alternate decoded source requires an explicit container repair')
    decoded_path, decoded_record = canonical if canonical is not None else (source, source_record)
    pixels = _read_png(decoded_path, decoded_record, 'RGBA', max_side=32768, max_pixels=MAX_RGBA_PIXELS)
    if not np.all(pixels[:, :, 3] == 255):
        raise ValueError('RGBA original contains nonopaque pixels')
    derivative = conversion['rgb_derivative']
    if derivative['size_px'] != report['source_size_px']:
        raise ValueError('RGB derivative dimensions differ from original')
    with_rgb = _read_png(contained(prep_root, derivative['path']), derivative, 'RGB',
                         max_side=32768, max_pixels=MAX_RGBA_PIXELS)
    if not np.array_equal(pixels[:, :, :3], with_rgb):
        raise ValueError('RGB derivative changed original RGB samples')
    del with_rgb
    base = conversion['rgb_preparation_report']
    base_path = contained(prep_root, base['path'])
    _sha(base_path, base['sha256'])
    native = read_json(base_path)
    if (native.get('status') != 'prepared' or native.get('source_integrity_validated') is not True
            or native['source_sha256'] != derivative['sha256']
            or native['source_size_px'] != report['source_size_px']):
        raise ValueError('Derived RGB preparation has incorrect provenance')
    if len(native['tiles']) != len(report['tiles']):
        raise ValueError('RGBA wrapper lost RGB preparation tiles')
    for original, wrapped in zip([native['overview'], native['ink_overview'], *native['tiles']],
                                 [report['overview'], report['ink_overview'], *report['tiles']]):
        expected_path = contained(base_path.parent, original['path'])
        if contained(prep_root, wrapped['path']) != expected_path or {k: v for k, v in wrapped.items() if k != 'path'} != {k: v for k, v in original.items() if k != 'path'}:
            raise ValueError('RGBA wrapper artifact differs from pinned RGB preparation')
    return pixels[:, :, :3], {'derivative_sha256': derivative['sha256'], 'rgb_preparation_report_sha256': base['sha256']}


def _validate_page(source, item, batch_item, prep_path, *, repair_case=None):
    report = read_json(prep_path)
    if (report.get('schema_version') != '1.0' or report.get('status') != 'prepared'
            or report.get('source_integrity_validated') is not True or report.get('source_modified') is not False
            or report.get('source_sha256') != item['input_sha256']):
        raise ValueError('Preparation source identity or completeness differs')
    width, height = _size(report['source_size_px'])
    if batch_item.get('source_size_px') != [width, height]:
        raise ValueError('Batch source dimensions differ from page preparation')
    w, h, depth, color, compression, filtering, interlace = _header(source)
    if (w, h) != (width, height) or (depth, compression, filtering, interlace) != (8, 0, 0, 0) or color not in (2, 6):
        raise ValueError('Original PNG format or dimensions do not match supported preparation')
    root = prep_path.parent
    rgba, rgba_provenance, repair_provenance, canonical = None, None, None, None
    if 'container_repair' in report:
        if color != 6:
            raise ValueError('Explicit container repair is supported only for opaque RGBA')
        path, record, repair_provenance = _container_repair(source, item, report, root, repair_case)
        canonical = (path, record)
    elif report.get('source_png_conformant') is False or 'source_integrity_scope' in report:
        raise ValueError('Nonconformant original requires an explicit container repair')
    if color == 6:
        rgba, rgba_provenance = _rgba_pixels(source,
            {'size_px': [width, height], 'size_bytes': item['input_size_bytes'], 'sha256': item['input_sha256']}, report, root, canonical=canonical)
    elif 'rgba_conversion' in report:
        raise ValueError('RGB source cannot claim an RGBA conversion')
    params = report['parameters']
    tile_size = _integer(params['tile_size'], 128, MAX_TILE_SIDE, 'tile size')
    overlap = _integer(params['overlap'], 0, tile_size//2, 'overlap')
    overview_max = _integer(params['overview_max_side'], 128, MAX_OVERVIEW_SIDE, 'overview side')
    stride = tile_size-overlap
    xs = list(range(0, max(1, width-overlap), stride))
    ys = list(range(0, max(1, height-overlap), stride))
    tiles = report['tiles']
    if (not isinstance(tiles, list) or not 1 <= len(tiles) <= MAX_TILES
            or len(tiles) != len(xs)*len(ys) or report['tile_count'] != len(tiles)
            or batch_item.get('tile_count') != len(tiles)):
        raise ValueError('Tile count does not cover the declared preparation grid')
    paths, by_origin = set(), {}
    for tile in tiles:
        box = tile.get('source_box_px')
        if (not isinstance(box, list) or len(box) != 4 or any(isinstance(v, bool) or not isinstance(v, int) for v in box)):
            raise ValueError('Invalid integer tile bounds')
        x0, y0, x1, y1 = box
        if x0 not in xs or y0 not in ys or (x1, y1) != (min(x0+tile_size, width), min(y0+tile_size, height)):
            raise ValueError('Tile bounds disagree with complete regular grid')
        if (x0, y0) in by_origin or tile['size_px'] != [x1-x0, y1-y0]:
            raise ValueError('Repeated tile origin or incorrect tile dimensions')
        path = contained(root, tile['path'])
        if path in paths:
            raise ValueError('Repeated artifact path')
        paths.add(path)
        if tile.get('pixel_encoding') != 'RGB8' or tile.get('source_samples_preserved') is not True:
            raise ValueError('Tile must declare unchanged RGB8 samples')
        _matrix(tile['tile_to_source'], _translation(x0, y0))
        _matrix(tile['source_to_tile'], _translation(-x0, -y0))
        by_origin[x0, y0] = tile
    compared = 0
    for row, y in enumerate(ys):
        for column, x in enumerate(xs):
            tile = by_origin[x, y]
            pixels = _read_png(contained(root, tile['path']), tile, 'RGB')
            x0, y0, x1, y1 = tile['source_box_px']
            if rgba is not None and not np.array_equal(pixels, rgba[y0:y1, x0:x1]):
                raise ValueError('Prepared tile differs from original RGBA RGB samples')
            # Neighbouring strips cover every shared sample, including four-way
            # corners. Two decoded tiles are held at a time, never a whole page.
            for neighbour_origin in ([(xs[column-1], y)] if column else []) + ([(x, ys[row-1])] if row else []):
                neighbour = by_origin[neighbour_origin]
                a, b, c, d = neighbour['source_box_px']
                ix0, iy0, ix1, iy1 = max(a, x0), max(b, y0), min(c, x1), min(d, y1)
                if ix0 < ix1 and iy0 < iy1:
                    previous = _read_png(contained(root, neighbour['path']), neighbour, 'RGB')
                    if not np.array_equal(pixels[iy0-y0:iy1-y0, ix0-x0:ix1-x0], previous[iy0-b:iy1-b, ix0-a:ix1-a]):
                        raise ValueError('Overlapping tiles have contradictory RGB pixels')
                    compared += (ix1-ix0)*(iy1-iy0)
                    del previous
            del pixels
    factor = max(1, math.ceil(max(width, height)/overview_max))
    size = [math.ceil(width/factor), math.ceil(height/factor)]
    for name, mode, method in [('overview', 'RGB', 'integer_area_mean_RGB_white_padding_round_half_up'),
                               ('ink_overview', 'L', 'minimum_of_RGB_channels_and_block_samples')]:
        overview = report[name]
        path = contained(root, overview['path'])
        if path in paths:
            raise ValueError('Repeated artifact path')
        paths.add(path)
        if overview['scale_factor'] != factor or overview['size_px'] != size or overview.get('method') != method:
            raise ValueError('Overview dimensions, scale or sampling method differ')
        _matrix(overview['source_to_overview'], [[1/factor, 0, (1/factor-1)/2], [0, 1/factor, (1/factor-1)/2], [0, 0, 1]])
        _matrix(overview['overview_to_source'], [[factor, 0, (factor-1)/2], [0, factor, (factor-1)/2], [0, 0, 1]])
        image = _read_png(path, overview, mode, max_side=MAX_OVERVIEW_SIDE, max_pixels=MAX_OVERVIEW_SIDE**2)
        del image
    if report['overview_padding_px'] != {'right': size[0]*factor-width, 'bottom': size[1]*factor-height, 'value': [255, 255, 255]}:
        raise ValueError('Overview padding differs from declared grid')
    return {'input_path': item['input_path'], 'source_sha256': item['input_sha256'],
            'preparation_report_sha256': batch_item['preparation_report_sha256'],
            'tile_count': len(tiles), 'image_artifacts_verified': len(tiles)+2,
            'source_pixel_count': width*height, 'overlap_rgb_pixels_compared': compared,
            'source_encoding': 'RGBA8_opaque' if rgba is not None else 'RGB8',
            'rgba_provenance': rgba_provenance, 'container_repair_provenance': repair_provenance,
            'source_rgb_pixels_independently_compared': width*height if rgba is not None else 0}


def validate_prepared_batch(dataset: Path, intake_path: Path, prepared_dir: Path) -> dict:
    """Rehash and validate every declared source and bounded prepared artifact."""
    dataset, intake_path, prepared_dir = Path(dataset).resolve(), Path(intake_path).resolve(), Path(prepared_dir).resolve()
    intake, batch = read_json(intake_path), read_json(prepared_dir/'report.json')
    if intake.get('schema_version') != '1.0' or intake.get('purpose') != 'training_intake_not_split':
        raise ValueError('Expected registered training intake')
    if (batch.get('schema_version') != '1.0' or batch.get('status') != 'complete'
            or batch.get('purpose') != 'raster_preparation_only' or batch.get('intake_sha256') != digest(intake_path)):
        raise ValueError('Batch must be complete and pinned to this intake')
    inputs, records = _index(intake['items']), _index(batch['items'])
    if set(inputs) != set(records):
        raise ValueError('Prepared input inventory differs from intake')
    counts = batch['counts']
    if any(counts.get(key) != value for key, value in {'intake_total': len(inputs), 'selected': len(inputs), 'prepared': len(inputs), 'failed': 0, 'not_selected': 0}.items()):
        raise ValueError('Batch counts do not describe a complete preparation')
    pages, prep_paths, used_repairs = [], set(), set()
    repair_cases = _repair_cases(batch)
    for name, item in sorted(inputs.items()):
        entry = records[name]
        if entry.get('status') != 'prepared' or entry.get('input_sha256') != item['input_sha256']:
            raise ValueError('Batch item differs from registered source')
        source = contained(dataset, name)
        if source.suffix.lower() != '.png' or source.stat().st_size != item['input_size_bytes']:
            raise ValueError('Original source format or size changed')
        _sha(source, item['input_sha256'])
        prep = contained(prepared_dir, entry['preparation_report'])
        if prep in prep_paths:
            raise ValueError('Several inputs share one preparation report')
        prep_paths.add(prep)
        if prep != contained(prepared_dir, entry['output_directory'])/'preparation.json':
            raise ValueError('Preparation path disagrees with output directory')
        _sha(prep, entry['preparation_report_sha256'])
        page = _validate_page(source, item, entry, prep, repair_case=repair_cases.get(item['input_sha256']))
        pages.append(page)
        if page['container_repair_provenance'] is not None:
            used_repairs.add(item['input_sha256'])
    if used_repairs != set(repair_cases):
        raise ValueError('Explicit container repair inventory contains unused cases')
    return {'schema_version': '1.0', 'status': 'verified', 'intake_sha256': digest(intake_path),
            'batch_report_sha256': digest(prepared_dir/'report.json'),
            'summary': {'source_count': len(pages), 'tile_count': sum(p['tile_count'] for p in pages),
                        'overview_count': len(pages)*2, 'source_files_rehashed': len(pages),
                        'preparation_reports_rehashed': len(pages),
                        'image_artifacts_verified': sum(p['image_artifacts_verified'] for p in pages),
                        'source_encodings': dict(Counter(p['source_encoding'] for p in pages)),
                        'source_pixel_count': sum(p['source_pixel_count'] for p in pages),
                        'container_repaired_sources': sum(p['container_repair_provenance'] is not None for p in pages),
                        'excluded_trailer_bytes_preserved': sum(p['container_repair_provenance']['excluded_trailer_size_bytes'] for p in pages if p['container_repair_provenance'] is not None),
                        'overlap_rgb_pixels_compared': sum(p['overlap_rgb_pixels_compared'] for p in pages),
                        'original_rgb_pixels_independently_compared': sum(p['source_rgb_pixels_independently_compared'] for p in pages)},
            'pages': pages, 'code_sha256': digest(Path(__file__)),
            'training_eligible': False, 'annotation_approved': False, 'source_modified': False,
            'scope': ['Every original SHA-256, preparation pin and produced image hash/CRC/mode/size was checked.',
                      'All tile grids, transforms, full coverage and adjacent overlap pixels were checked.',
                      'Large RGB originals were not independently decoded again; original-to-tile pixel equality is not claimed by this validator.',
                      'RGBA alpha and original RGB channels were compared with the derivative and every prepared tile.',
                      'Explicit listed container repairs additionally prove original bytes equal the pinned canonical PNG prefix plus the preserved excluded trailer; original PNG conformance is not asserted for those sources.',
                      'Overview sampling transforms are checked, but overview pixel aggregates are not independently recomputed.',
                      'This check does not verify PNG/DXF alignment, physical scale or semantic labels.']}
