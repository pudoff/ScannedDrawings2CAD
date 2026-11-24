"""Read-only content screening for the customer's explicit dataset layout."""
from collections import Counter, defaultdict
import hashlib
import html
import json
import math
from pathlib import Path
import re
import time
import unicodedata

import cv2
import ezdxf
import numpy as np
import pymupdf
from PIL import Image
from ezdxf.lldxf.tagwriter import TagCollector

from .dataset_layout_v1 import checked, digest, encoded, inside, pin, read, verify_layout_snapshot
from .family_catalog_v2 import require
from .png_stream import PNGStripReader
from .clear_pairs import normalize_raster
from .pairing import compare_renders, THRESHOLDS, _save_overlay
from .dxf_reference_v5 import render_dxf_reference

GEOMETRY = {'LINE', 'CIRCLE', 'ARC', 'LWPOLYLINE', 'ELLIPSE', 'SPLINE', 'SOLID', 'TRACE', '3DFACE', 'POINT'}
PHASES = ('native', 'pdf', 'controls', 'pdf_pairs')
MAX_SIDE = 1200
CODE = ['src/cad_recovery/' + n + '.py' for n in (
    'dataset_content_v1', 'dataset_layout_v1', 'family_catalog_v2', 'png_stream',
    'clear_pairs', 'pairing', 'dxf_reference', 'dxf_reference_v5', 'mtext_zero_height',
    'reference_resources', 'reference_completeness')]+['scripts/review_dataset_content_v1.py']


def stable(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def alias_key(name):
    value = unicodedata.normalize('NFKC', Path(name).stem).casefold().strip()
    if value.endswith('pdf'):
        value = value[:-3]
    return ' '.join(value.replace('_', ' ').split())


def bind_pairs(audit, aliases):
    files = {r['path']: r for r in audit['files']}
    pairs = [dict(r, input=r['input'][0], target=r['target'][0], basis='unique_layout_stem') for r in audit['candidate_pairs']]
    available = {p for r in audit['unmatched'] for key in ('input', 'target') for p in r[key]}
    used = set()
    for record in aliases:
        require(set(record) == {'input', 'target', 'input_sha256', 'target_sha256', 'basis'}, 'Invalid alias record')
        a, b = record['input'], record['target']
        require(a in available and b in available and not {a, b} & used, 'Alias does not bind unique unmatched endpoints')
        source, target = files[a], files[b]
        require(source['role'] == 'input' and target['role'] == 'target'
                and source['bundle'] == target['bundle'] and source['declared_split'] == target['declared_split'], 'Alias crosses bundle or split')
        require(source['sha256'] == record['input_sha256'] and target['sha256'] == record['target_sha256'], 'Alias content changed')
        require(alias_key(a) == alias_key(b), 'Alias requires a separate content decision')
        pairs.append({'bundle': source['bundle'], 'declared_split': source['declared_split'],
                      'input': a, 'target': b, 'basis': record['basis'], 'target_approved': False})
        used.update((a, b))
    require(used == available, 'Incomplete explicit alias inventory')
    require(len({r['input'] for r in pairs}) == len(pairs) == len({r['target'] for r in pairs}), 'Repeated pair endpoint')
    return sorted(pairs, key=lambda r: (r['bundle'], r['input']))


def prepare_config(root, output):
    root = Path(root).resolve()
    manifest = pin(root, 'artifacts/dataset-layout-v1/manifest.json')
    verify_layout_snapshot(root, manifest)
    audit = read(root/'artifacts/dataset-layout-v1/audit.json')
    files = {r['path']: r for r in audit['files']}
    inputs = [r['input'][0] for r in audit['unmatched'] if r['input']]
    targets = [r['target'][0] for r in audit['unmatched'] if r['target']]
    aliases = []
    for a in inputs:
        matches = [b for b in targets if files[b]['bundle'] == files[a]['bundle'] and alias_key(a) == alias_key(b)]
        require(len(matches) == 1, 'Alias is ambiguous')
        b = matches[0]
        aliases.append({'input': a, 'target': b, 'input_sha256': files[a]['sha256'],
                        'target_sha256': files[b]['sha256'], 'basis': 'explicit_unique_spaces_underscores_or_pdf_suffix_candidate'})
    pairs = bind_pairs(audit, aliases)
    config = {'schema_version': 'dataset-content-config/1.0', 'task': 'P08.9b', 'layout': manifest,
              'previous_handoff': pin(root, 'artifacts/pair-review-handoff-v5/report.json'),
              'aliases': aliases, 'pair_count': len(pairs), 'max_side': MAX_SIDE,
              'thresholds': dict(THRESHOLDS), 'code': [pin(root, n) for n in CODE],
              'policy': {'native_resources': 'all native entities; any IMAGE/OLE requires explicit completeness review',
                         'geometry_signature': 'exact selected primitive tags and recursive INSERT placements; ignores text/layers/handles; relationship hint only',
                         'pdf_vector': 'actual drawing paths/text and image placements per page; mixed and raster-only retained',
                         'png': 'fully validated strip decoding, row-area overview, no full source image allocation',
                         'control_comparison': 'full recorded modelspace render; original fixed SIFT thresholds; no target approval',
                         'independence': 'screened exact/specified geometry/text hints cannot prove family independence',
                         'training_eligible': False}}
    target = inside(root, output)
    require(not target.exists(), 'Config already exists')
    target.write_bytes(encoded(config))
    return pin(root, output)


def inputs(root, config_pin):
    config = read(checked(root, config_pin))
    require(config['schema_version'] == 'dataset-content-config/1.0' and config['task'] == 'P08.9b'
            and config['max_side'] == MAX_SIDE and config['thresholds'] == THRESHOLDS, 'Changed content protocol')
    require(config['code'] == [pin(root, n) for n in CODE], 'Content code changed')
    checked(root, config['layout'])
    checked(root, config['previous_handoff'])
    base = inside(root, config['layout']['path']).parent
    manifest = read(base/'manifest.json')
    for item in manifest['files']:
        checked(base, item)
    audit = read(base/'audit.json')
    for row in audit['files']:
        checked(root/'dataset', {'path': row['path'], 'sha256': row['sha256']})
    actual = {p.relative_to(root/'dataset').as_posix() for p in (root/'dataset').rglob('*') if p.is_file()}
    require(actual == {r['path'] for r in audit['files']}, 'Dataset inventory changed')
    pairs = bind_pairs(audit, config['aliases'])
    require(len(pairs) == config['pair_count'], 'Pair count changed')
    return config, audit, pairs


def _value(v):
    if isinstance(v, bytes):
        return v.hex()
    if isinstance(v, (float, np.floating)):
        require(math.isfinite(v), 'Nonfinite geometry')
        return round(float(v), 6)
    if isinstance(v, (int, str)):
        return v
    try:
        return [_value(x) for x in v]
    except TypeError:
        return str(v)


def inspect_native(path):
    document = ezdxf.readfile(path)
    require(len(document.entitydb) <= 200000, 'Native entity limit')
    native = Counter(e.dxftype() for e in document.entitydb.values() if e and e.is_alive)
    texts, resources, visited = set(), [], set()

    def walk(space, ancestors=()):
        require(len(ancestors) <= 16, 'Native INSERT depth limit')
        tokens = []
        for e in space:
            kind = e.dxftype()
            if kind in ('TEXT', 'ATTRIB', 'ATTDEF', 'MTEXT'):
                texts.add(e.plain_text() if kind == 'MTEXT' else e.dxf.get('text', ''))
            if kind in ('IMAGE', 'OLE2FRAME', 'ACAD_PROXY_ENTITY', 'PDFUNDERLAY', 'DWFUNDERLAY', 'DGNUNDERLAY'):
                record = {'type': kind, 'handle': e.dxf.get('handle'), 'block_path': list(ancestors)}
                if kind == 'IMAGE':
                    definition = e.image_def
                    record['declared_filename'] = definition.dxf.get('filename', '') if definition else None
                resources.append(record)
            if kind in GEOMETRY:
                collector = TagCollector(write_handles=False)
                e.export_dxf(collector)
                tags = [(t.code, _value(t.value)) for t in collector.tags
                        if t.code not in (5, 8, 62, 67, 100, 102, 330, 347, 370, 410, 420, 440)
                        and not 320 <= t.code <= 369 and t.code < 1000]
                tokens.append(stable(tags))
            if kind == 'INSERT':
                name = e.dxf.name
                require(name not in ancestors, 'Cyclic INSERT')
                block = e.block()
                require(block is not None, 'Missing INSERT block')
                # Cache geometry only; walk each block once still records its native text/resources.
                if name not in visited:
                    visited.add(name)
                    child = walk(block, (*ancestors, name))
                    cache[name] = child
                child = cache[name]
                if child:
                    placement = {k: _value(e.dxf.get(k, default)) for k, default in
                                 [('insert', (0, 0, 0)), ('xscale', 1), ('yscale', 1), ('zscale', 1),
                                  ('rotation', 0), ('row_count', 1), ('column_count', 1),
                                  ('row_spacing', 0), ('column_spacing', 0), ('extrusion', (0, 0, 1))]}
                    tokens.append(stable({'block_geometry': child, 'placement': placement}))
        return sorted(tokens)

    cache = {}
    geometry = walk(document.modelspace())
    # Inspect paperspace resources/text too, but keep modelspace signature separate.
    for layout in document.layouts:
        if layout.name != 'Model':
            walk(layout)
    images = [e for e in document.entitydb.values() if e and e.is_alive and e.dxftype() == 'IMAGE']
    image_refs = []
    for e in images:
        try:
            image_refs.append(e.image_def.dxf.get('filename', '') if e.image_def else None)
        except Exception:
            image_refs.append(None)
    audit = document.audit()
    return {'status': 'ok', 'version': document.dxfversion, 'units': int(document.units),
            'native_counts': dict(sorted(native.items())), 'modelspace_counts': dict(sorted(Counter(e.dxftype() for e in document.modelspace()).items())),
            'layout_names': sorted(document.layouts.names()), 'native_image_references': image_refs,
            'reached_resources': resources, 'resource_completeness': 'requires_review' if
            any(native[k] for k in ('IMAGE', 'OLE2FRAME', 'ACAD_PROXY_ENTITY', 'PDFUNDERLAY', 'DWFUNDERLAY', 'DGNUNDERLAY')) else 'no_listed_native_resource_dependency',
            'audit_errors': [{'code': int(x.code), 'message': x.message} for x in audit.errors],
            'audit_fixes': [{'code': int(x.code), 'message': x.message} for x in audit.fixes], 'fixes_saved': False,
            'texts': sorted(texts), 'geometry_token_count': len(geometry),
            'geometry_signature': stable(geometry) if len(geometry) >= 10 else None,
            'geometry_signature_is_relationship_hint_only': True}


def inspect_pdf(path):
    pages = []
    with pymupdf.open(path) as doc:
        require(not doc.needs_pass and 0 < len(doc) <= 100, 'Encrypted or excessive PDF')
        for page in doc:
            paths = page.get_drawings()
            text = page.get_text()
            images = page.get_image_info()
            pages.append({'number': page.number + 1, 'size_pt': [page.rect.width, page.rect.height],
                          'drawing_paths': len(paths), 'drawing_items': sum(len(p['items']) for p in paths),
                          'image_placements': len(images), 'text': text,
                          'classification': 'mixed' if paths and images else 'vector' if paths else
                          'raster_with_text' if images and text.strip() else 'raster' if images else
                          'text_only' if text.strip() else 'empty'})
    return {'status': 'ok', 'page_count': len(pages), 'pages': pages,
            'all_pages_have_vector_paths': all(p['drawing_paths'] > 0 for p in pages)}


def stream_overview(path, expected_sha, max_side=MAX_SIDE):
    reader = PNGStripReader(path, expected_sha, strip_rows=256)
    scale = min(1, max_side/max(reader.width, reader.height))
    width, height = max(1, round(reader.width*scale)), max(1, round(reader.height*scale))
    sums = np.zeros((height, width), np.float64)
    counts = np.zeros(height, np.int64)
    for start, pixels in reader.iter_strips():
        gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
        horizontal = cv2.resize(gray, (width, len(gray)), interpolation=cv2.INTER_AREA)
        bins = np.minimum(height-1, np.arange(start, start+len(gray), dtype=np.int64)*height//reader.height)
        np.add.at(sums, bins, horizontal)
        np.add.at(counts, bins, 1)
    require(reader.validated and np.all(counts > 0), 'PNG overview is provisional')
    return np.rint(sums/counts[:, None]).astype(np.uint8), reader.metadata


def compare_pair(root, pair, directory, files, png=True):
    source = root/'dataset'/pair['input']
    target = root/'dataset'/pair['target']
    result = {'input': pair['input'], 'target': pair['target'], 'bundle': pair['bundle'],
              'declared_split': pair['declared_split'], 'status': 'quarantine', 'errors': [],
              'input_sha256': files[pair['input']]['sha256'], 'target_sha256': files[pair['target']]['sha256'],
              'training_eligible': False, 'target_approved': False}
    try:
        if png:
            overview, result['source_metadata'] = stream_overview(source, result['input_sha256'])
        else:
            with pymupdf.open(source) as doc:
                require(len(doc) == 1, 'Multi-page PDF requires explicit page mapping')
                page = doc[0]
                pix = page.get_pixmap(matrix=pymupdf.Matrix(MAX_SIDE/max(page.rect.width, page.rect.height), MAX_SIDE/max(page.rect.width, page.rect.height)), colorspace=pymupdf.csGRAY, alpha=False)
                overview = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy()
        gray, result['normalization'] = normalize_raster(overview, MAX_SIDE)
        Image.fromarray(gray).save(directory/'source.png')
        reference, result['reference_metadata'] = render_dxf_reference(target, MAX_SIDE, allow_incomplete_reference=True, resource_root=root/'dataset')
        Image.fromarray(reference).save(directory/'reference.png')
        comparison = compare_renders(gray, reference)
        result['comparison'] = comparison
        complete = result['reference_metadata']['reference_complete']
        result['status'] = comparison['status'] if complete else 'incomplete_reference'
        _save_overlay(gray, reference, comparison, directory/'overlay.png')
    except Exception as exc:
        result['errors'].append({'type': type(exc).__name__, 'message': str(exc).replace(str(root), '<workspace>')})
    return result


def run_phase(workspace, config_pin, output, phase):
    root = Path(workspace).resolve()
    require(phase in PHASES, 'Unknown phase')
    config, audit, pairs = inputs(root, config_pin)
    base = inside(root, output)
    require(base.parent == root/'artifacts', 'Output must be a direct artifacts directory')
    directory = base/phase
    require(not directory.exists(), 'Phase already exists; select a new output')
    directory.mkdir(parents=True)
    files = {r['path']: r for r in audit['files']}
    if phase in ('native', 'pdf'):
        selected = [r for r in audit['files'] if Path(r['path']).suffix.lower() == ('.dxf' if phase == 'native' else '.pdf') and r['declared_split']]
        by_hash = {}
        for row in selected:
            by_hash.setdefault(row['sha256'], row)
        jobs = sorted(by_hash.values(), key=lambda r: (r['declared_split'] == 'train', r['path']))
    else:
        jobs = [p for p in pairs if (p['declared_split'] in ('validation', 'test') if phase == 'controls' else p['bundle'] == 'train_pdf')]
    results, timing = [], []
    cv2.setNumThreads(1)
    for i, row in enumerate(jobs):
        tick = time.perf_counter()
        if phase in ('native', 'pdf'):
            try:
                record = (inspect_native if phase == 'native' else inspect_pdf)(root/'dataset'/row['path'])
            except Exception as exc:
                record = {'status': 'error', 'error': {'type': type(exc).__name__, 'message': str(exc).replace(str(root), '<workspace>')}}
            record.update(path=row['path'], sha256=row['sha256'])
        else:
            image_dir = directory/f'pair-{i+1:04d}'
            image_dir.mkdir()
            record = compare_pair(root, row, image_dir, files, png=phase == 'controls')
            record['artifacts'] = [pin(directory, p.relative_to(directory).as_posix()) for p in sorted(image_dir.iterdir())]
        name = f'record-{i+1:04d}.json'
        (directory/name).write_bytes(encoded(record))
        results.append({'path': name, 'sha256': digest(directory/name)})
        timing.append({'index': i+1, 'seconds': time.perf_counter()-tick})
        if (i+1) % 10 == 0 or i+1 == len(jobs):
            print(json.dumps({'phase': phase, 'processed': i+1, 'total': len(jobs), 'last_status': record['status']}), flush=True)
    inputs(root, config_pin)
    (directory/'runtime.json').write_bytes(encoded(timing))
    manifest = {'schema_version': 'dataset-content-phase/1.0', 'phase': phase, 'config': config_pin,
                'job_count': len(jobs), 'records': results, 'runtime': pin(directory, 'runtime.json')}
    (directory/'manifest.json').write_bytes(encoded(manifest))
    return pin(root, (directory/'manifest.json').relative_to(root).as_posix())


def load_phase(root, base, phase, config_pin):
    directory = base/phase
    manifest = read(directory/'manifest.json')
    require(manifest['schema_version'] == 'dataset-content-phase/1.0' and manifest['phase'] == phase
            and manifest['config'] == config_pin, 'Changed phase identity')
    checked(directory, manifest['runtime'])
    records = [read(checked(directory, r)) for r in manifest['records']]
    require(len(records) == manifest['job_count'], 'Incomplete phase')
    for r in records:
        for a in r.get('artifacts', []):
            checked(directory, a)
    return records
