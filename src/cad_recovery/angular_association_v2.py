"""Corrected angle dispatch: a chamfer retains its distinct relationship type."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

from . import angular_association_v1 as core
from . import dimension_association_v3 as linear
from .family_catalog_v2 import require
from .notation import parse_notation

POLICY = {'id': 'angular-association/2.0', 'core': core.POLICY,
          'dispatch': 'Whole-string parsed chamfer is excluded from angular search; raw OCR and v3 unsupported chamfer remain intact.',
          'physical_geometry_approved': False, 'production_admitted': False}
angular_contract = core.angular_contract


def associate_angular_dimensions(source_image, output_dir, *, geometry, ocr, arrows, prefix='page-001'):
    output_dir = Path(output_dir)
    require(not output_dir.exists() and not output_dir.is_symlink(), 'Angular output must be new')
    require(isinstance(prefix, str) and prefix and Path(prefix).name == prefix and prefix not in ('.', '..'), 'Invalid prefix')
    # Dispatch uses a whole raw string, never a numerical comparison with geometry.
    lines = {r['id']: r for r in ocr.get('lines', [])}
    excluded = {n['id'] for n in ocr.get('notations', [])
                if n.get('source_line_id') in lines and parse_notation(lines[n['source_line_id']]['raw'])['kind'] == 'chamfer'}
    reduced = deepcopy(ocr); reduced['notations'] = [n for n in reduced.get('notations', []) if n['id'] not in excluded]
    report = core.associate_angular_dimensions(source_image, output_dir/'angular-v1-core', geometry=geometry, ocr=reduced, arrows=arrows, prefix=prefix)
    baseline = {k: 'angular-v1-core/'+v for k, v in report['linear_baseline_artifacts'].items()}
    if excluded:
        original = linear.associate_dimensions_v3(Path(source_image), output_dir/'linear-v3-complete', geometry=geometry, ocr=ocr,
                                                 arrows=arrows, prefix=prefix, record_timing=False)
        selected = {r['notation_id']: r for r in report['associations']}
        old_ids = {r['notation_id']: r['id'] for r in original['associations']}
        remap = {r['id']: old_ids[r['notation_id']] for r in report['associations']}
        def references(value):
            if isinstance(value, list): return [references(v) for v in value]
            if not isinstance(value, dict): return value
            result = {}
            for key, item in value.items():
                if key.endswith('association_id') and isinstance(item, str): result[key] = remap.get(item, item)
                elif key.endswith('association_ids') and isinstance(item, list): result[key] = [remap.get(v, v) for v in item]
                else: result[key] = references(item)
            return result
        selected = {key: references(value) for key, value in selected.items()}
        report['associations'] = [deepcopy(r if r['notation_id'] in excluded else selected[r['notation_id']]) for r in original['associations']]
        for i, row in enumerate(report['associations'], 1): row['id'] = f'association-{i:04d}'
        baseline = {k: 'linear-v3-complete/'+v for k, v in original['artifacts'].items()}
    report['algorithm_version'] = 'angular-association-v2'; report['angular_dispatch_policy'] = POLICY
    report['chamfer_notation_ids_preserved'] = sorted(excluded)
    report['linear_baseline_artifacts'] = baseline
    report['summary'] = {s: sum(r['status'] == s for r in report['associations']) for s in ('proposed', 'ambiguous', 'unresolved', 'unsupported')}
    report['summary']['total'] = len(report['associations'])
    report['artifacts'] = {'report': prefix+'-angular-associations.json', 'preview': prefix+'-angular-associations.png'}
    with Image.open(source_image) as original:
        image = original.convert('RGB'); image.thumbnail((1600, 1600))
    scale = np.array(image.size)/report['processed_size_px']; draw = ImageDraw.Draw(image)
    for row in report['associations']:
        color = '#008060' if row['status'] == 'proposed' else '#777777' if row['status'] == 'unsupported' else '#bd6320'
        if row['text_polygon_px']:
            polygon = (np.asarray(row['text_polygon_px'])*scale).tolist()
            draw.line([tuple(p) for p in polygon+[polygon[0]]], fill=color, width=2)
        if row['kind'] == 'angular' and len(row['anchors_px']) == 3:
            for point in row['anchors_px'][1:]:
                draw.line([tuple(np.asarray(row['anchors_px'][0])*scale), tuple(np.asarray(point)*scale)], fill=color, width=2)
    image.save(output_dir/report['artifacts']['preview'])
    report['artifact_sha256'] = {'preview': hashlib.sha256((output_dir/report['artifacts']['preview']).read_bytes()).hexdigest()}
    require(hashlib.sha256(Path(source_image).read_bytes()).hexdigest() == report['source_sha256'], 'Source changed during dispatch')
    (output_dir/report['artifacts']['report']).write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return report
