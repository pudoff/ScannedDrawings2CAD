"""Opt-in grouped ownership over the frozen angular-v4 support search."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import angular_association_v4 as baseline
from . import angular_group_ownership_v1 as ownership
from .family_catalog_v2 import require

POLICY = {'id': 'angular-association/5.0', 'baseline': baseline.POLICY, 'ownership': ownership.POLICY,
          'physical_geometry_approved': False, 'production_admitted': False, 'training_eligible': False}


def associate_angular_dimensions(source_image, output_dir, *, geometry, ocr, arrows, leader_geometry=None,
                                 upstream_incomplete=False, component_search_incomplete=False, parent_ocr=None, prefix='page-001'):
    source_image, output_dir = Path(source_image), Path(output_dir)
    require(not output_dir.exists() and not output_dir.is_symlink() and type(component_search_incomplete) is bool, 'Choose new v5 output')
    report = baseline.associate_angular_dimensions(source_image, output_dir/'ungrouped-v4', geometry=geometry, ocr=ocr,
        arrows=arrows, leader_geometry=leader_geometry, upstream_incomplete=upstream_incomplete or component_search_incomplete, prefix=prefix)
    parent_report = None
    if parent_ocr is not None:
        parent_report = baseline.associate_angular_dimensions(source_image, output_dir/'raw-parent-v4', geometry=geometry, ocr=parent_ocr,
            arrows=arrows, leader_geometry=leader_geometry, upstream_incomplete=upstream_incomplete, prefix=prefix)
    result = ownership.resolve(source_image, report, geometry=geometry, ocr=ocr, arrows=arrows, leader_geometry=leader_geometry,
        parent_ocr=parent_ocr, parent_report=parent_report, component_search_incomplete=component_search_incomplete)
    result['algorithm_version'] = 'angular-association-v5'; result['group_ownership_policy'] = POLICY
    result['raw_parent_artifacts'] = {k: 'raw-parent-v4/'+v for k, v in parent_report['artifacts'].items()} if parent_report else None
    result['linear_baseline_artifacts'] = {k: 'ungrouped-v4/'+v for k, v in report['linear_baseline_artifacts'].items()}
    result['summary'] = {s: sum(a['status'] == s for a in result['associations']) for s in ('proposed', 'ambiguous', 'unresolved', 'unsupported', 'associated_component')}
    result['summary']['total'] = len(result['associations'])
    result['artifacts'] = {'report': prefix+'-angular-associations.json', 'preview': prefix+'-angular-associations.png'}
    with Image.open(source_image) as image: image = image.convert('RGB'); image.thumbnail((1600, 1600))
    scale = np.asarray(image.size)/result['processed_size_px']; draw = ImageDraw.Draw(image)
    for item in result['associations']:
        color = '#008060' if item['status'] == 'proposed' else '#2878b5' if item['status'] == 'associated_component' else '#bd6320'
        p = [tuple(v) for v in np.asarray(item['text_polygon_px'])*scale]
        if p: draw.line(p+[p[0]], fill=color, width=2)
        if len(item['anchors_px']) == 3:
            for point in item['anchors_px'][1:]: draw.line([tuple(np.asarray(item['anchors_px'][0])*scale), tuple(np.asarray(point)*scale)], fill=color, width=2)
    image.save(output_dir/result['artifacts']['preview'])
    result['artifact_sha256'] = {'preview': hashlib.sha256((output_dir/result['artifacts']['preview']).read_bytes()).hexdigest()}
    require(hashlib.sha256(source_image.read_bytes()).hexdigest() == result['source_sha256'], 'Source changed during grouped association')
    (output_dir/result['artifacts']['report']).write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return result
