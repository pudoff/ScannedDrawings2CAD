"""Qualify the existing bounded axis detector and actual native CENTER entities."""
from collections import Counter
import math

import cv2
import numpy as np

from .cad_axis_pattern_v5 import apply as detect
from .cad_axis_pattern_v4 import ink_cell_errors
from .cad_object_export_v2 import APPID, read_provenance
from .closed_primitive_cycle_v1 import ledger
from .grid_role_revision_v1 import sha


def apply(assembly, gray, roles, geometry, heads=(), *, protected=None, text_protection=None):
    """Keep V5 thresholds/search, add invariants at its current integration point."""
    result = detect(assembly, gray, roles, geometry, heads,
                    protected=protected, text_protection=text_protection)
    if ledger(result['objects']+result.get('withheld', [])) != ledger(assembly['objects']+assembly.get('withheld', [])):
        raise ValueError('CENTER assembly changed source edge ownership')
    circular = lambda r: [o for o in r['objects'] if o['kind'] in ('arc', 'circle')]
    if circular(result) != circular(assembly):
        raise ValueError('CENTER assembly changed ARC/CIRCLE objects')
    audit = result['axis_pattern']
    if not audit['complete'] and result['objects'] != assembly['objects']:
        raise ValueError('Incomplete CENTER search published provisional objects')
    axes = [o for o in result['objects'] if o.get('axis_pattern_evidence')]
    if len({o['id'] for o in axes}) != len(axes):
        raise ValueError('CENTER identities must be unique')
    for obj in axes:
        if obj['kind'] != 'line' or obj.get('semantic_ownership_confirmed'):
            raise ValueError('Automatic CENTER must be a separate draft LINE')
    result['axis_component_contract'] = dict(version='axis-component-contract/1.0',
        accepted=list(audit['accepted']), search_complete=audit['complete'],
        source_edges_exact=True, circular_objects_unchanged=True,
        source_gray_sha256=sha(gray), pixel_roles_changed=False,
        manual_axis_groups_used=False, physical_geometry_approved=False)
    return result


def audit_entities(doc, gray, *, expected_ids=None):
    """Rebuild A-aligned visible dashes from DXF tags, independently of cached runs."""
    ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    rows = []
    for entity in doc.modelspace():
        if not entity.has_xdata(APPID):
            continue
        proof = read_provenance(entity)
        if not proof.get('axis_pattern_evidence'):
            continue
        if entity.dxftype() != 'LINE' or proof['source_sha256'] != sha(gray):
            raise ValueError('CENTER must be a source-bound LINE')
        layer = doc.layers.get(entity.dxf.layer)
        ltype = doc.linetypes.get(layer.dxf.linetype)
        tags = ltype.pattern_tags.tags
        pattern = np.array([tag.value for tag in tags if tag.code == 49], float)
        if (not entity.dxf.layer.startswith('AXES_PATTERN_DRAFT_')
                or not layer.dxf.linetype.startswith('CENTER_PATTERN_')
                or entity.dxf.linetype.upper() != 'BYLAYER'
                or entity.dxf.ltscale != 1 or doc.header['$LTSCALE'] != 1
                or tags.get_first_value(72) != 65 or len(pattern) != 4
                or not np.isfinite(pattern).all() or not np.array_equal(np.sign(pattern), [1, -1, 1, -1])
                or not np.allclose(pattern, proof['linetype_pattern_px'], rtol=0, atol=1e-10)):
            raise ValueError('Native CENTER layer/pattern differs from source contract')
        a = np.array(entity.dxf.start)[:2]; b = np.array(entity.dxf.end)[:2]
        length = float(np.linalg.norm(b-a)); period = float(abs(pattern).sum())
        if not math.isfinite(length) or length <= 0:
            raise ValueError('Degenerate CENTER LINE')
        cycles = int(math.floor(length/period)); terminal = (length-cycles*period+pattern[0])/2
        if cycles < 2 or terminal < pattern[0]/2:
            raise ValueError('CENTER lacks a supported repeating pattern')
        spans = [(0., terminal)]; cursor = terminal
        for index in range(cycles):
            cursor += abs(pattern[1]); spans.append((cursor, cursor+pattern[2]))
            cursor += pattern[2]+abs(pattern[3])
            end = length if index == cycles-1 else cursor+pattern[0]
            spans.append((cursor, end)); cursor = end
        u = (b-a)/length
        points = np.concatenate([a+np.arange(lo, hi+.01, .5)[:, None]*u for lo, hi in spans])
        points[:, 1] = gray.shape[0]-1-points[:, 1]
        errors = ink_cell_errors(ink, points)
        if errors.max() > 2 or np.percentile(errors, 95) > 1.25:
            raise ValueError('Actual visible CENTER dashes leave source ink')
        evidence = proof['axis_pattern_evidence']
        cached = np.array(evidence['A_aligned_runs'])-evidence['lo']
        if cached.shape != np.shape(spans) or not np.allclose(cached, spans, rtol=0, atol=1e-8):
            raise ValueError('Native A-aligned runs differ from measured axis extent')
        if proof.get('semantic_ownership_confirmed') is not False:
            raise ValueError('Automatic CENTER is not confirmed semantic ownership')
        rows.append(dict(object_id=proof['object_id'], handle=entity.dxf.handle,
            start=list(entity.dxf.start)[:2], end=list(entity.dxf.end)[:2], length_px=length,
            layer=entity.dxf.layer, linetype=layer.dxf.linetype,
            pattern_px=pattern.tolist(), cycles=cycles, native_visible_strokes=len(spans),
            native_dash_cell_max_px=float(errors.max()), native_dash_cell_p95_px=float(np.percentile(errors, 95))))
    if expected_ids is not None and Counter(r['object_id'] for r in rows) != Counter(expected_ids):
        raise ValueError('Expected whole CENTER objects are missing or duplicated in DXF')
    return rows
