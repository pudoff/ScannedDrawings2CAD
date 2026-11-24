"""Replace source-owned coarse geometry with admitted native subcell content.

The public runner re-observes and authorizes the source before calling these
low-level geometric utilities. Saved partition documents alone grant no roles.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import cv2
import ezdxf
import numpy as np

from .cad_object_export_v2 import entity_geometry, read_provenance, _provenance
from .cad_line_export_v3 import _record
from .demo_export_v1 import merged_runs
from .experimental_export_v1 import ROLE_LAYERS
from .native_dimension_transfer_v1 import _footprint, _sha, _supported_matrix
from .part_admission_v1 import object_sha
from .source_arrow_observation_v1 import checked_matrix

POLICY = {'maximum_cells': 4096, 'maximum_entities': 100000,
          'maximum_native_rectangles': 50000, 'epsilon': 1e-9}


def plan_patch(source_gray, link, roles, permitted, protection, assigned):
    """Plan an axis-aligned native patch under an already fresh semantic gate."""
    mask = np.zeros_like(roles, bool)
    report = {'schema_version': 'native-subcell-plan/1.0', 'status': 'withheld',
        'reasons': [], 'cells': [], 'physical_geometry_approved': False,
        'working_roles_changed': 0, 'new_TEXT_or_DIMENSION': False}
    result = {'report': report, 'mask': mask, 'cells': [], 'rectangles': [], 'axes': []}
    transfer = link['transfer']; partition = transfer['source_partition']
    if (link['status'] != 'source_dimension_bodies_admitted' or not link['anchor_prior']['accepted']
            or not link['anchor_prior']['complete'] or not transfer['complete']
            or transfer['holds'] or not partition.get('complete')):
        report['reasons'].append('fresh_semantic_chain_or_partition_not_admitted'); return result
    if (roles.ndim != 2 or roles.dtype != np.uint8 or assigned.shape != roles.shape
            or permitted.shape != roles.shape or permitted.dtype != bool
            or protection.shape != roles.shape):
        raise ValueError('Aligned semantic maps and boolean automatic authority required')
    if partition['source_window_sha256'] != _sha(source_gray) or partition['size_px'] != list(source_gray.shape[::-1]):
        raise ValueError('Native partition does not describe current source pixels')
    if partition['roles_sha256'] != _sha(roles) or partition['protection_sha256'] != _sha(protection):
        raise ValueError('Native partition semantic inputs changed')
    matrix = checked_matrix(partition['working_to_source'])
    if not _supported_matrix(matrix):
        report['reasons'].append('unsupported_registration'); return result
    plane = np.zeros(source_gray.shape, np.uint8)
    for y, a, b, code in partition['classes_rle']:
        if not (0 <= y < plane.shape[0] and 0 <= a < b <= plane.shape[1] and code in (1, 2, 3, 4)) or plane[y, a:b].any():
            raise ValueError('Invalid or overlapping native partition')
        plane[y, a:b] = code
    ink = cv2.threshold(source_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1] > 0
    if not np.array_equal(plane > 0, ink) or _sha(plane) != partition['class_map_sha256']:
        raise ValueError('Native partition lost or invented source ink')
    owners = [o for row in transfer['rows'] if row['role'] == 2
              for o in row['carrier_consensus']['owners']]
    if not owners:
        report['reasons'].append('measured_axis_missing'); return result
    observed = [np.asarray(o['bounded_axis_px'], float) for o in owners]
    axis = observed[0]
    if (axis.shape != (2, 2) or not np.isfinite(axis).all()
            or any(not np.allclose(axis, a, rtol=0, atol=1e-8) for a in observed[1:])):
        report['reasons'].append('measured_axes_not_identical'); return result
    delta = axis[1]-axis[0]
    vertical = abs(delta[0]) < 1e-8 and abs(delta[1]) > 1e-8
    horizontal = abs(delta[1]) < 1e-8 and abs(delta[0]) > 1e-8
    if not (vertical or horizontal):
        report['reasons'].append('diagonal_native_axis_requires_separate_export'); return result
    origin = np.asarray(partition['source_box_px'][:2], float)
    inverse = np.linalg.inv(matrix)
    # The complete carrier cell domain is derived from source pixels, not a
    # drawing name, OCR value, or manually selected working coordinates.
    yy, xx = np.nonzero(plane == 1)
    source = np.column_stack((xx, yy))+origin
    working = source@inverse[:2, :2].T+inverse[:2, 2]
    proposed = np.unique(np.rint(working).astype(int), axis=0)
    if len(proposed) > POLICY['maximum_cells']:
        report['reasons'].append('cell_budget'); return result
    holds = Counter()
    cells = []
    for x, y in proposed:
        if not (0 <= x < roles.shape[1] and 0 <= y < roles.shape[0]):
            holds['outside_working_frame'] += 1; continue
        cell = _footprint([x, y], matrix, origin, plane.shape)
        if cell is None:
            holds['source_cell_clipped'] += 1; continue
        a, t, b, d = cell; local = plane[t:d, a:b]
        if not (local == 1).any():
            continue
        if protection[y, x] or (local == 2).any():
            holds['protected_source_cell'] += 1; continue
        if int(roles[y, x]) not in (1, 2, 255) or (roles[y, x] == 1 and not permitted[y, x]):
            holds['original_role_not_authorized'] += 1; continue
        if int(assigned[y, x]) not in (1, 2, 255):
            holds['competing_CAD_role'] += 1; continue
        mask[y, x] = True; cells.append([int(x), int(y)])
    rectangles = [[a-.5, t-.5, b-.5, d-.5] for a, t, b, d in merged_runs(mask)]
    # Never span a missing or ambiguous source section with a new LINE.
    section = (plane == 1).T if horizontal else plane == 1
    center = float(axis[0, 1] if horizontal else axis[0, 0])
    present = np.zeros(len(section), bool)
    for i, row in enumerate(section):
        positions = np.flatnonzero(row)
        if not len(positions):
            continue
        if np.any(np.diff(positions) != 1) or not positions[0]-.5 <= center <= positions[-1]+.5:
            report['reasons'].append('carrier_section_not_unique'); return {**result, 'mask': np.zeros_like(mask)}
        present[i] = True
    edges = np.diff(np.pad(present.astype(np.int8), (1, 1)))
    axes = []
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        local = np.array([[center, start-.5], [center, end-.5]])
        if horizontal:
            local = local[:, ::-1]
        axes.append(((local+origin)@inverse[:2, :2].T+inverse[:2, 2]).tolist())
    if not cells:
        report['reasons'].append('no_authorized_native_cells'); return result
    report.update(status='admitted', cells=cells, withheld_cells=dict(holds),
        partition_sha256=object_sha(partition), source_ink_pixels=int(ink.sum()),
        source_window_sha256=partition['source_window_sha256'],
        source_box_px=partition['source_box_px'], working_to_source=matrix.tolist(),
        measured_axes_local_px=[a.tolist() for a in observed],
        mask_sha256=_sha(mask), roles_sha256=_sha(roles), authority_sha256=_sha(permitted),
        protection_sha256=_sha(protection), prior_CAD_roles_sha256=_sha(assigned),
        semantic_basis='fresh_two_head_chain_corroborated_PART_anchors_and_current_automatic_authority')
    result.update(mask=mask, cells=cells, rectangles=rectangles, axes=axes, axis=axes[0] if axes else [],
        plane=plane, matrix=matrix, origin=origin, inverse=inverse)
    report['plan_payload_sha256'] = _plan_hash(result)
    return result


def _plan_hash(plan):
    metadata = {k: v for k, v in plan['report'].items() if k != 'plan_payload_sha256'}
    return object_sha([metadata, _sha(plan['mask']), _sha(plan['plane']), plan['cells'],
        plan['rectangles'], plan['axes'], plan['matrix'].tolist(), plan['origin'].tolist(), plan['inverse'].tolist()])


def _subtract(rectangle, patches):
    pieces = [rectangle]
    for l, t, r, b in patches:
        output = []
        for a, c, d, e in pieces:
            x0, y0, x1, y1 = max(a, l), max(c, t), min(d, r), min(e, b)
            if x1-x0 <= 1e-10 or y1-y0 <= 1e-10:
                output.append([a, c, d, e]); continue
            output.extend(q for q in ([a, c, x0, e], [x1, c, d, e], [x0, c, x1, y0], [x0, y1, x1, e])
                          if q[2]-q[0] > 1e-10 and q[3]-q[1] > 1e-10)
        pieces = output
    return pieces


def _solid_bounds(entity, height):
    points = np.array([[entity.dxf.get(f'vtx{i}').x, height-1-entity.dxf.get(f'vtx{i}').y] for i in range(4)])
    low, high = points.min(0), points.max(0)
    if len(set(map(tuple, points))) != 4 or not all(x in (low[0], high[0]) and y in (low[1], high[1]) for x, y in points):
        raise ValueError('Only axis-aligned residual rectangles can be replaced')
    return [*low.tolist(), *high.tolist()]


def _add_solid(model, box, height, attrs):
    a, t, b, d = box
    return model.add_solid([(a, height-1-t), (b, height-1-t),
                            (a, height-1-d), (b, height-1-d)], dxfattribs=attrs)


def _set_provenance(entity, data):
    _provenance(entity, {**data, 'id': data['object_id']}, data['source_sha256'])
    _record(entity, data)


def entity_signature(entity):
    """Include residual rectangles and text, which the old model helper omits."""
    shape = entity_geometry(entity)
    if entity.dxftype() == 'SOLID':
        shape['vertices'] = [list(entity.dxf.get(f'vtx{i}')) for i in range(4)]
    elif entity.dxftype() == 'TEXT':
        shape.update(text=entity.dxf.text, insert=list(entity.dxf.insert),
                     height=entity.dxf.height, rotation=entity.dxf.rotation, style=entity.dxf.style)
    attrs = {k: getattr(entity.dxf, k) for k in ('layer', 'color', 'lineweight', 'linetype')}
    provenance = read_provenance(entity) if entity.has_xdata('CADRECOVERY_OBJECT') else None
    return object_sha([shape, attrs, provenance])


def _circle_touches(entity, patches, height):
    center = np.array([entity.dxf.center.x, height-1-entity.dxf.center.y]); radius = entity.dxf.radius
    start = -float(entity.dxf.end_angle) if entity.dxftype() == 'ARC' else 0.
    sweep = (float(entity.dxf.end_angle)-float(entity.dxf.start_angle)) % 360 if entity.dxftype() == 'ARC' else 360.
    def inside_arc(p):
        angle = math.degrees(math.atan2(p[1]-center[1], p[0]-center[0]))
        return sweep == 360 or (angle-start) % 360 <= sweep+1e-9
    candidates = [center+radius*np.array([math.cos(math.radians(a)), math.sin(math.radians(a))])
                  for a in (0, 90, 180, 270, start, start+sweep)]
    for l, t, r, b in patches:
        points = list(candidates)
        for x in (l, r):
            s = radius*radius-(x-center[0])**2
            if s >= 0:
                points.extend(np.array([x, center[1]+sign*math.sqrt(s)]) for sign in (-1, 1))
        for y in (t, b):
            s = radius*radius-(y-center[1])**2
            if s >= 0:
                points.extend(np.array([center[0]+sign*math.sqrt(s), y]) for sign in (-1, 1))
        if any(l-1e-9 <= p[0] <= r+1e-9 and t-1e-9 <= p[1] <= b+1e-9 and inside_arc(p) for p in points):
            return True
    return False


def export_patch(base_path, target, base_report, geometry, plan, assigned, *, render=False):
    """Write a separate replacement DXF; preflight all primitives before editing."""
    from .native_patch_clip_v1 import trim_model, clip_segment
    base_path, target = Path(base_path), Path(target)
    objects_path = target.with_name(target.stem+'-objects.json')
    if target.exists() or objects_path.exists():
        raise FileExistsError(target)
    if hashlib.sha256(base_path.read_bytes()).hexdigest() != base_report['dxf_sha256']:
        raise ValueError('Base DXF differs from its source catalog')
    if plan['report']['status'] == 'admitted' and (
            plan['report']['prior_CAD_roles_sha256'] != _sha(assigned)
            or plan['report']['plan_payload_sha256'] != _plan_hash(plan)):
        raise ValueError('Native plan or prior CAD role map changed after admission')
    before = object_sha([base_report, geometry, plan['report']]); old_bytes = base_path.read_bytes()
    doc = ezdxf.readfile(base_path); model = doc.modelspace(); height = base_report['size_px'][1]
    if len(model) > POLICY['maximum_entities'] or len(base_report['objects']) > POLICY['maximum_entities']:
        raise ValueError('Native replacement input entity budget exceeded')
    if doc.units != 0:
        raise ValueError('Native replacement requires working pixel coordinates')
    patches = plan['rectangles']; lookup = {r['id']: r for r in geometry}
    edits, solids = [], []; reasons = []
    if plan['report']['status'] == 'admitted':
        for row in base_report['objects']:
            entity = doc.entitydb[row['handle']]; kind = entity.dxftype()
            members = row['source_members']; supports = []
            for member in members:
                c = lookup[member['candidate_id']]; low, high = member['source_index_interval']; first = c['source_index_interval'][0]
                supports.extend(c.get('support_points', c.get('points_px'))[low-first:high-first+1])
            source_touches = any(any(l-1e-9 <= p[0] <= r+1e-9 and t-1e-9 <= p[1] <= b+1e-9
                                     for l, t, r, b in patches) for p in supports)
            if kind in ('ARC', 'CIRCLE'):
                if source_touches or _circle_touches(entity, patches, height):
                    reasons.append('curved_primitive_intersects_patch')
                continue
            if kind == 'POINT':
                p = entity.dxf.location
                if source_touches or any(l < p.x < r and t < height-1-p.y < b for l, t, r, b in patches):
                    reasons.append('isolated_point_intersects_patch')
                continue
            if kind not in ('LINE', 'LWPOLYLINE'):
                reasons.append('unsupported_native_primitive'); continue
            if kind == 'LINE':
                points = [[p.x, height-1-p.y] for p in (entity.dxf.start, entity.dxf.end)]
                shape = {'kind': 'line', 'points_px': points}
            else:
                vertices = list(entity.get_points('xyb'))
                if entity.closed or any(abs(p[2]) > 1e-12 for p in vertices):
                    # Closed geometry is safe outside the local patch.
                    points = np.asarray([[p[0], height-1-p[1]] for p in vertices])
                    if source_touches or (points[:, 0].max() >= min(q[0] for q in patches) and points[:, 0].min() <= max(q[2] for q in patches)
                            and points[:, 1].max() >= min(q[1] for q in patches) and points[:, 1].min() <= max(q[3] for q in patches)):
                        reasons.append('closed_or_bulged_primitive_near_patch')
                    continue
                shape = {'kind': 'polyline', 'points_px': [[p[0], height-1-p[1]] for p in vertices], 'closed': False}
            spatial = any(clip_segment(a, b, patches)['removed'] for a, b in zip(shape['points_px'], shape['points_px'][1:]))
            claimed = any(any(l-.000000001 <= p[0] <= r+.000000001 and t-.000000001 <= p[1] <= b+.000000001
                              for l, t, r, b in patches) for p in supports)
            if not spatial and not claimed:
                continue
            if len(members) != 1:
                reasons.append('multiple_source_members_need_separate_replacement'); continue
            try:
                replacement = trim_model(shape, supports, patches)
            except ValueError as exc:
                reasons.append(str(exc)); continue
            if replacement['changed']:
                edits.append((row, entity, replacement))
        for entity in model.query('SOLID'):
            if entity.dxf.layer == 'SOURCE_INK':
                continue
            rect = _solid_bounds(entity, height)
            pieces = _subtract(rect, patches)
            if pieces != [rect]:
                solids.append((entity, rect, pieces))
    proof = {'schema_version': 'native-subcell-export/1.0', 'plan': deepcopy(plan['report']),
        'status': 'withheld' if reasons or plan['report']['status'] != 'admitted' else 'applied',
        'reasons': sorted(set(reasons)), 'replaced_primitives': [], 'residual_rectangles': [],
        'native_lines': [], 'native_ink_rectangles': [], 'physical_geometry_approved': False,
        'new_DIMENSION_or_TEXT': False, 'production_changed': False}
    rows = deepcopy(base_report['objects'])
    if proof['status'] == 'applied':
        changed_handles = {row['handle'] for row, _, _ in edits}
        rows = [r for r in rows if r['handle'] not in changed_handles]
        for row, entity, replacement in edits:
            record = {'parent': deepcopy(row), **replacement, 'children': []}
            attrs = {k: entity.dxf.get(k) for k in ('layer', 'color', 'lineweight', 'linetype') if entity.dxf.get(k) is not None}
            model.delete_entity(entity)
            for piece in replacement['pieces']:
                points = [(p[0], height-1-p[1]) for p in piece['points_px']]
                e = model.add_line(*points, dxfattribs=attrs) if piece['kind'] == 'line' else model.add_lwpolyline(points, dxfattribs=attrs)
                provenance = {**row, 'object_id': 'native-patch-trim/'+object_sha([row['object_id'], piece])[:24],
                    'source_members': [], 'source_membership_model': 'fractional_frozen_parent',
                    'parent_source_members': deepcopy(row['source_members']), 'parent_object_id': row['object_id'],
                    'parent_model_interval': piece['model_span'], 'kind': piece['kind']}
                provenance.pop('handle', None); _set_provenance(e, provenance)
                child = {**provenance, 'handle': e.dxf.handle, 'layer': e.dxf.layer}
                rows.append(child); record['children'].append(child)
            proof['replaced_primitives'].append(record)
        for entity, rect, pieces in solids:
            attrs = {k: entity.dxf.get(k) for k in ('layer', 'color', 'lineweight', 'linetype') if entity.dxf.get(k) is not None}
            handle = entity.dxf.handle; model.delete_entity(entity)
            children = [_add_solid(model, box, height, attrs).dxf.handle for box in pieces]
            proof['residual_rectangles'].append({'old_handle': handle, 'old_bounds': rect, 'kept_bounds': pieces, 'new_handles': children})
        # Every admitted source section is a LINE. Missing sections and patch
        # gaps are separate objects; they are never silently bridged.
        for axis in plan['axes']:
            a, b = map(np.asarray, axis)
            intervals = clip_segment(a, b, patches)['removed']
            for low, high in intervals:
                points = np.asarray([a+low*(b-a), a+high*(b-a)])
                e = model.add_line(*[(float(x), height-1-float(y)) for x, y in points],
                    dxfattribs={'layer': ROLE_LAYERS[2], 'color': 256, 'lineweight': -1})
                provenance = {'object_id': 'native-source-line/'+object_sha(points.tolist())[:24], 'kind': 'line',
                    'role': 2, 'source_members': [], 'path_ids': [], 'source_sha256': plan['report']['source_window_sha256'],
                    'source_membership_model': 'native_source_partition', 'native_source_plan': object_sha(plan['report']),
                    'status': 'draft_requires_review', 'physical_geometry_approved': False}
                _set_provenance(e, provenance); rows.append({**provenance, 'handle': e.dxf.handle, 'layer': e.dxf.layer})
                proof['native_lines'].append({'handle': e.dxf.handle, 'points_px': points.tolist(), 'role': 2})
        inverse, origin, plane = plan['inverse'], plan['origin'], plan['plane']
        rectangle_count = 0
        for x, y in plan['cells']:
            footprint = _footprint([x, y], plan['matrix'], origin, plane.shape)
            a, t, b, d = footprint; actual = np.isin(plane[t:d, a:b], [2, 3, 4])
            role = int(assigned[y, x]); layer = ROLE_LAYERS.get(role, 'UNRESOLVED_INK')
            for l, top, r, bottom in merged_runs(actual):
                native = np.array([[a+l-.5, t+top-.5], [a+r-.5, t+bottom-.5]])+origin
                working = native@inverse[:2, :2].T+inverse[:2, 2]
                low, high = working.min(0), working.max(0)
                box = [max(float(low[0]), x-.5), max(float(low[1]), y-.5),
                       min(float(high[0]), x+.5), min(float(high[1]), y+.5)]
                if box[2]-box[0] <= 1e-10 or box[3]-box[1] <= 1e-10:
                    continue
                e = _add_solid(model, box, height, {'layer': layer, 'color': 256, 'lineweight': -1})
                proof['native_ink_rectangles'].append({'handle': e.dxf.handle, 'bounds': box,
                    'old_CAD_role': role, 'cell': [x, y], 'source_box_local_px': [a+l, t+top, a+r, t+bottom]})
                rectangle_count += 1
                if rectangle_count > POLICY['maximum_native_rectangles']:
                    raise ValueError('Native replacement rectangle budget exceeded')
    if len(model) > POLICY['maximum_entities']:
        raise ValueError('Native replacement entity budget exceeded')
    # Expected geometry/provenance is recorded before saving and rechecked on
    # reopening, including every untouched object and every fractional child.
    expected = {e.dxf.handle: entity_signature(e) for e in model}
    doc.saveas(target); reopened = ezdxf.readfile(target); audit = reopened.audit()
    if audit.errors or audit.fixes or reopened.units != 0:
        raise ValueError('Native replacement DXF failed readback')
    for handle, digest in expected.items():
        e = reopened.entitydb[handle]
        if entity_signature(e) != digest:
            raise ValueError('Native replacement geometry/provenance changed')
    if object_sha([base_report, geometry, plan['report']]) != before or base_path.read_bytes() != old_bytes:
        raise ValueError('Native replacement changed its baseline inputs')
    proof.update(source_replacement_exclusive=True, previous_report_sha256=object_sha(base_report),
        base_DXF_sha256=hashlib.sha256(old_bytes).hexdigest(), unchanged_object_count=len(base_report['objects'])-len(edits) if proof['status'] == 'applied' else len(rows))
    proof['all_entity_roundtrip_sha256'] = expected
    report = {**deepcopy(base_report), 'schema_version': 'native-subcell-CAD-export/1.0',
        'objects': rows, 'dxf': target.name, 'dxf_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
        'native_subcell_export': proof, 'audit_errors': 0, 'audit_fixes': 0,
        'source_membership_model': 'baseline_plus_fractional_parent_and_native_source_domains',
        'working_ownership_maps_apply_to': 'retained_baseline_only',
        'roundtrip_geometry_and_provenance_exact': True,
        'artifacts': {'objects': objects_path.name},
        'modelspace_counts': dict(Counter(e.dxftype() for e in reopened.modelspace()))}
    for key in ('geometry_preserving_role_export', 'residual_split_ownership'):
        if key in report:
            report['retained_baseline_'+key] = report.pop(key)
    if render:
        from .cad_axis_preview_v1 import render as preview
        preview_path = target.with_name(target.stem+'-preview.png'); preview(target, preview_path)
        report['artifacts']['preview'] = preview_path.name
    objects_path.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2)+'\n', encoding='utf-8')
    return report
