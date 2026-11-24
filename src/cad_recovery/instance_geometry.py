"""Resolve one explicit block-instance path without flattening or modifying CAD.

The returned entity is a detached copy in planar modelspace coordinates. Its
source identity stays in provenance, never in a fabricated native handle.
Geometry resolution does not assign a contour role or approve training labels.
"""
from __future__ import annotations

import math

import numpy as np
from ezdxf.upright import upright

MAX_INSTANCE_DEPTH = 8
MAX_TEXT_BYTES = 2 * 1024**2
PLANAR_TOLERANCE = 1e-9
MIN_SCALE = 1e-12
SUPPORTED_TYPES = {'LINE', 'CIRCLE', 'ARC', 'TEXT', 'MTEXT'}


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{label} must be a finite number')
    return float(value)


def _point(value, label):
    try:
        point = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{label} must be a finite XYZ point') from error
    if point.shape != (3,) or not np.isfinite(point).all() or abs(point[2]) > PLANAR_TOLERANCE:
        raise ValueError(f'{label} must be a finite point in the XY plane')
    return point


def _extrusion(entity):
    vector = np.asarray(entity.dxf.get('extrusion', (0, 0, 1)), dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all() or not np.allclose(vector, (0, 0, 1), rtol=0, atol=1e-12):
        raise ValueError('Only the default XY entity plane is supported')


def _geometry(entity):
    kind = entity.dxftype()
    if kind not in SUPPORTED_TYPES:
        raise ValueError(f'Unsupported instance entity type: {kind}')
    _extrusion(entity)
    if kind in ('LINE', 'CIRCLE', 'ARC', 'TEXT'):
        if abs(_number(entity.dxf.get('thickness', 0), 'Thickness')) > PLANAR_TOLERANCE:
            raise ValueError('Nonzero thickness is outside planar geometry support')
    if kind == 'LINE':
        start = _point(entity.dxf.start, 'Line start')
        end = _point(entity.dxf.end, 'Line end')
        if np.linalg.norm(end-start) <= MIN_SCALE:
            raise ValueError('Degenerate line cannot support instance geometry')
    elif kind in ('CIRCLE', 'ARC'):
        _point(entity.dxf.center, 'Curve center')
        if _number(entity.dxf.radius, 'Radius') <= MIN_SCALE:
            raise ValueError('Degenerate curve cannot support instance geometry')
        if kind == 'ARC':
            start = _number(entity.dxf.start_angle, 'Arc start angle')
            end = _number(entity.dxf.end_angle, 'Arc end angle')
            if (end-start) % 360 <= 1e-9:
                raise ValueError('Degenerate arc sweep')
    else:
        _point(entity.dxf.insert, 'Text insert')
        if entity.dxf.hasattr('align_point'):
            _point(entity.dxf.align_point, 'Text alignment point')
        height = entity.dxf.height if kind == 'TEXT' else entity.dxf.char_height
        if _number(height, 'Text height') <= 0:
            raise ValueError('Text height must be positive')
        for attribute in ('width', 'rotation', 'oblique'):
            if entity.dxf.hasattr(attribute):
                _number(entity.dxf.get(attribute), f'Text {attribute}')
        if entity.dxf.hasattr('text_direction'):
            if np.linalg.norm(_point(entity.dxf.text_direction, 'Text direction')) <= MIN_SCALE:
                raise ValueError('Text direction is degenerate')
        text = entity.dxf.text if kind == 'TEXT' else entity.text
        if not isinstance(text, str) or len(text.encode('utf-8')) > MAX_TEXT_BYTES:
            raise ValueError('Text exceeds the instance review budget')


def _handle(value):
    if not isinstance(value, str) or not value or len(value) > 64:
        raise ValueError('Expected a nonempty native handle')
    return value


def resolve_instance_entity(doc, handle: str, insert_path: list[str]):
    """Return ``(detached_copy, provenance)`` for a single explicit instance.

    ``insert_path`` lists INSERT handles from modelspace towards the leaf block;
    an empty path requires a native modelspace entity. Curves retain their type
    only under planar similarities at every step. Nonuniform or reflected text
    requires a dedicated glyph/layout review and is deliberately refused.
    """
    _handle(handle)
    if not isinstance(insert_path, list) or len(insert_path) > MAX_INSTANCE_DEPTH:
        raise ValueError('Instance path must be a list within the depth limit')
    for item in insert_path:
        _handle(item)
    if len(set(insert_path)) != len(insert_path):
        raise ValueError('Repeated INSERT in instance path')
    layout = doc.modelspace()
    owner = layout.block_record_handle
    seen_blocks = {owner}
    chain = []
    for insert_handle in insert_path:
        insert = doc.entitydb.get(insert_handle)
        if (insert is None or not insert.is_alive or insert.dxftype() != 'INSERT'
                or insert.dxf.owner != owner):
            raise ValueError('INSERT does not belong to the specified parent instance')
        _extrusion(insert)
        _point(insert.dxf.insert, 'INSERT insertion point')
        _number(insert.dxf.rotation, 'INSERT rotation')
        if insert.dxf.row_count != 1 or insert.dxf.column_count != 1:
            raise ValueError('MINSERT arrays require explicit row/column instance support')
        scales = [_number(insert.dxf.get(axis+'scale', 1), f'INSERT {axis} scale') for axis in 'xyz']
        if min(abs(x) for x in scales) <= MIN_SCALE:
            raise ValueError('INSERT has a zero or numerically singular scale')
        layout = insert.block()
        if layout is None:
            raise ValueError('INSERT references a missing block')
        owner = layout.block_record_handle
        if owner in seen_blocks:
            raise ValueError('Recursive block path is not supported')
        seen_blocks.add(owner)
        base = _point(layout.block.dxf.base_point, 'Block base point')
        matrix = insert.matrix44()
        rows = np.array(list(matrix.rows()), dtype=float)
        if rows.shape != (4, 4) or not np.isfinite(rows).all():
            raise ValueError('INSERT produces a nonfinite transformation')
        chain.append((matrix, {
            'insert_handle': insert_handle, 'block_name': layout.name,
            'owner_block_record_handle': insert.dxf.owner,
            'target_block_record_handle': owner, 'block_base_point': base.tolist(),
            'scale_xyz': scales, 'matrix_row_vector': rows.tolist(),
        }))
    source = doc.entitydb.get(handle)
    if source is None or not source.is_alive or source.dxf.owner != owner:
        raise ValueError('Leaf entity does not belong to the final instance block')
    _geometry(source)
    kind = source.dxftype()
    if kind in ('CIRCLE', 'ARC', 'TEXT', 'MTEXT'):
        for _, step in chain:
            sx, sy, _ = step['scale_xyz']
            if not math.isclose(abs(sx), abs(sy), rel_tol=1e-12, abs_tol=0):
                raise ValueError('Nonuniform curve or text instance scale is unsupported')
            if kind in ('TEXT', 'MTEXT') and sx*sy < 0:
                raise ValueError('Reflected text requires a separate glyph/layout review')
    result = source.copy()
    # Copies deliberately have no native handle/owner. Do not add them to doc.
    try:
        for matrix, _ in reversed(chain):
            result.transform(matrix)
            if kind in ('CIRCLE', 'ARC'):
                upright(result)
            _geometry(result)
    except (ArithmeticError, ValueError) as error:
        raise ValueError('Instance transformation failed or left supported planar geometry') from error
    provenance = {
        'schema_version': '1.0', 'source_handle': handle, 'insert_path': list(insert_path),
        'owner_block': doc.entitydb[owner].dxf.name, 'owner_block_record_handle': owner,
        'entity_type': kind, 'transforms': [step for _, step in chain],
        'transform_application_order': 'innermost_to_outermost',
        'coordinate_system': 'planar modelspace XY; default +Z extrusion',
        'curve_orientation_may_reverse': kind == 'ARC' and any(s['scale_xyz'][0]*s['scale_xyz'][1] < 0 for _, s in chain),
        'units': 'native CAD units; no physical conversion', 'source_modified': False,
        'geometry_role_verified': False, 'annotation_approved': False, 'training_eligible': False,
    }
    if kind in ('TEXT', 'MTEXT'):
        provenance['source_raw_text'] = source.dxf.text if kind == 'TEXT' else source.text
        provenance['source_plain_text'] = source.plain_text()
        provenance['raster_reading_verified'] = False
    return result, provenance
