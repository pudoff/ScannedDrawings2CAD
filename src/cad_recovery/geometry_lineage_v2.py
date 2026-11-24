"""Geometry-only block traversal: annotations are excluded before any transform."""
from collections import Counter
import math

from ezdxf.entities import Ellipse
from ezdxf.math import NonUniformScalingError, OCSTransform

from .family_catalog_v2 import require, stable_digest
from .geometry_lineage_v1 import ANNOTATIONS, GEOMETRY, entity_record


def has_width(entity):
    if entity.dxftype() == 'LWPOLYLINE':
        return bool(entity.dxf.get('const_width', 0) or any(p[2] or p[3] for p in entity.get_points('xyseb')))
    return bool(entity.dxf.get('default_start_width', 0) or entity.dxf.get('default_end_width', 0)
                or any(v.dxf.get('start_width', 0) or v.dxf.get('end_width', 0) for v in entity.vertices))


def fingerprint_document(document, *, entity_limit=100_000, depth_limit=12):
    """Transform detached supported leaves, never an INSERT with attached text.

    Native block bases, nested affine transforms and MINSERT offsets are retained.
    Unsupported entities, failed copies or lost width never receive a signature.
    """
    records, skipped, types, conversions, problems = [], Counter(), Counter(), Counter(), []
    visited = 0

    def visit():
        nonlocal visited
        visited += 1
        require(visited <= entity_limit, 'Geometry entity limit')

    def problem(entity, blocks, path, error):
        problems.append({'type': type(error).__name__, 'reason': str(error)[:300],
                         'entity_type': entity.dxftype(), 'native_handle': entity.dxf.handle,
                         'block_path': list(blocks), 'insert_path': list(path)})

    def apply(entity, matrices):
        if not matrices:
            yield entity
            return
        matrix, *remaining = matrices
        kind = entity.dxftype()
        if kind in ('LWPOLYLINE', 'POLYLINE'):
            is_2d = kind == 'LWPOLYLINE' or entity.is_2d_polyline
            require(not (is_2d and has_width(entity) and not OCSTransform(entity.dxf.extrusion, matrix).scale_uniform),
                    'Non-uniform polyline width is not preserved by primitive decomposition')
        # Transform a copy: failed transform implementations can mutate before raising.
        transformed = entity.copy()
        if hasattr(transformed, 'remove_association'):
            transformed.remove_association()
        try:
            transformed.transform(matrix)
        except NonUniformScalingError:
            if kind in ('CIRCLE', 'ARC'):
                require(math.isfinite(entity.dxf.radius) and entity.dxf.radius > 1e-12, 'Invalid transformed curve radius')
                transformed = Ellipse.from_arc(entity)
                transformed.transform(matrix)
                conversions[kind+'->ELLIPSE'] += 1
                yield from apply(transformed, remaining)
            elif kind in ('LWPOLYLINE', 'POLYLINE'):
                require(not has_width(entity), 'Polyline widths cannot be dropped')
                conversions[kind+'->primitives'] += 1
                for primitive in entity.virtual_entities():
                    visit()
                    require(primitive.dxftype() in GEOMETRY, 'Unsupported decomposed primitive')
                    yield from apply(primitive, matrices)
            else:
                raise
        else:
            yield from apply(transformed, remaining)

    def walk(entities, blocks=(), path=(), matrices=()):
        require(len(blocks) <= depth_limit, 'Block nesting limit')
        for entity in entities:
            visit()
            kind = entity.dxftype()
            if kind in ANNOTATIONS:
                skipped[kind] += 1
                continue
            try:
                if kind == 'INSERT':
                    name = entity.dxf.name
                    require(name not in blocks, 'Cyclic block reference')
                    require(name in document.blocks and not document.blocks[name].block.is_xref, 'Missing or external block')
                    require(entity.mcount <= entity_limit-visited+1, 'Block array instance limit')
                    inserts = entity.multi_insert() if entity.mcount > 1 else (entity,)
                    for insert in inserts:
                        transform = insert.matrix44()
                        require(all(math.isfinite(v) for row in transform.rows() for v in row)
                                and abs(transform.determinant()) > 1e-12, 'Nonfinite or singular INSERT transform')
                        walk(document.blocks[name], blocks+(name,), path+(entity.dxf.handle,), (transform,)+matrices)
                elif kind in GEOMETRY:
                    for transformed in apply(entity, matrices):
                        records.append(entity_record(transformed))
                        types[transformed.dxftype()] += 1
                else:
                    raise ValueError('Unsupported geometry-scope entity: '+kind)
            except Exception as error:
                # Continue the inventory, but any problem excludes the entire document from matching.
                problem(entity, blocks, path, error)

    try:
        walk(document.modelspace())
    except Exception as error:
        problems.append({'type': type(error).__name__, 'reason': str(error)[:300],
                         'entity_type': None, 'native_handle': None, 'block_path': [], 'insert_path': []})
    status = 'incomplete' if problems else 'complete_supported_geometry' if records else 'empty'
    hashes = sorted(stable_digest(record) for record in records)
    signature = stable_digest({'method': 'modelspace-native-tags/1.0', 'entities': hashes}) if status == 'complete_supported_geometry' else None
    return {'status': status, 'signature': signature, 'geometry_entities': len(records),
        'geometry_types': dict(sorted(types.items())), 'ignored_annotations': dict(sorted(skipped.items())),
        'visited_entities': visited, 'problems': problems, 'conversions': dict(sorted(conversions.items())),
        'physical_geometry_approved': False, 'semantic_family_approved': False,
        'annotations_filtered_before_transform': True, 'geometry_signature_format': 'modelspace-native-tags/1.0'}
