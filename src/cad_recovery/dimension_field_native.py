"""Pinned native facts for field review, without promoting CAD to label truth."""
from collections import Counter
from copy import deepcopy
import math
import re

from ezdxf.math import Matrix44
from ezdxf.tools.text import plain_mtext

from .dxf_semantics import _dimension
from .dimension_attachment_candidates import collect_attachment_candidates
from .instance_geometry import resolve_instance_entity


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite(values):
    values = [float(value) for value in values]
    _require(all(math.isfinite(value) for value in values), 'Native coordinates must be finite')
    return values


def normalize_dimension_override(raw):
    """Keep all source text; classify only H/W/Q formatting removal as safe.

    General MTEXT decoding is observational. Unknown commands, stacking,
    placeholders or meaningful prefixes cannot become a formatting correction.
    """
    _require(isinstance(raw, str) and len(raw) <= 100000, 'Invalid dimension override')
    normalized = plain_mtext(raw)
    number = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)'
    stripped, count = re.subn(r'\\[HWQ]'+number+r'x?;', '', raw)
    safe = count > 0 and stripped == normalized and '\\' not in stripped and '<>' not in stripped
    return {'raw': raw, 'plain': normalized, 'changed': raw != normalized,
            'formatting_only_removal_verified': safe,
            'method': 'ezdxf_plain_mtext_with_narrow_HWQ_removal_check',
            'source_modified': False}


def native_geometry_snapshot(entity):
    """Coordinates and presentation attributes, never a semantic role."""
    kind = entity.dxftype()
    _require(kind in ('LINE', 'CIRCLE', 'ARC'), 'Unsupported reviewed native geometry')
    row = {'handle': entity.dxf.handle, 'entity_type': kind, 'owner_handle': entity.dxf.owner,
           'layer': entity.dxf.layer, 'color': entity.dxf.color,
           'lineweight': entity.dxf.lineweight, 'linetype': entity.dxf.linetype,
           'semantic_role': 'not_inferred_from_presentation'}
    if kind == 'LINE':
        row.update(start=_finite(entity.dxf.start), end=_finite(entity.dxf.end))
        _require(row['start'] != row['end'], 'Degenerate reviewed native line')
    else:
        radius = float(entity.dxf.radius)
        _require(math.isfinite(radius) and radius > 0, 'Invalid reviewed native radius')
        row.update(center=_finite(entity.dxf.center), radius=radius,
                   extrusion=_finite(entity.dxf.extrusion))
        if kind == 'ARC':
            row.update(start_angle_deg=float(entity.dxf.start_angle), end_angle_deg=float(entity.dxf.end_angle))
            _require(math.isfinite(row['start_angle_deg']) and math.isfinite(row['end_angle_deg']),
                     'Nonfinite reviewed arc angles')
    return row


def extract_dimension_field_native(doc, stored_object):
    """Re-read the exact source object and incidence inventory, then add facts.

    The supported scope is a modelspace standard DIMENSION. Caller separately
    verifies file/report hashes and any raster readings or explicit role links.
    """
    identity = stored_object['identity']
    _require(identity.get('insert_path') == [] and identity.get('owner_block') == '*Model_Space',
             'Field review supports explicitly identified modelspace dimensions')
    handle = identity['handle']
    entity = doc.entitydb.get(handle)
    _require(entity is not None and entity.is_alive and entity.dxftype() == 'DIMENSION'
             and entity.dxf.owner == doc.modelspace().block_record_handle,
             'Dimension identity differs from native modelspace')
    fresh = _dimension(entity, doc, '*Model_Space', [], Matrix44())
    _require(fresh == stored_object['source_evidence'], 'Native dimension differs from pinned source evidence')
    _require(stored_object['units']['code'] == int(doc.units), 'Native unit declaration differs')
    _require(fresh['kind'] in ('linear', 'aligned', 'radius', 'diameter'),
             'Native dimension kind is not supported by this field review')
    _require(tuple(entity.dxf.extrusion) == (0., 0., 1.), 'Nonplanar dimension cannot receive planar field observations')
    measurement = fresh['measurement_raw']
    _require(type(measurement) in (int, float) and math.isfinite(measurement) and measurement > 0,
             'Expected a finite positive native measurement')
    attachment = collect_attachment_candidates(doc, [fresh])['dimensions'][0]
    _require(attachment == stored_object['cad_attachment_candidates'],
             'Native incidence candidates differ from pinned evidence')
    handles = sorted({candidate['geometry_handle'] for point in attachment.get('points', {}).values()
                      for candidate in point.get('candidates', [])})
    geometries = [native_geometry_snapshot(doc.entitydb[handle]) for handle in handles]
    normalized = normalize_dimension_override(fresh['text_override'])
    pieces = [row['text'] for row in fresh['stored_display_texts']]
    comparable = fresh['text_mode'] == 'explicit_override' and len(pieces) == 1
    factor = fresh['style_effective']['dimlfac']
    _require(type(factor) in (int, float) and math.isfinite(factor) and factor > 0,
             'Unsupported effective DIMLFAC')
    scaled_measurement = measurement*factor
    _require(math.isfinite(scaled_measurement), 'Nonfinite measurement times DIMLFAC')
    return {'identity': deepcopy(identity), 'units_header_code': int(doc.units),
            'native_kind': fresh['kind'], 'definition_points': deepcopy(fresh['definition_points_modelspace']),
            'definition_point_roles': deepcopy(fresh['definition_point_roles']),
            'measurement_raw': measurement, 'measurement_units': 'native_drawing_units_not_physical_truth',
            'dimlfac': factor, 'dimscale': fresh['style_effective']['dimscale'],
            'measurement_times_dimlfac': scaled_measurement,
            'angle_deg': fresh['angle_deg'], 'text_mode': fresh['text_mode'],
            'display_text_pieces': pieces, 'display_text_entities': deepcopy(fresh['stored_display_texts']),
            'override_normalization': normalized,
            'normalized_override_matches_single_display': normalized['plain'] == pieces[0] if comparable else None,
            'formatting_conflict_can_be_reinterpreted': comparable and normalized['formatting_only_removal_verified']
                and normalized['plain'] == pieces[0] and stored_object['field_statuses']['text'] == 'conflicting',
            'geometry_candidates': deepcopy(attachment), 'candidate_geometry': geometries,
            'physical_scale_confirmed': False, 'geometry_roles_approved': False,
            'source_modified': False, 'annotation_approved': False, 'training_eligible': False}


def extract_text_instance(doc, handle, insert_path):
    """Read an explicit supported text instance through the existing resolver."""
    entity, provenance = resolve_instance_entity(doc, handle, insert_path)
    _require(entity.dxftype() in ('TEXT', 'MTEXT'), 'Expected a native TEXT or MTEXT instance')
    return {'handle': handle, 'insert_path': list(insert_path), 'entity_type': entity.dxftype(),
            'plain_text': entity.plain_text(), 'world_insert_xyz': _finite(entity.dxf.insert),
            'native_owner_handle': doc.entitydb[handle].dxf.owner, 'provenance': provenance,
            'raster_reading_verified': False, 'annotation_approved': False, 'training_eligible': False}


def inspect_drawing_native(doc):
    """Inventory all live database resources and xrefs, including unused blocks.

    Their absence is scoped to native external-resource dependencies, not fonts,
    semantic completeness, physical units or approval of the drawing itself.
    """
    entities = [entity for entity in doc.entitydb.values() if entity.is_alive]
    _require(len(entities) <= 200000, 'Native inspection entity budget exceeded')
    counts = Counter(entity.dxftype() for entity in entities)
    resources = [{'handle': entity.dxf.handle, 'entity_type': entity.dxftype(),
                  'owner_handle': entity.dxf.get('owner')}
                 for entity in entities if entity.dxftype() in
                 ('IMAGE', 'IMAGEDEF', 'OLE2FRAME', 'OLEFRAME', 'PDFUNDERLAY', 'DWFUNDERLAY', 'DGNUNDERLAY',
                  'PDFDEFINITION', 'DWFDEFINITION', 'DGNDEFINITION', 'ACAD_PROXY_ENTITY', 'ACAD_PROXY_OBJECT')]
    xrefs = [{'name': block.name, 'block_handle': block.block.dxf.handle,
              'flags': block.block.dxf.flags, 'xref_path': block.block.dxf.get('xref_path', '')}
             for block in doc.blocks if block.block.dxf.flags & 12 or block.block.dxf.get('xref_path', '')]
    return {'scope': 'all_alive_entitydb_and_all_native_block_xref_flags',
            'entity_type_counts': dict(sorted(counts.items())), 'resource_entities': resources, 'xref_blocks': xrefs,
            'known_native_external_resources_absent': not resources and not xrefs,
            'wipeout_count': counts['WIPEOUT'], 'units_header_code': int(doc.units),
            'fonts_and_glyphs_verified': False, 'physical_scale_confirmed': False,
            'source_modified': False, 'annotation_approved': False, 'training_eligible': False}
