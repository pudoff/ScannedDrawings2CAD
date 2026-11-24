"""Separate dimension evidence fields without promoting CAD text into labels.

Syntax components are unit-neutral numbers read from displayed notation. Header
units, view scale, instance transforms and geometric attachment are separate
questions. In particular Unitless and metric-thread syntax do not assign mm to
native coordinates. Existing CAD audit measurements are retained, not repaired.
"""
from __future__ import annotations

from copy import deepcopy
import json
import re

from .notation import parse_notation

MAX_RECORD_BYTES = 2 * 1024**2
MAX_DISPLAY_TEXTS = 1000
STANDARD_KINDS = {'linear', 'aligned', 'diameter', 'radius', 'angular_two_lines',
                  'angular_three_points', 'ordinate'}


def _evidence(dimension):
    if not isinstance(dimension, dict):
        raise ValueError('Dimension evidence must be a JSON object')
    try:
        encoded = json.dumps(dimension, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError('Dimension evidence must contain finite JSON values') from error
    if len(encoded.encode('utf-8')) > MAX_RECORD_BYTES:
        raise ValueError('Dimension evidence exceeds its byte limit')
    handle, path = dimension.get('handle'), dimension.get('insert_path', [])
    if not isinstance(handle, str) or not handle or not isinstance(path, list) or any(not isinstance(p, str) or not p for p in path):
        raise ValueError('Dimension requires a handle and explicit instance path')
    raw = dimension.get('text_override')
    if raw is not None and not isinstance(raw, str):
        raise ValueError('Dimension override must be a string when present')
    stored = dimension.get('stored_display_texts', [])
    if not isinstance(stored, list) or len(stored) > MAX_DISPLAY_TEXTS:
        raise ValueError('Invalid stored display text inventory')
    if any(not isinstance(t, dict) or not isinstance(t.get('text'), str) for t in stored):
        raise ValueError('Stored display records require original text strings')
    return deepcopy(dimension)


def _syntax(raw, source):
    # Only well-defined text escapes are decoded. Formatting controls or stacked
    # tolerance syntax remain explicit unsupported text rather than being erased.
    decoded = re.sub(r'%%[cC]', 'Ø', raw)
    decoded = re.sub(r'%%[dD]', '°', decoded)
    decoded = re.sub(r'%%[pP]', '±', decoded)
    parsed = parse_notation(decoded)
    # X is also a case-sensitive fit designation (X7 differs from x7).
    # Preserve any valid original syntax, and only decode uppercase
    # multiplication in a recognized thread/chamfer/hole-count context.
    if parsed['status'] != 'parsed' and any(mark in decoded for mark in ('X', 'Х')):
        multiplied = decoded.replace('X', 'x').replace('Х', 'х')
        alternative = parse_notation(multiplied)
        if alternative['status'] == 'parsed' and (
                alternative['kind'] in ('thread', 'chamfer')
                or alternative['kind'] == 'diameter' and 'count' in alternative['values']):
            decoded, parsed = multiplied, alternative
    kind = {'nominal': 'scalar'}.get(parsed['kind'], parsed['kind'])
    components = {key.removesuffix('_mm'): value for key, value in parsed['values'].items()}
    base_kind = kind
    if parsed['status'] == 'parsed' and any(key in components for key in ('fit', 'tolerance_plus', 'tolerance_minus')) and kind != 'thread':
        kind = 'tolerance'
    if parsed['status'] != 'parsed':
        kind = 'other_complex' if any(mark in decoded for mark in ('±', '\\', '/', '°', '×', 'х', 'x', 'Х', 'X')) else 'unsupported'
    return {'source': source, 'raw': raw, 'decoded': decoded,
            'status': 'candidate' if parsed['status'] == 'parsed' else parsed['status'],
            'kind': kind, 'base_kind': base_kind, 'components': components,
            'number_units': 'unverified_display_units', 'issues': parsed['issues'],
            'annotation_approved': False}


def _same_text(a, b):
    if a['status'] == b['status'] == 'candidate':
        return a['kind'] == b['kind'] and a['base_kind'] == b['base_kind'] and a['components'] == b['components']
    return ' '.join(a['decoded'].split()) == ' '.join(b['decoded'].split())


def classify_dimension_observation(dimension: dict, *, units_code: int, pair_visual_status: str) -> dict:
    """Classify every valid audit record, retaining uncertainty field by field.

    ``candidate`` means recognized display syntax, including compound notation;
    it is never an accepted nominal, physical size, coordinate or CAD correction.
    An explicit override is compared with stored graphics, which may be stale.
    Multiple stored text entities are not silently selected or concatenated.
    """
    if isinstance(units_code, bool) or not isinstance(units_code, int) or not 0 <= units_code <= 24:
        raise ValueError('units_code must be an integer DXF unit code in 0..24')
    if pair_visual_status not in ('verified', 'quarantine', 'unknown', 'error'):
        raise ValueError('Invalid pair_visual_status')
    original = _evidence(dimension)
    handle = original['handle']
    path = original.get('insert_path', [])
    raw = original.get('text_override')
    mode = ('unresolved' if raw is None else 'suppressed' if raw == ' ' else
            'automatic' if raw in ('', '<>') else 'template' if '<>' in raw else 'explicit_override')
    reasons = []
    if original.get('text_mode', mode) != mode:
        reasons.append('recorded_text_mode_conflicts_with_raw_override')
    displays = original.get('stored_display_texts', [])
    candidates = [_syntax(item['text'], f'stored_display_texts[{index}]') for index, item in enumerate(displays)]
    override = _syntax(raw, 'text_override') if mode == 'explicit_override' else None
    if override:
        candidates.insert(0, override)
    selected = None
    if mode == 'suppressed':
        text_status = 'suppression_conflict' if displays else 'suppressed'
        if displays:
            reasons.append('suppressed_dimension_has_stored_text')
    elif len(displays) > 1:
        text_status = 'ambiguous'
        reasons.append('multiple_stored_text_entities_require_layout_review')
    elif override and displays and not _same_text(override, candidates[-1]):
        text_status = 'conflicting'
        reasons.append('override_and_stored_text_disagree_possible_stale_graphics')
    elif override:
        selected = override
        text_status = 'candidate' if displays else 'override_only'
        if not displays:
            reasons.append('explicit_override_has_no_stored_display_confirmation')
    elif displays:
        selected = candidates[0]
        text_status = 'candidate'
        reasons.append('stored_text_may_be_stale')
    else:
        text_status = 'unavailable'
        reasons.append('no_display_text_available')
    if 'recorded_text_mode_conflicts_with_raw_override' in reasons:
        text_status, selected = 'conflicting', None
    if selected is not None:
        notation = deepcopy(selected)
    else:
        notation = {'status': 'ambiguous' if text_status in ('ambiguous', 'conflicting', 'suppression_conflict') else 'unresolved',
                    'kind': mode if mode in ('automatic', 'template', 'suppressed') else 'unsupported',
                    'base_kind': None, 'components': {}, 'source': None,
                    'number_units': 'unverified_display_units', 'annotation_approved': False}
    units_status = 'declared_mm' if units_code == 4 else 'unknown_unitless' if units_code == 0 else 'other_declared_units'
    if units_code == 0:
        reasons.append('unitless_no_physical_unit_inference')
    if pair_visual_status != 'verified':
        reasons.append('pair_not_visually_verified')
    entity_supported = original.get('entity_type') == 'DIMENSION' and original.get('kind') in STANDARD_KINDS
    if not entity_supported:
        reasons.append('unsupported_dimension_subtype_or_missing_kind')
    if path:
        reasons.append('block_instance_transform_and_physical_scale_require_review')
    if mode == 'template':
        reasons.append('template_not_evaluated_or_regenerated')
    if notation['kind'] in ('thread', 'chamfer', 'tolerance', 'angle', 'scale'):
        reasons.append('compound_or_special_notation_requires_typed_geometry_review')
    points = original.get('definition_points_modelspace')
    if not isinstance(points, dict):
        points = {}
    required = {'linear': ('defpoint2', 'defpoint3'), 'aligned': ('defpoint2', 'defpoint3'),
                'diameter': ('defpoint', 'defpoint4'), 'radius': ('defpoint', 'defpoint4')}.get(original.get('kind'), ())
    def xyz(value):
        return isinstance(value, list) and len(value) == 3 and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)
    if not entity_supported or not required:
        anchor_status = 'unsupported'
    elif not all(name in points and xyz(points[name]) for name in required):
        anchor_status = 'not_available'
    else:
        anchor_status = 'provisional'
    comparison = deepcopy(original.get('display_comparison', {'status': 'unresolved'}))
    if not isinstance(comparison, dict):
        raise ValueError('Prior display comparison must be an object')
    numeric_status = comparison.get('status', 'unresolved')
    if numeric_status not in ('numeric_consistent', 'mismatch', 'unresolved'):
        numeric_status = 'unresolved'
        reasons.append('unrecognized_prior_numeric_comparison')
    return {'schema_version': '1.0',
            'identity': {'handle': handle, 'owner_block': original.get('owner_block'), 'insert_path': list(path)},
            'instance_key': json.dumps([*path, handle], ensure_ascii=False, separators=(',', ':')),
            'entity_type': original.get('entity_type'), 'native_kind': original.get('kind'),
            'text_mode': mode, 'notation': notation, 'text_candidates': candidates,
            'field_statuses': {'entity': 'native_observation' if entity_supported else 'unsupported_subtype',
                               'units': units_status, 'text': text_status, 'notation': notation['status'],
                               'numeric_consistency': numeric_status,
                               'instance': 'instance_review_required' if path else 'modelspace_observation',
                               'anchors': anchor_status, 'view_scale': 'not_reviewed',
                               'geometry_attachment': 'not_reviewed', 'physical_nominal': 'not_approved'},
            'units': {'code': units_code, 'status': units_status, 'source': 'DXF header only',
                      'external_evidence_verified': False, 'unit_conversion_applied': False},
            'numeric_consistency': {'source': 'prior_cad_semantics_audit', 'comparison': comparison,
                                    'meaning': 'Text/native-number consistency only; not validated geometry or physical size.'},
            'physical_nominal': {'status': 'not_approved', 'value': None, 'unit': None},
            'pair_visual_status': pair_visual_status, 'source_evidence': original,
            'review_reasons': sorted(set(reasons)), 'training_eligible': False,
            'annotation_approved': False, 'source_modified': False,
            'view_factor_applied': False, 'dimension_regenerated': False}
