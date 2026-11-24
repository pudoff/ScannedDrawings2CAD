"""Keep protection provenance while avoiding repeated own-head quantization.

The public caller re-observes the source and supplies its current chain/context.
Only duplicate coarse projections of that chain's two qualified native heads
can be replaced. Manual, TEXT, OCR and independent head hypotheses stay hard.
"""
from copy import deepcopy
import math

import cv2
import numpy as np

from .native_dimension_transfer_v1 import _sha, _supported_matrix
from .part_admission_v1 import object_sha
from .source_arrow_observation_v1 import checked_matrix, transform


POLICY = {'version': 'native-head-protection/1.0', 'maximum_working_pixels': 4_000_000,
    'maximum_roi_pixels': 4_000_000, 'maximum_protected_shapes': 2048,
    'maximum_protection_radius': 256, 'maximum_protection_raster_pixels': 64_000_000,
    'maximum_native_mask_pixels': 32_000_000, 'maximum_lifted_cell_pixels': 8_000_000}


class _Budget(ValueError):
    pass


def _polygon(value):
    points = np.asarray(value, float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
        raise ValueError('Finite protected polygon required')
    return points


def _uncertainty(value):
    uncertainty = float(value)
    if not math.isfinite(uncertainty) or uncertainty < 0:
        raise ValueError('Finite nonnegative protection uncertainty required')
    return uncertainty


def _paint(shape, points, radius):
    output = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(output, np.rint(points).astype(np.int32), 255)
    if radius:
        output = cv2.dilate(output, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                              (2*radius+1, 2*radius+1)))
    return output


def protection_ledger(shape, roles, ocr, arrows, heads, matrix, reviewed=None):
    """Reproduce the frozen working union, retaining each protective source."""
    matrix = checked_matrix(matrix)
    if (len(shape) != 2 or tuple(shape) != roles.shape or roles.ndim != 2
            or roles.dtype != np.uint8 or roles.size > POLICY['maximum_working_pixels']):
        raise ValueError('Bounded aligned uint8 role map required')
    if reviewed is not None and (reviewed.shape != roles.shape or reviewed.dtype != np.uint8):
        raise ValueError('Aligned uint8 reviewed protection required')
    before = object_sha([ocr, arrows, heads, matrix.tolist()])
    role_sha = _sha(roles); reviewed_sha = _sha(reviewed) if reviewed is not None else None
    inverse = np.linalg.inv(matrix)
    scale = float(np.linalg.svd(inverse[:2, :2], compute_uv=False).max())
    sources = []
    def add(kind, identity, row, mask, polygon=None, uncertainty=None):
        sources.append({'kind': kind, 'id': identity, 'row': deepcopy(row),
            'working_mask': mask, 'polygon_px': None if polygon is None else polygon.tolist(),
            'uncertainty_px': uncertainty})
    add('TEXT', 'known-TEXT', None, np.where(roles == 6, 255, 0).astype(np.uint8))
    add('manual', 'reviewed-protection', None, np.zeros_like(roles) if reviewed is None
        else np.where(reviewed > 0, 255, 0).astype(np.uint8))
    shapes = []
    for index, row in enumerate(ocr.get('lines', [])):
        shapes.append(('OCR', str(row.get('id', 'ocr/'+str(index))), row,
                       _polygon(row['polygon_px']), _uncertainty(row.get('uncertainty_px', 0.))))
    for rows, category in ((arrows.get('candidates', []), 'arrow'),
                           (heads.get('diagnostic_verified_candidates', []), 'diagnostic_native_head')):
        identities = [row['id'] for row in rows]
        if len(set(identities)) != len(identities):
            raise ValueError('Unique protective head identities required in each source domain')
        for row in rows:
            points = _polygon(row['polygon_px'])
            uncertainty = _uncertainty(row.get('uncertainty_px', 0.))
            if category == 'diagnostic_native_head':
                shapes.append((category, row['id'], row, transform(points, inverse), uncertainty*scale))
            else:
                evidence = row.get('evidence', {})
                kind = ('native_admitted_head' if row['id'].startswith('native/')
                    and evidence.get('original_head') and evidence.get('tip_admission') else 'working_arrow')
                shapes.append((kind, row['id'], row, points, uncertainty))
    if (len(shapes) > POLICY['maximum_protected_shapes']
            or len(shapes)*roles.size > POLICY['maximum_protection_raster_pixels']):
        raise _Budget('Protection ledger shape/raster budget exceeded')
    for kind, identity, row, points, uncertainty in shapes:
        radius = math.ceil(uncertainty+math.sqrt(.5))
        if radius > POLICY['maximum_protection_radius']:
            raise _Budget('Protection ledger uncertainty budget exceeded')
        add(kind, identity, row, _paint(shape, points, radius), points, uncertainty)
    mask = np.zeros_like(roles)
    for source in sources:
        mask |= source['working_mask']
    report = {'schema_version': 'native-head-protection-ledger/1.0', 'policy': deepcopy(POLICY),
        'roles_sha256': role_sha, 'reviewed_sha256': reviewed_sha,
        'ocr_sha256': object_sha(ocr), 'arrows_sha256': object_sha(arrows),
        'source_heads_sha256': object_sha(heads), 'source_sha256': heads.get('source_sha256'),
        'working_to_source': matrix.tolist(), 'mask_sha256': _sha(mask),
        'sources': [{'kind': s['kind'], 'id': s['id'], 'row_sha256': object_sha(s['row']),
            'working_mask_sha256': _sha(s['working_mask']),
            'working_pixels': int(np.count_nonzero(s['working_mask'])),
            'polygon_px': s['polygon_px'], 'uncertainty_px': s['uncertainty_px']} for s in sources],
        'original_working_union_preserved': True, 'roles_changed': 0,
        'source_head_uncertainty_reduced': False, 'physical_geometry_approved': False}
    if (before != object_sha([ocr, arrows, heads, matrix.tolist()]) or role_sha != _sha(roles)
            or reviewed_sha != (_sha(reviewed) if reviewed is not None else None)):
        raise ValueError('Protection ledger mutated its source inputs')
    return {'mask': mask, 'sources': sources, 'report': report}


def _same(a, b):
    try:
        x, y = np.asarray(a, float), np.asarray(b, float)
        return x.shape == y.shape and np.isfinite(x).all() and np.isfinite(y).all() and np.allclose(x, y, rtol=0, atol=1e-8)
    except (TypeError, ValueError):
        return False


def _qualify(identity, ledger, geometry, row, context, matrix, gray):
    """Match exact current provenance, never head proximity or appearance."""
    diagnostics = [s for s in ledger['sources'] if s['kind'] == 'diagnostic_native_head' and s['id'] == identity]
    projected = [s for s in ledger['sources'] if s['kind'] == 'native_admitted_head' and s['id'] == 'native/'+identity]
    if len(diagnostics) != 1 or len(projected) != 1:
        return None, 'unique_own_native_head_projection_missing'
    diagnostic, projected_source = diagnostics[0], projected[0]
    original, projected_head = diagnostic['row'], projected_source['row']
    evidence = projected_head.get('evidence', {}); prior = evidence.get('original_head', {})
    admission = evidence.get('tip_admission', {})
    source_sha = ledger['report']['source_sha256']; working_sha = geometry.get('source_sha256')
    if (not source_sha or not working_sha or not ledger.get('context_sha256')
            or original.get('source_ref', {}).get('sha256') != source_sha
            or original.get('source_ref', {}).get('coordinate_space') != 'original_image_pixels'
            or prior.get('id') != identity or prior.get('source_sha256') != source_sha
            or projected_head.get('source_ref', {}).get('sha256') != working_sha
            or projected_head.get('source_ref', {}).get('coordinate_space') != 'processed_image_pixels'
            or evidence.get('geometry_sha256') != object_sha(geometry)
            or evidence.get('source_heads_sha256') != ledger['report']['source_heads_sha256']):
        return None, 'own_head_source_or_geometry_provenance_differs'
    if (admission.get('source_arrow_id') != identity
            or admission.get('current_source_context_sha256') != ledger['context_sha256']
            or context['id'] not in admission.get('context_ids', [])
            or admission.get('source_box_px') != row['source_box_px']
            or admission.get('source_box_px') != context['source_box_px']
            or admission.get('source_window_sha256') != _sha(gray)
            or admission.get('current_local_recheck') != context.get('head_extension_bindings')
            or admission.get('measurement_coordinate_space') != 'original_image_pixels'
            or admission.get('original_owner_longitudinal_intervals_preserved') is not True
            or admission.get('semantic_dimension_approved') is not False
            or admission.get('physical_geometry_approved') is not False):
        return None, 'own_head_current_admission_differs'
    if (not _same(prior.get('tip_px'), original.get('tip_px'))
            or not _same(prior.get('polygon_px'), original.get('polygon_px'))
            or not _same(admission.get('original_tip_px'), original.get('tip_px'))):
        return None, 'original_head_observation_changed'
    groups = [g for g in context.get('head_extension_bindings', []) if g.get('head_id') == identity]
    options = [o for g in groups for o in g.get('extension_options', []) if o.get('source_shape_rechecked')]
    contacts = [c for c in row.get('contacts', []) if c.get('head_id') == identity]
    if len(groups) != 1 or len(options) != 1 or len(contacts) != 1:
        return None, 'unique_current_head_extension_missing'
    option, contact = options[0], contacts[0]; candidate = option.get('candidate') or {}
    candidate_ref = candidate.get('source_ref', {})
    if (candidate.get('id') != identity or candidate_ref.get('sha256') != source_sha
            or candidate_ref.get('coordinate_space') != 'source_window_pixels'
            or candidate_ref.get('box_original_px') != row['source_box_px']
            or option.get('extension_id') != admission.get('extension_id')
            or contact.get('extension_id') != option.get('extension_id')):
        return None, 'current_head_extension_identity_differs'
    origin = np.asarray(row['source_box_px'][:2], float)
    polygon = transform(projected_head['polygon_px'], matrix)-origin
    tip = transform(projected_head['tip_px'], matrix)-origin
    if (not _same(polygon, candidate.get('polygon_px')) or not _same(tip, candidate.get('tip_px'))
            or not _same(tip, contact.get('tip_px'))
            or not _same(tip+origin, admission.get('revised_tip_px'))):
        return None, 'current_head_source_shape_or_tip_differs'
    owners = candidate.get('evidence', {}).get('geometry_ids', [])
    if (len(owners) != 1 or set(owners) != set(original.get('evidence', {}).get('geometry_ids', []))
            or set(owners) != set(evidence.get('geometry_ids', []))
            or not set(owners) <= {r['id'] for r in row['carrier_observations']}):
        return None, 'current_head_shaft_ownership_differs'
    propagated = admission.get('uncertainty', {}).get('propagated_bound_px')
    inverse_scale = float(np.linalg.svd(np.linalg.inv(matrix)[:2, :2], compute_uv=False).max())
    if (propagated is None or isinstance(propagated, bool) or isinstance(projected_head['uncertainty_px'], bool)
            or not math.isfinite(float(propagated)) or float(propagated) < 0
            or not math.isclose(_uncertainty(projected_head['uncertainty_px']),
                                float(propagated)*inverse_scale, rel_tol=0, abs_tol=1e-8)):
        return None, 'current_head_uncertainty_differs'
    current_uncertainty = float(propagated)
    original_uncertainty = _uncertainty(original['uncertainty_px'])
    bound = _uncertainty(row['head_tolerance_px'])
    return {'identity': identity, 'sources': [diagnostic, projected_source],
        'polygons': [{'basis': 'original_diagnostic', 'polygon': _polygon(original['polygon_px'])-origin,
                      'uncertainty_px': max(original_uncertainty, bound)},
                     {'basis': 'current_admitted', 'polygon': _polygon(candidate['polygon_px']),
                      'uncertainty_px': max(current_uncertainty, bound)}],
        'original_uncertainty_px': original_uncertainty, 'current_uncertainty_px': current_uncertainty,
        'chain_head_tolerance_px': bound, 'owner_ids': owners}, None


def _lift(mask, matrix, origin, shape):
    output = np.zeros(shape, bool); yy, xx = np.nonzero(mask)
    centers = np.column_stack((xx, yy))@matrix[:2, :2].T+matrix[:2, 2]-origin
    half = np.abs(matrix[:2, :2]).sum(axis=1)/2
    low = np.ceil(centers-half-.5+1e-9).astype(int)
    high = np.floor(centers+half+.5-1e-9).astype(int)+1
    work = 0
    for a, b in zip(low, high):
        a = np.maximum(a, 0); b = np.minimum(b, shape[::-1])
        if (b <= a).any():
            continue
        work += int(np.prod(b-a))
        if work > POLICY['maximum_lifted_cell_pixels']:
            raise _Budget('Head protection lifted cell budget exceeded')
        output[a[1]:b[1], a[0]:b[0]] = True
    return output


def refine(gray, working_gray, geometry, roles, chain_row, context_row, matrix, *,
           protection, ledger, targets, ink, source_protection):
    """Return a source-only replacement; the published working mask is intact."""
    matrix = checked_matrix(matrix)
    if (gray.ndim != 2 or gray.dtype != np.uint8 or gray.size > POLICY['maximum_roi_pixels']
            or working_gray.shape != roles.shape or roles.dtype != np.uint8
            or protection.shape != roles.shape or protection.dtype != np.uint8
            or ink.shape != gray.shape or ink.dtype != bool
            or source_protection.shape != gray.shape or source_protection.dtype != bool):
        raise ValueError('Aligned bounded working/source protection maps required')
    if not _supported_matrix(matrix):
        raise ValueError('Supported source-cell registration required')
    if (ledger['report']['roles_sha256'] != _sha(roles)
            or ledger['report']['working_to_source'] != matrix.tolist()
            or ledger['report']['mask_sha256'] != _sha(protection)
            or not np.array_equal(ledger['mask'], protection)):
        raise ValueError('Current protection differs from its source ledger')
    current = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1] > 0
    if not np.array_equal(current, ink):
        raise ValueError('Source protection ink differs from current pixels')
    box = np.asarray(chain_row['source_box_px'], float)
    if (context_row['id'] != chain_row['id'] or context_row['source_box_px'] != chain_row['source_box_px']
            or tuple((box[2:]-box[:2])[::-1]) != gray.shape
            or chain_row.get('source_window_sha256') != _sha(gray)
            or context_row.get('source_window_sha256') != _sha(gray)):
        raise ValueError('Protection refinement source context differs')
    if chain_row.get('status') != 'source_linear_chain_observed' or not chain_row.get('geometry_chain_confirmed'):
        raise ValueError('Current geometric source chain required')
    before = object_sha([geometry, chain_row, context_row, ledger['report'], ledger.get('context_sha256')])
    hashes = [_sha(a) for a in (gray, working_gray, roles, protection, ink, source_protection)]
    source_hashes = [(s['kind'], s['id'], object_sha(s['row']), _sha(s['working_mask'])) for s in ledger['sources']]
    expected = [(s['kind'], s['id'], s['row_sha256'], s['working_mask_sha256']) for s in ledger['report']['sources']]
    if source_hashes != expected:
        raise ValueError('Protective source inventory changed after ledger construction')
    def unchanged():
        if (before != object_sha([geometry, chain_row, context_row, ledger['report'], ledger.get('context_sha256')])
                or hashes != [_sha(a) for a in (gray, working_gray, roles, protection, ink, source_protection)]
                or source_hashes != [(s['kind'], s['id'], object_sha(s['row']), _sha(s['working_mask'])) for s in ledger['sources']]):
            raise ValueError('Head protection refinement mutated its inputs')
    identities = chain_row['head_ids']
    if len(identities) != 2 or len(set(identities)) != 2:
        raise ValueError('Two different source heads required')
    qualified, failures = [], []
    for identity in identities:
        item, reason = _qualify(identity, ledger, geometry, chain_row, context_row, matrix, gray)
        if item is None:
            failures.append({'head_id': identity, 'reason': reason})
        else:
            qualified.append(item)
    # The pair is one geometric contract. A partial qualification cannot drop
    # either old head projection, even if the other one was individually valid.
    if failures:
        qualified = []
    removed = {id(source) for item in qualified for source in item['sources']}
    hard = np.zeros_like(protection)
    for source in ledger['sources']:
        if id(source) not in removed:
            hard |= source['working_mask']
    native = source_protection.copy(); polygons = []
    try:
        if 2*len(qualified)*gray.size > POLICY['maximum_native_mask_pixels']:
            raise _Budget('Native head mask budget exceeded')
        if any(math.ceil(shape['uncertainty_px']) > max(gray.shape)
               for item in qualified for shape in item['polygons']):
            raise _Budget('Native head uncertainty raster budget exceeded')
        for item in qualified:
            for shape in item['polygons']:
                radius = math.ceil(shape['uncertainty_px'])
                polygons.append({'head_id': item['identity'], 'basis': shape['basis'],
                    'polygon_px': shape['polygon'].tolist(), 'uncertainty_px': shape['uncertainty_px'],
                    'dilation_radius_px': radius})
                native |= _paint(gray.shape, shape['polygon'], radius) > 0
        lifted = _lift(hard, matrix, box[:2], gray.shape)
        baseline_native = source_protection | _lift(protection, matrix, box[:2], gray.shape)
        effective = native | lifted
    except _Budget as exc:
        unchanged()
        report = {'schema_version': 'native-head-protection-refinement/1.0', 'complete': False,
            'limits': [str(exc)], 'source_box_px': chain_row['source_box_px'],
            'source_window_sha256': _sha(gray), 'protection_sha256': _sha(protection),
            'ledger_sha256': object_sha(ledger['report']), 'qualified_head_ids': [],
            'qualification_failures': failures, 'replaced_duplicate_sources': [], 'native_polygons': [],
            'preserved_sources': [{'kind': s['kind'], 'id': s['id']} for s in ledger['sources']],
            'hard_protection_sha256': _sha(protection), 'native_protection_sha256': _sha(source_protection),
            'source_ink_released_from_projection_only': 0,
            'working_protection_pixels_delegated_to_native': 0,
            'original_working_union_preserved': True, 'intrinsic_source_protection_preserved': True,
            'head_uncertainty_reduced': False, 'roles_changed': 0,
            'physical_geometry_approved': False, 'partial_refinement_rolled_back': True}
        return {'hard_protection': protection.copy(), 'native_protection': source_protection.copy(),
                'report': report, 'complete': False}
    if np.any(source_protection & ~effective):
        raise ValueError('Intrinsic source protection was removed')
    report = {'schema_version': 'native-head-protection-refinement/1.0', 'complete': True,
        'source_box_px': chain_row['source_box_px'], 'source_window_sha256': _sha(gray),
        'working_to_source': matrix.tolist(), 'working_gray_sha256': _sha(working_gray),
        'roles_sha256': _sha(roles), 'protection_sha256': _sha(protection),
        'ledger_sha256': object_sha(ledger['report']), 'context_sha256': ledger.get('context_sha256'),
        'qualified_head_ids': [i['identity'] for i in qualified], 'qualification_failures': failures,
        'preserved_sources': [{'kind': s['kind'], 'id': s['id']} for s in ledger['sources'] if id(s) not in removed],
        'replaced_duplicate_sources': [{'kind': s['kind'], 'id': s['id']} for s in ledger['sources'] if id(s) in removed],
        'native_polygons': polygons, 'original_working_union_preserved': True,
        'hard_protection_sha256': _sha(hard), 'native_protection_sha256': _sha(native),
        'effective_source_protection_sha256': _sha(effective),
        'old_effective_source_protection_sha256': _sha(baseline_native),
        'source_ink_released_from_projection_only': int(np.count_nonzero(ink & baseline_native & ~effective)),
        'working_protection_pixels_delegated_to_native': int(np.count_nonzero((protection > 0) & (hard == 0))),
        'intrinsic_source_protection_preserved': True, 'head_uncertainty_reduced': False,
        'roles_changed': 0, 'physical_geometry_approved': False}
    unchanged()
    return {'hard_protection': hard, 'native_protection': native, 'report': report, 'complete': True}
