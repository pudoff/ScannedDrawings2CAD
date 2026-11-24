"""Inventory possible source-ink participants without promoting head semantics.

A head halo protects its complete inherited uncertainty.  A nominal triangle
or an eroded conditional interior is only a geometric observation: the current
tip evidence does not calibrate the base or the whole triangle.  In particular,
a single possible claimant is never an exclusive semantic ownership proof.
"""
from copy import deepcopy
import math

import cv2
import numpy as np

from .native_dimension_transfer_v1 import _sha, _supported_matrix
from .part_admission_v1 import object_sha
from .source_arrow_observation_v1 import checked_matrix, transform


POLICY = {'version': 'native-head-membership/1.0',
    'maximum_roi_pixels': 4_000_000, 'maximum_working_pixels': 4_000_000,
    'maximum_claimants': 32, 'maximum_mask_pixels': 200_000_000,
    'maximum_support_points': 2_000_000, 'maximum_lifted_cell_pixels': 8_000_000,
    'maximum_radius_px': 256, 'maximum_rle_runs': 500_000,
    'possible_membership_is_not_exclusive_ownership': True,
    'conditional_core_is_not_whole_triangle_bound': True,
    'semantic_exclusive_head_confirmed': False, 'cad_modification': False}

PIXEL_RADIUS = math.sqrt(.5)


class _Limit(ValueError):
    pass


def _same(a, b):
    try:
        a, b = np.asarray(a, float), np.asarray(b, float)
        return (a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
                and np.allclose(a, b, rtol=0, atol=1e-8))
    except (TypeError, ValueError):
        return False


def _nonnegative(value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError('Finite nonnegative membership uncertainty required')
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError('Finite nonnegative membership uncertainty required') from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError('Finite nonnegative membership uncertainty required')
    return number


def _triangle(value):
    points = np.asarray(value, float)
    if points.shape != (3, 2) or not np.isfinite(points).all():
        raise ValueError('Finite native head triangle required')
    area = abs(float(np.linalg.det(np.stack((points[1]-points[0], points[2]-points[0])))))
    if area <= 1e-8:
        raise ValueError('Nondegenerate native head triangle required')
    return points


def _unique(rows, label):
    result = {}
    for row in rows:
        identity = row.get('id')
        if not isinstance(identity, str) or not identity or identity in result:
            raise ValueError('Unique '+label+' identities required')
        result[identity] = row
    return result


def _checked_heads(geometry, chain, context, matrix, refinement, source_heads, admitted_heads):
    identities = chain['head_ids']
    if len(identities) != 2 or len(set(identities)) != 2:
        raise ValueError('Two different current source heads required')
    if (refinement.get('complete') is not True
            or refinement.get('qualified_head_ids') != identities
            or refinement.get('qualification_failures')
            or refinement.get('intrinsic_source_protection_preserved') is not True
            or refinement.get('original_working_union_preserved') is not True
            or refinement.get('roles_changed') != 0
            or refinement.get('head_uncertainty_reduced') is not False):
        raise ValueError('Complete current pair head protection refinement required')
    originals = _unique(source_heads.get('diagnostic_verified_candidates', []), 'diagnostic head')
    admitted = _unique(admitted_heads.get('candidates', []), 'admitted head')
    polygons = refinement.get('native_polygons', [])
    if len(polygons) != 4:
        raise ValueError('Both original and current refined head polygons required')
    source_sha = source_heads.get('source_sha256')
    working_sha = geometry.get('source_sha256')
    if not source_sha or not working_sha or not refinement.get('context_sha256'):
        raise ValueError('Current original, working and context source identities required')
    origin = np.asarray(chain['source_box_px'][:2], float)
    inverse_scale = float(np.linalg.svd(np.linalg.inv(matrix)[:2, :2], compute_uv=False).max())
    source_heads_sha, geometry_sha = object_sha(source_heads), object_sha(geometry)
    owners = {r['id'] for r in chain['carrier_observations']}
    output = []
    for identity in identities:
        if identity not in originals or 'native/'+identity not in admitted:
            raise ValueError('Qualified native head observation missing')
        old, current = originals[identity], admitted['native/'+identity]
        evidence = current.get('evidence', {})
        prior, admission = evidence.get('original_head', {}), evidence.get('tip_admission', {})
        if (old.get('source_ref', {}).get('sha256') != source_sha
                or old.get('source_ref', {}).get('coordinate_space') != 'original_image_pixels'
                or prior.get('id') != identity or prior.get('source_sha256') != source_sha
                or current.get('source_ref', {}).get('sha256') != working_sha
                or current.get('source_ref', {}).get('coordinate_space') != 'processed_image_pixels'
                or evidence.get('geometry_sha256') != geometry_sha
                or evidence.get('source_heads_sha256') != source_heads_sha
                or not _same(prior.get('tip_px'), old.get('tip_px'))
                or not _same(prior.get('polygon_px'), old.get('polygon_px'))):
            raise ValueError('Current head source or geometry provenance differs')
        if (admission.get('source_arrow_id') != identity
                or admission.get('current_source_context_sha256') != refinement['context_sha256']
                or context['id'] not in admission.get('context_ids', [])
                or admission.get('source_box_px') != chain['source_box_px']
                or admission.get('source_window_sha256') != chain['source_window_sha256']
                or admission.get('current_local_recheck') != context.get('head_extension_bindings')
                or admission.get('measurement_coordinate_space') != 'original_image_pixels'
                or admission.get('original_owner_longitudinal_intervals_preserved') is not True
                or admission.get('semantic_dimension_approved') is not False
                or admission.get('physical_geometry_approved') is not False
                or not _same(admission.get('original_tip_px'), old.get('tip_px'))):
            raise ValueError('Current native head admission differs')
        groups = [g for g in context.get('head_extension_bindings', []) if g.get('head_id') == identity]
        options = [o for g in groups for o in g.get('extension_options', []) if o.get('source_shape_rechecked')]
        contacts = [c for c in chain.get('contacts', []) if c.get('head_id') == identity]
        if len(groups) != 1 or len(options) != 1 or len(contacts) != 1:
            raise ValueError('Unique current head extension and contact required')
        option, contact = options[0], contacts[0]
        candidate = option.get('candidate') or {}
        polygon, tip = transform(current['polygon_px'], matrix)-origin, transform(current['tip_px'], matrix)-origin
        if (candidate.get('id') != identity
                or candidate.get('source_ref', {}).get('sha256') != source_sha
                or candidate.get('source_ref', {}).get('coordinate_space') != 'source_window_pixels'
                or candidate.get('source_ref', {}).get('box_original_px') != chain['source_box_px']
                or not _same(polygon, candidate.get('polygon_px'))
                or not _same(tip, candidate.get('tip_px')) or not _same(tip, contact.get('tip_px'))
                or not _same(tip+origin, admission.get('revised_tip_px'))
                or option.get('extension_id') != contact.get('extension_id')
                or option.get('extension_id') != admission.get('extension_id')):
            raise ValueError('Current head source shape or extension identity differs')
        candidate_owners = candidate.get('evidence', {}).get('geometry_ids', [])
        if (len(candidate_owners) != 1 or not set(candidate_owners) <= owners
                or set(candidate_owners) != set(old.get('evidence', {}).get('geometry_ids', []))
                or set(candidate_owners) != set(evidence.get('geometry_ids', []))):
            raise ValueError('Current head carrier owner differs')
        old_bound = _nonnegative(old['uncertainty_px'])
        own_bound = _nonnegative(admission.get('uncertainty', {}).get('propagated_bound_px'))
        if not math.isclose(_nonnegative(current['uncertainty_px']), own_bound*inverse_scale,
                            rel_tol=0, abs_tol=1e-8):
            raise ValueError('Current projected head uncertainty differs')
        old_polygon = _triangle(old['polygon_px'])-origin
        current_polygon = _triangle(candidate['polygon_px'])
        shapes = []
        for basis, expected, own in (('original_diagnostic', old_polygon, old_bound),
                                     ('current_admitted', current_polygon, own_bound)):
            matches = [r for r in polygons if r.get('head_id') == identity and r.get('basis') == basis]
            if len(matches) != 1:
                raise ValueError('Unique refined native head polygon required')
            shape = matches[0]
            bound = max(own, _nonnegative(chain['head_tolerance_px']))
            if (not _same(shape.get('polygon_px'), expected)
                    or not math.isclose(_nonnegative(shape['uncertainty_px']), bound, rel_tol=0, abs_tol=1e-8)
                    or shape.get('dilation_radius_px') != math.ceil(bound)):
                raise ValueError('Native head protection geometry or uncertainty differs')
            # Rasterize the exact native vertices that produced the refinement.
            # A harmless affine round trip can cross an x.5 rounding tie.
            shapes.append({'basis': basis, 'polygon': _triangle(shape['polygon_px']), 'own_bound': own,
                           'protection_bound': bound, 'radius': math.ceil(bound)})
        output.append({'id': identity, 'shapes': shapes, 'carrier_ids': candidate_owners,
                       'extension_id': contact['extension_id']})
    return output


def _paint(shape, polygon):
    mask = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(mask, np.rint(polygon).astype(np.int32), 1)
    return mask


def _strip(shape, points, radius):
    a, b = np.asarray(points, float)
    delta = b-a
    length = float(np.linalg.norm(delta))
    if not math.isfinite(length) or length <= 0:
        raise ValueError('Nondegenerate measured source line required')
    unit = delta/length
    y, x = np.ogrid[:shape[0], :shape[1]]
    along = (x-a[0])*unit[0]+(y-a[1])*unit[1]
    across = abs((x-a[0])*(-unit[1])+(y-a[1])*unit[0])
    return (along >= 0) & (along <= length) & (across <= radius)


def _lift(mask, matrix, origin, shape):
    """Lift every intersected working pixel cell; do not sample cell centres."""
    output = np.zeros(shape, bool)
    height, width = shape
    corners = np.array([[-.5, -.5], [width-.5, -.5], [width-.5, height-.5], [-.5, height-.5]])+origin
    working = transform(corners, np.linalg.inv(matrix))
    low = np.maximum(0, np.floor(working.min(0)-.5).astype(int))
    high = np.minimum(mask.shape[::-1], np.ceil(working.max(0)+.5).astype(int)+1)
    if (high <= low).any():
        return output
    y, x = np.nonzero(mask[low[1]:high[1], low[0]:high[0]])
    y, x = y+low[1], x+low[0]
    centers = np.column_stack((x, y))@matrix[:2, :2].T+matrix[:2, 2]-origin
    half = np.abs(matrix[:2, :2]).sum(axis=1)/2
    low = np.ceil(centers-half-.5+1e-9).astype(int)
    high = np.floor(centers+half+.5-1e-9).astype(int)+1
    work = 0
    for start, stop in zip(low, high):
        start, stop = np.maximum(start, 0), np.minimum(stop, shape[::-1])
        if (stop <= start).any():
            continue
        work += int(np.prod(stop-start))
        if work > POLICY['maximum_lifted_cell_pixels']:
            raise _Limit('independent_protection_lift_budget')
        output[start[1]:stop[1], start[0]:stop[0]] = True
    return output


def _rle(ink, possible, nominal, core):
    rows = []
    for y in range(ink.shape[0]):
        xs = np.flatnonzero(ink[y])
        if not len(xs):
            continue
        split = ((np.diff(xs) != 1) | (np.diff(possible[y, xs]) != 0)
                 | (np.diff(nominal[y, xs]) != 0) | (np.diff(core[y, xs]) != 0))
        bounds = np.r_[0, np.flatnonzero(split)+1, len(xs)]
        if len(rows)+len(bounds)-1 > POLICY['maximum_rle_runs']:
            raise _Limit('source_membership_rle_budget')
        for start, stop in zip(bounds[:-1], bounds[1:]):
            x = int(xs[start])
            rows.append([y, x, int(xs[stop-1])+1, int(possible[y, x]),
                         int(nominal[y, x]), int(core[y, x])])
    return rows


def observe(gray, working_gray, geometry, roles, chain_row, context_row, matrix, *,
            head_refinement, hard_protection, source_heads, admitted_heads):
    """Return a bounded, disjoint source-ink census; never alter CAD or roles."""
    matrix = checked_matrix(matrix)
    if not _supported_matrix(matrix):
        raise ValueError('Supported source-cell registration required')
    if (gray.ndim != 2 or gray.dtype != np.uint8
            or working_gray.ndim != 2 or working_gray.dtype != np.uint8
            or working_gray.shape != roles.shape or roles.dtype != np.uint8
            or hard_protection.shape != roles.shape or hard_protection.dtype != np.uint8):
        raise ValueError('Aligned uint8 source, working, role and hard protection maps required')
    refinement = head_refinement.get('report', head_refinement)
    source_sha = _sha(gray)
    box = np.asarray(chain_row['source_box_px'], float)
    if (box.shape != (4,) or not np.isfinite(box).all()
            or not np.array_equal(box, np.rint(box)) or (box[:2] < 0).any()
            or tuple((box[2:]-box[:2])[::-1]) != gray.shape
            or chain_row.get('id') != context_row.get('id')
            or context_row.get('source_box_px') != chain_row['source_box_px']
            or refinement.get('source_box_px') != chain_row['source_box_px']
            or any(row.get('source_window_sha256') != source_sha for row in (chain_row, context_row, refinement))
            or refinement.get('working_to_source') != matrix.tolist()
            or refinement.get('working_gray_sha256') != _sha(working_gray)
            or refinement.get('roles_sha256') != _sha(roles)
            or refinement.get('hard_protection_sha256') != _sha(hard_protection)):
        raise ValueError('Current membership source, registration or protection differs')
    if (chain_row.get('status') != 'source_linear_chain_observed'
            or chain_row.get('geometry_chain_confirmed') is not True):
        raise ValueError('Current observed source chain required')
    structures = [geometry, chain_row, context_row, refinement, source_heads, admitted_heads]
    arrays = [gray, working_gray, roles, hard_protection, matrix]
    before = object_sha(structures), [_sha(a) for a in arrays]
    heads = _checked_heads(geometry, chain_row, context_row, matrix, refinement, source_heads, admitted_heads)
    measured = _unique(context_row.get('stroke_observation', {}).get('lines', []), 'measured line')
    if context_row.get('stroke_observation', {}).get('complete') is not True:
        raise ValueError('Complete current source stroke observations required')
    carrier_ids = {r['id'] for r in chain_row['carrier_observations']}
    extension_ids = {r['extension_id'] for r in chain_row['contacts']}
    if not carrier_ids or not (carrier_ids | extension_ids) <= measured.keys() or carrier_ids & extension_ids:
        raise ValueError('Distinct current measured carriers and extensions required')
    report = {'schema_version': POLICY['version'], 'policy': deepcopy(POLICY),
        'complete': False, 'limits': [], 'source_context_id': chain_row['id'],
        'source_box_px': list(chain_row['source_box_px']), 'source_window_sha256': source_sha,
        'original_source_sha256': source_heads.get('source_sha256'),
        'working_source_sha256': geometry.get('source_sha256'),
        'working_to_source': matrix.tolist(), 'working_gray_sha256': _sha(working_gray),
        'roles_sha256': _sha(roles), 'hard_protection_sha256': _sha(hard_protection),
        'geometry_sha256': object_sha(geometry), 'source_heads_sha256': object_sha(source_heads),
        'admitted_heads_sha256': object_sha(admitted_heads), 'chain_sha256': object_sha(chain_row),
        'context_sha256': object_sha(context_row), 'head_refinement_sha256': object_sha(refinement),
        'qualified_head_ids': list(chain_row['head_ids']), 'claimants': [], 'heads': [],
        'classes_rle': [], 'rle_columns': ['y', 'x0', 'x1', 'possible_bits', 'nominal_head_bits', 'conditional_core_bits'],
        'rle_coordinate_space': 'source_window_pixels', 'rle_white_pixels_excluded': True,
        'semantic_exclusive_head_confirmed': False, 'conditional_core_semantics_withheld': True,
        'possible_membership_is_not_exclusive_ownership': True, 'roles_changed': 0,
        'cad_modification': False, 'cad_modification_performed': False,
        'physical_geometry_approved': False, 'head_uncertainty_reduced': False,
        'original_working_union_preserved': True, 'partial_census_rolled_back': False}

    def unchanged():
        if before != (object_sha(structures), [_sha(a) for a in arrays]):
            raise ValueError('Native head membership mutated its source inputs')

    try:
        if gray.size > POLICY['maximum_roi_pixels']:
            raise _Limit('source_membership_roi_budget')
        if roles.size > POLICY['maximum_working_pixels']:
            raise _Limit('source_membership_working_budget')
        count = len(heads)+len(measured)+1
        if count > min(32, POLICY['maximum_claimants']):
            raise _Limit('source_membership_claimant_budget')
        # Preflight all full-ROI model masks before allocating the census planes.
        if gray.size*(12*len(heads)+len(measured)+5) > POLICY['maximum_mask_pixels']:
            raise _Limit('source_membership_mask_budget')
        if sum(len(line.get('support_points', [])) for line in measured.values()) > POLICY['maximum_support_points']:
            raise _Limit('source_membership_support_points_budget')
        for head in heads:
            for shape in head['shapes']:
                radius = shape['radius']
                own_radius = math.ceil(shape['own_bound']+PIXEL_RADIUS)
                if max(radius, own_radius) > POLICY['maximum_radius_px']:
                    raise _Limit('source_head_radius_budget')
                points = shape['polygon']
                if (np.floor(points.min(0))-radius < 0).any() or (
                        np.ceil(points.max(0))+radius >= np.asarray(gray.shape[::-1])).any():
                    raise _Limit('head_possible_guard_clipped')
        ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1] > 0
        possible = np.zeros(gray.shape, np.uint64)
        nominal = np.zeros_like(possible)
        core = np.zeros_like(possible)
        head_diagnostics = []

        def claimant(kind, identity, mask, **extra):
            bit = 1 << len(report['claimants'])
            possible[ink & mask] |= np.uint64(bit)
            report['claimants'].append({'index': len(report['claimants']), 'bit': bit,
                'kind': kind, 'id': identity, 'source_ink_pixels': int(np.count_nonzero(ink & mask)), **extra})
            return bit

        for head in heads:
            fills, eroded, halo = [], [], np.zeros(gray.shape, bool)
            for shape in head['shapes']:
                fill = _paint(gray.shape, shape['polygon'])
                radius = shape['radius']
                halo |= cv2.dilate(fill, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                    (2*radius+1, 2*radius+1))) > 0
                own_radius = math.ceil(shape['own_bound']+PIXEL_RADIUS)
                inside = cv2.erode(fill, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                    (2*own_radius+1, 2*own_radius+1)), borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
                fills.append(fill > 0)
                eroded.append(inside)
            bit = claimant('head_possible', head['id'], halo,
                carrier_ids=list(head['carrier_ids']), extension_id=head['extension_id'],
                shape_bounds=[{'basis': s['basis'], 'protection_uncertainty_px': s['protection_bound'],
                    'dilation_radius_px': s['radius'], 'conditional_erosion_radius_px': math.ceil(s['own_bound']+PIXEL_RADIUS),
                    'own_observation_bound_px': s['own_bound']} for s in head['shapes']])
            observed = fills[0] & fills[1] & ink
            conditional = eroded[0] & eroded[1] & ink
            nominal[observed] |= np.uint64(bit)
            core[conditional] |= np.uint64(bit)
            head_diagnostics.append((head, bit, halo, fills, conditional))

        for identity, line in sorted(measured.items()):
            points = np.asarray(line['points_px'], float)
            width, uncertainty = float(line['width_px']), _nonnegative(line['uncertainty_px'])
            if (points.shape != (2, 2) or not np.isfinite(points).all()
                    or not math.isfinite(width) or width <= 0):
                raise ValueError('Finite current measured source line required')
            profile = line.get('source_profile', {})
            if (profile.get('input_pixels_sha256') != source_sha
                    or profile.get('measurement_coordinate_space') != 'source_window_pixels'
                    or profile.get('source_pixels_changed') != 0
                    or profile.get('support_samples_are_actual_ink') is not True
                    or profile.get('roles_inferred') is not False
                    or profile.get('extended_support_ownership_confirmed') is not False):
                raise ValueError('Current source line profile provenance differs')
            support = np.asarray(line.get('support_points', []), float)
            if (support.ndim != 2 or support.shape[1] != 2 or not len(support)
                    or not np.isfinite(support).all() or not np.array_equal(support, np.rint(support))
                    or (support < 0).any() or (support >= np.asarray(gray.shape[::-1])).any()):
                raise ValueError('Finite current native ink support points required')
            support = support.astype(int)
            if not np.all(ink[support[:, 1], support[:, 0]]):
                raise ValueError('Measured source line support is not current ink')
            kind = ('measured_carrier' if identity in carrier_ids else
                    'measured_extension' if identity in extension_ids else 'other_measured_line')
            radius = width/2+uncertainty+PIXEL_RADIUS
            if radius > POLICY['maximum_radius_px']:
                raise _Limit('source_line_radius_budget')
            claimant(kind, identity, _strip(gray.shape, points, radius),
                measured_points_px=points.tolist(), width_px=width,
                uncertainty_px=uncertainty, possible_radius_px=radius,
                observed_extent_is_not_original_ownership_extent=True)
        hard = _lift(hard_protection > 0, matrix, box[:2], gray.shape)
        claimant('hard_independent_protection', 'hard-independent-protection', hard,
                 own_pair_duplicate_projections_removed_only=True)

        def bits(kind):
            return np.uint64(sum(c['bit'] for c in report['claimants'] if c['kind'] == kind))

        head_bits, carrier_bits = bits('head_possible'), bits('measured_carrier')
        extension_bits, other_bits = bits('measured_extension'), bits('other_measured_line')
        hard_bits = bits('hard_independent_protection')
        for head, bit, halo, fills, conditional in head_diagnostics:
            own = np.uint64(bit)
            only = possible == own
            # Even these retained pixels are conditional geometric observations,
            # not a proof that the real head owns the complete triangle or base.
            core[(core & own != 0) & ~only] &= ~own
            h = ink & halo
            report['heads'].append({'head_id': head['id'], 'bit': bit,
                'possible_ink_pixels': int(np.count_nonzero(h)),
                'nominal_intersection_ink_pixels': int(np.count_nonzero(ink & fills[0] & fills[1])),
                'raw_current_nominal_disagreement_ink_pixels': int(np.count_nonzero(ink & (fills[0] ^ fills[1]))),
                'uncertainty_or_prior_disagreement_ink_pixels': int(np.count_nonzero(h & ~(fills[0] & fills[1]))),
                'geometric_observed_only_nominal_wing_pixels': int(np.count_nonzero(only & (nominal & own != 0))),
                'conditional_eroded_intersection_ink_pixels': int(np.count_nonzero(conditional)),
                'conditional_geometric_observed_only_core_pixels': int(np.count_nonzero(core & own)),
                'shared_carrier_ink_pixels': int(np.count_nonzero(h & (possible & carrier_bits != 0))),
                'shared_extension_ink_pixels': int(np.count_nonzero(h & (possible & extension_bits != 0))),
                'shared_other_measured_line_ink_pixels': int(np.count_nonzero(h & (possible & other_bits != 0))),
                'shared_independent_protection_ink_pixels': int(np.count_nonzero(h & (possible & hard_bits != 0))),
                'shared_other_head_ink_pixels': int(np.count_nonzero(h & (possible & (head_bits ^ own) != 0))),
                'conditional_core_is_not_whole_triangle_bound': True,
                'semantic_exclusive_head_confirmed': False})
        runs = _rle(ink, possible, nominal, core)
        encoded = sum(row[2]-row[1] for row in runs)
        if encoded != int(ink.sum()):
            raise ValueError('Membership source ink census is not complete')
        multi = (possible != 0) & ((possible & (possible-np.uint64(1))) != 0)
        report.update(complete=True, classes_rle=runs, classes_rle_sha256=object_sha(runs),
            counts={'source_ink_pixels': int(ink.sum()), 'encoded_source_ink_pixels': encoded,
                'unexplained_possible_bits_zero_ink_pixels': int(np.count_nonzero(ink & (possible == 0))),
                'multiple_possible_claimant_ink_pixels': int(np.count_nonzero(ink & multi)),
                'possible_head_ink_pixels': int(np.count_nonzero(ink & (possible & head_bits != 0))),
                'nominal_head_ink_pixels': int(np.count_nonzero(nominal)),
                'conditional_geometric_observed_only_core_ink_pixels': int(np.count_nonzero(core)),
                'rle_runs': len(runs)}, source_ink_partition_complete=True,
            whole_triangle_ownership_confirmed=False, complete_cad_head_approved=False)
    except _Limit as exc:
        report.update(complete=False, limits=[str(exc)], classes_rle=[], heads=[], claimants=[],
            partial_census_rolled_back=True, source_ink_partition_complete=False)
    unchanged()
    return {'report': report, 'complete': report['complete']}
