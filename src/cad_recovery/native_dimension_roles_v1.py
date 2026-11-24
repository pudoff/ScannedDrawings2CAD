"""Publish source-bound dimension bodies only with corroborated prior PART anchors.

The original source is re-observed here. Saved chain certificates cannot grant
roles. Known PART revisions additionally require a checked automatic producer.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .dataset_layout_v1 import checked, read
from .grid_role_revision_v1 import check_authority, sha
from .native_arc_part_prior_v1 import observe as observe_prior
from .native_dimension_chain_v1 import inspect
from .native_dimension_transfer_v1 import bind
from .part_admission_v1 import object_sha
from .source_arrow_observation_v1 import checked_matrix, digest, source_windows, transform


POLICY = {'version': 'native-dimension-roles/1.0', 'maximum_protected_shapes': 2048,
          'maximum_protection_radius': 256, 'maximum_working_pixels': 4_000_000,
          'maximum_protection_raster_pixels': 64_000_000,
          'maximum_source_window_pixels': 32_000_000,
          'new_roles': [2, 3], 'known_role_revision': 'automatic_PART_to_DIM_body_only',
          'anchor_basis': 'corroborated_existing_PART_on_two_observed_ARC_flanks',
          'new_PART_classification': False, 'geometry_changed': False,
          'physical_geometry_approved': False, 'production_changed': False}


class _Limit(Exception):
    pass


def _protection(shape, roles, ocr, arrows, heads, matrix, reviewed):
    mask = np.where(roles == 6, 255, 0).astype(np.uint8)
    if reviewed is not None:
        if reviewed.shape != shape or reviewed.dtype != np.uint8:
            raise ValueError('Aligned uint8 reviewed protection required')
        mask[reviewed > 0] = 255
    shapes = [(row['polygon_px'], float(row.get('uncertainty_px', 0.)))
              for row in ocr.get('lines', []) + arrows.get('candidates', [])]
    inverse = np.linalg.inv(matrix)
    scale = float(np.linalg.svd(inverse[:2, :2], compute_uv=False).max())
    # Keep every diagnostic source head, even if it was not admitted as support.
    for row in heads.get('diagnostic_verified_candidates', []):
        shapes.append((transform(row['polygon_px'], inverse), float(row['uncertainty_px'])*scale))
    if len(shapes) > POLICY['maximum_protected_shapes']:
        raise _Limit('protection_shape_budget')
    if len(shapes)*int(np.prod(shape)) > POLICY['maximum_protection_raster_pixels']:
        raise _Limit('protection_raster_budget')
    for polygon, uncertainty in shapes:
        points = np.asarray(polygon, float)
        if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 3
                or not np.isfinite(points).all() or not math.isfinite(uncertainty) or uncertainty < 0):
            raise ValueError('Finite protected polygon and uncertainty required')
        radius = math.ceil(uncertainty + math.sqrt(.5))
        if radius > POLICY['maximum_protection_radius']:
            raise _Limit('protection_uncertainty_budget')
        local = np.zeros(shape, np.uint8)
        cv2.fillConvexPoly(local, np.rint(points).astype(np.int32), 255)
        if radius:
            local = cv2.dilate(local, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1)))
        mask |= local
    return mask


def _authority(gray, roles, authority, root, original_sha):
    permitted = check_authority(gray, roles, authority)
    if authority is None or not permitted.any():
        return permitted
    producer = read(checked(Path(root), authority['producer']))
    expected = {'schema_version': 'automatic-dimension-role-input/1.0',
                'working_gray_sha256': sha(gray), 'roles_sha256': sha(roles),
                'original_source_sha256': original_sha, 'reviewed_input': False}
    if any(producer.get(key) != value for key, value in expected.items()):
        raise ValueError('Automatic producer is not bound to this exact role input')
    if not producer.get('code') or not producer.get('input_role_file'):
        raise ValueError('Automatic producer code and role-file pins required')
    for pin in producer['code']:
        checked(Path(root), pin)
    with Image.open(checked(Path(root), producer['input_role_file'])) as image:
        current = np.asarray(image.convert('L'))
    if not np.array_equal(current, roles):
        raise ValueError('Automatic producer role-file differs from input roles')
    return permitted


def propose(original_source, working_image, *, geometry, ocr, roles, source_heads,
            working_to_source, arrows, authority=None, authority_root=None, protected=None):
    """Re-observe the source, bind complete pixel cells, then admit safe bodies."""
    matrix = checked_matrix(working_to_source)
    with Image.open(working_image) as image:
        gray = np.asarray(image.convert('L'))
    if gray.size > POLICY['maximum_working_pixels']:
        raise ValueError('Bounded working raster required')
    if roles.shape != gray.shape or roles.dtype != np.uint8:
        raise ValueError('Aligned uint8 input roles required')
    chain = inspect(original_source, working_image, geometry=geometry, ocr=ocr,
        roles=roles, source_heads=source_heads, working_to_source=matrix, arrows=arrows)
    original_sha = chain['report']['source_sha256']
    permitted = _authority(gray, roles, authority,
        authority_root or Path(__file__).resolve().parents[2], original_sha)
    before = object_sha([geometry, ocr, source_heads, arrows])
    role_hash = sha(roles)
    protection_hash = sha(protected) if protected is not None else None
    authority_hash = sha(authority['mask']) if authority is not None else None
    authority_metadata = object_sha({k: v for k, v in authority.items() if k != 'mask'}) if authority is not None else None
    output = roles.copy()
    proposals = np.full(roles.shape, 255, np.uint8)
    accepted = np.zeros(roles.shape, np.uint16)
    records, limits = [], []
    contexts = {c['id']: c for c in chain['admission']['context']['contexts']}
    eligible = [row for row in chain['report']['links'] if row['status'] == 'source_linear_chain_observed']
    if not chain['report']['local_observation_complete'] or not chain['report']['head_domain_complete']:
        limits.append('source_chain_domain_incomplete')
    if geometry.get('complete') is False or ocr.get('coverage', {}).get('complete') is False:
        limits.append('input_observation_domain_incomplete')
    boxes = list(dict.fromkeys(tuple(row['source_box_px']) for row in eligible))
    pixels = sum((b[2]-b[0])*(b[3]-b[1]) for b in boxes)
    if pixels > POLICY['maximum_source_window_pixels']:
        limits.append('source_role_window_budget')
    mask = np.zeros_like(roles)
    try:
        all_arrows = {'candidates': [*arrows.get('candidates', []), *chain['admission']['arrows']['candidates']]}
        mask = _protection(gray.shape, roles, ocr,
            all_arrows, source_heads, matrix, protected)
    except _Limit as exc:
        limits.append(str(exc))
    windows = source_windows(original_source, original_sha,
        source_heads['processed_size_px'], boxes) if boxes and not limits else []
    by_box = dict(zip(boxes, windows))
    if not limits:
        for row in eligible:
            source = by_box[tuple(row['source_box_px'])]
            context = contexts[row['id']]
            transfer = bind(source, gray, geometry, roles, row, context, matrix, protection=mask)
            prior = observe_prior(source, gray, geometry, roles, row, context, matrix, protection=mask)
            record = {'id': row['id'], 'status': 'withheld', 'reasons': [],
                      'source_head_ids': row['head_ids'], 'transfer': transfer['report'],
                      'anchor_prior': prior['report'], 'roles_changed': 0,
                      'physical_geometry_approved': False, 'complete_DIMENSION_created': False}
            records.append(record)
            if not transfer['report']['complete'] or not prior['complete']:
                limits.append('source_role_observation_incomplete')
                record['reasons'].append('source_role_observation_incomplete')
                continue
            proposed = transfer['proposed_roles']
            competing = (proposals != 255) & (proposed != 255) & (proposals != proposed)
            if competing.any():
                limits.append('competing_source_role_proposals')
            proposals[proposed != 255] = proposed[proposed != 255]
            if transfer['report'].get('holds'):
                record['reasons'].extend(transfer['report']['holds'])
                continue
            if not prior['accepted']:
                record['reasons'].append('existing_PART_anchor_not_corroborated')
                continue
            incompatible = ((proposed == 3) & ~np.isin(roles, [255, 3])) | (
                (proposed == 2) & ~np.isin(roles, [255, 1, 2]))
            if incompatible.any():
                record['reasons'].append('known_competing_role_on_bound_source_body')
                record['known_competing_pixels'] = int(incompatible.sum())
                continue
            selected = (proposed != 255) & (mask == 0)
            allowed = (roles == 255) | ((roles == 1) & (proposed == 2) & permitted)
            selected &= allowed
            if not selected.any():
                record['reasons'].append('no_authorized_source_body_pixels')
                continue
            for role in (2, 3):
                accepted[selected & (proposed == role)] |= 1 << role
            record.update(status='source_dimension_bodies_admitted',
                prior_basis=POLICY['anchor_basis'],
                source_interval_ids=[item['interval_id'] for item in transfer['report']['rows'] if item['admitted_working_points']],
                proposed_role_pixels={str(role): int(np.count_nonzero(proposed == role)) for role in (2, 3)},
                published_role_pixels={str(role): int(np.count_nonzero(selected & (proposed == role))) for role in (2, 3)},
                prior_PART_revision_pixels=int(np.count_nonzero(selected & (roles == 1))),
                roles_changed=int(np.count_nonzero(selected & (roles != proposed))))
    if limits:
        accepted[:] = 0
        for record in records:
            if record['status'] == 'source_dimension_bodies_admitted':
                record.update(status='withheld', roles_changed=0, published_role_pixels={'2': 0, '3': 0},
                    prior_PART_revision_pixels=0, reasons=['source_role_publication_incomplete'])
    for role in (2, 3):
        output[accepted == 1 << role] = role
    if ((accepted != 0) & ~np.isin(accepted, [1 << 2, 1 << 3])).any():
        raise ValueError('Competing dimension roles reached publication')
    changed = output != roles
    revised = changed & (roles != 255)
    if np.any(revised & ~((roles == 1) & (output == 2) & permitted)) or np.any(changed & (mask > 0)):
        raise ValueError('Source role transfer changed a protected or unauthorized role')
    if (before != object_sha([geometry, ocr, source_heads, arrows]) or sha(roles) != role_hash
            or digest(original_source) != original_sha or digest(working_image) != chain['report']['working_source_sha256']
            or (protected is not None and sha(protected) != protection_hash)
            or (authority is not None and sha(authority['mask']) != authority_hash)):
        raise ValueError('Source role inputs changed during observation')
    if authority is not None:
        if authority_metadata != object_sha({k: v for k, v in authority.items() if k != 'mask'}):
            raise ValueError('Role authority changed during observation')
        _authority(gray, roles, authority, authority_root or Path(__file__).resolve().parents[2], original_sha)
    report = {'schema_version': POLICY['version'], 'policy': deepcopy(POLICY),
        'scope': 'observed_source_chain_bodies_only',
        'source_sha256': original_sha, 'working_source_sha256': chain['report']['working_source_sha256'],
        'geometry_sha256': object_sha(geometry), 'roles_sha256': role_hash,
        'source_chain_sha256': object_sha(chain['report']), 'protection_sha256': sha(mask),
        'output_roles_sha256': sha(output), 'proposed_roles_sha256': sha(proposals),
        'authority': {key: deepcopy(value) for key, value in authority.items() if key != 'mask'} if authority else None,
        'working_to_source': matrix.tolist(), 'links': records, 'limits': limits, 'complete': not limits,
        'summary': dict(Counter(record['status'] for record in records)),
        'roles_changed': int(changed.sum()), 'prior_PART_revised': int(revised.sum()),
        'roles_added': {str(role): int(np.count_nonzero(changed & (output == role))) for role in (2, 3)},
        'new_PART_classification': False, 'geometry_changed': False, 'CAD_DIMENSION_created': False,
        'physical_geometry_approved': False, 'production_changed': False,
        'reference_used_for_inference': False, 'source_context_complete': chain['report']['source_context_complete']}
    return {'report': report, 'roles': output, 'proposed_roles': proposals,
            'protection': mask, 'chain_result': chain}
