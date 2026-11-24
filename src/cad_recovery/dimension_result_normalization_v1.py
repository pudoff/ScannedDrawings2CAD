"""Normalize freshly admitted links; never interpret ink classes as semantic roles."""
import hashlib
from copy import deepcopy

import numpy as np

from .native_dimension_transfer_v1 import _footprint
from .source_arrow_observation_v1 import checked_matrix, source_windows


def arc_link(chain):
    origin = np.asarray(chain['source_box_px'][:2], float)
    tips = np.asarray([c['tip_px'] for c in chain['contacts']], float)+origin
    anchors = np.asarray(chain['anchors_px'], float)+origin
    return dict(tips=tips.tolist(), anchors=anchors.tolist(), dimension=[tips.tolist()],
                extensions=[[t.tolist(), a.tolist()] for t, a in zip(tips, anchors)])


def arc_mask(record, chain, native, before, ink, matrix):
    """Only actual newly published working cells can release their source ink.

    Deliberately conservative: unassigned source-body pixels are not filled in.
    Native partition values 1/2/3/4 have no role in this operation.
    """
    output = np.zeros(ink.shape, np.uint8)
    origin = np.asarray(chain['source_box_px'][:2])
    after, protected = native['roles'], native['protection']
    seen = set()
    for row in record['transfer']['rows']:
        role = row['role']
        if role not in (2, 3):
            raise ValueError('Unexpected published dimension role')
        for x, y in row['admitted_working_points']:
            if not (0 <= x < after.shape[1] and 0 <= y < after.shape[0]):
                raise ValueError('Admitted point outside working frame')
            if (after[y, x] != role or protected[y, x] or before[y, x] == role
                    or not (before[y, x] == 255 or (before[y, x] == 1 and role == 2))):
                continue
            identity = (x, y, role)
            if identity in seen:
                continue
            seen.add(identity)
            bounds = _footprint([x, y], matrix, origin, ink.shape)
            if bounds is None:
                raise ValueError('Published source footprint is clipped')
            l, t, r, b = bounds
            region = output[t:b, l:r]; actual = ink[t:b, l:r]
            if np.any(actual & (region != 0) & (region != role)):
                raise ValueError('Conflicting ARC role footprints')
            region[actual] = role
    return output


def merge_masks(records, box):
    """Merge only for display/scoring; keep individual links and masks as evidence."""
    x, y, right, bottom = box
    merged = np.zeros((bottom-y, right-x), np.uint8)
    for row, mask in records:
        l, t, r, b = row['source_box_px']
        if mask.dtype != np.uint8 or mask.shape != (b-t, r-l) or not set(np.unique(mask)) <= {0, 2, 3}:
            raise ValueError('Invalid semantic mask')
        a, c, d, e = max(x, l), max(y, t), min(right, r), min(bottom, b)
        if a >= d or c >= e:
            continue
        source = mask[c-t:e-t, a-l:d-l]
        target = merged[c-y:e-y, a-x:d-x]
        if np.any((source > 0) & (target > 0) & (source != target)):
            raise ValueError('Conflicting semantic roles across links')
        target[source > 0] = source[source > 0]
    return merged


def normalize(native, original_source, before, matrix, size):
    """Accept only the in-memory result of the current semantic publisher."""
    matrix = checked_matrix(matrix)
    report = native['report']; result = []; seen_ids = set(); seen_geometry = set()
    chains = {r['id']:r for r in native['chain_result']['report']['links']}
    requests = []
    for row in report['links']:
        if row['status'] == 'source_dimension_bodies_admitted':
            chain = chains[row['id']]
            if chain['status'] != 'source_linear_chain_observed':
                raise ValueError('Published role lacks current observed chain')
            requests.append(('arc_anchor', row, chain['source_box_px']))
    for row in report.get('source_short_linear_links', []):
        if row['status'] == 'source_short_linear_bodies_admitted':
            requests.append(('straight_anchor', row, row['source_box_px']))
    windows = source_windows(original_source, report['source_sha256'], size, [b for _, _, b in requests]) if requests else []
    for (branch, row, box), gray in zip(requests, windows):
        if row['id'] in seen_ids:
            raise ValueError('Duplicate accepted relation ID')
        seen_ids.add(row['id'])
        if branch == 'arc_anchor':
            chain = chains[row['id']]
            link = arc_link(chain)
            mask = arc_mask(row, chain, native, before, gray < 128, matrix)
        else:
            link = {k:deepcopy(row[k]) for k in ('tips', 'anchors', 'dimension', 'extensions')}
            mask = native['short_linear_roles'][row['id']].copy()
            if hashlib.sha256(mask.tobytes()).hexdigest() != row['source_roles_sha256']:
                raise ValueError('Short semantic mask hash differs')
        for key, shape in [('tips', (2, 2)), ('anchors', (2, 2))]:
            if np.asarray(link[key]).shape != shape or not np.isfinite(link[key]).all():
                raise ValueError('Two finite tips and anchors required')
        signature = tuple(sorted(tuple(t+a) for t, a in zip(link['tips'], link['anchors'])))
        if signature in seen_geometry:
            raise ValueError('Duplicate accepted geometric relation')
        seen_geometry.add(signature)
        if mask.shape != gray.shape or not set(np.unique(mask)) <= {0, 2, 3} or np.any((mask > 0) & (gray >= 128)):
            raise ValueError('Semantic role assigned outside current source ink')
        link.update(id=row['id'], branch=branch, source_box_px=box,
                    coordinate_space='input_source_pixel_centres',
                    role_counts={str(r):int(np.count_nonzero(mask == r)) for r in (2, 3)},
                    source_mask_basis=('published_working_cells_only' if branch == 'arc_anchor'
                                       else 'published_source_body_mask'))
        result.append((link, mask))
    return result
