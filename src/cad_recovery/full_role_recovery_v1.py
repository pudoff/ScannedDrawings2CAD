"""Reconcile source-backed role proposals in one native pixel frame."""
import cv2
import numpy as np

from .native_section_roles_v1 import axis_strokes
from .part_contact_v1 import bind_source
from .stroke_roles_v1 import checked_gray, foreground
from .supported_role_overlay_v1 import source_ownership


def observed_part_core(gray, graph, candidate, *, accepted=None):
    """Keep agreement only on unambiguous measured line or curve support."""
    if candidate.shape != gray.shape or candidate.dtype != bool:
        raise ValueError('Aligned native contour proposal required')
    if accepted is not None and (accepted.shape != gray.shape or accepted.dtype != bool):
        raise ValueError('Aligned accepted contour anchors required')
    if graph.get('search_incomplete') or not candidate.any():
        return np.zeros_like(candidate)
    items = graph['proposals']
    evidence, work = source_ownership(foreground(gray), items)
    if evidence is None or work['incomplete']:
        return np.zeros_like(candidate)
    owners = evidence['owner']
    radii = [0.]+[p['width_px']/2+np.sqrt(.5) for p in items]
    primitive = np.array([False]+[p['parent_kind'] in ('line', 'arc', 'circle') for p in items])
    allowed = np.array([False]+[p['role'] == 'part_contour' or (p['role'] == 'unknown' and typed)
                                for p, typed in zip(items, primitive[1:])])
    numbered = np.maximum(owners, 0)
    observed = (owners > 0) & (evidence['distance'] <= np.take(radii, numbered))
    if accepted is not None:
        area = np.bincount(owners[observed], minlength=len(allowed))
        support = np.bincount(owners[observed & accepted], minlength=len(allowed))
        allowed &= primitive & (support >= 32) & (support >= .7*area)
    return candidate & observed & allowed[numbered]


def reconcile(gray, baseline, protected, *, part, stamp, independent=()):
    """Fill unknown ink when the available positive witnesses agree on one role.

    Independent role maps must already be projected from verified views of
    this same native source. Their intersection is an additional proposal.
    It cannot overrule current roles, explicit protection or a source witness.
    """
    checked_gray(gray)
    roles = (baseline, *independent)
    if any(a.shape != gray.shape or a.dtype != np.uint8 or not np.isin(a, [*range(10), 255]).all() for a in roles):
        raise ValueError('Aligned role maps required')
    if any(a.shape != gray.shape or a.dtype != bool for a in (protected, part, stamp)):
        raise ValueError('Aligned native source masks required')
    if independent and len(independent) < 2:
        raise ValueError('Independent proposal requires two observations')

    eligible = (foreground(gray) > 0) & (baseline == 255) & ~protected
    # A 9-bit set preserves all contenders until conflicts can be resolved.
    choices = np.zeros(gray.shape, np.uint16)
    choices[part] |= 1 << 1
    choices[stamp] |= 1 << 8
    if independent:
        first = independent[0]
        agreement = np.logical_and.reduce([a == first for a in independent[1:]])
        for role in range(1, 10):
            choices[agreement & (first == role)] |= 1 << role

    output = baseline.copy()
    for role in range(1, 10):
        output[eligible & (choices == 1 << role)] = role
    conflict = eligible & ((choices & (choices - 1)) != 0)
    return {'roles': output, 'conflict': conflict,
            'new_assignments': int((output != baseline).sum())}


def corroborate_views(gray, baseline, protected, *, source_pin, receiver, donor):
    """Recover an agreed role in the common original-pixel frame."""
    if receiver['gray'].shape == donor['gray'].shape:
        raise ValueError('Distinct verified raster resolutions required')
    for observation in (receiver, donor):
        proof = observation['binding']
        actual = bind_source(observation['gray'], gray, proof['matrix'], source_pin,
                             resize_method=proof['resize_method'])
        if actual != proof:
            raise ValueError('Role observation belongs to a different raster or transform')
        roles = observation['roles']
        if roles.shape != gray.shape or roles.dtype != np.uint8 or not np.isin(roles, [*range(10), 255]).all():
            raise ValueError('Role observations must use the original raster frame')
    first, second = receiver['roles'], donor['roles']
    agreement = first == second
    return reconcile(gray, baseline, protected, part=agreement & (first == 1),
                     stamp=agreement & (first == 8))


def recover_tonal_edges(gray, roles, proposed, protected):
    """Confirm visible gray boundary samples next to a previously proven stroke."""
    checked_gray(gray)
    if any(m.shape != gray.shape or m.dtype != np.uint8 for m in (roles, proposed)):
        raise ValueError('Aligned original-pixel role maps required')
    if protected.shape != gray.shape or protected.dtype != bool:
        raise ValueError('Aligned protection mask required')
    ink = foreground(gray) > 0
    # A faint sample must carry genuine contrast in the authenticated original.
    # Use only its immediate digital neighbourhood, which limits the change to
    # edge samples and leaves an unobserved white gap untouched.
    gray_edge = (roles == 255) & ~protected & ~ink & (gray < 240)
    kernel = np.ones((3, 3), np.uint8)
    output = roles.copy()
    for role in (1, 8):
        certain = (roles == role) & ink
        nearby = cv2.filter2D(certain.astype(np.uint8), -1, kernel, borderType=cv2.BORDER_CONSTANT)
        other = (roles != 255) & (roles != role)
        conflict = cv2.dilate(other.astype(np.uint8), kernel) > 0
        output[gray_edge & (proposed == role) & (nearby >= 3) & ~conflict] = role
    return output


def open_component_contour(gray, receiver, donor):
    """Support an intact open component already measured in the paired view."""
    checked_gray(gray)
    if any(a.shape != gray.shape or a.dtype != np.uint8 for a in (receiver, donor)):
        raise ValueError('Aligned measured roles required for both raster views')
    ink = foreground(gray) > 0
    _, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    result = np.zeros_like(ink)
    inspected = 0
    for item, (x, y, width, height, area) in enumerate(stats[1:], 1):
        if (min(width, height) < 48 or x == 0 or y == 0
                or x+width == gray.shape[1] or y+height == gray.shape[0]):
            continue
        roi = np.s_[y:y+height, x:x+width]
        inspected += width*height
        if inspected > 16_777_216:
            return np.zeros_like(ink)
        member = labels[roi] == item
        accepted = donor[roi][member]
        if ((accepted == 1).sum() < .95*area or not np.isin(accepted, (1, 255)).all()
                or not np.isin(receiver[roi][member], (1, 255)).all()):
            continue
        if cv2.connectedComponents(np.pad((~member).astype(np.uint8), 1, constant_values=1), connectivity=4)[0] == 2:
            result[roi] |= member
    return result


def _line_kernels(radius, angles):
    for direction in np.arange(angles)*np.pi/angles:
        end = np.rint(radius*np.array([np.cos(direction), np.sin(direction)])).astype(int)
        end = np.rint(end*radius/np.max(np.abs(end))).astype(int)
        kernel = np.zeros((2*radius+1, 2*radius+1), np.uint8)
        cv2.line(kernel, tuple(radius-end), tuple(radius+end), 1)
        yield kernel


def recover_confirmed_gaps(gray, roles, proposed, protected):
    """Use one pass of two-sided role support and visible native ink at contacts."""
    checked_gray(gray)
    if any(m.shape != gray.shape or m.dtype != np.uint8 for m in (roles, proposed)):
        raise ValueError('Aligned original-pixel role maps required')
    if protected.shape != gray.shape or protected.dtype != bool:
        raise ValueError('Aligned protection mask required')
    visible = gray < 238
    eligible = (roles == 255) & visible & ~protected
    if not (eligible & np.isin(proposed, (1, 8))).any():
        return roles.copy()
    neighbors = np.ones((3, 3), np.uint8)
    weak_ink = foreground(gray) == 0
    output = roles.copy()
    for role, radius, directions in ((1, 21, 12), (8, 13, 2)):
        candidate = eligible & (proposed == role)
        if not candidate.any():
            continue
        anchors = (roles == role).astype(np.uint8)
        if not anchors.any():
            continue
        competition = (roles != 255) & (roles != 0) & (roles != role)
        clear = cv2.dilate(competition.astype(np.uint8), neighbors) == 0
        count = cv2.filter2D(anchors, -1, neighbors, borderType=cv2.BORDER_CONSTANT)
        supported = weak_ink & (count >= 3)
        for kernel in _line_kernels(radius, directions):
            bridge = cv2.morphologyEx(anchors, cv2.MORPH_CLOSE, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
            # Require observed continuity across at least a short window around
            # the candidate in the same direction as the two accepted edges.
            y, x = np.where(kernel)
            offset = np.rint(np.array([x[-1]-radius, y[-1]-radius])/radius).astype(int)
            local = np.zeros((3, 3), np.uint8)
            cv2.line(local, tuple(1-offset), tuple(1+offset), 1)
            continuous = cv2.erode(visible.astype(np.uint8), local, borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
            supported |= bridge & continuous
        output[candidate & clear & supported] = role
    return output


def peripheral_strokes(visible):
    """Measure long source rulings near the edge of a native table view."""
    result = np.zeros_like(visible)
    for direction in (0, 1):
        span = visible.shape[1-direction]
        minimum = int(.8*span) | 1
        if minimum < 35:
            continue
        for points, _, center, pen in axis_strokes(visible, direction, opening=minimum,
                                                  minimum_length=minimum, maximum_band=28):
            extent = visible.shape[direction]
            if pen >= 3 and min(center, extent-1-center) <= .13*extent:
                x, y = points.T
                result[y, x] = True
    return result


def _directional_support(visible, accepted):
    ink = visible.astype(np.uint8)
    supported = np.zeros_like(visible)
    for along, local in zip(_line_kernels(42, 24), _line_kernels(5, 24)):
        observed = cv2.morphologyEx(ink, cv2.MORPH_OPEN, local, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        bridged = cv2.morphologyEx(accepted, cv2.MORPH_CLOSE, along, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        supported |= (observed & bridged) > 0
    return supported


def recover_corroborated_contacts(gray, roles, paired, proposed, protected, *, bound_table, directional=False):
    """Confirm nearby source ink from the two aligned, previously qualified maps."""
    checked_gray(gray)
    if any(m.shape != gray.shape or m.dtype != np.uint8 for m in (roles, paired, proposed)):
        raise ValueError('Aligned original-pixel role maps required')
    if protected.shape != gray.shape or protected.dtype != bool:
        raise ValueError('Aligned protection mask required')
    visible = gray < 238
    source = visible & (roles == 255) & ~protected
    kernel = np.ones((3, 3), np.uint8)
    result = roles.copy()
    for role in (1, 8) if bound_table else (1,):
        candidate = source & (proposed == role)
        if not candidate.any():
            continue
        known_other = ((roles > 0) & (roles < 10) & (roles != role)
                       | (paired > 0) & (paired < 10) & (paired != role))
        candidate &= cv2.dilate(known_other.astype(np.uint8), kernel) == 0
        if not candidate.any():
            continue
        support = ((roles == role) | (paired == role)).astype(np.uint8)
        if directional and role == 1:
            candidate &= _directional_support(visible, support)
        else:
            count = cv2.filter2D(support, -1, kernel, borderType=cv2.BORDER_CONSTANT)
            candidate &= count >= (4 if directional else 3)
        if role == 8 and not directional and candidate.any():
            candidate &= ~peripheral_strokes(visible)
        result[candidate] = role
    return result
