"""Source-bound annotation support; never extends measured part geometry.

The head's observed longitudinal extent is distinct from localization error.
Only the explicitly named shaft is tested; nearby lines are not substitutes.
"""
from copy import deepcopy
from pathlib import Path
import math

import cv2
import numpy as np
from PIL import Image

from . import dimension_association_v3 as v3

POLICY = {
    'version': 'arrow-shaft-support/1.0',
    'angle_degrees': 12, 'minimum_confidence': .65,
    'contact_stroke_factor': 1.0, 'maximum_uncertainty_strokes': 2.,
    'minimum_axis_ink': .95, 'minimum_shaft_ink': .95,
    'minimum_taper_r_squared': .80, 'minimum_profile_fraction': .80,
    'width_ratio_range': [.65, 1.5], 'shaft_width_ratio_range': [.5, 2.],
    'maximum_pixels': 12_000_000, 'maximum_arrows': 256,
    'maximum_samples': 500_000, 'maximum_profile_samples': 100_000,
    'maximum_owners': 8, 'maximum_span_length_px': 2048,
    'uncertainty_is_calibrated_sigma': False,
    'production_admitted': False, 'physical_geometry_approved': False,
}
COS = math.cos(math.radians(POLICY['angle_degrees']))


class SampleLimit(Exception):
    pass


class Raster:
    def __init__(self, image):
        self.foreground = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1] > 0
        self.samples = 0

    def sample(self, points):
        shape = points.shape[:-1]
        if self.samples + int(np.prod(shape)) > POLICY['maximum_samples']:
            raise SampleLimit()
        self.samples += int(np.prod(shape))
        points = np.rint(points).astype(int).reshape(-1, 2)
        h, w = self.foreground.shape
        valid = ((points >= 0).all(axis=1) & (points[:, 0] < w) & (points[:, 1] < h))
        values = np.zeros(len(points), bool)
        values[valid] = self.foreground[points[valid, 1], points[valid, 0]]
        return values.reshape(shape)

    def widths(self, centers, normal, radius):
        radius = math.ceil(radius)
        cost = len(centers) * (2 * radius + 1)
        if cost > POLICY['maximum_profile_samples']:
            raise SampleLimit()
        offsets = np.arange(-radius, radius + 1)
        ink = self.sample(centers[:, None, :] + offsets[None, :, None] * normal)
        widths = []
        for row in ink:
            if not row[radius]:
                widths.append(0.)
                continue
            lo = hi = radius
            while lo > 0 and row[lo - 1]:
                lo -= 1
            while hi + 1 < len(row) and row[hi + 1]:
                hi += 1
            # A clipped profile cannot establish the surrounding white area.
            widths.append(float(hi - lo + 1) if lo > 0 and hi < len(row) - 1 else float('inf'))
        return np.array(widths)


def _positive(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def _check(raster, arrow, lines):
    event = {'arrow_id': arrow['id'], 'status': 'rejected', 'reasons': []}

    def reject(reason):
        event['reasons'].append(reason)
        return event

    evidence = arrow.get('evidence', {})
    owners = evidence.get('geometry_ids')
    if not isinstance(owners, list) or not owners or len(set(owners)) != len(owners):
        return reject('explicit_unique_shaft_ids_required')
    if len(owners) > POLICY['maximum_owners']:
        raise SampleLimit()
    if any(gid not in lines for gid in owners):
        return reject('named_shaft_missing_or_not_line')
    if arrow['confidence'] < POLICY['minimum_confidence']:
        return reject('low_arrow_confidence')
    polygon = np.array(arrow['polygon_px'], float)
    if polygon.shape != (3, 2):
        return reject('triangle_required')
    tip = np.array(arrow['tip_px'], float)
    base = (polygon[1] + polygon[2]) / 2
    delta = tip - base
    length = float(np.linalg.norm(delta))
    if length > POLICY['maximum_span_length_px']:
        raise SampleLimit()
    if length <= 0:
        return reject('degenerate_head')
    direction = delta / length
    normal = np.array([-direction[1], direction[0]])
    width = float(abs(np.dot(polygon[1] - polygon[2], normal)))
    stroke = evidence.get('shaft_width_px')
    uncertainty = arrow.get('uncertainty_px')
    if not _positive(stroke) or not _positive(uncertainty):
        return reject('positive_stroke_and_localization_uncertainty_required')
    if uncertainty > POLICY['maximum_uncertainty_strokes'] * stroke:
        return reject('localization_uncertainty_not_bounded_by_stroke')
    if (width <= stroke or np.linalg.norm(polygon[0] - tip) > .5 * stroke
            or np.dot(direction, arrow['direction']) < COS
            or abs(float(np.dot(polygon[1] - polygon[2], direction))) > stroke):
        return reject('triangle_tip_base_direction_inconsistent')
    event.update(tip_px=tip.tolist(), base_px=base.tolist(), head_length_px=length,
                 base_width_px=width, reported_shaft_width_px=stroke,
                 arrow_uncertainty_px=uncertainty)
    fractions = np.linspace(.15, .85, 15)
    widths = raster.widths(tip - fractions[:, None] * length * direction, normal, width * .8 + 2 * stroke)
    expected = fractions * width
    # One pixel is raster quantization, rather than a physical gap allowance.
    ratios = widths / np.maximum(expected, 1.)
    good = np.isfinite(widths) & (ratios >= POLICY['width_ratio_range'][0]) & (ratios <= POLICY['width_ratio_range'][1])
    profile_fraction = float(good.mean())
    if profile_fraction < POLICY['minimum_profile_fraction']:
        return reject('observed_triangle_widths_disagree')
    fit = np.polyfit(fractions[good], widths[good], 1)
    variance = float(np.sum((widths[good] - widths[good].mean()) ** 2))
    r_squared = 1 - float(np.sum((widths[good] - np.polyval(fit, fractions[good])) ** 2)) / variance if variance else 0.
    event.update(profile_fraction=profile_fraction, taper_r_squared=r_squared)
    if fit[0] <= 0 or r_squared < POLICY['minimum_taper_r_squared']:
        return reject('observed_taper_missing')
    # Exclude the analytically fitted subpixel apex itself, but check every
    # interior center sample. A long white gap cannot be licensed by endpoints.
    axis = np.linspace(tip - direction * min(stroke, length * .1), base, max(2, math.ceil(length)))
    axis_ink = float(raster.sample(axis).mean())
    event['axis_ink_fraction'] = axis_ink
    if axis_ink < POLICY['minimum_axis_ink']:
        return reject('head_centerline_not_continuous_ink')
    shaft_centers = base - np.linspace(2 * stroke, 8 * stroke, 13)[:, None] * direction
    shaft_widths = raster.widths(shaft_centers, normal, max(width, 4 * stroke))
    finite = shaft_widths[np.isfinite(shaft_widths) & (shaft_widths > 0)]
    if len(finite) < 12:
        return reject('shaft_profile_incomplete_or_crossing')
    measured = float(np.median(finite))
    event['measured_shaft_width_px'] = measured
    ratio = measured / stroke
    if not POLICY['shaft_width_ratio_range'][0] <= ratio <= POLICY['shaft_width_ratio_range'][1]:
        return reject('reported_stroke_disagrees_with_pixels')
    if (np.mean((shaft_widths >= .5 * measured) & (shaft_widths <= 2 * measured)) < .95
            or raster.sample(shaft_centers).mean() < POLICY['minimum_shaft_ink']):
        return reject('shaft_not_continuous_thin_ink')
    matches = []
    for gid in sorted(owners):
        line = lines[gid]
        geo_uncertainty = line['original'].get('uncertainty_px')
        if not _positive(geo_uncertainty) or geo_uncertainty > 2 * measured:
            continue
        tolerance = POLICY['contact_stroke_factor'] * measured + max(uncertainty, geo_uncertainty)
        distance = v3._segment_distance(base, line)
        distances = [v3._segment_distance(p, line) for p in shaft_centers]
        if abs(float(np.dot(line['u'], direction))) < COS or distance > tolerance or max(distances) > tolerance:
            continue
        matches.append({'shaft_id': gid, 'base_distance_px': distance,
                        'tip_distance_px': v3._segment_distance(tip, line),
                        'contact_tolerance_px': tolerance,
                        'geometry_uncertainty_px': geo_uncertainty})
    event['owner_checks'] = matches
    if not matches:
        return reject('no_observed_contact_with_explicit_own_shaft')
    # Multiple genuinely different owners are not resolved by nearest distance.
    if len(matches) != 1:
        event['status'] = 'ambiguous'
        return reject('multiple_explicit_shaft_owners')
    event['status'] = 'supported'
    return event


def observe(source_image, *, geometry, ocr, arrows):
    """No labels, nominal values, search radius overrides or physical scales."""
    source_image = Path(source_image)
    digest, size, normalized, _, _ = v3._validate(source_image, geometry, ocr, arrows)
    if size[0] * size[1] > POLICY['maximum_pixels']:
        raise ValueError('Arrow support raster exceeds resource limit')
    with Image.open(source_image) as image:
        raster = Raster(np.array(image.convert('L')))
    lines = {line['id']: line for line in normalized}
    events, spans = [], []
    complete = len(arrows['candidates']) <= POLICY['maximum_arrows']
    if not complete:
        events.append({'status': 'unknown', 'reasons': ['arrow_count_resource_limit']})
    try:
        if complete:
            for arrow in sorted(arrows['candidates'], key=lambda a: a['id']):
                event = _check(raster, arrow, lines)
                events.append(event)
                if event['status'] != 'supported':
                    continue
                span_id = 'annotation-head-span-' + arrow['id']
                if span_id in lines:
                    raise ValueError('Annotation support ID collides with input geometry')
                spans.append({'id': span_id, 'role': 'annotation_support_only',
                              'points_px': [event['tip_px'], event['base_px']],
                              'arrow_id': arrow['id'], 'shaft_id': event['owner_checks'][0]['shaft_id'],
                              'uncertainty_px': max(arrow['uncertainty_px'], event['owner_checks'][0]['geometry_uncertainty_px']),
                              'source_ref': {'sha256': digest, 'coordinate_space': 'processed_image_pixels'},
                              'support_to_processed': [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
                              'accepted_as_part_geometry': False})
    except SampleLimit:
        complete = False
        events.append({'status': 'unknown', 'reasons': ['resource_limit']})
    if len(geometry['candidates']) + len(spans) > v3.MAX_GEOMETRY:
        complete = False
        events.append({'status': 'unknown', 'reasons': ['association_view_geometry_limit']})
    if not complete:
        spans = []  # Do not silently admit an early subset of a capped search.
    return {'schema_version': POLICY['version'], 'source_sha256': digest,
            'processed_size_px': size, 'complete': complete, 'events': events,
            'spans': spans, 'sample_count': raster.samples, 'policy': deepcopy(POLICY),
            'status': 'needs_review', 'production_admitted': False,
            'physical_geometry_approved': False}


def association_view(geometry, arrows, support):
    """A disposable, typed annotation view. Original primitives stay intact."""
    for report in (geometry, arrows):
        if any(report.get(k) != support.get(k) for k in ('source_sha256', 'processed_size_px')):
            raise ValueError('Support frame mismatch')
    geo, heads = deepcopy(geometry), deepcopy(arrows)
    if not support['complete']:
        return geo, heads
    by_arrow = {a['id']: a for a in heads['candidates']}
    for span in support['spans']:
        geo['candidates'].append({**deepcopy(span), 'kind': 'line',
                                  'classification': 'annotation_support_only',
                                  'evidence': {'method': POLICY['version'], 'original_shaft_id': span['shaft_id']}})
        by_arrow[span['arrow_id']]['evidence']['geometry_ids'].append(span['id'])
    return geo, heads
