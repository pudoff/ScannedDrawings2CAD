"""Observe arrow heads in bounded, original-pixel windows along CV carriers.

This stage records source evidence, not semantic roles or CAD dimensions. A
completed search is distinct from recognition recall. Old detectors are frozen.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import tempfile
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from . import dimension_association_v3 as association
from . import arrow_head_completion_v1 as completion
from . import arrow_head_validation_v1 as validation
from .arrow_path_recovery_v1 import native_head_check, native_junction
from .source_arrow_junction_v1 import proposals as junction_proposals
from .source_arrow_taper_v1 import proposals as taper_proposals
from .png_stream import PNGStripReader
from .scaled_arrow_detection_v2 import detect as scaled_detect, clip_segment


POLICY = {'version': 'source-arrow-observation/1.0', 'window_side': 512,
          'window_step': 256, 'maximum_windows': 1024,
          'maximum_roi_pixels': 268_435_456, 'maximum_lines': 2048,
          'maximum_source_pixels': 256_000_000, 'maximum_candidates': 2048,
          'maximum_local_lines': 128, 'maximum_owner_count': 8,
          'maximum_small_decode_pixels': 12_000_000,
          'maximum_anisotropy': 1.02, 'minimum_owner_dot': .98,
          'production_admitted': False, 'physical_geometry_approved': False}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def transform(points, matrix):
    points = np.asarray(points, float)
    return points @ matrix[:2, :2].T + matrix[:2, 2]


def checked_matrix(value):
    matrix = np.asarray(value, float)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[2], [0, 0, 1], atol=1e-10)):
        raise ValueError('Finite affine pixel-center transform required')
    scales = np.linalg.svd(matrix[:2, :2], compute_uv=False)
    if scales.min() <= 1e-6 or scales.max()/scales.min() > POLICY['maximum_anisotropy']:
        raise ValueError('A nonsingular near-uniform image transform is required')
    return matrix


def plan_windows(lines, size):
    side, step = POLICY['window_side'], POLICY['window_step']
    boxes = set()
    for line in lines:
        ends = np.asarray(line['points_px'])
        clipped = clip_segment(ends, size)
        if clipped is None:
            continue
        a, b = np.asarray(clipped)
        count = max(2, math.ceil(np.linalg.norm(b-a)/step)+1)
        for point in np.linspace(a, b, count):
            center = np.floor(point/step)*step + step/2
            lo = np.maximum(0, np.floor(center-side/2)).astype(int)
            hi = np.minimum(size, np.ceil(center+side/2)).astype(int)
            if np.all(hi > lo):
                boxes.add(tuple(map(int, [*lo, *hi])))
    return sorted(boxes, key=lambda b: (b[1], b[0]))


def source_windows(path, source_sha, size, boxes):
    """Decode PNG once, retaining only requested source pixels; never resize."""
    if math.prod(size) <= POLICY['maximum_small_decode_pixels']:
        with Image.open(path) as image:
            if 'A' in image.getbands() or 'transparency' in image.info:
                canvas = Image.new('RGBA', image.size, 'white')
                canvas.alpha_composite(image.convert('RGBA'))
                gray = np.array(canvas.convert('L'))
            else:
                gray = np.array(image.convert('L'))
        if digest(path) != source_sha:
            raise ValueError('Source changed during observation')
        return [gray[t:b, l:r].copy() for l, t, r, b in boxes]
    # The existing streaming decoder checks PNG structure, CRC and complete
    # source SHA, including rows outside requested windows.
    reader = PNGStripReader(Path(path), expected_sha256=source_sha, strip_rows=128)
    if [reader.width, reader.height] != size:
        raise ValueError('Changed source dimensions')
    regions = [np.empty((b-t, r-l), np.uint8) for l, t, r, b in boxes]
    top = 0
    for pixels in reader.iter_strips():
        # iter_strips yields (row offset, RGB pixels).
        if isinstance(pixels, tuple):
            top, pixels = pixels
        end = top + pixels.shape[0]
        gray = np.array(Image.fromarray(pixels).convert('L'))
        for region, (left, upper, right, bottom) in zip(regions, boxes):
            low, high = max(top, upper), min(end, bottom)
            if high > low:
                region[low-upper:high-upper] = gray[low-top:high-top, left:right]
        top = end
    if top != size[1]:
        raise ValueError('Incomplete original raster decode')
    return regions


def local_lines(lines, box):
    origin = np.asarray(box[:2]); size = np.asarray(box[2:])-origin
    result = []
    for line in lines:
        clipped = clip_segment(np.asarray(line['points_px'])-origin, size)
        if clipped is None or np.linalg.norm(np.subtract(*clipped)) < 2:
            continue
        row = deepcopy(line)
        row['points_px'] = clipped
        if 'support_points' in row:
            points = np.asarray(row['support_points'], float)-origin
            row['support_points'] = points[((points >= 0) & (points < size)).all(1)].tolist()
        result.append(row)
    return result


def stroke_width(mask):
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    values = distance[(mask > 0) & (distance >= cv2.dilate(distance, np.ones((3, 3), np.uint8)))]
    return float(np.clip(np.quantile(2*values-1, .25), 1, 24)) if len(values) else 1.


def bounded_head_check(candidate, mask, stroke, text_polygons):
    # The frozen checker uses ceil(max)+1 for its polygon buffer. Reject a
    # polygon whose buffer crosses the ROI; another overlapping window can
    # establish a complete head. Never silently crop a triangle for validation.
    points = np.asarray(candidate['polygon_px'], float)
    if (not np.isfinite(points).all() or (points < 0).any()
            or (np.ceil(points.max(0))+1 > np.asarray(mask.shape[::-1])).any()):
        return False, {'reason': 'native_head_clipped_by_window'}
    return native_head_check(candidate, mask, stroke, text_polygons)


def merge_observations(candidates):
    """Prefer a source-checked junction over an extrapolated triangular apex."""
    output = []
    def quality(c):
        e = c['evidence']
        return (e.get('tip_localization') == 'native_unique_transverse_line_junction',
                e.get('source_verified', False), e.get('native_head_check', {}).get('triangle_ink', 0))
    for row in deepcopy(candidates):
        polygon = np.asarray(row['polygon_px'], np.float32)
        for index, previous in enumerate(output):
            if np.dot(row['direction'], previous['direction']) < .96:
                continue
            row_owners = set(row['evidence'].get('geometry_ids', []))
            old_owners = set(previous['evidence'].get('geometry_ids', []))
            if (row['evidence'].get('source_verified') and previous['evidence'].get('source_verified')
                    and row_owners and old_owners and row_owners.isdisjoint(old_owners)):
                continue
            other = np.asarray(previous['polygon_px'], np.float32)
            overlap, _ = cv2.intersectConvexConvex(polygon, other)
            union = abs(cv2.contourArea(polygon))+abs(cv2.contourArea(other))-overlap
            if (np.linalg.norm(np.asarray(row['tip_px'])-previous['tip_px']) <= 3
                    or union > 0 and overlap/union >= .35):
                if quality(row) > quality(previous):
                    output[index] = row
                break
        else:
            output.append(row)
    return output


def verify(candidate, gray, lines, text_polygons, stroke):
    """Independent native shape/shaft checks; confidence is never increased."""
    mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    c = deepcopy(candidate)
    p = np.asarray(c['polygon_px'], float); tip = np.asarray(c['tip_px'])
    base = p[1:].mean(0); length = float(np.linalg.norm(tip-base))
    if length <= 0 or not np.allclose(p[0], tip):
        return None, 'invalid_triangle'
    direction = (tip-base)/length
    if np.dot(direction, c['direction']) < .999:
        return None, 'inconsistent_direction'
    e = c['evidence']
    e.update(head_length_px=length, arrow_length_px=length,
             base_width_px=float(np.linalg.norm(p[1]-p[2])))
    if not completion.tip_has_native_ink(c, mask, stroke):
        return None, 'apex_without_native_ink'
    valid, head = bounded_head_check(c, mask, stroke, text_polygons)
    if not valid and head['reason'] == 'native_flanks_not_background':
        local = []
        for line in lines:
            ends = np.asarray(line['points_px']); delta = ends[1]-ends[0]
            extent = float(np.linalg.norm(delta))
            local.append((line['id'], ends, delta/extent, extent))
        valid, head = validation.attached_head_check(c, mask, stroke, text_polygons, local)
    if not valid:
        return None, head['reason']
    valid, shape = validation.shape_check(c, mask)
    if not valid:
        return None, shape['reason']
    valid, shaft = completion.shaft_evidence(c, mask, stroke)
    if not valid:
        return None, shaft['reason']
    measured = shaft['width_px']
    if shaft['base_to_shaft_ratio'] < validation.POLICY['minimum_base_to_shaft_ratio']:
        return None, 'insufficient_head_to_native_shaft_contrast'
    # The actual short shaft is measured behind the base. A geometrical line
    # farther away or extending only in front of the tip cannot own this head.
    probes = base - np.linspace(2*measured, min(shaft['observed_length_px'], 8*measured), 9)[:, None]*direction
    owners = []
    for line in lines:
        a, b = np.asarray(line['points_px']); delta = b-a
        extent = float(np.linalg.norm(delta)); unit = delta/extent
        if abs(float(unit@direction)) < POLICY['minimum_owner_dot']:
            continue
        uncertainty = float(line.get('uncertainty_px', 1.))
        if (not math.isfinite(uncertainty) or uncertainty < 0
                or uncertainty > 2*measured + line.get('source_projection_quantization_px', 0.)):
            continue
        tolerance = measured + uncertainty
        along = (probes-a)@unit
        across = abs((probes-a)@np.array([-unit[1], unit[0]]))
        if np.all((along >= -tolerance) & (along <= extent+tolerance) & (across <= tolerance)):
            owners.append(line)
    if not owners:
        return None, 'no_observed_carrier_owner'
    if len(owners) > POLICY['maximum_owner_count']:
        return None, 'ambiguous_carrier_owners'
    # Distinct parallel strokes must not be merged solely by a wide localization
    # band. Multiple IDs are retained only for the same observed centerline.
    offsets = [(np.asarray(line['points_px'])[0]-base)@np.array([-direction[1], direction[0]]) for line in owners]
    if np.ptp(offsets) > measured + math.sqrt(.5):
        return None, 'ambiguous_parallel_carriers'
    members = {json.dumps(x, sort_keys=True, ensure_ascii=False): x
               for line in owners for x in line.get('source_members', [])}
    e.update(geometry_ids=sorted(line['id'] for line in owners), shaft_width_px=measured,
             source_members=[deepcopy(members[k]) for k in sorted(members)],
             native_head_check=head, native_shape_check=shape, native_shaft=shaft,
             source_verified=True, shaft_owner_unresolved=False, confidence_recalibrated=False)
    return c, 'source_head_and_carrier_verified'


def observe(source_image, *, geometry, geometry_image=None, geometry_to_source=None, text_polygons=()):
    source_image = Path(source_image)
    geometry_image = Path(geometry_image) if geometry_image is not None else source_image
    common = {k: geometry.get(k) for k in ('source_sha256', 'processed_size_px')}
    association._validate(geometry_image, geometry, {**common, 'lines': [], 'notations': []}, {**common, 'candidates': []})
    matrix = checked_matrix(np.eye(3) if geometry_to_source is None else geometry_to_source)
    source_sha = digest(source_image)
    with Image.open(source_image) as image:
        size = list(image.size)
    if math.prod(size) > POLICY['maximum_source_pixels']:
        raise ValueError('Source raster exceeds bounded observation size')
    if geometry_image != source_image and geometry_to_source is None:
        raise ValueError('Explicit geometry-to-source transform required for different images')
    polygons = [transform(association._polygon(p, 'OCR exclusion polygon', size=geometry['processed_size_px']), matrix).tolist()
                for p in text_polygons]
    scale = float(np.linalg.svd(matrix[:2, :2], compute_uv=False).max())
    lines = []
    for row in geometry['candidates']:
        if row['kind'] != 'line':
            continue
        line = deepcopy(row)
        line['points_px'] = transform(row['points_px'], matrix).tolist()
        if 'support_points' in row:
            line['support_points'] = transform(row['support_points'], matrix).tolist()
        uncertainty = row.get('uncertainty_px', 1.)
        if (isinstance(uncertainty, bool) or not isinstance(uncertainty, (int, float))
                or not math.isfinite(uncertainty) or uncertainty < 0):
            raise ValueError('Finite nonnegative source line uncertainty required')
        line['uncertainty_px'] = max(.5, uncertainty*scale)
        line['source_projection_quantization_px'] = math.sqrt(.5)*scale
        lines.append(line)
    limits = []
    if len(lines) > POLICY['maximum_lines']:
        limits.append('line_budget')
    boxes = plan_windows(lines, size) if not limits else []
    pixels = sum((r-l)*(b-t) for l,t,r,b in boxes)
    if len(boxes) > POLICY['maximum_windows']:
        limits.append('window_budget')
    if pixels > POLICY['maximum_roi_pixels']:
        limits.append('roi_pixel_budget')
    candidates, verified, events = [], [], []
    if not limits:
        windows = source_windows(source_image, source_sha, size, boxes)
        with tempfile.TemporaryDirectory(prefix='cad-source-head-') as scratch:
            for index, (box, gray) in enumerate(zip(boxes, windows)):
                local = local_lines(lines, box)
                if len(local) > POLICY['maximum_local_lines']:
                    limits.append('local_line_budget')
                    events.append({'box_original_px': box, 'status': 'local_line_budget'})
                    continue
                origin = np.array(box[:2])
                local_text = [(np.asarray(p)-origin).tolist() for p in polygons
                              if (np.asarray(p).max(0) >= origin).all() and (np.asarray(p).min(0) < box[2:]).all()]
                # Existing head-first search accepts out-of-window polygons;
                # scaled search validates image bounds, so only use its fully
                # contained exclusions and apply all polygons in native checks.
                scaled_text = [p for p in local_text if (np.asarray(p) >= 0).all()
                               and (np.asarray(p) < gray.shape[::-1]).all()]
                path = Path(scratch)/f'{index}.png'
                Image.fromarray(gray).save(path)
                local_geometry = {'source_sha256': digest(path), 'processed_size_px': list(gray.shape[::-1]), 'candidates': local}
                scaled = scaled_detect(path, geometry=local_geometry, text_polygons=scaled_text)
                mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                stroke = stroke_width(mask)
                with patch.object(completion.previous, 'native_head_check', bounded_head_check):
                    extra = validation.search(gray, baseline=scaled['candidates'], lines=local, stroke_px=stroke, text_polygons=local_text)
                    junctions = junction_proposals(gray, lines=local, stroke_px=stroke, text_polygons=local_text)
                    tapers = taper_proposals(gray, lines=local, text_polygons=local_text)
                    refined = [native_junction(deepcopy(row), mask, stroke, local, local_text)
                               for row in extra['candidates']]
                combined = merge_observations(refined+junctions['candidates']+tapers['candidates'])
                incomplete = (scaled['analysis']['search_limited'] or extra['search_incomplete']
                              or junctions['search_incomplete'] or tapers['search_incomplete'])
                if incomplete:
                    limits.append('local_search_incomplete')
                rejected = Counter(); accepted = 0
                for raw in combined:
                    c, reason = verify(raw, gray, local, local_text, stroke)
                    rejected[reason] += 1
                    for row, target in ((deepcopy(raw), candidates), (c, verified)):
                        if row is None:
                            continue
                        row['tip_px'] = (np.asarray(row['tip_px'])+origin).tolist()
                        row['polygon_px'] = (np.asarray(row['polygon_px'])+origin).tolist()
                        row['source_ref'] = {'sha256': source_sha, 'image': source_image.name,
                                             'coordinate_space': 'original_image_pixels'}
                        row['evidence']['source_window'] = {'box_original_px': list(box),
                            'gray_pixels_sha256': hashlib.sha256(gray.tobytes()).hexdigest(), 'resampled': False}
                        row['evidence']['measurement_coordinate_space'] = 'source_window_pixels'
                        row['evidence']['window_to_original'] = [[1, 0, int(origin[0])], [0, 1, int(origin[1])], [0, 0, 1]]
                        target.append(row)
                    accepted += c is not None
                events.append({'box_original_px': list(box), 'lines': len(local), 'stroke_px': stroke,
                    'source_pixel_sha256': hashlib.sha256(gray.tobytes()).hexdigest(),
                    'scaled_count': len(scaled['candidates']), 'combined_count': len(combined),
                    'junction_count': len(junctions['candidates']), 'junction_events': junctions['events'],
                    'taper_count': len(tapers['candidates']), 'taper_events': tapers['events'],
                    'taper_analysis': tapers['analysis'],
                    'verified_count': accepted, 'reasons': dict(rejected),
                    'search_complete': not incomplete})
                if len(candidates) > POLICY['maximum_candidates']:
                    limits.append('candidate_budget')
                    break
    if geometry.get('complete', True) is False:
        limits.append('input_geometry_incomplete')
    candidates = merge_observations(candidates)
    diagnostic = merge_observations(verified)
    verified = [] if limits else diagnostic
    for group in (candidates, diagnostic):
        for index, row in enumerate(group, 1):
            row['id'] = f'source-arrow-{index:05d}'
    return {'schema_version': POLICY['version'], 'source_sha256': source_sha, 'processed_size_px': size,
        'geometry_source_sha256': common['source_sha256'], 'geometry_to_source': matrix.tolist(),
        'registration': {'matrix_supplied_by_caller': geometry_to_source is not None,
                         'full_view_pixel_correspondence_checked': False,
                         'native_head_and_local_carrier_pixels_checked': True},
        'candidates': candidates, 'verified_candidates': verified,
        'diagnostic_verified_candidates': diagnostic, 'search_complete': not limits,
        'analysis': {'limits': sorted(set(limits)), 'planned_windows': len(boxes), 'searched_windows': len(events),
                     'planned_roi_pixels': pixels, 'scope': 'windows_along_input_CV_lines',
                     'full_page_recall_claimed': False}, 'windows': events, 'policy': deepcopy(POLICY),
        'production_admitted': False, 'physical_geometry_approved': False, 'CAD_promotions': 0}
