"""Conservative geometric candidates in processed raster pixels.

These observations are not CAD geometry approved by a dimension solver. In
particular, text, frames, hatching and object edges have no semantic labels yet.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .circle_search_v2 import (build_circle_queue, propose_short_lines, validate_circle_proposal)

MAX_PIXELS = 25_000_000
MAX_BYTES = 50 * 1024 * 1024
ANALYSIS_SIDE = 2400
MAX_HOUGH_LINES = 2500
MAX_CONTOURS = 10000
MAX_SHORT_CONNECTOR_PAIRS = 10000
MAX_SHORT_CONNECTORS = 250


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _save_png(path: Path, image: Image.Image) -> None:
    with path.open("xb") as stream:
        image.save(stream, format="PNG")


def _skeleton(foreground: np.ndarray) -> tuple[np.ndarray, bool]:
    """Morphological centre-line evidence, bounded for filled regions."""
    remaining = foreground.copy()
    skeleton = np.zeros_like(foreground)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for _ in range(64):
        eroded = cv2.erode(remaining, element)
        opened = cv2.dilate(eroded, element)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(remaining, opened))
        remaining = eroded
        if not cv2.countNonZero(remaining):
            return skeleton, False
    # Thick filled areas have no dependable centre-line interpretation. Their
    # unprocessed cores are deliberately excluded, with a report issue.
    return skeleton, True


def _fit_circle(points: np.ndarray) -> tuple[np.ndarray, float, np.ndarray] | None:
    origin = points.mean(axis=0)
    local = points - origin
    design = np.column_stack((2 * local, np.ones(len(local))))
    fitted, _, rank, _ = np.linalg.lstsq(design, np.sum(local * local, axis=1), rcond=None)
    squared_radius = fitted[2] + np.dot(fitted[:2], fitted[:2])
    if rank < 3 or squared_radius <= 0:
        return None
    center = fitted[:2] + origin
    radius = float(np.sqrt(squared_radius))
    return center, radius, np.abs(np.linalg.norm(points - center, axis=1) - radius)


def _observed_angles(points: np.ndarray, center: np.ndarray) -> tuple[float, float]:
    angles = np.sort(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]) % (2 * np.pi))
    gaps = np.diff(np.r_[angles, angles[0] + 2 * np.pi])
    gap = int(np.argmax(gaps))
    return float(angles[(gap + 1) % len(angles)]), float(2 * np.pi - gaps[gap])


def _curve_support(center: np.ndarray, radius: float, start: float, sweep: float,
                   ink_distance: np.ndarray, tolerance: float) -> float:
    angles = np.linspace(start, start + sweep, max(72, min(720, int(radius * sweep))), endpoint=False)
    points = center + radius * np.column_stack((np.cos(angles), np.sin(angles)))
    indices = np.rint(points).astype(int)
    valid = ((indices[:, 0] >= 0) & (indices[:, 0] < ink_distance.shape[1])
             & (indices[:, 1] >= 0) & (indices[:, 1] < ink_distance.shape[0]))
    supported = np.zeros(len(indices), dtype=bool)
    supported[valid] = ink_distance[indices[valid, 1], indices[valid, 0]] <= tolerance
    return float(supported.mean())


def _curves(foreground: np.ndarray, stroke: float) -> tuple[list[dict], dict]:
    contours, _ = cv2.findContours(foreground, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # Stable priority limits resource use on noisy scans, without claiming all
    # candidates were visited when the limit was exceeded.
    contours = sorted(contours, key=len, reverse=True)
    distance = cv2.distanceTransform(255 - foreground, cv2.DIST_L2, 3)
    curves = []
    for contour in contours[:MAX_CONTOURS]:
        points = contour.reshape(-1, 2).astype(float)
        if len(points) < 36 or np.min(np.ptp(points, axis=0)) < 12:
            continue
        perimeter = cv2.arcLength(contour, True)
        if len(cv2.approxPolyDP(contour, 0.012 * perimeter, True)) < 6:
            continue
        fit = _fit_circle(points)
        if fit is None:
            continue
        center, radius, residual = fit
        if radius < 10 or radius > max(foreground.shape) * 2:
            continue
        rmse = float(np.sqrt(np.mean(residual ** 2)))
        if rmse > max(1.1, min(4., stroke * 0.7)) or rmse / radius > 0.045:
            continue
        if float(np.quantile(residual, 0.95)) > max(2., stroke * 0.95):
            continue
        start, sweep = _observed_angles(points, center)
        if sweep < np.pi / 4:
            continue
        # A complete sampled raster ring has a small gap between adjacent
        # contour pixels. Larger observed gaps remain arcs, even at 350°.
        kind = "circle" if 2 * np.pi - sweep <= max(np.deg2rad(3), 2 / radius) else "arc"
        if kind == "circle":
            start, sweep = 0., 2 * np.pi
        support = _curve_support(center, radius, start, sweep, distance, max(1.25, stroke * .65))
        if support < .92:
            continue
        curves.append({"kind": kind, "center": center, "radius": radius,
                       "start": start, "sweep": sweep, "rmse": rmse,
                       "support": support, "contour_count": 1, "point_count": len(points)})
    merged: list[dict] = []
    for curve in curves:
        for previous in merged:
            same_range = (curve["kind"] == previous["kind"] == "circle" or
                          (curve["kind"] == previous["kind"] == "arc"
                           and abs(curve["sweep"] - previous["sweep"]) < np.deg2rad(8)
                           and abs(math.atan2(math.sin(curve["start"] - previous["start"]),
                                             math.cos(curve["start"] - previous["start"]))) < np.deg2rad(8)))
            if (same_range and np.linalg.norm(curve["center"] - previous["center"]) <= max(2., stroke)
                    and abs(curve["radius"] - previous["radius"]) <= max(2., stroke + 1)):
                # Inner/outer stroke contours estimate one centre-line circle.
                count = previous["contour_count"]
                previous["center"] = (previous["center"] * count + curve["center"]) / (count + 1)
                previous["radius"] = (previous["radius"] * count + curve["radius"]) / (count + 1)
                previous["rmse"] = max(previous["rmse"], curve["rmse"])
                previous["support"] = min(previous["support"], curve["support"])
                previous["point_count"] += curve["point_count"]
                previous["contour_count"] += 1
                break
        else:
            merged.append(curve)
    return merged, {"contours_examined": min(len(contours), MAX_CONTOURS),
                    "contours_total": len(contours), "contour_limit_reached": len(contours) > MAX_CONTOURS}


def _hough_circles(analysis: np.ndarray, foreground: np.ndarray, stroke: float,
                   curves: list[dict], profile: str = 'regional_masked') -> dict:
    """Recover rings whose contours are broken by intersecting axes/leaders.

    Hough supplies proposals only. Acceptance requires angular coverage of ink
    on the source analysis raster after a local centre-line fit. Missing long
    arcs cannot become full circles merely because Hough votes for one.
    """
    if min(analysis.shape) < 30 or not cv2.countNonZero(foreground):
        return {"hough_circle_proposals": 0, "hough_circles_added": 0}
    proposals, search_stats = build_circle_queue(analysis, foreground, stroke, profile)
    # The public queue already maps regional centers/radii to analysis pixels.
    proposal_scale = 1.
    hough_count = len(proposals)
    distance = cv2.distanceTransform(255 - foreground, cv2.DIST_L2, 3)
    profiled_centers: list[np.ndarray] = []
    radial_proposal_count = 0

    def add_radial_proposals(center: np.ndarray) -> None:
        """Concentric rings cannot share a Hough centre due to minDist.

        Explore radii only at a centre already justified by a complete ring;
        the proposals still pass the same independent full-ring validation.
        """
        nonlocal radial_proposal_count
        if any(np.linalg.norm(center - previous) <= max(2., stroke) for previous in profiled_centers):
            return
        profiled_centers.append(center.copy())
        maximum = int(min(center[0], center[1], foreground.shape[1] - 1 - center[0],
                          foreground.shape[0] - 1 - center[1]))
        radii = np.arange(10, maximum + 1, dtype=float)
        if not len(radii):
            return
        angles = np.linspace(0., 2 * np.pi, 144, endpoint=False)
        probes = np.rint(center + radii[:, None, None] * np.column_stack((np.cos(angles), np.sin(angles)))).astype(int)
        scores = np.mean(distance[probes[:, :, 1], probes[:, :, 0]] <= max(.75, stroke * .35), axis=1)
        eligible = np.flatnonzero(scores >= .9)
        for group in np.split(eligible, np.flatnonzero(np.diff(eligible) > 1) + 1):
            if not len(group):
                continue
            radius = float(np.average(radii[group], weights=scores[group]))
            proposal_center = (center + .5) * proposal_scale - .5
            proposals.append([float(proposal_center[0]), float(proposal_center[1]), radius * proposal_scale])
            radial_proposal_count += 1

    for curve in curves:
        if curve["kind"] == "circle":
            add_radial_proposals(curve["center"])
    added = 0
    validation_rejections: dict[str, int] = {}
    validation_records = []
    for proposal_index, (cx, cy, proposal_radius) in enumerate(proposals):
        if proposal_index >= 300:
            break
        center = (np.array([cx, cy]) + .5) / proposal_scale - .5
        radius = proposal_radius / proposal_scale
        curve, reason = validate_circle_proposal(center, radius, foreground, distance, stroke, curves, _fit_circle)
        validation_records.append({'index': proposal_index, 'center_analysis_px': center.tolist(), 'radius_analysis_px': radius, 'reason': reason})
        if curve is None:
            validation_rejections[reason] = validation_rejections.get(reason, 0)+1
        else:
            if proposal_index >= hough_count: curve['method'] = 'radial_profile_angular_centerline_fit'
            curves.append(curve); added += 1; add_radial_proposals(curve['center'])
    return {**search_stats, "hough_circles_added": added,
            "radial_profile_proposals": radial_proposal_count,
            "hough_circle_proposal_scale": proposal_scale,
            "circle_validation_examined": len(validation_records), 'circle_validation_rejections': validation_rejections,
            'circle_validation_records': validation_records,
            "hough_circle_limit_reached": len(proposals) > 300 or search_stats['cheap_screen_limit_reached'] or search_stats['validation_queue_dropped'] > 0}


def _refine_line(endpoints: np.ndarray, foreground: np.ndarray, stroke: float) -> np.ndarray:
    direction = endpoints[1] - endpoints[0]
    length = np.linalg.norm(direction)
    direction /= length
    normal = np.array([-direction[1], direction[0]])
    margin = max(2., stroke * .75)
    left, top = np.maximum(np.floor(endpoints.min(axis=0) - margin).astype(int), 0)
    right, bottom = np.minimum(np.ceil(endpoints.max(axis=0) + margin).astype(int) + 1,
                               [foreground.shape[1], foreground.shape[0]])
    yy, xx = np.nonzero(foreground[top:bottom, left:right])
    points = np.column_stack((xx + left, yy + top)).astype(float)
    delta = points - endpoints[0]
    along, across = delta @ direction, delta @ normal
    points = points[(np.abs(across) <= margin) & (along >= -margin) & (along <= length + margin)]
    if len(points) < 4:
        return endpoints
    vx, vy, x, y = cv2.fitLine(points.astype(np.float32), cv2.DIST_HUBER, 0, .01, .01).reshape(-1).astype(float)
    fitted_direction = np.array([vx, vy])
    if abs(np.dot(fitted_direction, direction)) < .995:
        return endpoints
    if np.dot(fitted_direction, direction) < 0:
        fitted_direction *= -1
    center = np.array([x, y])
    projected = (points - center) @ fitted_direction
    result = np.array([center + projected.min() * fitted_direction,
                       center + projected.max() * fitted_direction])
    # Hough can stop short even on continuous ink, particularly near a crossing
    # where the skeleton branches. Follow only observed centre-line foreground;
    # the first white pixel terminates extension, preserving real dash gaps.
    for endpoint_index, sign in ((0, -1), (1, 1)):
        origin = result[endpoint_index].copy()
        for extension in np.arange(.5, max(20., stroke * 8), .5):
            point = origin + sign * extension * fitted_direction
            ix, iy = np.rint(point).astype(int)
            if not (0 <= ix < foreground.shape[1] and 0 <= iy < foreground.shape[0]) or not foreground[iy, ix]:
                break
            result[endpoint_index] = point
    return result


def _merge_lines(lines: list[dict], stroke: float, foreground: np.ndarray) -> list[dict]:
    merged: list[dict] = []
    for line in sorted(lines, key=lambda value: -np.linalg.norm(value["points"][1] - value["points"][0])):
        points = line["points"]
        direction = points[1] - points[0]
        direction /= np.linalg.norm(direction)
        for previous in merged:
            old = previous["points"]
            old_direction = old[1] - old[0]
            old_direction /= np.linalg.norm(old_direction)
            if abs(np.dot(direction, old_direction)) < math.cos(math.radians(2)):
                continue
            normal = np.array([-old_direction[1], old_direction[0]])
            perpendicular = np.abs((points - old[0]) @ normal)
            if float(perpendicular.max()) > max(1.4, stroke * .45):
                continue
            interval = np.sort((points - old[0]) @ old_direction)
            old_length = np.linalg.norm(old[1] - old[0])
            if interval[0] > old_length + 2 or interval[1] < -2:
                continue
            # Do not collapse two genuinely separate nearby parallel strokes:
            # a connecting middle strip must contain actual ink.
            middle = (points.mean(axis=0) + old.mean(axis=0)) / 2
            check = np.rint(middle).astype(int)
            if (0 <= check[0] < foreground.shape[1] and 0 <= check[1] < foreground.shape[0]
                    and not foreground[check[1], check[0]]):
                continue
            endpoints = np.array([old[0] + min(0., interval[0]) * old_direction,
                                  old[0] + max(old_length, interval[1]) * old_direction])
            previous["points"] = _refine_line(endpoints, foreground, stroke)
            previous["hough_count"] += line["hough_count"]
            break
        else:
            merged.append(line)
    return merged


def _line_on_curve(points: np.ndarray, curves: list[dict], stroke: float) -> bool:
    samples = points[0] + np.linspace(0, 1, 15)[:, None] * (points[1] - points[0])
    for curve in curves:
        radial = np.abs(np.linalg.norm(samples - curve["center"], axis=1) - curve["radius"])
        if np.mean(radial <= max(1.5, stroke * .85)) < .85:
            continue
        angle = np.arctan2(samples[:, 1] - curve["center"][1], samples[:, 0] - curve["center"][0])
        inside = ((angle - curve["start"]) % (2 * np.pi)) <= curve["sweep"] + .03
        if curve["kind"] == "circle" or inside.mean() >= .85:
            return True
    return False


def _short_connectors(lines: list[dict], foreground: np.ndarray, stroke: float,
                      minimum_length: float) -> tuple[list[dict], dict]:
    """Observe short caps missed by Hough between long parallel strokes.

    Only an existing endpoint can seed a cap. Its perpendicular projection must
    lie on another observed long segment, and the complete intervening raster
    must contain a thin, continuous stroke. This is raster evidence, not an
    inferred closure of an open contour or a semantic object-edge decision.
    """
    stats = {"short_connector_pairs_examined": 0, "short_connectors_added": 0,
             "short_connector_limit_reached": False}
    minimum_support = max(minimum_length * 2, stroke * 12)
    supports = [(index, line["points"]) for index, line in enumerate(lines)
                if np.linalg.norm(line["points"][1] - line["points"][0]) >= minimum_support]
    if len(supports) < 2:
        return [], stats
    indices = np.asarray([item[0] for item in supports])
    endpoints = np.asarray([item[1] for item in supports])
    lengths = np.linalg.norm(endpoints[:, 1] - endpoints[:, 0], axis=1)
    directions = (endpoints[:, 1] - endpoints[:, 0]) / lengths[:, None]
    minimum_gap = max(5., stroke * 2.5)
    maximum_gap = min(minimum_length * .95, max(16., stroke * 8))
    added: list[dict] = []

    def sample(points: np.ndarray) -> np.ndarray:
        pixels = np.rint(points).astype(int)
        inside = ((pixels[..., 0] >= 0) & (pixels[..., 0] < foreground.shape[1])
                  & (pixels[..., 1] >= 0) & (pixels[..., 1] < foreground.shape[0]))
        values = np.zeros(inside.shape, dtype=bool)
        values[inside] = foreground[pixels[..., 1][inside], pixels[..., 0][inside]] > 0
        return values

    for source_position, (source_index, observed) in enumerate(supports):
        parallel = np.abs(directions @ directions[source_position]) >= math.cos(math.radians(2))
        for endpoint in observed:
            along = np.sum((endpoint - endpoints[:, 0]) * directions, axis=1)
            projections = endpoints[:, 0] + along[:, None] * directions
            gaps = np.linalg.norm(projections - endpoint, axis=1)
            nearby = (parallel & (indices != source_index) & (gaps >= minimum_gap)
                      & (gaps <= maximum_gap) & (along >= 0) & (along <= lengths))
            for target_position in np.flatnonzero(nearby):
                if stats["short_connector_pairs_examined"] >= MAX_SHORT_CONNECTOR_PAIRS or len(added) >= MAX_SHORT_CONNECTORS:
                    stats["short_connector_limit_reached"] = True
                    stats["short_connectors_added"] = len(added)
                    return added, stats
                stats["short_connector_pairs_examined"] += 1
                end = projections[target_position]
                delta = end - endpoint
                gap = gaps[target_position]
                axis = delta / gap
                normal = np.array([-axis[1], axis[0]])
                # Ignore the supporting longitudinal strokes when testing cap
                # thickness; their ink cannot justify the middle of the gap.
                margin = max(2., stroke * .95)
                positions = np.linspace(margin, gap - margin, max(7, int(gap * 2)))
                core = endpoint + positions[:, None] * axis
                transverse = np.arange(-max(4., stroke * 2.5), max(4., stroke * 2.5) + .25, .5)
                best = None
                for offset in np.arange(-max(1., stroke * .75), max(1., stroke * .75) + .25, .5):
                    shifted = core + offset * normal
                    if sample(shifted).mean() < .95:
                        continue
                    profile = sample(shifted[:, None, :] + transverse[None, :, None] * normal)
                    # A cap must remain thin, with white space on both sides.
                    # Filled triangles/blobs and parallel-stroke bleed fail.
                    widths = profile.sum(axis=1) * .5
                    if (np.max(widths) > max(3., stroke * 1.75)
                            or np.ptp(widths) > max(1.5, stroke * .65)
                            or profile[:, :2].any() or profile[:, -2:].any()):
                        continue
                    centers = np.asarray([np.mean(transverse[row]) for row in profile])
                    if np.ptp(centers) > max(1.5, stroke * .65):
                        continue
                    center_offset = offset + float(np.median(centers))
                    trial = np.asarray([endpoint, end]) + center_offset * normal
                    if (np.any(trial < 0) or np.any(trial > [foreground.shape[1] - 1, foreground.shape[0] - 1])):
                        continue
                    if not sample(trial[0] + np.linspace(0, 1, max(12, int(gap * 2)))[:, None] * delta).all():
                        continue
                    score = (float(np.ptp(centers)), abs(center_offset))
                    if best is None or score < best[0]:
                        best = (score, trial)
                if best is None:
                    continue
                points = best[1]
                # Retain one observation when both parallel endpoints seed the
                # same cap, and do not duplicate an existing Hough segment.
                if any(min(np.max(np.linalg.norm(points - old["points"], axis=1)),
                           np.max(np.linalg.norm(points - old["points"][::-1], axis=1)))
                       <= max(2., stroke) for old in added):
                    continue
                midpoint = points.mean(axis=0)
                duplicate = False
                for old in lines:
                    old_delta = old["points"][1] - old["points"][0]
                    old_length = np.linalg.norm(old_delta)
                    old_axis = old_delta / old_length
                    if abs(np.dot(axis, old_axis)) < math.cos(math.radians(3)):
                        continue
                    relative = midpoint - old["points"][0]
                    if (-stroke <= np.dot(relative, old_axis) <= old_length + stroke
                            and abs(old_axis[0] * relative[1] - old_axis[1] * relative[0]) <= max(1.5, stroke * .5)):
                        # A short Hough observation may find the same cap first.
                        # Preserve the independently checked thin-ink support
                        # and its two source lines instead of losing provenance
                        # solely because the geometric segment already exists.
                        if old.get("method") == "automatic_proposal_mask_short_hough_centerline_fit":
                            old.update(points=points, method="short_connector_parallel_endpoint_ink_support",
                                       support_line_indices=[source_index, int(indices[target_position])],
                                       ink_support_ratio=1.)
                        duplicate = True
                        break
                if duplicate:
                    continue
                added.append({"points": points, "hough_count": 0,
                              "method": "short_connector_parallel_endpoint_ink_support",
                              "support_line_indices": [source_index, int(indices[target_position])],
                              "ink_support_ratio": 1.})
    stats["short_connectors_added"] = len(added)
    return added, stats


def detect_geometry(processed_image: Path, output_dir: Path, *, prefix: str = "page-001",
                    pixel_to_paper_mm: list | None = None, search_profile: str = 'regional_masked') -> dict:
    """Save pixel-space line/circle/arc candidates and a review overlay.

    The optional homogeneous matrix is retained as provenance only. An
    anisotropic or projective transform does not preserve circles. All output
    coordinates remain in original processed-image pixels, without rounding.
    Existing outputs and unsafe prefixes are refused before any output writes.
    """
    started = time.perf_counter()
    if search_profile not in ('ranked_global', 'regional_masked'):
        raise ValueError('Unknown geometry search profile')
    processed_image, output_dir = Path(processed_image), Path(output_dir)
    if not isinstance(prefix, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", prefix):
        raise ValueError("Prefix must contain 1-80 ASCII letters, digits, hyphens or underscores")
    if not processed_image.is_file() or processed_image.stat().st_size > MAX_BYTES:
        raise ValueError("Geometry input must be an existing raster of at most 50 MiB")
    matrix = None
    if pixel_to_paper_mm is not None:
        matrix = np.asarray(pixel_to_paper_mm, dtype=float)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-15:
            raise ValueError("pixel_to_paper_mm must be a finite invertible 3x3 matrix")
    names = {"preview": f"{prefix}-geometry-preview.png", "report": f"{prefix}-geometry.json"}
    for name in names.values():
        if (output_dir / name).exists() or (output_dir / name).is_symlink():
            raise FileExistsError(f"Geometry detection never overwrites existing artifacts: {output_dir / name}")
    with Image.open(processed_image) as source:
        if source.width * source.height > MAX_PIXELS or min(source.size) < 1:
            raise ValueError("Geometry raster exceeds 25 megapixel limit")
        if source.format not in {"PNG", "JPEG"}:
            raise ValueError("Geometry detection expects an intake PNG/JPEG raster")
        if "A" in source.getbands() or "transparency" in source.info:
            background = Image.new("RGBA", source.size, "white")
            background.alpha_composite(source.convert("RGBA"))
            gray = np.asarray(background.convert("L")).copy()
        else:
            gray = np.asarray(source.convert("L")).copy()
    scale = min(1., ANALYSIS_SIDE / max(gray.shape))
    analysis = gray
    if scale < 1:
        # An explicit isotropic map avoids silently turning circles into tiny
        # ellipses when destination integer dimensions round independently.
        affine = np.array([[scale, 0, (scale - 1) / 2], [0, scale, (scale - 1) / 2]])
        analysis = cv2.warpAffine(gray, affine,
                                  (math.ceil(gray.shape[1] * scale), math.ceil(gray.shape[0] * scale)),
                                  flags=cv2.INTER_AREA, borderValue=255)
    foreground = cv2.threshold(analysis, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    _, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= 6
    keep[0] = False
    foreground = (keep[labels].astype(np.uint8) * 255)
    skeleton, skeleton_limited = _skeleton(foreground)
    distance = cv2.distanceTransform(foreground, cv2.DIST_L2, 3)
    thicknesses = distance[skeleton > 0] * 2
    stroke = max(1., min(12., float(np.median(thicknesses)) if len(thicknesses) else 1.))
    prepared_at = time.perf_counter()
    curves, curve_stats = _curves(foreground, stroke)
    curve_stats.update(_hough_circles(analysis, foreground, stroke, curves, search_profile))
    curves_at = time.perf_counter()
    minimum_length = max(20., min(analysis.shape) * .025)
    detected = cv2.HoughLinesP(skeleton, 1, np.pi / 720, threshold=max(12, int(minimum_length * .55)),
                               minLineLength=minimum_length, maxLineGap=2)
    hough = [] if detected is None else list(detected.reshape(-1, 4).astype(float))
    hough.sort(key=lambda coordinates: -float(np.linalg.norm(coordinates[2:] - coordinates[:2])))
    global_hough_count = len(hough)
    short, regional_line_stats = propose_short_lines(skeleton, foreground, minimum_length,
        max(0, MAX_HOUGH_LINES-len(hough)), search_profile)
    # Keep the old long-segment priority and global cap. New short evidence is
    # independently refined/merged on the untouched source foreground.
    hough.extend(short)
    lines = []
    for source_index, coordinates in enumerate(hough[:MAX_HOUGH_LINES]):
        points = coordinates.reshape(2, 2).copy()
        lower = max(10., minimum_length*.55) if search_profile == 'regional_masked' else minimum_length
        if np.linalg.norm(points[1] - points[0]) < lower or _line_on_curve(points, curves, stroke):
            continue
        points = _refine_line(points, foreground, stroke)
        if points[1, 0] < points[0, 0] or (points[1, 0] == points[0, 0] and points[1, 1] < points[0, 1]):
            points = points[::-1].copy()
        item = {"points": points, "hough_count": 1}
        if source_index >= global_hough_count: item['method'] = 'automatic_proposal_mask_short_hough_centerline_fit'
        lines.append(item)
    lines = _merge_lines(lines, stroke, foreground)
    lines = _merge_lines(lines, stroke, foreground)
    connectors, connector_stats = _short_connectors(lines, foreground, stroke, minimum_length)
    lines.extend(line for line in connectors if not _line_on_curve(line["points"], curves, stroke))
    connector_stats["short_connectors_added"] = sum("support_line_indices" in line for line in lines)
    lines_at = time.perf_counter()
    candidates = []
    uncertainty = max(.75, stroke / 2) / scale
    for line in lines:
        points = (line["points"] + .5) / scale - .5
        candidates.append({"kind": "line", "points_px": points.tolist(),
                           "uncertainty_px": uncertainty,
                           "evidence": {"method": line.get("method", "morphological_skeleton_hough_centerline_fit"),
                                        "hough_segments_merged": line["hough_count"],
                                        "length_px": float(np.linalg.norm(points[1] - points[0])),
                                        "line_style": "unknown"}})
        if "support_line_indices" in line:
            candidates[-1]["evidence"].update(
                supporting_geometry_ids=[f"{prefix}-geometry-{index + 1:05d}"
                                         for index in line["support_line_indices"]],
                ink_support_ratio=line["ink_support_ratio"])
    for curve in curves:
        candidate = {"kind": curve["kind"], "center_px": ((curve["center"] + .5) / scale - .5).tolist(),
                     "radius_px": curve["radius"] / scale,
                     "uncertainty_px": max(uncertainty, curve["rmse"] / scale),
                     "evidence": {"method": curve.get("method", "contour_circle_least_squares_with_angular_ink_support"),
                                  "radial_rmse_px": curve["rmse"] / scale,
                                  "observed_sweep_deg": float(np.rad2deg(curve["sweep"])),
                                  "ink_support_ratio": curve["support"],
                                  "contours_merged": curve["contour_count"],
                                  "contour_points": curve["point_count"], "line_style": "unknown"}}
        if curve["kind"] == "arc":
            candidate.update(start_angle_deg=float(np.rad2deg(curve["start"])),
                             sweep_angle_deg=float(np.rad2deg(curve["sweep"])))
        if "maximum_missing_arc_deg" in curve:
            candidate["evidence"]["maximum_missing_arc_deg"] = curve["maximum_missing_arc_deg"]
        candidates.append(candidate)
    source_hash = _sha256(processed_image)
    for index, candidate in enumerate(candidates, 1):
        candidate.update(id=f"{prefix}-geometry-{index:05d}", classification="unclassified",
                         source_ref={"image": processed_image.name, "sha256": source_hash,
                                     "coordinate_space": "processed_image_pixels"})
    issues = [{"code": "geometry_semantics_unverified",
               "message": "Candidates may include text, frames, hatching and dimension strokes; object geometry and axes are unclassified."},
              {"code": "dimensions_not_reconstructed", "message": "No OCR dimension values, dimension associations or exact geometry constraints have been inferred."}]
    if not candidates:
        issues.append({"code": "no_geometry_candidates", "message": "No supported line/circle/arc candidates; inspect the source raster."})
    if skeleton_limited:
        issues.append({"code": "skeleton_iteration_limit", "message": "Thick filled cores excluded after 64 skeleton iterations."})
    if (len(hough) > MAX_HOUGH_LINES or curve_stats["contour_limit_reached"]
            or curve_stats.get("hough_circle_limit_reached") or connector_stats["short_connector_limit_reached"]
            or regional_line_stats['regional_short_line_limit_reached']):
        issues.append({"code": "geometry_candidate_limit", "message": "Analysis limit reached; candidate detection is incomplete."})
    if scale < 1:
        issues.append({"code": "geometry_analysis_downsampled", "message": "Detection used an isotropically downsampled raster; coordinates and uncertainty map back to source pixels."})
    overview = Image.fromarray(gray).convert("RGB")
    overview.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    sx, sy = overview.width / gray.shape[1], overview.height / gray.shape[0]
    drawing = ImageDraw.Draw(overview)
    for candidate in candidates:
        if candidate["kind"] == "line":
            drawing.line([(point[0] * sx, point[1] * sy) for point in candidate["points_px"]], fill="#dd4422", width=2)
        else:
            x, y = candidate["center_px"]
            radius = candidate["radius_px"]
            box = [(x - radius) * sx, (y - radius) * sy, (x + radius) * sx, (y + radius) * sy]
            if candidate["kind"] == "circle":
                drawing.ellipse(box, outline="#168c55", width=2)
            else:
                start = candidate["start_angle_deg"]
                drawing.arc(box, start, start + candidate["sweep_angle_deg"], fill="#3355dd", width=2)
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_png(output_dir / names["preview"], overview)
    report = {"schema_version": "1.0", "status": "needs_review", "source": processed_image.name,
              "source_sha256": source_hash, "processed_size_px": [gray.shape[1], gray.shape[0]],
              "coordinate_system": "pixel centers; top-left origin; x right, y down; angles clockwise from +x",
              "pixel_to_paper_mm": None if matrix is None else matrix.tolist(),
              "coordinate_policy": "pixel candidates only; optional transform retained without metric circle assumptions",
              "analysis": {"size_px": [analysis.shape[1], analysis.shape[0]], "scale": scale,
                           "processed_to_analysis": [[scale, 0., (scale - 1) / 2],
                                                     [0., scale, (scale - 1) / 2], [0., 0., 1.]],
                           "analysis_to_processed": [[1 / scale, 0., (.5 / scale) - .5],
                                                     [0., 1 / scale, (.5 / scale) - .5], [0., 0., 1.]],
                           "estimated_stroke_width_px": stroke / scale,
                           "minimum_line_length_px": minimum_length / scale,
                           "hough_segments": len(hough), 'global_hough_segments': global_hough_count,
                           **curve_stats, **connector_stats, **regional_line_stats},
              "candidates": candidates, "issues": issues,
              "artifacts": names, "preview_scale_xy": [sx, sy],
              "artifact_sha256": {"preview": _sha256(output_dir / names["preview"])},
              "timings": {"preparation_seconds": prepared_at - started,
                          "curves_seconds": curves_at - prepared_at,
                          "lines_seconds": lines_at - curves_at,
                          "total_seconds": time.perf_counter() - started},
              "dependencies": {"opencv": cv2.__version__, "numpy": np.__version__}}
    with (output_dir / names["report"]).open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return report
