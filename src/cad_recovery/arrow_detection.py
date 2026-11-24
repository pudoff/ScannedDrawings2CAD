"""Conservative filled arrow observations attached to raster line candidates.

No OCR nominal or accepted CAD entity is used by this detector. A tapered
transverse ink profile plus a connected thin shaft supplies the evidence.
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

MAX_PIXELS = 25_000_000
MAX_BYTES = 50 * 1024 * 1024
MAX_LINES = 2500
MAX_PROFILE_SAMPLES = 12_000_000
MAX_LINE_LENGTH = 8192
PROFILE_RADIUS = 40
MARGIN = 48


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _profile(foreground: np.ndarray, first: np.ndarray, unit: np.ndarray,
             length: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    along = np.arange(-MARGIN, math.ceil(length) + MARGIN + 1, dtype=np.float32)
    across = np.arange(-PROFILE_RADIUS, PROFILE_RADIUS + 1, dtype=np.float32)
    normal = np.array([-unit[1], unit[0]])
    coordinates = first + along[:, None] * unit
    map_x = (coordinates[:, 0][None, :] + across[:, None] * normal[0]).astype(np.float32)
    map_y = (coordinates[:, 1][None, :] + across[:, None] * normal[1]).astype(np.float32)
    # Sampling a local crop keeps OpenCV's remap dimension limit independent
    # of unusual, very long and narrow source rasters under the pixel budget.
    left = max(0, min(foreground.shape[1] - 1, math.floor(float(map_x.min())) - 1))
    top = max(0, min(foreground.shape[0] - 1, math.floor(float(map_y.min())) - 1))
    right = max(left + 1, min(foreground.shape[1], math.ceil(float(map_x.max())) + 2))
    bottom = max(top + 1, min(foreground.shape[0], math.ceil(float(map_y.max())) + 2))
    strip = cv2.remap(foreground[top:bottom, left:right], map_x - left, map_y - top, cv2.INTER_NEAREST,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
    # Permit up to two pixels of centre-line fitting error, without following
    # a detached text stroke farther from the supporting line.
    offsets = np.array([0, -1, 1, -2, 2])
    center_values = strip[PROFILE_RADIUS + offsets]
    centers = PROFILE_RADIUS + offsets[np.argmax(center_values, axis=0)]
    columns = np.arange(strip.shape[1])
    upper = np.clip(centers[None, :] - np.arange(PROFILE_RADIUS + 1)[:, None], 0, strip.shape[0] - 1)
    lower = np.clip(centers[None, :] + np.arange(PROFILE_RADIUS + 1)[:, None], 0, strip.shape[0] - 1)
    up_white = ~strip[upper, columns]
    down_white = ~strip[lower, columns]
    up = np.where(up_white.any(axis=0), up_white.argmax(axis=0), PROFILE_RADIUS + 1)
    down = np.where(down_white.any(axis=0), down_white.argmax(axis=0), PROFILE_RADIUS + 1)
    widths = np.maximum(0, up + down - 1).astype(np.float32)
    widths[~center_values.any(axis=0)] = 0
    midpoints = (centers - PROFILE_RADIUS + (down - up) / 2).astype(np.float32)
    # One-pixel intersection/noise spikes do not constitute a triangle.
    widths = cv2.medianBlur(widths[None, :], 3)[0]
    return along, widths, midpoints


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _shaft_support(widths: np.ndarray, position: float, direction: int, stroke: float) -> float:
    indices = np.rint(position + direction * np.arange(5, 22)).astype(int)
    valid = (indices >= 0) & (indices < len(widths))
    if valid.mean() < .8:
        return 0.
    sampled = widths[indices[valid]]
    return float(np.mean((sampled >= 1) & (sampled <= stroke + 2)))


def _line_arrows(foreground: np.ndarray, line: dict) -> list[dict]:
    points = np.asarray(line["points_px"], dtype=float)
    first, second = points
    length = float(np.linalg.norm(second - first))
    unit = (second - first) / length
    normal = np.array([-unit[1], unit[0]])
    along, widths, midpoints = _profile(foreground, first, unit, length)
    supported = widths[(widths > 0) & (widths < PROFILE_RADIUS)]
    if len(supported) < 20:
        return []
    stroke = max(1., min(10., float(np.quantile(supported, .25))))
    threshold = max(stroke + 2, stroke * 1.6, 4.)
    arrows = []
    for begin, end in _runs(widths >= threshold):
        if not 4 <= end - begin <= 72:
            continue
        x = np.arange(begin, end, dtype=float)
        y = widths[begin:end].astype(float)
        # Crossing extension lines may produce a short, very tall spike.
        # Exclude those saturated samples, retaining an explicit fit fraction.
        keep = y < PROFILE_RADIUS * 1.5
        if keep.sum() < 4:
            continue
        design = np.column_stack((x - x.mean(), np.ones(len(x))))
        fit = np.linalg.lstsq(design[keep], y[keep], rcond=None)[0]
        for _ in range(2):
            residual = np.abs(y - design @ fit)
            keep = keep & (residual <= max(2., float(np.median(residual[keep])) * 2.5))
            if keep.sum() < 4:
                break
            fit = np.linalg.lstsq(design[keep], y[keep], rcond=None)[0]
        if keep.sum() < 4 or keep.mean() < .7:
            continue
        slope, intercept = fit
        variance = float(np.sum((y[keep] - y[keep].mean()) ** 2))
        error = float(np.sum((y[keep] - (design @ fit)[keep]) ** 2))
        r_squared = 1 - error / variance if variance > 0 else 0.
        if abs(slope) < .18 or abs(slope) > 2.5 or r_squared < .68:
            continue
        if float(np.ptp(y[keep])) < 3 or float(y[keep].max()) < stroke + 3:
            continue
        center_slope = float(np.linalg.lstsq(design[keep], midpoints[begin:end][keep], rcond=None)[0][0])
        # Two crossing strokes often create a one-sided wedge in a width
        # profile. A filled arrow expands on both sides of its shaft; its
        # transverse midpoint remains approximately parallel to that shaft.
        if abs(center_slope) > max(.075, abs(slope) * .18):
            continue
        sign = 1 if slope < 0 else -1
        base_index = float(x[keep].min() if sign > 0 else x[keep].max())
        tip_index = float(x.mean() + (1. - intercept) / slope)
        arrow_length = abs(tip_index - base_index)
        base_width = float(intercept + slope * (base_index - x.mean()))
        if not 5 <= arrow_length <= 90 or not .65 <= arrow_length / base_width <= 6:
            continue
        behind_base = np.rint(base_index - sign * np.arange(2, 5)).astype(int)
        if ((behind_base < 0).any() or (behind_base >= len(widths)).any()
                or np.median(widths[behind_base]) > max(stroke + 2, base_width * .65)):
            # A triangular arrow has a base edge. A softly tapering scan line
            # or the flank of a crossing has no comparable abrupt back edge.
            continue
        shaft = max(_shaft_support(widths, base_index, -sign, stroke),
                    _shaft_support(widths, tip_index, sign, stroke))
        if shaft < .75:
            continue
        tip_distance = tip_index - MARGIN
        base_distance = base_index - MARGIN
        tip = first + unit * tip_distance
        base = first + unit * base_distance
        if not (0 <= tip[0] < foreground.shape[1] and 0 <= tip[1] < foreground.shape[0]):
            continue
        uncertainty = max(1.5, math.sqrt(error / keep.sum()) / abs(slope) + 1.)
        arrows.append({"tip_px": tip.tolist(), "direction": (sign * unit).tolist(),
                       "polygon_px": [tip.tolist(), (base + normal * base_width / 2).tolist(),
                                      (base - normal * base_width / 2).tolist()],
                       "uncertainty_px": uncertainty,
                       "confidence": float(min(.95, .55 + .25 * r_squared + .15 * shaft)),
                       "evidence": {"method": "filled_triangle_transverse_profile_with_connected_shaft",
                                    "geometry_ids": [line["id"]], "taper_r_squared": r_squared,
                                    "profile_inlier_fraction": float(keep.mean()),
                                    "transverse_midpoint_slope": center_slope,
                                    "shaft_support_ratio": shaft, "shaft_width_px": stroke,
                                    "arrow_length_px": arrow_length, "base_width_px": base_width,
                                    "confidence_calibrated": False}})
    return arrows


def detect_arrows(source_image: Path, output_dir: Path, *, geometry: dict,
                  prefix: str = "page-001", text_polygons: list | None = None) -> dict:
    """Write full-resolution pixel candidates; refuse mismatched provenance.

    Text polygons, if supplied, are polygons in the same processed image pixel
    coordinate space. Observations whose tip or centroid lies inside one are
    suppressed. They need not contain recognized strings.
    """
    started = time.perf_counter()
    source_image, output_dir = Path(source_image), Path(output_dir)
    if not isinstance(prefix, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", prefix):
        raise ValueError("Prefix must contain 1-80 ASCII letters, digits, hyphens or underscores")
    if not source_image.is_file() or source_image.stat().st_size > MAX_BYTES:
        raise ValueError("Arrow input must be an existing raster of at most 50 MiB")
    source_hash = _sha256(source_image)
    if geometry.get("source_sha256") != source_hash:
        raise ValueError("Arrow geometry source_sha256 does not match the processed image")
    names = {"report": f"{prefix}-arrows.json", "preview": f"{prefix}-arrows-preview.png"}
    for name in names.values():
        if (output_dir / name).exists() or (output_dir / name).is_symlink():
            raise FileExistsError(f"Arrow detection never overwrites existing artifacts: {output_dir / name}")
    with Image.open(source_image) as source:
        if source.width * source.height > MAX_PIXELS or min(source.size) < 1:
            raise ValueError("Arrow raster exceeds 25 megapixel limit")
        if source.format not in {"PNG", "JPEG"}:
            raise ValueError("Arrow detection expects an intake PNG/JPEG raster")
        if geometry.get("processed_size_px") != list(source.size):
            raise ValueError("Arrow geometry processed_size_px does not match the processed image")
        if "A" in source.getbands() or "transparency" in source.info:
            white = Image.new("RGBA", source.size, "white")
            white.alpha_composite(source.convert("RGBA"))
            gray = np.asarray(white.convert("L")).copy()
        else:
            gray = np.asarray(source.convert("L")).copy()
    polygons = []
    for polygon in text_polygons or []:
        array = np.asarray(polygon, dtype=np.float32)
        if array.ndim != 2 or array.shape[1:] != (2,) or len(array) < 3 or not np.isfinite(array).all():
            raise ValueError("text_polygons must contain finite polygons with at least three points")
        polygons.append(array)
    lines = []
    for candidate in geometry.get("candidates", []):
        if candidate.get("kind") != "line":
            continue
        points = np.asarray(candidate.get("points_px"), dtype=float)
        if points.shape != (2, 2) or not np.isfinite(points).all() or not candidate.get("id"):
            raise ValueError("Arrow geometry lines require an id and two finite pixel points")
        # Keep allocation and sampling bounded even for hand-edited reports.
        if ((points < -MARGIN).any() or (points[:, 0] > gray.shape[1] + MARGIN).any()
                or (points[:, 1] > gray.shape[0] + MARGIN).any()):
            raise ValueError("Arrow geometry line points fall outside the processed image")
        if np.linalg.norm(points[1] - points[0]) >= 20:
            lines.append(candidate)
    foreground = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    candidates = []
    sampled = examined = excluded = 0
    limited = len(lines) > MAX_LINES
    for line in lines[:MAX_LINES]:
        length = float(np.linalg.norm(np.diff(np.asarray(line["points_px"]), axis=0)))
        if length > MAX_LINE_LENGTH:
            limited = True
            continue
        cost = math.ceil(length) + 2 * MARGIN + 1
        if sampled + cost > MAX_PROFILE_SAMPLES:
            limited = True
            break
        sampled += cost
        examined += 1
        for candidate in _line_arrows(foreground, line):
            tip = tuple(candidate["tip_px"])
            centroid = tuple(np.mean(candidate["polygon_px"], axis=0))
            if any(cv2.pointPolygonTest(polygon, tip, False) >= 0
                   or cv2.pointPolygonTest(polygon, centroid, False) >= 0 for polygon in polygons):
                excluded += 1
                continue
            for previous in candidates:
                distance = np.linalg.norm(np.asarray(previous["tip_px"]) - candidate["tip_px"])
                if distance <= max(3., previous["uncertainty_px"], candidate["uncertainty_px"]) and np.dot(previous["direction"], candidate["direction"]) > .96:
                    previous["evidence"]["geometry_ids"] = sorted(set(previous["evidence"]["geometry_ids"] + candidate["evidence"]["geometry_ids"]))
                    break
            else:
                candidates.append(candidate)
    for index, candidate in enumerate(candidates, 1):
        candidate.update(id=f"{prefix}-arrow-{index:05d}", classification="filled_arrow_candidate",
                         source_ref={"image": source_image.name, "sha256": source_hash,
                                     "coordinate_space": "processed_image_pixels"})
    issues = [{"code": "arrow_candidates_unverified", "message": "Filled-arrow observations require review and dimension association; no CAD dimensions are accepted."},
              {"code": "arrow_detector_scope", "message": "Only filled arrows with a supported thin shaft are searched along geometry line candidates. Open arrows and missing geometry lines may be missed."}]
    if limited:
        issues.append({"code": "arrow_search_limit", "message": "Line/profile analysis limit reached; arrow search is incomplete."})
    if not candidates:
        issues.append({"code": "no_arrow_candidates", "message": "No supported filled arrows found; absence is not proof that the page has no dimensions."})
    overview = Image.fromarray(gray).convert("RGB")
    overview.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    sx, sy = overview.width / gray.shape[1], overview.height / gray.shape[0]
    drawing = ImageDraw.Draw(overview)
    for candidate in candidates:
        points = [(p[0] * sx, p[1] * sy) for p in candidate["polygon_px"]]
        drawing.line(points + points[:1], fill="#e33122", width=2)
        x, y = points[0]
        drawing.ellipse((x - 3, y - 3, x + 3, y + 3), outline="#155fdd", width=2)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / names["preview"]).open("xb") as stream:
        overview.save(stream, format="PNG")
    report = {"schema_version": "1.0", "status": "needs_review", "source": source_image.name,
              "source_sha256": source_hash, "processed_size_px": [gray.shape[1], gray.shape[0]],
              "coordinate_system": "pixel centers; top-left origin; x right, y down",
              "direction_convention": "unit vector from arrow base toward tip",
              "analysis": {"lines_examined": examined, "profile_samples": sampled,
                           "text_exclusions": excluded, "search_limited": limited,
                           "full_resolution": True},
              "candidates": candidates, "issues": issues, "artifacts": names,
              "preview_scale_xy": [sx, sy],
              "artifact_sha256": {"preview": _sha256(output_dir / names["preview"])},
              "timings": {"total_seconds": time.perf_counter() - started}}
    with (output_dir / names["report"]).open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return report
