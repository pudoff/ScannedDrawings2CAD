"""Conservative scan preparation with reversible pixel-coordinate provenance.

This module does not recognize drawing views or title-block semantics. Its grid
candidate is only a region worth inspecting and retaining at full resolution.
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
ANALYSIS_SIDE = 1800


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _check_size(width: int, height: int) -> None:
    if width < 1 or height < 1 or width * height > MAX_PIXELS:
        raise ValueError("Image or transformed canvas exceeds 25 megapixel limit")


def _analysis_image(gray: np.ndarray) -> tuple[np.ndarray, float]:
    scale = min(1.0, ANALYSIS_SIDE / max(gray.shape))
    if scale == 1.0:
        return gray, scale
    return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), scale


def _binary(gray: np.ndarray) -> np.ndarray:
    # Keep the grayscale input for OCR: thresholding must never become its only source.
    shortest = min(gray.shape)
    if shortest < 3:
        return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    window = min(101, max(15, shortest // 30)) | 1
    window = min(window, shortest if shortest % 2 else shortest - 1)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, window, 12)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order[np.searchsorted(cumulative, cumulative[-1] / 2)]])


def _estimate_skew(gray: np.ndarray) -> dict:
    small, scale = _analysis_image(gray)
    height, width = small.shape
    result = {"method": "long-line-hough-consensus", "analysis_scale": scale,
              "accepted": False, "estimated_skew_deg": None, "confidence": 0.0,
              "line_count": 0, "inlier_count": 0, "support_ratio": 0.0,
              "applied_rotation_deg": 0.0}
    edges = cv2.Canny(small, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 1800,
                           threshold=max(25, min(width, height) // 20),
                           minLineLength=max(35, min(width, height) * 0.15),
                           maxLineGap=max(3, min(width, height) // 100))
    if lines is None or len(lines) < 4:
        result["reason"] = "insufficient_long_lines"
        return result
    # OpenCV 4 returns (N, 1, 4); OpenCV 5 returns (N, 4).
    lines = lines.reshape(-1, 4).astype(float)
    dx, dy = lines[:, 2] - lines[:, 0], lines[:, 3] - lines[:, 1]
    lengths = np.hypot(dx, dy)
    angles = np.degrees(np.arctan2(dy, dx))
    residuals = (angles + 45) % 90 - 45
    eligible = np.abs(residuals) <= 10
    result["line_count"] = int(len(lines))
    if int(eligible.sum()) < 4:
        result["reason"] = "insufficient_near_axis_lines"
        return result
    estimate = _weighted_median(residuals[eligible], lengths[eligible])
    inliers = eligible & (np.abs(residuals - estimate) <= 0.8)
    support = float(lengths[inliers].sum() / lengths.sum())
    result.update(estimated_skew_deg=round(estimate, 6), inlier_count=int(inliers.sum()),
                  support_ratio=round(support, 6))
    if int(inliers.sum()) < 4 or support < 0.75:
        result["reason"] = "inconsistent_line_directions"
        return result
    horizontal = inliers & (np.abs(dx) >= np.abs(dy))
    vertical = inliers & ~horizontal
    result["horizontal_inliers"] = int(horizontal.sum())
    result["vertical_inliers"] = int(vertical.sum())
    # Parallel geometry alone can be intentionally oblique. Require two axis
    # families rather than mistaking a sloping edge/hatch for page orientation.
    if int(horizontal.sum()) < 2 or int(vertical.sum()) < 2:
        result["reason"] = "insufficient_orthogonal_support"
        return result
    spread = _weighted_median(np.abs(residuals[inliers] - estimate), lengths[inliers])
    result.update(accepted=True, confidence=round(support * max(0.0, 1 - spread / 0.8), 6),
                  reason="consensus", median_residual_deg=round(spread, 6))
    return result


def _expanded_rotation(width: int, height: int, clockwise_deg: float) -> tuple[np.ndarray, tuple[int, int]]:
    affine = cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), -clockwise_deg, 1)
    matrix = np.vstack((affine, [0, 0, 1]))
    corners = np.array([[0, 0, 1], [width - 1, 0, 1],
                        [width - 1, height - 1, 1], [0, height - 1, 1]], dtype=float)
    transformed = (matrix @ corners.T).T[:, :2]
    left, top = np.floor(transformed.min(axis=0) + 1e-9)
    right, bottom = np.ceil(transformed.max(axis=0) - 1e-9)
    size = int(right - left + 1), int(bottom - top + 1)
    _check_size(*size)
    matrix[0, 2] -= left
    matrix[1, 2] -= top
    return matrix, size


def _perspective(quad: list, width: int, height: int) -> tuple[np.ndarray, tuple[int, int]]:
    try:
        points = np.asarray(quad, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Perspective quad must contain four finite (x, y) points") from exc
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError("Perspective quad must contain four finite (x, y) points")
    if (points < 0).any() or (points[:, 0] > width - 1).any() or (points[:, 1] > height - 1).any():
        raise ValueError("Perspective quad must be inside the original image")
    edges = np.roll(points, -1, axis=0) - points
    next_edges = np.roll(edges, -1, axis=0)
    crosses = edges[:, 0] * next_edges[:, 1] - edges[:, 1] * next_edges[:, 0]
    area = abs(float(np.dot(points[:, 0], np.roll(points[:, 1], -1))
                     - np.dot(points[:, 1], np.roll(points[:, 0], -1)))) / 2
    if (crosses <= 1e-6).any() or area < 64 or np.min(np.linalg.norm(edges, axis=1)) < 4:
        raise ValueError("Perspective quad must be a nondegenerate convex TL/TR/BR/BL polygon")
    if (points[:2, 1].mean() >= points[2:, 1].mean()
            or points[[0, 3], 0].mean() >= points[[1, 2], 0].mean()):
        raise ValueError("Perspective quad point order must be TL/TR/BR/BL")
    target_width = round(max(np.linalg.norm(edges[0]), np.linalg.norm(edges[2]))) + 1
    target_height = round(max(np.linalg.norm(edges[1]), np.linalg.norm(edges[3]))) + 1
    size = int(target_width), int(target_height)
    _check_size(*size)
    destination = np.array([[0, 0], [size[0] - 1, 0],
                            [size[0] - 1, size[1] - 1], [0, size[1] - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(points.astype(np.float32), destination)
    if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError("Perspective quad produces a singular transform")
    return matrix, size


def _quarter_rotation(gray: np.ndarray, clockwise_deg: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = gray.shape
    matrix = {
        0: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        90: [[0, -1, height - 1], [1, 0, 0], [0, 0, 1]],
        180: [[-1, 0, width - 1], [0, -1, height - 1], [0, 0, 1]],
        270: [[0, 1, 0], [-1, 0, width - 1], [0, 0, 1]],
    }[clockwise_deg]
    return np.ascontiguousarray(np.rot90(gray, -clockwise_deg // 90)), np.array(matrix, dtype=float)


def _expand_connected_grid(foreground: np.ndarray, seed: list[int]) -> tuple[list[int], dict]:
    """Follow a right-hand grid seed into its connected left-hand table cells.

    Search the full width, but retain the seed's vertical extent so a connected
    page frame cannot grow the crop to the whole sheet. Two horizontal crossings
    and a vertical boundary on the left are required; proximity alone is unused.
    """
    height, width = foreground.shape
    left, top, right, bottom = seed
    band_top, band_bottom = max(0, top - 3), min(height, bottom + 3)
    band = foreground[band_top:band_bottom, :]
    horizontal = cv2.morphologyEx(band, cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, width // 12), 1)))
    vertical = cv2.morphologyEx(band, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, height // 30))))
    grid = cv2.bitwise_or(horizontal, vertical)
    # Close at most two analysis pixels of broken ink; this does not bridge a
    # meaningful whitespace gap between independent tables.
    grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(grid, connectivity=8)
    seed_labels = np.unique(labels[top - band_top:bottom - band_top, left:right])
    extended_left = left
    supporting_crossings, supporting_boundaries = 0, 0
    for label in seed_labels:
        if label == 0 or stats[label, cv2.CC_STAT_LEFT] >= left - 3:
            continue
        component = np.where(labels == label, 255, 0).astype(np.uint8)
        hstats = cv2.connectedComponentsWithStats(cv2.bitwise_and(horizontal, component), connectivity=8)[2][1:]
        crossings = [r for r in hstats if r[0] < left - 3 and r[0] + r[2] > left + 6
                     and top - band_top <= r[1] < bottom - band_top]
        vstats = cv2.connectedComponentsWithStats(cv2.bitwise_and(vertical, component), connectivity=8)[2][1:]
        boundaries = [r for r in vstats if r[0] < left - 3 and r[3] >= (bottom - top) * 0.25]
        if len(crossings) >= 2 and boundaries:
            extended_left = min(extended_left, int(stats[label, cv2.CC_STAT_LEFT]))
            supporting_crossings += len(crossings)
            supporting_boundaries += len(boundaries)
    return [extended_left, top, right, bottom], {
        "method": "connected_grid_in_full_width_seed_band",
        "expanded_left": extended_left < left,
        "horizontal_boundary_crossings": supporting_crossings,
        "left_vertical_boundaries": supporting_boundaries,
        "maximum_ink_gap_analysis_px": 2,
        "vertical_extent_policy": "unchanged_from_seed",
    }


def _regions(gray: np.ndarray, binary: np.ndarray) -> list[dict]:
    foreground = binary < 128
    xs, ys = np.flatnonzero(foreground.any(axis=0)), np.flatnonzero(foreground.any(axis=1))
    if not len(xs) or not len(ys):
        return []
    regions = [{"id": "content", "kind": "content_bbox", "confidence": 1.0,
                "confidence_meaning": "bounds of all thresholded foreground, not a semantic class",
                "bbox_px": [int(xs[0]), int(ys[0]), int(xs[-1] + 1), int(ys[-1] + 1)],
                "semantics_verified": False}]
    small, scale = _analysis_image(gray)
    height, width = small.shape
    xoffset, yoffset = int(width * 0.4), int(height * 0.55)
    small_foreground = 255 - _binary(small)
    crop = small_foreground[yoffset:, xoffset:]
    if min(crop.shape) < 12:
        return regions
    horizontal = cv2.morphologyEx(crop, cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, width // 12), 1)))
    vertical = cv2.morphologyEx(crop, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, height // 30))))
    hstats = cv2.connectedComponentsWithStats(horizontal, connectivity=8)[2][1:]
    # Ignore the page-wide frame: it alone is not evidence of a title block.
    hlines = [r for r in hstats if width * 0.15 <= r[2] < crop.shape[1] * 0.98 and r[3] <= 12]
    if len(hlines) < 3:
        return regions
    left, top = min(int(r[0]) for r in hlines), min(int(r[1]) for r in hlines)
    right = max(int(r[0] + r[2]) for r in hlines)
    bottom = max(int(r[1] + r[3]) for r in hlines)
    vstats = cv2.connectedComponentsWithStats(vertical, connectivity=8)[2][1:]
    vlines = [r for r in vstats if left - 3 <= r[0] < right + 3
              and r[1] < bottom and r[1] + r[3] > top and r[3] >= (bottom - top) * 0.5]
    intersections = cv2.bitwise_and(horizontal, vertical)[top:bottom, left:right]
    crossing_count = cv2.connectedComponents(intersections, connectivity=8)[0] - 1
    if len(vlines) < 3 or crossing_count < 6 or bottom - top < height * 0.03:
        return regions
    def to_full_bbox(box):
        return [max(0, int(math.floor((box[0] - 3) / scale))),
                max(0, int(math.floor((box[1] - 3) / scale))),
                min(gray.shape[1], int(math.ceil((box[2] + 3) / scale))),
                min(gray.shape[0], int(math.ceil((box[3] + 3) / scale)))]

    seed = [left + xoffset, top + yoffset, right + xoffset, bottom + yoffset]
    expanded, expansion = _expand_connected_grid(small_foreground, seed)
    expansion["seed_bbox_px"] = to_full_bbox(seed)
    bbox = to_full_bbox(expanded)
    regions.append({"id": "title-block-candidate", "kind": "title_block_grid_candidate",
                    "bbox_px": bbox, "semantics_verified": False,
                    "confidence": round(min(0.85, 0.45 + 0.025 * (len(hlines) + len(vlines))
                                            + 0.005 * crossing_count), 3),
                    "confidence_meaning": "uncalibrated grid support score; not a title-block probability",
                    "evidence": {"horizontal_lines": len(hlines), "vertical_lines": len(vlines),
                                 "intersections": crossing_count, "search_region": "bottom-right seed",
                                 "connected_expansion": expansion}})
    return regions


def _save_png(path: Path, image: Image.Image) -> None:
    with path.open("xb") as stream:
        image.save(stream, format="PNG")


def prepare_scan(source_image: Path, output_dir: Path, *, prefix: str = "page-001",
                 rotation_deg: int = 0, perspective_quad: list | None = None,
                 auto_deskew: bool = True) -> dict:
    """Prepare one intake raster, keeping original pixel provenance and OCR crops.

    Positive quarter-turns are clockwise. An optional quad is TL/TR/BR/BL in the
    original raster, before rotation. Coordinates address pixel centers (x right,
    y down); region boxes use exclusive right/bottom edges. EXIF was applied by
    intake and is deliberately not applied again here. Existing outputs are refused.
    """
    started = time.perf_counter()
    source_image, output_dir = Path(source_image), Path(output_dir)
    if not isinstance(prefix, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", prefix):
        raise ValueError("Prefix must contain 1-80 ASCII letters, digits, hyphens or underscores")
    if type(rotation_deg) is not int or rotation_deg not in (0, 90, 180, 270):
        raise ValueError("Explicit rotation must be 0, 90, 180 or 270 clockwise degrees")
    if type(auto_deskew) is not bool:
        raise ValueError("auto_deskew must be a boolean")
    if not source_image.is_file():
        raise ValueError(f"Source raster does not exist: {source_image}")
    if source_image.stat().st_size > MAX_BYTES:
        raise ValueError("Source raster exceeds 50 MiB limit")
    names = {"grayscale": f"{prefix}-processed.png", "binary": f"{prefix}-binary.png",
             "regions_overview": f"{prefix}-regions.png", "report": f"{prefix}-preprocessing.json",
             "roi_content": f"{prefix}-roi-content.png",
             "roi_title_block_candidate": f"{prefix}-roi-title-block-candidate.png"}
    for name in names.values():
        target = output_dir / name
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Preprocessing never overwrites existing artifacts: {target}")
    source_hash = _sha256(source_image)
    with Image.open(source_image) as original:
        _check_size(*original.size)
        if original.format not in {"PNG", "JPEG"}:
            raise ValueError("Preprocessing expects an intake PNG/JPEG raster")
        input_size = list(original.size)
        if "A" in original.getbands() or "transparency" in original.info:
            rgba = original.convert("RGBA")
            background = Image.new("RGBA", original.size, "white")
            background.alpha_composite(rgba)
            gray = np.asarray(background.convert("L")).copy()
        else:
            gray = np.asarray(original.convert("L")).copy()
    forward = np.eye(3)
    steps = [{"operation": "grayscale", "size_px": input_size,
              "exif_policy": "already_applied_by_intake", "alpha_background": "white"}]
    issues = []
    if perspective_quad is not None:
        matrix, size = _perspective(perspective_quad, *input_size)
        gray = cv2.warpPerspective(gray, matrix, size, flags=cv2.INTER_LINEAR, borderValue=255)
        forward = matrix @ forward
        steps.append({"operation": "explicit_perspective", "matrix": matrix.tolist(),
                      "quad_input_px": np.asarray(perspective_quad, dtype=float).tolist(),
                      "size_px": list(size), "metric_aspect_ratio_verified": False})
        issues.append({"code": "perspective_aspect_unverified",
                       "message": "Quad rectification uses observed edge lengths; physical aspect ratio is unverified."})
    if rotation_deg:
        gray, matrix = _quarter_rotation(gray, rotation_deg)
        forward = matrix @ forward
        steps.append({"operation": "explicit_rotation", "clockwise_deg": rotation_deg,
                      "matrix": matrix.tolist(), "size_px": [gray.shape[1], gray.shape[0]]})
    else:
        issues.append({"code": "orientation_unverified",
                       "message": "Line directions cannot determine upright text or distinguish 0/90/180/270 degrees."})
    deskew = {"method": "disabled", "accepted": False, "estimated_skew_deg": None,
              "applied_rotation_deg": 0.0, "confidence": 0.0}
    if auto_deskew:
        deskew = _estimate_skew(gray)
        if deskew["accepted"] and abs(deskew["estimated_skew_deg"]) >= 0.15:
            correction = -deskew["estimated_skew_deg"]
            try:
                matrix, size = _expanded_rotation(gray.shape[1], gray.shape[0], correction)
            except ValueError:
                deskew.update(accepted=False, reason="expanded_canvas_limit")
                issues.append({"code": "deskew_canvas_limit",
                               "message": "Deskew skipped: expanded canvas exceeds the pixel limit."})
            else:
                gray = cv2.warpAffine(gray, matrix[:2], size, flags=cv2.INTER_LINEAR, borderValue=255)
                forward = matrix @ forward
                deskew["applied_rotation_deg"] = correction
                steps.append({"operation": "deskew", "clockwise_deg": correction,
                              "matrix": matrix.tolist(), "size_px": list(size)})
        elif not deskew["accepted"]:
            issues.append({"code": "deskew_unresolved",
                           "message": f"No automatic correction: {deskew['reason']}."})
    else:
        issues.append({"code": "deskew_disabled", "message": "Automatic skew estimation was explicitly disabled."})
    binary = _binary(gray)
    steps.append({"operation": "adaptive_gaussian_threshold", "coordinate_transform": "identity",
                  "grayscale_preserved": True, "denoising": "none_to_preserve_thin_lines"})
    regions = _regions(gray, binary)
    if not regions:
        issues.append({"code": "no_foreground", "message": "No thresholded content; inspect the source raster."})
    issues.append({"code": "region_semantics_unverified",
                   "message": "Content bounds and optional grid candidate are not recognized drawing views or title-block fields."})
    output_dir.mkdir(parents=True, exist_ok=True)
    gray_image = Image.fromarray(gray)
    _save_png(output_dir / names["grayscale"], gray_image)
    _save_png(output_dir / names["binary"], Image.fromarray(binary))
    artifacts = {key: names[key] for key in ("grayscale", "binary", "regions_overview", "report")}
    overview = gray_image.copy()
    overview.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    overview = overview.convert("RGB")
    sx, sy = overview.width / gray.shape[1], overview.height / gray.shape[0]
    drawing = ImageDraw.Draw(overview)
    for region in regions:
        key = "roi_" + region["id"].replace("-", "_")
        bbox = region["bbox_px"]
        _save_png(output_dir / names[key], gray_image.crop(tuple(bbox)))
        artifacts[key] = names[key]
        region["roi_artifact"] = names[key]
        region["roi_to_processed"] = [[1, 0, bbox[0]], [0, 1, bbox[1]], [0, 0, 1]]
        display_box = [bbox[0] * sx, bbox[1] * sy, (bbox[2] - 1) * sx, (bbox[3] - 1) * sy]
        color = "#e67e22" if region["id"] != "content" else "#208060"
        drawing.rectangle(display_box, outline=color, width=2)
        drawing.text((display_box[0] + 3, display_box[1] + 3), region["id"], fill=color)
    _save_png(output_dir / names["regions_overview"], overview)
    report = {
        "schema_version": "1.0", "status": "needs_review" if issues else "succeeded",
        "source": source_image.name, "source_sha256": source_hash,
        "input_size_px": input_size, "processed_size_px": [gray.shape[1], gray.shape[0]],
        "coordinate_system": "pixel centers; top-left origin; x right, y down",
        "bbox_convention": "[left, top, right_exclusive, bottom_exclusive] in processed pixels",
        "transforms": {"input_to_processed": forward.tolist(),
                       "processed_to_input": np.linalg.inv(forward).tolist()},
        "parameters": {"rotation_deg_clockwise": rotation_deg, "auto_deskew": auto_deskew,
                       "perspective_quad_input_px": None if perspective_quad is None else
                       np.asarray(perspective_quad, dtype=float).tolist(), "max_pixels": MAX_PIXELS},
        "steps": steps, "deskew": deskew, "regions": regions, "issues": issues,
        "overview_size_px": list(overview.size), "overview_scale_xy": [sx, sy],
        "artifacts": artifacts,
        "artifact_sha256": {key: _sha256(output_dir / name)
                            for key, name in artifacts.items() if key != "report"},
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "dependencies": {"opencv": cv2.__version__, "numpy": np.__version__},
    }
    with (output_dir / names["report"]).open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return report
