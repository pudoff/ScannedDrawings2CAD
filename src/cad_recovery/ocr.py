"""Offline tiled OCR observations. No text is accepted as a CAD dimension or scale."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from decimal import Decimal
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .notation import parse_notation
from .title_recognition import propose_title_block

MAX_PIXELS = 25_000_000
MAX_WINDOWS = 64
MAX_OBSERVATIONS = 10_000


def _starts(length: int, size: int, overlap: int) -> list[int]:
    if length <= size:
        return [0]
    return sorted(set([*range(0, length - size + 1, size - overlap), length - size]))


def _windows(width, height, title_bbox, tile_size, overlap):
    regions = [("page", [0, 0, width, height])]
    if title_bbox is not None:
        regions.append(("title_block_candidate", title_bbox))
    windows, seen = [], set()
    for region, (left, top, right, bottom) in regions:
        for y in _starts(bottom - top, tile_size, overlap):
            for x in _starts(right - left, tile_size, overlap):
                box = (left + x, top + y, min(right, left + x + tile_size), min(bottom, top + y + tile_size))
                if box not in seen:
                    windows.append({"id": f"window-{len(windows) + 1:03d}", "region": region, "bbox_px": list(box)})
                    seen.add(box)
                if len(windows) > MAX_WINDOWS:
                    raise ValueError("OCR requires too many windows; process smaller page regions")
    return windows


def _matrix(value, width, height):
    if value is None:
        return None
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-15:
        raise ValueError("OCR coordinate transform must be a finite invertible 3x3 matrix")
    corners = np.array([[0, 0, 1], [width, 0, 1], [width, height, 1], [0, height, 1]])
    denominators = (corners @ matrix.T)[:, 2]
    if not (np.all(denominators > 1e-10) or np.all(denominators < -1e-10)):
        raise ValueError("OCR coordinate transform has a horizon in the page")
    return matrix


def _project(points, matrix):
    points = np.asarray(points, dtype=float)
    projected = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    if np.any(np.abs(projected[:, 2]) < 1e-10):
        raise ValueError("OCR polygon intersects the transform horizon")
    result = projected[:, :2] / projected[:, 2:]
    if not np.isfinite(result).all():
        raise ValueError("OCR coordinate projection is not finite")
    return result.tolist()


def _bbox(polygon):
    points = np.asarray(polygon)
    return [*points.min(axis=0).tolist(), *points.max(axis=0).tolist()]


def _iou(a, b):
    left, top, right, bottom = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    area = max(0, right - left) * max(0, bottom - top)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - area
    return area / union if union > 0 else 0


def _conflicting_overlap(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    # A seam-truncated reading can be much shorter than another reading of the
    # same title cell. IoU alone would miss that ambiguity.
    return _iou(a, b) >= 0.6 or (smaller > 0 and intersection / smaller >= 0.8)


def _notation_key(raw):
    """Only fully parsed, lossless values can establish spelling equivalence."""
    parsed = parse_notation(raw)
    if parsed["status"] != "parsed":
        return None
    # The parser exports JSON floats. Do not let their rounding make distinct
    # high-precision source numbers appear equal during deduplication.
    for token in re.findall(r"[0-9]+(?:[.,][0-9]+)?|[.,][0-9]+", parsed["normalized"]):
        number = Decimal(token.replace(",", "."))
        if number != Decimal(str(float(number))):
            return None
    return parsed["kind"], json.dumps(parsed["values"], sort_keys=True, allow_nan=False)


def _deduplicate(observations):
    """Retain all evidence; collapse overlapping equal or equivalent notations.

    The representative keeps the highest-scoring raw spelling and coordinates.
    Every spelling and observation remains accessible through ``raw_variants``.
    Equivalence compares kind and every parsed value, including deviations,
    thread pitch, fit, count and the explicit scale terms. It repairs no text.
    """
    notation_keys = {raw: _notation_key(raw) for raw in {item["raw"] for item in observations}}
    by_id = {item["id"]: item for item in observations}
    lines = []
    for observation in sorted(observations, key=lambda item: -item["confidence"]):
        duplicate = next((line for line in lines
                          if (line["raw"] == observation["raw"] or
                              (notation_keys[line["raw"]] is not None and
                               notation_keys[line["raw"]] == notation_keys[observation["raw"]]))
                          and _iou(line["bbox_px"], observation["bbox_px"]) >= 0.6), None)
        if duplicate is not None:
            duplicate["observation_ids"].append(observation["id"])
            variant = next((item for item in duplicate["raw_variants"] if item["raw"] == observation["raw"]), None)
            if variant is None:
                duplicate["raw_variants"].append({"raw": observation["raw"], "observation_ids": [observation["id"]]})
            else:
                variant["observation_ids"].append(observation["id"])
        else:
            lines.append({**observation, "observation_ids": [observation["id"]], "conflicting_line_ids": [],
                          "raw_variants": [{"raw": observation["raw"], "observation_ids": [observation["id"]]}]})
    lines.sort(key=lambda line: (round(line["bbox_px"][1] / 10), line["bbox_px"][0], line["raw"]))
    for i, line in enumerate(lines, 1):
        line["id"] = f"ocr-line-{i:04d}"
    for i, line in enumerate(lines):
        for other in lines[i + 1:]:
            # A conflicting third reading may overlap a retained variant more
            # closely than the representative; merging must not hide it.
            if line["raw"] != other["raw"] and any(
                    _conflicting_overlap(by_id[first]["bbox_px"], by_id[second]["bbox_px"])
                    for first in line["observation_ids"] for second in other["observation_ids"]):
                line["conflicting_line_ids"].append(other["id"])
                other["conflicting_line_ids"].append(line["id"])
    return lines


def recognize_text(source_image: Path, output_dir: Path, *, engine, prefix="page-001",
                   pixel_to_paper_mm=None, processed_to_input=None, title_bbox=None,
                   tile_size=1536, overlap=192, time_budget_seconds=240.0) -> dict:
    """Observe all text, retain conflicts and propose notation/title fields for review.

    ``engine.recognize`` accepts a BGR uint8 crop and returns raw/confidence/polygon_px.
    Transformations map pixel centers. Input is the full prepared grayscale image.
    """
    started = time.perf_counter()
    source_image, output_dir = Path(source_image), Path(output_dir)
    if Path(prefix).name != prefix or not prefix or prefix in {".", ".."}:
        raise ValueError("OCR prefix must be a filename component")
    if type(tile_size) is not int or not 512 <= tile_size <= 2048:
        raise ValueError("OCR tile_size must be an integer between 512 and 2048")
    if type(overlap) is not int or not 32 <= overlap <= tile_size // 3:
        raise ValueError("OCR overlap must be between 32 and tile_size / 3")
    if not math.isfinite(time_budget_seconds) or not 0 < time_budget_seconds <= 600:
        raise ValueError("OCR time budget must be positive and at most 600 seconds")
    names = {"report": f"{prefix}-ocr.json", "preview": f"{prefix}-ocr-preview.png"}
    for name in names.values():
        if (output_dir / name).exists() or (output_dir / name).is_symlink():
            raise FileExistsError(f"OCR never overwrites existing artifacts: {output_dir / name}")
    if source_image.stat().st_size > 50 * 1024 * 1024:
        raise ValueError("OCR source exceeds 50 MiB")
    with Image.open(source_image) as opened:
        width, height = opened.size
        if not 0 < width * height <= MAX_PIXELS:
            raise ValueError("OCR raster exceeds the pixel limit")
        rgba = opened.convert("RGBA")
        image = Image.new("RGBA", rgba.size, "white")
        image.alpha_composite(rgba)
        image = image.convert("RGB")
    paper_matrix = _matrix(pixel_to_paper_mm, width, height)
    input_matrix = _matrix(processed_to_input, width, height)
    if title_bbox is not None:
        values = np.asarray(title_bbox, dtype=float)
        if (values.shape != (4,) or not np.isfinite(values).all() or
                not (0 <= values[0] < values[2] <= width and 0 <= values[1] < values[3] <= height)):
            raise ValueError("Title region must lie inside the prepared raster")
        title_bbox = [math.floor(values[0]), math.floor(values[1]), math.ceil(values[2]), math.ceil(values[3])]
    windows = _windows(width, height, title_bbox, tile_size, overlap)
    source_hash = hashlib.sha256(source_image.read_bytes()).hexdigest()
    observations, issues = [], []
    completed = 0
    for window in windows:
        if time.perf_counter() - started >= time_budget_seconds:
            issues.append({"code": "ocr_time_budget", "message": "Time budget reached between windows; OCR coverage is incomplete."})
            break
        left, top, right, bottom = window["bbox_px"]
        crop = np.asarray(image.crop((left, top, right, bottom)))[:, :, ::-1].copy()
        result = engine.recognize(crop)
        if len(observations) + len(result) > MAX_OBSERVATIONS:
            raise ValueError("OCR observation limit exceeded")
        for item in result:
            raw, confidence = item["raw"], float(item["confidence"])
            polygon = np.asarray(item["polygon_px"], dtype=float)
            if (not isinstance(raw, str) or len(raw) > 4096 or not math.isfinite(confidence) or not 0 <= confidence <= 1 or
                    polygon.shape != (4, 2) or not np.isfinite(polygon).all()):
                raise ValueError("OCR engine returned an invalid observation")
            if not raw.strip():
                continue
            if (polygon[:, 0].min() < 0 or polygon[:, 1].min() < 0 or
                    polygon[:, 0].max() > right - left or polygon[:, 1].max() > bottom - top):
                raise ValueError("OCR engine returned a polygon outside its crop")
            polygon += [left, top]
            box = _bbox(polygon)
            if box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError("OCR engine returned an empty polygon")
            observation = {
                "id": f"ocr-observation-{len(observations) + 1:04d}", "raw": raw, "confidence": confidence,
                "status": "observed", "polygon_px": polygon.tolist(), "bbox_px": box,
                "source_ref": {"image": source_image.name, "sha256": source_hash, "window_id": window["id"],
                               "window_bbox_px": window["bbox_px"], "coordinate_space": "processed_pixel_centers"},
                "uncertainty": {"recognition_score_calibrated": False, "localization_error_px": None},
            }
            if paper_matrix is not None:
                observation["polygon_paper_mm"] = _project(polygon, paper_matrix)
            if input_matrix is not None:
                observation["polygon_input_px"] = _project(polygon, input_matrix)
            observations.append(observation)
        completed += 1
    lines = _deduplicate(observations)
    notations = []
    for line in lines:
        parsed = parse_notation(line["raw"])
        if parsed["status"] == "unparsed":
            continue
        box = line["bbox_px"]
        in_title = title_bbox is not None and title_bbox[0] <= (box[0] + box[2]) / 2 <= title_bbox[2] and title_bbox[1] <= (box[1] + box[3]) / 2 <= title_bbox[3]
        notations.append({"id": f"notation-{len(notations) + 1:04d}", "source_line_id": line["id"],
                          "source_ref": line["source_ref"], "polygon_px": line["polygon_px"],
                          "confidence": line["confidence"], "parsed": parsed,
                          "context": "title_block_candidate" if in_title else "drawing_or_notes",
                          "status": "ambiguous" if line["confidence"] < 0.65 or line["conflicting_line_ids"] or parsed["status"] != "parsed" else "proposed",
                          "accepted_as_dimension": False, "accepted_as_scale": False})
    # Keep conflicts as spatial obstacles, even though they cannot establish field values.
    title = propose_title_block(lines, title_bbox)
    issues.extend([
        {"code": "ocr_unverified", "message": "OCR scores are not calibrated correctness probabilities; all readings require review."},
        {"code": "notation_unassociated", "message": "Numbers may be dimensions, quantities, dates or notes. Geometry association and scale verification are pending."},
    ])
    if not lines:
        issues.append({"code": "ocr_no_text", "message": "No text detected; this does not prove the page contains no text."})
    if any(line["conflicting_line_ids"] for line in lines):
        issues.append({"code": "ocr_conflicting_readings", "message": "Overlapping crops produced different readings; alternatives retained."})
    output_dir.mkdir(parents=True, exist_ok=True)
    preview = image.copy()
    preview.thumbnail((1600, 1600))
    scale_x, scale_y = preview.width / width, preview.height / height
    draw = ImageDraw.Draw(preview)
    for line in lines:
        polygon = [(x * scale_x, y * scale_y) for x, y in line["polygon_px"]]
        color = "#b64a12" if line["confidence"] < 0.65 or line["conflicting_line_ids"] else "#00735c"
        draw.line([*polygon, polygon[0]], fill=color, width=2)
        draw.text(polygon[0], line["id"].removeprefix("ocr-line-"), fill=color)
    with (output_dir / names["preview"]).open("xb") as stream:
        preview.save(stream, format="PNG")
    report = {
        "schema_version": "1.0", "status": "needs_review", "source": source_image.name,
        "source_sha256": source_hash, "processed_size_px": [width, height],
        "coordinate_system": "pixel centers; top-left origin; x right, y down",
        "pixel_to_paper_mm": pixel_to_paper_mm, "processed_to_input": processed_to_input,
        "engine": engine.provenance, "parameters": {"tile_size": tile_size, "overlap": overlap, "time_budget_seconds": time_budget_seconds,
                                                   "time_budget_policy": "soft; checked before each window, no interruption of active inference"},
        "coverage": {"complete": completed == len(windows), "completed_windows": completed, "total_windows": len(windows)},
        "windows": windows, "observations": observations, "lines": lines, "notations": notations,
        "title_block": title, "issues": issues, "artifacts": names,
        "artifact_sha256": {"preview": hashlib.sha256((output_dir / names["preview"]).read_bytes()).hexdigest()},
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }
    with (output_dir / names["report"]).open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return report
