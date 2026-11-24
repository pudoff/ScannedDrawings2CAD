"""Conservative, reproducible visual verification of candidate drawing pairs.

Names propose candidates only. ``verified`` means the rendered views agree under
the recorded criteria, not that dimension values or CAD semantics were audited.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

import cv2
import ezdxf
import numpy as np
import pypdfium2 as pdfium
from PIL import Image, ImageOps

METHOD = "sift-similarity-ink-v1"
THRESHOLDS = {
    "inliers": 12, "interior_inliers": 8, "inlier_ratio": 0.45,
    "inlier_span_x": 0.25, "inlier_span_y": 0.25,
    "global_overlap": 0.80, "interior_overlap": 0.85,
    "worst_interior_tile_overlap": 0.65, "interior_ink_pixels": 100,
    "target_ink_visible": 0.98,
}


def _contained(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("Dataset path escapes its root")
    return path


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _normalize(image: Image.Image, max_side: int) -> np.ndarray:
    image = ImageOps.exif_transpose(image)
    if image.mode == "RGBA" or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        image = Image.alpha_composite(background, rgba)
    gray = np.asarray(image.convert("L"))
    y, x = np.where(gray < 210)
    if len(x) < 100:
        raise ValueError("Empty or insufficient drawing content")
    gray = gray[max(0, int(y.min()) - 3):int(y.max()) + 4,
                max(0, int(x.min()) - 3):int(x.max()) + 4]
    scale = max_side / max(gray.shape)
    return cv2.resize(gray, (max(1, round(gray.shape[1] * scale)),
                             max(1, round(gray.shape[0] * scale))),
                      interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)


def _render_pdf(source: Path | bytes, max_side: int) -> Image.Image:
    with pdfium.PdfDocument(source) as document:
        if len(document) != 1:
            raise ValueError("Multi-page input requires explicit page-to-DXF mapping")
        page = document[0]
        try:
            width, height = page.get_size()
            if not all(math.isfinite(v) and v > 0 for v in (width, height)):
                raise ValueError("Invalid page dimensions")
            bitmap = page.render(scale=max_side * 1.5 / max(width, height))
            try:
                return bitmap.to_pil().copy()
            finally:
                bitmap.close()
        finally:
            page.close()


def render_drawing(source: Path, max_side: int = 1200) -> np.ndarray:
    """Render one input page or DXF modelspace to a bounded grayscale image."""
    source = Path(source)
    if source.stat().st_size > 50 * 1024 * 1024:
        raise ValueError("Drawing exceeds the 50 MiB render limit")
    if source.suffix.lower() == ".dxf":
        from ezdxf.addons.drawing import Frontend, RenderContext, config, layout
        from ezdxf.addons.drawing.pymupdf import PyMuPdfBackend

        document = ezdxf.readfile(source)
        audit = document.audit()
        if audit.errors:
            raise ValueError("DXF contains unresolved audit errors")
        backend = PyMuPdfBackend()
        options = config.Configuration(
            background_policy=config.BackgroundPolicy.WHITE,
            color_policy=config.ColorPolicy.BLACK,
            hatching_timeout=5.0,
        )
        Frontend(RenderContext(document), backend, config=options).draw_layout(
            document.modelspace(), finalize=True)
        # A fixed bounded page avoids giant native CAD units creating giant PDFs.
        pdf = backend.get_pdf_bytes(layout.Page(297, 297,
                                                margins=layout.Margins.all(3)))
        image = _render_pdf(pdf, max_side)
    elif source.suffix.lower() == ".pdf":
        image = _render_pdf(source, max_side)
    else:
        with Image.open(source) as original:
            if original.width * original.height > 25_000_000:
                raise ValueError("Image exceeds the 25 megapixel render limit")
            image = ImageOps.exif_transpose(original).copy()
    return _normalize(image, max_side)


def _overlap(a: np.ndarray, b: np.ndarray, mask: np.ndarray,
             tolerance: int) -> tuple[float, float]:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (2 * tolerance + 1, 2 * tolerance + 1))
    near_a = cv2.dilate(a.astype(np.uint8), kernel).astype(bool)
    near_b = cv2.dilate(b.astype(np.uint8), kernel).astype(bool)
    selected_a, selected_b = a & mask, b & mask
    return (float(np.count_nonzero(selected_a & near_b) / max(1, selected_a.sum())),
            float(np.count_nonzero(selected_b & near_a) / max(1, selected_b.sum())))


def compare_renders(source: np.ndarray, target: np.ndarray) -> dict:
    """Register target to source and record reasons for verification/abstention."""
    cv2.setRNGSeed(0)
    detector = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.025)
    ka, da = detector.detectAndCompute(source, None)
    kb, db = detector.detectAndCompute(target, None)
    result = {"status": "quarantine", "method": METHOD, "reasons": [],
              "source_size": list(source.shape[::-1]),
              "target_size": list(target.shape[::-1]),
              "keypoints": [len(ka), len(kb)]}
    if da is None or db is None or min(len(ka), len(kb)) < THRESHOLDS["inliers"]:
        result["reasons"] = ["insufficient_features"]
        return result
    matches = cv2.BFMatcher().knnMatch(db, da, k=2)
    good = [m for group in matches if len(group) == 2
            for m, n in [group] if m.distance < 0.70 * n.distance]
    # Repeated glyphs must not vote multiple times for the same feature.
    unique = {}
    for match in sorted(good, key=lambda item: item.distance):
        unique.setdefault(match.trainIdx, match)
    good = list(unique.values())
    result["good_matches"] = len(good)
    if len(good) < THRESHOLDS["inliers"]:
        result["reasons"] = ["insufficient_matches"]
        return result
    a = np.float32([ka[m.trainIdx].pt for m in good])
    b = np.float32([kb[m.queryIdx].pt for m in good])
    tolerance = max(2, round(max(source.shape) / 600))
    matrix, inliers = cv2.estimateAffinePartial2D(
        b, a, method=cv2.RANSAC, ransacReprojThreshold=tolerance * 1.5,
        maxIters=3000, confidence=0.999, refineIters=20)
    if matrix is None or inliers is None:
        result["reasons"] = ["registration_failed"]
        return result
    scale = math.hypot(matrix[0, 0], matrix[1, 0])
    if not 0.25 <= scale <= 4:
        result["reasons"] = ["implausible_scale"]
        return result
    inlier_points = a[inliers.ravel().astype(bool)]
    height, width = source.shape
    xmin, xmax = round(width * 0.08), round(width * 0.92)
    ymin, ymax = round(height * 0.08), round(height * 0.78)
    interior = np.zeros(source.shape, bool)
    interior[ymin:ymax, xmin:xmax] = True
    interior_points = sum(xmin < x < xmax and ymin < y < ymax for x, y in inlier_points)
    warped = cv2.warpAffine(target, matrix, (width, height), borderValue=255)
    ink_a, ink_b = source < 210, warped < 210
    target_y, target_x = np.where(target < 210)
    projected = np.column_stack((target_x, target_y)) @ matrix[:, :2].T + matrix[:, 2]
    visible = ((projected[:, 0] >= -tolerance) & (projected[:, 0] < width + tolerance)
               & (projected[:, 1] >= -tolerance) & (projected[:, 1] < height + tolerance))
    global_overlap = _overlap(ink_a, ink_b, np.ones_like(interior), tolerance)
    interior_overlap = _overlap(ink_a, ink_b, interior, tolerance)
    tile_overlaps = []
    for ya, yb in zip(np.linspace(ymin, ymax, 4, dtype=int)[:-1],
                      np.linspace(ymin, ymax, 4, dtype=int)[1:]):
        for xa, xb in zip(np.linspace(xmin, xmax, 4, dtype=int)[:-1],
                          np.linspace(xmin, xmax, 4, dtype=int)[1:]):
            tile = np.zeros_like(interior)
            tile[ya:yb, xa:xb] = True
            if max(np.count_nonzero(ink_a & tile), np.count_nonzero(ink_b & tile)) >= 50:
                tile_overlaps.append(min(_overlap(ink_a, ink_b, tile, tolerance)))
    metrics = {
        "inliers": len(inlier_points), "interior_inliers": int(interior_points),
        "inlier_ratio": len(inlier_points) / len(good),
        "inlier_span_x": float(np.ptp(inlier_points[:, 0]) / width),
        "inlier_span_y": float(np.ptp(inlier_points[:, 1]) / height),
        "global_overlap": min(global_overlap), "interior_overlap": min(interior_overlap),
        "worst_interior_tile_overlap": min(tile_overlaps, default=0.0),
        "interior_ink_pixels": int(min((ink_a & interior).sum(), (ink_b & interior).sum())),
        "target_ink_visible": float(visible.mean()) if len(visible) else 0.0,
    }
    result.update(metrics=metrics, affine_target_to_source=matrix.tolist(),
                  tolerance_pixels=tolerance, global_directional_overlap=global_overlap,
                  interior_directional_overlap=interior_overlap,
                  interior_tile_overlaps=tile_overlaps)
    result["reasons"] = [f"below_{key}" for key, threshold in THRESHOLDS.items()
                         if metrics[key] < threshold]
    if not result["reasons"]:
        result["status"] = "verified"
    return result


def _save_overlay(source: np.ndarray, target: np.ndarray, evidence: dict, path: Path) -> None:
    matrix = evidence.get("affine_target_to_source")
    if matrix is None:
        return
    warped = cv2.warpAffine(target, np.array(matrix), source.shape[::-1], borderValue=255)
    # Input cyan, DXF red, agreement dark. White is empty space.
    overlay = np.stack([source, warped, warped], axis=-1)
    Image.fromarray(overlay).save(path)


def verify_pairs(dataset_root: Path, manifest: dict | Path, output_dir: Path, *,
                 max_side: int = 1200, limit: int | None = None) -> dict:
    """Check manifest candidates, save render evidence, and quarantine uncertainty.

    Full source hashes are checked against the audit manifest before rendering.
    This function never changes dataset files and never infers labels from names.
    """
    started = time.monotonic()
    root, output = Path(dataset_root).resolve(), Path(output_dir).resolve()
    if output.is_relative_to(root):
        raise ValueError("Verification output must be outside the dataset")
    if not 320 <= max_side <= 2400:
        raise ValueError("max_side must be between 320 and 2400")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if not isinstance(manifest, dict):
        manifest = json.loads(Path(manifest).read_text(encoding="utf-8"))
    records = {item["path"]: item for item in manifest["files"]}
    candidates = manifest["pair_candidates"][:limit]
    output.mkdir(parents=True, exist_ok=False)
    render_dir = output / "renders"
    render_dir.mkdir(exist_ok=True)
    report = {"schema_version": 1, "method": METHOD, "max_side": max_side,
              "thresholds": dict(THRESHOLDS), "candidate_count": len(manifest["pair_candidates"]),
              "scope": "rendered modelspace versus single input page; visual evidence only",
              "pairs": []}
    cv2.setNumThreads(1)
    cache = {}
    for index, candidate in enumerate(candidates):
        pair_started = time.monotonic()
        pair = {"input_path": candidate["input_path"], "target_path": candidate["target_path"],
                "set": candidate.get("set"), "status": "quarantine", "renders": {}}
        arrays = []
        try:
            for side in ("input", "target"):
                relative = pair[f"{side}_path"]
                record = records.get(relative)
                if not record or not record.get("sha256"):
                    raise ValueError("Missing audited source hash")
                path = _contained(root, relative)
                digest = _digest(path)
                pair[f"{side}_sha256"] = digest
                if digest != record["sha256"]:
                    raise ValueError("Source hash changed since dataset audit")
                if record.get("status") in {"error", "partial"} or record.get("errors"):
                    raise ValueError("Dataset audit marked source as erroneous")
                render_name = f"{digest}-{max_side}.png"
                if digest not in cache:
                    pixels = render_drawing(path, max_side)
                    Image.fromarray(pixels).save(render_dir / render_name)
                    if len(cache) >= 16:
                        cache.pop(next(iter(cache)))
                    cache[digest] = pixels
                arrays.append(cache[digest])
                pair["renders"][side] = f"renders/{render_name}"
            evidence = compare_renders(*arrays)
            pair["evidence"] = evidence
            pair["status"] = evidence["status"]
            if "affine_target_to_source" in evidence:
                overlay_name = f"pair-{index + 1:04d}-overlay.png"
                _save_overlay(*arrays, evidence, render_dir / overlay_name)
                pair["renders"]["overlay"] = f"renders/{overlay_name}"
        except (OSError, ValueError, RuntimeError, ezdxf.DXFError, cv2.error,
                Image.DecompressionBombError) as exc:
            pair["evidence"] = {"reasons": ["render_or_source_error"],
                                "error_type": type(exc).__name__, "error": str(exc)}
        pair["elapsed_seconds"] = round(time.monotonic() - pair_started, 3)
        report["pairs"].append(pair)
        # Keep a recoverable report during long runs without accepting unfinished pairs.
        if (index + 1) % 10 == 0:
            _write_report(report, output, started)
    _write_report(report, output, started)
    return report


def _write_report(report: dict, output: Path, started: float) -> None:
    report["summary"] = dict(Counter(pair["status"] for pair in report["pairs"]))
    report["processed_count"] = len(report["pairs"])
    report["complete"] = report["processed_count"] == report["candidate_count"]
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    temporary = output / "verification.json.tmp"
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output / "verification.json")
