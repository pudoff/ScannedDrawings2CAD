"""Independent, spatially matched evaluation of dimension association proposals.

This module deliberately does not import the association algorithm. Coordinates
refer to one hashed prepared raster, and omitted annotations never imply recall.
"""

from __future__ import annotations

from collections import Counter, deque
import math
import re

import cv2
import numpy as np


SUPPORTED_KINDS = {"linear", "radius", "diameter"}
STATUSES = {"proposed", "ambiguous", "unresolved", "unsupported"}
MAX_ITEMS = 2500


def _number(value, label, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    if positive and value <= 0:
        raise ValueError(f"{label} must be positive")
    return float(value)


def _points(value, size, label, *, polygon=False, empty=False):
    if not isinstance(value, list) or (not value and not empty):
        raise ValueError(f"{label} must contain points")
    if not value:
        return []
    if (polygon and not 3 <= len(value) <= 16) or (not polygon and len(value) != 2):
        raise ValueError(f"{label} has an invalid number of points")
    result = []
    for point in value:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError(f"{label} points must have two coordinates")
        x, y = [_number(item, label) for item in point]
        if not 0 <= x <= size[0] or not 0 <= y <= size[1]:
            raise ValueError(f"{label} is outside the prepared image")
        result.append([x, y])
    if polygon:
        contour = np.asarray(result, dtype=np.float32)
        if not cv2.isContourConvex(contour) or cv2.contourArea(contour) <= 0.01:
            raise ValueError(f"{label} must be a nondegenerate convex polygon")
    elif result[0] == result[1]:
        raise ValueError(f"{label} must contain distinct anchors")
    return result


def _entries(value, label, id_key):
    if not isinstance(value, list) or len(value) > MAX_ITEMS:
        raise ValueError(f"{label} must be a list of at most {MAX_ITEMS} entries")
    seen = set()
    for entry in value:
        if not isinstance(entry, dict):
            raise ValueError(f"{label} entries must be objects")
        identifier = entry.get(id_key)
        if not isinstance(identifier, str) or not identifier.strip() or identifier in seen:
            raise ValueError(f"{label} contains missing or duplicate {id_key}")
        seen.add(identifier)
    return value


def _inside(polygon, region):
    center = np.mean(polygon, axis=0)
    return cv2.pointPolygonTest(np.asarray(region, dtype=np.float32), tuple(center), False) >= 0


def _overlap(a, b):
    pa, pb = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    if np.any(pa.max(axis=0) < pb.min(axis=0)) or np.any(pb.max(axis=0) < pa.min(axis=0)):
        return 0.0
    intersection, _ = cv2.intersectConvexConvex(pa, pb)
    union = cv2.contourArea(pa) + cv2.contourArea(pb) - intersection
    return max(0.0, min(1.0, intersection / union)) if union > 0 else 0.0


def _match(predictions, expected, *, anchor_tolerance=None):
    """Maximum cardinality one-to-one matching; spatial IoU breaks ties.

    Text location, kind and nominal always gate an edge. The first evaluation
    pass additionally gates anchor accuracy, so a duplicate bad proposal cannot
    consume the ground truth before a good proposal merely by appearing first.
    """
    edges = []
    for prediction in predictions:
        candidates = []
        for index, truth in enumerate(expected):
            nominal = prediction.get("nominal_mm")
            if (prediction["kind"] != truth["kind"] or nominal is None or
                    not math.isclose(nominal, truth["nominal_mm"], rel_tol=1e-9, abs_tol=1e-12)):
                continue
            overlap = _overlap(prediction["text_polygon_px"], truth["text_polygon_px"])
            anchor_valid = (anchor_tolerance is None or
                            _anchor_error(prediction["anchors_px"], truth["anchors_px"])["max_endpoint_error_px"] <= anchor_tolerance)
            if overlap >= 0.25 and anchor_valid:
                candidates.append((overlap, index))
        edges.append([index for _, index in sorted(candidates, key=lambda item: (-item[0], item[1]))])
    matched_truth = {}
    for start in range(len(predictions)):
        queue, predecessor, visited_truth = deque([start]), {start: None}, set()
        endpoint = None
        while queue and endpoint is None:
            current = queue.popleft()
            for target in edges[current]:
                if target in visited_truth:
                    continue
                visited_truth.add(target)
                if target not in matched_truth:
                    endpoint = (current, target)
                    break
                displaced = matched_truth[target]
                if displaced not in predecessor:
                    predecessor[displaced] = (current, target)
                    queue.append(displaced)
        if endpoint is not None:
            current, target = endpoint
            while True:
                matched_truth[target] = current
                previous = predecessor[current]
                if previous is None:
                    break
                current, target = previous
    return sorted((prediction, truth) for truth, prediction in matched_truth.items())


def _anchor_error(predicted, expected):
    distances = []
    for ordered in (expected, expected[::-1]):
        errors = [math.dist(a, b) for a, b in zip(predicted, ordered)]
        distances.append((max(errors), sum(errors) / 2))
    maximum, mean = min(distances)
    return {"max_endpoint_error_px": maximum, "mean_endpoint_error_px": mean}


def evaluate_associations(prediction: dict, annotation: dict, *, anchor_tolerance_px=5) -> dict:
    """Evaluate supported proposals against explicit annotations of the same PNG.

    Proposal precision/recall require spatial text match, correct kind/nominal,
    and both anchors within tolerance (endpoint order is irrelevant). A wrong
    anchor is a false proposal and a missed expected link. Unsupported annotated
    forms are excluded from recall but a proposal on one remains a false positive.
    """
    tolerance = _number(anchor_tolerance_px, "anchor_tolerance_px", positive=True)
    if not isinstance(prediction, dict) or not isinstance(annotation, dict):
        raise ValueError("Prediction and annotation must be objects")
    if annotation.get("schema_version") != "1.0":
        raise ValueError("Unsupported annotation schema_version")
    for document in (prediction, annotation):
        digest = document.get("source_sha256")
        if not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest):
            raise ValueError("source_sha256 must identify the prepared image")
        size = document.get("processed_size_px")
        if (not isinstance(size, list) or len(size) != 2 or
                any(type(value) is not int or value <= 0 for value in size) or math.prod(size) > 25_000_000):
            raise ValueError("processed_size_px is invalid")
    if prediction["source_sha256"] != annotation["source_sha256"]:
        raise ValueError("Prediction and annotation source hashes differ")
    if prediction["processed_size_px"] != annotation["processed_size_px"]:
        raise ValueError("Prediction and annotation sizes differ")
    size = annotation["processed_size_px"]
    scope = annotation.get("scope")
    if scope not in {"full_page", "regions"}:
        raise ValueError("Annotation scope must be full_page or regions")
    region_values = annotation.get("regions_px", [])
    if not isinstance(region_values, list) or len(region_values) > MAX_ITEMS:
        raise ValueError("regions_px must be a bounded list")
    regions = [_points(value, size, "region", polygon=True) for value in region_values]
    if scope == "regions" and not regions:
        raise ValueError("Region scope requires explicit regions_px")
    ignored = _entries(annotation.get("ignored_regions", []), "ignored_regions", "id")
    excluded = _entries(annotation.get("excluded_notations", []), "excluded_notations", "id")
    negatives = _entries(annotation.get("negatives", []), "negatives", "id")
    for entry in [*ignored, *excluded, *negatives]:
        _points(entry.get("text_polygon_px"), size, "annotation polygon", polygon=True)
        if not isinstance(entry.get("reason"), str) or not entry["reason"].strip():
            raise ValueError("Excluded, negative and ignored annotations require a reason")
    def in_scope(polygon):
        return ((scope == "full_page" or any(_inside(polygon, region) for region in regions)) and
                not any(_inside(polygon, entry["text_polygon_px"]) for entry in ignored))

    expected = _entries(annotation.get("dimensions"), "dimensions", "id")
    all_annotation_ids = [item["id"] for item in [*expected, *ignored, *excluded, *negatives]]
    if len(all_annotation_ids) != len(set(all_annotation_ids)):
        raise ValueError("Annotation IDs must be unique across all categories")
    for truth in expected:
        if truth.get("kind") not in SUPPORTED_KINDS:
            raise ValueError("Unsupported ground truth must use excluded_notations")
        _number(truth.get("nominal_mm"), "nominal_mm", positive=True)
        _points(truth.get("text_polygon_px"), size, "text_polygon_px", polygon=True)
        _points(truth.get("anchors_px"), size, "anchors_px")
        if not in_scope(truth["text_polygon_px"]):
            raise ValueError("Expected dimension lies outside the evaluated scope")
    predictions = _entries(prediction.get("associations"), "associations", "notation_id")
    for item in predictions:
        if item.get("status") not in STATUSES:
            raise ValueError("Association status is invalid")
        if not isinstance(item.get("kind"), str) or not item["kind"]:
            raise ValueError("Association kind is required")
        if item.get("nominal_mm") is not None:
            _number(item["nominal_mm"], "nominal_mm", positive=True)
        if item["status"] == "proposed" and item.get("nominal_mm") is None:
            raise ValueError("Proposals require a nominal value")
        _points(item.get("text_polygon_px"), size, "text_polygon_px", polygon=True)
        _points(item.get("anchors_px"), size, "anchors_px", empty=item["status"] != "proposed")
    eligible = [item for item in predictions if in_scope(item["text_polygon_px"])]
    proposals = [item for item in eligible if item["status"] == "proposed"]
    abstentions = [item for item in eligible if item["status"] != "proposed"]
    correct_pairs = _match(proposals, expected, anchor_tolerance=tolerance)
    used_predictions = {p for p, _ in correct_pairs}
    used_expected = {t for _, t in correct_pairs}
    leftover_prediction_indices = [p for p in range(len(proposals)) if p not in used_predictions]
    leftover_expected_indices = [t for t in range(len(expected)) if t not in used_expected]
    residual_pairs = _match([proposals[p] for p in leftover_prediction_indices],
                            [expected[t] for t in leftover_expected_indices])
    pairs = sorted([*correct_pairs, *((leftover_prediction_indices[p], leftover_expected_indices[t]) for p, t in residual_pairs)])
    matches = []
    for p, t in pairs:
        error = _anchor_error(proposals[p]["anchors_px"], expected[t]["anchors_px"])
        matches.append({"notation_id": proposals[p]["notation_id"], "expected_id": expected[t]["id"],
                        "anchor_correct": error["max_endpoint_error_px"] <= tolerance, **error})
    correct = sum(match["anchor_correct"] for match in matches)
    matched_prediction_ids = {match["notation_id"] for match in matches}
    correct_truth = {match["expected_id"] for match in matches if match["anchor_correct"]}
    remaining_truth = [truth for truth in expected if truth["id"] not in correct_truth]
    abstained_expected = len(_match(abstentions, remaining_truth))
    errors = [match["max_endpoint_error_px"] for match in matches]
    return {
        "schema_version": "1.0", "source_sha256": prediction["source_sha256"],
        "processed_size_px": size, "scope": scope, "anchor_tolerance_px": tolerance,
        "matching": "one-to-one maximum anchor-correct matching, then residual text matching; polygon IoU >= 0.25; equal kind/nominal; unordered anchors",
        "counts": {"expected": len(expected), "proposed": len(proposals), "correct_proposals": correct,
                   "false_proposals": len(proposals) - correct, "missed_expected": len(expected) - correct,
                   "text_matches": len(matches), "anchor_failures": len(matches) - correct,
                   "abstentions": len(abstentions), "abstained_expected": abstained_expected,
                   "out_of_scope": len(predictions) - len(eligible), "excluded_notations": len(excluded),
                   "annotated_negatives": len(negatives)},
        "proposal_precision": correct / len(proposals) if proposals else None,
        "proposal_recall": correct / len(expected) if expected else None,
        "text_match_precision": len(matches) / len(proposals) if proposals else None,
        "text_match_recall": len(matches) / len(expected) if expected else None,
        "anchor_accuracy_on_text_matches": correct / len(matches) if matches else None,
        "max_anchor_error_px": max(errors) if errors else None,
        "mean_max_anchor_error_px": sum(errors) / len(errors) if errors else None,
        "abstention_statuses": dict(Counter(item["status"] for item in abstentions)),
        "matches": matches,
        "unmatched_proposal_ids": [item["notation_id"] for item in proposals if item["notation_id"] not in matched_prediction_ids],
        "missed_expected_ids": [item["id"] for item in expected if item["id"] not in correct_truth],
        "limitations": ["Measures only explicitly annotated support and scope; empty ground truth has no recall.",
                        "Does not assess physical scale, solver correctness, text recognition outside matched dimensions or CAD editability."],
    }
