"""Conservative, auditable dimension relationships in prepared raster pixels.

The result is a proposal report, never a scale estimate or accepted CAD geometry.
Every supported relationship needs several independent spatial observations.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .dimension_roles import (DimensionStrokeReview, MAX_STROKE_MEASUREMENTS,
                              MAX_CONTINUATION_PX, MIN_WIDTH_RATIO, MIN_WIDTH_DIFFERENCE_PX)

MAX_PIXELS = 25_000_000
MAX_GEOMETRY = 2500
MAX_ARROWS = 2000
MAX_NOTATIONS = 10000
MAX_OCR_LINES = 10000
MAX_NEAR_LINES = 64
MAX_NEAR_ARROWS = 24
MAX_ALTERNATIVES = 16
MAX_CHECKS_PER_NOTATION = 100_000
MAX_CHECKS_PER_PAGE = 2_000_000
_COS = math.cos(math.radians(12))
_SIN = math.sin(math.radians(12))


class _SearchLimit(Exception):
    pass


def _budgeted(items, budget):
    for item in items:
        budget["notation"] += 1
        budget["page"] += 1
        if budget["notation"] > MAX_CHECKS_PER_NOTATION or budget["page"] > MAX_CHECKS_PER_PAGE:
            raise _SearchLimit
        yield item


def _point(value, label):
    result = np.asarray(value, dtype=float)
    if result.shape != (2,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must be a finite pixel point")
    return result


def _polygon(value, label, *, size=None):
    result = np.asarray(value, dtype=float)
    if result.ndim != 2 or result.shape[1:] != (2,) or not 3 <= len(result) <= 16 or not np.isfinite(result).all():
        raise ValueError(f"{label} must be a finite pixel polygon")
    if size is not None and ((result < 0).any() or (result > np.asarray(size)).any()):
        raise ValueError(f"{label} is outside the prepared image")
    with np.errstate(over="ignore", invalid="ignore"):
        contour = result.astype(np.float32)
    if not np.isfinite(contour).all():
        raise ValueError(f"{label} coordinates exceed supported pixel precision")
    area = cv2.contourArea(contour)
    # Match the evaluator's convex, positive-area contract. OpenCV's convexity
    # test alone accepts some self-crossing stars, so compare with the hull too.
    if (not cv2.isContourConvex(contour) or area <= .01 or
            not math.isclose(area, cv2.contourArea(cv2.convexHull(contour)), rel_tol=1e-6, abs_tol=1e-6)):
        raise ValueError(f"{label} must be a nondegenerate convex polygon without self-intersections")
    return result


def _cross(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def _line(candidate):
    points = np.asarray(candidate["points_px"], dtype=float)
    if points.shape != (2, 2) or not np.isfinite(points).all():
        raise ValueError("Geometry lines must have two finite pixel endpoints")
    length = float(np.linalg.norm(points[1] - points[0]))
    if length < 1e-6:
        raise ValueError("Geometry lines must have positive length")
    return {"id": candidate["id"], "points": points, "u": (points[1] - points[0]) / length,
            "length": length, "original": candidate}


def _perp(point, line):
    return abs(_cross(point - line["points"][0], line["u"]))


def _segment_distance(point, line):
    projection = float(np.dot(point - line["points"][0], line["u"]))
    closest = line["points"][0] + np.clip(projection, 0, line["length"]) * line["u"]
    return float(np.linalg.norm(point - closest))


def _intersection(a, b):
    determinant = _cross(a["u"], b["u"])
    if abs(determinant) < 1e-6:
        return None
    t = _cross(b["points"][0] - a["points"][0], b["u"]) / determinant
    return a["points"][0] + t * a["u"]


def _unique_ids(candidates, label):
    identifiers = [item.get("id") for item in candidates]
    if any(not isinstance(identifier, str) or not identifier for identifier in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{label} require nonempty unique IDs")


def _validate(source_image, geometry, ocr, arrows):
    if source_image.stat().st_size > 50 * 1024 * 1024:
        raise ValueError("Association raster exceeds 50 MiB")
    with Image.open(source_image) as image:
        size = list(image.size)
        if not 0 < size[0] * size[1] <= MAX_PIXELS:
            raise ValueError("Association raster exceeds the pixel limit")
    digest = hashlib.sha256(source_image.read_bytes()).hexdigest()
    for label, report in (("geometry", geometry), ("OCR", ocr), ("arrows", arrows)):
        if report.get("source_sha256") != digest or report.get("processed_size_px") != size:
            raise ValueError(f"{label} provenance does not match the prepared raster hash and dimensions")
    candidates, arrow_candidates, notations = geometry.get("candidates", []), arrows.get("candidates", []), ocr.get("notations", [])
    ocr_lines = ocr.get("lines", [])
    if (len(candidates) > MAX_GEOMETRY or len(arrow_candidates) > MAX_ARROWS or
            len(notations) > MAX_NOTATIONS or len(ocr_lines) > MAX_OCR_LINES):
        raise ValueError("Dimension association input candidate limit exceeded")
    for collection, label in ((candidates, "Geometry"), (arrow_candidates, "Arrows"), (notations, "Notations"), (ocr_lines, "OCR lines")):
        _unique_ids(collection, label)
    lines, curves = [], []
    for item in candidates:
        if item["kind"] == "line":
            lines.append(_line(item))
        elif item["kind"] in {"circle", "arc"}:
            center = _point(item["center_px"], "Curve center")
            radius = float(item["radius_px"])
            if not math.isfinite(radius) or radius <= 0:
                raise ValueError("Curve radius must be finite and positive")
            if item["kind"] == "arc" and (not math.isfinite(item["start_angle_deg"]) or not math.isfinite(item["sweep_angle_deg"]) or not 0 < item["sweep_angle_deg"] <= 360):
                raise ValueError("Arc range must be finite with a positive sweep at most 360 degrees")
            curves.append({**item, "center": center, "radius": radius})
        else:
            raise ValueError("Unsupported geometry candidate kind")
    normalized_arrows = []
    for item in arrow_candidates:
        tip, direction = _point(item["tip_px"], "Arrow tip"), _point(item["direction"], "Arrow direction")
        norm, confidence = float(np.linalg.norm(direction)), float(item["confidence"])
        if not .99 <= norm <= 1.01 or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Arrow direction must be unit length and confidence between zero and one")
        _polygon(item["polygon_px"], "Arrow polygon")
        normalized_arrows.append({**item, "tip": tip, "u": direction / norm})
    for item in ocr_lines:
        _polygon(item["polygon_px"], "OCR polygon", size=size)
        confidence = float(item["confidence"])
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("OCR confidence must be between zero and one")
    ocr_line_ids = {item["id"] for item in ocr_lines}
    for item in notations:
        if item.get("source_line_id") not in ocr_line_ids and "polygon_px" not in item:
            raise ValueError("Notation requires a source OCR line or a localized polygon")
        if "polygon_px" in item:
            _polygon(item["polygon_px"], "Notation polygon", size=size)
        if "confidence" in item:
            confidence = float(item["confidence"])
            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError("Notation confidence must be between zero and one")
    return digest, size, lines, curves, normalized_arrows


def _text_geometry(line):
    polygon = _polygon(line["polygon_px"], "OCR polygon")
    # Detection polygons can stay TL/TR/BR/BL after a recognizer rotates its
    # crop. The long edge, not necessarily the first edge, is the text axis.
    edges = np.roll(polygon, -1, axis=0) - polygon
    axis = edges[int(np.argmax(np.linalg.norm(edges, axis=1)))]
    width = float(np.linalg.norm(axis))
    if width < 1e-6:
        raise ValueError("OCR text baseline must have positive length")
    u = axis / width
    height = max(1., float(np.ptp(polygon @ np.array([-u[1], u[0]]))))
    return {"id": line["id"], "polygon": polygon, "center": polygon.mean(axis=0), "u": u, "width": width,
            "height": height, "orientation_uncertain": width / height < 1.2,
            "tol": max(3., min(8., height * .35))}


def _candidate(anchors, measured, dimension, extensions, arrows, evidence):
    return {"anchors_px": [np.asarray(point, dtype=float).tolist() for point in anchors],
            "measured_candidate_ids": sorted(set(measured)), "dimension_line_candidate_ids": sorted(set(dimension)),
            "extension_line_candidate_ids": sorted(set(extensions)), "arrow_candidate_ids": sorted(set(arrows)),
            "evidence": evidence}


def _same_candidate(a, b, tolerance=2.):
    if a["measured_candidate_ids"] != b["measured_candidate_ids"] or a["arrow_candidate_ids"] != b["arrow_candidate_ids"]:
        return False
    ap, bp = np.asarray(a["anchors_px"]), np.asarray(b["anchors_px"])
    return bool(min(np.linalg.norm(ap - bp, axis=1).max(), np.linalg.norm(ap - bp[::-1], axis=1).max()) <= tolerance)


def _add(result, item):
    if not any(_same_candidate(item, previous) for previous in result):
        result.append(item)


def _arrow_pairs(candidate):
    return candidate["evidence"].get("equivalent_arrow_pairs", [candidate["arrow_candidate_ids"]])


def _add_linear(result, item):
    # At a chain's shared witness, opposite arrowheads can corroborate the same
    # measured binding. Different arrow IDs alone do not make anchors ambiguous.
    for previous in result:
        if (previous["extension_line_candidate_ids"] == item["extension_line_candidate_ids"] and
                previous["dimension_line_candidate_ids"] == item["dimension_line_candidate_ids"] and
                _same_candidate(previous, {**item, "arrow_candidate_ids": previous["arrow_candidate_ids"]})):
            pairs = sorted({tuple(pair) for pair in [*_arrow_pairs(previous), *_arrow_pairs(item)]})
            previous["evidence"]["equivalent_arrow_pairs"] = [list(pair) for pair in pairs]
            previous["arrow_candidate_ids"] = list(pairs[0])
            return
    result.append(item)


def _supported_span(a, b, members, text):
    """Require observed line coverage; allow only the text's explicit interruption."""
    axis = b - a
    span = float(np.linalg.norm(axis))
    axis /= span
    intervals = []
    for line in members:
        projected = sorted(float(np.dot(point - a, axis)) for point in line["points"])
        intervals.append((max(0., projected[0]), min(span, projected[1])))
    projected_text = (text["polygon"] - a) @ axis
    text_interval = (max(0., float(projected_text.min()) - text["tol"]), min(span, float(projected_text.max()) + text["tol"]))
    if text_interval[0] < text_interval[1]:
        intervals.append(text_interval)
    cursor = 0.
    for left, right in sorted(intervals):
        if right < left:
            continue
        if left > cursor + text["tol"] * 2:
            return False
        cursor = max(cursor, right)
    return cursor >= span - text["tol"] * 2


def _anchor_options(tip, baseline, lines, exclude, text, budget):
    tolerance = text["tol"]
    options = []
    for extension in _budgeted(lines, budget):
        if extension["id"] in exclude or abs(float(np.dot(extension["u"], baseline["u"]))) > _SIN:
            continue
        if _segment_distance(tip, extension) > tolerance * 2:
            continue
        minimum_offset = max(12., text["height"])
        distances = [float(np.linalg.norm(point - tip)) for point in extension["points"]]
        if max(distances) < minimum_offset:
            continue
        endpoint_tolerance = max(6., text["height"] * .65)
        intersections = {-1: [], 1: []}
        for measured in _budgeted(lines, budget):
            if measured["id"] in exclude or measured["id"] == extension["id"]:
                continue
            point = _intersection(extension, measured)
            if point is None or _segment_distance(point, extension) > endpoint_tolerance or _segment_distance(point, measured) > tolerance:
                continue
            offset = float(np.dot(point - tip, extension["u"]))
            if abs(offset) < minimum_offset:
                continue
            intersections[1 if offset > 0 else -1].append((abs(offset), point, measured["id"]))
        # Hough can merge a witness line with the side of a part. Its far end
        # then belongs to the opposite face. The first supported intersection
        # in each observed direction is the anchor; never skip a nearer face.
        for candidates in intersections.values():
            if not candidates:
                continue
            nearest = min(item[0] for item in candidates)
            for distance, point, measured_id in candidates:
                if distance <= nearest + max(tolerance, endpoint_tolerance):
                    options.append((point, measured_id, extension["id"]))
                    if len(options) >= MAX_ALTERNATIVES + 1:
                        return options, True
    return options, False


def _separate_text_domain(other, first, axis, members, tolerance, domains):
    """A complete nearer internal chain can disprove external text ownership."""
    domain = domains.get(other["id"]) if domains else None
    if domain is None:
        return None
    member_ids = {line["id"] for line in members}
    if member_ids.intersection(domain["dimension_line_candidate_ids"]):
        return None
    domain_axis = np.asarray(domain["axis"])
    domain_origin = np.asarray(domain["origin"])
    if abs(float(np.dot(domain_axis, axis))) < _COS:
        return None
    margin = max(tolerance, other["tol"])
    if abs(_cross(domain_origin - first["tip"], axis)) <= margin * 2:
        return None
    own_distance = abs(_cross(other["center"] - domain_origin, domain_axis))
    competing_distance = abs(_cross(other["center"] - first["tip"], axis))
    if own_distance + margin >= competing_distance:
        return None
    return {"source_line_id": other["id"], "supporting_association_id": domain["association_id"],
            "dimension_line_candidate_ids": domain["dimension_line_candidate_ids"],
            "arrow_candidate_ids": domain["arrow_candidate_ids"],
            "competing_dimension_line_candidate_ids": sorted(member_ids),
            "own_text_to_line_px": own_distance, "competing_text_to_line_px": competing_distance,
            "required_distance_margin_px": margin,
            "basis": "first_pass_complete_internal_chain_on_distinct_nearer_line"}


def _text_placement(text, first, second, members, arrows, other_texts, budget,
                    *, text_domains=None, blocked_other_ids=None):
    """Local dimension-chain topology, without nominal-to-length comparisons."""
    delta = second["tip"] - first["tip"]
    span = float(np.linalg.norm(delta))
    if span < max(12., text["tol"] * 2):
        return None
    axis = delta / span
    position = float(np.dot(text["center"] - first["tip"], axis))
    separation = abs(_cross(text["center"] - first["tip"], axis))
    if separation > text["height"] * 2.5 + 6:
        return None
    external = position < 0 or position > span
    interval = (min(0., position), max(span, position))
    # Opposite arrowheads at a shared witness may differ by a few pixels.
    # They belong to the endpoint cluster, not to an intervening dimension.
    for arrow in _budgeted(arrows, budget):
        projection = float(np.dot(arrow["tip"] - first["tip"], axis))
        if min(abs(projection), abs(projection - span)) <= text["tol"] * 2:
            continue
        if interval[0] + text["tol"] < projection < interval[1] - text["tol"]:
            return None
    if external:
        # This baseline covers the explicit outside-arrow convention only.
        # Outside text with inside arrowheads needs additional evidence.
        if float(np.dot(first["u"], axis)) < _COS:
            return None
        gap = max(-position, position - span)
        if gap > max(text["width"] * 3, text["height"] * 6):
            return None
        projection = first["tip"] + position * axis
        nearest_tip = first["tip"] if position < 0 else second["tip"]
        if not any(_segment_distance(projection, member) <= text["tol"] for member in members):
            return None
        if not _supported_span(projection, nearest_tip, members, text):
            return None
    # Uncertain/unsupported OCR also blocks ownership: do not jump past a
    # nearer annotation merely because the parser could not read it.
    separate_domains = []
    for other in _budgeted(other_texts, budget):
        if other["id"] == text["id"]:
            continue
        projection = float(np.dot(other["center"] - first["tip"], axis))
        if not interval[0] - text["tol"] <= projection <= interval[1] + text["tol"]:
            # A second outside label can claim the opposite side of the same
            # shaft even when its reading is uncertain or unparsable. Verify
            # its potential placement without recursively checking text peers.
            if ((other["orientation_uncertain"] or abs(float(np.dot(other["u"], axis))) >= _COS) and
                    _text_placement(other, first, second, members, arrows, (), budget) is not None):
                domain = _separate_text_domain(other, first, axis, members, text["tol"], text_domains)
                if domain is not None:
                    separate_domains.append(domain)
                    continue
                if blocked_other_ids is not None:
                    blocked_other_ids.add(other["id"])
                return None
            continue
        if abs(_cross(other["center"] - first["tip"], axis)) <= max(text["height"], other["height"]) * 2.5 + 6:
            return None
    if not _supported_span(first["tip"], second["tip"], members, text):
        return None
    result = {"arrow_span_px": span, "measurement_axis_px": axis.tolist(),
              "text_to_dimension_line_px": separation,
              "text_placement": "external" if external else "between_arrows",
              "arrow_placement": "outside" if float(np.dot(first["u"], axis)) > 0 else "inside"}
    if separate_domains:
        result["separate_text_domains"] = separate_domains
    return result


def _linear(text, lines, arrows, other_texts, budget, *, text_domains=None):
    # A neighboring dimension can cross the same witness lines closer to the
    # current baseline than the part. Known arrow shafts are annotations and
    # cannot supply measured object edges, even outside the text neighborhood.
    shaft_ids = {identifier for arrow in arrows if arrow["confidence"] >= .65
                 for identifier in arrow.get("evidence", {}).get("geometry_ids", [])}
    near_lines = [line for line in _budgeted(lines, budget) if abs(float(np.dot(line["u"], text["u"]))) >= _COS
                  and _segment_distance(text["center"], line) <= max(text["width"] * .6, text["height"] * 2.5 + 6)]
    truncated = len(near_lines) > MAX_NEAR_LINES
    near_lines = sorted(near_lines, key=lambda line: _segment_distance(text["center"], line))[:MAX_NEAR_LINES]
    result = []
    blocked_other_ids = set()
    diagnostics = {"text_parallel_lines": len(near_lines), "opposed_arrow_pairs": 0, "extension_anchor_pairs": 0}
    for baseline in near_lines:
        members = [line for line in _budgeted(lines, budget) if abs(float(np.dot(line["u"], baseline["u"]))) >= _COS
                   and max(_perp(point, baseline) for point in line["points"]) <= text["tol"]]
        truncated |= len(members) > MAX_NEAR_LINES
        members = sorted(members, key=lambda line: _segment_distance(text["center"], line))[:MAX_NEAR_LINES]
        if not members:
            continue
        relevant = [arrow for arrow in _budgeted(arrows, budget) if arrow["confidence"] >= .65
                    and abs(float(np.dot(arrow["u"], baseline["u"]))) >= _COS
                    and _perp(arrow["tip"], baseline) <= text["tol"]
                    and any(_segment_distance(arrow["tip"], member) <= text["tol"] * 3 for member in members)]
        truncated |= len(relevant) > MAX_NEAR_ARROWS
        relevant = sorted(relevant, key=lambda arrow: float(np.linalg.norm(arrow["tip"] - text["center"])))[:MAX_NEAR_ARROWS]
        for first, second in _budgeted(itertools.combinations(relevant, 2), budget):
            key = tuple(sorted((first["id"], second["id"])))
            if float(np.dot(first["u"], second["u"])) > -_COS:
                continue
            placement = _text_placement(text, first, second, members, relevant, other_texts, budget,
                                        text_domains=text_domains, blocked_other_ids=blocked_other_ids)
            if placement is None:
                continue
            axis = np.asarray(placement["measurement_axis_px"])
            diagnostics["opposed_arrow_pairs"] += 1
            excluded = {line["id"] for line in members} | shaft_ids
            first_options, cut_first = _anchor_options(first["tip"], baseline, lines, excluded, text, budget)
            second_options, cut_second = _anchor_options(second["tip"], baseline, lines, excluded, text, budget)
            truncated |= cut_first or cut_second
            for one, two in _budgeted(itertools.product(first_options, second_options), budget):
                point_a, measured_a, extension_a = one
                point_b, measured_b, extension_b = two
                if extension_a == extension_b or measured_a == extension_b or measured_b == extension_a:
                    continue
                offset_a, offset_b = _cross(point_a - first["tip"], axis), _cross(point_b - second["tip"], axis)
                # A projected linear dimension may join different steps of a
                # part. Witness lengths need not be equal; both anchors must
                # still lie on the same side of the dimension line.
                if offset_a * offset_b <= 0:
                    continue
                diagnostics["extension_anchor_pairs"] += 1
                _add_linear(result, _candidate([point_a, point_b], [measured_a, measured_b], excluded.intersection(line["id"] for line in members),
                                       [extension_a, extension_b], key,
                                       {"relation": "opposed_arrows_parallel_text_perpendicular_extensions_object_intersections",
                                        **placement,
                                        "anchor_extension_offsets_px": [offset_a, offset_b], "pixel_tolerance": text["tol"]}))
                if len(result) > MAX_ALTERNATIVES:
                    return result[:MAX_ALTERNATIVES], True, diagnostics
    if blocked_other_ids:
        diagnostics["competing_external_text_ids"] = sorted(blocked_other_ids)
    return result, truncated, diagnostics


def _text_domains(associations, arrows, incomplete_ids):
    """Freeze reliable first-pass line ownership without choosing object anchors.

    Multiple anchor alternatives can corroborate one baseline. Conflicting reading,
    different arrow pairs, partial searches and shared text ownership cannot.
    """
    arrow_by_id = {arrow["id"]: arrow for arrow in arrows}
    source_counts = {}
    for item in associations:
        source_counts[item["source_line_id"]] = source_counts.get(item["source_line_id"], 0) + 1
    result = {}
    for item in associations:
        if (item["kind"] != "linear" or item["notation_id"] in incomplete_ids or
                source_counts[item["source_line_id"]] != 1 or
                item["status"] not in {"proposed", "ambiguous"} or
                item["reasons"] not in (["complete_spatial_evidence_requires_human_or_solver_verification"],
                                        ["multiple_spatially_supported_associations"])):
            continue
        choices = [item] if item["status"] == "proposed" else item["alternatives"]
        if not choices or any(choice["evidence"].get("text_placement") != "between_arrows" for choice in choices):
            continue
        line_ids = choices[0]["dimension_line_candidate_ids"]
        arrow_ids = choices[0]["arrow_candidate_ids"]
        if (not line_ids or len(arrow_ids) != 2 or
                any(choice["dimension_line_candidate_ids"] != line_ids or
                    choice["arrow_candidate_ids"] != arrow_ids for choice in choices)):
            continue
        first, second = [arrow_by_id[identifier] for identifier in arrow_ids]
        delta = second["tip"] - first["tip"]
        span = float(np.linalg.norm(delta))
        if span < 1e-6:
            continue
        result[item["source_line_id"]] = {
            "association_id": item["id"], "dimension_line_candidate_ids": line_ids,
            "arrow_candidate_ids": arrow_ids, "origin": first["tip"].tolist(), "axis": (delta / span).tolist()}
    return result


def _apply_search_result(association, alternatives, truncated, diagnostics, issues):
    association["evidence"] = diagnostics
    if truncated:
        issues.append({"code": "association_candidate_limit", "notation_id": association["notation_id"],
                       "message": "Candidate search or alternatives were truncated; no unique association can be proposed."})
    if len(alternatives) == 1 and not truncated:
        association.update(alternatives[0], status="proposed")
        association["reasons"] = ["complete_spatial_evidence_requires_human_or_solver_verification"]
    elif alternatives or truncated:
        association.update(status="ambiguous", alternatives=alternatives)
        association["reasons"] = ["multiple_spatially_supported_associations" if len(alternatives) > 1 else "candidate_search_incomplete"]
    else:
        association["reasons"] = ["required_arrow_line_extension_and_object_evidence_missing" if association["kind"] == "linear"
                                  else "required_radial_leader_arrow_and_curve_evidence_missing"]


def _resolve_text_ownership(associations):
    """One arrow pair cannot uniquely belong to two distinct text observations."""
    claims = {}
    for item in associations:
        if item["kind"] != "linear":
            continue
        choices = [item] if item["status"] == "proposed" else item["alternatives"]
        for choice in choices:
            for pair in _arrow_pairs(choice):
                key = tuple(pair)
                if len(key) == 2:
                    claims.setdefault(key, set()).add(item["notation_id"])
    fields = ("anchors_px", "measured_candidate_ids", "dimension_line_candidate_ids",
              "extension_line_candidate_ids", "arrow_candidate_ids")
    for item in associations:
        if item["kind"] != "linear" or item["status"] != "proposed":
            continue
        owners = set().union(*(claims.get(tuple(pair), set()) for pair in _arrow_pairs(item)))
        if len(owners) > 1:
            item["alternatives"] = [{**{key: item[key] for key in fields}, "evidence": item["evidence"]}]
            for key in fields:
                item[key] = []
            item.update(status="ambiguous", reasons=["competing_texts_for_arrow_pair"],
                        evidence={"competing_notation_ids": sorted(owners)})


def _on_curve(point, curve, tolerance):
    if abs(float(np.linalg.norm(point - curve["center"])) - curve["radius"]) > tolerance:
        return False
    if curve["kind"] == "arc":
        angle = math.degrees(math.atan2(*(point - curve["center"])[::-1])) % 360
        relative = (angle - curve["start_angle_deg"]) % 360
        angular_tolerance = math.degrees(tolerance / curve["radius"])
        return relative <= curve["sweep_angle_deg"] + angular_tolerance or relative >= 360 - angular_tolerance
    return True


def _radial(text, lines, curves, arrows, kind, budget):
    result = []
    diagnostics = {"explicit_radial_notation": True, "curve_arrow_contacts": 0, "radial_leaders": 0}
    near_lines = [line for line in _budgeted(lines, budget) if _segment_distance(text["center"], line) <= text["height"] * 2 + text["width"] * .5]
    truncated = len(near_lines) > MAX_NEAR_LINES
    near_lines = sorted(near_lines, key=lambda line: _segment_distance(text["center"], line))[:MAX_NEAR_LINES]
    for curve in curves:
        if kind == "diameter" and curve["kind"] != "circle":
            continue
        for arrow in _budgeted(arrows, budget):
            if arrow["confidence"] < .65 or not _on_curve(arrow["tip"], curve, text["tol"] * 2):
                continue
            radial = arrow["tip"] - curve["center"]
            radial_length = float(np.linalg.norm(radial))
            if radial_length < max(1e-6, curve["radius"] * .5):
                continue
            radial /= radial_length
            if abs(float(np.dot(radial, arrow["u"]))) < _COS:
                continue
            diagnostics["curve_arrow_contacts"] += 1
            for leader in _budgeted(near_lines, budget):
                # geometry_ids are the detected arrow shaft, corroborating a
                # leader rather than excluding it as a triangle side.
                if abs(float(np.dot(radial, leader["u"]))) < _COS or _segment_distance(arrow["tip"], leader) > text["tol"] * 3:
                    continue
                if _perp(curve["center"], leader) > text["tol"] * 2 or leader["length"] < max(12., text["height"]):
                    continue
                if _perp(text["center"], leader) > text["height"] * 2 + 6:
                    continue
                diagnostics["radial_leaders"] += 1
                contact = curve["center"] + radial * curve["radius"]
                opposite = curve["center"] if kind == "radius" else curve["center"] - radial * curve["radius"]
                candidate = _candidate([opposite, contact], [curve["id"]], [leader["id"]], [], [arrow["id"]],
                                       {"relation": "explicit_symbol_radial_leader_arrow_on_curve", "tip_to_curve_error_px": abs(float(np.linalg.norm(arrow["tip"] - curve["center"])) - curve["radius"]),
                                        "leader_to_center_error_px": _perp(curve["center"], leader), "pixel_tolerance": text["tol"],
                                        "opposite_diameter_anchor": "derived from observed circle" if kind == "diameter" else None})
                # Two opposite arrows on the same diameter corroborate one
                # relationship. They are not competing circle associations.
                merged = False
                if kind == "diameter":
                    for previous in result:
                        equivalent = {**candidate, "arrow_candidate_ids": previous["arrow_candidate_ids"]}
                        if (_same_candidate(previous, equivalent) and
                                set(previous["dimension_line_candidate_ids"]) & set(candidate["dimension_line_candidate_ids"])):
                            previous["arrow_candidate_ids"] = sorted(set(previous["arrow_candidate_ids"] + candidate["arrow_candidate_ids"]))
                            previous["evidence"]["corroborated_by_opposite_arrow"] = True
                            merged = True
                            break
                if not merged:
                    _add(result, candidate)
                if len(result) > MAX_ALTERNATIVES:
                    return result[:MAX_ALTERNATIVES], True, diagnostics
    return result, truncated, diagnostics


def associate_dimensions(source_image: Path, output_dir: Path, *, geometry: dict, ocr: dict,
                         arrows: dict, prefix="page-001") -> dict:
    """Propose relationships from observations of the same prepared raster.

    Confidence is a gate, not a probability. Nominals are copied from parsed OCR,
    never compared to pixel lengths to choose a candidate or infer model scale.
    """
    started = time.perf_counter()
    source_image, output_dir = Path(source_image), Path(output_dir)
    if not isinstance(prefix, str) or not prefix or Path(prefix).name != prefix or prefix in {".", ".."}:
        raise ValueError("Association prefix must be a filename component")
    names = {"report": f"{prefix}-dimension-associations.json", "preview": f"{prefix}-dimension-associations-preview.png"}
    for name in names.values():
        if (output_dir / name).exists() or (output_dir / name).is_symlink():
            raise FileExistsError(f"Dimension association never overwrites existing artifacts: {output_dir / name}")
    digest, size, lines, curves, arrow_candidates = _validate(source_image, geometry, ocr, arrows)
    ocr_lines = {item["id"]: item for item in ocr.get("lines", [])}
    text_geometries = {identifier: _text_geometry(item) for identifier, item in ocr_lines.items()}
    associations, issues = [], []
    budget = {"notation": 0, "page": 0}
    notation_checks = {}
    for notation in ocr.get("notations", []):
        parsed = notation.get("parsed", {})
        source_line = ocr_lines.get(notation.get("source_line_id"))
        kind = {"nominal": "linear"}.get(parsed.get("kind"), parsed.get("kind")) or "unknown"
        nominal = parsed.get("values", {}).get("nominal_mm")
        if nominal is not None and (isinstance(nominal, bool) or not isinstance(nominal, (int, float)) or not math.isfinite(nominal) or nominal <= 0):
            raise ValueError("Notation nominal must be finite and positive")
        association = {"id": f"association-{len(associations) + 1:04d}", "notation_id": notation["id"],
                       "source_line_id": notation.get("source_line_id"), "raw": parsed.get("raw", source_line.get("raw", "") if source_line else ""),
                       "kind": kind, "nominal_mm": nominal, "status": "unresolved", "text_polygon_px": source_line["polygon_px"] if source_line else notation.get("polygon_px", []),
                       "source_ref": notation.get("source_ref", source_line.get("source_ref") if source_line else None),
                       "anchors_px": [], "measured_candidate_ids": [], "dimension_line_candidate_ids": [],
                       "extension_line_candidate_ids": [], "arrow_candidate_ids": [], "evidence": {}, "reasons": [], "alternatives": []}
        associations.append(association)
        if notation.get("context") == "title_block_candidate":
            association.update(status="unsupported", reasons=["title_block_context_is_not_dimension_evidence"])
            continue
        if source_line is None:
            association["reasons"].append("source_ocr_line_missing")
            continue
        if (parsed.get("status") != "parsed" or notation.get("status") != "proposed" or
                source_line["confidence"] < .65 or float(notation.get("confidence", source_line["confidence"])) < .65 or source_line.get("conflicting_line_ids")):
            association.update(status="ambiguous", reasons=["ocr_or_notation_reading_requires_review"])
            continue
        if kind not in {"linear", "radius", "diameter"}:
            association.update(status="unsupported", reasons=["notation_kind_requires_a_different_relationship_model"])
            continue
        if nominal is None:
            association.update(status="ambiguous", reasons=["parsed_nominal_missing"])
            continue
        text = text_geometries[source_line["id"]]
        if text["orientation_uncertain"] and kind == "linear":
            association.update(status="ambiguous", reasons=["text_orientation_uncertain"])
            continue
        budget["notation"] = 0
        try:
            if kind == "linear":
                alternatives, truncated, diagnostics = _linear(text, lines, arrow_candidates, text_geometries.values(), budget)
            else:
                alternatives, truncated, diagnostics = _radial(text, lines, curves, arrow_candidates, kind, budget)
        except _SearchLimit:
            alternatives, truncated, diagnostics = [], True, {"bounded_search_exhausted": True}
        notation_checks[notation["id"]] = budget["notation"]
        _apply_search_result(association, alternatives, truncated, diagnostics, issues)
    _resolve_text_ownership(associations)
    incomplete_ids = {issue["notation_id"] for issue in issues if issue["code"] == "association_candidate_limit"}
    domains = _text_domains(associations, arrow_candidates, incomplete_ids)
    # One bounded retry uses only frozen first-pass ownership. It cannot grow
    # into a self-reinforcing chain of inferred domains or reset search limits.
    for association in associations:
        if (association["kind"] != "linear" or association["status"] != "unresolved" or
                not set(association["evidence"].get("competing_external_text_ids", [])).intersection(domains)):
            continue
        budget["notation"] = notation_checks[association["notation_id"]]
        text = text_geometries[association["source_line_id"]]
        try:
            alternatives, truncated, diagnostics = _linear(text, lines, arrow_candidates,
                                                          text_geometries.values(), budget, text_domains=domains)
        except _SearchLimit:
            alternatives, truncated, diagnostics = [], True, {"bounded_search_exhausted": True}
        notation_checks[association["notation_id"]] = budget["notation"]
        diagnostics["text_domain_pass"] = 2
        _apply_search_result(association, alternatives, truncated, diagnostics, issues)
    _resolve_text_ownership(associations)
    stroke_review = DimensionStrokeReview(source_image, digest, lines, arrow_candidates)
    incomplete_ids = {issue["notation_id"] for issue in issues if issue["code"] == "association_candidate_limit"}
    for association in associations:
        if association["notation_id"] not in incomplete_ids:
            stroke_review.annotate(association)
    if stroke_review.limit_reached:
        issues.append({"code": "stroke_measurement_limit", "message": "Local stroke review was limited; unknown measurements do not resolve alternatives."})
    issues.append({"code": "dimension_associations_unverified", "message": "Pixel relationships are proposals. They do not establish scale, dimensions, units or CAD geometry."})
    summary = {status: sum(item["status"] == status for item in associations) for status in ("proposed", "ambiguous", "unresolved", "unsupported")}
    summary["total"] = len(associations)
    output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(source_image) as image:
        preview = image.convert("RGB")
    preview.thumbnail((1600, 1600))
    sx, sy = preview.width / size[0], preview.height / size[1]
    draw = ImageDraw.Draw(preview)
    for item in associations:
        color = {"proposed": "#00735c", "ambiguous": "#a85b00", "unresolved": "#c72535", "unsupported": "#777777"}[item["status"]]
        polygon = [(point[0] * sx, point[1] * sy) for point in item["text_polygon_px"]]
        if polygon:
            draw.line([*polygon, polygon[0]], fill=color, width=2)
        anchors = [(point[0] * sx, point[1] * sy) for point in item["anchors_px"]]
        if len(anchors) == 2:
            draw.line(anchors, fill=color, width=2)
            for x, y in anchors:
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline=color, width=2)
    with (output_dir / names["preview"]).open("xb") as stream:
        preview.save(stream, format="PNG")
    report = {"schema_version": "1.0", "status": "needs_review", "source": source_image.name, "source_sha256": digest,
              "processed_size_px": size, "coordinate_system": "pixel centers; top-left origin; x right, y down",
              "associations": associations, "summary": summary, "issues": issues, "artifacts": names,
              "stroke_measurements": stroke_review.measurements,
              "artifact_sha256": {"preview": hashlib.sha256((output_dir / names["preview"]).read_bytes()).hexdigest()},
              "parameters": {"minimum_ocr_confidence": .65, "minimum_arrow_confidence": .65, "angular_tolerance_deg": 12,
                             "max_near_lines": MAX_NEAR_LINES, "max_near_arrows": MAX_NEAR_ARROWS, "max_alternatives": MAX_ALTERNATIVES,
                             "max_checks_per_notation": MAX_CHECKS_PER_NOTATION, "max_checks_per_page": MAX_CHECKS_PER_PAGE,
                             "max_text_domain_passes": 2, "max_stroke_measurements": MAX_STROKE_MEASUREMENTS,
                             "max_stroke_continuation_px": MAX_CONTINUATION_PX,
                             "stroke_minimum_width_ratio": MIN_WIDTH_RATIO,
                             "stroke_minimum_width_difference_px": MIN_WIDTH_DIFFERENCE_PX,
                             "stroke_comparison_decision_effect": "review_only"},
              "search_checks": budget["page"],
              "search_checks_by_notation": notation_checks,
              "elapsed_seconds": round(time.perf_counter() - started, 4)}
    with (output_dir / names["report"]).open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return report
