"""Auditable local stroke comparisons for reviewing dimension alternatives.

These comparisons deliberately do not accept/reject geometric relationships:
thin object edges and thick unrelated strokes exist. The suggested ordering is
only a review aid, and every original alternative and status stays intact.
"""

from __future__ import annotations

import numpy as np

from .stroke_evidence import StrokeEvidence

MAX_STROKE_MEASUREMENTS = 1024
MAX_CONTINUATION_PX = 16.
MIN_WIDTH_RATIO = 1.6
MIN_WIDTH_DIFFERENCE_PX = 1.5


def _distance(point, line):
    distance = float(np.dot(point - line["points"][0], line["u"]))
    closest = line["points"][0] + np.clip(distance, 0, line["length"]) * line["u"]
    return float(np.linalg.norm(point - closest))


def _project(point, line):
    return line["points"][0] + np.dot(point - line["points"][0], line["u"]) * line["u"]


class DimensionStrokeReview:
    """Bounded, cached raster evidence; receives already validated observations."""

    def __init__(self, source_image, digest, lines, arrows):
        self.source_image, self.digest = source_image, digest
        self.lines = {line["id"]: line for line in lines}
        self.arrows = {arrow["id"]: arrow for arrow in arrows}
        self.measurements = []
        self._cache = {}
        self._raster = None
        self.limit_reached = False

    def _measure(self, point, direction, segment):
        if np.linalg.norm(segment[1] - segment[0]) < 1e-6:
            return None
        key = tuple(float(value) for value in np.concatenate((point, direction, segment.ravel())))
        if key in self._cache:
            return self._cache[key]
        if len(self.measurements) >= MAX_STROKE_MEASUREMENTS:
            self.limit_reached = True
            return None
        if self._raster is None:
            self._raster = StrokeEvidence(self.source_image, self.digest)
        result = self._raster.measure(point, direction, segment_endpoints=segment)
        result["id"] = f"stroke-{len(self.measurements) + 1:04d}"
        self.measurements.append(result)
        self._cache[key] = result
        return result

    def _anchor(self, point, choice, other_points):
        result = {"point_px": point.tolist(), "status": "unknown", "reasons": [],
                  "measured_profile_id": None, "witness_profile_id": None,
                  "continuation_profile_id": None, "widths_px": {}}
        measured = [self.lines[key] for key in choice["measured_candidate_ids"] if key in self.lines]
        extensions = [self.lines[key] for key in choice["extension_line_candidate_ids"] if key in self.lines]
        arrows = [self.arrows[key] for key in choice["arrow_candidate_ids"] if key in self.arrows]
        if not measured or not extensions or not arrows:
            result["reasons"] = ["local_geometry_missing"]
            return result
        # IDs are sorted in the report, whereas anchor order follows arrow order.
        # Establish each correspondence spatially; never zip the two lists.
        edge = min(measured, key=lambda line: (_distance(point, line), line["id"]))
        witness = min(extensions, key=lambda line: (_distance(point, line), line["id"]))
        arrow = min(arrows, key=lambda item: (np.linalg.norm(item["tip"] - _project(item["tip"], witness)),
                                              np.linalg.norm(item["tip"] - point), item["id"]))
        if (np.linalg.norm(point - _project(point, edge)) > 3 or
                np.linalg.norm(point - _project(point, witness)) > 3):
            result["reasons"] = ["intersection_outside_profile_tolerance"]
            return result
        tip = _project(arrow["tip"], witness)
        origin = _project(point, witness)
        distance = float(np.linalg.norm(origin - tip))
        if distance < 1e-6:
            result["reasons"] = ["zero_witness_span"]
            return result
        inward = (origin - tip) / distance
        result.update(measured_candidate_id=edge["id"], extension_candidate_id=witness["id"],
                      arrow_candidate_id=arrow["id"])
        profiles = {
            "measured": self._measure(point, edge["u"], edge["points"]),
            "witness": self._measure((origin + tip) / 2, inward, np.asarray([tip, origin])),
        }
        # Sample only observed continuation, away from the arrow. A neighboring
        # alternative can be another junction; stop halfway to it to avoid
        # borrowing its ink. Short/obscured windows remain unknown.
        length = min(MAX_CONTINUATION_PX, max(float(np.dot(end - origin, inward)) for end in witness["points"]))
        for other in other_points:
            along = float(np.dot(other - origin, inward))
            normal = float(np.linalg.norm(other - origin - along * inward))
            if along > 2 and normal <= 3:
                length = min(length, along / 2)
        profiles["continuation"] = self._measure(origin, inward, np.asarray([origin, origin + length * inward])) if length > 0 else None
        for name, profile in profiles.items():
            result[f"{name}_profile_id"] = profile["id"] if profile else None
            result["widths_px"][name] = profile["width_px"] if profile else None
        reference = profiles["witness"]
        if reference is None or reference["status"] != "supported":
            result["reasons"] = ["witness_width_unknown"]
            return result
        width = reference["width_px"]
        comparison = [profiles["measured"], profiles["continuation"]]
        strong = [name for name in ("measured", "continuation")
                  if profiles[name] and profiles[name]["status"] == "supported"
                  and profiles[name]["width_px"] >= MIN_WIDTH_RATIO * width
                  and profiles[name]["width_px"] >= width + MIN_WIDTH_DIFFERENCE_PX]
        if strong:
            result.update(status="boundary_supported", reasons=["local_width_contrast_with_witness"], contrast_sources=strong)
        elif all(profile and profile["status"] == "supported" and
                 profile["width_px"] <= 1.35 * width and profile["width_px"] <= width + 1.
                 for profile in comparison):
            result.update(status="similar_to_witness", reasons=["similar_local_width_does_not_establish_line_role"])
        else:
            result["reasons"] = ["incomplete_or_inconclusive_local_widths"]
        return result

    def annotate(self, association):
        """Append review evidence without modifying status, anchors or choices."""
        if association["kind"] != "linear" or association["status"] != "ambiguous":
            return
        choices = association["alternatives"]
        if len(choices) < 2 or "candidate_search_incomplete" in association["reasons"]:
            return
        all_points = [np.asarray(point, dtype=float) for choice in choices for point in choice["anchors_px"]]
        review = {"method": "local_stroke_comparison_v1", "status": "inconclusive",
                  "preferred_alternative_index": None, "alternatives": [],
                  "decision_effect": "review_only_all_geometric_alternatives_preserved"}
        for index, choice in enumerate(choices):
            anchors = [self._anchor(np.asarray(point, dtype=float), choice, all_points) for point in choice["anchors_px"]]
            states = [anchor["status"] for anchor in anchors]
            state = ("boundary_supported" if states == ["boundary_supported"] * 2 else
                     "similar_to_witness" if states == ["similar_to_witness"] * 2 else
                     "mixed" if "boundary_supported" in states else "unknown")
            review["alternatives"].append({"index": index, "status": state, "anchors": anchors})
        preferred = [item["index"] for item in review["alternatives"] if item["status"] == "boundary_supported"]
        if len(preferred) == 1 and all(
                item["index"] == preferred[0] or
                (all(anchor["status"] != "unknown" for anchor in item["anchors"]) and
                 any(anchor["status"] == "similar_to_witness" for anchor in item["anchors"]))
                for item in review["alternatives"]):
            review.update(status="preferred_for_review", preferred_alternative_index=preferred[0])
        review["measurement_limit_reached"] = self.limit_reached
        if self.limit_reached:
            review.update(status="inconclusive", preferred_alternative_index=None)
        association["evidence"]["stroke_review"] = review
