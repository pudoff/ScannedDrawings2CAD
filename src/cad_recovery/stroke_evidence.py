"""Local raster measurements for comparing competing geometric hypotheses.

Stroke width is evidence about the image, not a line-role classifier.  A thin
stroke can be a real object edge.  Measurements deliberately skip the queried
intersection and report unsupported or unstable profiles as unknown.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

_OFFSETS = (4., 6., 8., 10., 14., 18., 24.)
_PROFILE_RADIUS = 16.
_PROFILE_STEP = .5
_MAX_WIDTH = 12.
_MAX_PIXELS = 25_000_000


def _point(value, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (2,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite pixel point")
    return result


class StrokeEvidence:
    """Measure transverse ink runs near, but never at, a line intersection.

    ``direction`` is the measured line direction, not the extension direction.
    Optional ``segment_endpoints`` bounds sampling to the observed segment;
    this allows one-sided support at an endpoint without treating a blank side
    as a valid stroke.  No drawing coordinates or nominal values are inferred.
    """

    def __init__(self, source_image: Path, expected_sha256: str):
        source_image = Path(source_image)
        if source_image.stat().st_size > 50 * 1024 * 1024:
            raise ValueError("Stroke raster exceeds 50 MiB")
        digest = hashlib.sha256(source_image.read_bytes()).hexdigest()
        if digest != expected_sha256:
            raise ValueError("Stroke source_sha256 does not match the prepared image")
        with Image.open(source_image) as image:
            if not 0 < image.width * image.height <= _MAX_PIXELS:
                raise ValueError("Stroke raster exceeds the pixel limit")
            gray = np.asarray(image.convert("L"))
        threshold, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        self._ink = binary.astype(np.float32) / 255.
        self.source_sha256 = digest
        self.processed_size_px = [int(gray.shape[1]), int(gray.shape[0])]
        self.threshold = float(threshold)

    def measure(self, point_px, direction, *, segment_endpoints=None) -> dict:
        point = _point(point_px, "point_px")
        direction = _point(direction, "direction")
        length = float(np.linalg.norm(direction))
        if not math.isfinite(length) or length < 1e-9:
            raise ValueError("direction must have finite positive length")
        unit = direction / length
        normal = np.array([-unit[1], unit[0]])
        bounds = None
        sample_origin = point.copy()
        if segment_endpoints is not None:
            endpoints = np.asarray(segment_endpoints, dtype=float)
            if endpoints.shape != (2, 2) or not np.isfinite(endpoints).all():
                raise ValueError("segment_endpoints must contain two finite pixel points")
            delta = endpoints[1] - endpoints[0]
            segment_length = float(np.linalg.norm(delta))
            if not math.isfinite(segment_length) or segment_length < 1e-9:
                raise ValueError("segment_endpoints must have finite positive length")
            if abs(float(np.dot(delta / segment_length, unit))) < math.cos(math.radians(3)):
                raise ValueError("segment_endpoints must follow the measured direction")
            if abs(float(np.dot(point - endpoints[0], normal))) > 3.:
                raise ValueError("point_px must lie near the measured segment line")
            projections = (endpoints - point) @ unit
            bounds = (float(min(projections)), float(max(projections)))
            # A geometric intersection may extrapolate slightly beyond the
            # detected segment. Start inside its observed endpoint so the
            # first samples do not land on that endpoint's junction.
            origin_shift = float(np.clip(0., bounds[0], bounds[1]))
            sample_origin = point + origin_shift * unit
            bounds = (bounds[0] - origin_shift, bounds[1] - origin_shift)

        result = {
            "method": "local_transverse_ink_width_v1",
            "source_sha256": self.source_sha256,
            "point_px": point.tolist(), "direction": unit.tolist(),
            "sample_origin_px": sample_origin.tolist(),
            "segment_endpoints": None if segment_endpoints is None else endpoints.tolist(),
            "status": "unknown", "width_px": None, "spread_px": None,
            "observed_range_px": None,
            "support_count": 0, "sample_count": 0, "sides_supported": [],
            "threshold": self.threshold, "samples": [], "reasons": [],
        }
        if np.any(point < 0) or np.any(point >= self.processed_size_px):
            result["reasons"] = ["query_outside_image"]
            return result

        eligible = {"negative": 0, "positive": 0}
        supported = {"negative": [], "positive": []}
        for sign, side in ((-1, "negative"), (1, "positive")):
            for offset in _OFFSETS:
                distance = sign * offset
                position = sample_origin + distance * unit
                sample = {"position_px": position.tolist(), "offset_px": distance,
                          "side": side, "status": "unknown", "width_px": None}
                if bounds is not None and not bounds[0] <= distance <= bounds[1]:
                    sample["reason"] = "outside_segment"
                else:
                    eligible[side] += 1
                    sample.update(self._profile(position, normal))
                    if sample["status"] == "supported":
                        supported[side].append(sample["width_px"])
                result["samples"].append(sample)

        widths = supported["negative"] + supported["positive"]
        result["sample_count"] = sum(eligible.values())
        result["support_count"] = len(widths)
        result["sides_supported"] = [side for side, values in supported.items() if len(values) >= 2]
        # A known endpoint may be supported on its interior side.  Missing ink
        # on an otherwise eligible side is a gap, not an implicit endpoint.
        if len(widths) < 3 or len(widths) < .6 * sum(eligible.values()):
            result["reasons"].append("insufficient_clean_profiles")
        if any(count >= 2 and len(supported[side]) < 2 for side, count in eligible.items()):
            result["reasons"].append("missing_side_support")
        if widths:
            median = float(np.median(widths))
            # A single blemish may enlarge one profile. Quantiles tolerate
            # that isolated sample, while sustained width changes remain
            # unknown. Keep the full range and all profiles for inspection.
            spread = float(np.percentile(widths, 90) - np.percentile(widths, 10))
            result["observed_range_px"] = float(max(widths) - min(widths))
            if spread > max(1.5, .35 * median):
                result["reasons"].append("unstable_width")
            if not result["reasons"]:
                result.update(status="supported", width_px=median, spread_px=spread)
        return result

    def _profile(self, position, normal):
        offsets = np.arange(-_PROFILE_RADIUS, _PROFILE_RADIUS + _PROFILE_STEP / 2, _PROFILE_STEP)
        points = position + offsets[:, None] * normal
        if np.any(points < 0) or np.any(points >= np.asarray(self.processed_size_px) - 1):
            return {"status": "unknown", "reason": "profile_outside_image"}
        ink = cv2.remap(self._ink, points[:, 0].astype(np.float32).reshape(1, -1),
                        points[:, 1].astype(np.float32).reshape(1, -1),
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)[0] >= .5
        indices = np.flatnonzero(ink)
        if not len(indices):
            return {"status": "unknown", "reason": "blank_profile"}
        nearest = int(indices[np.argmin(abs(offsets[indices]))])
        if abs(float(offsets[nearest])) > 2.5:
            return {"status": "unknown", "reason": "no_ink_near_center"}
        left = right = nearest
        while left > 0 and ink[left - 1]:
            left -= 1
        while right < len(ink) - 1 and ink[right + 1]:
            right += 1
        if left == 0 or right == len(ink) - 1:
            return {"status": "unknown", "reason": "unbounded_or_filled_profile"}
        width = float((right - left + 1) * _PROFILE_STEP)
        if width > _MAX_WIDTH:
            return {"status": "unknown", "reason": "junction_or_filled_profile", "observed_width_px": width}
        return {"status": "supported", "width_px": width,
                "center_offset_px": float((offsets[left] + offsets[right]) / 2)}
