"""Guarded connector profile, isolated from the immutable geometry producer.

The connector loop is a readable fork of the pinned legacy loop. Its gates,
order, provenance and caps are retained; empty/nonfinite observations are refused.
Loading the base into a private namespace avoids patching shared module globals.
"""
import hashlib
import importlib.util
import math
from pathlib import Path

import numpy as np

from .dataset_layout_v1 import encoded
from .family_catalog_v2 import require

BASE_PATH = Path(__file__).with_name('geometry_detection.py')
BASE_SHA256 = 'b39a3065669ddd38b655a5da12141aabdbf457ff53a3aaf9747b5fb9a9231d46'
PROFILE = 'guarded-short-connector-profile/1.0'


def finite_sample(points, foreground, stats):
    """Reject before integer conversion; out-of-image samples never wrap."""
    points = np.asarray(points, dtype=float)
    require(points.ndim >= 1 and points.shape[-1] == 2, 'Expected pixel point pairs')
    finite = np.isfinite(points).all(axis=-1)
    stats['connector_nonfinite_samples_rejected'] += int((~finite).sum())
    # This broad float bound preserves rounding at the image borders and
    # prevents overflow even for finite values far outside int64 range.
    bounded = finite & (points >= -1).all(axis=-1) & (points <= [foreground.shape[1], foreground.shape[0]]).all(axis=-1)
    pixels = np.zeros(points.shape, dtype=np.int64)
    pixels[bounded] = np.rint(points[bounded]).astype(np.int64)
    inside = bounded & (pixels >= 0).all(axis=-1) & (pixels < [foreground.shape[1], foreground.shape[0]]).all(axis=-1)
    values = np.zeros(inside.shape, dtype=bool)
    values[inside] = foreground[pixels[..., 1][inside], pixels[..., 0][inside]] > 0
    return values


def find_connectors(lines: list[dict], foreground: np.ndarray, stroke: float,
                      minimum_length: float, *, maximum_pairs=10000, maximum_connectors=250) -> tuple[list[dict], dict]:
    """Observe short caps missed by Hough between long parallel strokes.

    Only an existing endpoint can seed a cap. Its perpendicular projection must
    lie on another observed long segment, and the complete intervening raster
    must contain a thin, continuous stroke. This is raster evidence, not an
    inferred closure of an open contour or a semantic object-edge decision.
    """
    require(isinstance(foreground, np.ndarray) and foreground.ndim == 2
            and foreground.dtype in (np.dtype('uint8'), np.dtype('bool'))
            and 0 < foreground.size <= 25000000, 'Expected bounded foreground raster')
    require(np.isfinite(stroke) and 0 < stroke <= 24
            and np.isfinite(minimum_length) and minimum_length > 0, 'Invalid connector scale')
    require(type(maximum_pairs) is int and 0 <= maximum_pairs <= 10000
            and type(maximum_connectors) is int and 0 <= maximum_connectors <= 250
            and len(lines) <= 2500, 'Connector budget exceeds frozen limits')
    stats = {"short_connector_pairs_examined": 0, "short_connectors_added": 0,
             "short_connector_limit_reached": False,
             "connector_empty_profiles_rejected": 0, "connector_empty_rows_rejected": 0,
             "connector_invalid_supports_rejected": 0, "connector_nonfinite_trials_rejected": 0,
             "connector_nonfinite_samples_rejected": 0}
    minimum_support = max(minimum_length * 2, stroke * 12)
    supports = []
    valid_lines = []
    for index, line in enumerate(lines):
        try:
            points = np.asarray(line['points'], dtype=float)
        except (KeyError, TypeError, ValueError):
            stats['connector_invalid_supports_rejected'] += 1
            continue
        if (points.shape != (2, 2) or not np.isfinite(points).all()
                or np.max(np.abs(points)) > max(foreground.shape) * 4 + 16):
            stats['connector_invalid_supports_rejected'] += 1
            continue
        length = float(np.linalg.norm(points[1] - points[0]))
        if not np.isfinite(length) or length <= 1e-9:
            stats['connector_invalid_supports_rejected'] += 1
            continue
        valid_lines.append(line)
        if length >= minimum_support:
            supports.append((index, points))
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
        return finite_sample(points, foreground, stats)

    for source_position, (source_index, observed) in enumerate(supports):
        parallel = np.abs(directions @ directions[source_position]) >= math.cos(math.radians(2))
        for endpoint in observed:
            along = np.sum((endpoint - endpoints[:, 0]) * directions, axis=1)
            projections = endpoints[:, 0] + along[:, None] * directions
            gaps = np.linalg.norm(projections - endpoint, axis=1)
            nearby = (parallel & (indices != source_index) & (gaps >= minimum_gap)
                      & (gaps <= maximum_gap) & (along >= 0) & (along <= lengths))
            for target_position in np.flatnonzero(nearby):
                if stats["short_connector_pairs_examined"] >= maximum_pairs or len(added) >= maximum_connectors:
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
                    empty = ~profile.any(axis=1)
                    if empty.any():
                        stats['connector_empty_profiles_rejected'] += 1
                        stats['connector_empty_rows_rejected'] += int(empty.sum())
                        continue
                    # A cap must remain thin, with white space on both sides.
                    # Filled triangles/blobs and parallel-stroke bleed fail.
                    widths = profile.sum(axis=1) * .5
                    if (np.max(widths) > max(3., stroke * 1.75)
                            or np.ptp(widths) > max(1.5, stroke * .65)
                            or profile[:, :2].any() or profile[:, -2:].any()):
                        continue
                    centers = np.asarray([np.mean(transverse[row]) for row in profile])
                    if not np.isfinite(centers).all():
                        stats['connector_nonfinite_trials_rejected'] += 1
                        continue
                    if np.ptp(centers) > max(1.5, stroke * .65):
                        continue
                    center_offset = offset + float(np.median(centers))
                    trial = np.asarray([endpoint, end]) + center_offset * normal
                    if not np.isfinite(trial).all() or not np.isfinite(center_offset):
                        stats['connector_nonfinite_trials_rejected'] += 1
                        continue
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
                for old in valid_lines:
                    old_points = np.asarray(old['points'], dtype=float)
                    old_delta = old_points[1] - old_points[0]
                    old_length = np.linalg.norm(old_delta)
                    if not np.isfinite(old_length) or old_length <= 1e-9:
                        stats['connector_invalid_supports_rejected'] += 1
                        continue
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



def isolated_kernel(*, maximum_pairs=10000, maximum_connectors=250):
    require(hashlib.sha256(BASE_PATH.read_bytes()).hexdigest() == BASE_SHA256,
            'Frozen geometry producer changed')
    spec = importlib.util.spec_from_file_location('cad_recovery._guarded_geometry_kernel_v1', BASE_PATH)
    kernel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernel)
    def guarded(lines, foreground, stroke, minimum_length):
        return find_connectors(lines, foreground, stroke, minimum_length,
                               maximum_pairs=maximum_pairs, maximum_connectors=maximum_connectors)
    kernel._short_connectors = guarded
    return kernel


def detect_geometry(source_image, output_dir, *, maximum_pairs=10000, maximum_connectors=250, **kwargs):
    kernel = isolated_kernel(maximum_pairs=maximum_pairs, maximum_connectors=maximum_connectors)
    report = kernel.detect_geometry(source_image, output_dir, **kwargs)
    report['producer_profile'] = {'id': PROFILE, 'base_code_sha256': BASE_SHA256,
        'kernel_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'maximum_pairs': maximum_pairs, 'maximum_connectors': maximum_connectors,
        'empty_profiles_inferred': False, 'production_admitted': False,
        'physical_geometry_approved': False}
    # This report was created by this invocation in its fresh output folder.
    (Path(output_dir) / report['artifacts']['report']).write_bytes(encoded(report))
    return report
