"""Offline diagnostic registration and retention metrics; no cleaning or acceptance."""
from __future__ import annotations

import cv2
import numpy as np


def transform_points(points, matrix):
    points = np.asarray(points, dtype=float)
    matrix = np.asarray(matrix, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or matrix.shape != (3, 3):
        raise ValueError('Expected Nx2 points and 3x3 transform')
    if not np.isfinite(points).all() or not np.isfinite(matrix).all():
        raise ValueError('Nonfinite coordinates')
    result = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    if np.any(np.abs(result[:, 2]) < 1e-12):
        raise ValueError('Point at infinity')
    return result[:, :2] / result[:, 2:]


def register_images(source, target):
    """Similarity registration with deterministic fitting/held-out correspondences."""
    cv2.setRNGSeed(152)
    sift = cv2.SIFT_create(nfeatures=5000)
    k1, d1 = sift.detectAndCompute(source, None)
    k2, d2 = sift.detectAndCompute(target, None)
    if d1 is None or d2 is None:
        raise ValueError('No registration features')
    matches = [m for m, n in cv2.BFMatcher().knnMatch(d1, d2, k=2)
               if m.distance < .7 * n.distance]
    # Deduplicate locations: several SIFT orientations at one location are not
    # independent validation landmarks.
    pairs = {}
    for m in sorted(matches, key=lambda m: m.distance):
        p, q = k1[m.queryIdx].pt, k2[m.trainIdx].pt
        key = (*np.round(p, 1), *np.round(q, 1))
        pairs.setdefault(key, (p, q))
    pairs = sorted(pairs.values())
    if len(pairs) < 30:
        raise ValueError('Insufficient matched landmarks')
    p, q = np.asarray(pairs, dtype=float).transpose(1, 0, 2)
    heldout = np.arange(len(p)) % 3 == 0
    affine, inliers = cv2.estimateAffinePartial2D(
        p[~heldout], q[~heldout], method=cv2.RANSAC,
        ransacReprojThreshold=1.5, maxIters=2000)
    if affine is None:
        raise ValueError('Registration failed')
    matrix = np.vstack((affine, [0, 0, 1]))
    error = np.linalg.norm(transform_points(p, matrix) - q, axis=1)
    # Report all held-out errors as well as the consistent subset. Outliers are
    # descriptor mismatches, never silently omitted from the denominator.
    held = error[heldout]
    consistent = held <= 1.5
    if len(held) < 10 or consistent.mean() < .8 or np.median(held) > .6:
        raise ValueError('Held-out registration gate failed')
    return matrix, {
        'matched_locations': len(p), 'fit_locations': int((~heldout).sum()),
        'fit_inliers': int(inliers.sum()), 'heldout_locations': len(held),
        'heldout_within_1_5px': int(consistent.sum()),
        'heldout_median_px': float(np.median(held)),
        'heldout_p95_all_px': float(np.percentile(held, 95)),
        'heldout_max_all_px': float(held.max()),
        'heldout_p95_consistent_px': float(np.percentile(held[consistent], 95)),
        'matrix': matrix.tolist(),
        'landmarks': [{'source': a.tolist(), 'target': b.tolist(),
                       'heldout': bool(h), 'error_px': float(e)}
                      for a, b, h, e in zip(p, q, heldout, error)],
    }


def marked_regions(rgb):
    """Return enclosed interiors of Paint markers, not a semantic noise mask."""
    red = ((rgb[:, :, 0] > 170) & (rgb[:, :, 1] < 120) & (rgb[:, :, 2] < 140))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(red.astype('uint8'))
    groups = sorted([(i, s) for i, s in enumerate(stats) if i and s[4] > 15],
                    key=lambda pair: (pair[1][1], pair[1][0]))
    results = []
    for i, (label, (x, y, w, h, area)) in enumerate(groups, 1):
        component = (labels == label).astype('uint8')
        contours, hierarchy = cv2.findContours(component, cv2.RETR_CCOMP,
                                              cv2.CHAIN_APPROX_SIMPLE)
        interior = np.zeros(red.shape, dtype='uint8')
        holes = 0
        for j, contour in enumerate(contours):
            if hierarchy[0, j, 3] >= 0 and cv2.contourArea(contour) > 4:
                cv2.drawContours(interior, [contour], -1, 1, -1)
                holes += 1
        interior[red] = 0
        interior = cv2.erode(interior, np.ones((3, 3), dtype='uint8'))
        if not interior.any():
            raise ValueError('Marker has no usable enclosed interior')
        results.append({'id': f'marker-{i:02}', 'bbox': [int(x), int(y), int(x+w), int(y+h)],
                        'holes': holes, 'mask': interior})
    return red, results


def retention_metrics(baseline, candidate, noise, useful, unknown):
    """Masks annotate baseline pixels only; newly added pixels are unclassified."""
    arrays = [np.asarray(a) for a in (baseline, candidate, noise, useful, unknown)]
    if any(a.dtype != bool or a.shape != arrays[0].shape for a in arrays):
        raise ValueError('Expected equal boolean masks')
    baseline, candidate, noise, useful, unknown = arrays
    if np.any((noise & useful) | (noise & unknown) | (useful & unknown)):
        raise ValueError('Conflicting annotation masks')
    if np.any((noise | useful | unknown) & ~baseline):
        raise ValueError('Annotations must select baseline ink')
    return {
        'noise_baseline_px': int(noise.sum()),
        'noise_removed_px': int((noise & ~candidate).sum()),
        'useful_baseline_px': int(useful.sum()),
        'useful_removed_px': int((useful & ~candidate).sum()),
        'unknown_baseline_px': int(unknown.sum()),
        'unknown_removed_px': int((unknown & ~candidate).sum()),
        'added_unclassified_px': int((candidate & ~baseline).sum()),
        'unlabelled_removed_px': int((baseline & ~(noise | useful | unknown) & ~candidate).sum()),
    }
