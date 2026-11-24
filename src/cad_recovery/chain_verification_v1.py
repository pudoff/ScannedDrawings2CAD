"""Public source-bound verification with explicit roles and shared budgets."""
from dataclasses import dataclass, asdict
from copy import deepcopy
from pathlib import Path
from typing import Protocol
import math
import cv2
import numpy as np
from PIL import Image
from .chain_resources_v1 import ResourceLedger, ResourceExhausted
from .chain_legacy_bridge_v1 import legacy_mask_coverage, triangle_sample_points, GeometryBackend


@dataclass(frozen=True)
class VerificationPolicy:
    role: str
    head_method: str = 'sampled'
    head_fraction: float = .85
    face_fraction: float = .85
    carrier_fraction: float = .85
    require_bounded_face: bool = True

    def __post_init__(self):
        if self.head_method not in ('mask', 'sampled') or not self.role:
            raise ValueError('Invalid verification policy')
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in
               (self.head_fraction, self.face_fraction, self.carrier_fraction)):
            raise ValueError('Ink thresholds must be finite fractions')


class ChainVerifier(Protocol):
    def verify(self, choice: dict, supported_arrows: set, topology: dict,
               policy: VerificationPolicy) -> dict: ...

    def report(self) -> dict: ...


class VerificationSession:
    """One raster/caches/counter per run, shared by all verification roles."""
    def __init__(self, source: Path, geometry: dict, arrows: dict, ledger: ResourceLedger):
        self.source, self.ledger = Path(source), ledger
        common = {k: geometry.get(k) for k in ('source_sha256', 'processed_size_px')}
        GeometryBackend(source, geometry, {**common, 'lines': [], 'notations': []}, arrows)
        with Image.open(source) as image:
            if image.width * image.height > 12_000_000:
                raise ValueError('Verification raster exceeds 12 Mpx')
            gray = np.array(image.convert('L'))
        _, white = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.distance = cv2.distanceTransform(white, cv2.DIST_L2, 5)
        self.geometry = {g['id']: deepcopy(g) for g in geometry['candidates']}
        self.arrows = {a['id']: deepcopy(a) for a in arrows['candidates']}
        self._samples = 0
        self._stage = 'verification'
        self._segments, self._heads = {}, {}
        self.cache_hits = 0

    @property
    def samples(self):
        return self._samples

    @samples.setter
    def samples(self, value):
        if type(value) is not int or value < self._samples:
            raise ValueError('Pixel counter cannot be reset')
        self.ledger.charge('pixel_samples', value - self._samples, self._stage)
        self._samples = value

    def _fraction(self, points, tolerance):
        self.samples += len(points)
        q = np.rint(points).astype(int)
        h, w = self.distance.shape
        inside = (q >= 0).all(axis=1) & (q[:, 0] < w) & (q[:, 1] < h)
        values = np.zeros(len(q), bool)
        values[inside] = self.distance[q[inside, 1], q[inside, 0]] <= tolerance
        return float(values.mean())

    def segment(self, points):
        key = tuple(tuple(p) for p in points)
        if key in self._segments:
            self.cache_hits += 1
            return self._segments[key]
        a, b = np.array(points, float)
        count = max(2, math.ceil(float(np.linalg.norm(b - a))) + 1)
        if count > 4096:
            raise ResourceExhausted('segment sample limit')
        self._segments[key] = self._fraction(np.linspace(a, b, count), 3.)
        return self._segments[key]

    def head(self, identifier, method):
        key = (identifier, method)
        if key in self._heads:
            self.cache_hits += 1
            return self._heads[key]
        arrow = self.arrows[identifier]
        if method == 'mask':
            value = legacy_mask_coverage(self, arrow)
        else:
            points = triangle_sample_points(arrow['polygon_px'])
            value = self._fraction(points, .5)
        self._heads[key] = value
        return value

    def verify(self, choice, supported_arrows, topology, policy):
        self._stage = policy.role
        if not set(choice['arrow_candidate_ids']).issubset(supported_arrows):
            return {'valid': False, 'reason': 'arrow_owned_shaft_unverified'}
        evidence = choice['evidence']
        if policy.require_bounded_face and (evidence.get('contour_support') != 'shared_bounded_face' or not evidence.get('contour_face_ids')):
            return {'valid': False, 'reason': 'shared_bounded_face_missing'}
        faces = {f['id']: f for f in topology['faces']}
        fractions = {'face': [], 'carriers': {}, 'heads': {}}
        for fid in evidence['contour_face_ids']:
            fractions['face'].extend(self.segment(e['points_px']) for e in faces[fid]['boundaries'])
        for gid in sorted(set(choice['dimension_line_candidate_ids'] + choice['extension_line_candidate_ids'])):
            fractions['carriers'][gid] = self.segment(self.geometry[gid]['points_px'])
        for aid in choice['arrow_candidate_ids']:
            fractions['heads'][aid] = self.head(aid, policy.head_method)
        valid = (min(fractions['face'], default=0 if policy.require_bounded_face else 1) >= policy.face_fraction
                 and min(fractions['carriers'].values(), default=0) >= policy.carrier_fraction
                 and min(fractions['heads'].values(), default=0) >= policy.head_fraction)
        return {'valid': bool(valid), 'reason': 'observed_chain_pixels' if valid else 'chain_pixels_missing',
                'fractions': fractions}

    def report(self):
        return {'version': 'chain-verification/1.0', 'samples': self.samples, 'cache_hits': self.cache_hits,
                'segment_cache_entries': len(self._segments), 'head_cache_entries': len(self._heads)}
