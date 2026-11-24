"""Bounded planar line topology for reviewing dimension-anchor candidates.

No geometry is moved or joined across unobserved gaps. A finite face is only a
connectivity hypothesis, never a semantic part label or physical measurement.
Interior/interior crossings do not become contour junctions merely by proximity.
"""
from collections import defaultdict
import math

import cv2
import numpy as np
from PIL import Image

MAX_PAIR_CHECKS = 250_000
MAX_VERTICES = 10_000
MAX_FACES = 2_000
MAX_FACE_VERTICES = 2_048
MAX_BRIDGE_EXCLUSION_CHECKS = 100_000
JUNCTION_TOLERANCE_PX = 4.5


def cross(a, b):
    return float(a[0]*b[1]-a[1]*b[0])


def segment_distance(point, a, b):
    delta = b-a
    position = float(np.dot(point-a, delta)/np.dot(delta, delta))
    closest = a+np.clip(position, 0., 1.)*delta
    return float(np.linalg.norm(point-closest))


def proper_cross(a, b, c, d):
    return cross(b-a, c-a)*cross(b-a, d-a) < -1e-8 and cross(d-c, a-c)*cross(d-c, b-c) < -1e-8


class ContourTopology:
    """Retain open edges, duplicate provenance and all bounded face alternatives."""
    report_tolerance = JUNCTION_TOLERANCE_PX
    def __init__(self, source_image, lines, *, excluded_ids=(), text_polygons=()):
        self.complete = True
        self.checks = 0
        self.face_checks = 0
        self.bridge_checks = 0
        self.faces = []
        self.vertices = []
        self.edges = {}
        self.excluded = sorted(set(excluded_ids))
        self.rejected_faces = []
        self._ink = None
        self._source = source_image
        self._vertex_bins = defaultdict(list)
        self._exact_vertices = {}
        self._by_geometry = defaultdict(list)
        self._face_geometry = {}
        self.lines = []
        self._annotation_lines = []
        polygons = [np.asarray(p, dtype=np.float32) for p in text_polygons]
        self._text_polygons = polygons
        for line in lines:
            original = line['original']
            # Explicit annotation classes, not line width, colour or control IDs.
            declared = original.get('semantic_role', original.get('classification'))
            in_text = any(all(cv2.pointPolygonTest(p, tuple(map(float, point)), False) >= 0
                                 for point in line['points']) for p in polygons)
            if line['id'] in self.excluded or declared in {'text', 'hatch', 'dimension', 'dimension_line'} or in_text:
                if line['id'] not in self.excluded:
                    self.excluded.append(line['id'])
                self._annotation_lines.append(line)
                continue
            self.lines.append(line)
        self._build()

    def _observed_bridge(self, a, b):
        if np.linalg.norm(a-b) <= 1e-5:
            return True
        if self._ink is None:
            with Image.open(self._source) as image:
                self._ink = np.asarray(image.convert('L')) < 200
        samples = np.linspace(a, b, max(2, math.ceil(np.linalg.norm(a-b)*2)+1))
        height, width = self._ink.shape
        for point in samples:
            x, y = np.rint(point).astype(int)
            if not 0 <= x < width or not 0 <= y < height or not self._ink[y, x]:
                return False
            # An excluded dimension/hatch/text stroke may visually bridge a
            # genuine opening. Such ink cannot establish a part junction.
            for polygon in self._text_polygons:
                self.bridge_checks += 1
                if self.bridge_checks > MAX_BRIDGE_EXCLUSION_CHECKS:
                    self.complete = False
                    return False
                if cv2.pointPolygonTest(polygon, tuple(map(float, point)), False) >= 0:
                    return False
            for line in self._annotation_lines:
                self.bridge_checks += 1
                if self.bridge_checks > MAX_BRIDGE_EXCLUSION_CHECKS:
                    self.complete = False
                    return False
                if segment_distance(point, *line['points']) <= 1.0:
                    return False
        return True

    def _simple_face(self, points):
        if len(points) > MAX_FACE_VERTICES:
            self.complete = False
            return False
        for i in range(len(points)):
            for j in range(i+2, len(points)):
                if (j+1) % len(points) == i:
                    continue
                self.checks += 1
                self.face_checks += 1
                if self.checks > MAX_PAIR_CHECKS:
                    self.complete = False
                    return False
                if proper_cross(points[i], points[(i+1) % len(points)], points[j], points[(j+1) % len(points)]):
                    return False
        return True

    def _vertex(self, point):
        if not self.complete:
            return None
        tolerance = JUNCTION_TOLERANCE_PX
        exact = tuple(np.round(point, 7))
        if exact in self._exact_vertices:
            return self._exact_vertices[exact]
        key = tuple(np.floor(point/tolerance).astype(int))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for index in self._vertex_bins[(key[0]+dx, key[1]+dy)]:
                    previous = self.vertices[index]
                    if np.linalg.norm(previous-point) <= tolerance and self._observed_bridge(previous, point):
                        self._exact_vertices[exact] = index
                        return index
        if len(self.vertices) >= MAX_VERTICES:
            self.complete = False
            return None
        index = len(self.vertices)
        self.vertices.append(point.copy())
        self._vertex_bins[key].append(index)
        self._exact_vertices[exact] = index
        return index

    def _build(self):
        tolerance = JUNCTION_TOLERANCE_PX
        split = [[*line['points']] for line in self.lines]
        if not split:
            return
        boxes = np.asarray([[*line['points'].min(axis=0), *line['points'].max(axis=0)] for line in self.lines])
        for i, one in enumerate(self.lines):
            a, b = one['points']
            overlap = np.where((boxes[:, 0] <= boxes[i, 2]+tolerance) & (boxes[:, 2] >= boxes[i, 0]-tolerance)
                               & (boxes[:, 1] <= boxes[i, 3]+tolerance) & (boxes[:, 3] >= boxes[i, 1]-tolerance))[0]
            for j in overlap:
                if not self.complete:
                    return
                if j <= i:
                    continue
                self.checks += 1
                if self.checks > MAX_PAIR_CHECKS:
                    self.complete = False
                    return
                two = self.lines[j]
                c, d = two['points']
                determinant = cross(one['u'], two['u'])
                if abs(determinant) <= 1e-6:
                    # Overlapping observations retain one geometric segment with all IDs.
                    for point in (a, b, c, d):
                        if segment_distance(point, a, b) < 1e-5 and segment_distance(point, c, d) < 1e-5:
                            split[i].append(point)
                            split[j].append(point)
                    continue
                point = a+cross(c-a, two['u'])/determinant*one['u']
                da, db = segment_distance(point, a, b), segment_distance(point, c, d)
                endpoint = min(np.linalg.norm(point-p) for p in (a, b, c, d))
                if da > tolerance or db > tolerance or endpoint > tolerance:
                    continue
                # Short detector truncations need visible ink; a real opening stays open.
                if (da > 1e-5 and not self._observed_bridge(point, min((a, b), key=lambda p: np.linalg.norm(point-p)))) or (
                        db > 1e-5 and not self._observed_bridge(point, min((c, d), key=lambda p: np.linalg.norm(point-p)))):
                    continue
                split[i].append(point)
                split[j].append(point)
        adjacency = defaultdict(set)
        for line, points in zip(self.lines, split):
            ids = []
            for point in sorted(points, key=lambda p: float(np.dot(p-line['points'][0], line['u']))):
                vertex = self._vertex(point)
                if vertex is None:
                    return
                if not ids or ids[-1] != vertex:
                    ids.append(vertex)
            for a, b in zip(ids, ids[1:]):
                if a == b:
                    continue
                key = tuple(sorted((a, b)))
                self.edges.setdefault(key, set()).add(line['id'])
                adjacency[a].add(b)
                adjacency[b].add(a)
        ordered = {a: sorted(neighbours, key=lambda b: math.atan2(*(self.vertices[b]-self.vertices[a])[::-1]))
                   for a, neighbours in adjacency.items()}
        visited = set()
        for a, b in [(a, b) for a in sorted(adjacency) for b in sorted(adjacency[a])]:
            if (a, b) in visited:
                continue
            walk, current = [], (a, b)
            while current not in visited:
                visited.add(current)
                u, v = current
                walk.append(u)
                neighbours = ordered[v]
                current = (v, neighbours[(neighbours.index(u)-1) % len(neighbours)])
            if current != (a, b) or len(walk) < 3:
                continue
            # A dangling observed edge may point into a bounded face. The face
            # walk traverses that edge out and back; remove only this immediate
            # backtrack from the face boundary, retaining the open edge itself.
            changed = True
            while changed and len(walk) >= 3:
                changed = False
                for index in range(len(walk)):
                    if walk[index-1] == walk[(index+1) % len(walk)]:
                        rotated = walk[index-1:]+walk[:index-1] if index else walk[-1:]+walk[:-1]
                        walk = rotated[2:]
                        changed = True
                        break
            if len(walk) < 3:
                continue
            points = [self.vertices[v] for v in walk]
            area = sum(cross(p, q) for p, q in zip(points, points[1:]+points[:1]))/2
            if area <= 1e-6:
                continue
            if len(set(walk)) != len(walk) or not self._simple_face(points):
                if not self.complete:
                    return
                self.rejected_faces.append({'reason': 'self_intersection_or_repeated_vertex', 'vertex_ids': walk})
                continue
            face_id = f'face-{len(self.faces)+1:04d}'
            boundaries = []
            for u, v in zip(walk, walk[1:]+walk[:1]):
                edge = {'points_px': [self.vertices[u].tolist(), self.vertices[v].tolist()],
                        'geometry_ids': sorted(self.edges[tuple(sorted((u, v)))])}
                boundaries.append(edge)
                for identifier in edge['geometry_ids']:
                    self._by_geometry[identifier].append((face_id, self.vertices[u], self.vertices[v]))
            self.faces.append({'id': face_id, 'area_px2': area, 'polygon_px': [p.tolist() for p in points],
                               'boundaries': boundaries, 'semantic_role': 'unconfirmed_closed_line_face'})
            self._face_geometry[face_id] = {identifier for edge in boundaries for identifier in edge['geometry_ids']}
            if len(self.faces) >= MAX_FACES:
                self.complete = False
                return

    def faces_at(self, point, identifier, *, excluded_ids=()):
        return sorted({face for face, a, b in self._by_geometry[identifier]
                       if segment_distance(point, a, b) <= JUNCTION_TOLERANCE_PX
                       and not self._face_geometry[face].intersection(excluded_ids)})

    def report(self):
        used = {identifier for face in self.faces for edge in face['boundaries'] for identifier in edge['geometry_ids']}
        return {'schema_version': '1.0', 'complete': self.complete, 'junction_tolerance_px': JUNCTION_TOLERANCE_PX,
                'pair_checks': self.checks, 'max_pair_checks': MAX_PAIR_CHECKS,
                'face_intersection_checks': self.face_checks, 'max_face_vertices': MAX_FACE_VERTICES,
                'bridge_exclusion_checks': self.bridge_checks, 'max_bridge_exclusion_checks': MAX_BRIDGE_EXCLUSION_CHECKS,
                'vertices_px': [point.tolist() for point in self.vertices], 'faces': self.faces,
                'excluded_annotation_ids': sorted(self.excluded), 'rejected_faces': self.rejected_faces,
                'open_or_unconfirmed_geometry_ids': sorted(line['id'] for line in self.lines if line['id'] not in used),
                'duplicate_edge_observations': sum(len(ids)-1 for ids in self.edges.values()),
                'geometry_modified': False, 'holes_filled': False, 'physical_scale_confirmed': False,
                'limitation': 'A bounded face establishes connectivity only; semantic part roles and holes require review.'}
