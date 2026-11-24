"""Exact source-point lookup for the existing raster junction decisions."""
from collections import defaultdict

import numpy as np

from .arc_junction_merge_v1 import context as original_context
from .cad_object_assembly_v5 import POLICY


class SourceIndex:
    """Skip only paths that the original context would discard as non-contacting.

    Original item order, point walks, connector eligibility and geometric tests
    are retained. Non-integral input falls back to the original complete scan.
    """

    def __init__(self, items):
        self.items = items
        self.owners = defaultdict(set)
        self.connectors = defaultdict(list)
        self.integral = True
        for identity, item in enumerate(items):
            points = item['points']
            if not np.isfinite(points).all() or not np.equal(points, np.rint(points)).all():
                self.integral = False
            if self.integral:
                for point in points:
                    self.owners[tuple(map(int, point))].add(identity)
            if len(points) < 2:
                continue
            a, b = points[[0, -1]]
            if np.max(abs(b-a)) <= POLICY['fit_max_px']:
                self.connectors[tuple(a)].append(tuple(b))
                self.connectors[tuple(b)].append(tuple(a))
        if not self.integral:
            self.owners.clear()

    def near(self, p, q):
        if not self.integral or not np.equal([p, q], np.rint([p, q])).all():
            return list(range(len(self.items)))
        identities = set()
        for point in (p, q):
            x, y = map(int, point)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    identities.update(self.owners.get((x+dx, y+dy), ()))
        return sorted(identities)

    def context(self, i, j, p, q, ink):
        identities = self.near(p, q)
        positions = {identity: position for position, identity in enumerate(identities)}
        safe, fragments = original_context([self.items[k] for k in identities],
            positions.get(i, -1), positions.get(j, -1), p, q, ink)
        return safe, [identities[k] for k in fragments]
