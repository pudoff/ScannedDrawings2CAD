"""The sole private-helper boundary for the supported versioned chain kernel.

Pinned producers remain immutable. This adapter names their semantics and
translates their resource signal; it never patches globals or compiles source.
"""
from copy import deepcopy
from pathlib import Path
import numpy as np
from . import dimension_association_v3 as legacy
from . import dimension_support_v1 as support
from .dimension_support_v2 import propose as geometry_propose
from .contour_topology_v3 import ContourTopology
from .dimension_support_verified_v1 import _arrow_ink as mask_fraction
from .chain_resources_v1 import ResourceExhausted

FIELDS = ('anchors_px', 'measured_candidate_ids', 'dimension_line_candidate_ids',
          'extension_line_candidate_ids', 'arrow_candidate_ids', 'evidence')


class SearchCounter(dict):
    def __init__(self, ledger, stage):
        super().__init__(notation=0, page=0)
        self.ledger, self.stage = ledger, stage

    def __setitem__(self, key, value):
        if key == 'page':
            delta = value - self.get('page', 0)
            self.ledger.charge('search_checks', delta, self.stage)
        super().__setitem__(key, value)


class GeometryBackend:
    version = 'frozen-v3-support-v2/1.0'

    def __init__(self, source, geometry, ocr, arrows):
        self.source = Path(source)
        self.digest, self.size, lines, self.curves, heads = legacy._validate(self.source, geometry, ocr, arrows)
        self.lines = sorted(lines, key=lambda x: x['id'])
        self.heads = sorted(heads, key=lambda x: x['id'])
        self.texts = {l['id']: legacy._text_geometry(l) for l in ocr['lines']}
        self.ocr, self.geometry, self.arrows = ocr, geometry, arrows

    def topology(self):
        excluded = {gid for a in self.heads if a['confidence'] >= .65
                    for gid in a.get('evidence', {}).get('geometry_ids', [])}
        return ContourTopology(self.source, self.lines, excluded_ids=excluded,
                               text_polygons=[l['polygon_px'] for l in self.ocr['lines']])

    def orientations(self, text_id, counter):
        text = self.texts[text_id]
        try:
            axes, cut = support._axes(text, self.lines, counter) if text['orientation_uncertain'] else ([text['u']], False)
            return [support._oriented(text, a) for a in axes], cut
        except legacy._SearchLimit as exc:
            raise ResourceExhausted('orientation search exhausted') from exc

    def linear(self, oriented, peers, counter, topology):
        try:
            return legacy._linear(oriented, self.lines, self.heads, peers, counter, topology=topology)
        except legacy._SearchLimit as exc:
            raise ResourceExhausted('linear search exhausted') from exc

    def baseline(self, output):
        return legacy.associate_dimensions_v3(self.source, Path(output), geometry=self.geometry,
                                              ocr=self.ocr, arrows=self.arrows, record_timing=False)

    def geometry_proposals(self, baseline):
        return geometry_propose(self.source, geometry=self.geometry, ocr=self.ocr, arrows=self.arrows, baseline=baseline)


def parse_reading(raw):
    return support.parse_composite(raw)


def merge_choice(choices, candidate):
    support._add_equivalent(choices, deepcopy(candidate))


def demote(row, reason):
    support._demote(row, reason)


def apply_choices(row, choices, issues):
    legacy._apply_search_result(row, choices, False, {}, issues)


def resolve_pair_ownership(rows):
    legacy._resolve_text_ownership(rows)


def legacy_mask_coverage(session, arrow):
    try:
        return mask_fraction(session, arrow)
    except legacy._SearchLimit as exc:
        raise ResourceExhausted('historical arrow mask limit') from exc


def triangle_sample_points(polygon):
    from .triangle_sampling_v1 import points
    try:
        return points(polygon)
    except legacy._SearchLimit as exc:
        raise ResourceExhausted('triangle sampling limit') from exc
