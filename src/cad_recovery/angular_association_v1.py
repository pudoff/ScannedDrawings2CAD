"""Versioned pixel angle proposals and guards for a missing degree sign.

This opt-in route wraps the frozen linear/radial v3 implementation. Neither a
pixel proposal nor its graph contract constitutes a physical/CAD approval.
"""
from copy import deepcopy
import hashlib
import itertools
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import dimension_association_v3 as linear
from .family_catalog_v2 import require
from .notation import parse_notation

POLICY = {
    'id': 'angular-association/1.0', 'max_near_arcs': 16, 'max_rays_per_endpoint': 16,
    'max_arrows_per_endpoint': 8, 'max_alternatives': 16,
    'max_checks_per_notation': 10000, 'max_checks_per_page': 100000,
    'max_pixel_cells': 2000000, 'contact_tolerance_px': 6,
    'minimum_ink_fraction': .75, 'minimum_visible_arc_samples': 12,
    'max_text_occlusion_fraction': .5,
    'tangent_tolerance_deg': 15, 'minimum_confidence': .65,
    'angle_comparison_tolerance_deg': 3,
    'coordinate_system': 'image x right y down; positive arc sweep clockwise',
    'nominal_used_for_support_selection': False, 'physical_geometry_approved': False,
    'production_admitted': False, 'training_eligible': False,
}


class SearchLimit(Exception):
    pass


class Budget:
    def __init__(self):
        self.page = self.notation = self.cells = 0

    def check(self, count=1):
        self.page += count; self.notation += count
        if self.page > POLICY['max_checks_per_page'] or self.notation > POLICY['max_checks_per_notation']:
            raise SearchLimit('candidate_check_limit')


class RasterEvidence:
    def __init__(self, source, ocr, arrows, budget):
        with Image.open(source) as image:
            self.gray = np.asarray(image.convert('L'))
        self.excluded = np.zeros(self.gray.shape, np.uint8)
        self.text_excluded = np.zeros(self.gray.shape, np.uint8)
        height, width = self.gray.shape
        for row in ocr.get('lines', []) + arrows.get('candidates', []):
            polygon = np.asarray(row['polygon_px'], float)
            polygon = np.clip(polygon, [0, 0], [width-1, height-1]).round().astype(np.int32)
            cv2.fillPoly(self.excluded, [polygon], 1)
        for row in ocr.get('lines', []):
            polygon = np.asarray(row['polygon_px'], float).round().astype(np.int32)
            cv2.fillPoly(self.text_excluded, [polygon], 1)
        self.budget = budget

    def sample(self, points):
        visible = supported = outside = excluded = text_excluded = 0; height, width = self.gray.shape
        for point in points:
            self.budget.check()
            x, y = np.rint(point).astype(int)
            if not (1 <= x < width-1 and 1 <= y < height-1):
                outside += 1
                continue
            if self.excluded[y, x]:
                excluded += 1
                text_excluded += bool(self.text_excluded[y, x])
                continue
            self.budget.cells += 9
            if self.budget.cells > POLICY['max_pixel_cells']:
                raise SearchLimit('pixel_probe_limit')
            visible += 1
            supported += bool((self.gray[y-1:y+2, x-1:x+2] < 200).any())
        return {'visible_samples': visible, 'ink_samples': supported,
                'assigned_samples': len(points), 'out_of_bounds_samples': outside, 'excluded_samples': excluded,
                'text_excluded_samples': text_excluded,
                'ink_fraction': supported/visible if visible else None}


def direction(angle):
    a = math.radians(angle)
    return np.array([math.cos(a), math.sin(a)])


def _arc_context(curve, text):
    delta = text['center']-curve['center']; radius = float(np.linalg.norm(delta))
    if radius < 1 or abs(radius-curve['radius']) > max(2*text['height'], .35*curve['radius']):
        return False
    angle = math.degrees(math.atan2(delta[1], delta[0]))
    offset = (angle-curve['start_angle_deg']) % 360
    return 0 < offset < curve['sweep_angle_deg']


def _rays(lines, center, contact, unit, budget):
    result = []; tolerance = POLICY['contact_tolerance_px']
    for row in lines:
        budget.check()
        if linear._segment_distance(center, row) > tolerance or linear._perp(contact, row) > tolerance:
            continue
        projections = (row['points']-center)@unit
        if float(projections.max()) < float(np.linalg.norm(contact-center))-tolerance:
            continue
        if abs(float(np.dot(row['u'], unit))) < math.cos(math.radians(4)):
            continue
        point = row['points'][int(np.argmax(projections))]
        result.append({'id': row['id'], 'point': point.tolist(),
                       'vertex_distance_px': linear._segment_distance(center, row), 'line': row})
    if len(result) > POLICY['max_rays_per_endpoint']:
        raise SearchLimit('ray_support_limit')
    return result


def _arrows(arrows, contact, unit, budget):
    result = []; tangent = np.array([-unit[1], unit[0]])
    for arrow in arrows:
        budget.check()
        if arrow['confidence'] < POLICY['minimum_confidence']:
            continue
        if (float(np.linalg.norm(arrow['tip']-contact)) <= POLICY['contact_tolerance_px'] and
                abs(float(np.dot(arrow['u'], tangent))) >= math.cos(math.radians(POLICY['tangent_tolerance_deg']))):
            result.append(arrow['id'])
    if len(result) > POLICY['max_arrows_per_endpoint']:
        raise SearchLimit('arrow_support_limit')
    return result


def hypotheses(text, lines, curves, arrows, raster, budget):
    """No nominal value, parsed kind or target labels enter geometric search."""
    results = []; near = []
    for curve in curves:
        budget.check()
        if curve['kind'] == 'arc' and 3 < curve['sweep_angle_deg'] < 357 and _arc_context(curve, text):
            near.append(curve)
    if len(near) > POLICY['max_near_arcs']:
        raise SearchLimit('arc_context_limit')
    for curve in near:
        center = curve['center']; radius = curve['radius']; start = curve['start_angle_deg'] % 360
        end = start+curve['sweep_angle_deg']; units = [direction(start), direction(end)]
        contacts = [center+radius*u for u in units]
        arc_points = [center+radius*direction(a) for a in np.linspace(start, end, min(256, max(32, math.ceil(radius*math.radians(end-start)))))]
        arc_ink = raster.sample(arc_points)
        if (arc_ink['visible_samples'] < POLICY['minimum_visible_arc_samples'] or
                arc_ink['ink_fraction'] < POLICY['minimum_ink_fraction']):
            continue
        complete_arc = (arc_ink['out_of_bounds_samples'] == 0 and
                        arc_ink['text_excluded_samples'] <= POLICY['max_text_occlusion_fraction']*arc_ink['assigned_samples'])
        rays = [_rays(lines, center, point, unit, budget) for point, unit in zip(contacts, units)]
        arrow_ends = [_arrows(arrows, point, unit, budget) for point, unit in zip(contacts, units)]
        if not (all(rays) or (any(rays) and any(arrow_ends))):
            continue
        for first, second in itertools.product(rays[0] or [None], rays[1] or [None]):
            budget.check()
            if first and second and first['id'] == second['id']:
                continue
            vertex = center
            complete_rays = first is not None and second is not None
            if complete_rays:
                vertex = linear._intersection(first['line'], second['line'])
                if vertex is None or float(np.linalg.norm(vertex-center)) > POLICY['contact_tolerance_px']:
                    continue
            ray_ink = []
            for ray, contact in zip((first, second), contacts):
                if ray:
                    proof = raster.sample(np.linspace(vertex, contact, 48))
                    ray_ink.append(proof)
            if not ray_ink or any(p['visible_samples'] < 8 or p['ink_fraction'] < POLICY['minimum_ink_fraction'] for p in ray_ink):
                continue
            for arrow_first, arrow_second in itertools.product(arrow_ends[0] or [None], arrow_ends[1] or [None]):
                budget.check()
                if arrow_first is not None and arrow_first == arrow_second:
                    continue
                complete = (complete_rays and complete_arc and arrow_first is not None and arrow_second is not None and
                            all(p['out_of_bounds_samples'] == 0 and p['text_excluded_samples'] <= POLICY['max_text_occlusion_fraction']*p['assigned_samples'] for p in ray_ink))
                points = [vertex.tolist(), first['point'] if first else contacts[0].tolist(), second['point'] if second else contacts[1].tolist()]
                observed = (math.degrees(math.atan2(points[2][1]-vertex[1], points[2][0]-vertex[0])) -
                            math.degrees(math.atan2(points[1][1]-vertex[1], points[1][0]-vertex[0]))) % 360
                results.append({'vertex_px': vertex.tolist(), 'points_px': points,
                                'measured_candidate_ids': [r['id'] for r in (first, second) if r],
                                'dimension_arc_candidate_id': curve['id'],
                                'arrow_candidate_ids': [a for a in (arrow_first, arrow_second) if a],
                                'complete_support': complete, 'observed_sector_deg': observed,
                                'sector_start_deg_image': start, 'sector_sweep_deg_image': curve['sweep_angle_deg'],
                                'graph_point_order_after_y_reflection': [0, 2, 1],
                                'evidence': {'arc_ink': arc_ink, 'ray_ink': ray_ink,
                                             'text_inside_observed_sector': True, 'tangent_arrow_contacts': len([a for a in (arrow_first, arrow_second) if a]),
                                             'nominal_used_for_selection': False, 'physical_geometry_approved': False}})
                if len(results) > POLICY['max_alternatives']:
                    raise SearchLimit('alternative_limit')
    return results


def angular_contract(item, *, purpose='review'):
    """Describe correspondence with DrawingGraph 0.2 without producing model-mm points."""
    require(purpose == 'review', 'Pixel angle proposals cannot be exported or used as physical targets')
    require(item['kind'] == 'angular' and item['nominal_unit'] == 'degree' and item['nominal_mm'] is None,
            'Not an angular degree proposal')
    return {'notation_kind': 'angle', 'notation_value_field': 'angle_deg', 'drawing_graph_kind': 'angular',
            'unit': 'degree', 'graph_points': 'vertex, ray1, ray2 in model CCW order',
            'pixel_to_model_transform_verified': False, 'exportable': False,
            'requires': ['verified finite straight rays and vertex', 'verified pixel-to-view transform with handedness',
                         'verified graph geometry and reviewed nominal/sector', 'explicit graph acceptance']}


def associate_angular_dimensions(source_image, output_dir, *, geometry, ocr, arrows, prefix='page-001'):
    source_image, output_dir = Path(source_image), Path(output_dir)
    require(not output_dir.exists() and not output_dir.is_symlink(), 'Angular output must be new')
    require(isinstance(prefix, str) and prefix and Path(prefix).name == prefix and prefix not in ('.', '..'), 'Invalid filename prefix')
    digest, size, lines, curves, arrow_candidates = linear._validate(source_image, geometry, ocr, arrows)
    for line in lines:
        require(np.abs(line['points']).max() <= 1e7, 'Angular geometry outside supported numeric bounds')
    for curve in curves:
        require(np.abs(curve['center']).max() <= 1e7 and curve['radius'] <= 1e7, 'Angular curve outside supported bounds')
    for arrow in arrow_candidates:
        require(np.abs(arrow['tip']).max() <= 1e7, 'Angular arrow outside supported numeric bounds')
    output_dir.mkdir(parents=True)
    base = linear.associate_dimensions_v3(source_image, output_dir/'linear-v3', geometry=geometry, ocr=ocr, arrows=arrows,
                                         prefix=prefix, record_timing=False)
    budget = Budget(); raster = RasterEvidence(source_image, ocr, arrows, budget)
    ocr_lines = {r['id']: r for r in ocr.get('lines', [])}; notations = {r['id']: r for r in ocr.get('notations', [])}
    report = deepcopy(base); report['algorithm_version'] = 'angular-association-v1'
    report['angular_policy'] = POLICY; report['physical_geometry_approved'] = False; report['production_admitted'] = False
    report['linear_baseline_artifacts'] = {k: 'linear-v3/'+v for k, v in base['artifacts'].items()}; incomplete = []; owners = {}
    for item in report['associations']:
        notation = notations[item['notation_id']]; row = ocr_lines.get(item['source_line_id'])
        if row is None or notation.get('context') == 'title_block_candidate':
            continue
        parsed = notation.get('parsed', {}); kind = parsed.get('kind')
        # Unsupported OCR containing a degree sign is still an angular/type-review context.
        explicit = kind == 'angle' or '°' in row['raw']
        if not explicit and kind not in ('nominal', None):
            continue
        budget.notation = 0; choices = []; limited = None
        try:
            choices = hypotheses(linear._text_geometry(row), lines, curves, arrow_candidates, raster, budget)
        except SearchLimit as error:
            limited = str(error); incomplete.append({'notation_id': item['notation_id'], 'reason': limited})
        if not explicit and not choices and limited is None:
            continue
        original = deepcopy(item)
        item.update(kind='angular' if explicit else 'unknown', nominal_mm=None,
                    nominal_unit='degree' if explicit else 'unverified', angle_deg=None,
                    status='ambiguous', anchors_px=[], measured_candidate_ids=[], dimension_line_candidate_ids=[],
                    extension_line_candidate_ids=[], arrow_candidate_ids=[], alternatives=choices, evidence={},
                    reasons=[], numeric_text_value=parsed.get('values', {}).get('nominal_mm'))
        item['type_hypotheses'] = ['angular'] if explicit else ['linear', 'angular_degree_sign_missing']
        item['prior_linear_hypothesis'] = original if not explicit else None
        if limited:
            item['reasons'].append('angular_search_incomplete:'+limited)
        if not explicit:
            item['reasons'].append('linear_reading_conflicts_with_observed_angular_context')
        else:
            syntax = parse_notation(row['raw'])
            value = syntax.get('values', {}).get('angle_deg')
            if syntax['kind'] == 'angle' and syntax['status'] == 'parsed' and value is not None and value < 360:
                item['angle_deg'] = value
                if parsed.get('kind') != 'angle' or parsed.get('values', {}).get('angle_deg') != value:
                    item['reasons'].append('notation_and_raw_angle_disagree')
            else:
                item['reasons'].append('angle_text_requires_review')
            item['graph_contract'] = angular_contract(item)
        if (row['confidence'] < POLICY['minimum_confidence'] or notation.get('confidence', row['confidence']) < POLICY['minimum_confidence']
                or row.get('conflicting_line_ids') or notation.get('status') != 'proposed'):
            item['reasons'].append('ocr_or_notation_reading_requires_review')
        if not choices:
            item['reasons'].append('angular_support_missing_or_incomplete')
        complete = [choice for choice in choices if choice['complete_support']]
        for choice in choices:
            owners.setdefault(choice['dimension_arc_candidate_id'], set()).add(item['notation_id'])
        if explicit and len(choices) == len(complete) == 1 and not item['reasons']:
            choice = complete[0]
            if abs(choice['observed_sector_deg']-item['angle_deg']) > POLICY['angle_comparison_tolerance_deg']:
                item['reasons'].append('observed_sector_disagrees_with_text')
            else:
                item.update(status='proposed', anchors_px=choice['points_px'], measured_candidate_ids=choice['measured_candidate_ids'],
                            dimension_line_candidate_ids=[choice['dimension_arc_candidate_id']], arrow_candidate_ids=choice['arrow_candidate_ids'],
                            evidence=choice['evidence'], observed_sector_deg=choice['observed_sector_deg'])
        elif choices:
            item['reasons'].append('angular_support_incomplete_or_not_unique')
    for item in report['associations']:
        choices = item.get('alternatives', [])
        contested = sorted({identifier for choice in choices if 'dimension_arc_candidate_id' in choice
                            for identifier in owners.get(choice['dimension_arc_candidate_id'], set()) if identifier != item['notation_id']})
        if contested:
            item.update(status='ambiguous', anchors_px=[], measured_candidate_ids=[], dimension_line_candidate_ids=[], arrow_candidate_ids=[])
            item['reasons'].append('competing_texts_for_dimension_arc'); item['competing_notation_ids'] = contested
    report['summary'] = {s: sum(r['status'] == s for r in report['associations']) for s in ('proposed', 'ambiguous', 'unresolved', 'unsupported')}
    report['summary']['total'] = len(report['associations'])
    report['angular_search'] = {'checks': budget.page, 'pixel_cells': budget.cells, 'incomplete': incomplete}
    report['issues'].append({'code': 'angular_pixel_proposals_only', 'message': 'Degree values and pixel supports are review proposals; no physical targets or graph acceptance.'})
    report['artifacts'] = {'report': prefix+'-angular-associations.json', 'preview': prefix+'-angular-associations.png'}
    with Image.open(source_image) as image:
        image = image.convert('RGB'); image.thumbnail((1600, 1600))
    draw = ImageDraw.Draw(image); scale = np.array(image.size)/size
    for item in report['associations']:
        color = '#008060' if item['status'] == 'proposed' else '#bd6320'
        polygon = (np.asarray(item['text_polygon_px'])*scale).tolist()
        if polygon: draw.line([tuple(p) for p in polygon+[polygon[0]]], fill=color, width=2)
        points = item['anchors_px']
        if item['kind'] == 'angular' and len(points) == 3:
            for p in points[1:]: draw.line([tuple(np.asarray(points[0])*scale), tuple(np.asarray(p)*scale)], fill=color, width=2)
    image.save(output_dir/report['artifacts']['preview'])
    report['artifact_sha256'] = {'preview': hashlib.sha256((output_dir/report['artifacts']['preview']).read_bytes()).hexdigest()}
    require(hashlib.sha256(source_image.read_bytes()).hexdigest() == digest, 'Source changed during association')
    (output_dir/report['artifacts']['report']).write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return report
