"""Sealed, source-pixel review of the Rib's two projected dimensions.

This is a deliberately small reviewed observation package. It never promotes
an interval feasibility witness to measured geometry or a released label.
"""
from __future__ import annotations

from copy import deepcopy
from html import escape
import json
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from .clear_pairs import contained, digest
from .graph_v2 import DrawingGraphV2
from .interval_scale_v1 import make_source_frame_binding_v1, solve_interval_scale_v1
from .label_release_v1 import checked_pin
from .label_schema_v1 import Pin, read_json_v1
from .raster_regions import read_prepared_region
from .rib_ring_readout_v1 import read_ring_center_v1


VIEW_BOX = [2400, 1160, 4140, 3420]
ANCHORS = ['rib_lower_vertical', 'rib_lower_flat', 'rib_upper_flat']
REVIEW_DIR = 'artifacts/rib-independent-review-v1/'
KEY_PATHS = {
    'source': 'dataset/clear_sum_dataset/vse_png/МТСП.24.08.012 Ребро.png',
    'native': 'dataset/clear_sum_dataset/vse_dxf/МТСП.24.08.012_Ребро.dxf',
    'preparation': 'artifacts/clear-raster-prepared-v1/source-0031-097db6a4e770/preparation.json',
    'references': 'configs/reviews/raster-point-references-v1.json',
    'reference_seal': 'artifacts/raster-point-references-v1/seal.json',
    'parent_fields': 'artifacts/dimension-field-review-v1/report.json',
    'parent_labels': 'artifacts/labels-corpus-v1/report.json',
    'parent_proof': 'artifacts/three-stage-batch-final-v1.json',
    'selection': REVIEW_DIR+'holdout-selection.json',
    'readout': REVIEW_DIR+'holdout-center-readout.json',
    'method': REVIEW_DIR+'center-readout-method.json',
    'review': REVIEW_DIR+'visual-review-notes.json',
    'crop_provenance': REVIEW_DIR+'crop-provenance.json',
    'independent_manifest': REVIEW_DIR+'manifest.json',
}
CODE_FILES = ['src/cad_recovery/'+name+'.py' for name in (
    'rib_mini_review_v1', 'rib_ring_readout_v1', 'interval_scale_v1', 'graph_v2',
    'raster_regions', 'label_schema_v1', 'label_release_v1', 'clear_pairs',
    'schema', 'family_catalog_v2')]
CONTEXTS = {'context': [1700, 1160, 3480, 3830],
            'lower-expanded': [1750, 2860, 2850, 3830],
            'upper-finite-flat': [2580, 1240, 3130, 1445]}


class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)


class ReviewConfigV1(Strict):
    schema_version: Literal['rib-mini-review-config/1.0']
    pair_id: Literal['pair-0031']
    view_id: Literal['main-profile']
    pins: dict[str, Pin] = Field(min_length=14, max_length=200)


class ReviewedFieldV1(Strict):
    subject_id: str
    field_path: str
    value: JsonValue
    unit: str | None
    frame: str | None
    decision: Literal['reviewed_observation', 'conditional_hypothesis', 'unresolved']
    evidence: list[Pin] = Field(min_length=1)
    accuracy_bound: None = None


class FieldReviewV1(Strict):
    schema_version: Literal['field-review/1.0'] = 'field-review/1.0'
    config: Pin
    source: Pin
    pair_id: Literal['pair-0031'] = 'pair-0031'
    view_id: Literal['main-profile'] = 'main-profile'
    reviewer: Literal['assistant'] = 'assistant'
    fields: list[ReviewedFieldV1]
    human_verified: Literal[False] = False
    physical_scale_confirmed: Literal[False] = False
    annotation_approved: Literal[False] = False
    training_eligible: Literal[False] = False
    split: None = None


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                               indent=2, allow_nan=False)+'\n', encoding='utf-8')


def source_to_view(point):
    return [point[0]-VIEW_BOX[0], VIEW_BOX[3]-1-point[1]]


def interval_to_view(box):
    corners = [source_to_view([x, y]) for x in (box[0], box[2]) for y in (box[1], box[3])]
    return [min(p[0] for p in corners), min(p[1] for p in corners),
            max(p[0] for p in corners), max(p[1] for p in corners)]


def verify_inputs_v1(workspace, config_pin):
    """Authenticate actual bytes and replay all selected raster observations."""
    workspace = Path(workspace).resolve()
    config_path = checked_pin(workspace, config_pin)
    config = ReviewConfigV1.model_validate(read_json_v1(config_path))
    pins = {key: value.model_dump() for key, value in config.pins.items()}
    for key, expected in KEY_PATHS.items():
        require(key in pins and pins[key]['path'] == expected, 'Missing or replaced review input: '+key)
    by_path = {v['path']: v for v in pins.values()}
    require(len(by_path) == len(pins), 'Repeated input paths')
    require(all(p in by_path for p in CODE_FILES), 'Runner and dependency code must be pinned')
    for pin in pins.values():
        checked_pin(workspace, pin)
    data = {key: read_json_v1(workspace/pin['path']) for key, pin in pins.items()
            if key in KEY_PATHS and key not in ('source', 'native')}
    seal = data['reference_seal']
    require(seal['reference_path'] == pins['references']['path'] and
            seal['reference_sha256'] == pins['references']['sha256'] and
            seal['point_count'] == len(data['references']['points']), 'Reference seal no longer binds the references')
    sealed_files = data['independent_manifest']['files']
    for name, sha in sealed_files.items():
        require(by_path.get(REVIEW_DIR+name) == {'path': REVIEW_DIR+name, 'sha256': sha},
                'Independent review manifest differs from pinned input: '+name)
    require(set(sealed_files) == {Path(p).name for p in by_path if p.startswith(REVIEW_DIR)
                                 and p != KEY_PATHS['independent_manifest']}, 'Independent manifest inventory differs')
    selection, review, readout = (data[k] for k in ('selection', 'review', 'readout'))
    for item in (selection, review):
        require(item['source']['sha256'] == pins['source']['sha256'] and
                item['source']['path'] == pins['source']['path'] and
                item['source']['size_px'] == [4986, 7042] and
                item['pair_id'] == 'pair-0031' and item['view_id'] == 'main-profile' and
                item['view_box_px'] == VIEW_BOX, 'Review source/view mismatch')
    require(selection['phase'] == 'selection_before_new_scale_or_holdout_residual' and
            selection['calibration_dimension_ids'] == ['1B2', '1C5'] and
            selection['holdout']['dimension_id'] == '1EB' and
            selection['holdout']['printed_text'] == '100' and
            selection['holdout']['used_for_calibration'] is False and
            selection['holdout']['center_interval_half_width_px_predeclared'] == 4,
            'Calibration/holdout selection changed')
    require(selection['visual_readings'] == {'chamfer-left-label': '20x45 degrees;2 chamfers',
            'inner-height-label': '160', 'hole-center-spacing-label': '100'}, 'Preselected readings changed')
    require(readout['source_sha256'] == pins['source']['sha256'] and
            readout['selection_sha256'] == pins['selection']['sha256'] and
            readout['method_sha256'] == pins['method']['sha256'] and
            readout['script_sha256'] == by_path[REVIEW_DIR+'read-raster-centers.py']['sha256'],
            'Holdout provenance mismatch')
    for item in (selection, review, readout):
        require(all(item[k] is False for k in ('human_verified', 'physical_accuracy_approved', 'training_eligible',
                                              'new_scale_computed', 'holdout_residual_computed')),
                'Independent observation record contains unsupported approval or fitted data')
    prep = data['preparation']
    require(prep['source_sha256'] == pins['source']['sha256'], 'Preparation source differs')
    parents = [p for p in data['parent_fields']['pairs'] if p['pair_id'] == 'pair-0031']
    require(len(parents) == 1, 'Parent pair must have one unambiguous identity')
    parent = parents[0]
    require(parent['input_sha256'] == pins['source']['sha256'] and
            parent['target_sha256'] == pins['native']['sha256'] and
            parent['annotation_approved'] is False and parent['training_eligible'] is False,
            'Parent identity or approval changed; requires a new review version')
    readings = {'1B2': '20×45°; 2 фаски', '1C5': '160', '1EB': '100'}
    require(len(review['dimensions']) == 3 and {r['id'] for r in review['dimensions']} == set(readings),
            'Reviewed dimension inventory changed')
    measurements = {'1B2': 'X projection between rib_lower_vertical and rib_lower_flat, not their diagonal distance',
                    '1C5': 'Y projection between rib_upper_flat and rib_lower_flat',
                    '1EB': 'Y projection between the two visible hole ring centers'}
    for identifier, text in readings.items():
        reviewed = next(r for r in review['dimensions'] if r['id'] == identifier)
        identities = [r for r in parent['objects'] if r['identity']['handle'] == identifier]
        require(len(identities) == 1, 'Dimension identity is ambiguous')
        native_identity = identities[0]
        require(reviewed['source_reading'] == native_identity['source_reading']['text'] == text,
                'Printed dimension reading differs from review recipe')
        require(reviewed['measurement'] == measurements[identifier], 'Reviewed projection semantics changed')
    require(review['dimensions'][0]['additional_semantics'] == {'chamfer_angle_deg': 45, 'count': 2},
            'Chamfer notation changed')
    records = {p['id']: p for p in data['references']['points']}
    require(len(records) == len(data['references']['points']), 'Repeated reference point IDs')
    roles = review['calibration_roles']
    require(len(roles) == 3 and {r['id'] for r in roles} == set(ANCHORS), 'Reviewed contour role inventory changed')
    calibration = selection['calibration_anchor_records']
    require([p['id'] for p in calibration] == ANCHORS, 'Calibration anchor inventory changed')
    regions = list(selection['evidence_regions'])+list(data['crop_provenance'])
    for point in calibration:
        original = records[point['id']]
        role = next(r for r in review['calibration_roles'] if r['id'] == point['id'])
        require(original['source_sha256'] == pins['source']['sha256'], 'Mixed source point')
        for key in ('center_source_xy', 'interval_source_xyxy'):
            require(point[key] == original[key] == role[key], 'Original raster anchor changed')
        e = point['evidence']
        require(e == original['evidence'], 'Original anchor evidence changed')
        require(seal['artifact_sha256'][Path(e['raw_crop_path']).name] == e['raw_crop_sha256'] and
                e['provenance']['preparation_report_sha256'] == pins['preparation']['sha256'] and
                e['provenance']['source_sha256'] == pins['source']['sha256'] and
                e['provenance']['source_box_px'] == e['source_box_px'],
                'Anchor seal or preparation binding differs')
        regions.append({'path': e['raw_crop_path'], 'sha256': e['raw_crop_sha256'],
                        'source_box_px': e['source_box_px']})
    prep_dir = workspace/pins['preparation']['path']
    pixels = {}
    for region in regions:
        pin = {'path': region['path'], 'sha256': region['sha256']}
        require(by_path.get(pin['path']) == pin, 'Raster evidence missing from pinned inventory')
        rgb, provenance = read_prepared_region(prep_dir.parent, region['source_box_px'])
        with Image.open(workspace/pin['path']) as image:
            require(np.array_equal(rgb, np.asarray(image.convert('RGB'))), 'Crop differs from prepared source pixels')
        pixels[pin['path']] = rgb
        require(provenance['source_sha256'] == pins['source']['sha256'], 'Crop has mixed source')
    for point in readout['points']:
        require(point['nominal_used_in_center_estimation'] is False and point['cad_coordinates_used'] is False,
                'Holdout center used calibration information')
        region = point['evidence']
        require(region in selection['evidence_regions'], 'Holdout region was not preselected')
        replay = read_ring_center_v1(pixels[region['path']], data['method'], region['source_box_px'])
        for key, value in replay.items():
            require(point[key] == value, 'Raster center replay differs: '+key)
    require([p['id'] for p in readout['points']] == ['rib_holdout_upper-hole', 'rib_holdout_lower-hole'],
            'Unexpected holdout support inventory')
    for point in calibration+readout['points']:
        x, y = point['center_source_xy']
        x0, y0, x1, y1 = point['interval_source_xyxy']
        require(0 <= x0 <= x <= x1 <= 4985 and 0 <= y0 <= y <= y1 <= 7041,
                'Authored center must lie inside its source interval and pixel extent')
    return pins, data, calibration, readout['points']


def build_problem_v1(pins, calibration, holdouts):
    frame = make_source_frame_binding_v1(pins['source']['sha256'], 0, 'original-png', [4986, 7042])
    def ref(key, locator):
        return {**pins[key], 'locator': locator}
    points = calibration+holdouts
    anchors = [{'id': p['id'], 'frame_sha256': frame.frame_sha256,
                'box_xyxy': p['interval_source_xyxy'], 'reviewed': True,
                'review_refs': [ref('review' if p['id'] in ANCHORS else 'readout', 'anchor/'+p['id'])]}
               for p in points]
    def span(identifier, a, b, axis, direction, nominal):
        return {'id': identifier, 'frame_sha256': frame.frame_sha256, 'anchor_a': a, 'anchor_b': b,
                'axis': axis, 'source_direction': direction, 'nominal_mm': nominal, 'unit': 'mm',
                'nominal_reviewed': True, 'direction_reviewed': True,
                'review_refs': [ref('review', 'dimension/'+identifier)]}
    return {'binding': frame.model_dump(mode='json'), 'anchors': anchors,
            'calibration': [span('1B2', ANCHORS[0], ANCHORS[1], 'X', 'positive', 20.),
                            span('1C5', ANCHORS[1], ANCHORS[2], 'Y', 'negative', 160.)],
            'holdouts': [{**span('1EB', holdouts[0]['id'], holdouts[1]['id'], 'Y', 'positive', 100.),
                          'independent_review': True}]}


def build_graph_v1(pins, data, calibration):
    source_hash = pins['source']['sha256']
    def evidence(box):
        return {'source_ref': pins['source']['path'], 'method': 'derived', 'status': 'observed',
                'review_ref': pins['review']['path'],
                'source_region': {'source_sha256': source_hash, 'coordinate_space': 'source_pixel', 'box': box}}
    pts = [source_to_view(p['center_source_xy']) for p in calibration]
    # D is a clipped endpoint along a visibly continuous flat, not a design corner.
    clipped = source_to_view([3100., calibration[2]['center_source_xy'][1]])
    common = {'view_id': 'main-profile', 'status': 'proposed'}
    primitives = [dict(common, id='lower-chamfer', kind='line', start=pts[0], end=pts[1],
                       evidence=evidence(CONTEXTS['lower-expanded'])),
                  dict(common, id='upper-flat-fragment', kind='line', start=pts[2], end=clipped,
                       evidence=evidence(CONTEXTS['upper-finite-flat']))]
    dims = [dict(common, id='1B2', kind='linear', nominal=20., raw_text='20×45°; 2 фаски',
                 points=[pts[1], pts[0]], placement=source_to_view([2560., 3608.]),
                 measured_entity_ids=['lower-chamfer'], angle_deg=0., evidence=evidence(CONTEXTS['lower-expanded'])),
            dict(common, id='1C5', kind='linear', nominal=160., raw_text='160',
                 points=[pts[1], pts[2]], placement=source_to_view([1880., 2250.]),
                 measured_entity_ids=['lower-chamfer', 'upper-flat-fragment'], angle_deg=90.,
                 evidence=evidence(CONTEXTS['context']))]
    meta = data['preparation']['decoder']['physical_pixel_dimensions']
    require(meta['unit'] == 'meter' and meta['pixels_per_unit_x'] == meta['pixels_per_unit_y'] == 3780,
            'Image density metadata changed')
    q = 1000./meta['pixels_per_unit_x']
    graph = DrawingGraphV2.model_validate({
        'source': {**pins['source'], 'page_index': 0, 'method': 'source_pixel_review'},
        'page': {'width_mm': 4986*q, 'height_mm': 7042*q, 'size_source': 'image_dpi', 'kind': 'raster'},
        'views': [{'id': 'main-profile', 'coordinate_space': 'pixel', 'evidence': evidence(VIEW_BOX),
                   'source_to_view': [[1, 0, -2400], [0, -1, 3419], [0, 0, 1]],
                   'view_to_sheet': [[q, 0, 2400*q], [0, q, (7042-3420)*q], [0, 0, 1]]}],
        'primitives': primitives, 'dimensions': dims,
        'issues': [{'code': code, 'message': message} for code, message in (
            ('conditional_scale', 'Printed nominals are a conditional mm design hypothesis; physical scale is unconfirmed.'),
            ('partial_geometry', 'Two finite observed fragments only; upper flat end is deliberately clipped.'),
            ('metadata_page_size', 'Image DPI is metadata, not original paper format or model scale.'),
            ('authored_intervals', 'Authored pixel intervals are not accuracy or confidence bounds.'),
            ('no_label_release', 'No physical target or training split has been approved.'))],
        'provenance': {'pair_id': 'pair-0031', 'input_pins': pins,
            'geometry_coordinate_basis': 'unchanged_authored_raster_centers_not_solver_witness',
            'nominal_unit_assumption': 'mm_per_printed_linear_dimension_convention_not_verified_physical_size',
            'notation': {'1B2': {'projected_length': 20, 'axis': 'X', 'chamfer_angle_deg': 45, 'count': 2}},
            'clipped_endpoints': [{'primitive_id': 'upper-flat-fragment', 'endpoint': 'end',
                                   'source_xy': [3100., calibration[2]['center_source_xy'][1]],
                                   'design_corner': False}],
            'physical_scale_confirmed': False, 'annotation_approved': False, 'training_eligible': False}})
    return graph.model_dump(mode='json')


def assemble_review_v1(pins, data, calibration, holdouts, config_pin):
    problem = build_problem_v1(pins, calibration, holdouts)
    result = solve_interval_scale_v1(problem)
    baseline = deepcopy(problem)
    baseline['holdouts'] = []
    require(solve_interval_scale_v1(baseline)['calibration'] == result['calibration'], 'Holdout leaked into calibration')
    graph = build_graph_v1(pins, data, calibration)
    fields = []
    def add(subject, path, value, unit=None, frame=None, decision='reviewed_observation', keys=('review',)):
        fields.append({'subject_id': subject, 'field_path': path, 'value': value, 'unit': unit,
                       'frame': frame, 'decision': decision, 'evidence': [pins[k] for k in keys]})
    for p in calibration+holdouts:
        keys = ('references', 'review') if p['id'] in ANCHORS else ('selection', 'readout')
        add(p['id'], 'center', p['center_source_xy'], 'source_pixel', 'original-png', keys=keys)
        add(p['id'], 'authored_interval', p['interval_source_xyxy'], 'source_pixel', 'original-png', keys=keys)
    for role in data['review']['calibration_roles']:
        add(role['id'], 'contour_junction_role', role['role'])
    for identifier, text, nominal, axis in [('1B2', '20×45°; 2 фаски', 20, 'X'), ('1C5', '160', 160, 'Y'), ('1EB', '100', 100, 'Y')]:
        add(identifier, 'printed_text', text)
        add(identifier, 'printed_numeric_value', nominal)
        add(identifier, 'projection_axis', axis, frame='original-png')
        add(identifier, 'physical_unit_assumption', 'mm', decision='conditional_hypothesis')
    add('1B2', 'chamfer_angle', 45, 'degree')
    add('1B2', 'chamfer_count', 2)
    add('main-profile', 'scale_interval', result['calibration']['scale_mm_per_px'], 'mm_per_source_pixel',
        'original-png', 'conditional_hypothesis', ('review', 'references', 'selection'))
    descriptor = FieldReviewV1.model_validate({'config': config_pin, 'source': pins['source'], 'fields': fields}).model_dump(mode='json')
    bindings = [
        {'dimension_id': '1B2', 'endpoint': 'defpoint2', 'anchor_id': ANCHORS[1], 'primitive_id': 'lower-chamfer', 'primitive_endpoint': 'end'},
        {'dimension_id': '1B2', 'endpoint': 'defpoint3', 'anchor_id': ANCHORS[0], 'primitive_id': 'lower-chamfer', 'primitive_endpoint': 'start'},
        {'dimension_id': '1C5', 'endpoint': 'defpoint2', 'anchor_id': ANCHORS[1], 'primitive_id': 'lower-chamfer', 'primitive_endpoint': 'end'},
        {'dimension_id': '1C5', 'endpoint': 'defpoint3', 'anchor_id': ANCHORS[2], 'primitive_id': 'upper-flat-fragment', 'primitive_endpoint': 'start'}]
    ledger = {'coordinate_convention': 'pixel_centers_inclusive_intervals', 'view_box_half_open': VIEW_BOX,
              'anchors': [{**{k: p[k] for k in ('id', 'center_source_xy', 'interval_source_xyxy')},
                           'center_view_xy': source_to_view(p['center_source_xy']),
                           'interval_view_xyxy': interval_to_view(p['interval_source_xyxy'])} for p in calibration+holdouts],
              'finite_bindings': bindings, 'native_handles_scope': 'identity_only_not_coordinate_truth',
              'holdout_used_for_calibration': False, 'source_intervals_unchanged': True}
    for row in bindings:
        add(row['dimension_id'], 'finite_binding/'+row['endpoint'], row)
    # Include bindings after assembling all rows; validation forbids accidental approval.
    descriptor['fields'] = [ReviewedFieldV1.model_validate(row).model_dump(mode='json') for row in fields]
    return {'graph.json': graph, 'field-review.json': descriptor, 'anchor-ledger.json': ledger,
            'interval-problem.json': problem, 'interval-result.json': result}


def preview_svg(calibration, holdouts):
    box = CONTEXTS['context']
    width, height = box[2]-box[0], box[3]-box[1]
    elements = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{box[0]} {box[1]} {width} {height}">',
                f'<image href="context.png" x="{box[0]}" y="{box[1]}" width="{width}" height="{height}"/>']
    a, b, c = [p['center_source_xy'] for p in calibration]
    for start, end in [(a, b), (c, [3100, c[1]])]:
        elements.append(f'<path d="M{start[0]},{start[1]} L{end[0]},{end[1]}" stroke="#be245c" stroke-width="5" fill="none"/>')
    for index, p in enumerate(calibration+holdouts):
        x, y = p['center_source_xy']; lo_x, lo_y, hi_x, hi_y = p['interval_source_xyxy']
        elements.append(f'<rect x="{lo_x}" y="{lo_y}" width="{hi_x-lo_x}" height="{hi_y-lo_y}" fill="none" stroke="#005ac2" stroke-width="3"/>')
        elements.append(f'<circle cx="{x}" cy="{y}" r="12" fill="none" stroke="#005ac2" stroke-width="3"/>')
        elements.append(f'<text x="{x+20}" y="{y-18}" font-size="32" fill="#005ac2">{escape(chr(65+index))}</text>')
    return ''.join(elements)+'</svg>\n'


def run_rib_mini_review_v1(workspace, config_pin, output_dir):
    workspace = Path(workspace).resolve()
    output = contained(workspace, output_dir)
    artifacts = (workspace/'artifacts').resolve()
    require(artifacts != workspace and artifacts.is_relative_to(workspace) and
            output != artifacts and output.is_relative_to(artifacts),
            'Review output must be a new directory under workspace/artifacts')
    require(not output.exists(), 'Use a new output version; never overwrite review artifacts')
    pins, data, calibration, holdouts = verify_inputs_v1(workspace, config_pin)
    documents = assemble_review_v1(pins, data, calibration, holdouts, config_pin)
    output.mkdir(parents=True)
    contexts = []
    for name, box in CONTEXTS.items():
        rgb, provenance = read_prepared_region((workspace/pins['preparation']['path']).parent, box, max_pixels=6_000_000)
        Image.fromarray(rgb).save(output/(name+'.png'))
        contexts.append({'path': name+'.png', 'sha256': digest(output/(name+'.png')), 'provenance': provenance,
                         'original_source_rehashed_by_review_runner': True})
    documents['contexts.json'] = contexts
    result = documents['interval-result.json']
    report = {'schema_version': 'rib-mini-review/1.0', 'config': config_pin, 'inputs': pins,
              'pair_id': 'pair-0031', 'calibration_ids': ['1B2', '1C5'], 'holdout_ids': ['1EB'],
              'scale_mm_per_source_pixel': result['calibration']['scale_mm_per_px'],
              'calibration_status': result['calibration']['status'],
              'holdout_status': result['holdouts'][0]['status'],
              'holdout_span_px': result['holdouts'][0]['span_px'],
              'holdout_did_not_change_calibration': True, 'finite_binding_count': 4,
              'field_count': len(documents['field-review.json']['fields']),
              'physical_scale_confirmed': False, 'annotation_approved': False,
              'training_eligible': False, 'split': None,
              'limitations': ['Assistant-authored +/-4px boxes are not statistical or metrological accuracy bounds.',
                  'Uniform scale and mm printed nominals are conditional design assumptions.',
                  'The holdout is excluded from calibration; prior CAD metadata exposure prevents a blind-test claim.',
                  'Only two finite raster geometry fragments; no complete contour or target reconstruction.',
                  'Image DPI does not establish original paper format or part scale.',
                  'Parent label and batch report pins are preserved lineage snapshots; this runner does not reapprove their contents.',
                  'No frozen labels, family allocations, resource exclusions or training data were changed.']}
    documents['report.json'] = report
    for name, value in documents.items():
        write_json(output/name, value)
    (output/'overlay.svg').write_text(preview_svg(calibration, holdouts), encoding='utf-8')
    scale = report['scale_mm_per_source_pixel']
    scale_text = f"{scale['lower']:.9f}–{scale['upper']:.9f}" if scale else 'конфликт'
    html = ('<!doctype html><meta charset="utf-8"><title>Ребро — проверка опор</title>'
            '<style>body{max-width:1000px;margin:2em auto;font:18px system-ui;line-height:1.5}object,img{max-width:100%}</style>'
            '<h1>Ребро: два размера и отдельный контроль</h1>'
            '<p>20×45°, 2 фаски: калибровка по проекции X=20. Второе направление: Y=160. '
            'Контроль 100 между центрами отверстий исключён из калибровки.</p>'
            f'<p>Условный диапазон: {scale_text} мм/пиксель. Контроль: {escape(report["holdout_status"])}.</p>'
            '<p>Синие отметки A/B/C — исходные опоры, D/E — центры отверстий; прямоугольники — авторские ±4 px. '
            'Малиновые линии — два конечных фрагмента графа. Правый конец верхней линии обрезан для этого обзора.</p>'
            '<p>Диапазоны не являются допуском точности. Физический масштаб и обучающие метки не утверждены.</p>'
            '<p><a href="context.png">Исходные пиксели</a> · <a href="graph.json">Граф</a> · '
            '<a href="anchor-ledger.json">Привязки</a> · <a href="field-review.json">Поля</a> · '
            '<a href="interval-result.json">Расчёт</a> · <a href="report.json">Отчёт</a></p>'
            '<object data="overlay.svg" type="image/svg+xml" width="100%"></object>')
    (output/'review.html').write_text(html, encoding='utf-8')
    # A completed manifest must never attest inputs which changed during the run.
    checked_pin(workspace, config_pin)
    for pin in pins.values():
        checked_pin(workspace, pin)
    write_json(output/'manifest.json', {'schema_version': 'rib-mini-manifest/1.0', 'config': config_pin,
               'files': [{'path': p.name, 'sha256': digest(p)} for p in sorted(output.iterdir())]})
    return report
