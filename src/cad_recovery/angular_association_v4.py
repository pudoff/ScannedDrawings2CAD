"""Opt-in angular association experiment with evidenced leader and occlusion paths."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import angular_association_v1 as core
from . import angular_association_v2 as baseline
from . import angular_leader_search_v2 as search
from . import dimension_association_v3 as linear
from .family_catalog_v2 import require
from .notation import parse_notation

POLICY = {'id': 'angular-association/4.0', 'search': search.POLICY,
          'leader_observations': 'Separate same-image line packet; only shelf/leader roles, never measured rays.',
          'physical_geometry_approved': False, 'production_admitted': False, 'training_eligible': False}
angular_contract = core.angular_contract


def associate_angular_dimensions(source_image, output_dir, *, geometry, ocr, arrows,
                                 leader_geometry=None, upstream_incomplete=False, prefix='page-001'):
    source_image, output_dir = Path(source_image), Path(output_dir)
    require(type(upstream_incomplete) is bool, 'Upstream completeness must be explicit boolean')
    require(not output_dir.exists() and not output_dir.is_symlink(), 'Angular v4 output must be new')
    digest, size, lines, curves, heads = linear._validate(source_image, geometry, ocr, arrows)
    leader_geometry = geometry if leader_geometry is None else leader_geometry
    _, _, leader_lines, _, _ = linear._validate(source_image, leader_geometry, ocr, arrows)
    require(all(np.abs(r['points']).max() <= 1e7 for r in leader_lines), 'Leader geometry outside supported bounds')
    report = baseline.associate_angular_dimensions(source_image, output_dir/'baseline-v2', geometry=geometry,
                                                   ocr=ocr, arrows=arrows, prefix=prefix)
    report['linear_baseline_artifacts'] = {k: 'baseline-v2/'+v for k, v in report['linear_baseline_artifacts'].items()}
    report['algorithm_version'] = 'angular-association-v4'
    report['angular_leader_policy'] = POLICY
    budget = search.Budget(); raster = core.RasterEvidence(source_image, ocr, arrows, budget)
    text_rows = {r['id']: r for r in ocr.get('lines', [])}
    notations = {r['id']: r for r in ocr.get('notations', [])}
    owners = {}; incomplete = []
    for item in report['associations']:
        row = text_rows.get(item['source_line_id']); notation = notations[item['notation_id']]
        if row is None or notation.get('context') == 'title_block_candidate':
            continue
        syntax = parse_notation(row['raw']); kind = syntax.get('kind')
        explicit = kind == 'angle' or '\u00b0' in row['raw']
        if kind == 'chamfer' or not explicit and kind not in ('nominal', None):
            continue
        budget.notation = 0; budget.junction_notation = 0; limited = None; extended = []
        try:
            extended = search.hypotheses(linear._text_geometry(row), lines, curves, heads, leader_lines, raster, budget)
        except core.SearchLimit as error:
            limited = str(error); incomplete.append({'notation_id': item['notation_id'], 'reason': limited})
        if not extended and limited is None:
            continue
        original = deepcopy(item)
        choices = extended
        # Keep prior direct contexts as alternatives when the new search did not reproduce them.
        for old in original.get('alternatives', []):
            if 'dimension_arc_candidate_id' not in old:
                continue
            if not any(old['dimension_arc_candidate_id'] == c['dimension_arc_candidate_id'] and
                       old['measured_candidate_ids'] == c['measured_candidate_ids'] and
                       old['arrow_candidate_ids'] == c['arrow_candidate_ids'] for c in choices):
                choices.append(old)
        item.update(kind='angular' if explicit else 'unknown', nominal_mm=None,
                    nominal_unit='degree' if explicit else 'unverified', angle_deg=None,
                    status='ambiguous', anchors_px=[], measured_candidate_ids=[], dimension_line_candidate_ids=[],
                    extension_line_candidate_ids=[], arrow_candidate_ids=[], alternatives=choices,
                    evidence={}, reasons=[])
        item.pop('observed_sector_deg', None)
        item['type_hypotheses'] = ['angular'] if explicit else ['linear', 'angular_degree_sign_missing']
        if not explicit:
            item['prior_linear_hypothesis'] = original.get('prior_linear_hypothesis') or original
            item['reasons'].append('linear_reading_conflicts_with_observed_angular_context')
        else:
            value = syntax.get('values', {}).get('angle_deg')
            if syntax.get('status') == 'parsed' and kind == 'angle' and value is not None and 0 < value < 360:
                item['angle_deg'] = value
                parsed = notation.get('parsed', {})
                if parsed.get('kind') != 'angle' or parsed.get('values', {}).get('angle_deg') != value:
                    item['reasons'].append('notation_and_raw_angle_disagree')
            else:
                item['reasons'].append('angle_text_requires_review')
            item['graph_contract'] = angular_contract(item)
        if limited or upstream_incomplete or report['angular_search']['incomplete']:
            item['reasons'].append('angular_search_incomplete:'+str(limited or 'upstream_or_baseline'))
        if (row['confidence'] < .65 or notation.get('confidence', row['confidence']) < .65 or
                row.get('conflicting_line_ids') or notation.get('status') != 'proposed'):
            item['reasons'].append('ocr_or_notation_reading_requires_review')
        if len(choices) == 1 and choices[0]['complete_support'] and explicit and not item['reasons']:
            choice = choices[0]
            # A mismatch is a review reason after support selection, never a ranking feature.
            if abs(choice['observed_sector_deg']-item['angle_deg']) > 3:
                item['reasons'].append('observed_sector_disagrees_with_text')
            else:
                item.update(status='proposed', anchors_px=choice['points_px'],
                            measured_candidate_ids=choice['measured_candidate_ids'],
                            dimension_line_candidate_ids=[choice['dimension_arc_candidate_id']],
                            arrow_candidate_ids=choice['arrow_candidate_ids'], evidence=choice['evidence'],
                            observed_sector_deg=choice['observed_sector_deg'])
        else:
            item['reasons'].append('angular_support_incomplete_or_not_unique')
    # Include unchanged baseline contexts in competition with the new paths.
    for item in report['associations']:
        for c in item.get('alternatives', []):
            if 'dimension_arc_candidate_id' in c:
                owners.setdefault(c['dimension_arc_candidate_id'], set()).add(item['notation_id'])
    for item in report['associations']:
        competing = []
        for c in item.get('alternatives', []):
            if 'dimension_arc_candidate_id' in c:
                competing += sorted(owners[c['dimension_arc_candidate_id']]-{item['notation_id']})
        if competing or upstream_incomplete:
            item.update(status='ambiguous', anchors_px=[], measured_candidate_ids=[],
                        dimension_line_candidate_ids=[], arrow_candidate_ids=[], evidence={})
            item.pop('observed_sector_deg', None)
            if competing:
                item['competing_notation_ids'] = sorted(set(competing))
                if 'competing_texts_for_dimension_arc' not in item['reasons']:
                    item['reasons'].append('competing_texts_for_dimension_arc')
            if upstream_incomplete:
                item['reasons'].append('upstream_search_incomplete')
    report['leader_search'] = {'checks': budget.page, 'pixel_cells': budget.cells, 'incomplete': incomplete,
                               'upstream_incomplete': upstream_incomplete, 'local_junctions': budget.stats(),
                               'leader_geometry_sha256': hashlib.sha256(json.dumps(leader_geometry, sort_keys=True).encode()).hexdigest()}
    report['summary'] = {s: sum(r['status'] == s for r in report['associations']) for s in ('proposed', 'ambiguous', 'unresolved', 'unsupported')}
    report['summary']['total'] = len(report['associations'])
    report['artifacts'] = {'report': prefix+'-angular-associations.json', 'preview': prefix+'-angular-associations.png'}
    with Image.open(source_image) as source:
        image = source.convert('RGB'); image.thumbnail((1600, 1600))
    scale = np.asarray(image.size)/size; draw = ImageDraw.Draw(image)
    for item in report['associations']:
        color = '#008060' if item['status'] == 'proposed' else '#bd6320'
        polygon = (np.asarray(item['text_polygon_px'])*scale).tolist()
        if polygon:
            draw.line([tuple(p) for p in polygon+[polygon[0]]], fill=color, width=2)
        if len(item['anchors_px']) == 3:
            for p in item['anchors_px'][1:]:
                draw.line([tuple(np.asarray(item['anchors_px'][0])*scale), tuple(np.asarray(p)*scale)], fill=color, width=2)
    image.save(output_dir/report['artifacts']['preview'])
    report['artifact_sha256'] = {'preview': hashlib.sha256((output_dir/report['artifacts']['preview']).read_bytes()).hexdigest()}
    require(hashlib.sha256(source_image.read_bytes()).hexdigest() == digest, 'Source changed during association')
    (output_dir/report['artifacts']['report']).write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return report
