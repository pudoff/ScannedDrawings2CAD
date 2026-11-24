"""Two-pass linear-chain ownership, with an immutable hypothesis inventory.

Peer-free geometry is only a hypothesis, never a published relationship.
Certificates require a uniquely separated, pixel-supported internal chain;
unknown, shared and externally placed texts remain blocking observations.
"""
from copy import deepcopy
from pathlib import Path
import math
import numpy as np

from . import dimension_association_v3 as v3
from .contour_topology_v3 import ContourTopology
from .dimension_support_v1 import PixelSupport, parse_composite, _axes, _oriented, _add_equivalent, _demote
from .dimension_support_verified_v1 import _arrow_ink

POLICY = {
    'version': 'text-chain-ownership/1.0', 'minimum_confidence': .65,
    'distance_dominance_factor': 1.5, 'angle_degrees': 12,
    'minimum_carrier_ink': .85, 'minimum_face_ink': .85, 'minimum_head_ink': .85,
    'maximum_texts': 64, 'maximum_inventory_choices': 512,
    'maximum_revalidation_calls': 64,
    'external_text_can_supply_certificate': False,
    'maximum_checks_per_text': v3.MAX_CHECKS_PER_NOTATION,
    'maximum_checks_per_page': v3.MAX_CHECKS_PER_PAGE,
    'production_admitted': False, 'physical_geometry_approved': False,
}
COS = math.cos(math.radians(POLICY['angle_degrees']))
FIELDS = ('anchors_px', 'measured_candidate_ids', 'dimension_line_candidate_ids',
          'extension_line_candidate_ids', 'arrow_candidate_ids', 'evidence')


def _key(choice):
    # Different anchors/faces on the SAME annotation chain stay alternatives.
    return (tuple(choice['dimension_line_candidate_ids']), tuple(choice['arrow_candidate_ids']))


def _distance(text, choice, heads):
    first, second = [heads[k] for k in choice['arrow_candidate_ids']]
    delta = np.array(second['tip_px']) - first['tip_px']
    axis = delta / np.linalg.norm(delta)
    return abs(v3._cross(text['center'] - first['tip_px'], axis))


class PixelChecks:
    def __init__(self, source, geometry, arrows, topology):
        self.pixels = PixelSupport(source)
        self.geometry = {g['id']: g for g in geometry['candidates']}
        self.arrows = {a['id']: a for a in arrows['candidates']}
        self.faces = {f['id']: f for f in topology.report()['faces']}
        self.segments, self.heads = {}, {}

    def segment(self, points):
        key = tuple(tuple(p) for p in points)
        if key not in self.segments:
            self.segments[key] = self.pixels.segment(*np.array(points, dtype=float))
        return self.segments[key]

    def verify(self, choice, supported_arrows):
        if not set(choice['arrow_candidate_ids']).issubset(supported_arrows):
            return {'valid': False, 'reason': 'arrow_owned_shaft_unverified'}
        evidence = choice['evidence']
        if evidence.get('contour_support') != 'shared_bounded_face' or not evidence.get('contour_face_ids'):
            return {'valid': False, 'reason': 'shared_bounded_face_missing'}
        fractions = {'face': [], 'carriers': {}, 'heads': {}}
        for fid in evidence['contour_face_ids']:
            for edge in self.faces[fid]['boundaries']:
                fractions['face'].append(self.segment(edge['points_px']))
        for gid in set(choice['dimension_line_candidate_ids'] + choice['extension_line_candidate_ids']):
            fractions['carriers'][gid] = self.segment(self.geometry[gid]['points_px'])
        for aid in choice['arrow_candidate_ids']:
            if aid not in self.heads:
                self.heads[aid] = _arrow_ink(self.pixels, self.arrows[aid])
            fractions['heads'][aid] = self.heads[aid]
        valid = (min(fractions['face'], default=0) >= POLICY['minimum_face_ink']
                 and min(fractions['carriers'].values(), default=0) >= POLICY['minimum_carrier_ink']
                 and min(fractions['heads'].values(), default=0) >= POLICY['minimum_head_ink'])
        return {'valid': bool(valid), 'reason': 'observed_chain_pixels' if valid else 'chain_pixels_missing',
                'fractions': fractions}


def resolve(source_image, *, geometry, ocr, arrows, support, baseline):
    """All inputs are source observations. No ground truth or nominal matching."""
    source_image = Path(source_image)
    digest, size, lines, _, heads_normalized = v3._validate(source_image, geometry, ocr, arrows)
    lines = sorted(lines, key=lambda r: r['id'])
    heads_normalized = sorted(heads_normalized, key=lambda r: r['id'])
    for report in (baseline, support):
        if report.get('source_sha256') != digest or report.get('processed_size_px') != size:
            raise ValueError('Ownership input frame mismatch')
    result = deepcopy(baseline)
    result['algorithm_version'] = POLICY['version']
    supported_arrows = {s['arrow_id'] for s in support['spans']}
    heads = {a['id']: a for a in arrows['candidates']}
    texts = {l['id']: v3._text_geometry(l) for l in ocr['lines']}
    by_line = {l['id']: l for l in ocr['lines']}
    notation_by = {n['source_line_id']: n for n in ocr['notations']}
    for line in ocr['lines']:
        parsed_raw = notation_by.get(line['id'], {}).get('parsed', {}).get('raw', line['raw'])
        if parsed_raw != line['raw']:
            raise ValueError('Parsed notation does not match the observed raw reading')
    for row in baseline['associations']:
        if row['source_line_id'] in by_line and row['raw'] != by_line[row['source_line_id']]['raw']:
            raise ValueError('Baseline is bound to a different OCR reading')
    eligible = []
    for line in sorted(ocr['lines'], key=lambda l: l['id']):
        parsed = parse_composite(line['raw'])
        if (parsed['status'] == 'parsed' and parsed['kind'] in ('nominal', 'linear')
                and line['confidence'] >= POLICY['minimum_confidence']
                and not line.get('conflicting_line_ids')
                and notation_by.get(line['id'], {}).get('context') != 'title_block_candidate'):
            eligible.append(line['id'])
    audit = {'policy': deepcopy(POLICY), 'complete': True, 'inventory': [], 'certificates': {},
             'blocked_certificates': [], 'peer_decisions': [], 'revalidations': [],
             'search_checks_by_text': {}, 'search_checks': 0, 'pixel_samples': 0,
             'scope': 'linear chains in provided CV graph; frozen v3 neighborhood rules and caps',
             'global_native_graph_solved': False, 'upstream_acceptance_implied': False}
    result['text_chain_ownership'] = audit
    complete = support['complete'] and len(ocr['lines']) <= POLICY['maximum_texts']
    if not complete:
        audit['blocked_certificates'].append({'reason': 'support_or_text_budget_incomplete'})
    excluded = {gid for a in heads_normalized if a['confidence'] >= .65
                for gid in a.get('evidence', {}).get('geometry_ids', [])}
    topology = None
    pixels = None
    choices_by = {}
    orientations = {}
    counts = {}
    budget = {'notation': 0, 'page': 0}
    try:
        if complete and eligible:
            topology = ContourTopology(source_image, lines, excluded_ids=excluded,
                                       text_polygons=[l['polygon_px'] for l in ocr['lines']])
            pixels = PixelChecks(source_image, geometry, arrows, topology)
            complete = topology.complete
            if not complete:
                audit['blocked_certificates'].append({'reason': 'contour_topology_incomplete'})
        if complete:
            for tid in eligible:
                budget['notation'] = 0
                text = texts[tid]
                axes, cut = _axes(text, lines, budget) if text['orientation_uncertain'] else ([text['u']], False)
                orientations[tid] = [_oriented(text, a) for a in axes]
                choices = []
                for oriented in orientations[tid]:
                    found, truncated, _ = v3._linear(oriented, lines, heads_normalized, (), budget, topology=topology)
                    cut |= truncated
                    for choice in found:
                        _add_equivalent(choices, deepcopy(choice))
                counts[tid] = budget['notation']
                entry = {'source_line_id': tid, 'raw': by_line[tid]['raw'], 'complete': not cut,
                         'candidate_checks': budget['notation'], 'choices': []}
                audit['inventory'].append(entry)
                choices_by[tid] = []
                if sum(len(v['choices']) for v in audit['inventory']) + len(choices) > POLICY['maximum_inventory_choices']:
                    raise v3._SearchLimit()
                for choice in choices:
                    check = pixels.verify(choice, supported_arrows)
                    entry['choices'].append({'choice': deepcopy(choice), 'pixel_check': check})
                    if check['valid']:
                        choices_by[tid].append(choice)
                if sum(len(v) for v in choices_by.values()) > POLICY['maximum_inventory_choices']:
                    raise v3._SearchLimit()
                complete &= not cut
                if cut:
                    break  # No certificate or peer waiver can use this prefix.
    except v3._SearchLimit:
        complete = False
        audit['blocked_certificates'].append({'reason': 'inventory_resource_limit'})
    # Freeze all certificates simultaneously. Never manufacture a new domain
    # from a second-pass success or iterate ownership to a convenient fixed point.
    certificates = {}
    if complete:
        for tid in eligible:
            groups = {}
            for choice in choices_by.get(tid, []):
                groups.setdefault(_key(choice), []).append(choice)
            ranked = sorted(((_distance(texts[tid], choices[0], heads), key, choices)
                             for key, choices in groups.items()), key=lambda r: (r[0], r[1]))
            if not ranked:
                continue
            distance, key, choices = ranked[0]
            if any(c['evidence'].get('text_placement') != 'between_arrows' for c in choices):
                audit['blocked_certificates'].append({'source_line_id': tid, 'reason': 'external_domain_not_certifiable'})
                continue
            margin = texts[tid]['tol']
            if len(ranked) > 1 and distance * POLICY['distance_dominance_factor'] + margin >= ranked[1][0]:
                audit['blocked_certificates'].append({'source_line_id': tid, 'reason': 'chain_distance_not_separated'})
                continue
            certificates[tid] = {'source_line_id': tid, 'key': key, 'choices': deepcopy(choices),
                'dimension_line_candidate_ids': list(key[0]), 'arrow_candidate_ids': list(key[1]),
                'own_text_to_line_px': distance, 'distance_margin_px': margin,
                'runner_up_distance_px': ranked[1][0] if len(ranked) > 1 else None,
                'basis': 'complete_pixel_supported_internal_chain_with_distance_dominance'}
        claims = {}
        for tid, cert in certificates.items():
            claims.setdefault(cert['key'], []).append(tid)
        shared = {tid for owners in claims.values() if len(owners) > 1 for tid in owners}
        for tid in sorted(shared):
            certificates.pop(tid)
            audit['blocked_certificates'].append({'source_line_id': tid, 'reason': 'shared_chain_claim'})
    # Unknown peers still block; certified peers may be separated only from
    # this target's independently certified chain, not from every line globally.
    rows = {a['source_line_id']: a for a in result['associations']}
    if complete:
        audit['certificates'] = deepcopy(certificates)
        for tid in eligible:
            if tid not in certificates and choices_by.get(tid) and tid in rows and rows[tid]['status'] == 'proposed':
                rows[tid]['ownership_original_record'] = deepcopy(rows[tid])
                _demote(rows[tid], 'chain_ownership_not_uniquely_supported')
        try:
            for tid, own in sorted(certificates.items()):
                if tid not in rows:
                    continue
                if len(audit['revalidations']) >= POLICY['maximum_revalidation_calls']:
                    raise v3._SearchLimit()
                peers = []
                for oid, other in sorted(texts.items()):
                    if oid == tid:
                        continue
                    cert = certificates.get(oid)
                    separate = False
                    reason = 'unknown_or_shared_ownership'
                    if cert:
                        a = [heads[k] for k in own['arrow_candidate_ids']]
                        b = [heads[k] for k in cert['arrow_candidate_ids']]
                        origin = np.array(a[0]['tip_px']);axis = np.array(a[1]['tip_px']) - origin
                        axis /= np.linalg.norm(axis)
                        other_axis = np.array(b[1]['tip_px']) - b[0]['tip_px']
                        other_axis /= np.linalg.norm(other_axis)
                        competing_distance = abs(v3._cross(other['center'] - origin, axis))
                        shared_lines = set(own['dimension_line_candidate_ids']) & set(cert['dimension_line_candidate_ids'])
                        shared_heads = set(own['arrow_candidate_ids']) & set(cert['arrow_candidate_ids'])
                        margin = max(texts[tid]['tol'], other['tol'])
                        separate = (not shared_lines and not shared_heads
                            and abs(float(np.dot(axis, other_axis))) >= COS
                            and cert['own_text_to_line_px'] * POLICY['distance_dominance_factor'] + margin < competing_distance
                            and abs(v3._cross(np.array(b[0]['tip_px']) - origin, axis)) > 2 * margin)
                        reason = 'distinct_observed_chain' if separate else 'certified_but_not_separated'
                    audit['peer_decisions'].append({'target_text_id': tid, 'other_text_id': oid,
                                                   'waived': bool(separate), 'reason': reason})
                    if not separate:
                        peers.append(other)
                budget['notation'] = counts[tid]  # Continue this text's original budget.
                found = [];cut = False
                for oriented in orientations[tid]:
                    candidates, truncated, _ = v3._linear(oriented, lines, heads_normalized, peers, budget, topology=topology)
                    cut |= truncated
                    for c in candidates:
                        # Only the independently certified target domain can
                        # use its peer-specific waiver. Other domains remain
                        # visible in the immutable first-pass inventory.
                        if _key(c) == own['key'] and pixels.verify(c, supported_arrows)['valid']:
                            _add_equivalent(found, deepcopy(c))
                counts[tid] = budget['notation']
                audit['revalidations'].append({'source_line_id': tid, 'truncated': cut,
                                              'candidate_count': len(found), 'total_text_checks': counts[tid]})
                if cut:
                    raise v3._SearchLimit()
                if not found:
                    continue
                row = rows[tid]
                # Keep old competing anchors, never choose a part face by
                # distance, nominal or the label's certificate.
                previous = [row] if row['status'] == 'proposed' else row.get('alternatives', [])
                for c in previous:
                    _add_equivalent(found, {k: deepcopy(c[k]) for k in FIELDS})
                row['ownership_original_record'] = deepcopy(row)
                for field in FIELDS[:-1]:
                    row[field] = []
                row['alternatives'] = []
                v3._apply_search_result(row, found, False, {}, result['issues'])
                row['evidence']['text_chain_ownership'] = {'certificate_text_id': tid,
                    'peer_waivers': [p for p in audit['peer_decisions'] if p['target_text_id'] == tid and p['waived']],
                    'inventory_frozen_before_revalidation': True}
        except v3._SearchLimit:
            complete = False
            audit['blocked_certificates'].append({'reason': 'revalidation_resource_limit'})
    v3._resolve_text_ownership(result['associations'])
    audit.update(complete=bool(complete), search_checks=budget['page'],
                 search_checks_by_text=counts, pixel_samples=pixels.pixels.samples if pixels else 0,
                 contour_topology=topology.report() if topology else None)
    result['complete'] = bool(baseline.get('complete')) and bool(complete)
    if not complete:
        audit['certificates'] = {}
        for row in result['associations']:
            _demote(row, 'text_chain_ownership_search_incomplete')
    return result
