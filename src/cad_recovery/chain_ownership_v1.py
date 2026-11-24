"""Supported ownership phases with an explicitly supplied verifier.

The c.13 algorithm is expressed as inventory, certificates, and revalidation;
the frozen monolithic solver is neither copied, evaluated nor patched.
"""
from copy import deepcopy
from dataclasses import dataclass, asdict
import math
import numpy as np
from .chain_resources_v1 import ResourceExhausted
from .chain_legacy_bridge_v1 import (SearchCounter, FIELDS, parse_reading,
    merge_choice, demote, apply_choices, resolve_pair_ownership)


@dataclass(frozen=True)
class OwnershipPolicy:
    minimum_confidence: float = .65
    distance_factor: float = 1.5
    maximum_texts: int = 64
    maximum_choices: int = 512
    maximum_revalidations: int = 64


def chain_key(choice):
    return (tuple(choice['dimension_line_candidate_ids']), tuple(choice['arrow_candidate_ids']))


def line_axis(choice, heads):
    first, second = (np.array(heads[k]['tip_px']) for k in choice['arrow_candidate_ids'])
    return first, (second-first)/np.linalg.norm(second-first)


def cross(a, b):
    return float(a[0]*b[1]-a[1]*b[0])


class OwnershipResolver:
    """One invocation owns one inventory; second-pass output cannot seed it."""
    def __init__(self, backend, verifier, ledger, verification_policy, policy=None):
        self.backend, self.verifier, self.ledger = backend, verifier, ledger
        self.verification_policy = verification_policy
        self.policy = policy or OwnershipPolicy()
        self.heads = {a['id']: a for a in backend.arrows['candidates']}
        self.counter = SearchCounter(ledger, 'ownership_search')
        self.counts, self.orientations, self.choices = {}, {}, {}
        self.audit = {'version':'chain-ownership/1.0', 'policy':asdict(self.policy), 'complete':False,
            'inventory':[], 'certificates':{}, 'blocked_certificates':[], 'peer_decisions':[],
            'revalidations':[], 'search_checks_by_text':{}, 'search_checks':0,
            'scope':'provided CV graph; frozen v3 neighborhood rules and caps',
            'global_native_graph_solved':False, 'upstream_acceptance_implied':False}

    def _eligible(self, baseline):
        lines = {l['id']: l for l in self.backend.ocr['lines']}
        notations = {n['source_line_id']: n for n in self.backend.ocr['notations']}
        for tid, line in lines.items():
            if notations.get(tid, {}).get('parsed', {}).get('raw', line['raw']) != line['raw']:
                raise ValueError('Notation is bound to another raw reading')
        for row in baseline['associations']:
            if row['source_line_id'] in lines and row['raw'] != lines[row['source_line_id']]['raw']:
                raise ValueError('Baseline is bound to another raw reading')
        return [tid for tid, line in sorted(lines.items())
                if parse_reading(line['raw'])['status']=='parsed'
                and parse_reading(line['raw'])['kind'] in ('nominal','linear')
                and line['confidence']>=self.policy.minimum_confidence
                and not line.get('conflicting_line_ids')
                and notations.get(tid,{}).get('context')!='title_block_candidate']

    def _search(self, tid, peers):
        self.counter['notation'] = self.counts.get(tid, 0)
        result = []
        try:
            for text in self.orientations[tid]:
                found, cut, _ = self.backend.linear(text, peers, self.counter, self.topology)
                if cut: raise ResourceExhausted('linear candidate inventory truncated')
                for c in found: merge_choice(result, c)
        finally:
            self.counts[tid] = self.counter['notation']
        return result

    def _inventory(self, eligible):
        total = 0
        for tid in eligible:
            self.counter['notation'] = 0
            axes, cut = self.backend.orientations(tid, self.counter)
            self.counts[tid] = self.counter['notation']
            if cut: raise ResourceExhausted('orientation candidates truncated')
            self.orientations[tid] = axes
            candidates = self._search(tid, ())
            total += len(candidates)
            if total > self.policy.maximum_choices: raise ResourceExhausted('inventory choice limit')
            entry = {'source_line_id':tid, 'complete':True, 'candidate_checks':self.counts[tid], 'choices':[]}
            self.audit['inventory'].append(entry)
            self.choices[tid] = []
            for choice in candidates:
                check = self.verifier.verify(choice, self.supported, self.topology.report(), self.verification_policy)
                entry['choices'].append({'choice':deepcopy(choice), 'pixel_check':check})
                if check['valid']: self.choices[tid].append(choice)

    def _certificates(self, eligible):
        result = {}
        for tid in eligible:
            groups = {}
            for c in self.choices.get(tid, []): groups.setdefault(chain_key(c), []).append(c)
            ranked = []
            for key, choices in groups.items():
                origin, axis = line_axis(choices[0], self.heads)
                ranked.append((abs(cross(self.backend.texts[tid]['center']-origin, axis)), key, choices))
            ranked.sort(key=lambda x:(x[0],x[1]))
            if not ranked: continue
            distance, key, choices = ranked[0]
            reason = None
            if any(c['evidence'].get('text_placement')!='between_arrows' for c in choices):
                reason = 'external_domain_not_certifiable'
            elif len(ranked)>1 and distance*self.policy.distance_factor+self.backend.texts[tid]['tol']>=ranked[1][0]:
                reason = 'chain_distance_not_separated'
            if reason:
                self.audit['blocked_certificates'].append({'source_line_id':tid,'reason':reason});continue
            result[tid] = {'source_line_id':tid, 'key':key, 'choices':deepcopy(choices),
                           'own_text_to_line_px':distance, 'dimension_line_candidate_ids':list(key[0]),
                           'arrow_candidate_ids':list(key[1])}
        claims = {}
        for tid,c in result.items(): claims.setdefault(c['key'], []).append(tid)
        for tid in sorted({tid for ids in claims.values() if len(ids)>1 for tid in ids}):
            result.pop(tid)
            self.audit['blocked_certificates'].append({'source_line_id':tid,'reason':'shared_chain_claim'})
        return result

    def _peers(self, tid, certificates):
        own = certificates[tid];origin, axis = line_axis(own, self.heads);peers = []
        for oid, text in sorted(self.backend.texts.items()):
            if oid==tid: continue
            other = certificates.get(oid);separate = False
            if other:
                other_origin, other_axis = line_axis(other, self.heads)
                margin = max(self.backend.texts[tid]['tol'],text['tol'])
                separate = (not set(own['dimension_line_candidate_ids'])&set(other['dimension_line_candidate_ids'])
                    and not set(own['arrow_candidate_ids'])&set(other['arrow_candidate_ids'])
                    and abs(float(np.dot(axis,other_axis)))>=math.cos(math.radians(12))
                    and other['own_text_to_line_px']*self.policy.distance_factor+margin<abs(cross(text['center']-origin,axis))
                    and abs(cross(other_origin-origin,axis))>2*margin)
            self.audit['peer_decisions'].append({'target_text_id':tid,'other_text_id':oid,'waived':bool(separate)})
            if not separate: peers.append(text)
        return peers

    def resolve(self, baseline, support):
        result = deepcopy(baseline)
        result['algorithm_version'] = 'supported-chain-ownership/1.0'
        result['text_chain_ownership'] = self.audit
        for report in (baseline,support):
            if report.get('source_sha256')!=self.backend.digest or report.get('processed_size_px')!=self.backend.size:
                raise ValueError('Ownership frame mismatch')
        self.supported = {s['arrow_id'] for s in support['spans']}
        try:
            if not support['complete'] or len(self.backend.ocr['lines'])>self.policy.maximum_texts:
                raise ResourceExhausted('support or text inventory incomplete')
            eligible = self._eligible(baseline)
            self.topology = self.backend.topology() if eligible else None
            if self.topology and not self.topology.complete: raise ResourceExhausted('contour topology incomplete')
            self._inventory(eligible)
            certificates = self._certificates(eligible)
            self.audit['certificates'] = deepcopy(certificates)
            rows = {r['source_line_id']:r for r in result['associations']}
            for tid in eligible:
                if tid in rows and tid not in certificates and self.choices.get(tid):
                    demote(rows[tid], 'chain_ownership_not_uniquely_supported')
            for tid, cert in sorted(certificates.items()):
                if tid not in rows: continue
                if len(self.audit['revalidations'])>=self.policy.maximum_revalidations:
                    raise ResourceExhausted('revalidation count limit')
                found = [c for c in self._search(tid,self._peers(tid,certificates))
                         if chain_key(c)==cert['key'] and self.verifier.verify(c,self.supported,self.topology.report(),self.verification_policy)['valid']]
                self.audit['revalidations'].append({'source_line_id':tid,'candidate_count':len(found),
                                                   'total_text_checks':self.counts[tid]})
                if not found: continue
                row = rows[tid];old = [row] if row['status']=='proposed' else row['alternatives']
                for c in old: merge_choice(found,{k:deepcopy(c[k]) for k in FIELDS})
                row['ownership_original_record'] = deepcopy(row)
                for key in FIELDS[:-1]: row[key] = []
                row['alternatives'] = []
                apply_choices(row, found, result['issues'])
                row['evidence']['text_chain_ownership'] = {'certificate_text_id':tid,
                    'peer_waivers':[p for p in self.audit['peer_decisions'] if p['target_text_id']==tid and p['waived']]}
            resolve_pair_ownership(result['associations'])
            self.audit['complete'] = True
        except ResourceExhausted as exc:
            self.audit['certificates'] = {}
            self.audit['blocked_certificates'].append({'reason':str(exc)})
            for row in result['associations']: demote(row,'text_chain_ownership_search_incomplete')
        self.audit.update(search_checks=self.counter['page'],search_checks_by_text=dict(self.counts),
                          contour_topology=self.topology.report() if getattr(self,'topology',None) else None)
        result['complete'] = bool(baseline.get('complete')) and self.audit['complete']
        return result
