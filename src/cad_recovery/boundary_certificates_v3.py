"""Cache-independent logical proof budget, with bounded actual work and memo bytes.

Geometry and incidence retain the frozen v2 implementation below. Replaying a
successful certificate consumes the same logical charge sequence as proving it,
including the exact successful prefix when the remaining budget is insufficient.
"""
from array import array
from bisect import bisect_right
from collections import Counter,OrderedDict
from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np

from . import robust_topology_v1 as previous
from . import primitive_graph_qa_v1 as kernel
from .cad_validation_topology_v1 import loop_data

Unresolved=previous.Unresolved
numeric_envelope=previous.numeric_envelope
POLICY={**previous.POLICY,'version':'boundary-certificates/3.0',
    'maximum_certificate_entries':256,'maximum_cache_payload_bytes':8*1024**2,
    'index_leaf_segments':8,'lookup_reserve_units':20,'maximum_actual_predicates':250000,
    'budget_policy':'logical proof charges independent of cache history; actual work separately counted',
    'cache_authority':'complete exact geometry and policy fingerprint only',
    'roles':'unchanged baseline; no semantic ownership approval'}


@dataclass(frozen=True)
class Proof:
    state: object
    trace: bytes

    @property
    def checkpoints(self):
        return np.frombuffer(self.trace,dtype='<u4')

    @property
    def payload_bytes(self):
        # Trace is exact; 1024 bytes additionally reserve bounded entry/key metadata.
        return len(self.trace)+1024


class CertificateCache:
    def __init__(self):
        self._values=OrderedDict()
        self.payload_bytes=0
        self.peak_entries=0
        self.peak_payload_bytes=0

    def get(self,key):
        value=self._values.get(key)
        if value is not None:self._values.move_to_end(key)
        return value

    def put(self,key,state,checkpoints):
        points=np.asarray(checkpoints,dtype='<u4')
        if (not len(points) or len(points)>250000 or int(points[0])<1 or int(points[-1])>250000
                or np.any(points[1:]<=points[:-1])):
            raise ValueError('Invalid successful proof trace')
        value=Proof(state,points.tobytes())
        old=self._values.pop(key,None)
        if old is not None:self.payload_bytes-=old.payload_bytes
        self._values[key]=value;self.payload_bytes+=value.payload_bytes
        while (len(self._values)>256 or self.payload_bytes>POLICY['maximum_cache_payload_bytes']):
            _,removed=self._values.popitem(last=False);self.payload_bytes-=removed.payload_bytes
        self.peak_entries=max(self.peak_entries,len(self._values))
        self.peak_payload_bytes=max(self.peak_payload_bytes,self.payload_bytes)


class Budget(previous.Budget):
    def __init__(self,cache=None):
        super().__init__()
        if cache is not None and not isinstance(cache,CertificateCache):
            raise ValueError('Only this policy version can supply certificates')
        self.cache=cache if cache is not None else CertificateCache()
        self.stats=Counter()
        self.actual_used=0
        self.trace=array('I')

    def spend(self,count):
        super().spend(count)
        self.actual_used+=count
        if count:self.trace.append(self.used)
        assert self.actual_used<=self.used<=250000

    def charge(self,category,count=1):
        self.spend(count);self.stats[category]+=count

    def remember(self,key,state,mark,start):
        points=np.asarray(self.trace[mark:],dtype='<u4')-start
        self.cache.put(key,state,points)

    def replay(self,proof):
        # Lookup reserves 20 actual units: dictionary access, remaining-budget
        # comparison and at most 18 binary-search comparisons over 250000 entries.
        points=proof.checkpoints
        remaining=250000-self.used;total=int(points[-1])
        if total>remaining:
            index=bisect_right(points,remaining)
            prefix=int(points[index-1]) if index else 0
            self.used+=prefix;self.stats['replayed_logical_predicates']+=prefix
            self.stats['failed_cached_budget_replays']+=1
            raise Unresolved('topology_predicate_budget')
        self.used+=total;self.stats['replayed_logical_predicates']+=total
        assert self.actual_used<=self.used<=250000
        return proof.state


def fingerprint(flat,chord,numeric,circle,budget):
    budget.charge('fingerprinted_vertices',len(flat))
    raw=json.dumps([POLICY,chord,numeric,circle],sort_keys=True,allow_nan=False).encode()
    return hashlib.sha256(raw+np.asarray(flat,dtype='<f8').tobytes()).hexdigest()


class SegmentIndex:
    """Balanced AABB tree. Every visited node and tested segment consumes the shared cap."""
    def __init__(self, flat, budget):
        self.a = np.asarray(flat, dtype=float)
        self.b = np.roll(self.a,-1,axis=0)
        self.low, self.high = np.minimum(self.a,self.b), np.maximum(self.a,self.b)
        self.budget = budget
        self.root = self._build(np.arange(len(self.a)))

    def _build(self, ids):
        self.budget.charge('index_nodes_built')
        low, high = self.low[ids].min(axis=0), self.high[ids].max(axis=0)
        if len(ids) <= POLICY['index_leaf_segments']:
            return low, high, ids, None
        axis = int(np.argmax(high-low))
        ids = ids[np.argsort(self.low[ids,axis]+self.high[ids,axis],kind='stable')]
        middle = len(ids)//2
        return low, high, self._build(ids[:middle]), self._build(ids[middle:])

    def nearby(self, a, b, clearance):
        low, high = np.minimum(a,b), np.maximum(a,b)
        stack = [self.root]
        while stack:
            left,right,first,second = stack.pop()
            self.budget.charge('index_nodes_queried')
            if np.any(low > right+clearance) or np.any(left > high+clearance):
                self.budget.stats['index_subtrees_excluded'] += 1
                continue
            if second is None:
                for i in first:
                    self.budget.charge('segment_boxes_tested')
                    if not (np.any(low > self.high[i]+clearance) or np.any(self.low[i] > high+clearance)):
                        yield int(i)
            else:
                stack.extend((second,first))


def as_circle(walk, edges):
    """Recognize a complete positive analytic circle, not a fitted polygon."""
    arc = edges[walk[0][0]]['segment']
    if arc['kind'] != 'arc':
        return None
    sweep = 0.
    for j, sign in walk:
        edge = edges[j]; segment = edge['segment']
        if (segment['kind'] != 'arc' or sign != 1 or segment['radius'] != arc['radius']
                or not np.array_equal(segment['center'],arc['center'])):
            return None
        sweep += segment['sweep']*(edge['high']-edge['low'])
    if abs(sweep-2*math.pi) > 64*np.finfo(float).eps*2*math.pi:
        return None
    return [arc['center'].tolist(),float(arc['radius'])]


def certify_boundary(flat, chord, numeric, budget, circle=None):
    if len(flat) > 512:
        raise Unresolved('flattened_vertex_budget')
    key = fingerprint(flat,chord,numeric,circle,budget)
    budget.charge('certificate_lookups',POLICY['lookup_reserve_units'])
    proof=budget.cache.get(('simple',key))
    if proof is not None:
        budget.stats['simple_cache_hits'] += 1
        budget.replay(proof)
        return key
    mark,start=len(budget.trace),budget.used
    if circle is not None:
        budget.charge('analytic_circle_simple')
    else:
        tree = SegmentIndex(flat,budget)
        for i,(a,b) in enumerate(zip(tree.a,tree.b)):
            ids = [j for j in tree.nearby(a,b,2*chord+numeric)
                   if j > i+1 and not (i == 0 and j == len(flat)-1)]
            if not ids:
                continue
            budget.charge('exact_segment_distances',len(ids))
            if np.any(previous.segment_distances(a,b,tree.a[ids],tree.b[ids]) <= 2*chord+numeric):
                raise Unresolved('self_intersection_or_unresolved_curve_clearance')
    budget.remember(('simple',key),True,mark,start)
    return key


def relation(left, right, budget):
    a,b=np.asarray(left['flat']),np.asarray(right['flat'])
    envelope=left['chord_error_bound_px']+right['chord_error_bound_px']+max(
        left['numeric_envelope_native_px'],right['numeric_envelope_native_px'])
    budget.charge('boundary_boxes_tested')
    if np.any(a.min(axis=0)>b.max(axis=0)+envelope) or np.any(b.min(axis=0)>a.max(axis=0)+envelope):
        return 'disjoint'
    # Vertex incidence must be checked even on a cache hit in a different graph.
    if set(left['vertices']) & set(right['vertices']):
        raise Unresolved('boundary_contact')
    keys=(left['_certificate_key'],right['_certificate_key'])
    budget.charge('certificate_lookups',POLICY['lookup_reserve_units'])
    proof=budget.cache.get(('relation',*keys))
    if proof is not None:
        budget.stats['relation_cache_hits']+=1
        return budget.replay(proof)
    mark,start=len(budget.trace),budget.used
    if left['_circle'] is not None and right['_circle'] is not None:
        budget.charge('analytic_circle_relations')
        ac,ar=left['_circle'];bc,br=right['_circle']
        d=math.dist(ac,bc)
        numeric=max(left['numeric_envelope_native_px'],right['numeric_envelope_native_px'])
        if d > ar+br+numeric:
            state='disjoint'
        elif d+ar < br-numeric:
            state='left_inside_right'
        elif d+br < ar-numeric:
            state='right_inside_left'
        else:
            raise Unresolved('analytic_circle_contact_or_crossing')
    else:
        tree=SegmentIndex(b,budget)
        for p,q in zip(a,np.roll(a,-1,axis=0)):
            ids=list(tree.nearby(p,q,envelope))
            if not ids:
                continue
            budget.charge('exact_segment_distances',len(ids))
            if np.any(previous.segment_distances(p,q,tree.a[ids],tree.b[ids]) <= envelope):
                raise Unresolved('boundary_contact_or_unresolved_curve_clearance')
        left_inside=previous.inside(a[0],b,budget)
        right_inside=previous.inside(b[0],a,budget)
        if left_inside and right_inside:
            raise Unresolved('inconsistent_containment')
        state='left_inside_right' if left_inside else 'right_inside_left' if right_inside else 'disjoint'
    budget.remember(('relation',*keys),state,mark,start)
    return state


def face_boundaries(nodes, edges, adjacency, budget):
    def tangent(half):
        index, sign = half
        edge = edges[index]
        segment = edge['segment']
        t = edge['low'] if sign == 1 else edge['high']
        if segment['kind'] == 'line':
            d = sign*(segment['b']-segment['a'])
        else:
            angle = segment['start']+t*segment['sweep']
            d = sign*np.array([-math.sin(angle), math.cos(angle)])
        return math.atan2(d[1], d[0])

    ordered = [sorted(a, key=tangent) for a in adjacency]
    positions = [{h: i for i, h in enumerate(a)} for a in ordered]
    seen, output, unresolved = set(), [], []
    for index in range(len(edges)):
        for sign in (1, -1):
            first = (index, sign)
            if first in seen:
                continue
            walk, current = [], first
            while current not in seen:
                seen.add(current)
                walk.append(current)
                j, direction = current
                edge = edges[j]
                end = edge['b'] if direction == 1 else edge['a']
                order = ordered[end]
                current = order[(positions[end][(j, -direction)]-1) % len(order)]
            if current != first:
                raise kernel.Incomplete('face_walk_inconsistent')
            reduced = []
            for half in walk:
                if reduced and reduced[-1] == (half[0], -half[1]):
                    reduced.pop()
                else:
                    reduced.append(half)
            while len(reduced) > 1 and reduced[0] == (reduced[-1][0], -reduced[-1][1]):
                reduced = reduced[1:-1]
            if not reduced:
                continue
            vertices, pieces, owners = [], [], set()
            for j, direction in reduced:
                edge = edges[j]
                segment = edge['segment']
                low, high = (edge['low'], edge['high']) if direction == 1 else (edge['high'], edge['low'])
                a, b = kernel.point(segment, low), kernel.point(segment, high)
                bulge = math.tan(segment['sweep']*(high-low)/4) if segment['kind'] == 'arc' else 0.
                pieces.append((a, b, bulge))
                vertices.append(edge['a'] if direction == 1 else edge['b'])
                owners.update(edge['owners'])
            if len(set(vertices)) != len(vertices):
                unresolved.append('non_simple_boundary_walk')
                continue
            origin = pieces[0][0]
            local = [(tuple(a-origin), tuple(b-origin), bulge) for a, b, bulge in pieces]
            try:
                loop = loop_data(local, 1e-7)
                flat = np.asarray(loop['flat'], float)+origin
                numeric = numeric_envelope(flat)
                # Area is computed locally, avoiding subtraction of products of global coordinates.
                if loop['signed_area'] <= 1e-7:
                    continue
                if len(flat) > 512:
                    raise Unresolved('flattened_vertex_budget')
                circle = as_circle(reduced, edges)
                if circle is not None and any(len(edges[j]['primitives']) > 1 for j, _ in reduced):
                    raise Unresolved('coincident_analytic_boundaries')
                certificate_key = certify_boundary(flat, loop['chord_error_bound'], numeric, budget, circle)
            except ValueError as exc:
                unresolved.append(str(exc))
                continue
            if len(output) >= kernel.POLICY['maximum_faces']:
                raise kernel.Incomplete('face_budget')
            output.append({'id': f'face-{len(output):05d}', 'area_px2': loop['signed_area'],
                'owners': sorted(owners), 'vertices': vertices, 'flat': flat.tolist(),
                'chord_error_bound_px': loop['chord_error_bound'], 'numeric_envelope_native_px': numeric,
                'simple': True, 'boundary_certified': True, 'parent': None, 'depth': None,
                '_circle': circle, '_certificate_key': certificate_key})
    return output, unresolved


def nesting(loops, budget):
    uncertain, failures, parents = set(), [], {f['id']: [] for f in loops}
    checks = 0
    for i, left in enumerate(loops):
        for right in loops[i+1:]:
            checks += 1
            if checks > 100000:
                raise kernel.Incomplete('nesting_budget')
            try:
                state = relation(left, right, budget)
            except Unresolved as exc:
                uncertain.update((left['id'], right['id']))
                failures.append(str(exc))
                continue
            if state == 'left_inside_right':
                parents[left['id']].append(right)
            elif state == 'right_inside_left':
                parents[right['id']].append(left)
    for loop in loops:
        if loop['id'] not in uncertain and parents[loop['id']]:
            loop['parent'] = min(parents[loop['id']], key=lambda f: f['area_px2'])['id']
    lookup = {f['id']: f for f in loops}
    for loop in loops:
        parent, visited = loop['parent'], set()
        if loop['id'] in uncertain:
            continue
        while parent:
            if parent in visited:
                raise kernel.Incomplete('cyclic_nesting')
            visited.add(parent)
            parent = lookup[parent]['parent']
        loop['depth'] = len(visited)
    return checks, sorted(uncertain), failures


def build(candidates, *, cache=None):
    """Reuse v1 analytic intersections/incidence; only boundary QA is versioned."""
    try:
        parts, isolated, definitions = kernel.segments(candidates)
        checks, entries = kernel.split_intersections(parts)
        nodes, edges, isolated_owners, duplicates = kernel.assemble(parts, isolated)
        adj = kernel.adjacency(nodes, edges)
        component, count = [-1]*len(nodes), 0
        for first in range(len(nodes)):
            if component[first] >= 0:
                continue
            stack, component[first] = [first], count
            while stack:
                v = stack.pop()
                for j, _ in adj[v]:
                    edge = edges[j]
                    w = edge['b'] if edge['a'] == v else edge['a']
                    if component[w] < 0:
                        component[w] = count
                        stack.append(w)
            count += 1
        budget = Budget(cache)
        loops, unresolved = face_boundaries(nodes, edges, adj, budget)
        nest_checks, uncertain, nesting_failures = nesting(loops, budget)
        structural, component_owners = [], [set() for _ in range(count)]
        for i, p in enumerate(nodes):
            owners = set(isolated_owners[i])
            for j, _ in adj[i]:
                owners.update(edges[j]['owners'])
            component_owners[component[i]].update(owners)
            if len(adj[i]) != 2:
                structural.append({'id': f'node-{i:06d}', 'point_px': p.tolist(), 'degree': len(adj[i]),
                                   'owners': sorted(owners), 'component': component[i]})
        return {'complete': True, 'NoP': len(candidates), 'NoV_definition': definitions,
            'graph_vertices': len(nodes), 'graph_edges': len(edges), 'components': count,
            'cycle_rank': len(edges)-len(nodes)+count, 'degree_histogram': dict(sorted(Counter(map(len, adj)).items())),
            'structural_nodes': structural, 'component_owners': sorted(sorted(o) for o in component_owners),
            'faces': [{k: v for k, v in f.items() if k not in ('flat', 'vertices', '_circle', '_certificate_key')} for f in loops],
            'bounded_faces': len(loops), 'nesting_complete': not unresolved and not uncertain,
            'unresolved_face_reasons': dict(Counter(unresolved)), 'uncertain_nesting_ids': uncertain,
            'nesting_failure_reasons': dict(Counter(nesting_failures)), 'topology_predicates': budget.used,
            'certificate_stats': {**budget.stats,'actual_predicates':budget.actual_used,
                'logical_predicates':budget.used,'peak_cache_payload_bytes':budget.cache.peak_payload_bytes}, 'certificate_cache_peak_entries': budget.cache.peak_entries,
            'duplicate_geometric_pieces': duplicates, 'pair_checks': checks, 'grid_entries': entries,
            'nesting_checks': nest_checks, 'geometric_crossings_are_semantic_junctions': False,
            'endpoint_snapping_performed': False, 'physical_geometry_approved': False}
    except kernel.Incomplete as exc:
        return {'complete': False, 'reason': str(exc), 'physical_geometry_approved': False}
