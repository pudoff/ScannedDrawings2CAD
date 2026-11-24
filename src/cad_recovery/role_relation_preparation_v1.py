"""Source-only interval graphs; separately reviewed object/relation targets.

An observed path is an input relation, never a CAD object identity. Unknown
targets and unreviewed pairs are not negatives. This module performs no fit.
"""
from collections import defaultdict
import math
import re

import numpy as np

from .role_interval_dataset_v1 import FEATURES

POLICY = dict(maximum_nodes_per_patch=4096, maximum_directed_edges_per_node=64,
    local_neighbours=8, local_radius_in_widths=64, maximum_local_radius_px=256,
    junction_distance_px=1.5, orientation_tolerance_deg=10,
    edge_types=['source_path_continuation', 'source_junction', 'near_parallel',
                'near_perpendicular', 'near_other'],
    node_features=list(FEATURES), labels_in_graph=False,
    source_path_is_whole_CAD_object=False, unknown_target=255)
RELATIONS = ('same_primitive', 'dimension_extension', 'reviewed_unrelated')
PRIMITIVES = ('LINE', 'ARC', 'CIRCLE', 'CENTER_LINE')


def select_sources(rows, decisions):
    """Every row is screened before any graph or target array is consumed."""
    kept, rejected = [], []
    for index, row in enumerate(rows):
        d = decisions[row['source']['sha256']]
        reasons = list(d['reasons'])
        if row['declared_split'] != 'train': reasons.append('not_original_train')
        if d['development_related_family']: reasons.append('development_related_family_no_new_fit')
        if d['blocked'] and not reasons: reasons.append('blocked_source')
        if row['canonical_family'] != d['canonical_family']: reasons.append('family_identity_changed')
        if reasons: rejected.append(dict(index=index, reasons=sorted(set(reasons))))
        else: kept.append(index)
    return kept, rejected


def build_graph(features, points, local_ids):
    """Only geometry/21 source features enter this interface; no target argument."""
    x = np.asarray(features, np.float64); n = len(x)
    if (x.shape != (n, len(FEATURES)) or not np.isfinite(x).all()
            or not 0 < n <= POLICY['maximum_nodes_per_patch'] or len(points) != n
            or len(local_ids) != n or len(set(local_ids)) != n):
        raise ValueError('Invalid source graph observations')
    paths = defaultdict(list); clouds = []; centers = []; widths = []
    for index, (p, identity) in enumerate(zip(points, local_ids, strict=True)):
        p = np.asarray(p, float)
        m = re.fullmatch(r'(path-\d+)/(\d+)-(\d+)', identity)
        if (m is None or int(m[2]) >= int(m[3]) or p.ndim != 2 or p.shape[1] != 2
                or len(p) < 2 or not np.isfinite(p).all() or not np.array_equal(p, np.round(p))):
            raise ValueError('Invalid lattice interval identity or coordinates')
        length = np.linalg.norm(np.diff(p, axis=0), axis=1).sum()
        ratio = np.expm1(x[index, 0])
        if ratio <= 0 or not math.isfinite(ratio): raise ValueError('Invalid measured stroke scale')
        widths.append(max(1., float(length/ratio)))
        paths[m[1]].append((int(m[2]), int(m[3]), index))
        clouds.append(p); centers.append(p.mean(0))
    centers = np.asarray(centers); widths = np.asarray(widths)
    edges = set()
    def connect(a, b, kind):
        if a != b: edges.update(((a, b, kind), (b, a, kind)))
    for intervals in paths.values():
        ordered = sorted(intervals)
        for a, b in zip(ordered, ordered[1:]):
            if a[1] == b[0]: connect(a[2], b[2], 0)
    # Lattice endpoints, rather than endpoint bbox overlap, identify junctions.
    ends = defaultdict(set)
    for i, p in enumerate(clouds):
        for point in (p[0], p[-1]): ends[tuple(point.astype(int))].add(i)
    for point, nodes in ends.items():
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if math.hypot(dx, dy) > POLICY['junction_distance_px']: continue
                for a in nodes:
                    for b in ends.get((point[0]+dx, point[1]+dy), ()):
                        connect(a, b, 1)
    threshold = math.cos(math.radians(2*POLICY['orientation_tolerance_deg']))
    for i in range(n):
        distance = np.linalg.norm(centers-centers[i], axis=1)
        radius = min(POLICY['maximum_local_radius_px'], POLICY['local_radius_in_widths']*widths[i])
        indices = np.flatnonzero((distance <= radius) & (np.arange(n) != i))
        nearest = indices[np.lexsort((indices, distance[indices]))[:POLICY['local_neighbours']]]
        for j in nearest:
            cosine = float(x[i, 3:5] @ x[j, 3:5])
            kind = 2 if cosine >= threshold else 3 if cosine <= -threshold else 4
            connect(i, int(j), kind)
    if len(edges) > POLICY['maximum_directed_edges_per_node']*n:
        raise ValueError('Source graph edge budget exceeded; do not silently truncate')
    ordered = sorted(edges)
    # Distances/directions are source-only edge features, not semantic labels.
    attributes = []
    for a, b, kind in ordered:
        delta = (centers[b]-centers[a])/max(1., math.sqrt(widths[a]*widths[b]))
        attributes.append([float(delta[0]), float(delta[1]), float(np.linalg.norm(delta)),
                           float(x[a, 3:5] @ x[b, 3:5]), float(math.log(widths[b]/widths[a]))])
    return dict(node_ids=list(local_ids), node_count=n, directed_edges=[list(e) for e in ordered],
                edge_features=attributes, edge_types=POLICY['edge_types'],
                graph_from_source_only=True, same_primitive_targets_inferred=False)


def validate_review(review, node_ids):
    """Validate supplied supervision without inventing annotations for absent data."""
    if review.get('authority') != 'assistant_source_review' or review.get('human_verified') is not False:
        raise ValueError('Explicit review authority required')
    nodes = set(node_ids); objects = {}; membership = {}
    for obj in review['objects']:
        if (obj['id'] in objects or obj['primitive'] not in PRIMITIVES
                or obj.get('complete_visible_primitive') is not True
                or obj.get('physical_geometry_approved') is not False
                or not obj['nodes'] or len(set(obj['nodes'])) != len(obj['nodes'])
                or not set(obj['nodes']) <= nodes):
            raise ValueError('Invalid whole primitive supervision')
        objects[obj['id']] = obj
        for node in obj['nodes']:
            if node in membership: raise ValueError('One interval cannot be two whole primitives')
            membership[node] = obj['id']
    pairs = {}; counts = {r: 0 for r in RELATIONS}
    for row in review['relations']:
        a, b = row['nodes']; relation = row['relation']; key = tuple(sorted((a, b)))
        if (a == b or a not in nodes or b not in nodes or key in pairs
                or relation not in RELATIONS or row.get('source_evidence_reviewed') is not True):
            raise ValueError('Invalid or conflicting reviewed relation')
        if relation == 'same_primitive' and (a not in membership or membership.get(a) != membership.get(b)):
            raise ValueError('Same-primitive relation needs one reviewed complete primitive')
        if relation == 'reviewed_unrelated' and a in membership and membership.get(a) == membership.get(b):
            raise ValueError('Unrelated relation contradicts reviewed object membership')
        if relation == 'dimension_extension' and (row.get('dimension_endpoint') not in (a, b)
                or row.get('extension_endpoint') not in (a, b)
                or row['dimension_endpoint'] == row['extension_endpoint']):
            raise ValueError('Dimension/extension directions must be explicit')
        pairs[key] = relation; counts[relation] += 1
    return dict(objects=len(objects), explicitly_reviewed_relations=counts,
                absent_pairs_are_negative=False, coordinate_regression_admitted=False)


def data_gate(supported_roles, reviews):
    reasons = []
    if list(supported_roles) != list(range(1, 10)): reasons.append('nine_role_family_coverage_not_ready')
    if not reviews or not all(r.get('status') == 'complete' for r in reviews):
        reasons.append('finite_object_relation_review_not_complete')
    for kind in RELATIONS:
        families = {r['family'] for r in reviews if r['counts']['explicitly_reviewed_relations'][kind]}
        by_family = {f: sum(r['counts']['explicitly_reviewed_relations'][kind]
                           for r in reviews if r['family'] == f) for f in families}
        valid = [f for f in families if len(families-{f}) >= 2
                 and sum(n for other, n in by_family.items() if other != f) >= 16]
        if len(families) < 3: reasons.append(kind+'_fewer_than_three_train_groups')
        if len(valid) < 2 or sum(by_family[f] for f in valid) < 8:
            reasons.append(kind+'_insufficient_family_held_pairs')
    return dict(fit_authorized=not reasons, reasons=reasons, thresholds_lowered=False)


def load_for_fit(root, manifest_ref):
    """A packet of source graphs is not permission to train on missing targets."""
    from pathlib import Path
    from .dataset_layout_v1 import checked, read
    root = Path(root)
    manifest = read(checked(root, manifest_ref))
    if manifest.get('schema_version') != 'role-relation-preparation-release/1.0':
        raise ValueError('Unsupported relation preparation release')
    files = {ref['path']: ref for ref in manifest['files']}
    report_refs = [r for p,r in files.items() if p.endswith('/report.json')]
    if len(report_refs) != 1: raise ValueError('Missing unique packet report')
    report = read(checked(root, report_refs[0]))
    if not report['readiness']['fit_authorized']:
        raise ValueError('Source graphs alone do not authorize fit: '+', '.join(report['readiness']['reasons']))
    # Admission of reviewed targets must be implemented as its own release;
    # changing a status flag in this preparation report cannot enable training.
    raise ValueError('Use a separately admitted object/relation supervision release')
