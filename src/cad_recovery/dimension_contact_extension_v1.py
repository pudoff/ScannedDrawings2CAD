"""Finite train-only extension; source selection precedes raster/label reads."""
import numpy as np

from .atomic_source_graph_v1 import build
from .object_supervision_protocol_v1 import endpoint_graph, pair_channels, role_channels
from .role_interval_dataset_v1 import observe
from .role_relation_preparation_v1 import build_graph
from .role_relation_release_v1 import assemble_patch

POLICY = dict(maximum_sources=4, maximum_patches=16, maximum_crop_side=1024,
              source_selection='ascending source SHA/path; first source of each new conservative group',
              crop_selection='central rectangle <=2048x2048, four disjoint quadrants; integer source pixels',
              maximum_source_pixels=64_000_000, source_QA_distance_px=3, source_QA_minimum_fraction=.95,
              review='all fully visible straight DIM and EXT primitives and their explicit contacts in every selected crop',
              clipped_or_ambiguous_primitives='unknown; never fabricate or extend beyond crop',
              contact_unit='one reviewed physical DIM/EXT contact; repeated atoms/pairs do not add units',
              new_primary_held_targets=False, automatic_expansion=False, models_fitted=0)


def select_sources(inventory, decisions, old_groups):
    admitted = []
    rejected = []
    for row in inventory:
        if row.get('declared_split') != 'train' or row.get('role') != 'input' or not row['path'].casefold().endswith('.png'):
            continue
        decision = decisions[row['sha256']]
        reasons = list(decision['reasons'])
        if decision['blocked'] and not reasons:
            reasons.append('blocked')
        if decision['development_related_family']:
            reasons.append('development_related')
        if decision['canonical_family'] in old_groups:
            reasons.append('existing_six_train_groups')
        source = dict(path='dataset/' + row['path'], sha256=row['sha256'])
        item = dict(source=source, family=decision['canonical_family'])
        if reasons:
            rejected.append(dict(**item, reasons=sorted(set(reasons))))
        else:
            admitted.append(item)
    admitted.sort(key=lambda x: (x['source']['sha256'], x['source']['path']))
    chosen = []
    seen = set()
    for row in admitted:
        if row['family'] in seen:
            continue
        chosen.append(row)
        seen.add(row['family'])
        if len(chosen) == POLICY['maximum_sources']:
            break
    return chosen, admitted, rejected


def fixed_crops(size):
    w, h = size
    if min(w, h) < 2 or w * h > POLICY['maximum_source_pixels']:
        raise ValueError('Source header outside frozen bounds; no replacement source')
    cw, ch = min(w, 2 * POLICY['maximum_crop_side']), min(h, 2 * POLICY['maximum_crop_side'])
    left, top = (w - cw) // 2, (h - ch) // 2
    xs, ys = [left, left + cw // 2, left + cw], [top, top + ch // 2, top + ch]
    return [[xs[x], ys[y], xs[x + 1], ys[y + 1]] for y in range(2) for x in range(2)]


def source_graph(gray, transform, size):
    rows, audit = observe(gray, patch_to_source=transform, source_size=size)
    if not audit['complete']:
        raise ValueError('Incomplete interval trace')
    features = np.asarray([r['features'] for r in rows], np.float32).reshape(-1, 21)
    nodes = [{k: r[k] for k in ('id', 'points_px')} for r in rows]
    if rows:
        base = build_graph(features, [r['points_px'] for r in rows], [r['id'] for r in rows])
        base = endpoint_graph(features, [r['points_px'] for r in rows], base)
    else:
        base = dict(node_ids=[], directed_edges=[])
    graph, result = build(gray, nodes, features, base, patch_to_source=transform, source_size=size)
    return graph, result.reshape(-1, 21)


def attach_review(reading, gray, graph):
    if any(o['role_id'] not in (2, 3) or o['primitive'] != 'LINE' for o in reading['objects']):
        raise ValueError('Only source-reviewed straight DIM/EXT objects are admitted')
    contact_ids = [e['id'] for e in reading['contacts']]
    if len(set(contact_ids)) != len(contact_ids):
        raise ValueError('Duplicate physical contact ID')
    by_id = {o['id']: o for o in reading['objects']}
    contacts = set()
    for event in reading['contacts']:
        if event['relation'] != 'dimension_extension':
            raise ValueError('New negative contacts are outside this finite review')
        if tuple(event['objects']) in contacts:
            raise ValueError('Repeated DIM/EXT physical contact')
        contacts.add(tuple(event['objects']))
        if len(event['objects']) != 2 or any(o not in by_id for o in event['objects']):
            raise ValueError('Contact references absent objects')
    pixels = np.full(len(graph['node_ids']), 255, np.uint8)
    rows = [dict(points_px=r['points_px'], target_reason='clipped_source_path' if r['source_path_clipped'] else 'source_observation')
            for r in graph['nodes']]
    result = assemble_patch(reading, gray, graph, rows, pixels,
                            dict(distance_px=POLICY['source_QA_distance_px'], minimum_fraction=POLICY['source_QA_minimum_fraction']))
    fit, provenance = role_channels(pixels, result['source_objects'], graph['node_ids'])
    pairs = pair_channels(result, graph, [r['points_px'] for r in rows])
    pairs['legacy_evaluation_pairs'] = []
    result.update(pairs=pairs, role_fit_provenance=provenance,
                  new_primary_evaluation_labels=False, review_scope='straight DIM/EXT only; other roles unknown')
    return result, fit
