"""Numerical source packet, with separate FIT truth and immutable held channels."""
from collections import defaultdict
import hashlib

import numpy as np

from .dataset_layout_v1 import checked, encoded, pin, read
from .typed_graph_solver_v1 import balanced_weights, normalizer, standardize

KINDS = ('same_primitive','dimension_extension','reviewed_unrelated')


def unit(value, prefix):
    return prefix + hashlib.sha256(encoded(value)).hexdigest()


def primitive_unit(obj, source, offset):
    g = obj['geometry_px']
    shift = np.asarray(offset)
    if obj['primitive'] in ('LINE','CENTER_LINE'):
        geometry = sorted([(np.asarray(g[k])+shift).tolist() for k in ('start','end')])
    else:
        geometry = dict(g,center=(np.asarray(g['center'])+shift).tolist())
    return unit([source,obj['primitive'],geometry],'primitive:')


def pair_geometry(packet, pairs):
    pairs = np.asarray(pairs,np.int64).reshape(-1,2)
    if len(pairs) and ((pairs < 0).any() or (pairs >= len(packet['features'])).any() or (pairs[:,0] == pairs[:,1]).any()):
        raise ValueError('Bounded distinct source-node pairs required')
    a,b = np.sort(pairs,axis=1).T
    keys = a*len(packet['features'])+b
    positions = np.searchsorted(packet['candidate_keys'],keys)
    if len(positions) and ((positions >= len(packet['candidate_keys'])).any() or not np.array_equal(packet['candidate_keys'][positions],keys)):
        raise ValueError('Relation pair absent from frozen source candidates')
    width = np.sqrt(packet['widths'][a]*packet['widths'][b])
    delta = packet['centres'][a]-packet['centres'][b]
    endpoints = packet['endpoints']
    nearest = np.linalg.norm(endpoints[a,:,None,:]-endpoints[b,None,:,:],axis=-1).min(axis=(1,2))
    descriptors = [np.log1p(np.linalg.norm(delta,axis=1)/width),np.log1p(nearest/width),
                   np.log1p(np.abs(delta[:,0])/width),np.log1p(np.abs(delta[:,1])/width),
                   np.sum(packet['features'][a,3:5]*packet['features'][b,3:5],axis=1),
                   np.abs(np.log(packet['lengths'][a]/packet['lengths'][b])),
                   np.abs(np.log(packet['widths'][a]/packet['widths'][b])),
                   (packet['paths'][a] == packet['paths'][b]).astype(float)]
    masks = packet['candidate_types'][positions]
    return np.array(descriptors + [((masks >> k)&1).astype(float) for k in range(8)],np.float32).T


def prepare(root, output):
    """No PNG decoding, OCR, DXF parsing or extraction tuning at this stage."""
    data = root/'artifacts/dimension-contact-extension-v1'
    old = root/'artifacts/atomic-source-graph-v1'
    features = np.load(data/'features.npy',allow_pickle=False)
    primary = np.load(data/'primary-role-targets.npy',allow_pickle=False)
    fit = np.load(data/'fit-role-targets.npy',allow_pickle=False)
    metadata = read(data/'node-metadata.json')
    n = len(features)
    centres, endpoints = np.zeros((n,2),np.float32),np.zeros((n,2,2),np.float32)
    lengths,widths = np.zeros(n,np.float32),np.zeros(n,np.float32)
    paths = np.zeros(n,np.int32)
    owner_units = np.full(n,'',dtype='U80')
    family = np.array([r['canonical_family'] for r in metadata])
    source_refs = []
    edges, patches, fit_pairs, held_pairs, objects, events = [],[],[],[],[],[]
    source_paths = {}
    edge_owner = {}
    parent = []
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a,b):
        a,b = find(a),find(b)
        if a != b:
            parent[max(a,b)] = min(a,b)
    cases = [(old,c) for c in read(root/'configs/atomic-source-graph-v1.json')['cases']]
    cases += [(data,c) for c in read(root/'configs/dimension-contact-extension-v1.json')['cases']]
    coverage = np.zeros(n,int)
    for folder,case in cases:
        name = case['patch_id']
        sr = pin(root,(folder/name/'supervision.json').relative_to(root).as_posix())
        s = read(checked(root,sr))
        graph = read(checked(root,s['graph']))
        source_refs.extend([sr,s['graph']])
        idx = np.asarray(s['global_indices'],np.int32)
        coverage[idx] += 1
        assert all(family[idx] == case['family'])
        offset = np.array(case['patch_to_source'])[:2,2].astype(int)
        ids = {identity:int(i) for identity,i in zip(graph['node_ids'],idx,strict=True)}
        local_paths = {}
        for i,node in zip(idx,graph['nodes'],strict=True):
            p = np.asarray(node['points_px'],np.float32)
            centres[i], endpoints[i] = p.mean(0),p[[0,-1]]
            lengths[i] = np.linalg.norm(np.diff(p,axis=0),axis=1).sum()
            widths[i] = max(1.,float(lengths[i]/np.expm1(features[i,0])))
            key = (case['source']['sha256'],name,node['path_index'])
            if key not in source_paths:
                source_paths[key] = len(parent)
                parent.append(len(parent))
            path_id = source_paths[key]
            paths[i] = path_id
            local_paths[node['path_index']] = path_id
            if node['legacy']:
                continue
            q = p.astype(int)+offset
            for a,b in zip(q,q[1:]):
                e = (case['source']['sha256'], *sorted((tuple(a),tuple(b))))
                previous = edge_owner.setdefault(e,path_id)
                union(path_id,previous)
        e = np.asarray(graph['directed_edges'],np.int32).reshape(-1,3)
        if len(e):
            mapped = np.c_[idx[e[:,:2]],e[:,2]]
            edges.append(mapped)
        obj_units = {}
        for obj in s['source_objects']:
            key = primitive_unit(obj,case['source']['sha256'],offset)
            obj_units[obj['id']] = key
            nodes = [ids[k] for k in obj['nodes']]
            for i in nodes:
                if fit[i] == obj['role_id']:
                    if owner_units[i] and owner_units[i] != key:
                        raise ValueError('One FIT node has conflicting source owners')
                    owner_units[i] = key
            objects.append(dict(patch_id=name,family=case['family'],unit=key,role=obj['role_id'],nodes=nodes,
                                source_QA_pass=obj['source_QA']['source_QA_pass'],candidate_available=bool(nodes)))
        contacts = {e['id']:e for e in s['source_contacts']}
        source_events = {e['id']:e for e in s['pairs']['source_events']}
        def pair_unit(row):
            identity = row.get('unit','')
            if row['relation'] == 'same_primitive':
                return obj_units[row.get('object_id',identity.removeprefix('primitive:'))]
            contact = row.get('physical_contact_id',identity.removeprefix('contact:') if identity.startswith('contact:') else None)
            if contact is not None:
                event = contacts[contact]
                return unit([case['source']['sha256'],(np.asarray(event['point_px'])+offset).tolist(),
                             sorted(obj_units[o] for o in event['objects']),event['relation']],'contact:')
            names = row.get('source_object_ids',identity.removeprefix('object-pair:').split('|'))
            return unit(sorted(obj_units[o] for o in names),'object-pair:')
        def convert(row):
            return dict(nodes=[ids[k] for k in row['nodes']],target=KINDS.index(row['relation']),
                        unit=pair_unit(row),family=case['family'],patch_id=name)
        fit_pairs.extend(convert(row) for row in s['pairs']['fit_pairs'])
        held_pairs.extend(convert(row) for row in s['pairs']['legacy_evaluation_pairs'])
        for event in s['source_contacts']:
            selected = source_events[event['id']]['selected_nodes']
            events.append(dict(patch_id=name,family=case['family'],id=event['id'],target=KINDS.index(event['relation']),
                               nodes=[ids[k] for k in selected] if selected else None,missing_counts_as_FN=selected is None))
        patches.append(dict(patch_id=name,family=case['family'],source=case['source'],indices=idx.tolist()))
    assert np.all(coverage == 1) and np.all(lengths > 0)
    canonical_paths = np.array([find(i) for i in paths],np.int32)
    node_units = owner_units.copy()
    path_keys = list(source_paths)
    path_labels = {p:unit([path_keys[p]],'path:') for p in set(canonical_paths)}
    for i in np.flatnonzero(node_units == ''):
        node_units[i] = path_labels[canonical_paths[i]]
    all_edges = np.concatenate(edges).astype(np.int32)
    assert np.all(family[all_edges[:,0]] == family[all_edges[:,1]])
    a,b = np.sort(all_edges[:,:2],axis=1).T
    keys, inverse = np.unique(a.astype(np.int64)*n+b,return_inverse=True)
    masks = np.zeros(len(keys),np.uint16)
    np.bitwise_or.at(masks,inverse,(1 << all_edges[:,2]).astype(np.uint16))
    arrays = dict(features=features,primary=primary,fit_roles=fit,families=family,node_units=node_units,
                  centres=centres,endpoints=endpoints,lengths=lengths,widths=widths,paths=canonical_paths,
                  edges=all_edges,candidate_keys=keys,candidate_types=masks)
    output.mkdir()
    files = []
    for name,values in arrays.items():
        with (output/(name+'.npy')).open('xb') as stream:
            np.save(stream,values,allow_pickle=False)
        files.append(pin(root,(output/(name+'.npy')).relative_to(root).as_posix()))
    support = {f['held_family']:f['roles'] for f in read(data/'role-folds.json')}
    relation_support = {f['held_family']:f['support'] for f in read(data/'pair-folds.json')}
    documents = dict(patches=patches,fit_pairs=fit_pairs,held_pairs=held_pairs,source_objects=objects,
                     source_events=events,role_support=support,relation_support=relation_support)
    for name,value in documents.items():
        with (output/(name+'.json')).open('xb') as stream:
            stream.write(encoded(value))
        files.append(pin(root,(output/(name+'.json')).relative_to(root).as_posix()))
    result = dict(nodes=n,features=21,patches=len(patches),groups=len(set(family)),directed_edges=len(all_edges),
                  primary_roles=int(np.count_nonzero(primary != 255)),primary_pairs=len(held_pairs),
                  source_objects=len(objects),source_events=len(events),source_events_missing=sum(e['nodes'] is None for e in events),
                  canonical_reviewed_primitive_units=len({o['unit'] for o in objects}),
                  node_units_with_fit_roles=len(set(node_units[fit != 255])),source_inputs=source_refs,files=files)
    with (output/'manifest.json').open('xb') as stream:
        stream.write(encoded(result))
    return result


def load(root, manifest_ref):
    manifest = read(checked(root,manifest_ref))
    for ref in manifest['files']:
        checked(root,ref)
    directory = checked(root,manifest_ref).parent
    arrays = {p.stem:np.load(p,allow_pickle=False) for p in directory.glob('*.npy')}
    arrays.update({p.stem:read(p) for p in directory.glob('*.json') if p.name != 'manifest.json'})
    return arrays


def fold(packet, held):
    """All FIT quantities depend only on non-held source rows and target slices."""
    families = packet['families']
    if held not in set(families):
        raise ValueError('Unregistered conservative holdout group')
    nodes = np.flatnonzero(families != held)
    remap = np.full(len(families),-1,np.int32)
    remap[nodes] = np.arange(len(nodes))
    role_indices = nodes[packet['fit_roles'][nodes] != 255]
    y = packet['fit_roles'][role_indices].astype(int)-1
    base,w = balanced_weights(y,packet['node_units'][role_indices],families[role_indices])
    node_stats = normalizer(packet['features'][role_indices],base)
    role_support = np.array([packet['role_support'][held][str(k)]['fit_support_sufficient'] for k in range(1,10)])
    relation_support = np.array([packet['relation_support'][held][k]['fit_support_sufficient'] for k in KINDS])
    pair_rows = [p for p in packet['fit_pairs'] if p['family'] != held and relation_support[p['target']]]
    pairs = np.array([r['nodes'] for r in pair_rows],np.int32).reshape(-1,2)
    labels = np.array([r['target'] for r in pair_rows],int)
    pair_base,pair_weights = balanced_weights(labels,[r['unit'] for r in pair_rows],[r['family'] for r in pair_rows])
    geometry = pair_geometry(packet,pairs)
    pair_stats = normalizer(geometry,pair_base)
    edges = packet['edges']
    edges = edges[(families[edges[:,0]] != held) & (families[edges[:,1]] != held)]
    edges = np.c_[remap[edges[:,:2]],edges[:,2]].astype(np.int32)
    if np.any(remap[pairs] < 0):
        raise ValueError('Held source leaked into FIT pairs')
    return dict(global_nodes=nodes,edges=edges,x=standardize(packet['features'][nodes],node_stats),
                role_indices=remap[role_indices],role_global_indices=role_indices,role_targets=y,role_base=base,role_weights=w,
                pairs=remap[pairs],pair_global_indices=pairs,pair_targets=labels,pair_base=pair_base,pair_weights=pair_weights,
                geometry=standardize(geometry,pair_stats),node_stats=node_stats,pair_stats=pair_stats,
                role_support=role_support,relation_support=relation_support,pair_units=[p['unit'] for p in pair_rows])
