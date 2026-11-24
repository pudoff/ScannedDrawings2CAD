"""Join analysis tracks at exact observed nodes; source geometry stays unchanged."""
from collections import defaultdict
from copy import deepcopy
import math
import cv2
import numpy as np
from .open_section_witness_v3 import observed_graph
from .grid_role_revision_v1 import sha
from .part_admission_v1 import object_sha
from .stroke_roles_v1 import checked_gray,foreground

POLICY = {'version':'open-source-continuation/1.0','maximum_tracks':512,
          'maximum_source_points':2000000,'maximum_checks':250000,
          'direction_tolerance_deg':8.,'maximum_pen_ratio':1.67,
          'join':'exact_observed_endpoint_reciprocal_unique',
          'geometry_modified':False,'semantic_PART_assigned':False}


def observe(gray,graph):
    checked_gray(gray); before = object_sha(graph); pixels = sha(gray)
    probe,upstream = observed_graph(graph)
    audit = {'source_points':0,'checks':0,'limits':[]}
    if not upstream['complete']: audit['limits'].append('incomplete_source_observer')
    if len(probe.get('tracks',[])) > POLICY['maximum_tracks']: audit['limits'].append('track_budget')
    lookup = {p['id']:p for p in probe['proposals']}; ink = foreground(gray)
    distance = cv2.distanceTransform(ink,cv2.DIST_L2,5); edges = []; incidence = defaultdict(list)
    for index,track in enumerate(probe.get('tracks',[])) if not audit['limits'] else []:
        members = [lookup[i] for i in track['interval_ids']]
        points = np.concatenate([np.asarray(p['points_px'],float) for p in members])
        audit['source_points'] += len(points)
        if audit['source_points'] > POLICY['maximum_source_points']: audit['limits'].append('source_point_budget'); break
        q = np.rint(points).astype(int)
        if (q < 0).any() or (q[:,0] >= gray.shape[1]).any() or (q[:,1] >= gray.shape[0]).any():
            raise ValueError('Source track outside current pixels')
        ends = np.asarray(track['points'],float); vector = ends[1]-ends[0]; length = float(np.linalg.norm(vector))
        values = distance[q[:,1],q[:,0]]; values = values[values > 0]
        usable = (length > 0 and len(values) > 0 and all(p.get('parent_kind') not in ('arc','circle') for p in members)
                  and np.array_equal(ends,np.rint(ends)) and np.all(ink[q[:,1],q[:,0]] > 0))
        pen = max(1.,float(2*np.quantile(values,.25)-1)) if len(values) else 1.
        edges.append({'index':index,'track':track,'ends':ends,'u':vector/length if length else np.zeros(2),
                      'pen':pen,'usable':usable,'owners':set(track['interval_ids'])})
        if usable:
            incidence[tuple(ends[0])].append((index,0)); incidence[tuple(ends[1])].append((index,1))
    next_half = {}; cosine = math.cos(math.radians(POLICY['direction_tolerance_deg']))
    for edge in edges if not audit['limits'] else []:
        if not edge['usable']: continue
        for endpoint in (0,1):
            incoming = edge['u'] if endpoint == 1 else -edge['u']; options = []
            for index,other_endpoint in incidence[tuple(edge['ends'][endpoint])]:
                audit['checks'] += 1
                if audit['checks'] > POLICY['maximum_checks']: audit['limits'].append('continuation_budget'); break
                other = edges[index]
                if index == edge['index'] or edge['owners'] & other['owners']: continue
                outgoing = other['u'] if other_endpoint == 0 else -other['u']
                ratio = edge['pen']/other['pen']
                if float(incoming@outgoing) >= cosine and 1/POLICY['maximum_pen_ratio'] <= ratio <= POLICY['maximum_pen_ratio']:
                    options.append((index,other_endpoint))
            next_half[edge['index'],endpoint] = options[0] if len(options) == 1 else None
            if audit['limits']: break
        if audit['limits']: break
    links = {half:nxt for half,nxt in next_half.items() if nxt is not None and next_half.get(nxt) == half}
    done = set(); tracks = []; groups = []
    order = sorted(edges,key=lambda e:(sum((e['index'],end) in links for end in (0,1)),e['index']))
    for edge in order if not audit['limits'] else []:
        if edge['index'] in done: continue
        start_endpoint = 0 if (edge['index'],0) not in links else 1
        route = []; current = edge['index']; entry = start_endpoint; visited = set()
        while current not in visited and current not in done:
            visited.add(current); route.append((current,entry))
            nxt = links.get((current,1-entry))
            if nxt is None: break
            current,entry = nxt
        done.update(visited)
        if len(route) < 2 or (links.get((route[-1][0],1-route[-1][1])) is not None and current in visited):
            tracks.extend(deepcopy(edges[i]['track']) for i,_ in route); continue
        first,last = route[0],route[-1]
        ends = [edges[first[0]]['ends'][first[1]],edges[last[0]]['ends'][1-last[1]]]
        if np.array_equal(ends[0],ends[1]):
            tracks.extend(deepcopy(edges[i]['track']) for i,_ in route); continue
        ids = [identity for i,_ in route for identity in edges[i]['track']['interval_ids']]
        if len(set(ids)) != len(ids): raise ValueError('Continuation repeats a source owner')
        delta = ends[1]-ends[0]; length = float(np.linalg.norm(delta))
        name = 'source-continuation/'+object_sha([edges[i]['track']['id'] for i,_ in route])[:24]
        tracks.append({'id':name,'points':np.asarray(ends).tolist(),'u':(delta/length).tolist(),
                       'length':length,'width':float(np.median([edges[i]['pen'] for i,_ in route])),
                       'interval_ids':ids})
        groups.append({'id':name,'source_track_ids':[edges[i]['track']['id'] for i,_ in route],
                       'interval_ids':ids,'join_nodes':[edges[i]['ends'][1-e].tolist() for i,e in route[:-1]]})
    if audit['limits']:
        probe['search_incomplete'] = True
    else:
        probe['tracks'] = tracks
    if before != object_sha(graph) or pixels != sha(gray): raise ValueError('Continuation mutated source')
    return probe,{'complete':not audit['limits'],'groups':groups,'audit':audit,'policy':deepcopy(POLICY),
        'original_graph_sha256':before,'observed_graph_sha256':object_sha(probe),
        'input_tracks':len(graph.get('tracks',[])),'output_tracks':len(probe.get('tracks',[])),
        'source_immutable':True,'geometry_modified':False}
