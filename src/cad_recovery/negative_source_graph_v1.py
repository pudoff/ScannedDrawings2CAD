"""Independent source-bound negative observation graph; legacy graph stays intact."""
from collections import Counter,defaultdict
from copy import deepcopy
import math
import cv2
import numpy as np
from .cad_object_assembly_v5 import assemble
from .local_contour_probes_v1 import probes
from .local_contour_witness_v1 import inventory
from .ownership_contours_v1 import boundaries
from .part_admission_v1 import object_sha
from .grid_role_revision_v1 import sha
from .stroke_roles_v1 import foreground,checked_gray

POLICY={'version':'negative-source-graph/1.0','maximum_directions':512,'maximum_objects':4096,
    'maximum_source_points':2000000,'maximum_raster_pixels':64000000,
    'source_owner_remapped':False,'legacy_graph_modified':False,'creates_positive_roles':False}


def build(packet):
    gray,roles=packet['gray'],packet['roles'];checked_gray(gray)
    old=packet['graph'];owner=packet['source_owner'];ambiguous=packet['ambiguous_owner_mask']
    if (packet.get('input_sha256')!=sha(gray) or roles.shape!=gray.shape or roles.dtype!=np.uint8
            or not np.isin(roles,[*range(10),255]).all() or owner.shape!=gray.shape or owner.dtype.kind not in 'ui'
            or owner.min(initial=0)<0 or owner.max(initial=0)>len(old['proposals'])
            or ambiguous.shape!=gray.shape or ambiguous.dtype!=np.uint8 or np.any((ambiguous>0)&(owner!=0))):
        raise ValueError('Checked aligned negative source observations required')
    source_inventory=inventory(packet)
    before=object_sha([old,packet['geometry'],packet['heads']]);arrays=[sha(a) for a in (gray,roles,packet['source_owner'],packet['ambiguous_owner_mask'])]
    result=assemble(gray,packet['geometry'],roles=roles)
    objects,probe_audit=probes(result,packet['geometry'],maximum_segments=POLICY['maximum_objects'])
    audit={'limits':[],'source_points':source_inventory['source_points'],'raster_pixels':0,'native_line_groups':0,
        'individual_interval_tracks':0,'text_only_interval_ids':[],'covered_interval_ids':[],
        'source_assembly_complete':result['search_complete'],'analysis_probes':probe_audit}
    if packet.get('fresh_source_extraction_complete',True) is not True:audit['limits'].append('source_extraction_incomplete')
    if not result['search_complete'] or not probe_audit['complete']:audit['limits'].append('source_observations_incomplete')
    items=deepcopy(old['proposals']);by_parent=defaultdict(list);text=set();assigned=set();tracks=[]
    for n,p in enumerate(items):
        q=np.asarray(p['points_px'],int)
        if np.all(roles[q[:,1],q[:,0]]==6):text.add(n);audit['text_only_interval_ids'].append(p['id'])
        by_parent[p['parent_primitive_id']].append(n)
    ink=foreground(gray);dt=cv2.distanceTransform(ink,cv2.DIST_L2,5)
    for p in items:
        q=np.asarray(p['points_px'],int)
        if not math.isfinite(float(p.get('width_px',1.))) or float(p.get('width_px',1.))<=0:raise ValueError('Finite negative source width required')
        if 'width_px' not in p:p['width_px']=max(1.,float(2*np.median(dt[q[:,1],q[:,0]])))
    def add(numbers,axis,origin,identity):
        if not numbers:return
        q=np.unique(np.concatenate([np.asarray(items[n]['points_px'],int) for n in numbers]),axis=0)
        lo,hi=q.min(0),q.max(0)+1;audit['raster_pixels']+=int(np.prod(hi-lo))
        if audit['raster_pixels']>POLICY['maximum_raster_pixels']:audit['limits'].append('source_connectivity_budget');return
        raster=np.zeros(tuple((hi-lo)[::-1]),np.uint8);raster[q[:,1]-lo[1],q[:,0]-lo[0]]=1
        _,labels=cv2.connectedComponents(raster,connectivity=8);parts=defaultdict(list)
        for n in numbers:
            p=np.asarray(items[n]['points_px'],int);ls=np.unique(labels[p[:,1]-lo[1],p[:,0]-lo[0]])
            if len(ls)!=1 or ls[0]==0:raise ValueError('Disconnected observed interval')
            parts[int(ls[0])].append(n)
        for component,ns in sorted(parts.items()):
            points=np.concatenate([np.asarray(items[n]['points_px'],float) for n in ns]);t=(points-origin)@axis
            if np.ptp(t)<=0:continue
            ends=np.array([origin+t.min()*axis,origin+t.max()*axis]);width=float(np.median([items[n]['width_px'] for n in ns]))
            for n in ns:items[n]['linear_context']=True;assigned.add(n)
            tracks.append({'id':identity+'/'+str(component),'points':ends.tolist(),'u':axis.tolist(),
                'length':float(np.ptp(t)),'width':width,'interval_ids':[items[n]['id'] for n in sorted(ns)]})
    for obj in objects if not audit['limits'] else []:
        if obj['kind']!='line':continue
        numbers=[]
        for m in obj['source_members']:
            for n in by_parent[m['candidate_id']]:
                if n in text or n in assigned or n in numbers:continue
                low,high=items[n]['source_index_interval']
                if low<high and m['source_index_interval'][0]<=low<=high<=m['source_index_interval'][1]:numbers.append(n)
        if not numbers:continue
        ends=np.asarray(obj['points_px'],float);axis=ends[1]-ends[0];axis/=np.linalg.norm(axis)
        add(numbers,axis,ends[0],'negative-line/'+object_sha(obj['id'])[:24]);audit['native_line_groups']+=1
        if audit['limits']:break
    for n,p in enumerate(items) if not audit['limits'] else []:
        if n in text or n in assigned or not p.get('linear_context'):continue
        q=np.asarray(p['points_px'],float);axis=q[-1]-q[0];norm=np.linalg.norm(axis)
        if norm==0:raise ValueError('Declared linear interval is degenerate')
        add([n],axis/norm,q[0],'negative-interval/'+object_sha(p['id'])[:24]);audit['individual_interval_tracks']+=1
        if audit['limits']:break
    audit['covered_interval_ids']=[items[n]['id'] for n in sorted(assigned)]
    if len(tracks)>POLICY['maximum_directions']:audit['limits'].append('direction_budget')
    counts=Counter(i for t in tracks for i in t['interval_ids'])
    if any(v!=1 for v in counts.values()):raise ValueError('A negative interval belongs to multiple tracks')
    if any(p['linear_context'] and n not in text and p['id'] not in counts for n,p in enumerate(items)):
        audit['limits'].append('uncovered_linear_source')
    # Excluded TEXT is still retained, with its exact points and old owners.
    # It must not re-enter straight context through an old linear flag.
    for n in text:items[n]['linear_context']=False
    stroke=float(old.get('stroke_px',max(1.,min(24.,float(np.quantile([p['width_px'] for p in items],.25)) if items else 1.))))
    topology=deepcopy(old.get('boundaries',{'search_incomplete':True}))
    if topology.get('search_incomplete'):
        geometry=[{'id':p['id'],'path_id':p['id'],'kind':'line' if p['linear_context'] else 'polyline',
            'points_px':[p['points_px'][0],p['points_px'][-1]] if p['linear_context'] else p['points_px'],
            'closed':len(p['points_px'])>1 and p['points_px'][0]==p['points_px'][-1]} for p in items]
        topology=boundaries(geometry,gray.shape,stroke);audit['topology_origin']='rebuilt_from_checked_intervals'
    else:audit['topology_origin']='retained_checked_source_boundary_observations'
    if topology.get('search_incomplete'):audit['limits'].append('source_topology_incomplete')
    ids={p['id'] for p in items}
    for boundary in topology.get('boundaries',[]):
        if not set(boundary['owners'])<=ids:raise ValueError('Foreign negative boundary source')
    graph={**deepcopy(old),'schema':'independent-negative-source-graph/1.0','proposals':items,'tracks':tracks,
        'boundaries':topology,'stroke_px':stroke,'dimension_chains':deepcopy(old.get('dimension_chains',[])),
        'search_incomplete':bool(audit['limits'])}
    certificate={'policy':deepcopy(POLICY),'complete':not audit['limits'],'audit':audit,
        'source_sha256':sha(gray),'roles_sha256':sha(roles),'legacy_graph_sha256':object_sha(old),
        'source_owner_sha256':sha(packet['source_owner']),'legacy_graph_search_incomplete':old.get('search_incomplete'),
        'ambiguous_owner_sha256':sha(packet['ambiguous_owner_mask']),
        'geometry_and_heads_sha256':object_sha([packet['geometry'],packet['heads']]),
        'graph_sha256':object_sha(graph),'source_interval_ids_unchanged':[p['id'] for p in items]==[p['id'] for p in old['proposals']],
        'observed_inventory':source_inventory,'direction_count':len(tracks),'full_CV_or_semantic_completeness_claimed':False}
    if before!=object_sha([old,packet['geometry'],packet['heads']]) or arrays!=[sha(a) for a in (gray,roles,packet['source_owner'],packet['ambiguous_owner_mask'])]:
        raise ValueError('Negative source builder changed original observations')
    return graph,certificate
