"""Read-only joint qualification; no target files or production promotion."""
from collections import Counter,defaultdict
import numpy as np
from .part_admission_v1 import object_sha


def produce(root,view,folder):
    """Capture actual source/geometry/roles together using frozen pixel producers."""
    import cv2
    from PIL import Image
    from source_chain_inputs_v1 import prepare
    from run_head_recall_v1 import infer
    from .source_chain_profile_v2 import analyze as routes,diagnostic_heads
    from .head_junction_ownership_v1 import apply
    from .open_reference_chains_v2 import extend
    from .open_reference_context_v2 import filter_context
    from .joint_cv_profile_v1 import analyze as joint
    from .dataset_layout_v1 import pin
    from .pipeline import write_json
    folder.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(1);cv2.setRNGSeed(0)
    gray,source,geometry,inherited,graph,local,old_heads,extra,calls=prepare(root,view)
    routed=routes(gray,graph,inherited,local['local']['text_mask'],old_heads,diagnostic_heads=extra)
    advanced=diagnostic_heads(gray,old_heads);raw_heads=advanced['candidates']
    raw=routes(gray,graph,routed['pixel_roles'],local['local']['text_mask'],raw_heads,diagnostic_heads=extra)
    base=apply(gray,graph,routed['pixel_roles'],raw,raw_heads)
    localized,context,detector,heads=infer(gray,raw_heads,graph['tracks'])
    allowed=set(base['head_ids_used']);original_ids={h['id'] for h in localized['localized']}
    effective=[h for h in heads['guarded'] if h['id'] in allowed or h['id'] not in original_ids]
    links=filter_context(gray,graph,base,extend(gray,graph,base,effective))
    fullbase={**base,'dimension_chains':base['dimension_chains']+links['open_reference_chains']}
    packet=joint(gray,graph,fullbase,heads['previous'],heads['guarded'],prior_binding_ids=allowed&{h['id'] for h in heads['previous']})
    Image.fromarray(packet['joint']['pixel_roles']).save(folder/'roles.png')
    values={'graph':graph,'heads':packet['width_contract']['normalized_heads'],'links':packet['joint']['dimension_chains'],
            'source':source,'geometry':geometry}
    for name,value in values.items():write_json(folder/(name+'.json'),value)
    complete=not any((graph.get('search_incomplete',False),routed['search_incomplete'],advanced['search_incomplete'],
        localized['search_incomplete'],not context['complete'],not detector['complete'],links['search_incomplete'],packet['joint']['search_incomplete']))
    write_json(folder/'producer.json',{'complete':complete,'GT_used':False,'cached_predictions_used':False,'calls':calls})
    return {k:pin(root,(folder/f).relative_to(root).as_posix()) for k,f in
        [('graph','graph.json'),('heads','heads.json'),('links','links.json'),('roles','roles.png')]}, {
        k:pin(root,(folder/(k+'.json')).relative_to(root).as_posix()) for k in ('source','geometry')},complete


def inspect_bundle(gray,graph,heads,chains,roles,matrix,*,geometry=None,source=None):
    """Check identities and index ownership without inventing missing geometry."""
    before=object_sha([graph,heads,chains,geometry,source]);issues=[]
    proposals=graph['proposals'];ids=[p['id'] for p in proposals];parents=defaultdict(list)
    if len(ids)!=len(set(ids)):issues.append('duplicate_interval_ids')
    for p in proposals:
        low,high=p['source_index_interval'];points=np.asarray(p['points_px'],float)
        if not 0<=low<=high or points.shape!=(high-low+1,2) or not np.isfinite(points).all():issues.append('invalid_source_index_interval:'+p['id'])
        parents[p['parent_primitive_id']].append(p)
    geometry_binding=None
    if geometry is not None and source is not None:
        models={g['id']:g for g in geometry};paths={s['path_id']:s for s in source};binding=[]
        for p in proposals:
            g=models.get(p['parent_primitive_id']);s=paths.get(p['path_id']);lo,hi=p['source_index_interval']
            if g is None or s is None:binding.append('missing_parent_or_source:'+p['id']);continue
            if g['path_id']!=p['path_id'] or g['kind']!=p['parent_kind']:binding.append('parent_identity_mismatch:'+p['id'])
            if p['source_observation_id']!=s['id'] or g['source_observation_id']!=s['id']:binding.append('source_identity_mismatch:'+p['id'])
            if not np.array_equal(np.asarray(p['points_px']),np.asarray(s['points_px'])[lo:hi+1]):binding.append('source_points_mismatch:'+p['id'])
            gl,gh=g['source_index_interval']
            if not gl<=lo<=hi<=gh:binding.append('interval_outside_parent:'+p['id'])
        for g in geometry:
            spans=sorted((p['source_index_interval'] for p in parents.get(g['id'],[])))
            if not spans or spans[0][0]!=g['source_index_interval'][0] or spans[-1][1]!=g['source_index_interval'][1] or any(a[1]!=b[0] for a,b in zip(spans,spans[1:])):
                binding.append('parent_edges_not_partitioned:'+g['id'])
        geometry_binding={'available':True,'valid':not binding,'issues':binding,'primitive_count':len(geometry),
            'source_path_count':len(source),'kind_counts':dict(Counter(g['kind'] for g in geometry))}
    else:geometry_binding={'available':False,'valid':None,'issues':['typed_geometry_snapshot_unavailable']}
    head_ids=[h['id'] for h in heads];head_set=set(head_ids);interval_set=set(ids);track_set={t['id'] for t in graph['tracks']};chain_issues=[]
    if len(head_ids)!=len(head_set):chain_issues.append('duplicate_head_ids')
    for i,c in enumerate(chains):
        if not set(c.get('head_ids',[]))<=head_set:chain_issues.append({'chain':i,'reason':'missing_head','ids':sorted(set(c.get('head_ids',[]))-head_set)})
        for key in ('shaft_interval_ids','contour_interval_ids'):
            if not set(c.get(key,[]))<=interval_set:chain_issues.append({'chain':i,'reason':'missing_'+key})
        for group in c.get('extension_interval_ids',[]):
            if isinstance(group,str):group=[group]
            if not set(group)<=interval_set:chain_issues.append({'chain':i,'reason':'missing_extension_interval'})
        if c.get('shaft_track') and c['shaft_track'] not in track_set:chain_issues.append({'chain':i,'reason':'missing_shaft_track'})
    m=np.asarray(matrix,float);samples=defaultdict(Counter)
    for p in proposals:
        q=np.rint(np.asarray(p['points_px'])@m[:2,:2].T+m[:2,2]).astype(int)
        valid=(q>=0).all(1)&(q[:,0]<roles.shape[1])&(q[:,1]<roles.shape[0]);q=q[valid]
        samples[p['parent_primitive_id']].update(map(int,roles[q[:,1],q[:,0]]))
    ambiguous=sum(sum(v>0 for k,v in counts.items() if k not in (0,255))>1 for counts in samples.values())
    return {'source_graph_identity_issues':issues,'geometry_binding':geometry_binding,'chain_identity_issues':chain_issues,
        'head_count':len(heads),'chain_count':len(chains),'interval_count':len(proposals),'parent_primitive_count':len(parents),
        'parents_with_multiple_sampled_roles':ambiguous,'sampling_is_not_CAD_role_authority':True,
        'source_immutable':before==object_sha([graph,heads,chains,geometry,source]),'GT_used':False}


def geometry_qa(source,geometry,matrix):
    from .curve_graph_metrics_v1 import native
    from .robust_topology_v1 import build
    from .robust_graph_qa_v1 import guard,verified_provenance
    before=object_sha([source,geometry]);a=[native(c,matrix) for c in source];b=[native(c,matrix) for c in geometry]
    ga,gb=build(a),build(b)
    if ga['complete'] and gb['complete']:
        ends=defaultdict(list)
        for path in a:
            points=path['points_px'];ends[tuple(points[0])].append((path['path_id'],0))
            if len(points)>1:ends[tuple(points[-1])].append((path['path_id'],-1))
        evidence=[{'observed_native_px':list(point),'estimated_native_px':list(point),'incidences':incident}
            for point,incident in sorted(ends.items())]
        proof=verified_provenance(ga,gb,a,b,evidence);result=guard(a,b,graphs=(ga,gb),provenance=proof)
    else:result={'decision':{'complete':False,'accepted':False,'reasons':['incomplete_observed_geometry_graph'],'distance':None}}
    return {'decision':result['decision'],'source_graph':{k:ga.get(k) for k in ('complete','components','cycle_rank','bounded_faces')},
        'candidate_graph':{k:gb.get(k) for k in ('complete','components','cycle_rank','bounded_faces')},
        'source_immutable':before==object_sha([source,geometry]),'fallback_not_applied_to_role_graph':True,
        'reason':'Qualification only; replacing primitives requires rebuilding and validating interval ownership'}
