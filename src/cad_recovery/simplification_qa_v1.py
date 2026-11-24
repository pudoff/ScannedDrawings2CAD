"""Fail-closed simplification decisions and separate geometric/export evidence."""
from collections import Counter,defaultdict
import copy
import math

import cv2
import numpy as np

from .primitive_graph_qa_v1 import build
from .curve_graph_v1 import graph_paths,curve_distance,sample
from .curve_graph_metrics_v1 import native
from .stroke_curve_control_v1 import maximum_matching,sample_polyline,distance_to_polyline

POLICY={'version':'simplification-qa/1.0','matching_native_px':3.,'maximum_bidirectional_native_px':2.,
    'maximum_relative_face_area_change':.10,'maximum_sampling_points_per_path':120000,
    'sampling_spacing_native_px':.5,'distance_sampling_bound_native_px':.25,
    'maximum_owner_distance_tests':30000000,'unknown_semantic_role':'unchanged',
    'gate_scope':'observed geometric graph only; not physical CAD approval'}


def source_paths(skeleton,matrix):
    paths,stats=graph_paths(skeleton)
    if stats['search_incomplete']:raise ValueError('Incomplete source paths')
    values=[{'id':f'source-{i:05d}','path_id':f'observed-path-{i:05d}','kind':'polyline',
        'points_px':p['points'].tolist(),'closed':p['closed'],'review_status':'observed_unknown'} for i,p in enumerate(paths)]
    # graph_paths exposes isolated pixel counts but no path for them. Keep them explicitly in QA.
    occupied=np.zeros_like(skeleton,bool)
    for p in paths:
        points=p['points'].astype(int);occupied[points[:,1],points[:,0]]=True
    y,x=np.nonzero((skeleton>0)&~occupied)
    for i,(px,py) in enumerate(zip(x,y)):
        values.append({'id':f'isolated-{i:05d}','path_id':f'isolated-{i:05d}','kind':'polyline',
            'points_px':[[int(px),int(py)]],'closed':False,'review_status':'observed_unknown'})
    return [native(c,matrix) for c in values],stats


def match_nodes(before,after):
    output={};details=[]
    for category,test in [('junction',lambda d:d>=3),('endpoint',lambda d:d==1),('isolated',lambda d:d==0)]:
        a=[n for n in before['structural_nodes'] if test(n['degree'])]
        b=[n for n in after['structural_nodes'] if test(n['degree'])]
        edges=[];distances=[]
        for expected in a:
            d={j:math.dist(expected['point_px'],p['point_px']) for j,p in enumerate(b)}
            edges.append(sorted((j for j in d if d[j]<=3),key=d.get));distances.append(d)
        assignment=maximum_matching(edges)
        rows=[{'before_id':a[i]['id'],'after_id':b[j]['id'],'distance_px':distances[i][j],
            'degree_before':a[i]['degree'],'degree_after':b[j]['degree'],
            'owners_preserved':a[i]['owners']==b[j]['owners']} for i,j in assignment.items()]
        output[category]={'expected':len(a),'predicted':len(b),'matched':len(rows),'missed':len(a)-len(rows),
            'extra':len(b)-len(rows),'wrong_degree':sum(r['degree_before']!=r['degree_after'] for r in rows),
            'wrong_owners':sum(not r['owners_preserved'] for r in rows),
            'RMSE_px':math.sqrt(sum(r['distance_px']**2 for r in rows)/len(rows)) if rows else None,
            'matches':rows}
    return output


def face_comparison(before,after):
    a=defaultdict(list);b=defaultdict(list)
    for face in before['faces']:a[tuple(face['owners'])].append(face)
    for face in after['faces']:b[tuple(face['owners'])].append(face)
    old_lookup={f['id']:tuple(f['owners']) for f in before['faces']}
    new_lookup={f['id']:tuple(f['owners']) for f in after['faces']}
    rows=[];missing=[];extra=[];ambiguous=[]
    for owners in sorted(a.keys()|b.keys()):
        if len(a[owners])!=1 or len(b[owners])!=1:
            if not b[owners]:missing.extend(f['id'] for f in a[owners])
            elif not a[owners]:extra.extend(f['id'] for f in b[owners])
            else:ambiguous.append(list(owners))
            continue
        old,new=a[owners][0],b[owners][0]
        rows.append({'owners':list(owners),'relative_area_change':abs(new['area_px2']-old['area_px2'])/old['area_px2'],
            'before_area_px2':old['area_px2'],'after_area_px2':new['area_px2'],
            'parent_owners_preserved':old_lookup.get(old['parent'])==new_lookup.get(new['parent']),
            'depth_preserved':old['depth']==new['depth']})
    return {'matches':rows,'missing':missing,'extra':extra,'ambiguous_owner_sets':ambiguous,
        'complete':before['nesting_complete'] and after['nesting_complete'] and not ambiguous}


def compare_graphs(before,after,*,require_owners=True):
    if not before['complete'] or not after['complete']:
        return {'complete':False,'accepted':False,'reasons':['incomplete_graph'],
            'editing_cost_seconds':None,'scope':'no authority from incomplete analysis'}
    nodes=match_nodes(before,after);faces=face_comparison(before,after);reasons=[]
    for name in ('components','cycle_rank','bounded_faces'):
        if before[name]!=after[name]:reasons.append(name+'_changed')
    if require_owners and before['component_owners']!=after['component_owners']:reasons.append('component_ownership_changed')
    for category,m in nodes.items():
        if m['missed'] or m['extra'] or m['wrong_degree']:reasons.append(category+'_incidence_changed')
        if require_owners and m['wrong_owners']:reasons.append(category+'_ownership_changed')
    if not faces['complete']:reasons.append('unresolved_face_or_nesting')
    if require_owners and (faces['missing'] or faces['extra']):reasons.append('face_ownership_changed')
    if require_owners and any(not f['parent_owners_preserved'] or not f['depth_preserved'] for f in faces['matches']):reasons.append('nesting_changed')
    if require_owners and any(f['relative_area_change']>.1 for f in faces['matches']):reasons.append('protected_face_area_changed')
    if after['duplicate_geometric_pieces']>before['duplicate_geometric_pieces']:reasons.append('new_duplicate_pieces')
    operations={'insert_missing_nodes':sum(m['missed'] for m in nodes.values()),
        'delete_extra_nodes':sum(m['extra'] for m in nodes.values()),
        'repair_degree':sum(m['wrong_degree'] for m in nodes.values()),
        'cycle_rank_difference':abs(before['cycle_rank']-after['cycle_rank']),
        'missing_faces':len(faces['missing']),'extra_faces':len(faces['extra']),
        'repair_nesting':sum(not f['parent_owners_preserved'] or not f['depth_preserved'] for f in faces['matches'])}
    return {'complete':True,'accepted':not reasons,'reasons':reasons,'nodes':nodes,'faces':faces,
        'component_difference':after['components']-before['components'],
        'cycle_rank_difference':after['cycle_rank']-before['cycle_rank'],
        'editing_operations_proxy':operations,'unit_operation_proxy':sum(operations.values()),
        'editing_cost_seconds':None,'editing_cost_note':'diagnostic operation vector, not minimal graph edit distance or measured human time',
        'owner_correspondence_required':require_owners,'physical_geometry_approved':False}


def dense_samples(c):
    if c['kind']=='polyline':
        p=c['points_px'];result=np.asarray(p,float) if len(p)==1 else sample_polyline(p,.5)
        if c.get('closed',False) and len(p)>1 and math.dist(p[0],p[-1])>1e-7:
            result=np.r_[result,sample_polyline([p[-1],p[0]],.5)]
    else:result=sample(c,.5)
    if len(result)>120000:raise ValueError('sampling_budget')
    return result


def owner_errors(before,after):
    left=defaultdict(list);right=defaultdict(list)
    for c in before:left[c.get('path_id',c['id'])].append(c)
    for c in after:right[c.get('path_id',c['id'])].append(c)
    rows=[];work=0
    for owner in sorted(left.keys()|right.keys()):
        if not left[owner] or not right[owner]:
            rows.append({'owner':owner,'complete':False,'reason':'missing_owner','max_px':None});continue
        def directed(a,b):
            nonlocal work
            samples=np.concatenate([dense_samples(c) for c in a]);d=np.full(len(samples),np.inf)
            for c in b:
                if c['kind'] in ('arc','circle'):work+=len(samples);values=curve_distance(samples,c)
                elif len(c['points_px'])==1:work+=len(samples);values=np.linalg.norm(samples-np.asarray(c['points_px'][0]),axis=1)
                else:
                    points=c['points_px']
                    if c.get('closed',False) and math.dist(points[0],points[-1])>1e-7:points=points+[points[0]]
                    work+=len(samples)*(len(points)-1)
                    if work>30000000:raise ValueError('owner_distance_budget')
                    values=distance_to_polyline(samples,points)
                if work>30000000:raise ValueError('owner_distance_budget')
                d=np.minimum(d,values)
            return d
        try:
            errors=np.r_[directed(left[owner],right[owner]),directed(right[owner],left[owner])]
            rows.append({'owner':owner,'complete':True,'max_px':float(errors.max()),
                'p95_px':float(np.percentile(errors,95)),'mean_px':float(errors.mean()),
                'sampling_bound_px':.25,'units':'native_px'})
        except ValueError as exc:rows.append({'owner':owner,'complete':False,'max_px':None,'reason':str(exc)})
    return {'rows':rows,'complete':all(r['complete'] for r in rows),'distance_tests':work,
        'max_px':max((r['max_px'] for r in rows if r['max_px'] is not None),default=None),
        'missing_or_incomplete_owners':sum(not r['complete'] for r in rows),'sampling_bound_px':.25}


def guard(before,proposal,*,graphs=None):
    """Select the unchanged source on any QA failure; never repair or move endpoints."""
    a,b=graphs if graphs is not None else (build(before),build(proposal))
    result=compare_graphs(a,b)
    errors=owner_errors(before,proposal) if a['complete'] and b['complete'] else {
        'complete':False,'reason':'incomplete_graph','max_px':None,'rows':[]}
    if not errors['complete']:result['reasons'].append('incomplete_geometric_error')
    # Add the sampling envelope to the measured maximum, rather than silently dropping it.
    if errors['max_px'] is not None and errors['max_px']+.25>2:result['reasons'].append('geometric_error_with_sampling_bound')
    result['accepted']=not result['reasons'];result['distance']=errors
    result['selection']='proposal' if result['accepted'] else 'unchanged_source'
    result['production_admitted']=False
    return {'decision':result,'selected':copy.deepcopy(proposal if result['accepted'] else before)}


def read_diagnostic_dxf(path,height,templates):
    """Independent reader of actual modelspace; unsupported/extra native types fail."""
    import ezdxf
    doc=ezdxf.readfile(path);audit=doc.audit();entities=list(doc.modelspace())
    if len(entities)!=len(templates):raise ValueError('Native primitive count mismatch')
    def flip(p):return [float(p[0]),height-1-float(p[1])]
    rows=[];layer_mismatch=[];style_mismatch=[]
    expected_layers={'line':'CV_LINES','arc':'CV_ARCS','circle':'CV_CIRCLES','polyline':'UNRESOLVED_PATHS'}
    for entity,template in zip(entities,templates):
        if entity.dxf.hasattr('extrusion') and math.dist(tuple(entity.dxf.extrusion),(0,0,1))>1e-10:raise ValueError('Unexpected native OCS')
        if entity.dxf.hasattr('thickness') and abs(entity.dxf.thickness)>1e-10:raise ValueError('Unexpected native thickness')
        for attribute in ('start','end','center'):
            if entity.dxf.hasattr(attribute):
                p=getattr(entity.dxf,attribute)
                if not np.isfinite(tuple(p)).all() or abs(p.z)>1e-8:raise ValueError('Unexpected nonplanar native point')
        if entity.dxftype()=='LWPOLYLINE' and abs(entity.dxf.elevation)>1e-8:raise ValueError('Unexpected native elevation')
        c={k:v for k,v in template.items() if k in ('id','path_id')};kind=entity.dxftype()
        if kind=='LINE':c.update(kind='line',points_px=[flip(entity.dxf.start),flip(entity.dxf.end)])
        elif kind=='CIRCLE':c.update(kind='circle',center_px=flip(entity.dxf.center),radius_px=entity.dxf.radius)
        elif kind=='ARC':c.update(kind='arc',center_px=flip(entity.dxf.center),radius_px=entity.dxf.radius,
            start_angle_deg=(-entity.dxf.end_angle)%360,sweep_angle_deg=(entity.dxf.end_angle-entity.dxf.start_angle)%360)
        elif kind=='LWPOLYLINE':
            points=entity.get_points()
            if any(any(abs(p[i])>1e-12 for i in (2,3,4)) for p in points):raise ValueError('Unexpected diagnostic polyline width/bulge')
            c.update(kind='polyline',points_px=[flip(p) for p in points],closed=bool(entity.closed))
        else:raise ValueError('Unexpected native modelspace entity')
        if c['kind']!=template['kind']:raise ValueError('Native kind changed')
        if entity.dxf.layer!=expected_layers[c['kind']]:layer_mismatch.append(c['id'])
        if entity.dxf.color!=256 or entity.dxf.lineweight!=-1 or entity.dxf.linetype.upper()!='BYLAYER':style_mismatch.append(c['id'])
        c['native_layer']=entity.dxf.layer;rows.append(c)
    return rows,{'audit_errors':len(audit.errors),'audit_fixes':len(audit.fixes),'INSUNITS':doc.units,
        'layer_mismatch_ids':layer_mismatch,'by_layer_mismatch_ids':style_mismatch,
        'modelspace_entities':len(entities),'handle_locator':'file order checked against type, geometry and layers; not an expected-value source'}
