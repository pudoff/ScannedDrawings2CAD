"""Assigned arrow, dash/gap and open-arc diagnostics, never CAD acceptance."""
import math

import numpy as np

from .family_catalog_v2 import require
from .hard_raster_metrics_v1 import stroke_coverage, transform_points

POLICY={'version':'stroke-curve-control/1.0','tip_tolerance_px':8,'direction_tolerance_deg':20,
    'curve_band_px':8,'curve_endpoint_tolerance_px':12,'curve_length_ratio':[.7,1.3],
    'line_coverage_min':.8,'full_precision':None,'independent_evaluation':False,
    'physical_geometry_approved':False,'CAD_promotions':0}


def validate_references(reference,size):
    width,height=size
    def point(p):
        return len(p)==2 and np.isfinite(p).all() and 0<=p[0]<width and 0<=p[1]<height
    all_rows=sum([reference[k] for k in ('arrows','strokes','gaps','curves','negatives','excluded_arrows')],[])
    require(len({r['id'] for r in all_rows})==len(all_rows), 'Repeated control identity')
    for row in reference['arrows']+reference['excluded_arrows']:
        require(point(row['tip']) and np.isfinite(row['direction']).all()
                and np.linalg.norm(row['direction'])>0 and len(row['observation'])>20, 'Invalid arrow reference')
    for row in reference['strokes']+reference['gaps']+reference['curves']:
        require(len(row['points'])>=2 and all(point(p) for p in row['points'])
                and len(row['observation'])>20, 'Invalid stroke/curve reference')
        require(all(np.linalg.norm(np.asarray(a)-b)>0 for a,b in zip(row['points'],row['points'][1:])), 'Degenerate guide')
    for row in reference['negatives']:
        x,y,right,bottom=row['box']
        require(0<=x<right<=width and 0<=y<bottom<=height and len(row['observation'])>20, 'Invalid negative region')


def maximum_matching(edges):
    """Maximum cardinality; a single candidate cannot explain two assigned tips."""
    owner={}
    def assign(target,visited):
        for candidate in edges[target]:
            if candidate in visited:continue
            visited.add(candidate)
            if candidate not in owner or assign(owner[candidate],visited):
                owner[candidate]=target
                return True
        return False
    for target in range(len(edges)):assign(target,set())
    return {target:candidate for candidate,target in owner.items()}


def arrow_scores(references,candidates,matrix):
    require(np.allclose(np.asarray(matrix)[2],[0,0,1]), 'Only affine pixel transforms are supported')
    predicted=[]
    for candidate in candidates:
        require(candidate.get('classification')=='filled_arrow_candidate', 'Only real arrow observations may score as arrows')
        point=transform_points([candidate['tip_px']],matrix)[0]
        direction=np.asarray(matrix)[:2,:2]@np.asarray(candidate['direction'],float)
        require(np.isfinite(direction).all() and np.linalg.norm(direction)>0, 'Invalid predicted arrow direction')
        predicted.append((point,direction/np.linalg.norm(direction)))
    edges=[];distances=[]
    for ref in references:
        direction=np.asarray(ref['direction'],float);direction/=np.linalg.norm(direction)
        eligible=[];details={}
        for i,(tip,unit) in enumerate(predicted):
            distance=float(np.linalg.norm(tip-np.asarray(ref['tip'])))
            angle=math.degrees(math.acos(float(np.clip(direction@unit,-1,1))))
            if distance<=POLICY['tip_tolerance_px'] and angle<=POLICY['direction_tolerance_deg']:
                eligible.append((distance,i));details[i]=(distance,angle)
        edges.append([i for _,i in sorted(eligible)]);distances.append(details)
    matched=maximum_matching(edges)
    return [{'id':r['id'],'matched':i in matched,'candidate_id':candidates[matched[i]]['id'] if i in matched else None,
        'tip_distance_px':distances[i][matched[i]][0] if i in matched else None,
        'direction_error_deg':distances[i][matched[i]][1] if i in matched else None} for i,r in enumerate(references)]


def sample_polyline(points,spacing=2):
    points=np.asarray(points,float);samples=[]
    for a,b in zip(points,points[1:]):
        length=float(np.linalg.norm(b-a));count=max(2,math.ceil(length/spacing))
        samples.extend(a+np.linspace(0,1,count)[:,None]*(b-a))
    return np.asarray(samples)


def distance_to_polyline(points,vertices):
    points=np.asarray(points,float);vertices=np.asarray(vertices,float)
    distances=np.full(len(points),np.inf)
    for a,b in zip(vertices,vertices[1:]):
        delta=b-a;denom=float(delta@delta)
        if denom<=0:continue
        t=np.clip((points-a)@delta/denom,0,1)
        distances=np.minimum(distances,np.linalg.norm(points-a-t[:,None]*delta,axis=1))
    return distances


def curve_points(candidate,matrix):
    if candidate['kind'] not in ('arc','circle'):return None
    radius=candidate['radius_px'];center=np.asarray(candidate['center_px'],float)
    start=candidate.get('start_angle_deg',0);sweep=candidate.get('sweep_angle_deg',360)
    angles=np.radians(np.linspace(start,start+sweep,max(8,math.ceil(abs(math.radians(sweep)*radius)/2))))
    return transform_points(center+radius*np.c_[np.cos(angles),np.sin(angles)],matrix)


def curve_scores(references,candidates,matrix):
    output=[]
    for ref in references:
        expected=np.asarray(ref['points'],float)
        length=float(np.linalg.norm(np.diff(expected,axis=0),axis=1).sum())
        assigned=sample_polyline(expected)
        rows=[]
        for candidate in candidates:
            predicted=curve_points(candidate,matrix)
            if predicted is None:continue
            coverage=float(np.mean(distance_to_polyline(assigned,predicted)<=POLICY['curve_band_px']))
            if not coverage:continue
            endpoints=min(max(np.linalg.norm(predicted[0]-expected[0]),np.linalg.norm(predicted[-1]-expected[-1])),
                          max(np.linalg.norm(predicted[-1]-expected[0]),np.linalg.norm(predicted[0]-expected[-1])))
            ratio=float(np.linalg.norm(np.diff(predicted,axis=0),axis=1).sum()/length)
            p95=float(np.percentile(distance_to_polyline(predicted,expected),95))
            qualified=(candidate['kind']=='arc' and coverage>=.8 and p95<=POLICY['curve_band_px']
                and endpoints<=POLICY['curve_endpoint_tolerance_px'] and POLICY['curve_length_ratio'][0]<=ratio<=POLICY['curve_length_ratio'][1])
            rows.append({'candidate_id':candidate['id'],'kind':candidate['kind'],'guide_coverage':coverage,
                'candidate_p95_distance_px':p95,'endpoint_max_error_px':float(endpoints),'length_ratio':ratio,'qualified_open_arc':bool(qualified)})
        output.append({'id':ref['id'],'qualified_open_arc':any(r['qualified_open_arc'] for r in rows),
                       'nearby_curve_candidates':rows,'circle_completion_is_not_a_match':True})
    return output


def score(reference,geometry,arrows,matrix):
    candidates=geometry['candidates']
    arrow_rows=arrow_scores(reference['arrows'],arrows['candidates'],matrix)
    endpoints=np.asarray([p for c in candidates if c['kind']=='line' for p in transform_points(c['points_px'],matrix)]).reshape(-1,2)
    for ref,row in zip(reference['arrows'],arrow_rows):
        row['nearby_line_endpoint_only']=bool(len(endpoints) and np.min(np.linalg.norm(endpoints-ref['tip'],axis=1))<=POLICY['tip_tolerance_px'])
    strokes=[]
    for row in reference['strokes']+reference['gaps']:
        result=stroke_coverage({**row,'state':'visible','band_px':3},candidates,matrix)
        result['assigned_kind']='gap' if row in reference['gaps'] else 'visible_dash'
        result['correct_dash_style_claimed']=False
        strokes.append(result)
    predicted_tips=[(a['id'],transform_points([a['tip_px']],matrix)[0]) for a in arrows['candidates']]
    negatives=[]
    for row in reference['negatives']:
        x,y,right,bottom=row['box']
        ids=[name for name,(px,py) in predicted_tips if x<=px<right and y<=py<bottom]
        negatives.append({'id':row['id'],'reason':row['role'],'arrow_tip_candidate_ids':ids,'false_arrow_tip_count':len(ids)})
    curves=curve_scores(reference['curves'],candidates,matrix)
    return {'arrows':arrow_rows,'strokes_and_gaps':strokes,'open_curves':curves,'negative_regions':negatives,
        'assigned_arrow_tips':len(arrow_rows),'matched_directed_arrows':sum(r['matched'] for r in arrow_rows),
        'line_endpoints_near_tips':sum(r['nearby_line_endpoint_only'] for r in arrow_rows),
        'visible_dash_guides':len(reference['strokes']),
        'covered_dash_guides':sum(r['covered'] for r in strokes if r['assigned_kind']=='visible_dash'),
        'gap_guides':len(reference['gaps']),'bridged_gap_guides':sum(r['covered'] for r in strokes if r['assigned_kind']=='gap'),
        'assigned_open_arcs':len(curves),'qualified_open_arcs':sum(r['qualified_open_arc'] for r in curves),
        'false_arrows_in_assigned_negative_regions':sum(r['false_arrow_tip_count'] for r in negatives),
        'full_arrow_precision':None,'full_geometric_precision':None,'CAD_promotions':0}
