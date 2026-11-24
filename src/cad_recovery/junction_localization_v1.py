"""Source-core contraction and line/circle position views; no raster editing.

Nearby nodes are never merged merely by distance. A common observed critical
region proposes a contraction; separate source branches must support one center.
Raw observations and every contributing pixel remain in the returned evidence.
"""
from copy import deepcopy
import itertools
import math
import cv2
import numpy as np
from .junction_topology_v1 import adjacency_bits,OFFSETS,topology
from .junction_topology_v2 import trace_branches
from .geometry_detection import _fit_circle
from .raster_junctions import _circular_runs

POLICY={'version':'junction-localization/1.0','maximum_critical_pixels':4096,'maximum_nodes':512,
    'maximum_trace_steps':200000,'maximum_source_samples':8000000,'maximum_condition_number':6.,
    'maximum_fit_p95_px':1.5,'minimum_source_coverage':.95,'production_admitted':False,
    'physical_geometry_approved':False,'source_erasure_allowed':False}


def source_centers(path,ink,dt,audit):
    along=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
    pixels=np.rint(path).astype(int);pen=max(1.,float(np.median(2*dt[pixels[:,1],pixels[:,0]]-1)))
    low=max(1.5,min(1.5*pen,.2*along[-1]));high=min(64.,max(16.,8*pen),.92*along[-1]);points=[]
    for index in np.flatnonzero((along>=low)&(along<=high)):
        a=max(0,index-3);b=min(len(path),index+4);local=path[a:b]
        if len(local)<3:continue
        _,_,vectors=np.linalg.svd(local-local.mean(0),full_matrices=False);unit=vectors[0];normal=np.array([-unit[1],unit[0]])
        offsets=np.arange(-max(3.,1.5*pen),max(3.,1.5*pen)+.25,.5)
        audit['source_samples']+=len(offsets)
        if audit['source_samples']>POLICY['maximum_source_samples']:
            audit['limits'].append('source_samples');return np.empty((0,2)),pen
        q=np.rint(path[index]+offsets[:,None]*normal).astype(int)
        if (q<0).any() or (q[:,0]>=ink.shape[1]).any() or (q[:,1]>=ink.shape[0]).any():continue
        dark=ink[q[:,1],q[:,0]]>0;changes=np.diff(np.r_[False,dark,False].astype(int))
        runs=list(zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1)))
        if len(runs)!=1:continue
        start,end=runs[0]
        if start==0 or end==len(offsets):continue
        center=(offsets[start]+offsets[end-1])/2
        if abs(center)>pen:continue
        points.append(path[index]+center*normal)
    return np.asarray(points).reshape(-1,2),pen


def fitted_line(points,seed):
    if len(points)<3 or np.linalg.norm(np.ptp(points,axis=0))<2:return None
    center=points.mean(0);_,_,vectors=np.linalg.svd(points-center,full_matrices=False);unit=vectors[0]
    if unit@(center-seed)<0:unit=-unit
    normal=np.array([-unit[1],unit[0]]);errors=np.abs((points-center)@normal)
    if np.percentile(errors,95)>1.5:return None
    return {'center':center,'unit':unit,'normal':normal,'rms':float(np.sqrt(np.mean(errors**2)))}


def source_support(ink,points,audit):
    audit['source_samples']+=len(points)
    if audit['source_samples']>POLICY['maximum_source_samples']:
        audit['limits'].append('source_samples');return False
    q=np.rint(points).astype(int)
    if (q<0).any() or (q[:,0]>=ink.shape[1]).any() or (q[:,1]>=ink.shape[0]).any():return False
    dark=ink[q[:,1],q[:,0]]>0;changes=np.diff(np.r_[False,~dark,False].astype(int))
    gaps=np.flatnonzero(changes==-1)-np.flatnonzero(changes==1)
    return bool(dark.mean()>=.95 and max(gaps,default=0)*.5<=1.)


def ring_evidence(ink,point,models,pen,available,audit,strict):
    maximum=min(2*pen+9,.85*min(available));minimum=max(3.,1.25*pen)
    if maximum<minimum+1:
        return (not strict),{'status':'source_graph_only_short_context','strict_merge':strict}
    radii=[max(minimum,.75*maximum),maximum];reports=[]
    for radius in radii:
        angles=np.linspace(0,2*np.pi,720,endpoint=False);radial=np.c_[np.cos(angles),np.sin(angles)]
        q=np.rint(point+radius*radial).astype(int);audit['source_samples']+=len(q)
        if audit['source_samples']>POLICY['maximum_source_samples']:
            audit['limits'].append('source_samples');return False,{'status':'sample_budget'}
        if (q<0).any() or (q[:,0]>=ink.shape[1]).any() or (q[:,1]>=ink.shape[0]).any():return False,{'status':'clipped_ring'}
        runs=_circular_runs(ink[q[:,1],q[:,0]]>0);expected=[]
        for model in models:
            if model['kind']=='line':expected.append(model['unit'])
            else:
                outward=(point-model['center'])/model['radius'];tangent=model['sign']*np.array([-outward[1],outward[0]])
                if radius>=2*model['radius']:return False,{'status':'unresolved_curve_ring'}
                shift=-radius**2/(2*model['radius'])*outward
                vector=shift+radius*math.sqrt(1-radius**2/(4*model['radius']**2))*tangent
                expected.append(vector/np.linalg.norm(vector))
        tolerance=math.radians(12)+math.asin(min(1.,math.sqrt(.5)/radius));matches=[]
        for run in runs:
            direction=radial[run].mean(0);norm=np.linalg.norm(direction)
            if norm<=0:return False,{'status':'ambiguous_ring'}
            options=[i for i,d in enumerate(expected) if direction@d/norm>=math.cos(tolerance)]
            matches.append(options)
        good=len(runs)==len(models) and all(len(m)==1 for m in matches) and len({m[0] for m in matches if m})==len(models)
        reports.append({'radius_px':radius,'run_count':len(runs),'matches':matches,'supported':good})
        if not good:return False,{'status':'unresolved_source_topology','rings':reports}
    return True,{'status':'observed_source_topology','rings':reports}


def locate(node,packet,bits,dt,audit,strict=False):
    point=np.asarray(node['point_px'],float);paths=trace_branches(node,bits,audit);features=[]
    for path in paths:
        if len(path)<2:return deepcopy(node),{'status':'insufficient_observed_path'}
        points,pen=source_centers(path,packet['ink'],dt,audit)
        features.append({'path':path,'points':points,'pen':pen,'line':fitted_line(points,point)})
    if len(features)!=node['degree'] or audit['limits']:return deepcopy(node),{'status':'unresolved_branch_count'}
    candidates=[];valid=[f for f in features if f['line'] is not None]
    if len(valid)>=2 and (not strict or len(valid)==len(features)):
        normals=np.array([f['line']['normal'] for f in valid]);centers=np.array([f['line']['center'] for f in valid])
        singular=np.linalg.svd(normals,compute_uv=False)
        if singular[-1]>1e-12 and singular[0]/singular[-1]<=6:
            fitted=np.linalg.lstsq(normals,np.sum(normals*centers,axis=1),rcond=None)[0]
            residual=np.abs(np.sum(normals*(fitted-centers),axis=1))
            if residual.max()<=1.5:
                models=[]
                for f in features:
                    direction=f['line']['unit'] if f['line'] is not None else (f['path'][-1]-fitted)/np.linalg.norm(f['path'][-1]-fitted)
                    models.append({'kind':'line','unit':direction})
                candidates.append((float(np.mean([f['line']['rms']**2 for f in valid])+np.mean(residual**2)),fitted,models,'lines'))
    if len(features)==3 and not strict:
        for first,second in itertools.combinations(range(3),2):
            other=3-first-second;a,b,c=features[first],features[second],features[other]
            if len(a['points'])<6 or len(b['points'])<6 or c['line'] is None:continue
            points=np.r_[a['points'],b['points']];fit=_fit_circle(points)
            if fit is None:continue
            center,radius,errors=fit;pen=max(a['pen'],b['pen'])
            if not 3*pen<=radius<=100000 or np.percentile(errors,95)>1.5:continue
            # Curvature must be observed: straight opposed rays cannot acquire
            # an arbitrary huge fitted circle with an indistinguishable sagitta.
            spread=np.linalg.norm(np.ptp(points,axis=0))
            if spread>=2*radius or radius-math.sqrt(max(0.,radius**2-spread**2/4))<max(1.,.25*pen):continue
            line=c['line'];delta=line['center']-center;projection=float(delta@line['unit'])
            discriminant=projection**2-(delta@delta-radius**2)
            if discriminant<0:continue
            solutions=[line['center']+(-projection+sign*math.sqrt(discriminant))*line['unit'] for sign in (-1,1)]
            near=[p for p in solutions if np.linalg.norm(p-point)<=max(8.,2*max(f['pen'] for f in features))]
            if len(near)!=1:continue
            fitted=near[0];outward=(fitted-center)/radius;tangent=np.array([-outward[1],outward[0]]);models=[]
            for i,f in enumerate(features):
                if i==other:models.append({'kind':'line','unit':line['unit']})
                else:
                    direction=f['path'][min(5,len(f['path'])-1)]-fitted
                    sign=1. if direction@tangent>=0 else -1.
                    models.append({'kind':'circle','center':center,'radius':radius,'sign':sign})
            if models[first]['sign']==models[second]['sign']:continue
            candidates.append((float(np.mean(errors**2)+line['rms']**2),fitted,models,'circle_line'))
    rejected=[]
    for cost,fitted,models,kind in sorted(candidates,key=lambda c:c[0]):
        pen=max(f['pen'] for f in features)
        if np.linalg.norm(fitted-point)>max(8.,2*pen):continue
        available=[float(np.linalg.norm(f['path'][-1]-fitted)) for f in features];supported=True
        for f,model in zip(features,models):
            extent=min(max(4.,2*f['pen']),float(np.linalg.norm(f['path'][-1]-fitted)))
            along=np.linspace(0,extent,max(3,math.ceil(extent/.5)+1))
            if model['kind']=='line':q=fitted+along[:,None]*model['unit']
            else:
                angle=math.atan2(*(fitted-model['center'])[::-1])+model['sign']*along/model['radius']
                q=model['center']+model['radius']*np.c_[np.cos(angle),np.sin(angle)]
            if not source_support(packet['ink'],q,audit):supported=False;break
        if not supported:rejected.append({'model':kind,'reason':'source_gap'});continue
        good,rings=ring_evidence(packet['ink'],fitted,models,pen,available,audit,strict)
        if not good:rejected.append({'model':kind,'reason':rings['status']});continue
        result=deepcopy(node);result['topological_point_px']=node['point_px'];result['point_px']=fitted.tolist()
        result['position_evidence']='source_'+kind
        return result,{'status':'localized','model':kind,'source_cost':cost,'rings':rings,
            'raw_point_px':node['point_px'],'point_px':result['point_px'],'fitted_branch_count':len(valid),'rejected':rejected}
    return deepcopy(node),{'status':'unchanged_source_ambiguous','rejected':rejected}


def core_proposals(packet,bits):
    binary=(packet['skeleton']>0).astype(np.uint8)
    degree=cv2.filter2D(binary,cv2.CV_16S,np.ones((3,3),np.int16),borderType=cv2.BORDER_CONSTANT)-binary
    critical=(binary>0)&(degree>=3)
    if int(critical.sum())>4096:return [],{'complete':False,'reason':'critical_pixel_budget'}
    count,labels,_,_=cv2.connectedComponentsWithStats(critical.astype(np.uint8),connectivity=8)
    old=packet['topological_nodes'];proposals=[];decisions=[]
    for label in range(1,count):
        members={tuple(map(int,p)) for p in np.argwhere(labels==label)}
        expanded=set(members)
        for y,x in members:
            for dy,dx in ((-1,-1),(-1,1),(1,-1),(1,1)):
                if (y+dy,x+dx) not in members:continue
                for p in ((y+dy,x),(y,x+dx)):
                    if binary[p]:expanded.add(p)
        owned=[n for n in old if n['kind']=='junction' and any(tuple(map(int,p[::-1])) in expanded for p in n['source_cluster_px'])]
        if not owned:continue
        boundary=internal=0
        for y,x in expanded:
            for bit,(dy,dx) in enumerate(OFFSETS):
                if int(bits[y,x])&(1<<bit):
                    if (y+dy,x+dx) in expanded:internal+=1
                    else:boundary+=1
        pixels=np.array([[x,y] for y,x in sorted(expanded)],float)
        if boundary<=2:
            lo=pixels.min(0).astype(int);hi=pixels.max(0).astype(int)+1
            hole=topology(packet['ink'][lo[1]:hi[1],lo[0]:hi[0]])['holes']
            if hole:proposals+=deepcopy(owned)
            decisions.append({'raw_ids':[n['id'] for n in owned],'status':'unresolved_cyclic_core' if hole else 'two_port_source_core',
                'source_cluster_px':pixels.tolist(),'external_ports':boundary});continue
        point=pixels.mean(0).tolist() if len(owned)>1 else owned[0]['point_px']
        proposals.append({'id':'core-'+str(len(proposals)),'point_px':point,'kind':'junction','degree':boundary,
            'source_cluster_px':pixels.tolist(),'raw_node_ids':[n['id'] for n in owned],'raw_nodes':deepcopy(owned),
            'boundary_edge_count':boundary,'internal_cycle_rank':max(0,internal//2-len(expanded)+1)})
    proposals+=deepcopy([n for n in old if n['kind']!='junction'])
    return proposals,{'complete':True,'core_decisions':decisions}


def refine_packet(packet):
    raw=deepcopy(packet['nodes']);base=packet.get('topological_nodes',packet['nodes'])
    previous={n['id']:n for n in raw}
    if not packet['complete']:
        return {'raw_observations':raw,'core_only':raw,'pose_only':raw,'combined':raw,'complete':False,'audit':{'limits':['previous_incomplete']}}
    bits=adjacency_bits(packet['skeleton']);dt=cv2.distanceTransform(packet['ink'],cv2.DIST_L2,5)
    audit={'trace_steps':0,'source_samples':0,'limits':[]};poses=[];core=[];combined=[];decisions=[]
    for node in base:
        result,evidence=locate(node,packet,bits,dt,audit) if node['kind']=='junction' else (deepcopy(node),{'status':'endpoint_retained'})
        if evidence['status']!='localized':result=deepcopy(previous.get(node['id'],result))
        poses.append(result);decisions.append({'arm':'pose_only','raw_id':node['id'],**evidence})
    proposals,core_audit=core_proposals(packet,bits)
    for node in proposals:
        strict=len(node.get('raw_node_ids',[]))>1
        result,evidence=locate(node,packet,bits,dt,audit,strict) if node['kind']=='junction' else (deepcopy(node),{'status':'endpoint_retained'})
        if strict and evidence['status']!='localized':
            originals=node['raw_nodes'];core+=deepcopy(originals)
            combined += [next((p for p in poses if p['id']==n['id']),deepcopy(n)) for n in originals]
            decisions.append({'arm':'combined','status':'source_merge_rejected','raw_ids':node['raw_node_ids'],'fit':evidence})
        else:
            if evidence['status']!='localized' and len(node.get('raw_node_ids',[]))==1:
                result['point_px']=deepcopy(previous[node['raw_node_ids'][0]]['point_px'])
                result['position_evidence']='previous_position_retained_without_new_source_proof'
            core.append(deepcopy(node));combined.append(result)
            decisions.append({'arm':'combined','raw_ids':node.get('raw_node_ids',[node['id']]),**evidence})
    def renumber(nodes):
        output=deepcopy(sorted(nodes,key=lambda n:(n['point_px'][1],n['point_px'][0],n['kind'])))
        for i,n in enumerate(output):n['raw_view_id']=n['id'];n['id']='node-'+str(i)
        return output
    complete=core_audit['complete'] and not audit['limits'] and max(len(core),len(combined),len(poses))<=512
    if not complete:core=combined=poses=deepcopy(raw)
    return {'raw_observations':raw,'core_only':renumber(core),'pose_only':renumber(poses),'combined':renumber(combined),
        'complete':complete,'audit':{**audit,'cores':core_audit},'decisions':decisions,
        'source_erased_pixels':0,'production_admitted':False,'physical_geometry_approved':False}
