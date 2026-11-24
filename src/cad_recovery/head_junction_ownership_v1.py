"""Abstain from new extension paint on source-supported shaft/head contacts."""
from collections import defaultdict
import math

import cv2
import numpy as np

from .stroke_roles_v1 import checked_gray,foreground

POLICY={'version':'head-junction-ownership/1.0','maximum_source_points':2000000,
        'maximum_polygon_checks':2000000,'maximum_combined_raster_domains':40000000,
        'maximum_heads':512,'maximum_intervals':4096,'maximum_chains':4096,
        'pixel_footprint_radius_px':2**.5/2,'direction_agreement_deg':20.,
        'affected_new_role':'extension_line','withdrawal_only':True,
        'observations_immutable':True,'production_admitted':False,'physical_geometry_approved':False}


def apply(gray,previous,inherited_roles,packet,heads):
    checked_gray(gray)
    labels=packet['pixel_roles'];owners=packet['pixel_interval_owner']
    if labels.shape!=gray.shape or inherited_roles.shape!=gray.shape or owners.shape!=gray.shape:
        raise ValueError('Aligned role/owner maps required')
    if labels.dtype!=np.uint8 or inherited_roles.dtype!=np.uint8:raise ValueError('Uint8 roles required')
    if owners.dtype.kind not in 'ui':raise ValueError('Integer source owner map required')
    known=inherited_roles!=255
    if not np.array_equal(labels[known],inherited_roles[known]):raise ValueError('Candidate changed inherited roles')
    intervals=previous['proposals'];chains=packet['dimension_chains']
    if len(intervals)>4096 or len(chains)>4096 or len(heads)>512:raise ValueError('Object budget')
    lookup={p['id']:p for p in intervals};head_lookup={h['id']:h for h in heads}
    if len(lookup)!=len(intervals) or len(head_lookup)!=len(heads):raise ValueError('Unique provenance IDs required')
    output=labels.copy();assigned=owners.copy();withdrawn=np.zeros(gray.shape,np.uint8)
    selected=(labels==3)&(inherited_roles==255)
    if np.any(selected&((owners<=0)|(owners>len(intervals)))):raise ValueError('New extension pixels need a source owner')
    inherited_domains=packet.get('contact_audit',{}).get('combined_raster_pixels',0)+packet.get('boundary_competitor_audit',{}).get('combined_raster_domain_pixels',0)
    audit={'source_points':0,'polygon_checks':0,'inherited_raster_domains':int(inherited_domains),
           'combined_raster_domains':int(inherited_domains+gray.size),'contexts':0,'removed_pixels':0,'limits':[]}
    def limit(reason):
        if reason not in audit['limits']:audit['limits'].append(reason)
    if audit['combined_raster_domains']>40000000:limit('junction_raster_domain_budget')
    distance=None if audit['limits'] else cv2.distanceTransform(foreground(gray),cv2.DIST_L2,5)
    tracks={t['id']:t for t in previous['tracks']};contexts=defaultdict(list);cache={};records=[]
    # Group changed pixels once. Do not scan the full image once per interval.
    y,x=np.nonzero(selected);numbers=owners[y,x];order=np.argsort(numbers,kind='stable')
    y,x,numbers=y[order],x[order],numbers[order]
    boundaries=np.r_[0,np.flatnonzero(np.diff(numbers))+1,len(numbers)]
    groups={int(numbers[a]):(y[a:b],x[a:b]) for a,b in zip(boundaries,boundaries[1:]) if b>a}
    for number,chain in enumerate(chains):
        for group in chain['extension_interval_ids']:
            for key in group:contexts[key].append((number,chain))
    for owner,(yy,xx) in groups.items():
        interval=intervals[owner-1];claims=contexts[interval['id']]
        points=np.column_stack((xx,yy));conflict=np.zeros(len(points),bool);evidence=[]
        if not claims:
            conflict[:]=True
        for chain_index,chain in claims:
            if audit['limits']:break
            key=tuple(sorted(chain['shaft_interval_ids']))
            if key not in cache:
                samples=[]
                for identifier in key:
                    p=np.asarray(lookup[identifier]['points_px'],float)
                    if p.ndim!=2 or p.shape[1]!=2 or not np.isfinite(p).all():raise ValueError('Finite source shaft required')
                    audit['source_points']+=len(p)
                    if audit['source_points']>2000000:limit('junction_source_point_budget');break
                    q=np.rint(p).astype(int)
                    if (q<0).any() or (q[:,0]>=gray.shape[1]).any() or (q[:,1]>=gray.shape[0]).any():raise ValueError('Source shaft outside image')
                    values=distance[q[:,1],q[:,0]];samples.extend(values[values>0].tolist())
                cache[key]=float(np.quantile(samples,.25)) if len(samples)>=3 and not audit['limits'] else None
            radius=cache[key];track=tracks.get(chain['shaft_track'])
            if radius is None or track is None:
                conflict[:]=True;continue
            axis=np.asarray(track['points'],float);axis=axis[1]-axis[0];norm=float(np.linalg.norm(axis))
            if norm<=0 or not np.isfinite(axis).all():raise ValueError('Finite nonzero shaft axis required')
            axis/=norm
            for identity in chain['head_ids']:
                if audit['limits']:break
                if identity not in head_lookup:
                    conflict[:]=True;continue
                head=head_lookup[identity];polygon=np.asarray(head['polygon_px'],float);direction=np.asarray(head['direction'],float)
                uncertainty=float(head.get('uncertainty_px',0.))
                if polygon.shape!=(3,2) or direction.shape!=(2,) or not np.isfinite(polygon).all() or not np.isfinite(direction).all() or not math.isfinite(uncertainty) or uncertainty<0:
                    raise ValueError('Finite observed head/uncertainty required')
                dnorm=float(np.linalg.norm(direction))
                if dnorm<=0:raise ValueError('Nonzero head direction required')
                if abs(axis@(direction/dnorm))<math.cos(math.radians(20)):
                    conflict[:]=True;continue
                tip=polygon[0];base=polygon[1:].mean(0);delta=base-tip;length2=float(delta@delta)
                if length2<=1e-12:raise ValueError('Nonzero observed head length required')
                band=radius+uncertainty+2**.5/2
                near=((points>=polygon.min(0)-band)&(points<=polygon.max(0)+band)).all(1)
                indices=np.flatnonzero(near)
                if audit['polygon_checks']+len(indices)>2000000:limit('junction_polygon_budget');break
                audit['polygon_checks']+=len(indices);audit['contexts']+=1
                if len(indices):
                    p=points[indices];along=np.clip((p-tip)@delta/length2,0.,1.)
                    projected=tip+along[:,None]*delta
                    shaft_support=np.linalg.norm(p-projected,axis=1)<=band
                    poly=polygon.astype(np.float32)
                    halo=np.array([cv2.pointPolygonTest(poly,(float(v[0]),float(v[1])),True)>=-band for v in p])
                    conflict[indices]|=shaft_support&halo
                evidence.append({'chain_index':chain_index,'head_id':identity,'shaft_interval_ids':list(key),
                                 'observed_shaft_radius_px':radius,'head_uncertainty_px':uncertainty,
                                 'ambiguity_extent_px':band})
        if audit['limits']:break
        output[yy[conflict],xx[conflict]]=255;assigned[yy[conflict],xx[conflict]]=0;withdrawn[yy[conflict],xx[conflict]]=255
        count=int(conflict.sum());audit['removed_pixels']+=count
        records.append({'source_interval_id':interval['id'],'tested_pixels':len(points),'removed_pixels':count,
                        'reason':'source_supported_shaft_head_competes_with_extension' if claims else 'missing_verified_chain_context',
                        'contexts':evidence})
    if audit['limits']:
        # An unchecked extra guard grants no new roles. Preserve the complete
        # inherited map and flag the incomplete evaluation for the caller.
        output[:]=inherited_roles;assigned[:]=0;withdrawn[:]=np.where(labels!=inherited_roles,255,0)
        audit['removed_pixels']=int(np.count_nonzero(labels!=inherited_roles))
    return {**packet,'schema':POLICY['version'],'pixel_roles':output,'pixel_interval_owner':assigned,
            'junction_abstention_mask':withdrawn,'junction_ownership_audit':audit,'junction_decisions':records,
            'unguarded_added_pixels':int(np.count_nonzero(labels!=inherited_roles)),
            'added_pixels':int(np.count_nonzero(output!=inherited_roles)),
            'search_incomplete':packet['search_incomplete'] or bool(audit['limits']),
            'new_role_changes_are_withdrawals':bool(np.all((output==labels)|(output==inherited_roles))),
            'production_admitted':False,'physical_geometry_approved':False}
