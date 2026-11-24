"""Bounded dash-dot proposals from real ink components, without joining gaps."""
from collections import Counter
import math

import cv2
import numpy as np

from .family_catalog_v2 import require

POLICY = {
    'version': 'component-dash-dot-search/1.0',
    'profiles': ['otsu_components', 'local_contrast_components'],
    'background_kernel_px': 31, 'minimum_local_contrast': 12, 'maximum_ink_gray': 245,
    'maximum_pixels': 25000000, 'maximum_components_examined': 50000,
    'maximum_component_points': 300000, 'maximum_linear_components': 512,
    'minimum_component_area': 6, 'maximum_component_area': 4096,
    'minimum_length_px': 8, 'maximum_length_px': 256, 'maximum_width_px': 16,
    'minimum_elongation': 3, 'minimum_rectangular_fill': .55,
    'minimum_component_centerline_support': .9,
    'maximum_direction_error_deg': 3, 'minimum_gap_px': 2,
    'maximum_gap_in_shaft_widths': 6, 'minimum_maximum_gap_px': 12,
    'minimum_long_short_ratio': 2.5, 'minimum_pattern_members': 4,
    'maximum_gap_coefficient_of_variation': .35, 'maximum_groups': 64,
    'continuous_axis_inferred': False, 'physical_geometry_approved': False,
    'production_admitted': False, 'manual_guides_used_for_search': False,
}


def foreground(gray, profile):
    require(profile in POLICY['profiles'], 'Unknown dash-search profile')
    require(isinstance(gray,np.ndarray) and gray.ndim == 2 and gray.dtype == np.uint8
            and 0 < gray.size <= POLICY['maximum_pixels'], 'Expected bounded uint8 grayscale raster')
    if profile == 'otsu_components':
        threshold,mask = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
        return mask, {'method':'Otsu', 'threshold':float(threshold)}
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (POLICY['background_kernel_px'],)*2)
    background = cv2.morphologyEx(gray,cv2.MORPH_CLOSE,kernel)
    contrast = cv2.subtract(background,gray)
    mask = ((contrast >= POLICY['minimum_local_contrast']) &
            (gray <= POLICY['maximum_ink_gray'])).astype(np.uint8)*255
    return mask, {'method':'local_closing_background_minus_original',
                  'minimum_contrast':POLICY['minimum_local_contrast'],
                  'foreground_morphology':'none; closing estimates background only'}


def linear_components(mask):
    count,labels,stats,_ = cv2.connectedComponentsWithStats(mask,connectivity=8)
    counters = Counter();rows=[];point_count=0;limited=[]
    for identity in range(1,count):
        if counters['components_examined'] >= POLICY['maximum_components_examined']:
            limited.append('component_count_budget');break
        counters['components_examined'] += 1
        x,y,width,height,area = (int(v) for v in stats[identity])
        if not POLICY['minimum_component_area'] <= area <= POLICY['maximum_component_area']:
            counters['area_rejected'] += 1;continue
        if max(width,height) > POLICY['maximum_length_px']+POLICY['maximum_width_px']:
            counters['bounding_extent_rejected'] += 1;continue
        if point_count+area > POLICY['maximum_component_points']:
            limited.append('component_point_budget');break
        point_count += area
        yy,xx = np.nonzero(labels[y:y+height,x:x+width] == identity)
        points = np.c_[xx+x,yy+y].astype(float)
        center = points.mean(axis=0)
        values,vectors = np.linalg.eigh((points-center).T@(points-center)/len(points))
        unit = vectors[:,-1]
        if unit[0] < -1e-12 or (abs(unit[0]) <= 1e-12 and unit[1] < 0):unit=-unit
        normal = np.array([-unit[1],unit[0]])
        along = (points-center)@unit;across=(points-center)@normal
        length = float(np.ptp(along)+1);thickness=float(np.ptp(across)+1)
        elongation = math.sqrt(float(values[-1])/max(float(values[0]),.01))
        if (not POLICY['minimum_length_px'] <= length <= POLICY['maximum_length_px']
                or thickness > POLICY['maximum_width_px'] or elongation < POLICY['minimum_elongation']):
            counters['shape_rejected'] += 1;continue
        fill = area/(length*thickness)
        if fill < POLICY['minimum_rectangular_fill']:
            counters['irregular_shape_rejected'] += 1;continue
        # Finite endpoints stay inside the observed component extent, never the gap.
        endpoints = np.array([center+unit*along.min(),center+unit*along.max()])
        samples=np.rint(endpoints[0]+np.linspace(0,1,max(2,math.ceil(length*2)))[:,None]*
                        (endpoints[1]-endpoints[0])).astype(int)
        valid=((samples>=0).all(axis=1)&(samples[:,0]<mask.shape[1])&(samples[:,1]<mask.shape[0]))
        supported=np.zeros(len(samples),bool)
        supported[valid]=labels[samples[valid,1],samples[valid,0]]==identity
        support=float(supported.mean())
        if support<POLICY['minimum_component_centerline_support']:
            counters['centerline_support_rejected']+=1;continue
        rows.append({'component_id':identity,'points_px':endpoints.tolist(),'center':center.tolist(),
                     'unit':unit.tolist(),'length_px':length,'width_px':thickness,
                     'area':area,'rectangular_fill':fill,'elongation':elongation,
                     'centerline_own_component_support':support})
    rows.sort(key=lambda r:(-r['length_px'],r['component_id']))
    if len(rows) > POLICY['maximum_linear_components']:
        limited.append('linear_component_budget')
        rows=rows[:POLICY['maximum_linear_components']]
    return rows, {'connected_components':count-1,'component_points_examined':point_count,
                  **dict(counters),'linear_components_retained':len(rows),'limits':limited}


def pattern_groups(rows):
    size=len(rows)
    parent=list(range(size))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(i,j):
        a,b=find(i),find(j)
        if a!=b:parent[max(a,b)]=min(a,b)
    comparisons=0
    for i,left in enumerate(rows):
        unit=np.asarray(left['unit']);normal=np.array([-unit[1],unit[0]])
        a=np.asarray(left['points_px'])@unit
        for j in range(i+1,size):
            comparisons+=1;right=rows[j]
            if abs(unit@np.asarray(right['unit'])) < math.cos(math.radians(POLICY['maximum_direction_error_deg'])):continue
            delta=np.asarray(right['center'])-left['center']
            tolerance=max(2.,(left['width_px']+right['width_px'])/2)
            if abs(delta@normal)>tolerance:continue
            b=np.asarray(right['points_px'])@unit
            gap=max(float(a.min()-b.max()),float(b.min()-a.max()))
            maximum=max(POLICY['minimum_maximum_gap_px'],
                        POLICY['maximum_gap_in_shaft_widths']*min(left['width_px'],right['width_px']))
            if POLICY['minimum_gap_px']<=gap<=maximum:union(i,j)
    clusters={}
    for i in range(size):clusters.setdefault(find(i),[]).append(i)
    groups=[];reasons=Counter()
    for members in clusters.values():
        if len(members)<POLICY['minimum_pattern_members']:
            reasons['too_few_collinear_members']+=1;continue
        unit=np.asarray(rows[members[0]]['unit'])
        normal=np.array([-unit[1],unit[0]])
        centers=np.asarray([rows[i]['center'] for i in members])
        if (np.max(np.abs((centers-centers[0])@normal))>max(2.,np.median([rows[i]['width_px'] for i in members])/2)
                or any(abs(np.asarray(rows[i]['unit'])@unit)<math.cos(math.radians(POLICY['maximum_direction_error_deg'])) for i in members)):
            reasons['global_axis_drift']+=1;continue
        members.sort(key=lambda i:float(np.asarray(rows[i]['center'])@unit))
        lengths=sorted(rows[i]['length_px'] for i in members)
        ratios=[lengths[i+1]/lengths[i] for i in range(len(lengths)-1)]
        split=int(np.argmax(ratios))
        if ratios[split]<POLICY['minimum_long_short_ratio']:
            reasons['no_long_short_separation']+=1;continue
        threshold=math.sqrt(lengths[split]*lengths[split+1])
        kinds=['short_dash' if rows[i]['length_px']<threshold else 'long_dash' for i in members]
        # Split at repeated kinds; only maximal alternating chains can qualify.
        cuts=[0]+[i for i in range(1,len(kinds)) if kinds[i]==kinds[i-1]]+[len(kinds)]
        for begin,end in zip(cuts,cuts[1:]):
            selected=members[begin:end];roles=kinds[begin:end]
            if len(selected)<4 or min(roles.count('short_dash'),roles.count('long_dash'))<2:
                reasons['insufficient_alternation']+=1;continue
            gaps=[]
            for i,j in zip(selected,selected[1:]):
                gaps.append(float((np.asarray(rows[j]['points_px'])@unit).min()-
                                  (np.asarray(rows[i]['points_px'])@unit).max()))
            maximum=max(POLICY['minimum_maximum_gap_px'],
                        POLICY['maximum_gap_in_shaft_widths']*min(rows[i]['width_px'] for i in selected))
            if min(gaps)<POLICY['minimum_gap_px'] or max(gaps)>maximum:
                reasons['overlap_or_excessive_gap']+=1;continue
            variation=float(np.std(gaps)/np.mean(gaps))
            if variation>POLICY['maximum_gap_coefficient_of_variation']:
                reasons['irregular_spacing']+=1;continue
            groups.append({'member_indices':selected,'member_roles':roles,'gaps_px':gaps,
                           'gap_coefficient_of_variation':variation,
                           'continuous_axis_inferred':False,'classification':'dash_dot_pattern_candidate'})
    limited=len(groups)>POLICY['maximum_groups']
    groups=groups[:POLICY['maximum_groups']]
    return groups, {'pair_comparisons':comparisons,'cluster_rejections':dict(reasons),
                    'group_limit_reached':limited,'groups_retained':len(groups)}


def search(gray, *, profile):
    mask,threshold=foreground(gray,profile)
    rows,components=linear_components(mask)
    groups,group_stats=pattern_groups(rows)
    candidates=[]
    for index,group in enumerate(groups,1):
        group['id']=f'axis-pattern-{index:04d}'
        for member,role in zip(group['member_indices'],group['member_roles']):
            row=rows[member]
            candidates.append({'id':f'axis-dash-{len(candidates)+1:05d}','kind':'line',
                'points_px':row['points_px'],'classification':'axis_dash_candidate',
                'uncertainty_px':max(.75,row['width_px']/2),
                'evidence':{'method':POLICY['version'],'profile':profile,
                    'component_id':row['component_id'],'group_id':group['id'],'member_role':role,
                    'length_px':row['length_px'],'width_px':row['width_px'],
                    'centerline_own_component_support':row['centerline_own_component_support'],
                    'observed_ink_only':True,'continuous_axis_inferred':False}})
    return {'schema_version':'axis-dash-proposals/1.0','profile':profile,'policy':POLICY,
        'source_size_px':[gray.shape[1],gray.shape[0]],'threshold':threshold,
        'raw_linear_components':rows,'groups':groups,'candidates':candidates,
        'analysis':{**components,**group_stats},
        'search_incomplete':bool(components['limits'] or group_stats['group_limit_reached']),
        'production_admitted':False,'physical_geometry_approved':False,'CAD_promotions':0}
