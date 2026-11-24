"""Additive P12.4c.10 proposals from explicit text and observed pixel chains.

Frozen v3 remains the baseline. This experiment does not change raw OCR,
confidence, semantic view ownership, physical units or admitted service profiles.
"""
from collections import defaultdict
from copy import deepcopy
import math
import re

import cv2
import numpy as np
from PIL import Image

from . import dimension_association_v3 as v3
from .contour_topology_v3 import ContourTopology
from .dimension_chain_audit_v1 import _arrow_shape
from .notation import parse_notation

POLICY={
    'version':'dimension-support/1.0','confidence':.65,'angle_degrees':12,
    'maximum_texts':128,'maximum_axes':4,'maximum_leader_lines':64,
    'maximum_leader_segments':3,'maximum_alternatives':64,
    'maximum_pixels':12_000_000,'maximum_pixel_samples':250000,
    'maximum_segment_samples':4096,'line_ink_fraction':.85,'curve_ink_fraction':.75,
    'pixel_distance_px':3,'minimum_curve_samples':64,'maximum_curve_samples':512,
    'search_checks_per_notation':v3.MAX_CHECKS_PER_NOTATION,'search_checks_per_page':v3.MAX_CHECKS_PER_PAGE,
    'physical_geometry_approved':False,'production_admitted':False,
}
ARMS={
    'orientation':{'orientation'}, 'composite':{'composite'},
    'leaders':{'leaders'}, 'flanks':{'flanks'},
    'all':{'orientation','composite','leaders','flanks'},
}
COS=math.cos(math.radians(POLICY['angle_degrees']))
SIN=math.sin(math.radians(POLICY['angle_degrees']))


def parse_composite(raw):
    """Whole-string quantity grammar; never repair a digit into a symbol."""
    original=parse_notation(raw)
    if original['status']=='parsed' or len(raw)>512 or any(c in raw for c in '\r\n\v\f'):
        return original
    normalized=' '.join(raw.split())
    match=re.fullmatch(r'(?P<count>[1-9][0-9]{0,4})\s+(?:радиус|радиуса|радиусов|radii)\s+(?P<body>R.{1,96})',normalized)
    if not match or int(match['count'])>10000:return original
    body=parse_notation(match['body'])
    if body['status']!='parsed' or body['kind']!='radius':return original
    result=deepcopy(body)
    result.update(raw=raw,normalized=normalized,component_parse=body,
                  component_span_in_normalized=list(match.span('body')),
                  interpretation='one_observed_radial_relation_with_quantity; not count cloned dimensions')
    result['values']['count']=int(match['count'])
    result['issues'].append({'code':'quantity_scope_requires_geometry','message':'Quantity does not identify all repeated features or confirm units.'})
    return result


class PixelSupport:
    def __init__(self,path):
        with Image.open(path) as image:
            if image.width*image.height>POLICY['maximum_pixels']:raise ValueError('Pixel support budget exceeded')
            gray=np.array(image.convert('L'))
        _,white=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        self.distance=cv2.distanceTransform(white,cv2.DIST_L2,5)
        self.samples=0

    def points(self,points):
        a=np.asarray(points,dtype=float)
        if a.ndim!=2 or a.shape[1]!=2 or not np.isfinite(a).all():raise ValueError('Finite pixel samples required')
        if self.samples+len(a)>POLICY['maximum_pixel_samples']:raise v3._SearchLimit()
        self.samples+=len(a)
        q=np.rint(a).astype(int);h,w=self.distance.shape
        inside=(q[:,0]>=0)&(q[:,0]<w)&(q[:,1]>=0)&(q[:,1]<h)
        values=np.zeros(len(q),dtype=bool)
        values[inside]=self.distance[q[inside,1],q[inside,0]]<=POLICY['pixel_distance_px']
        return float(values.mean()) if len(values) else 0.

    def segment(self,a,b):
        length=float(np.linalg.norm(np.asarray(b)-a));n=max(2,math.ceil(length)+1)
        if n>POLICY['maximum_segment_samples']:raise v3._SearchLimit()
        return self.points(np.linspace(a,b,n))

    def curve(self,curve):
        sweep=curve.get('sweep_angle_deg',360);start=curve.get('start_angle_deg',0)
        n=min(POLICY['maximum_curve_samples'],max(POLICY['minimum_curve_samples'],math.ceil(curve['radius']*math.radians(sweep))))
        angle=np.radians(np.linspace(start,start+sweep,n,endpoint=sweep!=360))
        points=curve['center']+curve['radius']*np.c_[np.cos(angle),np.sin(angle)]
        return self.points(points)


def _axes(text,lines,budget):
    """Axes come from observed CV lines; old OCR aspect threshold stays intact."""
    axes=[];limit=False
    for line in v3._budgeted(sorted(lines,key=lambda a:a['id']),budget):
        if v3._segment_distance(text['center'],line)>max(text['width']*.6,text['height']*2.5+6):continue
        u=line['u'].copy()
        if u[0]<0 or (abs(u[0])<1e-8 and u[1]<0):u=-u
        if any(abs(float(np.dot(u,p)))>=COS for p in axes):continue
        if len(axes)==POLICY['maximum_axes']:limit=True;continue
        axes.append(u)
    return axes,limit


def _oriented(text,u):
    result=deepcopy(text);n=np.array([-u[1],u[0]])
    result.update(u=u,width=max(1.,float(np.ptp(text['polygon']@u))),
                  height=max(1.,float(np.ptp(text['polygon']@n))),orientation_uncertain=False)
    result['tol']=max(3.,min(8.,result['height']*.35))
    return result


def _add_equivalent(choices,choice):
    for previous in choices:
        if v3._same_candidate(previous,choice):
            paths=previous['evidence'].setdefault('corroborating_paths',[])
            ids=choice['dimension_line_candidate_ids']
            if ids not in paths:paths.append(deepcopy(ids))
            return
    choices.append(choice)


def _linear_candidates(text,lines,arrows,other_texts,budget,topology,*,orientation):
    axes,cut=(_axes(text,lines,budget) if text['orientation_uncertain'] and orientation else ([text['u']],False))
    if text['orientation_uncertain'] and not orientation:return [],False,{'orientation':'uncertain_unchanged'}
    choices=[];diagnostics={'orientation_originally_uncertain':text['orientation_uncertain'],'axes':[]}
    for axis in axes:
        oriented=_oriented(text,axis)
        found,truncated,details=v3._linear(oriented,lines,arrows,other_texts,budget,topology=topology)
        cut|=truncated;diagnostics['axes'].append({'axis':axis.tolist(),'diagnostics':details})
        for c in found:
            c=deepcopy(c)
            c['evidence']['orientation_source']='observed_arrow_line_extension_contour_chain'
            c['evidence']['source_ocr_polygon_unchanged']=True
            _add_equivalent(choices,c)
    return choices,cut,diagnostics


def _flank_candidates(text,lines,arrows,other_texts,budget,topology,pixels,*,orientation):
    """Extension and measured flank are collinear, unlike linear v3 anchors.

    Each pair belongs to one observed bounded face. Never choose a nearer face
    using the printed diameter; different faces remain different alternatives.
    """
    if text['orientation_uncertain'] and not orientation:return [],False,{'orientation':'uncertain_unchanged'}
    axes,cut=_axes(text,lines,budget) if text['orientation_uncertain'] else ([text['u']],False)
    choices=[];diagnostics={'opposed_arrow_pairs':0,'face_flank_pairs':0}
    shaft_ids={i for a in arrows for i in a.get('evidence',{}).get('geometry_ids',[])}
    for axis in axes:
        oriented=_oriented(text,axis)
        bases=[l for l in v3._budgeted(lines,budget) if abs(float(np.dot(l['u'],axis)))>=COS
               and v3._segment_distance(text['center'],l)<=max(text['width']*.6,text['height']*2.5+6)]
        cut|=len(bases)>v3.MAX_NEAR_LINES
        for base in sorted(bases,key=lambda l:l['id'])[:v3.MAX_NEAR_LINES]:
            members=[l for l in v3._budgeted(lines,budget) if abs(float(np.dot(l['u'],base['u'])))>=COS
                     and max(v3._perp(p,base) for p in l['points'])<=oriented['tol']]
            cut|=len(members)>v3.MAX_NEAR_LINES;members=members[:v3.MAX_NEAR_LINES]
            relevant=[a for a in arrows if abs(float(np.dot(a['u'],base['u'])))>=COS
                      and v3._perp(a['tip'],base)<=oriented['tol']
                      and any(v3._segment_distance(a['tip'],m)<=oriented['tol']*3 for m in members)]
            cut|=len(relevant)>v3.MAX_NEAR_ARROWS;relevant=relevant[:v3.MAX_NEAR_ARROWS]
            import itertools
            for first,second in v3._budgeted(itertools.combinations(relevant,2),budget):
                if float(np.dot(first['u'],second['u']))>-COS:continue
                placement=v3._text_placement(oriented,first,second,members,relevant,other_texts,budget)
                if placement is None:continue
                diagnostics['opposed_arrow_pairs']+=1
                measured_axis=np.array(placement['measurement_axis_px'])
                excluded={m['id'] for m in members}|shaft_ids
                for face in v3._budgeted(topology.faces,budget):
                    if any(set(edge['geometry_ids'])&excluded for edge in face['boundaries']):continue
                    options=[]
                    for arrow in (first,second):
                        one=[]
                        for edge in v3._budgeted(face['boundaries'],budget):
                            a,b=np.array(edge['points_px']);delta=b-a;length=float(np.linalg.norm(delta))
                            if length<1:continue
                            u=delta/length
                            if abs(float(np.dot(u,measured_axis)))>SIN:continue
                            point=a+np.clip(float(np.dot(arrow['tip']-a,u)),0,length)*u
                            if (abs(v3._cross(arrow['tip']-a,u))>oriented['tol']
                                    or np.linalg.norm(point-arrow['tip'])<max(12.,oriented['height'])):continue
                            for extension in v3._budgeted(lines,budget):
                                if extension['id'] in excluded or extension['id'] in edge['geometry_ids']:continue
                                if (abs(float(np.dot(extension['u'],u)))<COS
                                        or v3._segment_distance(arrow['tip'],extension)>oriented['tol']*2
                                        or v3._segment_distance(point,extension)>oriented['tol']*3):continue
                                if pixels.segment(arrow['tip'],point)<POLICY['line_ink_fraction']:continue
                                for mid in edge['geometry_ids']:one.append((point,mid,extension['id']))
                        options.append(one)
                    for one,two in v3._budgeted(itertools.product(*options),budget):
                        a,ma,ea=one;b,mb,eb=two
                        if ma==mb or ea==eb:continue
                        # Both contacts must lie on the same side of the shaft.
                        if v3._cross(a-first['tip'],measured_axis)*v3._cross(b-second['tip'],measured_axis)<=0:continue
                        diagnostics['face_flank_pairs']+=1
                        candidate=v3._candidate([a,b],[ma,mb],[m['id'] for m in members],[ea,eb],
                            [first['id'],second['id']],{'relation':'explicit_diameter_opposed_arrows_parallel_feature_flanks',
                            **placement,'explicit_symbol_required':True,'contour_face_ids':[face['id']],
                            'contour_support':'shared_bounded_face','contour_role_approved':False,
                            'extension_to_feature_relation':'collinear_observed_carriers',
                            'pixel_tolerance':oriented['tol']})
                        _add_equivalent(choices,candidate)
                        if len(choices)>POLICY['maximum_alternatives']:return choices[:POLICY['maximum_alternatives']],True,diagnostics
    return choices,cut,diagnostics


def _leader_candidates(text,lines,curves,arrows,kind,budget,pixels):
    choices=[];cut=False
    diagnostics={'curve_arrow_contacts':0,'connected_paths':0,'pixel_rejected_paths':0,'branching_junctions':0}
    near=[l for l in v3._budgeted(lines,budget)
          if v3._segment_distance(text['center'],l)<=max(text['width']*3,text['height']*12)]
    if len(near)>POLICY['maximum_leader_lines']:cut=True
    near=sorted(near,key=lambda l:(v3._segment_distance(text['center'],l),l['id']))[:POLICY['maximum_leader_lines']]
    adjacency=defaultdict(list);junctions={};support_cache={}
    def line_supported(line):
        if line['id'] not in support_cache:
            support_cache[line['id']]=pixels.segment(*line['points'])
        return support_cache[line['id']]>=POLICY['line_ink_fraction']
    for i,left in enumerate(near):
        for right in v3._budgeted(near[i+1:],budget):
            if abs(float(np.dot(left['u'],right['u'])))>=COS:continue
            pairs=[(float(np.linalg.norm(a-b)),a,b) for a in left['points'] for b in right['points']]
            distance,a,b=min(pairs,key=lambda p:p[0])
            if distance>text['tol']*3:continue
            if pixels.segment(a,b)<POLICY['line_ink_fraction']:continue
            point=(a+b)/2
            degree=sum(v3._segment_distance(point,l)<=text['tol'] for l in near)
            key=tuple(sorted((left['id'],right['id'])))
            junctions[key]={'point_px':point.tolist(),'endpoint_gap_px':distance,'incident_cv_lines':degree}
            adjacency[left['id']].append(right);adjacency[right['id']].append(left)
    for curve in v3._budgeted(curves,budget):
        if kind=='diameter' and curve['kind']!='circle':continue
        curve_support=None
        for arrow in v3._budgeted(arrows,budget):
            if not v3._on_curve(arrow['tip'],curve,text['tol']*2):continue
            delta=arrow['tip']-curve['center'];length=float(np.linalg.norm(delta))
            if length<max(1e-6,curve['radius']*.5):continue
            radial=delta/length
            if abs(float(np.dot(radial,arrow['u'])))<COS:continue
            diagnostics['curve_arrow_contacts']+=1
            if curve_support is None:curve_support=pixels.curve(curve)
            if curve_support<POLICY['curve_ink_fraction']:continue
            starts=[l for l in near if abs(float(np.dot(l['u'],radial)))>=COS
                    and v3._segment_distance(arrow['tip'],l)<=text['tol']*3
                    and v3._perp(curve['center'],l)<=text['tol']*2 and line_supported(l)]
            for start in starts:
                stack=[([start],[])]
                while stack:
                    path,joins=stack.pop();last=path[-1]
                    # New branch handles an observed elbow, not a direct leader.
                    if (len(path)>=2 and abs(float(np.dot(last['u'],text['u'])))>=COS
                            and v3._segment_distance(text['center'],last)<=text['height']*1.5+6):
                        contact=curve['center']+radial*curve['radius']
                        opposite=curve['center'] if kind=='radius' else curve['center']-radial*curve['radius']
                        branching=any(j['incident_cv_lines']>2 for j in joins)
                        diagnostics['connected_paths']+=1;diagnostics['branching_junctions']+=branching
                        candidate=v3._candidate([opposite,contact],[curve['id']],[l['id'] for l in path],[],[arrow['id']],
                            {'relation':'explicit_radial_symbol_curve_arrow_observed_elbow',
                             'path_order':[l['id'] for l in path],'junctions':joins,'branching_requires_review':branching,
                             'curve_ink_fraction':curve_support,'path_ink_fractions':{l['id']:support_cache[l['id']] for l in path},
                             'pixel_tolerance':text['tol'],'arrow_shape':_arrow_shape(arrow)})
                        _add_equivalent(choices,candidate)
                    if len(path)==POLICY['maximum_leader_segments']:continue
                    for following in v3._budgeted(adjacency[last['id']],budget):
                        if following['id'] in {l['id'] for l in path}:continue
                        if not line_supported(following):diagnostics['pixel_rejected_paths']+=1;continue
                        stack.append((path+[following],joins+[junctions[tuple(sorted((last['id'],following['id'])))]]))
                    if len(choices)>POLICY['maximum_alternatives']:return choices[:POLICY['maximum_alternatives']],True,diagnostics
    return choices,cut,diagnostics


def _demote(row,reason):
    if row['status']=='proposed':
        row['alternatives']=[{k:deepcopy(row[k]) for k in ('anchors_px',*v3_fields(),'evidence')}]
        for field in ('anchors_px',*v3_fields()):row[field]=[]
        row['status']='ambiguous';row['reasons'].append(reason)


def v3_fields():
    return ('measured_candidate_ids','dimension_line_candidate_ids','extension_line_candidate_ids','arrow_candidate_ids')


def propose(source_image,*,geometry,ocr,arrows,baseline,arm='all'):
    """No labels/expected values/bboxes; returns a separate needs_review report."""
    if arm not in ARMS:raise ValueError('Unknown frozen experiment arm')
    digest,size,lines,curves,normalized_arrows=v3._validate(source_image,geometry,ocr,arrows)
    if (baseline.get('algorithm_version')!='contour-association-v3' or baseline.get('source_sha256')!=digest
            or baseline.get('processed_size_px')!=size):raise ValueError('Baseline provenance mismatch')
    if len(ocr['lines'])>POLICY['maximum_texts']:raise ValueError('Support text budget exceeded')
    v3._unique_ids(baseline['associations'],'Baseline associations')
    flags=ARMS[arm];pixels=PixelSupport(source_image)
    good=[];rejected=[]
    for arrow in normalized_arrows:
        shape=_arrow_shape(arrow)
        if arrow['confidence']>=POLICY['confidence'] and shape.get('base_to_tip_dot',-1)>=COS:good.append(arrow)
        else:rejected.append({'id':arrow['id'],'shape':shape,'confidence':arrow['confidence']})
    by_id={line['id']:line for line in ocr['lines']}
    text_geometries={k:v3._text_geometry(line) for k,line in by_id.items()}
    records=deepcopy(baseline['associations']);by_text={r['source_line_id']:r for r in records}
    notation_by_text={n['source_line_id']:n for n in ocr['notations']};new_notations=[]
    topology=None;budget={'notation':0,'page':0};issues=[];history=[];modified=set();incomplete=False
    for line in ocr['lines']:
        if line['id'] in notation_by_text and notation_by_text[line['id']]['parsed']['raw']!=line['raw']:
            raise ValueError('Parsed notation cannot substitute another source reading')
        parsed=parse_composite(line['raw']) if 'composite' in flags else parse_notation(line['raw'])
        old=by_text.get(line['id']);stage={'source_line_id':line['id'],'raw':line['raw'],'confidence':line['confidence'],
            'parsed':parsed,'view_id':None,'baseline_status':old['status'] if old else 'no_record'};history.append(stage)
        if old and old['status']=='proposed':stage['action']='baseline_proposal_preserved';continue
        if (parsed['status']!='parsed' or parsed['kind'] not in {'nominal','linear','radius','diameter'}
                or line['confidence']<POLICY['confidence'] or line.get('conflicting_line_ids')
                or notation_by_text.get(line['id'],{}).get('context')=='title_block_candidate'):
            stage['action']='notation_or_confidence_gate';continue
        kind={'nominal':'linear'}.get(parsed['kind'],parsed['kind']);text=text_geometries[line['id']]
        quantity='component_parse' in parsed
        enabled=(kind=='linear' and text['orientation_uncertain'] and 'orientation' in flags
                 or quantity or kind in {'radius','diameter'} and ('leaders'in flags or kind=='diameter' and 'flanks'in flags))
        if not enabled:stage['action']='outside_extension_scope';continue
        row=deepcopy(old) if old else {'id':'support-association-'+line['id'],'notation_id':'support-notation-'+line['id'],
            'source_line_id':line['id'],'raw':line['raw'],'kind':kind,'nominal_mm':parsed['values'].get('nominal_mm'),
            'status':'unresolved','text_polygon_px':line['polygon_px'],'source_ref':line.get('source_ref'),
            'anchors_px':[],'measured_candidate_ids':[],'dimension_line_candidate_ids':[],'extension_line_candidate_ids':[],
            'arrow_candidate_ids':[],'evidence':{},'reasons':[],'alternatives':[]}
        row['baseline_record']=deepcopy(old);row['support_version']=POLICY['version'];row['accepted']=False
        if quantity:
            new_notations.append({'id':row['notation_id'],'source_line_id':line['id'],'parsed':parsed,
                                  'confidence':line['confidence'],'accepted_as_dimension':False,'accepted_as_scale':False})
        budget['notation']=0;choices=deepcopy(row['alternatives']);cut=False;diagnostics={};new_evidence=False
        try:
            if kind=='linear' or kind=='diameter' and 'flanks'in flags:
                if topology is None:
                    topology=ContourTopology(source_image,lines,
                        excluded_ids={v for a in normalized_arrows if a['confidence']>=.65 for v in a.get('evidence',{}).get('geometry_ids',[])},
                        text_polygons=[l['polygon_px'] for l in ocr['lines']])
                if kind=='diameter':
                    found,truncated,details=_flank_candidates(text,lines,good,text_geometries.values(),budget,topology,pixels,
                        orientation='orientation'in flags)
                else:
                    found,truncated,details=_linear_candidates(text,lines,good,text_geometries.values(),budget,topology,
                        orientation='orientation'in flags)
                new_evidence|=bool(found)
                for c in found:_add_equivalent(choices,c)
                cut|=truncated;diagnostics['line_chains']=details
            if quantity:
                found,truncated,details=v3._radial(text,lines,curves,good,kind,budget)
                # A new grammar must not license curves absent from the pixels.
                curve_map={c['id']:c for c in curves}
                for c in found:
                    fractions=[pixels.curve(curve_map[i]) for i in c['measured_candidate_ids']]
                    if fractions and min(fractions)>=POLICY['curve_ink_fraction']:
                        c['evidence']['curve_ink_fractions']=fractions;_add_equivalent(choices,c);new_evidence=True
                cut|=truncated;diagnostics['quantity_radial']=details
            if kind in {'radius','diameter'} and 'leaders'in flags:
                found,truncated,details=_leader_candidates(text,lines,curves,good,kind,budget,pixels)
                new_evidence|=bool(found)
                for c in found:_add_equivalent(choices,c)
                cut|=truncated;diagnostics['elbow_leaders']=details
        except v3._SearchLimit:
            cut=True;incomplete=True;diagnostics['bounded_search_exhausted']=True
        stage.update(action='support_search',diagnostics=diagnostics,search_checks=budget['notation'],truncated=cut)
        if old and not new_evidence:
            stage['action']='no_additional_spatial_support';continue
        v3._apply_search_result(row,choices,cut,diagnostics,issues)
        if old and old['status']=='ambiguous' and old['alternatives']:
            _demote(row,'baseline_ambiguity_preserved')
        if row['status']=='proposed':
            evidence=row['evidence']
            if evidence.get('branching_requires_review'):_demote(row,'branching_leader_requires_review')
            if evidence.get('relation') in {'opposed_arrows_parallel_text_perpendicular_extensions_object_intersections',
                                            'explicit_diameter_opposed_arrows_parallel_feature_flanks'}:
                if not topology.complete or evidence.get('contour_support')!='shared_bounded_face':
                    _demote(row,'unconfirmed_contour_role_requires_review')
        if old:records[records.index(old)]=row
        else:records.append(row)
        modified.add(row['id'])
    incomplete|=any(h.get('truncated',False) for h in history)
    # Do not let a new radial/flank claim silently reuse another text's arrow.
    claims=defaultdict(set)
    for row in records:
        choices=[row] if row['status']=='proposed' else row['alternatives']
        for c in choices:
            for aid in c.get('arrow_candidate_ids',[]):claims[aid].add(row['source_line_id'])
    for row in records:
        if row['id'] in modified and any(len(claims[aid])>1 for aid in row['arrow_candidate_ids']):
            _demote(row,'competing_observed_text_for_arrow')
        if incomplete and row['id'] in modified:_demote(row,'page_support_search_incomplete')
    return {'schema_version':'dimension-support/1.0','algorithm_version':'dimension-support-v1','arm':arm,
        'status':'needs_review','source':str(source_image),'source_sha256':digest,'processed_size_px':size,
        'associations':records,'new_notations':new_notations,'history':history,'issues':issues,
        'rejected_arrows':rejected,'search_checks':budget['page'],'pixel_samples':pixels.samples,
        'complete':not incomplete and not any(h.get('truncated') for h in history),
        'contour_topology':topology.report() if topology else None,
        'policy':POLICY,'baseline_proposals_preserved':True,'view_id':None,
        'physical_geometry_approved':False,'production_admitted':False,'accepted_geometry':False}
