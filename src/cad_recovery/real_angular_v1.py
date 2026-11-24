"""Adapter and partial spatial evaluation for reviewed real angle contexts."""
from copy import deepcopy
import math
import numpy as np

from .angular_link_metrics_v1 import overlap
from .family_catalog_v2 import require

MODES = ('reviewed_observations', 'automatic_cv_reviewed_text', 'automatic_cv_ocr')
POLICY = {'id':'real-angular/1.0','maximum_contexts':4,'maximum_pixels':4000000,
          'cv_text_exclusions':'none; no reviewed polygons or labels enter automatic CV',
          'geometry_limit':2500,'arrow_limit':2000,'text_match_iou':.35,
          'vertex_tolerance_px':8,'ray_tolerance_deg':4,'sector_tolerance_deg':4,
          'incomplete_search':'all notation proposals require review; retain raw/alternatives',
          'nominal_texts_are_components':True,'independent_evaluation':False,
          'physical_geometry_approved':False,'production_admitted':False,'training_eligible':False}


def validate_case(case):
    require(case['split'] in ('train','development') and case['group_id'], 'Closed/unknown split or missing group')
    require(all(case[k] is False for k in ('human_verified','independent_evaluation','training_eligible','physical_geometry_approved','production_admitted')),
            'Real-angle review cannot promote physical/training authority')
    box,size = case['source_box'],case['size_px']
    require(len(box)==4 and all(type(v)is int for v in box) and 0<=box[0]<box[2]<=case['source_size_px'][0]
            and 0<=box[1]<box[3]<=case['source_size_px'][1] and size==[box[2]-box[0],box[3]-box[1]]
            and size[0]*size[1]<=POLICY['maximum_pixels'],'Invalid native context box')
    require(case['context_to_source']==[[1,0,box[0]],[0,1,box[1]],[0,0,1]] and case['source_file_rehashed']is True,
            'Changed native coordinate transform/provenance')
    p=np.asarray(case['text_polygon_px'],float)
    require(p.shape==(4,2) and np.isfinite(p).all() and (p>=0).all() and (p<=size).all(),'Invalid reviewed text polygon')
    if case['reviewed_geometry_state']=='unresolved':
        require(case['expected_points_px']is None and case['expected_sector_deg']is None and case['unresolved_reason'],
                'Unknown supports acquired an invented target')
    else:
        points=np.asarray(case['expected_points_px'],float)
        require(case['reviewed_geometry_state']=='complete_pixel_supports' and points.shape==(3,2)
                and np.isfinite(points).all() and (points>=0).all() and (points<=size).all()
                and isinstance(case['expected_sector_deg'],(int,float)) and 0<case['expected_sector_deg']<360,
                'Invalid reviewed angular supports')


def adapt_native(report, *, image_sha256, image_size, RGB_equal):
    """Change only the packet schema, after proving identical native RGB pixels."""
    require(RGB_equal is True and report['native_size_px']==image_size and report['production_admitted']is False
            and report['physical_geometry_approved']is False and report['profile']['id']=='native-cv-experimental/3.0',
            'Native CV is not an equal-pixel v3 observation packet')
    require(report['source']['native_to_source']==[[1,0,0],[0,1,0],[0,0,1]],'Unexpected CV input transform')
    rows=report['observations']; ids={r['id'] for r in rows}
    require(len(ids)==len(rows) and all(set(r['supporting_observation_ids'])<=ids for r in rows),'Dangling native producer references')
    geom=[]; heads=[]; dropped_geometry=dropped_arrows=0
    for row in rows:
        if row.get('kind') in ('line','arc','circle'):
            if len(geom)>=POLICY['geometry_limit']: dropped_geometry+=1;continue
            # Axis observations are finite ink members; no gap is bridged here.
            geom.append(deepcopy(row))
    retained={r['id'] for r in geom}
    all_geometry_ids={r['id'] for r in rows if r.get('kind') in ('line','arc','circle')}
    for row in rows:
        if row['producer_ref']['stage']!='arrow_head':continue
        supports=[v for v in row['supporting_observation_ids'] if v in all_geometry_ids]
        if len(heads)>=POLICY['arrow_limit'] or not set(supports)<=retained:
            dropped_arrows+=1;continue
        head=deepcopy(row);head['evidence']={'geometry_ids':supports,'method':'unified_native_id_adapter',
                                           'producer_evidence_preserved':True}
        heads.append(head)
    common={'source_sha256':image_sha256,'processed_size_px':image_size}
    incomplete=report['search_incomplete'] or bool(dropped_geometry or dropped_arrows)
    meta={'native_image_sha256':report['native_image']['sha256'],'canonical_image_sha256':image_sha256,
          'equal_RGB_proved':True,'source_to_context':report['source']['native_to_source'],
          'incomplete_stages':report['incomplete_stages'],'search_incomplete':incomplete,
          'dropped_geometry':dropped_geometry,'dropped_arrows':dropped_arrows,
          'geometry':len(geom),'arrows':len(heads),'physical_geometry_approved':False}
    return dict(common,candidates=geom),dict(common,candidates=heads),meta


def gate_notations(ocr, incomplete):
    result=deepcopy(ocr)
    if incomplete:
        for row in result.get('notations',[]):
            row['status']='ambiguous';row['upstream_search_incomplete']=True
    return result


def point_error(points, reference):
    p,q=np.asarray(points,float),np.asarray(reference,float)
    if p.shape!=(3,2) or not np.isfinite(p).all():return None
    a,b=p[1:]-p[0],q[1:]-q[0]
    if min(np.linalg.norm(v) for v in [*a,*b])<1e-6:return None
    # Compare directed observed rays, preserving their sector order.
    angular=[math.degrees(math.acos(float(np.clip(np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v)),-1,1)))) for u,v in zip(a,b)]
    return {'vertex_px':float(np.linalg.norm(p[0]-q[0])),'maximum_ray_error_deg':max(angular)}


def evaluate(case, report):
    validate_case(case)
    require(report['source_sha256']==case['image']['sha256'] and report['processed_size_px']==case['size_px'], 'Evaluation image mismatch')
    rows=report['associations']
    choices=[(overlap(case['text_polygon_px'],r['text_polygon_px']),i,r) for i,r in enumerate(rows)]
    choices=[r for r in choices if r[0]>=POLICY['text_match_iou']]
    matched=max(choices,key=lambda r:(r[0],-r[1])) if choices else None
    row=matched[2] if matched else None; known=case['reviewed_geometry_state']=='complete_pixel_supports'
    proposed=bool(row and row['status']=='proposed'); error=None; correct=False
    if known and proposed and row['kind']=='angular':
        error=point_error(row['anchors_px'],case['expected_points_px'])
        correct=bool(error and error['vertex_px']<=POLICY['vertex_tolerance_px']
                     and error['maximum_ray_error_deg']<=POLICY['ray_tolerance_deg']
                     and abs(row.get('observed_sector_deg',-1000)-case['expected_sector_deg'])<=POLICY['sector_tolerance_deg']
                     and row.get('nominal_mm')is None and row.get('nominal_unit')=='degree')
    alternatives=[]
    if known:
        for r in rows:
            if overlap(case['text_polygon_px'],r['text_polygon_px'])<POLICY['text_match_iou']:continue
            for candidate in r.get('alternatives',[]):
                if 'points_px'not in candidate:continue
                e=point_error(candidate['points_px'],case['expected_points_px'])
                if e and e['vertex_px']<=POLICY['vertex_tolerance_px'] and e['maximum_ray_error_deg']<=POLICY['ray_tolerance_deg']:
                    alternatives.append({'notation_id':r['notation_id'],'errors':e,'complete_support':candidate['complete_support']})
    return {'known_geometry_targets':int(known),'unresolved_geometry_targets':int(not known),
            'localized_nominal_components':int(row is not None),'correct_links':int(correct),
            'false_links_in_known_target':int(known and proposed and not correct),
            'unverifiable_proposals':int(not known and proposed),
            'linear_claim_in_angular_label_context':int(bool(proposed and row['kind']=='linear')),
            'literal_equal_component':int(bool(row and row['raw']==case['reference'])),
            'target_status':row['status'] if row else 'not_localized','raw':row['raw'] if row else None,
            'target_kind':row['kind'] if row else None,'errors':error,
            'correct_support_among_alternatives':bool(alternatives),'alternative_supports':alternatives,
            'text_iou':matched[0] if matched else None,
            'non_target_proposals_not_scored':sum(r['status']=='proposed' and (row is None or r['id']!=row['id']) for r in rows),
            'component_comparison_is_not_full_ROI_OCR_accuracy':True}


def totals(rows):
    keys=('known_geometry_targets','unresolved_geometry_targets','localized_nominal_components','correct_links',
          'false_links_in_known_target','unverifiable_proposals','linear_claim_in_angular_label_context','literal_equal_component')
    return {k:sum(row[k] for row in rows) for k in keys}
