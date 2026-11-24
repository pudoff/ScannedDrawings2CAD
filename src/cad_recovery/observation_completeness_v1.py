"""Public, fail-closed completeness gate for source-bound observation reports.

Completion describes visited search space, not recognition accuracy, view
ownership or physical acceptance. Missing evidence remains unknown.
"""
from copy import deepcopy
from . import dimension_association_v3 as v3

VERSION='observation-completeness/1.0'
GEOMETRY_FLAGS=('contour_limit_reached','hough_circle_limit_reached',
                'short_connector_limit_reached','regional_short_line_limit_reached')
GEOMETRY_ISSUES={'geometry_candidate_limit','geometry_time_budget','geometry_search_incomplete','skeleton_iteration_limit'}

def _state(reasons,known):
    return {'state':'incomplete' if reasons else 'complete' if known else 'unknown',
            'reasons':reasons if reasons or known else ['completeness_evidence_missing_or_invalid']}

def assess(source_image,*,geometry,ocr,arrows,proposals):
    digest,size,*_=v3._validate(source_image,geometry,ocr,arrows)
    if proposals.get('source_sha256')!=digest or proposals.get('processed_size_px')!=size:
        raise ValueError('Completeness gate proposal provenance mismatch')
    stages={};analysis=geometry.get('analysis',{});issues=geometry.get('issues')
    reasons=[i['code'] for i in issues or [] if i.get('code') in GEOMETRY_ISSUES]
    reasons += [k for k in GEOMETRY_FLAGS if analysis.get(k) is True]
    if analysis.get('validation_queue_dropped',0)>0:reasons.append('validation_queue_dropped')
    if analysis.get('cheap_screen_limit_reached') is True:reasons.append('cheap_screen_limit_reached')
    known=isinstance(issues,list) and all(type(analysis.get(k))is bool for k in GEOMETRY_FLAGS)
    stages['geometry']=_state(sorted(set(reasons)),known)
    analysis=arrows.get('analysis',{});issues=arrows.get('issues')
    reasons=[i['code'] for i in issues or [] if i.get('code')=='arrow_search_limit']
    if analysis.get('search_limited') is True:reasons.append('search_limited')
    stages['arrows']=_state(sorted(set(reasons)),isinstance(issues,list) and type(analysis.get('search_limited'))is bool)
    coverage=ocr.get('coverage',{});complete=coverage.get('complete')
    known=(type(complete)is bool and type(coverage.get('completed_windows'))is int
           and type(coverage.get('total_windows'))is int and 0<=coverage['completed_windows']<=coverage['total_windows'])
    reasons=[]
    if complete is False:reasons.append('ocr_windows_incomplete')
    if known and complete and coverage['completed_windows']!=coverage['total_windows']:reasons.append('contradictory_ocr_coverage')
    stages['ocr']=_state(reasons,known)
    reasons=[];known=type(proposals.get('complete'))is bool
    if proposals.get('complete') is False:reasons.append('association_search_incomplete')
    for key in ('pixel_verification','carrier_guard','contour_topology'):
        if isinstance(proposals.get(key),dict) and proposals[key].get('complete')is False:reasons.append(key+'_incomplete')
    if any(h.get('truncated') for h in proposals.get('history',[])):reasons.append('truncated_association_history')
    if proposals.get('algorithm_version')=='contour-association-v3':
        known=isinstance(proposals.get('issues'),list) and isinstance(proposals.get('contour_topology'),dict)
        if any(i.get('code')=='association_candidate_limit' for i in proposals['issues']):reasons.append('association_candidate_limit')
    stages['associations']=_state(sorted(set(reasons)),known)
    passed=all(s['state']=='complete' for s in stages.values())
    return {'version':VERSION,'source_sha256':digest,'processed_size_px':size,'stages':stages,
            'complete':passed,'promotion_allowed_by_completeness':passed,
            'semantic_or_physical_acceptance':False}

def gate(source_image,*,geometry,ocr,arrows,proposals):
    assessment=assess(source_image,geometry=geometry,ocr=ocr,arrows=arrows,proposals=proposals)
    result=deepcopy(proposals)
    if not assessment['complete']:
        for row in result['associations']:
            if row['status']!='proposed':continue
            fields=('anchors_px','measured_candidate_ids','dimension_line_candidate_ids',
                    'extension_line_candidate_ids','arrow_candidate_ids','evidence')
            choice={k:deepcopy(row[k]) for k in fields}
            row.setdefault('alternatives',[]).append(choice)
            row['completeness_original_proposal']=deepcopy(next(a for a in proposals['associations'] if a['id']==row['id']))
            for k in fields[:-1]:row[k]=[]
            row.update(status='ambiguous',accepted=False)
            row['reasons'].append('upstream_or_association_completeness_not_confirmed')
    result['completeness_gate']=assessment;result['complete']=assessment['complete']
    result['production_admitted']=False;result['physical_geometry_approved']=False
    return result

def native_assessment(report,window_assessments=None):
    windows=report.get('windows',[]);assessments=window_assessments or []
    bound=(bool(windows) and len(assessments)==len(windows)
           and len({a['window_id'] for a in assessments})==len(assessments)
           and {a['window_id'] for a in assessments}=={w['id'] for w in windows})
    checks={
        'source_pixels':report.get('pixel_coverage_complete') is True,
        'ocr_windows':report.get('ocr_windows_complete') is True,
        'geometry_search':report.get('geometry_search_incomplete') is False,
        'deferred_contexts':type(report.get('deferred_retries'))is int and report['deferred_retries']==0,
        'all_window_stages_bound_and_complete':bound and all(a['assessment']['complete'] for a in assessments),
    }
    return {'version':VERSION,'checks':checks,'complete':all(checks.values()),
            'promotion_allowed_by_completeness':all(checks.values()),'physical_geometry_approved':False}

def gate_native(report,window_assessments):
    result=deepcopy(report);assessment=native_assessment(report,window_assessments)
    if not assessment['complete']:
        for a in result['associations']:
            if a['status']!='proposed':continue
            a['completeness_original_proposal']=deepcopy(a)
            a['status']='ambiguous';a['accepted']=False;a['anchors_px']=[]
            a['references']={k:[] for k in a['references']}
            a['completeness_reason']='native_or_window_observations_incomplete_or_unknown'
    result['completeness_gate']=assessment
    return result
