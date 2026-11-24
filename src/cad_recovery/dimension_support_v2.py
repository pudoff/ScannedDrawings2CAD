"""Conservative carrier-face guard over the preserved train-qualified v1.

An intervening closed face is an alternative, not permission to choose the
nearer or farther feature. Only new proposals can be demoted by this guard.
"""
from copy import deepcopy
import numpy as np
from . import dimension_support_v1 as base

ARMS=base.ARMS
POLICY={'version':'dimension-support/2.0','base_policy':base.POLICY,'maximum_carrier_checks':100000,
        'physical_geometry_approved':False,'production_admitted':False}


def _intervening_faces(row,faces,arrows,budget):
    axis=np.array(row['evidence']['measurement_axis_px']);tol=row['evidence']['pixel_tolerance']
    own=set(row['evidence']['contour_face_ids']);targets=[]
    # Candidate ID lists are sorted sets, not an anchor-order contract.
    for aid in row['arrow_candidate_ids']:
        tip=np.array(arrows[aid]['tip_px'])
        anchors=[np.array(p) for p in row['anchors_px'] if abs(float(np.dot(np.array(p)-tip,axis)))<=tol]
        if len(anchors)!=1:return [{'reason':'carrier_anchor_pairing_not_unique','arrow_id':aid}]
        delta=anchors[0]-tip;length=float(np.linalg.norm(delta))
        targets.append((aid,tip,delta/length,length))
    found=[]
    for face in faces:
        if face['id'] in own:continue
        contacts=[]
        for aid,tip,direction,length in targets:
            support=[]
            for edge in face['boundaries']:
                budget['checks']+=1
                if budget['checks']>POLICY['maximum_carrier_checks']:raise base.v3._SearchLimit()
                a,b=np.array(edge['points_px']);delta=b-a;span=float(np.linalg.norm(delta))
                if span<1:continue
                u=delta/span
                if abs(float(np.dot(u,axis)))>base.SIN or abs(base.v3._cross(tip-a,u))>tol:continue
                p=a+np.clip(float(np.dot(tip-a,u)),0,span)*u
                projection=float(np.dot(p-tip,direction))
                if tol*2<projection<length-tol:
                    support.append({'point_px':p.tolist(),'geometry_ids':edge['geometry_ids']})
            if support:contacts.append({'arrow_id':aid,'supports':support})
        if len(contacts)==len(targets):found.append({'face_id':face['id'],'carrier_contacts':contacts})
    return found


def propose(source_image,*,geometry,ocr,arrows,baseline,arm='all'):
    result=base.propose(source_image,geometry=geometry,ocr=ocr,arrows=arrows,baseline=baseline,arm=arm)
    result['algorithm_version']='dimension-support-v2';result['policy']=deepcopy(POLICY)
    faces=(result.get('contour_topology') or {}).get('faces',[])
    by_arrow={a['id']:a for a in arrows['candidates']};budget={'checks':0};events=[];incomplete=False
    for row in result['associations']:
        if (row['status']!='proposed' or not row.get('support_version')
                or row['evidence'].get('relation')!='explicit_diameter_opposed_arrows_parallel_feature_flanks'):continue
        try:competing=_intervening_faces(row,faces,by_arrow,budget)
        except base.v3._SearchLimit:
            competing=[{'reason':'carrier_guard_budget'}];incomplete=True
        if competing:
            row['evidence']['competing_unresolved_faces']=competing
            base._demote(row,'competing_unresolved_contour_face_on_carriers')
            events.append({'association_id':row['id'],'competing':competing})
    if incomplete:
        result['complete']=False
        for row in result['associations']:
            if row.get('support_version'):base._demote(row,'carrier_guard_incomplete')
    result['carrier_guard']={'checks':budget['checks'],'events':events,'complete':not incomplete}
    return result
