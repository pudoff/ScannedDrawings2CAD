"""Explicit bootstrap/measured/unavailable width views of immutable heads."""
from copy import deepcopy
import cv2
import numpy as np
from .stroke_roles_v1 import checked_gray,foreground

POLICY={'version':'head-width-contract/1.0','maximum_heads':512,'maximum_path_samples':500000,
    'physical_geometry_approved':False,'confidence_calibrated':False}


def normalize(gray,heads,*,immutable_ids=()):
    checked_gray(gray);old=set(immutable_ids);identities={h['id'] for h in heads}
    if len(heads)>512 or len(identities)!=len(heads) or not old<=identities:raise ValueError('Bounded unique heads and existing immutable IDs required')
    output=deepcopy(list(heads));records=[];used=0;limits=[];eligible=[];distance=None;ink=None
    for h in output:
        if h['id'] in old or h.get('evidence',{}).get('method')!='curved-source-heads/1.0':
            eligible.append(h['id']);records.append({'head_id':h['id'],'status':'inherited_contract_unchanged'});continue
        e=h['evidence'];bootstrap=e.get('local_shaft_width_px');original={k:deepcopy(e.get(k)) for k in ('local_shaft_width_px','observed_path_width_px')}
        if bootstrap is not None and (not np.isfinite(bootstrap) or bootstrap<=0):raise ValueError('Finite positive bootstrap width required')
        e['initial_shaft_width_px']=bootstrap;e['local_shaft_width_px']=None;e['observed_path_width_px']=None
        path=e.get('source_path',{});raw=path.get('path_px',[]);status='unavailable_no_source_samples';measure=None
        if raw:
            p=np.asarray(raw,float)
            if p.ndim!=2 or p.shape[1]!=2 or not np.isfinite(p).all() or not np.array_equal(p,np.rint(p)):
                status='unavailable_invalid_source_samples'
            elif used+len(p)>POLICY['maximum_path_samples']:
                limits.append('path_sample_budget');status='unavailable_budget'
            else:
                used+=len(p);p=p.astype(int)
                inside=(p>=0).all() and (p[:,0]<gray.shape[1]).all() and (p[:,1]<gray.shape[0]).all()
                connected=len(p)==1 or bool(np.all((np.abs(np.diff(p,axis=0)).max(1)<=1)&(np.abs(np.diff(p,axis=0)).sum(1)>0)))
                if ink is None:ink=foreground(gray);distance=cv2.distanceTransform(ink,cv2.DIST_L2,5)
                supported=path.get('supported') and path.get('source_only') and inside and connected and bool(np.all(ink[p[:,1],p[:,0]]>0))
                if not supported:status='unavailable_unverified_source_samples'
                else:
                    widths=2*distance[p[:,1],p[:,0]]-1
                    measure={'median_px':float(np.median(widths)),'q25_px':float(np.quantile(widths,.25)),
                        'q75_px':float(np.quantile(widths,.75)),'samples':len(p),'method':'source_DT_2r_minus_1',
                        'confidence_calibrated':False,'physical_units':None}
                    e['local_shaft_width_px']=e['observed_path_width_px']=measure['median_px'];status='measured_source_path'
                    eligible.append(h['id'])
        e['width_contract']={'version':POLICY['version'],'status':status,'bootstrap_px':bootstrap,
            'measurement':measure,'original_fields':original,'eligible_for_role_binding':status=='measured_source_path',
            'physical_geometry_approved':False}
        records.append({'head_id':h['id'],**deepcopy(e['width_contract'])})
    if limits:
        eligible=[h['id'] for h in heads if h['id'] in old or h.get('evidence',{}).get('method')!='curved-source-heads/1.0']
        for h in output:
            contract=h.get('evidence',{}).get('width_contract')
            if h['id'] not in old and contract:contract['eligible_for_role_binding']=False
        for r in records:
            if r.get('version')==POLICY['version']:r['eligible_for_role_binding']=False
    return {'raw_observations':deepcopy(list(heads)),'normalized_heads':output,
        'role_heads':[deepcopy(h) for h in output if h['id'] in eligible],'decisions':records,
        'audit':{'source_path_samples':used,'limits':limits},'complete':not limits,
        'source_pixels_erased':0,'physical_geometry_approved':False,'production_admitted':False}
