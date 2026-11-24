"""Verify a narrow quantity-prefix seam conflict without rewriting raw evidence.

Only an unparsed prefix of an explicitly parsed grouped radius/diameter is
eligible. Both source windows, native overlap pixels and two new OCR readings
must agree. Competing complete readings, values and tolerances stay unresolved.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import math
import re
import cv2
import numpy as np
from .dimension_support_v1 import parse_composite
from .notation_recovery_v1 import bounds,rectify

POLICY={'version':'partial-ocr-consistency/1.0','confidence':.65,
        'maximum_lines':128,'maximum_pairs':8,'maximum_rec_calls':16,
        'maximum_roi_pixels':200000,'full_window_clearance_px':2,
        'angle_degrees':12,'height_ratio':1.8,'minimum_containment':.8,
        'minimum_ink_fraction':.01,'maximum_ink_fraction':.5,
        'production_admitted':False,'physical_geometry_approved':False}

def literal(raw):
    return re.sub(r'\s+',' ',raw).strip()

def _intersect(a,b):
    return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))

def _area(b):return max(0,b[2]-b[0])*max(0,b[3]-b[1])

def _representative(line,ocr):
    observations={o['id']:o for o in ocr.get('observations',[])}
    for oid in line.get('observation_ids',[]):
        o=observations.get(oid)
        if (o and o['raw']==line['raw'] and o['confidence']==line['confidence']
                and o['polygon_px']==line['polygon_px'] and o['source_ref']==line['source_ref']):return o
    return None

def _window(line,ocr):
    ref=line.get('source_ref',{});windows={w['id']:w['bbox_px'] for w in ocr.get('windows',[])}
    if ref.get('sha256')!=ocr['source_sha256'] or ref.get('coordinate_space')!='processed_pixel_centers':return None
    window=windows.get(ref.get('window_id'))
    if window!=ref.get('window_bbox_px') or window is None:return None
    w,h=ocr['processed_size_px']
    if (not all(type(v)is int for v in window) or not 0<=window[0]<window[2]<=w
            or not 0<=window[1]<window[3]<=h):return None
    b=bounds(line['polygon_px'])
    if _intersect(window,b)<_area(b)-1e-6:return None
    return window

def _axis(poly):
    p=np.array(poly);edges=[p[1]-p[0],p[3]-p[0]];u=max(edges,key=np.linalg.norm);norm=float(np.linalg.norm(u))
    if not norm:raise ValueError('Nonzero text axis required')
    u=u/norm;n=np.array([-u[1],u[0]]);height=float(np.ptp(p@n))
    if height<=0:raise ValueError('Positive text height required')
    return u,height

def candidate(full,partial,ocr,gray):
    a,b=literal(full['raw']),literal(partial['raw']);parsed=parse_composite(full['raw'])
    if (parsed['status']!='parsed' or parsed['kind'] not in {'radius','diameter'}
            or not parsed['values'].get('count') or parse_composite(partial['raw'])['status']=='parsed'):
        return None,'outside_quantity_prefix_scope'
    if not a.startswith(b) or len(a)<=len(b) or not a[len(b)].isspace():return None,'not_exact_token_prefix'
    if (full['confidence']<POLICY['confidence'] or partial['confidence']<POLICY['confidence']):return None,'low_original_confidence'
    if (partial['id'] not in full.get('conflicting_line_ids',[]) or full['id'] not in partial.get('conflicting_line_ids',[])):
        return None,'not_a_mutual_observed_conflict'
    if not _representative(full,ocr) or not _representative(partial,ocr):return None,'unbound_original_observation'
    fw,pw=_window(full,ocr),_window(partial,ocr)
    if fw is None or pw is None or fw==pw:return None,'unbound_or_same_source_window'
    f,p=bounds(full['polygon_px']),bounds(partial['polygon_px']);margin=POLICY['full_window_clearance_px']
    if not(f[0]>=fw[0]+margin and f[1]>=fw[1]+margin and f[2]<=fw[2]-margin and f[3]<=fw[3]-margin):
        return None,'complete_reading_touches_its_window'
    if _intersect(f,pw)>=_area(f)-1e-6:return None,'partial_window_does_not_clip_full_reading'
    if _intersect(f,p)/_area(p)<POLICY['minimum_containment']:return None,'partial_polygon_not_contained'
    fu,fh=_axis(full['polygon_px']);pu,ph=_axis(partial['polygon_px'])
    if abs(float(fu@pu))<math.cos(math.radians(POLICY['angle_degrees'])) or max(fh,ph)/min(fh,ph)>POLICY['height_ratio']:
        return None,'incompatible_baselines_or_heights'
    x,y,r,z=[int(v) for v in (math.floor(p[0]),math.floor(p[1]),math.ceil(p[2]),math.ceil(p[3]))]
    if not (fw[0]<=x<r<=fw[2] and fw[1]<=y<z<=fw[3] and pw[0]<=x<r<=pw[2] and pw[1]<=y<z<=pw[3]):
        return None,'overlap_patch_outside_registered_windows'
    # Independently apply both registered crop translations to the same pinned
    # raster. No resized overview, inferred characters or target string enters.
    full_patch=gray[fw[1]:fw[3],fw[0]:fw[2]][y-fw[1]:z-fw[1],x-fw[0]:r-fw[0]]
    partial_patch=gray[pw[1]:pw[3],pw[0]:pw[2]][y-pw[1]:z-pw[1],x-pw[0]:r-pw[0]]
    if not np.array_equal(full_patch,partial_patch):raise ValueError('Registered overlap pixels disagree')
    if full_patch.size>POLICY['maximum_roi_pixels']:return None,'overlap_pixel_budget'
    ink=float(np.mean(full_patch<200))
    if not POLICY['minimum_ink_fraction']<=ink<=POLICY['maximum_ink_fraction']:return None,'missing_or_solid_prefix_ink'
    return {'full_id':full['id'],'partial_id':partial['id'],'full_window':fw,'partial_window':pw,
            'overlap_box_px':[x,y,r,z],'overlap_sha256':hashlib.sha256(full_patch.tobytes()).hexdigest(),
            'overlap_ink_fraction':ink,'full_observation_ids':full['observation_ids'],
            'partial_observation_ids':partial['observation_ids']},None

def verify(gray,ocr,recognize):
    if gray.dtype!=np.uint8 or gray.ndim!=2 or gray.size>12_000_000:raise ValueError('Bounded native grayscale required')
    if list(gray.shape[::-1])!=ocr['processed_size_px']:raise ValueError('OCR frame size mismatch')
    lines=ocr['lines']
    if len(lines)>POLICY['maximum_lines']:raise ValueError('Consistency line budget')
    for rows in (lines,ocr.get('observations',[]),ocr.get('windows',[])):
        if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate provenance identifier')
    if any(len(l.get('conflicting_line_ids',[]))>len(lines) or
           len(set(l.get('conflicting_line_ids',[])))!=len(l.get('conflicting_line_ids',[])) for l in lines):
        raise ValueError('Unbounded or duplicate conflict references')
    by_id={l['id']:l for l in lines};candidates=[];rejections=[]
    for full in lines:
        for pid in full.get('conflicting_line_ids',[]):
            if pid not in by_id:rejections.append({'full_id':full['id'],'partial_id':pid,'reason':'missing_conflict_reference'});continue
            item,reason=candidate(full,by_id[pid],ocr,gray)
            if item:candidates.append(item)
            else:rejections.append({'full_id':full['id'],'partial_id':pid,'reason':reason})
    ownership=Counter(i['full_id'] for i in candidates)+Counter(i['partial_id'] for i in candidates)
    eligible=[]
    for item in candidates:
        if ownership[item['full_id']]!=1 or ownership[item['partial_id']]!=1:
            rejections.append(item|{'reason':'competing_consistency_groups'})
        else:eligible.append(item)
    if len(eligible)>POLICY['maximum_pairs']:
        return {'groups':[],'verifications':[],'rejections':rejections,'complete':False,
                'reason':'consistency_pair_budget','rec_calls':0,'original_evidence_changed':False}
    groups=[];verifications=[];calls=0
    for item in eligible:
        pair=[]
        for role,identifier in [('full',item['full_id']),('partial',item['partial_id'])]:
            line=by_id[identifier];roi,proof=rectify(gray,line['polygon_px'])
            if roi.size>POLICY['maximum_roi_pixels']:raise ValueError('Re-recognition ROI budget')
            observed=recognize(np.repeat(roi[:,:,None],3,axis=2));calls+=1
            if not isinstance(observed['raw'],str) or not math.isfinite(observed['confidence']) or not 0<=observed['confidence']<=1:
                raise ValueError('Invalid fresh OCR observation')
            entry={'role':role,'source_line_id':identifier,'raw':observed['raw'],'confidence':observed['confidence'],
                   'roi_sha256':hashlib.sha256(roi.tobytes()).hexdigest(),'transform':proof,
                   'matches_original_literal':literal(observed['raw'])==literal(line['raw'])}
            pair.append(entry);verifications.append(entry)
        if all(p['matches_original_literal'] and p['confidence']>=POLICY['confidence'] for p in pair):
            groups.append(item|{'id':'consistent-prefix-'+item['full_id'],'verification_indices':[len(verifications)-2,len(verifications)-1],
                               'accepted_as_dimension':False,'original_conflicts_preserved':True})
        else:rejections.append(item|{'reason':'fresh_ocr_did_not_confirm_both_observed_literals'})
    return {'policy':POLICY,'groups':groups,'verifications':verifications,'rejections':rejections,
            'complete':True,'rec_calls':calls,'original_evidence_changed':False}

def association_view(ocr,evidence):
    """A separate hypothesis view, with every original conflict archived."""
    result=deepcopy(ocr);result['original_lines']=deepcopy(ocr['lines']);result['original_notations']=deepcopy(ocr['notations'])
    result['consistency_evidence']=deepcopy(evidence)
    if not evidence['complete']:return result
    removed={g['partial_id'] for g in evidence['groups']};fulls={g['full_id']:g for g in evidence['groups']}
    result['lines']=[line for line in result['lines'] if line['id'] not in removed]
    for line in result['lines']:
        if line['id'] in fulls:
            line['original_observation']=deepcopy(line)
            line['conflicting_line_ids']=[i for i in line['conflicting_line_ids'] if i!=fulls[line['id']]['partial_id']]
            line['consistency_group_id']=fulls[line['id']]['id']
    result['notations']=[]
    for line in result['lines']:
        parsed=parse_composite(line['raw'])
        if parsed['status']=='unparsed':continue
        original=next((n for n in ocr['notations'] if n['source_line_id']==line['id']),{})
        result['notations'].append({'id':'consistency-notation-'+line['id'],'source_line_id':line['id'],
            'source_ref':line['source_ref'],'polygon_px':line['polygon_px'],'confidence':line['confidence'],
            'context':original.get('context','drawing_or_notes'),'parsed':parsed,
            'status':'proposed' if parsed['status']=='parsed' and line['confidence']>=.65 and not line['conflicting_line_ids'] else 'ambiguous',
            'accepted_as_dimension':False,'accepted_as_scale':False})
    return result
