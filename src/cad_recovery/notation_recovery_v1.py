"""Bounded research hypotheses for diameter search and row re-recognition.

No target strings or annotation boxes are runtime inputs. Original observations
are immutable; template scores are neither OCR confidence nor semantic proof.
This module is deliberately separate from the admitted service profiles.
"""
from copy import deepcopy
import math
import re

import cv2
import numpy as np

POLICY = {
    'version': 'notation-recovery/1.0', 'maximum_lines': 128,
    'maximum_rec_calls': 256, 'maximum_template_lines': 64,
    'maximum_band_pixels': 393216, 'maximum_band_width': 2048,
    'rectified_height': 128, 'template_scales': [.70, .85, 1., 1.15],
    'template_angles': [-10, 0, 10], 'maximum_templates': 4,
    'threshold_candidates': [.60, .65, .70, .75, .80, .85, .90, .95],
    'group_angle_degrees': 10, 'group_height_ratio': 1.8,
    'group_center_offset_heights': .30, 'group_gap_heights': .75,
    'maximum_group_members': 6, 'maximum_group_width_heights': 12,
    'thin_width_height_ratio': .045, 'thin_width_px': 8,
    'dilation_kernel': [2, 2], 'confidence_threshold': .65,
    'physical_geometry_approved': False, 'production_admitted': False,
}


def bounds(polygon):
    a = np.asarray(polygon, dtype=float)
    if a.shape != (4, 2) or not np.isfinite(a).all():
        raise ValueError('Finite quadrilateral required')
    return [float(a[:, 0].min()), float(a[:, 1].min()),
            float(a[:, 0].max()), float(a[:, 1].max())]


def polygon(box):
    x, y, r, b = box
    return [[x,y], [r,y], [r,b], [x,b]]


def project(points, matrix):
    a = np.asarray(points, dtype=float)
    values = np.c_[a, np.ones(len(a))] @ np.asarray(matrix, dtype=float).T
    return (values[:, :2] / values[:, 2:]).tolist()


def _basis(poly):
    p = np.asarray(poly, dtype=float); bounds(p)
    edges = [p[1]-p[0], p[3]-p[0]]
    e = max(edges, key=np.linalg.norm); length = np.linalg.norm(e)
    if length < 2:
        raise ValueError('Degenerate text polygon')
    e = e/length
    # Engineering vertical dimensions read upwards. Other vertical orientations
    # remain diagnostic; no source string is used to select the direction.
    if (abs(e[0]) >= abs(e[1]) and e[0] < 0) or (abs(e[0]) < abs(e[1]) and e[1] > 0):
        e = -e
    n = np.array([-e[1], e[0]])
    u, v = p @ e, p @ n
    return e, n, [float(u.min()), float(v.min()), float(u.max()), float(v.max())]


def rectify(gray, poly, *, search=False):
    """Return observed pixels and exact affine map back to the context image."""
    if gray.dtype != np.uint8 or gray.ndim != 2 or gray.size > 12_000_000:
        raise ValueError('Bounded native grayscale context required')
    p = np.asarray(poly, dtype=float); bounds(p)
    if p.min() < 0 or p[:,0].max() > gray.shape[1] or p[:,1].max() > gray.shape[0]:
        raise ValueError('Text polygon outside context')
    e, n, (x,y,r,b) = _basis(p); h = b-y
    if h < 3:
        raise ValueError('Insufficient text height')
    pad_x, pad_y = ((.9, .25) if search else (.08, .08))
    x -= pad_x*h; r += .08*h; y -= pad_y*h; b += pad_y*h
    scale = min(1., POLICY['rectified_height']/h)
    w, height = math.ceil((r-x)*scale), math.ceil((b-y)*scale)
    if w > POLICY['maximum_band_width'] or w*height > POLICY['maximum_band_pixels']:
        raise ValueError('Rectification pixel budget exceeded')
    matrix = np.array([[e[0]*scale,e[1]*scale,-x*scale],
                       [n[0]*scale,n[1]*scale,-y*scale], [0,0,1.]])
    result = cv2.warpAffine(gray, matrix[:2], (w,height), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return result, {'context_to_roi':matrix.tolist(), 'roi_to_context':np.linalg.inv(matrix).tolist(),
                    'size_px':[w,height], 'text_height_px':h*scale,
                    'original_text_height_px':h, 'padding_native_heights':[pad_x,pad_y],
                    'sampling':'bilinear; white outside source; original pixels retained separately'}


def ink(gray):
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]


def trim_ink(gray):
    a = ink(gray); y,x = np.where(a > 0)
    if not len(x):
        raise ValueError('Empty template')
    return a[y.min():y.max()+1, x.min():x.max()+1]


def _rotated(template, height, angle):
    width = max(3, round(template.shape[1]*height/template.shape[0]))
    a = cv2.resize(template, (width,height), interpolation=cv2.INTER_NEAREST)
    if angle:
        pad = math.ceil(.2*max(width,height)); a = cv2.copyMakeBorder(a,pad,pad,pad,pad,cv2.BORDER_CONSTANT)
        matrix = cv2.getRotationMatrix2D(((a.shape[1]-1)/2,(a.shape[0]-1)/2),angle,1.)
        a = cv2.warpAffine(a,matrix,(a.shape[1],a.shape[0]),flags=cv2.INTER_NEAREST)
        y,x = np.where(a > 0); a = a[y.min():y.max()+1,x.min():x.max()+1]
    return a


def search_diameter(gray, lines, templates, threshold):
    if not 0 <= threshold <= 1 or not 0 < len(templates) <= POLICY['maximum_templates']:
        raise ValueError('Invalid frozen template policy')
    if len(lines) > POLICY['maximum_lines']:
        raise ValueError('Line budget exceeded')
    proposals=[]; skipped=[]; searches=0; correlations=0
    for line in lines:
        if not re.search(r'[0-9]', line['raw']) or any(s in line['raw'] for s in 'Ø⌀∅ø'):
            continue
        if searches == POLICY['maximum_template_lines']:
            skipped.append({'line_id':line['id'], 'reason':'template_line_budget'}); continue
        searches += 1
        try:
            band, proof = rectify(gray, line['polygon_px'], search=True)
        except ValueError as exc:
            skipped.append({'line_id':line['id'], 'reason':str(exc)}); continue
        band = ink(band); candidates=[]
        for ti, template in enumerate(templates):
            for scale in POLICY['template_scales']:
                for angle in POLICY['template_angles']:
                    t = _rotated(template, max(4, round(proof['text_height_px']*scale)), angle)
                    h,w = t.shape
                    if h > band.shape[0] or w > band.shape[1]: continue
                    scores = cv2.matchTemplate(band, t, cv2.TM_CCOEFF_NORMED); correlations += 1
                    _, score, _, (x,y) = cv2.minMaxLoc(scores)
                    local_poly = project(polygon([x,y,x+w,y+h]),proof['roi_to_context'])
                    bb = bounds(local_poly)
                    # Warped white padding cannot authorize evidence outside the
                    # actual input. Keep clipping failures explicit.
                    if bb[0] < 0 or bb[1] < 0 or bb[2] > gray.shape[1] or bb[3] > gray.shape[0]: continue
                    candidates.append({'score':float(score), 'template_index':ti,
                                       'scale':scale, 'angle_degrees':angle,
                                       'polygon_px':local_poly, 'bbox_px':bb, 'roi_box':[x,y,x+w,y+h],
                                       'transform':proof})
        if not candidates: continue
        best=max(candidates,key=lambda r:(r['score'],-r['template_index'],-r['scale'],-r['angle_degrees']))
        best.update(id=f'diameter-pixel-{len(proposals)+1:04d}', source_line_id=line['id'],
                    threshold_passed=best['score'] >= threshold, status='pixel_proposal',
                    semantic_diameter_confirmed=False, accepted=False,
                    original_raw=line['raw'], original_confidence=line['confidence'])
        proposals.append(best)
    return {'proposals':proposals, 'skipped':skipped, 'searches':searches, 'correlations':correlations,
            'coverage_complete':not skipped, 'threshold':threshold}


def diameter_hypotheses(lines, search):
    """Separate string hypothesis; never replace a backend observation in place."""
    by_id={p['source_line_id']:p for p in search['proposals'] if p['threshold_passed']}
    result=[]
    for line in lines:
        original=deepcopy(line); p=by_id.get(line['id']); candidate=None
        if p:
            e,n,b=_basis(line['polygon_px']); h=b[3]-b[1]
            u=np.array(p['polygon_px']) @ e
            # Only the prefix is unambiguous enough to propose a replacement.
            # Internal symbols need a new OCR observation, never character-index
            # guessing from average widths or an expected numeric value.
            if abs(float(u.min())-b[0]) <= .9*h and float(u.max()) <= b[0]+1.15*h:
                raw=line['raw']
                if re.match(r'^[0OО][1-9][0-9]*(?:[,.][0-9]+)?',raw): candidate='⌀'+raw[1:]
                elif re.match(r'^[1-9][0-9]*(?:[,.][0-9]+)?',raw): candidate='⌀'+raw
        if candidate:
            original.update(id='hypothesis-'+line['id'],raw=candidate,
                            original_observation=deepcopy(line),
                            hypothesis={'kind':'diameter_prefix','pixel_proposal_id':p['id'],
                                        'semantic_diameter_confirmed':False,'accepted':False})
        result.append(original)
    return result


def _barrier(gray, left, right):
    """Reject an observed long separator in the gap; no nominal is inspected."""
    a,b=bounds(left['polygon_px']),bounds(right['polygon_px'])
    x,r=math.ceil(a[2]),math.floor(b[0]); y,z=math.floor(min(a[1],b[1])),math.ceil(max(a[3],b[3]))
    if r<=x or z<=y: return False
    crop=ink(gray[max(0,y):min(gray.shape[0],z),max(0,x):min(gray.shape[1],r)])
    return bool(crop.size and np.any(np.count_nonzero(crop,axis=0) >= .8*crop.shape[0]))


def assemble_rows(gray, lines):
    """Conservative horizontal baseline grouping; vertical singletons retained."""
    if len(lines)>POLICY['maximum_lines']: raise ValueError('Line budget exceeded')
    by_id={p['id']:p for p in lines}
    if len(by_id)!=len(lines): raise ValueError('Duplicate observation ID')
    pending=sorted(lines,key=lambda p:(bounds(p['polygon_px'])[0],p['id']))
    groups=[]; reasons=[]
    for line in pending:
        e,n,b=_basis(line['polygon_px']); angle=math.degrees(math.atan2(e[1],e[0])); chosen=None
        if abs(angle)<=POLICY['group_angle_degrees']:
            for g in groups:
                previous=g[-1]; pe,pn,pb=_basis(previous['polygon_px'])
                pa=math.degrees(math.atan2(pe[1],pe[0])); h=b[3]-b[1]; ph=pb[3]-pb[1]
                gap=b[0]-pb[2]; center=abs((b[1]+b[3]-pb[1]-pb[3])/2)
                if (len(g)>=POLICY['maximum_group_members'] or abs(pa-angle)>POLICY['group_angle_degrees']
                        or max(h,ph)/min(h,ph)>POLICY['group_height_ratio']
                        or not 0 <= gap <= POLICY['group_gap_heights']*min(h,ph)
                        or center > POLICY['group_center_offset_heights']*min(h,ph)):
                    continue
                # A group cannot drift into a stacked tolerance through a chain.
                if any(abs((bounds(v['polygon_px'])[1]+bounds(v['polygon_px'])[3])/2
                           -(bounds(line['polygon_px'])[1]+bounds(line['polygon_px'])[3])/2)
                       > POLICY['group_center_offset_heights']*min(h,bounds(v['polygon_px'])[3]-bounds(v['polygon_px'])[1]) for v in g):
                    continue
                x=min(bounds(v['polygon_px'])[0] for v in g+[line]); r=max(bounds(v['polygon_px'])[2] for v in g+[line])
                if r-x > POLICY['maximum_group_width_heights']*min(h,ph):continue
                if _barrier(gray,previous,line):
                    reasons.append({'left':previous['id'],'right':line['id'],'reason':'observed_separator'});continue
                if chosen is not None:
                    chosen=None;reasons.append({'line':line['id'],'reason':'competing_rows'});break
                chosen=g
        if chosen is None:groups.append([line])
        else:chosen.append(line)
    return {'groups':[[p['id'] for p in g] for g in groups], 'rejections':reasons,
            'ordering':'baseline-compatible groups, then projected left-to-right; never top-first',
            'coverage_complete':True}


def stroke_width(gray):
    binary=ink(gray)
    dist=cv2.distanceTransform(binary,cv2.DIST_L2,5)
    peaks=(dist>0)&(dist>=cv2.dilate(dist,np.ones((3,3),np.uint8))-.01)
    return float(np.median(2*dist[peaks])) if np.any(peaks) else 0.


def thicken(gray, original_height):
    width=stroke_width(gray)
    # Width is measured in this rectified image; the height is in the same units.
    eligible=0 < width <= POLICY['thin_width_px'] and width/original_height<=POLICY['thin_width_height_ratio']
    result=cv2.erode(gray,np.ones(POLICY['dilation_kernel'],np.uint8)) if eligible else gray.copy()
    return result, {'stroke_width_px':width,'text_height_px':original_height,
                    'applied':bool(eligible),'kernel':POLICY['dilation_kernel'],
                    'scope':'experimental derivative; native image retained'}


def line_for_recognition(members, result, identifier):
    if len(members)==1: poly=members[0]['polygon_px']
    else:
        boxes=[bounds(p['polygon_px']) for p in members]
        poly=polygon([min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[2] for b in boxes),max(b[3] for b in boxes)])
    return {'id':identifier,'raw':result['raw'],'confidence':result['confidence'],
            'polygon_px':poly,'bbox_px':bounds(poly),'conflicting_line_ids':[],
            'source_ref':deepcopy(members[0]['source_ref']),
            'parent_observation_ids':[p['id'] for p in members],
            'has_unresolved_parent_conflicts':any(p.get('conflicting_line_ids') for p in members),
            'observation_kind':'new_recognition','recognition':result}
