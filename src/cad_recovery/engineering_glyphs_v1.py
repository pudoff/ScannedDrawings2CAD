"""Pixel glyph annotations with explicit uncertainty and diagnostic-only admission."""
from collections import Counter, defaultdict
import re

import cv2
import numpy as np

from .family_catalog_v2 import require

CLASSES = ('diameter_mark', 'letter_R', 'letter_M', 'plus_minus', 'degree', 'multiplication',
    'digit_zero', 'round_letter', 'phi_like_letter', 'currency_dollar', 'circle_geometry', 'crossed_geometry')
TEXT_CLASSES = {'⌀':'diameter_mark','Ø':'diameter_mark','ø':'diameter_mark','R':'letter_R','M':'letter_M',
    '±':'plus_minus','°':'degree','×':'multiplication','x':'multiplication','0':'digit_zero',
    'O':'round_letter','О':'round_letter','φ':'phi_like_letter','Φ':'phi_like_letter',
    'ф':'phi_like_letter','Ф':'phi_like_letter','$':'currency_dollar'}
POLICY = {'id':'engineering-glyph-diagnostic/1.0','bbox':'integer half-open native context pixels',
    'reviewer':'assistant','human_verified':False,'training_admission':'not_released',
    'physical_geometry_approved':False,'independent_evaluation':False,
    'literal_policy':'Source font code points unknown; preserve stated transcription conventions. Visual family is not semantic or Unicode identity.',
    'unresolved_policy':'No class/literal target; expected context reading may be retained separately.',
    'pixel_policy':'Exact RGB crop, no cleaning, resampling or paint. Neighbouring ink may intersect a rectangular box.'}


def component_boxes(rgb):
    """Navigation proposals only; never constitutes label approval."""
    require(isinstance(rgb,np.ndarray) and rgb.dtype==np.uint8 and rgb.ndim==3 and rgb.shape[2]==3
        and 0<rgb.shape[0]*rgb.shape[1]<=4000000,'Expected bounded RGB context')
    _,_,stats,_=cv2.connectedComponentsWithStats((rgb.min(axis=2)<245).astype('uint8'),8)
    boxes=[[int(x),int(y),int(x+w),int(y+h)]for x,y,w,h,a in stats[1:]if a>=3 and
        not(w>.7*rgb.shape[1] and h<=max(4,.12*rgb.shape[0]))]
    changed=True
    while changed:
        changed=False
        for i,a in enumerate(boxes):
            for j in range(i+1,len(boxes)):
                b=boxes[j];wa=a[2]-a[0];wb=b[2]-b[0]
                if (min(a[2],b[2])-max(a[0],b[0])>.65*min(wa,wb) and
                    abs((a[0]+a[2]-b[0]-b[2])/2)<.3*max(wa,wb)):
                    boxes[i]=[min(a[0],b[0]),min(a[1],b[1]),max(a[2],b[2]),max(a[3],b[3])]
                    boxes.pop(j);changed=True;break
            if changed:break
    return sorted(boxes,key=lambda b:(b[0],b[1]))


def translation(x,y): return [[1,0,x],[0,1,y],[0,0,1]]


def valid_box(box,size):
    require(isinstance(box,list) and len(box)==4 and all(type(v)is int for v in box) and
        0<=box[0]<box[2]<=size[0] and 0<=box[1]<box[3]<=size[1],'Invalid native glyph bounds')


def validate(contexts,glyphs):
    require(0<len(contexts)<=192 and 0<len(glyphs)<=512,'Glyph package exceeds diagnostic scope')
    by={c['id']:c for c in contexts};require(len(by)==len(contexts),'Duplicate context IDs')
    require(len({g['id']for g in glyphs})==len(glyphs),'Duplicate glyph IDs')
    groups={};sources={};images={};seen=set()
    for c in contexts:
        require(re.fullmatch(r'[A-Za-z0-9_.-]+',c['id']) and c['split']in('train','validation','development','synthetic'),
                'Invalid context ID or closed-test scope')
        require(c['reference']is None or isinstance(c['reference'],str),'Invalid context transcription')
        require(len(c['size_px'])==2 and all(type(v)is int and v>0 for v in c['size_px']) and
            c['size_px'][0]*c['size_px'][1]<=4000000,'Invalid context pixel budget')
        if c['split']in('train','validation'):
            for owners,key in [(groups,c['group_id']),(images,c['image']['sha256'])]:
                require(owners.setdefault(key,c['split'])==c['split'],'Cross-split glyph group/image leakage')
            if c.get('source_sha256'):
                require(sources.setdefault(c['source_sha256'],c['split'])==c['split'],'Cross-split glyph source leakage')
        if c['context_to_source']is not None:
            require('source_box'in c and len(c['source_box'])==4 and all(type(v)is int for v in c['source_box']) and
                c['context_to_source']==translation(c['source_box'][0],c['source_box'][1]) and
                c['size_px']==[c['source_box'][2]-c['source_box'][0],c['source_box'][3]-c['source_box'][1]],
                'Source transform must describe the exact native crop')
    for g in glyphs:
        require(re.fullmatch(r'[A-Za-z0-9_.-]+',g['id']) and g['context_id']in by,'Invalid glyph identity')
        c=by[g['context_id']];valid_box(g['box'],c['size_px'])
        require(g['split']==c['split'] and g['group_id']==c['group_id'],'Glyph scope differs from context')
        require(g['state']in('accepted_diagnostic','unresolved') and g['human_verified']is False and
            g['training_admission']=='not_released' and g['physical_geometry_approved']is False,'Changed glyph authority')
        require((g['context_id'],tuple(g['box']))not in seen,'Duplicate annotated glyph box');seen.add((g['context_id'],tuple(g['box'])))
        if g['state']=='unresolved':require(g['class']is None and g['literal']is None and g['reason'],'Unresolved glyph contains a target')
        else:
            require(g['class']in CLASSES,'Unknown glyph class')
            if g['class']not in('circle_geometry','crossed_geometry'):
                require(g['literal']in TEXT_CLASSES and TEXT_CLASSES[g['literal']]==g['class'],'Literal/class conflict')
                index=g['index'];require(type(index)is int and c['reference']is not None and
                    0<=index<len(c['reference']) and c['reference'][index]==g['literal'],'Glyph index differs from context')
        require(g['glyph_to_context']==translation(g['box'][0],g['box'][1]),'Changed glyph coordinate transform')
        expected=(np.asarray(c['context_to_source'])@np.asarray(g['glyph_to_context'])).tolist() if c['context_to_source']is not None else None
        require(g['glyph_to_source']==expected,'Changed glyph/source transform')


def summary(contexts,glyphs):
    validate(contexts,glyphs)
    coverage={}
    for split in ('train','validation','development','synthetic'):
        rows=[g for g in glyphs if g['split']==split];accepted=[g for g in rows if g['state']=='accepted_diagnostic']
        coverage[split]={'contexts':sum(c['split']==split for c in contexts),'glyph_slots':len(rows),
            'accepted':len(accepted),'unresolved':len(rows)-len(accepted),
            'classes':dict(sorted(Counter(g['class']for g in accepted).items())),
            'literals':dict(sorted(Counter(g['literal']for g in accepted if g['literal']is not None).items())),
            'unique_accepted_images':len({g['image']['sha256']for g in accepted}),
            'unique_images_by_class':{cls:len({g['image']['sha256']for g in accepted if g['class']==cls})for cls in sorted({g['class']for g in accepted})}}
    duplicates=defaultdict(list)
    for g in glyphs:
        if g['state']=='accepted_diagnostic':duplicates[g['image']['sha256']].append(g)
    conflicts=[]
    for sha,rows in duplicates.items():
        if len({g['class']for g in rows})>1:conflicts.append({'sha256':sha,'glyph_ids':[g['id']for g in rows]})
        real_splits={g['split']for g in rows if g['split']in('train','validation')}
        require(len(real_splits)<=1,'Identical glyph pixels cross train/validation')
    require(not conflicts,'Identical glyph pixels have contradictory classes; resolve before release')
    return {'schema_version':'engineering-glyph-report/1.0','coverage':coverage,
        'glyph_slots':len(glyphs),'accepted':sum(g['state']=='accepted_diagnostic'for g in glyphs),
        'unresolved':sum(g['state']=='unresolved'for g in glyphs),'context_count':len(contexts),
        'identical_pixel_groups':[{'sha256':k,'ids':[g['id']for g in v]}for k,v in sorted(duplicates.items())if len(v)>1],
        'unique_accepted_glyph_images':len(duplicates),'policy':POLICY,'training_eligible':False,
        'OCR_executed':False,'model_training_executed':False,'closed_test_used':False}


def require_purpose(purpose):
    require(purpose in('diagnostic_glyph_evaluation','annotation_review'),
            'Glyph data are diagnostic-only; classifier training and physical targets are not released')
