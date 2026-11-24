"""Fail closed on tiny-band template aliasing, preserving the frozen v1 kernel.

Qualification and native validation had no empty resized templates. Isolated
legacy glyph probes exposed that boundary. Such lines now remain explicitly
unknown, rather than crashing or treating disappeared ink as a negative sign.
"""
import re
from .notation_recovery_v1 import POLICY, rectify, _rotated, search_diameter


def guarded_search(gray, lines, templates, threshold):
    if len(lines)>POLICY['maximum_lines'] or not 0<len(templates)<=POLICY['maximum_templates'] or not 0<=threshold<=1:
        raise ValueError('Invalid bounded recovery input')
    retained=[];skipped=[];examined=0
    for line in lines:
        eligible=re.search(r'[0-9]',line['raw']) and not any(s in line['raw'] for s in 'Ø⌀∅ø')
        if not eligible:
            retained.append(line);continue
        if examined==POLICY['maximum_template_lines']:
            skipped.append({'line_id':line['id'],'reason':'template_line_budget'});continue
        examined+=1
        try:
            _,proof=rectify(gray,line['polygon_px'],search=True)
        except ValueError:
            # Frozen kernel already records these validation failures explicitly.
            retained.append(line);continue
        valid=True
        for template in templates:
            for scale in POLICY['template_scales']:
                for angle in POLICY['template_angles']:
                    try:_rotated(template,max(4,round(proof['text_height_px']*scale)),angle)
                    except ValueError:valid=False
        if valid:retained.append(line)
        else:skipped.append({'line_id':line['id'],'reason':'empty_template_after_small_height_resampling',
                             'status':'unknown','semantic_diameter_confirmed':False})
    result=search_diameter(gray,retained,templates,threshold)
    result['skipped']+=skipped;result['coverage_complete']=not result['skipped']
    result['guard_version']='notation-template-aliasing-guard/1.0'
    return result
