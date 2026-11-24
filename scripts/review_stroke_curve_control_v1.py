"""Freeze/build/replay arrow and curve controls on the screened raw-pixel cohort."""
import argparse
from html import escape
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cad_recovery import stroke_curve_control_v1 as control
from cad_recovery import pixel_role_draft_v1 as draft
from cad_recovery.arrow_detection import detect_arrows
from cad_recovery.dataset_layout_v1 import checked,encoded,inside,pin,read
from cad_recovery.family_catalog_v2 import require
from cad_recovery.source_coallocation_v1 import package

GEOMETRY=draft.GEOMETRY
PROFILES=['baseline','ranked_global','regional_masked']
CODE=['scripts/review_stroke_curve_control_v1.py','src/cad_recovery/stroke_curve_control_v1.py',
      'src/cad_recovery/arrow_detection.py','src/cad_recovery/hard_raster_metrics_v1.py']


def freeze(reading,output):
    require(not output.exists(), 'Choose new control config')
    value={'schema_version':'stroke-curve-control-config/1.0','task':'P10.2c.2a','geometry':GEOMETRY,
        'draft':{'path':'artifacts/pixel-role-draft-v2/manifest.json','sha256':'4e4acc98088dc22a5b8f3de3349c5bcafab171bf3b6377629c0274b215cda741'},
        'reading':pin(ROOT,reading),'code':[pin(ROOT,n) for n in CODE], 'policy':control.POLICY,
        'profiles':PROFILES,'model_inference_executed':False,'test_used':False,'physical_geometry_approved':False}
    output.write_bytes(encoded(value));return pin(ROOT,output.relative_to(ROOT).as_posix())


def context(record):
    config=read(checked(ROOT,record))
    require(config['policy']==control.POLICY and config['code']==[pin(ROOT,n) for n in CODE]
            and config['geometry']==GEOMETRY and config['profiles']==PROFILES, 'Changed control protocol')
    require(all(config[k] is False for k in ('model_inference_executed','test_used','physical_geometry_approved')), 'Changed control scope')
    _,parents=draft.verify(ROOT,config['draft'])
    by_id={r['id']:r for r in parents}
    geom,manifest=package(ROOT,GEOMETRY,'geometry-search-manifest/2.0')
    geom_config=read(checked(ROOT,manifest['config']))
    base=(ROOT/draft.BASE['path']).parent
    cases={r['id']:r for r in geom_config['cases']}
    reading=read(checked(ROOT,config['reading']))
    require(reading['reviewer']=='assistant' and all(reading[k] is False for k in
        ('human_verified','training_eligible','physical_geometry_approved','independent_evaluation')), 'Changed control authority')
    require(len(reading['patches'])==4 and len({r['patch_id'] for r in reading['patches']})==4, 'Lost control patch')
    for ref in reading['patches']:
        require(ref['patch_id'] in by_id and by_id[ref['patch_id']]['split'] in ('train','validation'), 'No JPEG/test control admission')
        control.validate_references(ref,by_id[ref['patch_id']]['size_px'])
    return config,reading,by_id,geom,cases,base


def score_saved(config,reading,parents,geom,cases,base,target):
    rows=[]
    for ref in reading['patches']:
        parent=parents[ref['patch_id']];case=cases[parent['case_id']]
        x,y=parent['patch_box_in_context'][:2]
        matrix=np.array([[1,0,-x],[0,1,-y],[0,0,1]])@np.asarray(case['processed_to_source'])
        profiles={}
        for profile in PROFILES:
            if profile=='baseline':
                geometry=read(checked(ROOT,case['baseline_geometry']))
                arrows=read(base/'runs'/case['id']/'cli_analyze_dimensions/result/page-001-arrows.json')
            else:
                geometry=read(geom/'runs'/case['id']/profile/'result/page-001-geometry.json')
                arrows=read(target/'runs'/case['id']/profile/'page-001-arrows.json')
            profiles[profile]=control.score(ref,geometry,arrows,matrix)
        rows.append({'patch_id':ref['patch_id'],'case_id':case['id'],'split':parent['split'],
            'processed_to_patch':matrix.tolist(),'profiles':profiles,'excluded_arrow_tips':ref['excluded_arrows']})
    keys=['assigned_arrow_tips','matched_directed_arrows','line_endpoints_near_tips','visible_dash_guides',
        'covered_dash_guides','gap_guides','bridged_gap_guides','assigned_open_arcs','qualified_open_arcs',
        'false_arrows_in_assigned_negative_regions']
    aggregate={p:{k:sum(r['profiles'][p][k] for r in rows) for k in keys} for p in PROFILES}
    return {'schema_version':'stroke-curve-control-report/1.0','policy':control.POLICY,'rows':rows,
        'aggregate':aggregate,'full_precision':None,'models_executed':False,'test_used':False,
        'physical_geometry_approved':False,'CAD_promotions':0,'production_changes':False,
        'arrow_inputs':'same saved OCR polygons; new geometry profiles rerun existing filled-arrow detector',
        'notes':['Assigned guides only; partial review does not establish full precision.',
                 'Line-end proximity is reported separately from one-to-one directed arrow matches.',
                 'Dash coverage by a merged long line does not validate dashed style or preserve gaps.',
                 'Full circles and truncated/overextended arcs cannot pass the open-arc endpoint/length checks.']}


def build(record,target):
    config,reading,parents,geom,cases,base=context(record)
    require(target.parent==ROOT/'artifacts' and not target.exists(),'Choose new control artifacts')
    target.mkdir()
    runtimes=[]
    for ref in reading['patches']:
        parent=parents[ref['patch_id']];case=cases[parent['case_id']]
        ocr=read(base/'runs'/case['id']/'cli_analyze_dimensions/result/page-001-ocr.json')
        for profile in PROFILES[1:]:
            geometry=read(geom/'runs'/case['id']/profile/'result/page-001-geometry.json')
            folder=target/'runs'/case['id']/profile
            start=time.perf_counter()
            result=detect_arrows(checked(ROOT,case['source']),folder,geometry=geometry,
                                 text_polygons=[line['polygon_px'] for line in ocr['lines']])
            runtimes.append({'case_id':case['id'],'profile':profile,'wall_seconds':time.perf_counter()-start,
                             'arrow_candidates':len(result['candidates']),'analysis':result['analysis']})
        source=(ROOT/config['draft']['path']).parent/parent['image']
        image=Image.open(source).convert('RGB');draw=ImageDraw.Draw(image)
        for arrow in ref['arrows']:
            x,y=arrow['tip'];direction=np.asarray(arrow['direction'],float);direction/=np.linalg.norm(direction)
            draw.ellipse((x-8,y-8,x+8,y+8),outline='red',width=2)
            draw.line([tuple(np.array([x,y])-direction*35),(x,y)],fill='red',width=3)
        for row in ref['strokes']:draw.line([tuple(p) for p in row['points']],fill='#079447',width=4)
        for row in ref['gaps']:draw.line([tuple(p) for p in row['points']],fill='#e89500',width=4)
        for row in ref['curves']:draw.line([tuple(p) for p in row['points']],fill='#a12cc9',width=3)
        for row in ref['negatives']:draw.rectangle(tuple(row['box']),outline='#0764ad',width=2)
        image.save(target/(ref['patch_id']+'-controls.png'))
    report=score_saved(config,reading,parents,geom,cases,base,target)
    (target/'report.json').write_bytes(encoded(report));(target/'runtimes.json').write_bytes(encoded(runtimes))
    (target/'references.json').write_bytes(encoded(reading))
    cards=['<!doctype html><meta charset="utf-8"><title>Arrow and curve controls</title>',
        '<style>body{font:16px sans-serif;margin:2rem}img{max-width:100%}pre{white-space:pre-wrap}</style>',
        '<h1>Контроль стрелок, штрихов и дуг</h1><p>Красный — направленные стрелки; зелёный — штрихи; оранжевый — пробелы; фиолетовый — открытые дуги; синий — отрицательные области.</p>',
        '<p><a href="report.json">Метрики и ограничения</a></p>']
    for row in reading['patches']:cards.append('<h2>'+escape(row['patch_id'])+'</h2><img src="'+row['patch_id']+'-controls.png">')
    (target/'index.html').write_text(''.join(cards),encoding='utf-8')
    files=sorted(p.relative_to(target).as_posix() for p in target.rglob('*') if p.is_file())
    (target/'manifest.json').write_bytes(encoded({'schema_version':'stroke-curve-control-manifest/1.0',
        'config':record,'files':[pin(target,n) for n in files]}))
    return {'manifest':pin(ROOT,(target/'manifest.json').relative_to(ROOT).as_posix()),'aggregate':report['aggregate']}


def verify(record):
    folder,manifest=package(ROOT,record,'stroke-curve-control-manifest/1.0')
    config,*args=context(manifest['config'])
    require(read(folder/'report.json')==score_saved(config,*args,folder), 'Changed directed-arrow/curve metrics')
    return {'verified':True,'new_models_executed':False,'production_changes':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['freeze','build','verify'])
    for name in ('reading','config','manifest','sha256','output'):p.add_argument('--'+name)
    a=p.parse_args()
    if a.command=='freeze':value=freeze(a.reading,ROOT/a.output)
    elif a.command=='build':value=build({'path':a.config,'sha256':a.sha256},ROOT/a.output)
    else:value=verify({'path':a.manifest,'sha256':a.sha256})
    print(json.dumps(value))
