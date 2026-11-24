"""Opt-in isolated line extraction ablations and native-coordinate diagnostics."""
import hashlib
import importlib.util
import math
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
from PIL import Image,ImageDraw

from . import geometry_detection as detector
from .local_stroke_evidence_v1 import line_width
from .stroke_curve_control_v1 import maximum_matching, sample_polyline, distance_to_polyline
from .dataset_layout_v1 import encoded

VARIANTS=('baseline','zhang','medial','parallel_reference')


def topology(mask):
    mask=(mask>0).astype(np.uint8)
    return {'components':cv2.connectedComponents(mask,connectivity=8)[0]-1,
        'holes':cv2.connectedComponents(np.pad(1-mask,1,constant_values=1),connectivity=4)[0]-2}


def foreground(gray):
    binary=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]
    _,labels,stats,_=cv2.connectedComponentsWithStats(binary,connectivity=8)
    keep=stats[:,cv2.CC_STAT_AREA]>=6;keep[0]=False
    return keep[labels].astype(np.uint8)*255


def extract(root, folder, gray, variant):
    if variant not in VARIANTS:raise ValueError('Unknown experiment variant')
    folder.mkdir(parents=True,exist_ok=False)
    source=folder/'input.png';Image.fromarray(gray).save(source)
    original=detector._skeleton; captured={}
    start=time.perf_counter()
    if variant=='parallel_reference':
        private=folder/'reference';private.mkdir()
        shutil.copyfile(root/'parallel_project/pipeline.py',private/'pipeline.py')
        spec=importlib.util.spec_from_file_location('isolated_parallel_cv',private/'pipeline.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.WORK=str(private/'work')
        ink=foreground(gray)
        lines=module.stage7_lines(ink)
        with Image.open(private/'work/s7_skel.png') as im:skel=255-np.asarray(im)
        candidates=[{'id':f'parallel-{i:05d}','kind':'line','points_px':[[r['p'][0],r['p'][1]],[r['p'][2],r['p'][3]]],
            'reference_width_px':r['w'],'reference_coverage':r['cov']} for i,r in enumerate(lines)]
        stats={'skeleton_iteration_limit':False,'reference_fixed_thresholds':True}
    else:
        def skeleton(ink):
            from skimage.morphology import skeletonize,medial_axis
            if variant=='baseline':result,limited=original(ink)
            elif variant=='zhang':result=skeletonize(ink>0,method='zhang').astype(np.uint8)*255;limited=False
            else:result=medial_axis(ink>0,rng=0).astype(np.uint8)*255;limited=False
            captured.update(ink=ink.copy(),skeleton=result.copy(),limited=limited)
            return result,limited
        detector._skeleton=skeleton
        try:report=detector.detect_geometry(source,folder/'geometry',prefix='cv',search_profile='regional_masked')
        finally:detector._skeleton=original
        ink,skel=captured['ink'],captured['skeleton'];candidates=report['candidates']
        stats={'skeleton_iteration_limit':captured['limited'],'analysis':report['analysis'],
            'issues':report['issues'],'timings':report['timings']}
    extraction_seconds=time.perf_counter()-start
    dt=cv2.distanceTransform(ink,cv2.DIST_L2,5)
    widths=[]
    for c in candidates:
        if c['kind']=='line':
            widths.append({'id':c['id'],**line_width(c['points_px'],ink,dt)})
    Image.fromarray(255-skel).save(folder/'skeleton.png')
    # All overlays use the fixed input frame, never fitted extents.
    preview=Image.fromarray(gray).convert('RGB');draw=ImageDraw.Draw(preview)
    for c in candidates:
        if c['kind']=='line':draw.line([tuple(p) for p in c['points_px']],fill=(230,60,30),width=1)
    preview.save(folder/'overlay.png')
    return {'variant':variant,'candidates':candidates,'widths':widths,'extraction_seconds':extraction_seconds,
        'total_seconds':time.perf_counter()-start,'foreground_sha256':hashlib.sha256(ink.tobytes()).hexdigest(),
        'input_pixel_sha256':hashlib.sha256(gray.tobytes()).hexdigest(),
        'skeleton_sha256':hashlib.sha256(skel.tobytes()).hexdigest(),
        'input_topology':topology(ink),'skeleton_topology':topology(skel),
        'source_mask_changed':False,'role_inference':False,'stats':stats}


def native_points(points, matrix):
    array=np.c_[np.asarray(points,float),np.ones(len(points))]@np.asarray(matrix,float).T
    return array[:,:2]/array[:,2:]


def line_scores(case, result, matrix):
    predictions=[{**c,'native':native_points(c['points_px'],matrix)} for c in result['candidates'] if c['kind']=='line']
    exact=case['scope']=='analytical_control'
    targets=case.get('labels',{}).get('geometry',[]) if exact else []
    targets=[g for g in targets if g['kind']=='line']
    widths={r['id']:r for r in result['widths']}; edges=[]; details=[]
    for t in targets:
        expected=np.asarray(t['points'],float);samples=sample_polyline(expected,.5)
        direction=expected[1]-expected[0];direction/=np.linalg.norm(direction)
        options=[];values={}
        for i,p in enumerate(predictions):
            segment=p['native'];delta=segment[1]-segment[0];length=np.linalg.norm(delta)
            if length==0:continue
            angle=math.degrees(math.acos(float(np.clip(abs(direction@(delta/length)),0,1))))
            forward=distance_to_polyline(samples,segment)
            reverse=distance_to_polyline(sample_polyline(segment,.5),expected)
            coverage=float((forward<=2).mean());p95=float(np.percentile(np.r_[forward,reverse],95))
            endpoints=min(max(np.linalg.norm(segment[0]-expected[0]),np.linalg.norm(segment[1]-expected[1])),
                          max(np.linalg.norm(segment[1]-expected[0]),np.linalg.norm(segment[0]-expected[1])))
            if angle<=5 and coverage>=.8 and p95<=2:
                options.append((p95,i));values[i]={'p95_px':p95,'coverage':coverage,'endpoint_error_px':float(endpoints),
                    'width_native_px':None if widths[p['id']]['width_px'] is None else widths[p['id']]['width_px']*abs(matrix[0][0]),
                    'nominal_width_px':t['stroke_width_px']}
        edges.append([i for _,i in sorted(options)]);details.append(values)
    assignment=maximum_matching(edges)
    matches=[{'target':targets[i]['id'],'candidate':predictions[j]['id'],**details[i][j]} for i,j in assignment.items()]
    # Independent continuous coverage diagnoses fragmented lines without crediting them as a single object.
    all_segments=[p['native'] for p in predictions]
    coverage_rows=[]
    guides=targets if exact else case.get('geometry_guides',[])
    for guide in guides:
        if guide.get('kind') not in (None,'line','polyline') or 'points' not in guide:continue
        samples=sample_polyline(guide['points'],.5)
        distance=np.full(len(samples),np.inf)
        for segment in all_segments:distance=np.minimum(distance,distance_to_polyline(samples,segment))
        tolerance=2 if exact else guide.get('band_px',8)
        coverage_rows.append({'id':guide['id'],'role':guide.get('role'),'coverage':float((distance<=tolerance).mean()),
            'p95_px':None if not np.isfinite(distance).all() else float(np.percentile(distance,95)),
            'target_exact':exact})
    gaps=[]
    for gap in case.get('labels',{}).get('gaps',[]):
        if 'points' not in gap:continue
        samples=sample_polyline(gap['points'],.5)
        bridged=any(np.mean(distance_to_polyline(samples,p)<=2)>=.8 for p in all_segments)
        gaps.append({'id':gap['id'],'bridged':bool(bridged)})
    duplicate=sum(max(0,len(option)-1) for option in edges)
    junctions=[]
    for j in case.get('labels',{}).get('junctions',[]):
        point=np.asarray(j['point']); directions=[]
        for segment in all_segments:
            if distance_to_polyline([point],segment)[0]>2:continue
            for endpoint in segment:
                delta=endpoint-point
                if np.linalg.norm(delta)<=3:continue
                angle=math.atan2(delta[1],delta[0])
                if not any(abs(math.atan2(math.sin(angle-a),math.cos(angle-a)))<math.radians(8) for a in directions):
                    directions.append(angle)
        junctions.append({'point':j['point'],'expected_degree':j['degree'],'observed_degree':len(directions),
                          'preserved':len(directions)==j['degree']})
    return {'targets':len(targets),'predicted_lines':len(predictions),'true_positives':len(matches),
        'false_positives':len(predictions)-len(matches) if exact else None,
        'precision':len(matches)/len(predictions) if exact and predictions else None,
        'recall':len(matches)/len(targets) if targets else None,'matches':matches,
        'duplicate_eligible_candidates':duplicate,'guides':coverage_rows,'gaps':gaps,'junctions':junctions,
        'real_full_precision':None,'source_mask_removed_pixels':0,
        'curves_and_roles':'not evaluated for admission in P10.2c.3a; unchanged curve fitter retained'}
