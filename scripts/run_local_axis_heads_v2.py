"""Paired local-axis/path experiment; frozen development and untouched holdout."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

from run_local_width_heads_v1 import read, write, fingerprint

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/local-axis-heads-v2'
PREVIOUS={'path':'artifacts/head-gate-diagnostics-release-v1.json',
    'sha256':'f57e5051c4a05f13f607abe13cfc48cc4ad135be902da3c3df74061dc203f610'}
CONTROLS={'path':'artifacts/curved-head-controls-v1/contract.json',
    'sha256':'d5e7e2160d214797e42228f468ed5f73e8081d0e485f92ffff735c62a0c10e95'}
CORE={'path':'artifacts/cv-comparison-protocol-v1/contract.json',
    'sha256':'815c8a5ee7f47714066206b2a7a55cbdd94328bb75fdc2ce5383b02ca300d901'}
LARGE=('historical-legacy-val149-upper','historical-legacy-val162-left')


def freeze():
    from cad_recovery.dataset_layout_v1 import checked,pin
    from cad_recovery.local_axis_heads_v2 import POLICY
    from verify_cv_comparison_contract_v1 import verify_contract
    verify_contract(ROOT,CORE);checked(ROOT,PREVIOUS)
    old=read(checked(ROOT,{'path':'artifacts/streamed-heads-v1/plan.json',
        'sha256':'516c4f3e45a12c9b701571838af8bf5579ae82b314be37b4f45a0fe9d2ce8792'}))
    controls=read(checked(ROOT,CONTROLS));checked(ROOT,controls['generator'])
    index={(r['case'],r['mode']):r for r in read(ROOT/'artifacts/streamed-heads-v1/results-index.json') if not r['repeat']}
    jobs=[]
    for job in old['jobs']:
        views=[]
        for view in job['views']:
            row=index[job['case']['id'],view['mode']]
            views.append({'mode':view['mode'],'gray':view['gray'],'input_to_native':view['input_to_native'],
                'comparison_cache':pin(ROOT,row['artifact']),'binding_cache':view.get('binding_cache')})
        jobs.append({'case':job['case'],'views':views,'new_control':False})
    jobs.extend(controls['jobs'])
    names=['src/cad_recovery/local_head_axis_support_v2.py','src/cad_recovery/local_axis_heads_v2.py',
        'src/cad_recovery/local_axis_ownership_v1.py','tests/test_local_axis_heads_v2.py',
        'scripts/run_local_axis_heads_v2.py','src/cad_recovery/streamed_head_profiles_v1.py','src/cad_recovery/streamed_local_heads_v1.py',
        'scripts/prepare_streamed_head_controls_v1.py','scripts/run_streamed_heads_v1.py',
        'scripts/qualify_streamed_heads_v1.py','tests/test_streamed_head_profiles_v1.py']+[p['path'] for p in old['code']]
    value={'task':'P10.2c.2b.7c.2a.3a','previous':PREVIOUS,'core':CORE,'controls':CONTROLS,
        'code':[pin(ROOT,n) for n in dict.fromkeys(names)],'jobs':jobs,'reading':old['reading'],
        'policy':POLICY,'repeats':2,'gates':read(ROOT/'artifacts/local-axis-heads-protocol-v1.json')['primary_gates'],
        'controls_prepared_before_new_rules':True,'old_viewed_validation_used_as_development':True,
        'new_validation_used_for_parameters':False,'new_validation_opened_before_freeze':False,
        'protocol':pin(ROOT,'artifacts/local-axis-heads-protocol-v1.json'),
        'numerics':'Unchanged deterministic float32-map/float64-bilinear/fsum source sampler shared by original profiles and new path sections.',
        'profile_oracle':'Previously sealed streamed sampler tests retained; no new eager corpus calls.',
        'comparison_scope':'Fresh local-axis raw+path ownership on every source; frozen current packets reused on 743 old scenes; original/width/previous-streamed/previous-ownership fully prepared on 72 new scenes.',
        'budgets':{'child_seconds':60,'RSS_MiB':1024,'output_MiB':64,'files':128,'CPU_threads':1,'total_child_seconds':2400},
        'new_OCR_ML_calls':0,'training':False,'closed_test_used':False,'production_changed':False,'physical_geometry_approved':False}
    write(BASE/'plan.json',value);print(json.dumps(pin(ROOT,'artifacts/local-axis-heads-v2/plan.json')))


def child(path):
    sys.path.insert(0,str(ROOT/'artifacts/cv-skeleton-runtime-v1/deps'))
    import cv2
    import numpy as np
    from PIL import Image
    from cad_recovery.dataset_layout_v1 import checked
    from cad_recovery.local_axis_heads_v2 import search
    from cad_recovery.local_axis_ownership_v1 import apply
    from cad_recovery.head_support_ownership_v1 import apply as old_apply
    from cad_recovery.stroke_role_metrics_v1 import head_scores
    from run_stroke_roles_v1 import refs_for
    cv2.setNumThreads(1);cv2.setRNGSeed(0)
    request=read(path);plan=read(checked(ROOT,request['plan']))
    for p in plan['code']:checked(ROOT,p)
    job=request['job'];case=job['case'];folder=path.parent/'output';folder.mkdir();rows=[]
    for view in job['views']:
        gray=np.array(Image.open(checked(ROOT,view['gray'])).convert('L'));sha=hashlib.sha256(gray.tobytes()).hexdigest();prep=None
        if view.get('comparison_cache'):
            old=read(checked(ROOT,view['comparison_cache']))
            if old['input_pixel_sha256']!=sha:raise ValueError('Mixed source raster')
            original,v1,previous_raw,previous_guarded=old['original_candidates'],old['v1_search'],old['streamed_raw'],old['streamed_guarded']
        else:
            from skimage.morphology import skeletonize
            from cad_recovery.streamed_local_heads_v1 import search as old_search
            from cad_recovery.local_width_heads_v1 import search as width_search
            from cad_recovery.baseline_anchor_windows_v1 import extract
            from cad_recovery.arrow_head_completion_v1 import search as completion
            from cad_recovery.native_head_bounds_v1 import guarded_native_heads
            from cad_recovery.stroke_roles_v1 import foreground,stroke_floor
            from run_stroke_ownership_v1 import widths_for
            ink=foreground(gray);skeleton=skeletonize(ink>0,method='zhang').astype(np.uint8)*255
            packet=extract(ink,skeleton,conditioned=False,input_to_native=view['input_to_native']);geometry=packet['candidates']
            prep={'fit_calls':packet['analysis']['fit_calls'],'windows':len(packet['analysis'].get('windows',[])),
                  'search_incomplete':packet['analysis']['search_incomplete']}
            with guarded_native_heads():
                original=completion(gray,lines=[g for g in geometry if g['kind']=='line'],arcs=[g for g in geometry if g['kind']=='arc'],
                    stroke_px=stroke_floor(widths_for(geometry)) or 3.)['candidates']
            v1=width_search(gray,baseline=original);previous_raw=old_search(gray,baseline=v1['candidates'])
            previous_guarded=old_apply(gray,previous_raw,baseline=v1['candidates'])
        baseline=v1['candidates'];before=fingerprint([baseline,previous_raw,previous_guarded])
        started=time.perf_counter();raw=search(gray,baseline=baseline);guarded=apply(gray,raw,baseline=baseline)
        elapsed=time.perf_counter()-started
        eager_raw=eager_guarded=None
        binding=None
        if view.get('binding_cache'):
            from cad_recovery.local_role_guard_v1 import analyze as bind
            saved=view['binding_cache'];parent=read(checked(ROOT,saved['cache']))
            geometry=read(checked(ROOT,saved['geometry']));evidence=parent['previous_interval_details']
            text=np.array(Image.open(checked(ROOT,saved['text_mask'])));roles=np.array(Image.open(checked(ROOT,saved['baseline_roles'])))
            immutable=fingerprint([geometry,evidence])
            attached=bind(gray,evidence,roles,text,guarded['candidates'],diagnostic_heads=parent['head_candidates']['DT_novel'])['hybrid']
            binding={'dimension_chains':attached['dimension_chains'],'search_incomplete':attached['search_incomplete'],
                'source_geometry_unchanged':immutable==fingerprint([geometry,evidence]),'semantic_CAD_promotions':0}
        # Evaluation starts after source processing and binding, never inside the predictor.
        labels=case.get('labels',{})
        if 'arrows' in labels:refs,negatives=labels['arrows'],labels.get('negative_regions',[])
        else:refs,negatives=refs_for(case,read(checked(ROOT,plan['reading'])))
        dense=case['scope'].endswith('analytical_control') or case['scope'] in (
            'local_width_head_control','sparse_head_control','sparse_head_large_control','economical_head_control',
            'economical_head_large_control','local_head_context_control','local_head_context_control_v2',
            'head_support_control','streamed_head_control','streamed_head_large_control','curved_head_control')
        variants={'original':original,'v1':baseline,'previous':previous_guarded['candidates'],
                  'local_axis_raw':raw['candidates'],'local_axis':guarded['candidates']}
        heads={n:head_scores(refs,negatives,h,view['input_to_native'],dense=dense) for n,h in variants.items()}
        row={k:case.get(k) for k in ('scope','group_id','split','cohort')}
        row.update(case=case['id'],mode=view['mode'],repeat=request['repeat'],new_control=job['new_control'],
            input_to_native=view['input_to_native'],input_pixel_sha256=sha,source_pixel_sha256_after=hashlib.sha256(gray.tobytes()).hexdigest(),
            packets_sha256_before=before,packets_sha256_after=fingerprint([baseline,previous_raw,previous_guarded]),
            baseline_exact_prefix_preserved=guarded['candidates'][:len(baseline)]==baseline,
            original_candidates=original,v1_search=v1,previous_raw=previous_raw,previous_guarded=previous_guarded,
            local_axis_raw=raw,local_axis_guarded=guarded,eager_raw=eager_raw,eager_guarded=eager_guarded,
            heads=heads,baseline_preparation=prep,binding=binding,timing={'local_axis_with_guard_seconds':elapsed},
            new_OCR_ML_calls=0,CAD_promotions=0,production_changed=False,physical_geometry_approved=False)
        write(folder/view['mode']/'result.json',row);rows.append(row)
    write(folder/'result.json',{'rows':rows})


def run(sha):
    from experiment_children_v1 import run as execute
    execution=execute(ROOT,BASE,{'path':'artifacts/local-axis-heads-v2/plan.json','sha256':sha},Path(__file__).resolve())
    index=read(BASE/'results-index.json');rows=[read(ROOT/r['artifact']) for r in index];fresh=sum(r['new_control'] for r in rows)
    eager=sum(r['eager_raw'] is not None for r in rows)
    execution.update(local_axis_search_calls=len(rows),eager_reference_calls=eager,
        path_ownership_guard_calls=len(rows)+eager,previous_ownership_guard_calls=fresh,cached_previous_views=len(rows)-fresh,
        fresh_completion_width_previous_streamed_skeleton_calls=fresh,baseline_geometry_dispatches=fresh,
        actual_baseline_geometry_extractions=sum(r['baseline_preparation']['windows'] or 1 for r in rows if r['new_control']),
        binding_guard_calls=sum(bool(r['binding']) for r in rows),new_OCR_ML_calls=0,
        training=False,closed_test_used=False,production_changed=False)
    write(BASE/'execution.json',execution)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['freeze','run','child'])
    parser.add_argument('--sha256');parser.add_argument('--request',type=Path);args=parser.parse_args()
    if args.mode=='freeze':freeze()
    elif args.mode=='child':child(args.request)
    elif args.sha256:run(args.sha256)
    else:parser.error('--sha256 required')
