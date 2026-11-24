"""Fresh streamed detector comparison against frozen packets and a full oracle."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

from run_local_width_heads_v1 import read, write, fingerprint

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/streamed-heads-v1'
PREVIOUS={'path':'artifacts/head-support-ownership-release-v1.json',
    'sha256':'d14ced8b32950b37b810b362b67f67a322482fc4eb379d6f44ddc17e569667bd'}
CONTROLS={'path':'artifacts/streamed-head-controls-v1/contract.json',
    'sha256':'e503ef843e281eb02e461841c95b6ca63ef3bf6134f60ae7390983ab762cd1eb'}
CORE={'path':'artifacts/cv-comparison-protocol-v1/contract.json',
    'sha256':'815c8a5ee7f47714066206b2a7a55cbdd94328bb75fdc2ce5383b02ca300d901'}
LARGE=('historical-legacy-val149-upper','historical-legacy-val162-left')


def freeze():
    from cad_recovery.dataset_layout_v1 import checked,pin
    from cad_recovery.streamed_local_heads_v1 import POLICY
    from verify_cv_comparison_contract_v1 import verify_contract
    verify_contract(ROOT,CORE);checked(ROOT,PREVIOUS)
    old=read(checked(ROOT,{'path':'artifacts/head-support-ownership-v1/plan.json',
        'sha256':'0c9f4a520eb92fb757b4345b3e829ba554b6a2f56b186d6270c85b42456a39d3'}))
    controls=read(checked(ROOT,CONTROLS));checked(ROOT,controls['generator'])
    index={(r['case'],r['mode']):r for r in read(ROOT/'artifacts/head-support-ownership-v1/results-index.json') if not r['repeat']}
    jobs=[]
    for job in old['jobs']:
        views=[]
        for view in job['views']:
            row=index[job['case']['id'],view['mode']]
            views.append({'mode':view['mode'],'gray':view['gray'],'input_to_native':view['input_to_native'],
                'comparison_cache':pin(ROOT,row['artifact']),'binding_cache':view.get('binding_cache')})
        jobs.append({'case':job['case'],'views':views,'new_control':False})
    jobs.extend(controls['jobs'])
    names=['src/cad_recovery/streamed_head_profiles_v1.py','src/cad_recovery/streamed_local_heads_v1.py',
        'scripts/prepare_streamed_head_controls_v1.py','scripts/run_streamed_heads_v1.py',
        'scripts/qualify_streamed_heads_v1.py','tests/test_streamed_head_profiles_v1.py']+[p['path'] for p in old['code']]
    value={'task':'P10.2c.2b.7c.2a.2','previous':PREVIOUS,'core':CORE,'controls':CONTROLS,
        'code':[pin(ROOT,n) for n in dict.fromkeys(names)],'jobs':jobs,'reading':old['reading'],
        'policy':POLICY,'repeats':2,'gates':controls['gates'],
        'controls_prepared_before_profile':True,'parameters_selected_on_viewed_validation':False,
        'numerics':'Same source sampling grid; deterministic float64 bilinear/fsum replaces layout-dependent float32 remap reductions. Legacy drift measured; lazy/full new backend must be bit exact.',
        'profile_oracle':'Eager new backend measures every full row and body using the same unchanged fitter; incomplete references remain incomplete.',
        'comparison_scope':'Fresh streamed raw+ownership on every source; old baselines/previous packets reused, 40 new scenes fully prepared.',
        'budgets':{'child_seconds':60,'RSS_MiB':1024,'output_MiB':64,'files':128,'CPU_threads':1,'total_child_seconds':1800},
        'new_OCR_ML_calls':0,'training':False,'closed_test_used':False,'production_changed':False,'physical_geometry_approved':False}
    write(BASE/'plan.json',value);print(json.dumps(pin(ROOT,'artifacts/streamed-heads-v1/plan.json')))


def child(path):
    sys.path.insert(0,str(ROOT/'artifacts/cv-skeleton-runtime-v1/deps'))
    import cv2
    import numpy as np
    from PIL import Image
    from cad_recovery.dataset_layout_v1 import checked
    from cad_recovery.streamed_local_heads_v1 import search
    from cad_recovery.head_support_ownership_v1 import apply
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
            original,v1,previous_raw,previous_guarded=old['original_candidates'],old['v1_search'],old['raw_search'],old['guarded_search']
        else:
            from skimage.morphology import skeletonize
            from cad_recovery.local_head_context_v2 import search as old_search
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
            previous_guarded=apply(gray,previous_raw,baseline=v1['candidates'])
        baseline=v1['candidates'];before=fingerprint([baseline,previous_raw,previous_guarded])
        started=time.perf_counter();raw=search(gray,baseline=baseline);guarded=apply(gray,raw,baseline=baseline)
        elapsed=time.perf_counter()-started
        eager_raw=eager_guarded=None
        if job['new_control'] or case['id'] in LARGE:
            eager_raw=search(gray,baseline=baseline,eager_reference=True);eager_guarded=apply(gray,eager_raw,baseline=baseline)
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
            'head_support_control','streamed_head_control','streamed_head_large_control')
        variants={'original':original,'v1':baseline,'previous':previous_guarded['candidates'],
                  'streamed_raw':raw['candidates'],'streamed':guarded['candidates']}
        heads={n:head_scores(refs,negatives,h,view['input_to_native'],dense=dense) for n,h in variants.items()}
        row={k:case.get(k) for k in ('scope','group_id','split','cohort')}
        row.update(case=case['id'],mode=view['mode'],repeat=request['repeat'],new_control=job['new_control'],
            input_to_native=view['input_to_native'],input_pixel_sha256=sha,source_pixel_sha256_after=hashlib.sha256(gray.tobytes()).hexdigest(),
            packets_sha256_before=before,packets_sha256_after=fingerprint([baseline,previous_raw,previous_guarded]),
            baseline_exact_prefix_preserved=guarded['candidates'][:len(baseline)]==baseline,
            original_candidates=original,v1_search=v1,previous_raw=previous_raw,previous_guarded=previous_guarded,
            streamed_raw=raw,streamed_guarded=guarded,eager_raw=eager_raw,eager_guarded=eager_guarded,
            heads=heads,baseline_preparation=prep,binding=binding,timing={'streamed_with_guard_seconds':elapsed},
            new_OCR_ML_calls=0,CAD_promotions=0,production_changed=False,physical_geometry_approved=False)
        write(folder/view['mode']/'result.json',row);rows.append(row)
    write(folder/'result.json',{'rows':rows})


def run(sha):
    from experiment_children_v1 import run as execute
    execution=execute(ROOT,BASE,{'path':'artifacts/streamed-heads-v1/plan.json','sha256':sha},Path(__file__).resolve())
    index=read(BASE/'results-index.json');rows=[read(ROOT/r['artifact']) for r in index];fresh=sum(r['new_control'] for r in rows)
    eager=sum(r['eager_raw'] is not None for r in rows)
    execution.update(streamed_search_calls=len(rows),eager_reference_calls=eager,
        ownership_guard_calls=len(rows)+eager+fresh,cached_previous_views=len(rows)-fresh,
        fresh_completion_v1_oldraw_skeleton_calls=fresh,baseline_geometry_dispatches=fresh,
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
