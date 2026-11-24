"""Frozen train-ROI classification and positive-only validation localization."""
from pathlib import Path
import math
import time
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image

from .dataset_layout_v1 import checked, encoded, pin, read
from .family_catalog_v2 import require
from .page_region_review_v1 import context, authenticated_source, INPUTS
from .page_region_release_v1 import load
from .semantic_regions_v1 import PARAMETERS, classify, legacy, candidate, map_box
from .hard_raster_runs_v1 import process_peaks
from .service_v1.process import ChildGuard

LABELS = {'path':'artifacts/page-region-labels-v1/manifest.json',
          'sha256':'e408eda44e91a6e9dbb90004ce69ce2f5872da330268571fa6b4ac8380f2c7e6'}
PREVIOUS = {'path':'artifacts/page-region-release-v1.json',
            'sha256':'c124de65975690308b241d0aea831c3778ccc548aadfb251ec4518b3382a4917'}
CONFIG = 'configs/reviews/semantic-region-protocol-v1.json'
CODE = ['src/cad_recovery/semantic_regions_v1.py','src/cad_recovery/semantic_region_experiment_v1.py',
        'src/cad_recovery/preprocessing.py','scripts/semantic_region_worker_v1.py',
        'scripts/run_semantic_regions_v1.py','src/cad_recovery/hard_raster_runs_v1.py',
        'src/cad_recovery/service_v1/process.py']
GATES = {'iou':.5,'positive_recall':.8,'title_recall':1.,'geometric_context_recall':.75,
         'max_worker_seconds':90,'max_case_seconds':8,'max_worker_rss_mib':1024,
         'exact_typed_semantic_recall':.8}


def protected(root):
    old = read(checked(root,PREVIOUS)); values = old['prior_protected_pins']|old['files']|{PREVIOUS['path']:PREVIOUS['sha256']}
    for path in (root/'artifacts/external-method-review-v1').rglob('*'):
        if path.is_file():values[path.relative_to(root).as_posix()] = pin(root,path.relative_to(root).as_posix())['sha256']
    name = 'docs/external-cad-methods-review-v1.md'; values[name] = pin(root,name)['sha256']
    for name,sha in values.items():checked(root,{'path':name,'sha256':sha})
    return values


def prepare(root,folder):
    require(not folder.exists() and not (root/CONFIG).exists(),'Choose new frozen experiment paths')
    previous = protected(root)
    # Screen all sources and labels before opening even a prepared overview.
    _,sources,excluded = context(root); require(not excluded,'Changed source boundary')
    train = load(root,LABELS,split='train',purpose='positive_roi_classification')
    valid = load(root,LABELS,split='validation',purpose='diagnostic_positive_box_localization')
    reports = {s['id']:authenticated_source(root,s) for s in sources}
    protocol = {'schema_version':'semantic-region-protocol/1.0','labels':LABELS,'selection':INPUTS,
        'parameters':PARAMETERS,'gates':GATES,'code':{p:pin(root,p)['sha256'] for p in CODE},
        'methods':['legacy_grid','context_cv'],'train_use':'positive_roi_classification_only',
        'validation_use':'known_positive_context_box_localization_only','tuning_runs':0,
        'targets_blinded_from_worker':True,'classification':'title_block/table/view_or_detail/unknown',
        'detail_subtype':'unresolved_without_OCR; typed view and detail both abstain',
        'matching':'one-to-one maximum-cardinality at IoU>=threshold, then maximum total IoU',
        'unknown_policy':'unmatched predictions are unscored, never false positives',
        'independent_evaluation':False,'already_reviewed_sources':True,'dense_training_admitted':False,
        'physical_geometry_approved':False,'production_admitted':False,
        'previous_pins':len(previous),'previous':PREVIOUS}
    (root/CONFIG).write_bytes(encoded(protocol))
    folder.mkdir(); (folder/'images').mkdir()
    cases = []; targets = []; ledger = []; index = 0; total = 0
    for s in sources:
        prep,report = reports[s['id']]; o = report['overview']; factor = o['scale_factor']
        p = checked(root,{'path':(prep.parent/o['path']).relative_to(root).as_posix(),'sha256':o['sha256']})
        with Image.open(p) as im:
            require(im.mode == 'RGB' and im.width*im.height <= PARAMETERS['max_pixels'],'Invalid bounded overview')
            overview = im.copy()
        selected = [r for r in train if r['source_id'] == s['id']] if s['split']=='train' else [None]
        for label in selected:
            index += 1; identifier = f'case-{index:02d}'
            box = ([math.floor(label['bbox'][0]/factor),math.floor(label['bbox'][1]/factor),
                    math.ceil(label['bbox'][2]/factor),math.ceil(label['bbox'][3]/factor)]
                   if label else [0,0,*overview.size])
            image = overview.crop(box); path = folder/'images'/f'{identifier}.png'; image.save(path)
            x,y = box[:2]; source_to_input = np.asarray(o['source_to_overview'],dtype=float)
            source_to_input = np.array([[1,0,-x],[0,1,-y],[0,0,1]])@source_to_input
            edge = [[factor,0,factor*x],[0,factor,factor*y],[0,0,1]]
            cases.append({'id':identifier,'image':pin(root,path.relative_to(root).as_posix()),
                'size_px':list(image.size),'source_size_px':s['size_px'],
                'source_to_input_centres':source_to_input.tolist(),
                'input_to_source_centres':np.linalg.inv(source_to_input).tolist(),
                'input_to_source_edges':edge,'source_to_input_edges':np.linalg.inv(edge).tolist(),
                'quarter_turns_ccw':0,'task':'classify_roi' if label else 'localize_page'})
            targets.append({'id':identifier,'split':s['split'],'source_id':s['id'],'group_id':s['group_id'],
                'labels':[label] if label else [r for r in valid if r['source_id']==s['id']]})
            ledger.append({'case_id':identifier,'source':s['source'],'preparation':s['preparation'],
                           'overview':{'path':p.relative_to(root).as_posix(),'sha256':o['sha256']},
                           'overview_box':box,'sampling':o['method'],'integer_factor':factor})
            total += image.width*image.height
    require(len(cases)<=PARAMETERS['max_cases'] and total<=PARAMETERS['max_total_pixels'],'Input budget exceeded')
    value = {'schema_version':'semantic-region-inputs/1.0','protocol':pin(root,CONFIG),'cases':cases}
    (folder/'requests.json').write_bytes(encoded(value)); (folder/'targets.json').write_bytes(encoded(targets))
    (folder/'lineage.json').write_bytes(encoded(ledger))
    return {'requests':pin(root,(folder/'requests.json').relative_to(root).as_posix()),
            'targets':pin(root,(folder/'targets.json').relative_to(root).as_posix()),
            'protocol':pin(root,CONFIG),'total_pixels':total,'previous_pins':previous}


def validated_protocol(root,protocol_pin):
    protocol = read(checked(root,protocol_pin))
    require(protocol['parameters']==PARAMETERS and protocol['gates']==GATES and protocol['labels']==LABELS,
            'Frozen parameters changed')
    require(protocol['tuning_runs']==0 and protocol['independent_evaluation'] is False
            and protocol['production_admitted'] is False,'Protocol authority changed')
    for name,sha in protocol['code'].items():checked(root,{'path':name,'sha256':sha})
    require(set(protocol['code'])==set(CODE),'Incomplete code pins')
    return protocol


def worker(root,request_pin,output,task):
    """Only opaque image requests; no labels, split, source name or group."""
    require(task in ('classify_roi','localize_page') and not output.exists(),'Invalid worker output/task')
    request = read(checked(root,request_pin)); validated_protocol(root,request['protocol'])
    require(set(request)=={'schema_version','protocol','cases'} and request['schema_version']=='semantic-region-inputs/1.0','Unblinded request')
    cases = request['cases']; require(len(cases)<=PARAMETERS['max_cases'],'Too many cases')
    require(len({c['id'] for c in cases})==len(cases),'Duplicate case IDs')
    allowed = {'id','image','size_px','source_size_px','source_to_input_centres','input_to_source_centres',
               'input_to_source_edges','source_to_input_edges','quarter_turns_ccw','task'}
    total = 0
    for case in cases:
        require(set(case)==allowed and case['task'] in ('classify_roi','localize_page')
                and case['quarter_turns_ccw']==0,'Unblinded or unsupported case')
        w,h=case['size_px']; require(type(w)is int and type(h)is int and 0<w<=32768 and 0<h<=32768
                                   and w*h<=PARAMETERS['max_pixels'],'Case size budget')
        total += w*h
        for forward,inverse in [('source_to_input_centres','input_to_source_centres'),('source_to_input_edges','input_to_source_edges')]:
            a=np.asarray(case[forward],float); b=np.asarray(case[inverse],float)
            require(a.shape==b.shape==(3,3) and np.isfinite(a).all() and np.isfinite(b).all()
                    and np.allclose(a@b,np.eye(3),atol=1e-10),'Non-invertible coordinate transform')
    require(total<=PARAMETERS['max_total_pixels'],'Worker input budget')
    cv2.setNumThreads(1); cv2.setRNGSeed(0); rows=[]; started=time.perf_counter()
    for case in cases:
        if case['task']!=task:continue
        begin=time.perf_counter()
        with Image.open(checked(root,case['image'])) as im:
            require(im.mode=='RGB' and list(im.size)==case['size_px'],'Image dimensions differ')
            rgb=np.asarray(im).copy()
        if task=='classify_roi':
            prediction={'classification':classify(rgb)}
        else:
            arms={}
            for name,algorithm in [('legacy_grid',legacy),('context_cv',candidate)]:
                candidates=algorithm(rgb)
                for i,row in enumerate(candidates,1):
                    row['id']=f'{case["id"]}-{name}-{i:02d}'
                    row['native_box']=map_box(row['box'],case['input_to_source_edges'],case['source_size_px'])
                    row['transform']={k:case[k] for k in allowed if 'centres' in k or 'edges' in k}
                arms[name]={'candidates':candidates,'incomplete':len(candidates)>=PARAMETERS['max_candidates']}
            prediction={'arms':arms}
        elapsed=time.perf_counter()-begin
        require(elapsed<=GATES['max_case_seconds'],'Case time budget exceeded')
        rows.append({'id':case['id'],'prediction':prediction,'seconds':elapsed})
    value={'schema_version':'semantic-region-worker/1.0','request':request_pin,'protocol':request['protocol'],
           'task':task,'rows':rows,'seconds':time.perf_counter()-started,'model_calls':0,'training_runs':0}
    output.write_bytes(encoded(value))


def run_worker(root,spec,output,task):
    command=[sys.executable,str(root/'scripts/semantic_region_worker_v1.py'),
             spec['requests']['path'],spec['requests']['sha256'],str(output),task]
    started=time.perf_counter(); peak=0; pid_peaks={}
    with (output.with_suffix('.log')).open('wb') as log:
        import os
        environment=os.environ.copy(); environment.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        process=subprocess.Popen(command,cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=log,env=environment,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        guard=None
        try:
            guard=ChildGuard(process)
            while True:
                for pid,value in process_peaks(process,guard).items():pid_peaks[pid]=max(pid_peaks.get(pid,0),value)
                peak=sum(pid_peaks.values())
                require(time.perf_counter()-started<=GATES['max_worker_seconds'],'Worker wall time exceeded')
                require(peak<=GATES['max_worker_rss_mib']*1024**2,'Worker RSS budget exceeded')
                if process.poll() is not None:break
                time.sleep(.05)
        except BaseException:
            process.kill(); process.wait(); raise
        finally:
            if guard:guard.close()
    require(process.returncode==0,'CV worker failed; inspect '+str(output.with_suffix('.log')))
    require(bool(pid_peaks),'Missing OS memory evidence')
    return {'wall_seconds':time.perf_counter()-started,'observed_tree_peak_rss_mib':peak/1024**2,
            'observed_process_count':len(pid_peaks),
            'memory_measurement':'sum of observed per-PID OS lifetime peaks, 50ms sampling; not simultaneous RSS; short-lived children can be missed'}


def iou(a,b):
    overlap=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-overlap
    return overlap/union if union>0 else 0.


def coarse(role):
    return 'view_or_detail' if role in ('view','detail') else role


def match(labels,candidates,*,typed=False):
    """One candidate cannot satisfy several positives; invalid pairs cost zero."""
    if not labels:return []
    require(len(labels)<=PARAMETERS['max_match_targets'] and len(candidates)<=PARAMETERS['max_candidates'],
            'Matching budget exceeded')
    scores=np.zeros((len(labels),len(candidates)))
    for i,label in enumerate(labels):
        for j,p in enumerate(candidates):
            value=iou(label['bbox'],p['native_box'])
            compatible=(label['role']==p['role']) if typed else (coarse(label['role'])==p['role'])
            if compatible and value>=GATES['iou']:scores[i,j]=len(labels)+1+value
    # Bounded subset DP: no optional SciPy dependency; exact maximum matching
    # for the small admitted per-page positive set, including dummy unmatched.
    states={0:(0.,{})}
    for j in range(len(candidates)):
        updated=dict(states)
        for mask,(value,assignment) in states.items():
            for i in range(len(labels)):
                if mask&(1<<i) or scores[i,j]<=0:continue
                next_mask=mask|(1<<i); score=value+float(scores[i,j])
                if next_mask not in updated or score>updated[next_mask][0]:
                    updated[next_mask]=(score,{**assignment,i:j})
        states=updated
    chosen=max(states,key=lambda mask:(mask.bit_count(),states[mask][0],-mask))
    selected=states[chosen][1]
    return [{'label_id':r['id'],'role':r['role'],'matched':i in selected,
             'candidate_id':candidates[selected[i]]['id'] if i in selected else None,
             'iou':iou(r['bbox'],candidates[selected[i]]['native_box']) if i in selected else 0.,
             'best_compatible_iou':max((iou(r['bbox'],p['native_box']) for p in candidates
                    if ((r['role']==p['role']) if typed else (coarse(r['role'])==p['role']))),default=0.)}
            for i,r in enumerate(labels)]


def evaluate(root,spec,classification,localization):
    validated_protocol(root,spec['protocol']); targets=read(checked(root,spec['targets']))
    train={r['id']:r for r in targets if r['split']=='train'}; valid={r['id']:r for r in targets if r['split']=='validation'}
    require({r['id'] for r in classification['rows']}==set(train)
            and {r['id'] for r in localization['rows']}==set(valid),'Missing or mixed assignments')
    classification_rows=[]
    for row in classification['rows']:
        target=train[row['id']]['labels'][0]; pred=row['prediction']['classification']['role']
        classification_rows.append({'id':row['id'],'label_id':target['id'],'role':target['role'],'predicted':pred,
                                    'coarse_correct':coarse(target['role'])==pred,'typed_correct':target['role']==pred})
    arms={}
    for name in ('legacy_grid','context_cv'):
        rows=[]; typed=[]; unscored=0; incomplete=False
        for row in localization['rows']:
            arm=row['prediction']['arms'][name]; p=arm['candidates']; labels=valid[row['id']]['labels']
            found=match(labels,p); exact=match(labels,p,typed=True)
            rows += [{'case_id':row['id'],**r} for r in found]; typed += exact
            unscored += len(p)-sum(r['matched'] for r in found); incomplete |= arm['incomplete']
        by_role={role:{'known':sum(r['role']==role for r in rows),
                      'matched':sum(r['role']==role and r['matched'] for r in rows)} for role in ('view','detail','title_block')}
        count=sum(r['matched'] for r in rows); exact_count=sum(r['matched'] for r in typed)
        geometric=[r for r in rows if r['role'] in ('view','detail')]; title=by_role['title_block']
        gates={'positive_recall':count/len(rows)>=GATES['positive_recall'],
               'title_recall':title['matched']/title['known']>=GATES['title_recall'],
               'geometric_context_recall':sum(r['matched'] for r in geometric)/len(geometric)>=GATES['geometric_context_recall'],
               'exact_typed_semantic_recall':exact_count/len(typed)>=GATES['exact_typed_semantic_recall'],
               'complete_search':not incomplete}
        arms[name]={'rows':rows,'typed_rows':typed,'known_positives':len(rows),'matched':count,
                    'positive_recall':count/len(rows),'by_role':by_role,'typed_matched':exact_count,
                    'mean_matched_iou':sum(r['iou'] for r in rows)/count if count else None,
                    'unmatched_predictions_unscored':unscored,'incomplete':incomplete,'gates':gates,
                    'decision':'diagnostic_gates_passed' if all(gates.values()) else 'reject_automatic_semantic_route'}
    return {'schema_version':'semantic-region-evaluation/1.0','labels':LABELS,'protocol':spec['protocol'],
        'train_roi_classification':{'rows':classification_rows,'coarse_correct':sum(r['coarse_correct'] for r in classification_rows),
                                    'typed_correct':sum(r['typed_correct'] for r in classification_rows),'count':len(classification_rows)},
        'validation':arms,'full_page_precision':None,'AP':None,'table_validation':'not_evaluable_no_labels',
        'frame_validation':'not_evaluable_no_labels','independent_evaluation':False,'model_calls':0,'training_runs':0,
        'physical_geometry_approved':False,'production_admitted':False}
