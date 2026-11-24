"""Frozen isolated detector experiment: geometry, data boundaries and decisions."""
from collections import Counter
import math
from pathlib import Path
import shutil

import cv2
import numpy as np

from .dataset_layout_v1 import checked,encoded,pin,read
from .family_catalog_v2 import require
from .layout_region_labels_v1 import CLASSES,load
from .semantic_region_experiment_v1 import iou,match,coarse
from .yolox_model_v1 import ASSETS,SEED

PREVIOUS={'path':'artifacts/layout-region-labels-release-v1.json',
          'sha256':'271f90ff95a87dd7e469dd6b9d99ef0da546e9368290ed68c9780becc1b611f7'}
LABELS={'path':'artifacts/layout-region-labels-v1/manifest.json',
        'sha256':'708a31568919a723a19c9fd92c43dbb6f8836fd50370b30e1d1803d5a4c182b5'}
QUALIFICATION={'path':'artifacts/yolox-qualification-v1/report.json',
               'sha256':'29c0cad23868d115ed9c9af7d9278f21783c3a134a051e02730d7b80fcc79130'}
CONFIG='configs/reviews/yolox-experiment-v1.json'
FOLDER='artifacts/yolox-protocol-v1'
POLICY={'classes':CLASSES,'primary':'rgb','control':'gray','repeats':2,'seed':SEED,
    'side':640,'batch':2,'updates':300,'optimizer':'AdamW','learning_rate':.0003,
    'weight_decay':.01,'warmup_updates':30,'final_learning_rate':.00003,'clip_grad_norm':10.,
    'checkpoint_selection':'fixed_final_update_300_only_no_validation_or_best_checkpoint',
    'schedule':'round_robin_six_train_pages_two_per_update_100_complete_passes',
    'augmentations':{'mosaic':False,'mixup':False,'flip':False,'rotation':False,'crop':False,
                     'perspective':False,'HSV':False,'random_resize':False},
    'preprocessing':'whole_overview_RGB_or_BT601_gray_then_BGR_FP32_0_to_255_top_left_letterbox_white',
    'tiling':{'mode':'one_whole_page','overlap':0,'native_tiles':False,
              'reason':'large layout context boxes exceed local tiles; no clipped object targets'},
    'resizing':'cv2_INTER_LINEAR_truncated_size_actual_independent_xy_edge_ratios',
    'training_FP32':True,'AMP':False,'TF32':False,'EMA':False,'workers':0,'threads':4,
    'deterministic_algorithms':True,'cpu_training_qualified':False,
    'score':'objectness_times_max_class_probability','confidence':.25,'nms_iou':.5,
    'pre_nms_topk':1000,'max_detections':24,'ap_min_score':.001,'ap_max_detections':100,
    'NMS':'class_aware_stable_sort_score_role_box_original_index_CPU',
    'matching':'IoU>=0.5; exact maximum cardinality then total IoU; one-to-one',
    'AP':'confidence_greedy_per_class_per_page; 101_point_interpolated_AP_at_0.50_and_0.50:0.05:0.95',
    'cv_roles':'view_or_detail coarse only; exact view/detail abstain; no invented notes predictions',
    'customer_model_calls_during_preparation':0,'production_admitted':False,'independent_evaluation':False,
    'closed_test_used':False,'physical_geometry_approved':False}
GATES={'iou':.5,'precision_min':.8,'recall_min':.8,'macro_F1_min':.8,'each_class_recall_min':.75,
    'title_block_recall_min':1.,'typed_macro_F1_gain_over_best_CV':.10,'coarse_F1_regression_max':0.,
    'baseline_geometric_targets_newly_missed_max':0,'gray_control_macro_F1_drop_max':.05,
    'gray_control_geometric_TP_drop_max':0,'detail_caption_and_marker_coverage_min':.99,
    'required_repeat_equal':True,'downstream_acceptance_required_for_integration':True}
BUDGETS={'synthetic_worker_seconds':600,'train_worker_seconds':1200,'inference_worker_seconds':180,
    'inference_page_seconds':10,'cpu_inference_page_seconds':15,'host_tree_peak_MiB':6144,
    'cuda_allocated_peak_MiB':4096,'cuda_reserved_peak_MiB':6144,
    'cv_worker_seconds':90,'cv_worker_host_peak_MiB':1024,
    'maximum_input_page_pixels':4194304,'maximum_inference_batch_pixels':24000000,
    'maximum_final_training_updates':300}


def geometry(size,side=640):
    require(len(size)==2 and all(type(v) is int and v>0 for v in size),'Invalid image grid')
    w,h=size; scale=min(side/w,side/h); rw=max(1,int(w*scale)); rh=max(1,int(h*scale))
    sx,sy=rw/w,rh/h
    return {'size_px':size,'side':side,'resized_size_px':[rw,rh],'pad_left_top':[0,0],
        'edges_to_model':[[sx,0,0],[0,sy,0],[0,0,1]],
        'model_to_edges':[[1/sx,0,0],[0,1/sy,0],[0,0,1]],
        'centres_to_model':[[sx,0,(sx-1)/2],[0,sy,(sy-1)/2],[0,0,1]]}


def map_edges(box,matrix,size=None):
    require(len(box)==4 and all(math.isfinite(v) for v in box) and box[0]<box[2] and box[1]<box[3],'Invalid detection box')
    matrix=np.asarray(matrix,float)
    require(matrix.shape==(3,3) and np.isfinite(matrix).all() and np.array_equal(matrix[2],[0,0,1])
            and matrix[0,1]==matrix[1,0]==0 and matrix[0,0]>0 and matrix[1,1]>0,'Positive axis-aligned affine required')
    x,y,r,b=box; result=[x*matrix[0,0]+matrix[0,2],y*matrix[1,1]+matrix[1,2],
                       r*matrix[0,0]+matrix[0,2],b*matrix[1,1]+matrix[1,2]]
    if size: result=[max(0,min(v,size[i%2])) for i,v in enumerate(result)]
    return [float(v) for v in result]


def preprocess(rgb,arm):
    require(arm in ('rgb','gray'),'Unknown image arm')
    require(isinstance(rgb,np.ndarray) and rgb.dtype==np.uint8 and rgb.ndim==3 and rgb.shape[2]==3
            and 0<rgb.shape[0]*rgb.shape[1]<=BUDGETS['maximum_input_page_pixels'],'Bounded RGB overview required')
    spec=geometry(list(rgb.shape[:2][::-1])); rw,rh=spec['resized_size_px']
    image=rgb if arm=='rgb' else cv2.cvtColor(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),cv2.COLOR_GRAY2RGB)
    canvas=np.full((640,640,3),255,dtype=np.uint8)
    canvas[:rh,:rw]=cv2.resize(image,(rw,rh),interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(canvas[:,:,::-1].transpose(2,0,1),dtype=np.float32),spec


def learning_rate(update):
    require(type(update) is int and 0<=update<POLICY['updates'],'Invalid optimizer update')
    if update<POLICY['warmup_updates']:
        return POLICY['learning_rate']*(update+1)/POLICY['warmup_updates']
    progress=(update-POLICY['warmup_updates'])/(POLICY['updates']-POLICY['warmup_updates']-1)
    return POLICY['final_learning_rate']+(.5+.5*math.cos(math.pi*progress))*(POLICY['learning_rate']-POLICY['final_learning_rate'])


def schedule(train_ids):
    require(len(train_ids)==6 and len(set(train_ids))==6,'Exactly six admitted train pages required')
    return [[train_ids[(2*i+j)%6] for j in range(2)] for i in range(POLICY['updates'])]


def nms(rows,*,confidence=.25,maximum=24):
    require(len(rows)<=1000 and maximum in (24,100),'Detection budget exceeded')
    for r in rows:
        require(r['role'] in CLASSES and math.isfinite(r['score']) and 0<=r['score']<=1,'Invalid detection class/score')
        map_edges(r['box'],np.eye(3))
    ranked=sorted(enumerate(rows),key=lambda pair:(-pair[1]['score'],CLASSES.index(pair[1]['role']),pair[1]['box'],pair[0]))
    kept=[]
    for _,row in ranked:
        if row['score']<confidence: continue
        if any(p['role']==row['role'] and iou(p['box'],row['box'])>.5 for p in kept): continue
        kept.append(row)
        if len(kept)==maximum: break
    return kept


def operating_matches(targets,predictions,*,typed):
    require(type(typed) is bool,'Explicit matching mode required')
    candidates=[{'id':p['id'],'role':p['role'] if typed else coarse(p['role']),
                 'native_box':p['box']} for p in predictions]
    return match(targets,candidates,typed=typed)


def validate_policy(protocol):
    require(protocol['policy']==POLICY and protocol['gates']==GATES and protocol['budgets']==BUDGETS,'Experiment changed after registration')
    require(protocol['labels']==LABELS and protocol['assets']==ASSETS and protocol['qualification']==QUALIFICATION,'Data/model qualification rebound')
    require(protocol['customer_training_runs']==protocol['customer_inference_calls']==0
            and protocol['quality_evaluated'] is False,'Preparation became a quality/training result')


def detector_decision(candidate,baselines,gray,*,repeat_equal,resources_pass,context_pass,downstream_pass=False):
    """No candidate selection by validation; classify a predeclared primary only.

    Metrics come from saved, one-to-one evaluated predictions in P09.3e.2.
    Preparation never calls this with customer quality numbers.
    """
    require(set(baselines)=={'legacy_grid','context_cv_rejected_candidate'},'Both frozen CV baselines required')
    require(all(type(v) is bool for v in (repeat_equal,resources_pass,context_pass,downstream_pass)),'Explicit acceptance booleans required')
    for value in [candidate,gray,*baselines.values()]:
        require(all(math.isfinite(value[k]) and 0<=value[k]<=1 for k in ('precision','recall','macro_F1','coarse_F1')),'Invalid evaluation metrics')
        require(set(value['class_recall'])==set(CLASSES) and all(0<=v<=1 for v in value['class_recall'].values()),'Incomplete per-class recall')
        require(type(value['geometric_TP']) is int and value['geometric_TP']>=0,'Invalid geometric count')
    require(type(candidate['baseline_geometric_targets_newly_missed']) is int and candidate['baseline_geometric_targets_newly_missed']>=0,'Missing source-level regression audit')
    checks={
        'repeat_equal':repeat_equal,'resources':resources_pass,'context_preserved':context_pass,
        'precision':candidate['precision']>=GATES['precision_min'],
        'recall':candidate['recall']>=GATES['recall_min'],
        'macro_F1':candidate['macro_F1']>=GATES['macro_F1_min'],
        'each_class_recall':all(v>=GATES['each_class_recall_min'] for v in candidate['class_recall'].values()),
        'title_block_recall':candidate['class_recall']['title_block']==1.,
        'typed_gain':candidate['macro_F1']+1e-12>=max(v['macro_F1'] for v in baselines.values())+.10,
        'coarse_non_regression':candidate['coarse_F1']+1e-12>=max(v['coarse_F1'] for v in baselines.values()),
        'no_new_geometric_misses':candidate['baseline_geometric_targets_newly_missed']==0,
        'gray_control':gray['macro_F1']+1e-12>=candidate['macro_F1']-.05 and gray['geometric_TP']>=candidate['geometric_TP']}
    passed=all(checks.values())
    return {'detector_experiment_passed':passed,'checks':checks,
        'eligible_for_separate_integration_review':passed and downstream_pass,
        'downstream_pass':downstream_pass,'production_admitted':False,'independent_evaluation':False}


def verify_prepared(root,config_pin):
    protocol=read(checked(root,config_pin)); validate_policy(protocol)
    train=load(root,LABELS,split='train',purpose='large_layout_detector_experiment')
    valid=load(root,LABELS,split='validation',purpose='diagnostic_large_layout_detector_evaluation')
    require(set(protocol['inputs'])=={'downstream-coverage.json','inference-train.json','inference-validation.json',
            'lineage.json','projected-size-audit.json','targets.json','train.json'},'Incomplete data registration')
    for p in protocol['inputs'].values(): checked(root,p)
    for p in protocol['baselines'].values(): checked(root,p)
    ledger=read(checked(root,protocol['inputs']['lineage.json']))
    targets=read(checked(root,protocol['inputs']['targets.json']))
    training=read(checked(root,protocol['inputs']['train.json']))
    pages={p['id']:p for p in train['pages']+valid['pages']}
    records=train['records']+valid['records']; case_ids=set(); all_cases={}
    coverage=read(checked(root,protocol['inputs']['downstream-coverage.json']))
    require(coverage['page_OCR_quality_scorable'] is False and coverage['physical_dimension_quality_scorable'] is False
            and {r['source_id'] for r in coverage['sources']}==set(pages)
            and all(r['globally_locatable']==0 for r in coverage['sources']),'Unsupported downstream quality authority')
    require(len(ledger)==len(targets)==10 and len(training['cases'])==6,'Incomplete prepared data')
    by_case={l['id']:l for l in ledger}
    for split in ('train','validation'):
        request=read(checked(root,protocol['inputs'][f'inference-{split}.json']))
        require(set(request)=={'cases'},'Unblinded request envelope')
        require(len(request['cases'])==(6 if split=='train' else 4),'Lost/duplicate split cases')
        for case in request['cases']:
            require(set(case)=={'id','image','size_px'} and case['id'] not in case_ids,'Unblinded or duplicate inference request')
            case_ids.add(case['id']); all_cases[case['id']]=case
            line=by_case[case['id']]; page=pages[line['source_id']]
            require(line['split']==page['split']==split and line['group_id']==page['group_id']
                    and line['source']==page['source'] and case['size_px']==page['overview_size_px'],'Split/source rebound')
            factor=page['factor']
            require(line['native_size_px']==page['size_px'] and line['overview']==page['overview']
                    and line['preparation']==page['preparation']
                    and line['native_to_overview_edges']==[[1/factor,0,0],[0,1/factor,0],[0,0,1]]
                    and line['overview_to_native_edges']==[[factor,0,0],[0,factor,0],[0,0,1]],'Native/overview lineage changed')
            require(case['image']['sha256']==page['overview']['sha256'],'Annotation overlay or changed RGB reached model input')
            checked(root,case['image'])
            require(line['model_geometry']==geometry(case['size_px']),'Letterbox geometry rebound')
    require(case_ids==set(by_case) and len(pages)==len({l['source_id'] for l in ledger}),'Missing source lineage')
    target_by_case={t['id']:t for t in targets}
    require(set(target_by_case)==case_ids,'Missing/duplicate evaluator targets')
    manifest=read(checked(root,LABELS)); links=read(checked(root,manifest['files']['detail-links.json']))
    for identifier,line in by_case.items():
        expected=[r for r in records if r['source_id']==line['source_id']]
        target=target_by_case[identifier]
        require(target['split']==line['split'] and target['regions']==expected
                and target['detail_links']==[l for l in links if l['source_id']==line['source_id']],
                'Targets differ from reviewed release')
    train_ids=[]
    for case in training['cases']:
        require(set(case)=={'id','image','size_px','labels_cxcywh_model_pixels'},'Unexpected train input fields')
        identifier=case['id']; train_ids.append(identifier); line=by_case[identifier]
        require(line['split']=='train' and {k:case[k] for k in all_cases[identifier]}==all_cases[identifier],'Validation entered training')
        page=pages[line['source_id']]; expected=[]
        for row in target_by_case[identifier]['regions']:
            box=map_edges([v/page['factor'] for v in row['bbox']],line['model_geometry']['edges_to_model'])
            x,y,r,b=box; expected.append([CLASSES.index(row['role']),(x+r)/2,(y+b)/2,r-x,b-y])
        require(expected==case['labels_cxcywh_model_pixels'],'Repinned train labels differ from native supervision')
    require(training['schedule']==protocol['training_schedule']==schedule(train_ids),'Training schedule changed')
    return {'pages':10,'train_pages':6,'train_regions':28,'validation_pages':4,'validation_regions':19,
        'training_updates':300,'repeats_per_arm':2,'arms':['rgb','gray'],'customer_model_calls':0,
        'protocol_validated':True,'native_labels_validated':True,'blinded_inference_validated':True,
        'production_admitted':False,'independent_evaluation':False}


def prepare(root):
    folder=root/FOLDER; require(not folder.exists() and not (root/CONFIG).exists(),'Choose a new protocol version')
    # load() finishes group/closed-source screening and validates the reviewed release.
    train=load(root,LABELS,split='train',purpose='large_layout_detector_experiment')
    valid=load(root,LABELS,split='validation',purpose='diagnostic_large_layout_detector_evaluation')
    qualification=read(checked(root,QUALIFICATION)); process=read(root/'artifacts/yolox-qualification-process-v1/process.json')
    require(qualification['qualified'] and qualification['batch']==2 and qualification['side']==640
            and qualification['customer_image_calls']==0 and process['budget_pass'],'YOLOX lacks its own qualification')
    require(qualification['results'][0]['initial']==qualification['results'][1]['initial'],'Unstable pretrained initialization')
    sources=train['pages']+valid['pages']; labels=train['records']+valid['records']
    folder.mkdir(); (folder/'images').mkdir()
    cases=[]; targets=[]; ledger=[]; training=[]; projected=[]
    manifest=read(checked(root,LABELS))
    detail_links=read(checked(root,manifest['files']['detail-links.json']))
    for index,page in enumerate(sources,1):
        identifier=f'page-{index:03d}'; name=f'images/{identifier}.png'; dest=folder/name
        shutil.copyfile(checked(root,page['overview']),dest)
        case={'id':identifier,'image':pin(root,dest.relative_to(root).as_posix()),'size_px':page['overview_size_px']}
        cases.append(case); spec=geometry(case['size_px'])
        rows=[r for r in labels if r['source_id']==page['id']]
        targets.append({'id':identifier,'split':page['split'],'regions':rows,
            'detail_links':[l for l in detail_links if l['source_id']==page['id']]})
        ledger.append({'id':identifier,'source_id':page['id'],'split':page['split'],'group_id':page['group_id'],
            'source':page['source'],'preparation':page['preparation'],'overview':page['overview'],
            'native_size_px':page['size_px'],'native_to_overview_edges':[[1/page['factor'],0,0],[0,1/page['factor'],0],[0,0,1]],
            'overview_to_native_edges':[[page['factor'],0,0],[0,page['factor'],0],[0,0,1]],'model_geometry':spec})
        boxes=[]
        for row in rows:
            overview=[v/page['factor'] for v in row['bbox']]
            box=map_edges(overview,spec['edges_to_model'])
            projected.append({'region_id':row['id'],'split':page['split'],'width':box[2]-box[0],'height':box[3]-box[1]})
            x,y,r,b=box; boxes.append([CLASSES.index(row['role']),(x+r)/2,(y+b)/2,r-x,b-y])
        if page['split']=='train': training.append({**case,'labels_cxcywh_model_pixels':boxes})
    train_ids=[p['id'] for p in training]
    requests={}
    for split in ('train','validation'):
        ids={t['id'] for t in targets if t['split']==split}
        rows=[c for c in cases if c['id'] in ids]
        require(sum(c['size_px'][0]*c['size_px'][1] for c in rows)<=BUDGETS['maximum_inference_batch_pixels'],'Inference batch pixel budget')
        name=f'inference-{split}.json'; (folder/name).write_bytes(encoded({'cases':rows}))
        requests[split]=pin(root,(folder/name).relative_to(root).as_posix())
    (folder/'train.json').write_bytes(encoded({'cases':training,'schedule':schedule(train_ids)}))
    (folder/'targets.json').write_bytes(encoded(targets)); (folder/'lineage.json').write_bytes(encoded(ledger))
    (folder/'projected-size-audit.json').write_bytes(encoded(projected))
    # A new audit records why inherited OCR metrics cannot be presented as page OCR quality.
    ocr_config=read(checked(root,pin(root,'configs/reviews/ocr-model-comparison-v1.json')))
    ocr_rows=read(checked(root,ocr_config['data']['lineage']))
    coverage=[]
    for page in sources:
        matches=[r for r in ocr_rows if r['provenance'].get('source_sha256')==page['source']['sha256']]
        coverage.append({'source_id':page['id'],'matching_ocr_records':len(matches),
            'globally_locatable':sum(r['provenance'].get('context_to_source') is not None and r['provenance'].get('source_box') is not None for r in matches),
            'by_corpus':dict(Counter(r['provenance']['corpus'] for r in matches))})
    (folder/'downstream-coverage.json').write_bytes(encoded({'sources':coverage,
        'page_OCR_quality_scorable':False,'physical_dimension_quality_scorable':False,
        'followup':'P09.3e.2a: targeted globally locatable text/sign/leader annotations before coupled OCR/association acceptance; no GT cropping as detector input',
        'existing_ocr_lineage':ocr_config['data']['lineage']}))
    baselines={p:pin(root,p) for p in ('src/cad_recovery/preprocessing.py','src/cad_recovery/semantic_regions_v1.py',
        'src/cad_recovery/semantic_region_experiment_v1.py','configs/reviews/semantic-region-protocol-v1.json',
        'src/cad_recovery/ocr_engine.py','src/cad_recovery/data/ocr-models-v1.json')}
    runtime='path: '+str(train['yolo_root'])+'\ntrain: images/train\nval: images/validation\nnames:\n'+''.join(f'  {i}: {c}\n' for i,c in enumerate(CLASSES))
    (folder/'runtime-data.yaml').write_text(runtime,encoding='utf-8',newline='\n')
    inputs={p.name:pin(root,p.relative_to(root).as_posix()) for p in sorted(folder.glob('*.json'))}
    protocol={'schema_version':'yolox-protocol/1.0','task':'P09.3e.1','previous':PREVIOUS,
        'labels':LABELS,'assets':ASSETS,'qualification':QUALIFICATION,'qualification_process':pin(root,'artifacts/yolox-qualification-process-v1/process.json'),
        'policy':POLICY,'gates':GATES,'budgets':BUDGETS,'inputs':inputs,'baselines':baselines,
        'train_pages':6,'train_regions':28,'validation_pages':4,'validation_regions':19,
        'training_schedule':schedule(train_ids),'baseline_methods':['legacy_grid','context_cv_rejected_candidate'],
        'small_sample_limit':'diagnostic only; validation table1/detail2; no generalization claim',
        'customer_training_runs':0,'customer_inference_calls':0,'quality_evaluated':False,
        'downstream_quality_gate':'not_scorable_before_P09.3e.2a; no automatic integration',
        'minimum_model_target_width':min(r['width'] for r in projected),
        'minimum_model_target_height':min(r['height'] for r in projected)}
    validate_policy(protocol); (root/CONFIG).write_bytes(encoded(protocol))
    return {'protocol':pin(root,CONFIG),'minimum_width':protocol['minimum_model_target_width'],
            'minimum_height':protocol['minimum_model_target_height'],'requests':requests,'inputs':inputs}
