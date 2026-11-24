"""Targeted downstream diagnostics; native labels never enter runtime requests."""
from datetime import datetime, timezone
import math
import numpy as np
from PIL import Image
from .dataset_layout_v1 import checked, encoded, pin, read
from .family_catalog_v2 import require
from .ocr_baseline_metrics_v1 import normalize_text_v1, score_text_v1

FOLDER='artifacts/joint-region-comparison-v1'
LABELS='artifacts/joint-native-labels-v1'
PREVIOUS={'path':'artifacts/yolox-comparison-release-v1.json','sha256':'0984921606cfc0fbcc503d31d9e539c2e8aec2c9d4bc8a0640d91ab9a15ea362'}
PARENT={'path':'configs/reviews/yolox-experiment-v1.json','sha256':'67e98548fafeb76c50d29046f2b4a7f2761341e28413903d5fb3a7bec5262c6f'}
METHODS={'whole_page':None,'legacy_grid':'legacy_grid-validation-0',
 'context_cv':'context_cv_rejected_candidate-validation-0',
 'yolox_rgb':'infer-rgb-0-validation','yolox_gray':'infer-gray-0-validation'}
CODE=['src/cad_recovery/joint_region_comparison_v1.py','scripts/joint_region_worker_v1.py',
 'scripts/run_joint_region_comparison_v1.py','src/cad_recovery/preprocessing.py',
 'src/cad_recovery/ocr_engine.py','src/cad_recovery/ocr.py','src/cad_recovery/geometry_detection.py',
 'src/cad_recovery/arrow_detection.py','src/cad_recovery/dimension_association.py',
 'src/cad_recovery/ocr_baseline_metrics_v1.py']
POLICY={'task':'P09.3e.2a','repeats':2,'inference_split':'validation','resolution':'frozen YOLO overview',
 'training':False,'test_inference':False,'parameter_search':False,'physical_geometry_approved':False,
 'independent_evaluation':False,'production_admitted':False,'auto_deskew':False,
 'search_profile':'regional_masked','tile_size':1536,'overlap':192,'ocr_time_budget_seconds':240,
 'worker_seconds':900,'worker_memory_MiB':3072,'spatial_matching_iou':.5,
 'matching':'descending IoU then confidence then IDs; one-to-one; no target text selection',
 'duplicate_policy':'same raw and IoU >= .8; highest confidence retained across crops; conflicts retained',
 'unlabelled':'unknown, never negative; no full-page OCR precision',
 'sign_score':'aligned normalized character at known text occurrence; not symbol localization accuracy',
 'association_score':'only proposed links on spatially matched selected texts; wrong known target or known non-dimension; other targets unknown',
 'crop_policy':'all frozen operating proposals/all roles; floor/ceil outward to overview edges; no GT crop or padding',
 'context':'full text and complete observed leader polyline in one crop; features separately retained as partial GT',
 'neighbour_capture':'per predicted view/detail ROI, count other reviewed view/detail boxes with >=10% target area after dominant overlap',
 'decision':'diagnostic only; rejected YOLO remains rejected; whole_page is control, legacy_grid is ROI ablation'}

def intersection(a,b):
    return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))

def area(b):return max(0,b[2]-b[0])*max(0,b[3]-b[1])

def iou(a,b):
    x=intersection(a,b);return x/(area(a)+area(b)-x) if area(a)+area(b)-x else 0

def crop_box(native_box,factor,size):
    b=[max(0,math.floor(native_box[0]/factor)),max(0,math.floor(native_box[1]/factor)),
       min(size[0],math.ceil(native_box[2]/factor)),min(size[1],math.ceil(native_box[3]/factor))]
    require(area(b)>0 and all(math.isfinite(x) for x in b),'Invalid crop')
    return b

def native_points(points,box,factor):
    a=np.asarray(points,dtype=float)
    require(a.ndim==2 and a.shape[1]==2 and np.isfinite(a).all(),'Invalid pixel centres')
    return ((a+np.array(box[:2]))*factor+(factor-1)/2).tolist()

def polygon_box(points):
    p=np.asarray(points,float)+.5
    return [float(p[:,0].min()),float(p[:,1].min()),float(p[:,0].max()),float(p[:,1].max())]

def validate_request(request):
    require(set(request)=={'cases'},'Unblinded request')
    ids=[]
    for c in request['cases']:
        require(set(c)=={'id','image','size_px','crops'},'Unblinded case')
        ids.append(c['id']);w,h=c['size_px'];require(0<w*h<=4_000_000,'Overview pixel budget')
        seen=set()
        for r in c['crops']:
            require(set(r)=={'id','bbox','proposal_ids','role'},'Unblinded crop')
            b=r['bbox'];require(len(b)==4 and all(type(x) is int for x in b)
                and 0<=b[0]<b[2]<=w and 0<=b[1]<b[3]<=h,'Invalid crop coordinates')
            require(r['id'] not in seen,'Repeated crop ID');seen.add(r['id'])
    require(len(ids)==len(set(ids))==4,'Lost validation page')

def protected(root):
    previous=read(checked(root,PREVIOUS))
    records=previous['prior_protected_pins']|previous['files']|{PREVIOUS['path']:PREVIOUS['sha256']}
    for name,sha in records.items():checked(root,{'path':name,'sha256':sha})
    return records

def prepare(root):
    folder=root/FOLDER;require(not folder.exists(),'Choose new execution version')
    prior=protected(root)
    from .yolox_protocol_v1 import verify_prepared
    verify_prepared(root,PARENT)
    acceptance=read(root/LABELS/'acceptance.json')
    for ref in acceptance['payloads']:checked(root,ref)
    sources=read(root/LABELS/'sources.json');texts=read(root/LABELS/'texts.json')
    require(len(texts)==22 and sum(t['split']=='validation' for t in texts)==16,'Changed target registry')
    require(not ({p['group_id'] for p in sources if p['split']=='train'} &
                 {p['group_id'] for p in sources if p['split']=='validation'}),'Cross-split family')
    lineage=read(root/'artifacts/yolox-protocol-v1/lineage.json')
    preds=read(root/'artifacts/yolox-comparison-v2/decoded-predictions.json')
    folder.mkdir();(folder/'requests').mkdir();jobs=[]
    for method,key in METHODS.items():
        cases=[]
        for source in sources:
            if source['split']!='validation':continue
            line=next(x for x in lineage if x['source_id']==source['id'])
            require(line['source']==source['source'] and line['group_id']==source['group_id'],'Rebound ancestry')
            overview=checked(root,source['overview'])
            with Image.open(overview) as image:size=list(image.size)
            crops=[];by_box={}
            proposals=[{'id':'full','role':'whole_page','box':[0,0,*source['size_px']]}] if key is None else preds[key]['operating'][line['id']]
            for p in proposals:
                b=crop_box(p['box'],source['factor'],size)
                if tuple(b) in by_box:
                    by_box[tuple(b)]['proposal_ids'].append(p['id']);continue
                r={'id':f'roi-{len(crops)+1:03d}','bbox':b,'proposal_ids':[p['id']],'role':p['role']}
                by_box[tuple(b)]=r;crops.append(r)
            cases.append({'id':line['id'],'image':source['overview'],'size_px':size,'crops':crops})
        request={'cases':cases};validate_request(request)
        path=folder/'requests'/f'{method}.json';path.write_bytes(encoded(request))
        for repeat in range(2):jobs.append({'id':f'{method}-{repeat}','method':method,'repeat':repeat,
                                           'request':pin(root,path.relative_to(root).as_posix())})
    inputs=[pin(root,str(p.relative_to(root).as_posix())) for p in sorted((root/LABELS).rglob('*')) if p.is_file()]
    models=[pin(root,p.relative_to(root).as_posix()) for p in sorted((root/'artifacts/models/ppocr-v5').rglob('*.onnx'))]
    config={'schema_version':'joint-region-comparison/1.0','created_before_inference_utc':datetime.now(timezone.utc).isoformat(),
        'policy':POLICY,'previous':PREVIOUS,'parent':PARENT,'code':[pin(root,n) for n in CODE],
        'models':models,'inputs':inputs,'prior_protected_pins':prior,'jobs':jobs,
        'lineage':pin(root,'artifacts/yolox-protocol-v1/lineage.json'),
        'predictions':pin(root,'artifacts/yolox-comparison-v2/decoded-predictions.json'),
        'roi_counts':{m:sum(len(c['crops']) for c in read(root/FOLDER/'requests'/f'{m}.json')['cases']) for m in METHODS}}
    (folder/'protocol.json').write_bytes(encoded(config));return config

def authenticate(root,record):
    config=read(checked(root,record));require(config['policy']==POLICY,'Changed fixed policy')
    require(config['code']==[pin(root,n) for n in CODE],'Producer/evaluator changed')
    for p in config['models']+config['inputs']:checked(root,p)
    for job in config['jobs']:validate_request(read(checked(root,job['request'])))
    return config

def deduplicate(lines):
    kept=[]
    for line in sorted(lines,key=lambda x:(-x['confidence'],x['id'])):
        if not any(line['raw']==p['raw'] and iou(line['bbox'],p['bbox'])>=.8 for p in kept):kept.append(line)
    return kept

def spatial_match(texts,lines):
    options=[(-iou(t['bbox'],p['bbox']),-p['confidence'],t['id'],p['id'])
             for t in texts for p in lines if iou(t['bbox'],p['bbox'])>=.5]
    result={};used=set()
    for _,_,tid,pid in sorted(options):
        if tid not in result and pid not in used:result[tid]=pid;used.add(pid)
    return result

def max_coverage(box,crops):
    return max((intersection(box,b)/area(box) for b in crops),default=0)

def segment_fraction(a,b,box):
    """Liang-Barsky interval, no point sampling bias on a clipped leader."""
    t0,t1=0.,1.;dx=b[0]-a[0];dy=b[1]-a[1]
    for p,q in [(-dx,a[0]-box[0]),(dx,box[2]-a[0]),(-dy,a[1]-box[1]),(dy,box[3]-a[1])]:
        if p==0:
            if q<0:return 0.
        elif p<0:t0=max(t0,q/p)
        else:t1=min(t1,q/p)
    return max(0.,t1-t0)

def leader_coverage(polyline,box):
    lengths=[math.dist(a,b) for a,b in zip(polyline,polyline[1:])]
    return sum(length*segment_fraction(a,b,box) for a,b,length in zip(polyline,polyline[1:],lengths))/sum(lengths)

def point_segment_distance(p,segment):
    p=np.array(p);a,b=np.array(segment);d=b-a
    t=np.clip(np.dot(p-a,d)/max(1e-12,np.dot(d,d)),0,1)
    return float(np.linalg.norm(p-a-t*d))

def correct_relation(link,association):
    if association.get('kind')!=link['kind']:return False
    anchors=association.get('native_anchors',[]);tol=link['tolerance_px']
    if len(anchors)!=2:return False
    if link['scope']=='observed_circle':
        cx,cy,r=link['circle'];center=np.array([cx,cy]);a,b=np.array(anchors)
        return (np.linalg.norm((a+b)/2-center)<=tol and
                abs(np.linalg.norm(a-b)/2-r)<=tol)
    if link['scope']=='leader_contact_only':return min(math.dist(p,link['contact']) for p in anchors)<=tol
    features=link['features']
    return any(max(point_segment_distance(anchors[0],features[i]),
                   point_segment_distance(anchors[1],features[1-i]))<=tol for i in (0,1))

def character_preserved(reference,prediction,literal):
    # Frozen normalizer only; no decimal/script/diameter guesses in model output.
    r=normalize_text_v1(reference);p=normalize_text_v1(prediction)
    index=r.find(literal)
    if index<0:return False
    current=-1
    for entry in score_text_v1(r,p)['alignment']:
        if entry['reference'] is not None:current+=1
        if current==index and entry['reference'] is not None:return entry['operation']=='match'
    return False

def summarize_texts(rows):
    result={'count':len(rows),'localized':sum(r['prediction_id'] is not None for r in rows)}
    for name in ('raw','normalized'):
        scores=[r[name] for r in rows];den=sum(s['reference_characters'] for s in scores)
        result[name]={'exact':sum(s['exact_match'] for s in scores),
                      'cer':sum(s['distance'] for s in scores)/den if den else None}
    return result

def evaluate(root,config):
    sources=read(root/LABELS/'sources.json');source_by={p['id']:p for p in sources}
    lineage=read(checked(root,config['lineage']));mapping={p['id']:source_by[p['source_id']] for p in lineage if p['source_id'] in source_by}
    texts=read(root/LABELS/'texts.json');links=read(root/LABELS/'links.json');signs=read(root/LABELS/'signs.json')
    layout={p['id']:p for p in read(root/'artifacts/yolox-protocol-v1/targets.json')}
    all_rows=[];reports=[]
    literals={'diameter':'⌀','radius_prefix':'R','plus_minus':'±','degree':'°','reference_asterisk':'*','multiplication':'x'}
    for job in config['jobs']:
        request=read(checked(root,job['request']));folder=root/FOLDER/job['id'];runtime=read(folder/'process.json')
        saved=read(folder/'result/results.json') if (folder/'result/results.json').exists() else {'cases':[]}
        text_rows=[];sign_rows=[];link_rows=[];negative_rows=[];neighbours=[]
        for case in request['cases']:
            s=mapping[case['id']];f=s['factor'];case_text=[t for t in texts if t['source_id']==s['id']]
            predictions=[];associations={};crops=[]
            output=next((c for c in saved['cases'] if c['id']==case['id']),{'crops':[]})
            for crop in case['crops']:
                box=crop['bbox'];crops.append([v*f for v in box])
                out=next((o for o in output['crops'] if o['id']==crop['id']),None)
                if not out or out.get('error'):continue
                for line in out['ocr']['lines']:
                    uid=crop['id']+'/'+line['id'];p={**line,'id':uid,
                        'bbox':polygon_box(native_points(line['polygon_px'],box,f))}
                    predictions.append(p)
                    associations[uid]=[{**a,'native_anchors':native_points(a['anchors_px'],box,f)} for a in out['associations']['associations']
                                       if a['source_line_id']==line['id']]
            predictions=deduplicate(predictions);by_id={p['id']:p for p in predictions};matches=spatial_match(case_text,predictions)
            for t in case_text:
                pid=matches.get(t['id']);prediction=by_id[pid]['raw'] if pid else ''
                row={'id':t['id'],'source_id':s['id'],'reference':t['text'],'prediction':prediction,'prediction_id':pid,
                     'max_crop_coverage':max_coverage(t['bbox'],crops),
                     'raw':score_text_v1(t['text'],prediction),
                     'normalized':score_text_v1(normalize_text_v1(t['text']),normalize_text_v1(prediction))}
                text_rows.append(row)
                if not t['is_dimension']:
                    proposed=[a for a in associations.get(pid,[]) if a['status']=='proposed']
                    negative_rows.append({'text_id':t['id'],'localized':pid is not None,'false_proposed':len(proposed)})
                for sign in [x for x in signs if x['text_id']==t['id']]:
                    sign_rows.append({'id':sign['id'],'text_id':t['id'],'kind':sign['kind'],
                       'retained_character':character_preserved(t['text'],prediction,literals[sign['kind']]),
                       'max_crop_coverage':max_coverage(sign['bbox'],crops)})
                for link in [x for x in links if x['text_id']==t['id']]:
                    proposed=[a for a in associations.get(pid,[]) if a['status']=='proposed']
                    correct=[a for a in proposed if correct_relation(link,a)]
                    complete=any(max_coverage(t['bbox'],[b])>=1-1e-9 and leader_coverage(link['polyline'],b)>=1-1e-9 for b in crops)
                    link_rows.append({'id':link['id'],'text_id':t['id'],'complete_text_leader_context':complete,
                        'leader_max_coverage':max((leader_coverage(link['polyline'],b) for b in crops),default=0),
                        'proposed':len(proposed),'correct_proposed':len(correct),
                        'wrong_proposed':len(proposed)-len(correct),'associations':associations.get(pid,[])})
            targets=[r for r in layout[case['id']]['regions'] if r['role'] in ('view','detail')]
            for crop,b in zip(case['crops'],crops):
                if crop['role'] not in ('view','detail'):continue
                overlaps=sorted([(intersection(b,t['bbox']),t) for t in targets],key=lambda v:(-v[0],v[1]['id']))
                captured=[t['id'] for overlap,t in overlaps[1:] if overlap/area(t['bbox'])>=.1]
                neighbours.append({'page_id':case['id'],'crop_id':crop['id'],'neighbour_ids':captured})
        detail={'job':job['id'],'texts':text_rows,'signs':sign_rows,'links':link_rows,'negative_controls':negative_rows,'neighbours':neighbours}
        all_rows.append(detail)
        reports.append({'job':job['id'],'method':job['method'],'repeat':job['repeat'],
            'texts':summarize_texts(text_rows),'signs':{'count':len(sign_rows),'retained':sum(s['retained_character'] for s in sign_rows)},
            'text_context_complete':sum(t['max_crop_coverage']>=1-1e-9 for t in text_rows),
            'sign_context_complete':sum(s['max_crop_coverage']>=1-1e-9 for s in sign_rows),
            'links':{'count':len(link_rows),'context_complete':sum(l['complete_text_leader_context'] for l in link_rows),
                'correct':sum(l['correct_proposed']>0 for l in link_rows),'wrong':sum(l['wrong_proposed'] for l in link_rows)},
            'negative_controls':{'count':len(negative_rows),'localized':sum(n['localized'] for n in negative_rows),
                                 'false_proposed':sum(n['false_proposed'] for n in negative_rows)},
            'view_rois':len(neighbours),'neighbour_capturing_rois':sum(bool(n['neighbour_ids']) for n in neighbours),
            'runtime':runtime,'completion':saved.get('completion'),
            'model_calls':saved.get('model_calls',0),'roi_runs':sum(len(c['crops']) for c in saved['cases'])})
    repeats={m:all_rows[2*i]['texts']==all_rows[2*i+1]['texts'] and all_rows[2*i]['links']==all_rows[2*i+1]['links'] for i,m in enumerate(METHODS)}
    report={'task':'P09.3e.2a','scope':'targeted overview-scale validation diagnostics, not dense native-page or physical CAD quality',
        'policy':POLICY,'runs':reports,'repeat_target_outputs_equal':repeats,
        'production_admitted':False,'yolo_decision':'remains rejected under P09.3e.2',
        'notes':['Legacy-grid ROI-only is an ablation; current pipeline control is whole_page.',
                 'Five selected positive relations and five non-dimension controls only; unknown relations not negatives.',
                 'Normalized character retention is alignment-dependent, not glyph detector accuracy.',
                 'Neighbour overlap is a routing-risk diagnostic; whole-page processing intentionally includes all views.']}
    (root/FOLDER/'details.json').write_bytes(encoded(all_rows));(root/FOLDER/'report.json').write_bytes(encoded(report))
    return report
