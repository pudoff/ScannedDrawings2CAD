"""Frozen model acceptance diagnostics; labels/production models remain unchanged."""
import argparse
from copy import deepcopy
from html import escape
from io import BytesIO
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cad_recovery import segmentation_acceptance_v1 as metrics
from cad_recovery import segmentation_experiment_v1 as experiment
from cad_recovery import pixel_role_expansion_v1 as labels
from cad_recovery import arrow_path_recovery_v1 as arrows
from cad_recovery import open_arc_search_v1 as arcs
from cad_recovery import stroke_curve_control_v1 as scoring
from cad_recovery.dataset_layout_v1 import checked,encoded,inside,pin,read
from cad_recovery.dataset_preparation_v1 import crop_source
from cad_recovery.family_catalog_v2 import require
from cad_recovery.geometry_detection import detect_geometry
from cad_recovery.gpu_comparison_v1 import DeviceMemorySampler
from cad_recovery.hard_raster_runs_v1 import execute
from cad_recovery.source_coallocation_v1 import package

LABELS={'path':'artifacts/pixel-role-expansion-v3/manifest.json','sha256':'ba7836cfa899633e3291f95c6715fbd9b64418322e2ff14377e4143dbb320746'}
OLD={'path':'artifacts/segmentation-experiment-v2/manifest.json','sha256':'3d3c649d2928cc7fabfe6f8f00d9f8c331169f35b06b91e40fce9b39a9824e91'}
CODE=list(dict.fromkeys(labels.CODE+['src/cad_recovery/'+n+'.py' for n in
    ('segmentation_acceptance_v1','segmentation_experiment_v1','gpu_training_probe_v1','gpu_inference_v1',
     'gpu_comparison_v1','geometry_detection','circle_search_v2','arrow_detection','arrow_path_recovery_v1',
     'open_arc_search_v1','stroke_curve_control_v1')]+['scripts/review_segmentation_acceptance_v1.py','scripts/segmentation_acceptance_child_v1.py']))


def frozen_models():
    folder,_=package(ROOT,OLD,'segmentation-experiment-manifest/1.0')
    previous=read(folder/'summary.json')['results'];models=[]
    for variant in metrics.POLICY['variants']:
        original='pretrained-gray' if variant.endswith('-repeat') else variant
        saved=next(v for v in previous if v['variant']==original)
        models.append({'variant':variant,'mode':'rgb' if variant=='pretrained-rgb' else 'gray',
            'weights':pin(ROOT,(folder/original/'result/final.pt').relative_to(ROOT).as_posix()),'tensor_digest':saved['final_parameters_sha256']})
    return models


def freeze(output):
    labels.verify(ROOT,LABELS);models=frozen_models()
    require(not output.exists(),'Choose new acceptance protocol')
    output.write_bytes(encoded({'schema_version':'segmentation-acceptance-config/1.0','task':'P10.2b.2',
        'labels':LABELS,'previous_experiment':OLD,'policy':metrics.POLICY,'models':models,
        'environment':pin(ROOT,'configs/gpu-training-v1.json'),'lock':pin(ROOT,'configs/gpu-training-v1.lock'),
        'python':'artifacts/environments/gpu-training-v1/Scripts/python.exe','code':[pin(ROOT,n) for n in CODE],
        'frozen_before_new_inference':True,'training_executed':False}))
    return pin(ROOT,output.relative_to(ROOT).as_posix())


def authenticate(record):
    c=read(checked(ROOT,record))
    require(c['policy']==metrics.POLICY and c['labels']==LABELS and c['previous_experiment']==OLD
            and c['code']==[pin(ROOT,n) for n in CODE] and c['frozen_before_new_inference'] is True
            and c['training_executed'] is False,'Changed acceptance protocol')
    require(c['models']==frozen_models(),'Changed assigned frozen model')
    for v in c['models']:checked(ROOT,v['weights'])
    checked(ROOT,c['environment']);checked(ROOT,c['lock'])
    return c


def png(image):
    stream=BytesIO();image.save(stream,format='PNG');return stream.getvalue()


def inputs():
    _,rows=labels.verify(ROOT,LABELS);folder=(ROOT/LABELS['path']).parent
    arrays={r['id']:(np.asarray(Image.open(folder/r['paths']['image']).convert('RGB')),
                     np.asarray(Image.open(folder/r['paths']['labels']))) for r in rows}
    items=[];assets={}
    for r in rows:
        name=r['id'];image='inputs/'+name+'.png';target='targets/'+name+'.png'
        assets[image]=(folder/r['paths']['image']).read_bytes();assets[target]=(folder/r['paths']['labels']).read_bytes()
        items.append({'id':name,'kind':'patch','split':r['split'],'size_px':r['size_px'],'source':r['source'],
            'patch_to_source':r['patch_to_source'],'image_name':image,'target_name':target,
            'new':bool('source_id' in r),'monochrome_train':name.startswith('train-extra-')})
    sources={r['id']:r['source_record'] for r in read(folder/'selection.json')['sources']}
    regions=read(folder/'page-regions.json')
    for source_id in metrics.POLICY['full_view_sources']:
        source=sources[source_id];page=next(r for r in regions if r['source_id']==source_id)
        box=next(r['box'] for r in page['regions'] if r['role']=='drawing_view')
        require((box[2]-box[0])*(box[3]-box[1])<=metrics.POLICY['maximum_full_view_pixels'],'Full-view resource ceiling')
        pixels,_=crop_source(ROOT,source,box)
        target,conflicts=metrics.merged_targets(rows,arrays,page['source'],box)
        name='full-'+source_id;image='inputs/'+name+'.png';mask='targets/'+name+'.png'
        assets[image]=png(Image.fromarray(pixels));assets[mask]=png(Image.fromarray(target))
        items.append({'id':name,'kind':'full_view','split':source['declared_split'],'size_px':[pixels.shape[1],pixels.shape[0]],
            'source':page['source'],'source_box':box,'image_name':image,'target_name':mask,'overlap_conflicts':conflicts,
            'coarse_box_is_not_dense_GT':True,'new':True,'monochrome_train':source_id=='train-extra-1'})
    return items,assets,rows


def evaluate(folder,items):
    variants={};patch_items=[r for r in items if r['kind']=='patch']
    for variant in metrics.POLICY['variants']+['ink_floor']:
        evaluations=[]
        for r in patch_items:
            target=np.asarray(Image.open(folder/r['target_name']))
            prediction=(experiment.heuristic(np.asarray(Image.open(folder/r['image_name']).convert('RGB')))
                if variant=='ink_floor' else np.asarray(Image.open(folder/variant/'result'/(r['id']+'.png'))))
            evaluations.append({'id':r['id'],'split':r['split'],'new':r['new'],'monochrome_train':r['monochrome_train'],**metrics.evaluate(target,prediction)})
        strata={}
        for name,chosen in [('train',[r for r in evaluations if r['split']=='train']),
            ('validation',[r for r in evaluations if r['split']=='validation']),
            ('new_validation',[r for r in evaluations if r['split']=='validation' and r['new']]),
            ('old_validation',[r for r in evaluations if r['split']=='validation' and not r['new']]),
            ('monochrome_train',[r for r in evaluations if r['monochrome_train']])]:
            strata[name]=metrics.combine_evaluations(chosen)
        variants[variant]={'strata':strata,'patches':evaluations}
    full=[]
    for r in items:
        if r['kind']!='full_view':continue
        target=np.asarray(Image.open(folder/r['target_name']));pred=np.asarray(Image.open(folder/'pretrained-gray/result'/(r['id']+'.png')))
        matrix=experiment.confusion(target,pred)
        full.append({'id':r['id'],'size_px':r['size_px'],'reviewed_pixels':int((target!=255).sum()),
            'unknown_pixels':int((target==255).sum()),'partial_metrics':experiment.metrics(matrix),'full_page_precision':None})
    return variants,full


def old_predictions(folder,items):
    old=(ROOT/OLD['path']).parent;counts={}
    for variant in metrics.POLICY['variants']:
        original='pretrained-gray' if variant.endswith('-repeat') else variant
        selected=[r for r in items if r['kind']=='patch' and not r['new']]
        counts[variant]=sum(np.array_equal(np.asarray(Image.open(folder/variant/'result'/(r['id']+'.png'))),
            np.asarray(Image.open(old/original/'result'/(r['id']+'-prediction.png')))) for r in selected)
        require(counts[variant]==len(selected),'Frozen predictions changed on old inputs')
    return counts


def cv_score(row,target,geometry,heads):
    reference=[{'id':c['id'],'tip':c['tip_px'],'direction':c['direction']} for c in row['review']['arrow_controls'] if c['state']=='visible_head']
    matched=scoring.arrow_scores(reference,heads['candidates'],np.eye(3));negative=[]
    for control in row['review']['arrow_controls']:
        if control['state']!='negative_region':continue
        x,y,r,b=control['box'];ids=[c['id'] for c in heads['candidates'] if x<=c['tip_px'][0]<r and y<=c['tip_px'][1]<b]
        negative.append({'id':control['id'],'candidate_ids':ids})
    return {'assigned':len(matched),'matched':sum(r['matched'] for r in matched),'arrows':matched,
        'false_tips':sum(len(r['candidate_ids']) for r in negative),'negatives':negative,
        'head_candidates':len(heads['candidates']),'primitive_candidates':len(geometry['candidates']),
        'primitive_coverage':metrics.primitive_coverage(target,geometry),'arrow_search_incomplete':heads['search_incomplete']}


def cv_run(record,folder,output):
    authenticate(record);_,_,rows=inputs();require(not output.exists(),'Choose new CV output');output.mkdir()
    cv2.setNumThreads(4);cv2.setRNGSeed(20261110);results=[]
    for row in rows:
        if 'source_id' not in row:continue
        case=output/row['id'];case.mkdir();original=folder/'inputs'/(row['id']+'.png')
        rgb=np.asarray(Image.open(original).convert('RGB'));gray=np.asarray(Image.fromarray(rgb).convert('L'))
        target=np.asarray(Image.open(folder/'targets'/(row['id']+'.png')))
        prediction=np.asarray(Image.open(folder/'pretrained-gray/result'/(row['id']+'.png')))
        filtered=case/'guided.png';Image.fromarray(metrics.guided_rgb(rgb,prediction)).save(filtered)
        scores={}
        for variant,path in [('original',original),('guided',filtered)]:
            destination=case/variant
            geometry=detect_geometry(path,destination,search_profile='regional_masked')
            stroke=geometry['analysis']['estimated_stroke_width_px']/geometry['analysis']['scale']
            query=np.asarray(Image.open(path).convert('L'))
            arc=arcs.search(query,stroke_px=stroke,lines=geometry['candidates'])
            heads=arrows.search(gray,lines=geometry['candidates'],arcs=arc['candidates'],stroke_px=stroke)
            geometry.pop('timings',None)
            (destination/'geometry-canonical.json').write_bytes(encoded(geometry))
            (destination/'arcs.json').write_bytes(encoded(arc));(destination/'heads.json').write_bytes(encoded(heads))
            scores[variant]={**cv_score(row,target,geometry,heads),'arc_search_incomplete':arc['search_incomplete'],
                'geometry_issues':geometry['issues'],'source_heads_checked_on_original_RGB':True}
        results.append({'id':row['id'],'split':row['split'],**scores})
        print('CV',row['id'],scores['original']['matched'],'->',scores['guided']['matched'],flush=True)
    (output/'results.json').write_bytes(encoded(results))


def build(record,target):
    c=authenticate(record);require(target.parent==ROOT/'artifacts' and not target.exists(),'Choose fresh acceptance artifacts')
    items,assets,rows=inputs();target.mkdir()
    for name,data in assets.items():
        p=inside(target,name);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    records=[{**r,'image':pin(ROOT,(target/r['image_name']).relative_to(ROOT).as_posix())} for r in items]
    (target/'inputs.json').write_bytes(encoded(records));input_pin=pin(ROOT,(target/'inputs.json').relative_to(ROOT).as_posix())
    runtimes={}
    for variant in metrics.POLICY['variants']:
        dest=target/variant;dest.mkdir()
        command=[str(ROOT/c['python']),str(ROOT/'scripts/segmentation_acceptance_child_v1.py'),
            '--config',record['path'],'--sha256',record['sha256'],'--inputs',input_pin['path'],'--inputs-sha256',input_pin['sha256'],
            '--variant',variant,'--output',(dest/'result').relative_to(ROOT).as_posix()]
        with DeviceMemorySampler() as sampler:runtime=execute(ROOT,command,dest/'process',c['policy']['max_child_seconds'],c['policy']['max_host_MiB'])
        runtime['gpu_memory']=sampler.report();(dest/'runtime.json').write_bytes(encoded(runtime))
        require(runtime['budget_pass'] and not runtime['gpu_memory']['errors']
                and runtime['gpu_memory']['peak_MiB'] is not None and runtime['gpu_memory']['peak_MiB']<=c['policy']['max_device_MiB'],'Frozen inference failed: '+variant)
        runtimes[variant]=runtime;print('inference saved',variant,flush=True)
    cv_runtime=execute(ROOT,[str(ROOT/'.venv/Scripts/python.exe'),str(Path(__file__).resolve()),'cv',
        '--config',record['path'],'--sha256',record['sha256'],'--folder',target.relative_to(ROOT).as_posix(),
        '--output',(target/'cv').relative_to(ROOT).as_posix()],target/'cv-process',900,3072)
    (target/'cv-runtime.json').write_bytes(encoded(cv_runtime));require(cv_runtime['budget_pass'],'CV comparison failed')
    variants,full=evaluate(target,items)
    repeat=all((target/'pretrained-gray/result'/(r['id']+'.png')).read_bytes()==(target/'pretrained-gray-repeat/result'/(r['id']+'.png')).read_bytes() for r in items)
    require(repeat and variants['pretrained-gray']==variants['pretrained-gray-repeat'],'Independent inference repeat differs')
    cv_rows=read(target/'cv/results.json')
    report={'schema_version':'segmentation-acceptance-report/1.0','policy':metrics.POLICY,'variants':variants,
        'full_views':full,'cv':cv_rows,'primary_repeat_identical':repeat,
        'old_predictions_unchanged':old_predictions(target,items),
        'decision':metrics.admission(variants['pretrained-gray']['strata']['new_validation'],cv_rows,repeat,True),
        'training_executed':False,'closed_test_used':False,'production_admitted':False,
        'limitations':['Partial labels and viewed diagnostic validation; no independent or dense full-page evaluation.',
            'Reference island connectivity is not CAD topology; no physical units or dimension associations are accepted.',
            'Guided CV is an explicit experimental ablation; native head checks use the original source.',
            'Ink floor has no semantic separation; unclassified CV coverage is a separate diagnostic, not semantic IoU.']}
    (target/'report.json').write_bytes(encoded(report));preview(target,items,report)
    files=sorted(p.relative_to(target).as_posix() for p in target.rglob('*') if p.is_file())
    (target/'manifest.json').write_bytes(encoded({'schema_version':'segmentation-acceptance-manifest/1.0','config':record,'files':[pin(target,n) for n in files]}))
    print(json.dumps({'manifest':pin(ROOT,(target/'manifest.json').relative_to(ROOT).as_posix()),'repeat_identical':repeat,'decision':report['decision']}))


def preview(folder,items,report):
    palette=np.array([[255,255,255],[0,116,210],[235,58,85],[255,158,0],[139,61,198],[0,140,120],[196,57,148],[120,77,45],[99,105,0],[14,178,80]],np.uint8)
    cards=['<!doctype html><meta charset="utf-8"><title>Expanded segmentation acceptance</title>',
        '<style>body{font:16px sans-serif;margin:2rem}img{max-width:32%;vertical-align:top}pre{white-space:pre-wrap}</style>',
        '<h1>Расширенная проверка сегментации</h1><p>Исходник / частичные targets (серый — ignore) / предсказание. Качество неизвестных пикселей не оценено. Приёмки в CAD нет.</p>',
        '<p><a href="report.json">Метрики и решение</a></p>']
    dest=folder/'preview';dest.mkdir()
    for r in items:
        target=np.asarray(Image.open(folder/r['target_name']));pred=np.asarray(Image.open(folder/'pretrained-gray/result'/(r['id']+'.png')))
        mask=np.full((*target.shape,3),215,np.uint8);valid=target!=255;mask[valid]=palette[target[valid]]
        for name,a in [('target',mask),('prediction',palette[pred])]:
            im=Image.fromarray(a);im.thumbnail((1400,1400));im.save(dest/(r['id']+'-'+name+'.png'))
        original=Image.open(folder/r['image_name']).convert('RGB');original.thumbnail((1400,1400));original.save(dest/(r['id']+'-source.png'))
        cards.append('<h2>'+escape(r['id'])+'</h2>'+''.join('<img src="preview/'+r['id']+'-'+n+'.png">' for n in ('source','target','prediction')))
    (folder/'index.html').write_text(''.join(cards),encoding='utf-8')


def verify(record):
    folder,manifest=package(ROOT,record,'segmentation-acceptance-manifest/1.0');c=authenticate(manifest['config'])
    items,assets,rows=inputs()
    require(all((folder/n).read_bytes()==v for n,v in assets.items()),'Changed native acceptance inputs or targets')
    inventory={p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}
    require(inventory=={r['path'] for r in manifest['files']}|{'manifest.json'},'Changed acceptance inventory')
    input_records=read(folder/'inputs.json')
    require(len(input_records)==len(items),'Lost native input assignment')
    for saved,item in zip(input_records,items):
        require({k:v for k,v in saved.items() if k!='image'}==item
            and saved['image']['sha256']==pin(ROOT,(folder/item['image_name']).relative_to(ROOT).as_posix())['sha256'],'Changed frozen inference input')
    for variant in metrics.POLICY['variants']:
        saved=read(folder/variant/'result/result.json');model=next(v for v in c['models'] if v['variant']==variant)
        require(saved['tensor_digest_before']==saved['tensor_digest_after']==model['tensor_digest']
            and saved['training_executed'] is False and saved['targets_read'] is False and saved['model_eval'] is True
            and saved['input_mode']==model['mode'] and saved['variant']==variant
            and saved['AMP'] is saved['TF32'] is saved['gradients_enabled'] is False
            and saved['network_disabled'] is True and saved['torch']=='2.14.1+cu130'
            and saved['protocol']==manifest['config'] and saved['inputs_sha256']==pin(ROOT,(folder/'inputs.json').relative_to(ROOT).as_posix())['sha256'],'Changed frozen inference evidence')
        require(0<saved['allocated_peak_MiB']<=c['policy']['max_allocated_MiB']
            and saved['allocated_peak_MiB']<=saved['reserved_peak_MiB']<=c['policy']['max_reserved_MiB'],'Invalid CUDA resource evidence')
        runtime=read(folder/variant/'runtime.json')
        require(runtime['budget_pass'] and runtime['exit_code']==0 and runtime['failure'] is None
            and runtime['walltime_seconds']<=c['policy']['max_child_seconds'] and runtime['peak_rss_MiB']<=c['policy']['max_host_MiB']
            and not runtime['gpu_memory']['errors'] and runtime['gpu_memory']['peak_MiB']<=c['policy']['max_device_MiB'],'Changed inference resource decision')
        expected=[r['id'] for r in items if r['kind']=='patch' or variant in ('pretrained-gray','pretrained-gray-repeat')]
        require([r['id'] for r in saved['items']]==expected,'Lost inference assignment')
        for r in saved['items']:
            item=next(i for i in items if i['id']==r['id']);w,h=item['size_px']
            require(r['prediction']['sha256']==pin(ROOT,(folder/variant/'result'/(r['id']+'.png')).relative_to(ROOT).as_posix())['sha256']
                and Path(r['prediction']['path']).name==r['id']+'.png'
                and r['size_px']==[w,h] and r['tiles']==len(experiment.boxes(w,h,stride=c['policy']['stride']))
                and r['coverage_min']>0,'Changed inference assignment or tile coverage')
    variants,full=evaluate(folder,items);report=read(folder/'report.json');cv_rows=read(folder/'cv/results.json')
    require([r['id'] for r in cv_rows]==[r['id'] for r in rows if 'source_id' in r],'Lost CV control assignment')
    cv_runtime=read(folder/'cv-runtime.json')
    require(cv_runtime['budget_pass'] and cv_runtime['failure'] is None and cv_runtime['exit_code']==0
            and cv_runtime['walltime_seconds']<=900 and cv_runtime['peak_rss_MiB']<=3072,'Changed CV resource decision')
    by_id={r['id']:r for r in rows}
    for r in cv_rows:
        row=by_id[r['id']];target=np.asarray(Image.open(folder/'targets'/(r['id']+'.png')))
        for v in ('original','guided'):
            base=folder/'cv'/r['id']/v;geometry=read(base/'geometry-canonical.json');heads=read(base/'heads.json')
            gray=np.asarray(Image.open(folder/'inputs'/(r['id']+'.png')).convert('L'))
            stroke=geometry['analysis']['estimated_stroke_width_px']/geometry['analysis']['scale']
            require(heads==arrows.search(gray,lines=geometry['candidates'],arcs=read(base/'arcs.json')['candidates'],stroke_px=stroke),'Changed native head recipe')
            expected=cv_score(row,target,geometry,heads)
            require(all(r[v][k]==value for k,value in expected.items()),'Changed CV scores')
    repeat=all((folder/'pretrained-gray/result'/(r['id']+'.png')).read_bytes()==(folder/'pretrained-gray-repeat/result'/(r['id']+'.png')).read_bytes() for r in items)
    require(report['variants']==variants and report['full_views']==full and report['cv']==cv_rows
        and report['primary_repeat_identical']==repeat and repeat
        and report['old_predictions_unchanged']==old_predictions(folder,items)
        and report['decision']==metrics.admission(variants['pretrained-gray']['strata']['new_validation'],cv_rows,repeat,True)
        and report['training_executed'] is report['closed_test_used'] is report['production_admitted'] is False,'Changed acceptance metrics/admission')
    return {'verified':True,'patches':25,'full_views':len(full),'repeat_equal':repeat,'decision':report['decision']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['freeze','build','cv','verify'])
    for n in ('config','sha256','output','manifest','folder'):p.add_argument('--'+n)
    a=p.parse_args()
    if a.command=='freeze':print(json.dumps(freeze(ROOT/a.output)))
    elif a.command=='build':build({'path':a.config,'sha256':a.sha256},ROOT/a.output)
    elif a.command=='cv':cv_run({'path':a.config,'sha256':a.sha256},ROOT/a.folder,ROOT/a.output)
    else:print(json.dumps(verify({'path':a.manifest,'sha256':a.sha256})))
