"""Independent input/pixel/protocol/output audit after frozen CV comparison."""
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw

from .dataset_layout_v1 import checked,encoded,pin,read
from .family_catalog_v2 import require
from .page_region_review_v1 import context,authenticated_source
from .page_region_release_v1 import load
from .semantic_regions_v1 import classify,legacy,candidate,map_box
from .semantic_region_experiment_v1 import (LABELS,CONFIG,PARAMETERS,GATES,validated_protocol,evaluate)

SPEC = {'path':'artifacts/semantic-region-experiment-v1/spec.json',
        'sha256':'007dd75d045982cca576c4514402faecb254b898beff89060fe07896b18f5921'}


def replay(root,*,render=False):
    spec=read(checked(root,SPEC)); validated_protocol(root,spec['protocol'])
    for name,sha in spec['previous_pins'].items():checked(root,{'path':name,'sha256':sha})
    requests=read(checked(root,spec['requests'])); targets=read(checked(root,spec['targets']))
    require(requests['protocol']==spec['protocol'],'Request protocol rebound')
    inputs={r['id']:r for r in requests['cases']}; target_map={r['id']:r for r in targets}
    require(set(inputs)==set(target_map) and len(inputs)==14,'Changed input assignments')
    folder=root/'artifacts/semantic-region-experiment-v1'
    lineage=read(folder/'lineage.json'); lineage_map={r['case_id']:r for r in lineage}
    require(set(lineage_map)==set(inputs),'Changed lineage inventory')
    _,sources,excluded=context(root); require(not excluded,'Closed source in experiment')
    source_map={s['id']:s for s in sources}
    approved=load(root,LABELS,split='train',purpose='positive_roi_classification')+load(
        root,LABELS,split='validation',purpose='diagnostic_positive_box_localization')
    request_pixels=0; candidate_count=0; native_witnesses=0; predictions={}
    for identifier,request in inputs.items():
        target=target_map[identifier]; source=source_map[target['source_id']]; line=lineage_map[identifier]
        require(target['group_id']==source['group_id'] and target['split']==source['split'],'Group or split rebound')
        selected=[r for r in approved if r['source_id']==source['id']]
        require(target['labels'] and all(r in selected for r in target['labels']),'Unapproved target')
        if source['split']=='train':require(request['task']=='classify_roi' and len(target['labels'])==1,'Train localization not admitted')
        else:require(request['task']=='localize_page' and target['labels']==selected,'Validation target set changed')
        prep,report=authenticated_source(root,source); overview=report['overview']; factor=overview['scale_factor']
        expected_pin={'path':(prep.parent/overview['path']).relative_to(root).as_posix(),'sha256':overview['sha256']}
        require(line['source']==source['source'] and line['preparation']==source['preparation']
                and line['overview']==expected_pin and line['integer_factor']==factor,'Source pixels rebound')
        with Image.open(checked(root,expected_pin)) as im:
            box=line['overview_box']; expected=np.asarray(im.crop(box)).copy()
        with Image.open(checked(root,request['image'])) as im:
            require(im.mode=='RGB' and list(im.size)==request['size_px'] and np.array_equal(np.asarray(im),expected),'Prepared input pixels differ')
        x,y=box[:2]; edge=[[factor,0,factor*x],[0,factor,factor*y],[0,0,1]]
        centre=np.array([[1,0,-x],[0,1,-y],[0,0,1]])@np.asarray(overview['source_to_overview'])
        require(request['input_to_source_edges']==edge and np.array_equal(request['source_to_input_centres'],centre)
                and request['source_size_px']==source['size_px'],'Native coordinate map differs')
        for forward,inverse in [('source_to_input_centres','input_to_source_centres'),('source_to_input_edges','input_to_source_edges')]:
            require(np.allclose(np.asarray(request[forward])@request[inverse],np.eye(3),atol=1e-10),'Invalid inverse matrix')
        request_pixels += expected.shape[0]*expected.shape[1]
        if request['task']=='classify_roi': predictions[identifier]={'classification':classify(expected)}
        else:
            arms={}
            for name,algorithm in [('legacy_grid',legacy),('context_cv',candidate)]:
                rows=algorithm(expected)
                for i,row in enumerate(rows,1):
                    row['id']=f'{identifier}-{name}-{i:02d}'
                    row['native_box']=map_box(row['box'],edge,source['size_px'])
                    row['transform']={k:request[k] for k in request if 'centres' in k or 'edges' in k}
                    native_witnesses += 1
                candidate_count += len(rows)
                arms[name]={'candidates':rows,'incomplete':len(rows)>=PARAMETERS['max_candidates']}
            predictions[identifier]={'arms':arms}
    require(request_pixels==spec['total_pixels'],'Pixel budget ledger differs')
    raw={}
    for task in ('classify_roi','localize_page'):
        expected_ids=[r['id'] for r in requests['cases'] if r['task']==task]
        for repeat in (1,2):
            value=read(folder/f'{task}-{repeat}.json')
            require(value['request']==spec['requests'] and value['protocol']==spec['protocol']
                    and value['task']==task and value['model_calls']==value['training_runs']==0,'Worker contract differs')
            require([r['id'] for r in value['rows']]==expected_ids,'Changed or duplicate worker assignments')
            for row in value['rows']:
                require(row['prediction']==predictions[row['id']],'CV prediction replay differs')
                require(0<=row['seconds']<=GATES['max_case_seconds'],'Case budget differs')
            raw[task,repeat]=value
    result=evaluate(root,spec,raw['classify_roi',1],raw['localize_page',1]); actual=read(folder/'report.json')
    require({k:v for k,v in actual.items() if k not in ('resources','two_fresh_process_repeats_equal')}==result,
            'Evaluation or admission replay differs')
    require(actual['two_fresh_process_repeats_equal'] is True and len(actual['resources'])==4,'Missing repeat/resources')
    for resource in actual['resources']:
        require(0<resource['wall_seconds']<=GATES['max_worker_seconds']
                and 0<resource['observed_tree_peak_rss_mib']<=GATES['max_worker_rss_mib']
                and resource['observed_process_count']>=1,'Resource gate failed')
    if render:
        gallery=root/'artifacts/semantic-region-gallery-v1'; require(not gallery.exists(),'Choose new gallery directory');gallery.mkdir()
        for identifier,request in inputs.items():
            if request['task']!='localize_page':continue
            with Image.open(checked(root,request['image'])) as im:canvas=im.copy()
            canvas.thumbnail((1150,1150)); sx=canvas.width/request['size_px'][0]; sy=canvas.height/request['size_px'][1]
            draw=ImageDraw.Draw(canvas)
            for row in target_map[identifier]['labels']:
                box=map_box(row['bbox'],request['source_to_input_edges'],request['size_px'])
                b=[box[0]*sx,box[1]*sy,box[2]*sx,box[3]*sy]
                draw.rectangle(b,outline='#008650',width=3);draw.text((b[0]+3,b[1]+3),row['id'],fill='#008650')
            for row in predictions[identifier]['arms']['context_cv']['candidates']:
                box=row['box'];b=[box[0]*sx,box[1]*sy,box[2]*sx,box[3]*sy]
                draw.rectangle(b,outline='#d35f00',width=2);draw.text((b[0]+3,b[1]+17),row['id'].split('-')[-1]+' '+row['role'],fill='#d35f00')
            canvas.save(gallery/f'{identifier}.png')
    return {'schema_version':'semantic-region-replay/1.0','spec':SPEC,'protocol':spec['protocol'],
        'request_images_equal':len(inputs),'request_pixels':request_pixels,'native_bbox_witnesses':native_witnesses,
        'candidate_count_both_arms':candidate_count,'classification_records_replayed':12,
        'worker_reports_replayed':4,'previous_pins_preserved':len(spec['previous_pins']),
        'predictions_and_evaluation_equal':True,'model_calls':0,'training_runs':0,
        'closed_test_used':False,'physical_geometry_approved':False,'production_admitted':False}
