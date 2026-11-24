"""Expanded, source-screened partial pixel supervision; no physical CAD targets."""
from collections import Counter
from copy import deepcopy
from html import escape
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageColor, ImageDraw

from . import pixel_role_release_v1 as previous
from . import pixel_role_draft_v1 as draft
from .dataset_layout_v1 import checked, encoded, inside, pin, read
from .dataset_preparation_v1 import crop_source
from .family_catalog_v2 import require
from .hard_raster_runs_v1 import screen
from .source_coallocation_v1 import package

PARENT={'path':'artifacts/pixel-role-release-v2/manifest.json',
        'sha256':'ed44beaedaae7e24bb322b41406d86e21361c08c5efa80e0ef9ca1f0083216ba'}
GROUPS={'path':'artifacts/role-admission-labels-v1-final/groups.json',
        'sha256':'d9c89858ec82cf93e00339ef938be1e40365f6127fecd414e42c2b040c94b3e6'}
INPUTS=['artifacts/ocr-current-release-v2/source-dispositions.json',
        'artifacts/dataset-content-review-v1/catalog.json',
        'artifacts/ocr-control-release-v2/task-splits.json',
        'artifacts/ocr-current-release-v2/task-splits.json',
        'configs/reviews/ocr-fresh-holdout-selection-v1.json']
CODE=['src/cad_recovery/'+n+'.py' for n in ('pixel_role_expansion_v1','pixel_role_release_v1',
      'pixel_role_draft_v1','dataset_preparation_v1','dataset_layout_v1','raster_regions',
      'hard_raster_runs_v1','source_coallocation_v1')]+['scripts/expand_pixel_roles_v1.py',
      'scripts/prepare_pixel_role_expansion_reading_v1.py']
POLICY={'task':previous.TASK,'classes':previous.POLICY['classes'],'ignore':255,
        'maximum_new_patches':16,'maximum_patch_side':1024,'maximum_annotations_per_patch':200,
        'maximum_new_sources':6,'maximum_regions_per_source':12,
        'authority':'assistant_visual_review','human_verified':False,'dense_ground_truth':False,
        'physical_geometry_approved':False,'coordinate_regression_approved':False,
        'independent_evaluation':False,'production_admitted':False,
        'validation':'diagnostic_only_no_training_or_model_selection',
        'training_scope':'fixed experimental partial supervision only',
        'region_targets':'coarse visual proposals only; not admitted as view detector targets',
        'selection':'two new groups per split by minimum SHA; plus two unused train sources by SHA; no readability replacement',
        'mask_recipe':previous.POLICY}


def select_sources(sources,catalog,groups,old_rows,registries,fresh_sources):
    """Screen metadata first; source identities and groups cannot be relabelled."""
    membership={p:g for g in groups for p in g['members']}
    entries={r['input']:r for r in catalog['entries']}
    require(len(entries)==len(catalog['entries']) and len({s['input'] for s in sources})==len(sources),'Repeated source')
    old_paths={r['source']['path'][len('dataset/'):] for r in old_rows}
    old_groups={r['group_id'] for r in old_rows}
    fresh=[{'source_path':s['input'],'source_sha256':s['input_sha256']} for s in fresh_sources]
    eligible=[]
    for s in sources:
        split=s['declared_split'];g=membership[s['input']]
        if (split not in ('train','validation') or s['source_kind']!='png' or s['input'] in old_paths
            or g['cross_split'] or g['declared_splits']!=[split]
            or entries[s['input']]['visual_status'] not in ('verified','inherited_verified')):continue
        boundary=screen([{'id':'candidate','declared_split':split,'input':s['input'],
                         'source_sha256':s['input_sha256']}],groups,registries,fresh)[0]
        if not boundary['blocked']:eligible.append(s)
    key=lambda s:(s['input_sha256'],s['input'])
    selected=[];counts={}
    for split in ('train','validation'):
        pool={}
        for s in sorted(eligible,key=key):
            g=membership[s['input']]['id']
            if s['declared_split']==split and g not in old_groups:pool.setdefault(g,s)
        counts[split]=len(pool)
        require(len(pool)>=2,'Insufficient unclosed new groups')
        for i,s in enumerate(sorted(pool.values(),key=key)[:2],1):
            selected.append({'id':split+'-new-'+str(i),'group_id':membership[s['input']]['id'],'source_record':s})
    used={s['source_record']['input'] for s in selected}
    extra=[s for s in sorted(eligible,key=key) if s['declared_split']=='train' and s['input'] not in used][:2]
    require(len(extra)==2,'Insufficient extra train rasters')
    for i,s in enumerate(extra,1):selected.append({'id':'train-extra-'+str(i),'group_id':membership[s['input']]['id'],'source_record':s})
    return selected,counts


def selection(root):
    old_folder,_=package(root,PARENT,'pixel-role-release-manifest/1.0')
    groups=read(checked(root,GROUPS))['groups']
    values=[read(root/name) for name in INPUTS]
    sources,counts=select_sources(values[0],values[1],groups,read(old_folder/'records.json'),values[2:4],values[4]['selected_sources'])
    return {'schema_version':'pixel-role-expansion-selection/1.0','parent':PARENT,'groups':GROUPS,
            'inputs':[pin(root,n) for n in INPUTS],'selection':POLICY['selection'],
            'new_group_counts':counts,'sources':sources,'independent_evaluation':False}


def color_statistics(image,labels):
    rgb=np.asarray(image.convert('RGB'));gray=np.asarray(image.convert('L'));ink=gray<240
    chromatic=(rgb.max(axis=2).astype(int)-rgb.min(axis=2).astype(int))>20
    roles=[]
    for index,name in enumerate(POLICY['classes']):
        selected=labels==index;v=gray[selected]
        roles.append({'class':name,'pixels':int(selected.sum()),
                      'gray_quantiles_10_50_90':np.percentile(v,[10,50,90]).tolist() if len(v) else None})
    return {'ink_pixels':int(ink.sum()),'nonneutral_ink_fraction':float((chromatic&ink).sum()/max(1,ink.sum())),
            'targets_by_role':roles,'grayscale_removes_color_contrast':False}


def validate_patch(patch,size):
    require(patch['source_box'] and len(patch['observations'])>=2
            and all(len(s)>20 for s in patch['observations']),'Missing patch review')
    valid_source_box(patch['source_box'],size)
    w,h=patch['source_box'][2]-patch['source_box'][0],patch['source_box'][3]-patch['source_box'][1]
    require(max(w,h)<=POLICY['maximum_patch_side'],'Patch exceeds resource bound')
    annotations=patch['annotations']
    require(len(annotations)<=POLICY['maximum_annotations_per_patch']
            and len({a['id'] for a in annotations})==len(annotations),'Invalid annotation inventory')
    require(len(patch['arrow_controls'])<=100,'Arrow control budget exceeded')
    for control in patch['arrow_controls']:
        require(control['state'] in ('visible_head','negative_region') and len(control['observation'])>20,'Invalid arrow review')
        if control['state']=='visible_head':
            tip=np.asarray(control['tip_px']);direction=np.asarray(control['direction'])
            require(tip.shape==direction.shape==(2,) and np.isfinite([*tip,*direction]).all()
                    and (tip>=0).all() and (tip<[w,h]).all() and abs(np.linalg.norm(direction)-1)<1e-6,'Invalid directed tip')
        else:draft.rasterize((h,w),control)
    return w,h


def valid_source_box(box,size):
    require(isinstance(box,list) and len(box)==4 and all(type(v) is int for v in box)
            and 0<=box[0]<box[2]<=size[0] and 0<=box[1]<box[3]<=size[1],'Source box outside image')


def freeze(root,selection_path,reading_path,output):
    require(not output.exists(),'Choose new expansion config')
    require(read(root/selection_path)==selection(root),'Source selection changed')
    output.write_bytes(encoded({'schema_version':'pixel-role-expansion-config/1.0','task':'P10.2a.3',
        'parent':PARENT,'policy':POLICY,'selection':pin(root,selection_path),'reading':pin(root,reading_path),
        'code':[pin(root,n) for n in CODE]}))
    return pin(root,output.relative_to(root).as_posix())


def materialize(root,record):
    config=read(checked(root,record))
    require(config['schema_version']=='pixel-role-expansion-config/1.0' and config['task']=='P10.2a.3'
            and config['parent']==PARENT and config['policy']==POLICY
            and config['code']==[pin(root,n) for n in CODE],'Changed expansion recipe')
    selected=read(checked(root,config['selection']))
    require(selected==selection(root),'Changed source selection')
    reading=read(checked(root,config['reading']))
    require(reading['reviewer']=='assistant' and all(reading[f] is False for f in
        ('human_verified','physical_geometry_approved','dense_ground_truth','independent_evaluation')),'Changed annotation authority')
    sources={s['id']:s for s in selected['sources']};patches=reading['patches']
    require(0<len(patches)<=POLICY['maximum_new_patches'] and len({p['id'] for p in patches})==len(patches)
            and {p['source_id'] for p in patches}==set(sources),'Changed patch inventory')
    require(len(reading['source_reviews'])==len(sources)
            and {s['source_id'] for s in reading['source_reviews']}==set(sources),'Missing source review')
    kinds={r['source_id']:r['drawing_kind'] for r in reading['source_reviews']}
    require(set(kinds.values())<= {'component_drawing','purchased_items_list'},'Unknown page kind')
    old_folder,old_manifest=package(root,PARENT,'pixel-role-release-manifest/1.0')
    previous.verify(root,PARENT)
    records=deepcopy(read(old_folder/'records.json'));stats=[];assets={}
    # Carry original partial targets unchanged; expansion cannot rewrite past labels.
    for r in old_manifest['files']:
        if r['path'].startswith('patches/'):assets[r['path']]=(old_folder/r['path']).read_bytes()
    old_stats=read(old_folder/'report.json')['patches']
    stats.extend(deepcopy(old_stats));taxonomy=read((root/previous.PARENT['path']).parent/'taxonomy.json')
    for p in patches:
        require(p['source_id'] in sources and p['id'] not in {r['id'] for r in records},'Unknown or duplicate patch')
        source=sources[p['source_id']];s=source['source_record'];size=validate_patch(p,s['source_size_px'])
        if kinds[p['source_id']]=='purchased_items_list':
            require(all(a['role'] in ('text','sheet_frame','title_grid') for a in p['annotations'])
                    and all(c['state']=='negative_region' for c in p['arrow_controls']),'Inventory cannot supply part geometry')
        pixels,_=crop_source(root,s,p['source_box']);image=Image.fromarray(pixels)
        require(image.size==size,'Wrong native crop size')
        arrays=previous.reviewed_masks(image,p['annotations'],p['unknown_regions'],p['background_regions'])
        prefix='patches/'+p['id'];paths={n:prefix+'/'+n+'.png' for n in ('image','labels','role-bits','state','preview')}
        assets[paths['image']]=draft.png(image)
        for name,a in [('labels',arrays['labels']),('role-bits',arrays['bits']),('state',arrays['state'])]:assets[paths[name]]=draft.png(Image.fromarray(a))
        overlay=pixels.copy()
        for item in taxonomy:
            target=arrays['labels']==item['id'];color=np.array(ImageColor.getrgb(item['color']))
            overlay[target]=(overlay[target]*.25+color*.75).astype(np.uint8)
        target=arrays['labels']==0;overlay[target]=(overlay[target]*.75+[0,50,0]).astype(np.uint8)
        overlay[arrays['ambiguous']&arrays['ink']]=[110,110,110]
        assets[paths['preview']]=draft.png(Image.fromarray(overlay))
        controls=image.copy();draw=ImageDraw.Draw(controls)
        for c in p['arrow_controls']:
            if c['state']=='visible_head':
                tip=np.asarray(c['tip_px']);direction=np.asarray(c['direction'])
                x,y=tip;draw.ellipse((x-8,y-8,x+8,y+8),outline='#009c4b',width=2)
                draw.line([tuple(tip-25*direction),tuple(tip)],fill='#b00042',width=2)
            else:
                l,t,r,b=c['box'];draw.rectangle((l,t,r-1,b-1),outline='#0966d0',width=2)
        assets[prefix+'/arrow-review.png']=draft.png(controls)
        x,y=p['source_box'][:2]
        row={'id':p['id'],'source_id':p['source_id'],'split':s['declared_split'],'group_id':source['group_id'],
             'source':{'path':'dataset/'+s['input'],'sha256':s['input_sha256']},'size_px':list(size),
             'patch_to_source':[[1,0,x],[0,1,y],[0,0,1]],'source_to_patch':[[1,0,-x],[0,1,-y],[0,0,1]],
             'paths':paths,'task':previous.TASK,'partial_targets':True,'dense_ground_truth':False,
             'training_eligible':s['declared_split']=='train','diagnostic_only':s['declared_split']=='validation',
             'independent_evaluation':False,'physical_geometry_approved':False,'review':p,
             'annotations':p['annotations'],'unknown_regions':p['unknown_regions']}
        records.append(row);assets[prefix+'/record.json']=encoded(row)
        stats.append({'id':p['id'],'split':row['split'],'group_id':row['group_id'],'pixels':image.width*image.height,
            'targets_by_class':[int(np.count_nonzero(arrays['labels']==i)) for i in range(10)],
            'ignore_pixels':int(np.count_nonzero(arrays['labels']==255)),
            'multiple_candidate_pixels':int(np.count_nonzero(arrays['state']==2)),
            'explicit_unknown_pixels':int(np.count_nonzero(arrays['state']==3))})
    regions=[]
    for review in reading['source_reviews']:
        s=sources[review['source_id']]['source_record'];w,h=s['source_size_px']
        require(len(review['regions'])<=POLICY['maximum_regions_per_source'] and len(review['observations'])>=2,'Missing page review')
        for region in review['regions']:
            require(region['role'] in ('drawing_view','title_block','parts_list','notes') and len(region['observation'])>20,'Unknown page role')
            require(region['kind']=='box','Coarse page regions require finite boxes')
            valid_source_box(region['box'],[w,h])
        assets['overviews/'+review['source_id']+'.png']=checked(root,next(a for a in s['prepared_assets'] if a['path'].endswith('/overview.png'))).read_bytes()
        overview=Image.open(BytesIO(assets['overviews/'+review['source_id']+'.png'])).convert('RGB');draw=ImageDraw.Draw(overview)
        for r in review['regions']:
            l,t,right,bottom=r['box'];l,right=l*overview.width/w,right*overview.width/w;t,bottom=t*overview.height/h,bottom*overview.height/h
            draw.rectangle((l,t,right,bottom),outline='#0966d0',width=2);draw.text((l+3,t+3),r['role'],fill='#b00042')
        assets['overviews/'+review['source_id']+'-regions.png']=draft.png(overview)
        regions.append({**review,'source':{'path':'dataset/'+s['input'],'sha256':s['input_sha256']},
                        'source_size_px':[w,h],'training_eligible':False,'coarse_only':True})
    train={r['group_id'] for r in records if r['split']=='train'};val={r['group_id'] for r in records if r['split']=='validation'}
    require(not train&val,'Cross-split group leakage')
    colors=[]
    for row in records:
        image=Image.open(BytesIO(assets[row['paths']['image']])).convert('RGB')
        labels=np.asarray(Image.open(BytesIO(assets[row['paths']['labels']])))
        colors.append({'id':row['id'],'split':row['split'],**color_statistics(image,labels)})
    coverage={s:np.sum([r['targets_by_class'] for r in stats if r['split']==s],axis=0).tolist() for s in ('train','validation')}
    admission={'task':previous.TASK,'train':'experimental_partial_supervision_only','validation':POLICY['validation'],
        'test':'not_used','dense_segmentation_admitted':False,'physical_geometry_approved':False,
        'coordinate_regression_approved':False,
        'production_admitted':False,'human_verified':False,'independent_evaluation':False,
        'coarse_region_training_admitted':False,'automatic_arrow_training_admitted':False}
    report={'schema_version':'pixel-role-expansion-report/1.0','policy':POLICY,'admission':admission,
        'patches':stats,'patch_counts':dict(Counter(r['split'] for r in records)),
        'sources_by_split':{s:len({r['source']['sha256'] for r in records if r['split']==s}) for s in ('train','validation')},
        'groups_by_split':{'train':sorted(train),'validation':sorted(val)},'targets_by_split':coverage,
        'color_statistics':colors,'closed_test_images_or_targets_used':False,'models_executed':False,'training_started':False,
        'arrow_control_counts':{s:dict(Counter(c['state'] for p in patches if sources[p['source_id']]['source_record']['declared_split']==s for c in p['arrow_controls'])) for s in ('train','validation')},
        'limitations':reading['limitations']}
    assets.update({'records.json':encoded(records),'report.json':encoded(report),'admission.json':encoded(admission),
        'reading.json':encoded(reading),'selection.json':encoded(selected),'page-regions.json':encoded(regions),
        'taxonomy.json':encoded(taxonomy),'arrow-controls.json':encoded({
            'schema_version':'expanded-arrow-controls/1.0','tip_tolerance_px':8,'direction_tolerance_deg':20,
            'training_eligible':False,'independent_evaluation':False,'controls_exhaustive':False,
            'authority':'assistant_visual_review','rows':[{'patch_id':r['id'],'split':r['split'],
                'source':r['source'],'patch_to_source':r['patch_to_source'],'controls':r['review']['arrow_controls']}
                for r in records if 'source_id' in r]})})
    for s in ('train','validation'):assets[s+'.jsonl']=b''.join(encoded(r).replace(b'\n',b'')+b'\n' for r in records if r['split']==s)
    assets['test.jsonl']=b''
    cards=['<!doctype html><meta charset="utf-8"><title>Expanded partial masks</title>',
        '<style>body{font:16px sans-serif;margin:2rem}img{max-width:48%;vertical-align:top}section{border-top:1px solid #aaa}</style>',
        '<h1>Расширенные частичные маски</h1><p>Проверка ассистента; diagnostic validation; неизвестные штрихи — ignore. Полные виды и физическая геометрия не допущены.</p>',
        '<p><a href="report.json">Метрики покрытия/контраста</a> · <a href="page-regions.json">Крупные области, только просмотр</a></p>']
    for r in records:cards.append('<section><h2>'+escape(r['id'])+' / '+r['split']+'</h2><img src="'+r['paths']['image']+'"><img src="'+r['paths']['preview']+'"></section>')
    for p in patches:cards.append('<section><h2>Arrow review: '+escape(p['id'])+'</h2><img src="patches/'+p['id']+'/arrow-review.png"></section>')
    for s in sources:cards.append('<section><h2>Page proposals: '+escape(s)+'</h2><img src="overviews/'+s+'-regions.png"></section>')
    assets['index.html']=''.join(cards).encode('utf-8')
    return assets,report,records


def build(root,record,output):
    target=inside(root,output)
    require(target.parent==root/'artifacts' and not target.exists(),'Choose new artifacts directory')
    assets,report,_=materialize(root,record);target.mkdir()
    for name,data in assets.items():
        path=inside(target,name);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    (target/'manifest.json').write_bytes(encoded({'schema_version':'pixel-role-expansion-manifest/1.0',
        'config':record,'files':[pin(target,n) for n in sorted(assets)]}))
    return {'manifest':pin(root,(target/'manifest.json').relative_to(root).as_posix()),'patch_counts':report['patch_counts'],
            'sources_by_split':report['sources_by_split'],'targets_by_split':report['targets_by_split']}


def verify(root,record):
    folder,manifest=package(root,record,'pixel-role-expansion-manifest/1.0')
    assets,report,rows=materialize(root,manifest['config'])
    require(manifest['files']==[pin(folder,n) for n in sorted(assets)],'Changed expansion inventory')
    require({p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}==set(assets)|{'manifest.json'},'Unexpected expansion files')
    require(all((folder/n).read_bytes()==v for n,v in assets.items()),'Changed expansion labels or authority')
    return report,rows


def load(root,record,*,split,task):
    require(task==previous.TASK and split in ('train','validation'),'Only experimental partial supervision is admitted')
    _,rows=verify(Path(root).resolve(),record)
    return {'records':[r for r in rows if r['split']==split],'task':task,'training_eligible':split=='train',
            'diagnostic_only':split=='validation','independent_evaluation':False,'dense_ground_truth':False}
