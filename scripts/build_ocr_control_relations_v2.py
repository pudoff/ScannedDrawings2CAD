"""Build or replay the complete control-source relationship screening; no OCR."""
import argparse
from datetime import datetime
from html import escape
import json
from pathlib import Path
import runpy

import numpy as np
from PIL import Image

from cad_recovery.clear_pairs import contained, digest
from cad_recovery.coallocation_review_v1 import source_regions_v1
from cad_recovery.family_catalog_v2 import stable_digest
from cad_recovery.label_release_v1 import checked_pin
from cad_recovery.ocr_control_relations_v2 import build_relations_v2
from cad_recovery.ocr_release_v3 import load_ocr_training_v3, _package
from cad_recovery.prepared_batch_validation import _rgba_pixels
from cad_recovery.raster_regions import read_prepared_region

SELECTION = {'path':'artifacts/ocr-control-selection-v2.json','sha256':'c5debe99b3b15405d7e85be5efd356ef2287afbea46281e4e5d521b50abbf7bb'}
PROBES = {'path':'artifacts/ocr-control-title-probes-v2/manifest.json','sha256':'8c63908a06fbe6b33daca783e15cff101ef0cce3a0cc5a7f2704652ef980a908'}
PARENT = {'path':'artifacts/ocr-baseline-final-v2.json','sha256':'6ee85d8133ce9758b590ad891890430ec6f8232e3b20cf3a9efa611d0b744dc6'}
RELEASE = {'path':'artifacts/ocr-release-v3/manifest.json','sha256':'40459498a211b577e6cdfbca9d8772b4c9b04a9768b2d9d842395995abb87fdd'}
READINGS = ['artifacts/ocr-control-relations-'+letter+'-v2.json' for letter in ('a','b','c')]
CODE = ['scripts/build_ocr_control_relations_v2.py','scripts/prepare_ocr_control_v2.py', *[
    'src/cad_recovery/'+name+'.py' for name in ('ocr_control_relations_v2','ocr_control_selection_v2',
    'coallocation_policy_v1','coallocation_review_v1','ocr_release_v3','raster_regions',
    'png_stream','prepared_batch_validation','opaque_rgba','family_catalog_v2','clear_pairs','label_release_v1')]]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def encoded(value):
    return (json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode('utf-8')


def pin(root,path):
    return {'path':path,'sha256':digest(contained(root,path))}


def same(left,right):
    return stable_digest(left)==stable_digest(right)


def prepare(root,output):
    target=contained(root,output)
    require(target.parent==root/'configs/reviews' and not target.exists(),'Use a new review configuration')
    config={'schema_version':'ocr-control-relations-config/2.0','task':'P14.1b.7a',
        'selection':SELECTION,'probes':PROBES,'parent':PARENT,'release':RELEASE,
        'readings':[pin(root,p) for p in READINGS],'code':[pin(root,p) for p in CODE]}
    inputs(root,config)
    target.write_bytes(encoded(config)); return pin(root,output)


def inputs(root,config):
    require(set(config)=={'schema_version','task','selection','probes','parent','release','readings','code'}
        and config['schema_version']=='ocr-control-relations-config/2.0' and config['task']=='P14.1b.7a'
        and config['selection']==SELECTION and config['probes']==PROBES and config['parent']==PARENT
        and config['release']==RELEASE,'Unexpected relationship review scope')
    require(config['code']==[pin(root,p) for p in CODE],'Relationship implementation changed')
    require([p['path'] for p in config['readings']]==READINGS,'Missing authored reviews')
    checked_pin(root,PARENT)
    selector=runpy.run_path(str(root/'scripts/prepare_ocr_control_v2.py'),run_name='control_selection_recipe')
    selector['verify'](root,SELECTION)
    selection=read(checked_pin(root,SELECTION)); queue=selection['queue']
    context=selector['inputs'](root)
    loaded=load_ocr_training_v3(root,RELEASE)
    require(loaded['count']==90 and len(queue['entries'])==383,'Changed inherited or candidate inventory')
    probes=read(checked_pin(root,PROBES)); checked_pin(root,probes['extractor'])
    require(probes['selection']==SELECTION and len(probes['sources'])==80,'Changed source probes')
    prepared={r['input_path']:r for r in read(checked_pin(root,probes['prepared_catalog']))['entries']}
    for record in probes['contacts']:
        checked_pin(root,record)
    authored=[]; reading_pins={}
    fields={'rank','id','input_path','input_sha256','title_image','title_designation','title_name',
            'related_designations','decision','notes','visual_checks'}
    for record in config['readings']:
        packet=read(checked_pin(root,record))
        require(set(packet)=={'schema_version','selection','probes','reviewer','human_verified','ocr_run',
            'reviewed_at_utc','current_date','sources'} and packet['schema_version']=='ocr-control-relations/2.0'
            and packet['selection']==SELECTION and packet['probes']==PROBES and packet['reviewer']=='assistant'
            and packet['human_verified'] is False and packet['ocr_run'] is False,'Invalid authored review claims')
        reviewed=datetime.fromisoformat(packet['reviewed_at_utc'])
        require(reviewed.tzinfo is not None and reviewed>=datetime.fromisoformat(selection['selected_at_utc']), 'Review predates selection')
        for row in packet['sources']:
            require(fields<=set(row)<=fields|{'additional_viewed_images'} and type(row['rank']) is int
                and 0<=row['rank']<80 and row['id'] not in reading_pins,'Unexpected or duplicated reviewed source')
            probe=probes['sources'][row['rank']]
            require(all(same(row[key],probe[key]) for key in ('id','rank','input_path','input_sha256','title_image')),'Review/probe identity differs')
            binding=prepared[row['input_path']]
            require(all(same(probe[key],binding[key]) for key in ('input_sha256','preparation','source_size_px')),'Probe preparation differs from catalog')
            prep_path=checked_pin(root,binding['preparation']); prep=read(prep_path)
            require(prep['source_sha256']==row['input_sha256'] and prep['source_size_px']==probe['source_size_px'],'Changed preparation identity')
            allowed={prep_path.parent/item['path']:item['sha256'] for item in [prep['overview'],*prep['tiles']]}
            require(checked_pin(root,probe['overview']) in allowed and probe['overview']['sha256']==allowed[root/probe['overview']['path']], 'Probe overview differs')
            for extra in row.get('additional_viewed_images',[]):
                path=checked_pin(root,extra)
                require(path in allowed and allowed[path]==extra['sha256'],'Extra reviewed image belongs to another source')
            checked_pin(root,row['title_image'])
            require(set(row['visual_checks'])=={'title_seen','overview_seen'} and row['visual_checks']['title_seen'] is True
                and type(row['visual_checks']['overview_seen']) is bool,'Incomplete visual evidence')
            authored.append(row);reading_pins[row['id']]=record
    require(len(authored)==80,'Incomplete visual review inventory')
    return queue,context,probes,sorted(authored,key=lambda row:row['rank']),reading_pins


def pixels(root,probe,additional=()):
    source=checked_pin(root,{'path':'dataset/'+probe['input_path'],'sha256':probe['input_sha256']})
    prep_path=checked_pin(root,probe['preparation']);prep=read(prep_path);box=probe['title_box_source_px']
    require('container_repair' not in prep,'Unexpected repaired source in control probes')
    tile_by_path={prep_path.parent/item['path']:item for item in prep['tiles']}
    extra_tiles=[(record,tile_by_path[checked_pin(root,record)]) for record in additional
                 if checked_pin(root,record) in tile_by_path]
    boxes={'title':box,**{str(i):tile['source_box_px'] for i,(_,tile) in enumerate(extra_tiles)}}
    if 'rgba_conversion' in prep:
        rgb,conversion=_rgba_pixels(source,{'sha256':probe['input_sha256'],'size_px':probe['source_size_px'],
            'size_bytes':source.stat().st_size},prep,prep_path.parent)
        raw={key:rgb[b[1]:b[3],b[0]:b[2]] for key,b in boxes.items()};mode='original_opaque_RGBA_RGB_channels'
    else:
        raw=source_regions_v1(source,probe['input_sha256'],boxes);conversion=None;mode='original_RGB_stream'
    original=raw['title']
    tiled,provenance=read_prepared_region(prep_path.parent,box,max_pixels=16_777_216)
    require(np.array_equal(original,tiled),'Prepared title differs from original pixels')
    with Image.open(checked_pin(root,probe['title_image'])) as image:
        require(image.mode=='RGB' and np.array_equal(np.asarray(image),original),'Authored title image differs from original')
    extras=[]
    for i,(record,tile) in enumerate(extra_tiles):
        with Image.open(checked_pin(root,record)) as image:
            require(image.mode=='RGB' and np.array_equal(np.asarray(image),raw[str(i)]),'Additional reviewed tile differs from original')
        extras.append({'image':record,'box_source_px':tile['source_box_px'],'original_pixel_replay_equal':True})
    return {'rank':probe['rank'],'source':{'path':'dataset/'+probe['input_path'],'sha256':probe['input_sha256']},
        'box_source_px':box,'title_image':probe['title_image'],'original_pixel_replay_equal':True,
        'mode':mode,'rgba_provenance':conversion,'tile_provenance':provenance,'additional_original_regions':extras}


def documents(root,config_pin,config,verified,progress=False):
    queue,context,probes,authored,reading_pins=verified
    result=build_relations_v2(queue,context['catalog'],context['policy'],context['exclusions'],context['overlay'],
        authored,SELECTION,reading_pins,task_splits=context['task_splits'])
    files={};evidence=[];by_id={row['id']:row for row in authored}
    for probe in probes['sources']:
        additional=by_id[probe['id']].get('additional_viewed_images',[])
        evidence.append(pixels(root,probe,additional))
        files[f"images/{probe['rank']:02d}-title.png"]=checked_pin(root,probe['title_image']).read_bytes()
        for index,record in enumerate(additional):
            files[f"images/{probe['rank']:02d}-extra-{index}.png"]=checked_pin(root,record).read_bytes()
        if progress and (probe['rank']+1)%10==0:
            print(json.dumps({'original_pixel_replay':probe['rank']+1,'total':80}),flush=True)
    report={**result['report'],'config':config_pin,'parent':PARENT,'previous_ocr_release':RELEASE,
        'scope':'complete_relationship_screening_without_control_admission','date':'2026-11-08',
        'source_modified':False,'actual_bytes_authenticated':True,
        'new_OCR_labels':0,'released_validation_samples':0,'released_test_samples':0,
        'OCR_run':False,'training_started':False,'independent_evaluation':False,
        'original_pixel_regions_replayed':80+sum(len(r['additional_original_regions']) for r in evidence),
        'opaque_RGBA_regions_replayed':sum(r['mode']=='original_opaque_RGBA_RGB_channels' for r in evidence)}
    artifacts={**{name+'.json':value for name,value in result.items() if name!='report'},
        'report.json':report,'readings.json':authored,'evidence.json':evidence}
    for name,value in artifacts.items():files[name]=encoded(value)
    body=['<!doctype html><html lang="ru"><meta charset="utf-8"><title>Проверка связей OCR</title>',
        '<style>body{font:17px system-ui;max-width:1100px;margin:2em auto}img{max-width:100%}section{margin:2em 0}pre{white-space:pre-wrap}</style>',
        '<h1>80 представителей: проверка связей перед контролем OCR</h1>',
        '<p>Контрольные выборки не выпущены. <a href="report.json">Результат</a> · <a href="dispositions.json">Полный реестр</a></p>']
    for row in authored:
        body.append(f"<section><h2>{row['rank']:02d}: {escape(row['input_path'])}</h2><p>{escape(row['decision'])}</p>"
            f"<img loading='lazy' src='images/{row['rank']:02d}-title.png' alt='Исходная область штампа'>"
            '<pre>'+escape(json.dumps({key:row[key] for key in ('title_designation','title_name','related_designations','notes')},ensure_ascii=False,indent=2))+'</pre>'+
            ''.join(f"<img loading='lazy' src='images/{row['rank']:02d}-extra-{i}.png' alt='Дополнительная просмотренная область'>" for i in range(len(row.get('additional_viewed_images',[]))))+'</section>')
    files['index.html']=(''.join(body)+'</html>\n').encode('utf-8')
    return files,report


def build(root,config_pin,output):
    target=contained(root,output)
    require(target.parent==root/'artifacts' and not target.exists(),'Use a new direct artifacts directory')
    config=read(checked_pin(root,config_pin));verified=inputs(root,config)
    files,report=documents(root,config_pin,config,verified,True)
    inputs(root,config)
    target.mkdir()
    for name,data in files.items():
        path=contained(target,name);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    manifest={'schema_version':'ocr-control-relations-manifest/2.0','config':config_pin,'files':[pin(target,name) for name in sorted(files)]}
    (target/'manifest.json').write_bytes(encoded(manifest))
    return report


def verify(root,record):
    path=checked_pin(root,record);manifest=read(path)
    require(set(manifest)=={'schema_version','config','files'} and manifest['schema_version']=='ocr-control-relations-manifest/2.0','Invalid screening manifest')
    config=read(checked_pin(root,manifest['config']));verified=inputs(root,config)
    files,report=documents(root,manifest['config'],config,verified,True)
    require(manifest['files']==[pin(path.parent,name) for name in sorted(files)],'Screening inventory or hashes differ')
    require({p.relative_to(path.parent).as_posix() for p in path.parent.rglob('*') if p.is_file()}==set(files)|{'manifest.json'},'Unlisted screening files')
    for name,data in files.items():require((path.parent/name).read_bytes()==data,'Screening differs from original evidence')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--workspace',type=Path,default=Path.cwd())
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--output',default='configs/reviews/ocr-control-relations-v2.json')
    p=sub.add_parser('build');p.add_argument('--config',default='configs/reviews/ocr-control-relations-v2.json');p.add_argument('--config-sha256',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('verify');p.add_argument('--manifest',required=True);p.add_argument('--manifest-sha256',required=True)
    args=parser.parse_args();root=args.workspace.resolve()
    if args.command=='prepare': result=prepare(root,args.output)
    elif args.command=='build':result=build(root,{'path':args.config,'sha256':args.config_sha256},args.output)
    else:result=verify(root,{'path':args.manifest,'sha256':args.manifest_sha256})
    print(json.dumps(result,ensure_ascii=True))


if __name__=='__main__':main()
