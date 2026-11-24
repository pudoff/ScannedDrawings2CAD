"""Build/replay all remaining source reviews and freeze a limited control protocol."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import runpy

from cad_recovery.clear_pairs import contained, digest
from cad_recovery.family_catalog_v2 import require, stable_digest
from cad_recovery.label_release_v1 import checked_pin
from cad_recovery.ocr_control_scope_v3 import build_scope_v3

PARENT={'path':'artifacts/ocr-control-relations-final-v2.json','sha256':'05e2332182c0dcc0c1822978509181c7d8a43eb076c839afbda6b278dee1a53b'}
SCREEN={'path':'artifacts/ocr-control-relations-v2/manifest.json','sha256':'6ddb15361a9266d67e3475f8c6d2198391282a1c8b49c6948b91941304c2ba93'}
PROBES={'path':'artifacts/ocr-control-scope-probes-v3/manifest.json','sha256':'b75fe90665fcef2f2bca6a7be3f34aaebe52ff68b71cca9549c562c7da789e38'}
READINGS={'path':'artifacts/ocr-control-scope-readings-v3.json','sha256':'02e085b6929b46010b88dabf029b1bb5859a1d3bcd0065048b3ce5d666ffbc50'}
PREVIOUS_PROTOCOL={'path':'configs/reviews/ocr-control-protocol-v2.json','sha256':'994a682600858400ba499ccab64be8c56d66afc313c5ab2497784236f11919db'}
CODE=['scripts/build_ocr_control_scope_v3.py','src/cad_recovery/ocr_control_scope_v3.py',
      'artifacts/prepare-control-scope-probes-v3.py','artifacts/author-control-scope-readings-v3.py']
CONFIG_FIELDS={'schema_version','task','parent','screening','probes','readings','previous_protocol','code'}


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def encoded(value):return (json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode('utf-8')
def pin(root,path):return {'path':path,'sha256':digest(contained(root,path))}
def same(a,b):return stable_digest(a)==stable_digest(b)


def package(root,record):
    path=checked_pin(root,record);manifest=read(path)
    names=[r['path'] for r in manifest['files']]
    require(len(names)==len(set(names)),'Repeated package entry')
    require({p.relative_to(path.parent).as_posix() for p in path.parent.rglob('*') if p.is_file()}
        ==set(names)|{'manifest.json'},'Package inventory differs')
    for r in manifest['files']:checked_pin(path.parent,r)
    return path.parent,manifest


def inputs(root,config):
    require(set(config)==CONFIG_FIELDS and config['schema_version']=='ocr-control-scope-config/3.0'
        and config['task']=='P14.1b.7b','Changed scope configuration')
    for key,expected in [('parent',PARENT),('screening',SCREEN),('probes',PROBES),('readings',READINGS),('previous_protocol',PREVIOUS_PROTOCOL)]:
        require(config[key]==expected,'Unexpected canonical dependency: '+key);checked_pin(root,expected)
    require(config['code']==[pin(root,p) for p in CODE],'Scope implementation changed')
    base,screen=package(root,SCREEN)
    old=runpy.run_path(str(root/'scripts/build_ocr_control_relations_v2.py'),run_name='scope_inherited_relations')
    queue,context,_,previous,_=old['inputs'](root,read(checked_pin(root,screen['config'])))
    probes=read(checked_pin(root,PROBES));readings=read(checked_pin(root,READINGS))
    require(probes['schema_version']=='ocr-control-scope-probes/3.0' and probes['parent']==PARENT
        and probes['screening']==SCREEN and probes['source_count']==25,'Changed complete review scope')
    checked_pin(root,probes['extractor'])
    protocol=read(checked_pin(root,probes['protocol']))
    require(readings['probes']==PROBES and readings['protocol']==probes['protocol'],'Reading provenance differs')
    reviewed=datetime.fromisoformat(readings['reviewed_at_utc']);prepared_at=datetime.fromisoformat(protocol['created_at_utc'])
    require(reviewed.tzinfo is not None and prepared_at.tzinfo is not None and reviewed>=prepared_at,'Review predates probe protocol')
    require(protocol['OCR_run'] is False and protocol['new_OCR_labels']==0 and protocol['validation_test_admitted'] is False
        and protocol['all_25_sources_fixed_before_new_review'] is True,'Invalid source review protocol claims')
    prepared={r['input_path']:r for r in read(checked_pin(root,probes['prepared_catalog']))['entries']}
    for contact in probes['contacts']:checked_pin(root,contact)
    for row in probes['sources']:
        binding=prepared[row['input_path']]
        require(all(same(row[k],binding[k]) for k in ('input_sha256','source_size_px','preparation')),'Probe preparation differs')
        prep_path=checked_pin(root,row['preparation']);prep=read(prep_path)
        require(prep['source_sha256']==row['input_sha256'] and prep['source_size_px']==row['source_size_px'],'Preparation source differs')
        require(row['overview']==pin(root,(prep_path.parent/prep['overview']['path']).relative_to(root).as_posix())
            and row['overview']['sha256']==prep['overview']['sha256'],'Overview binding differs')
        w,h=row['source_size_px'];rw,rh=min(3200,w),min(1500,h)
        require([(r['name'],r['box_source_px']) for r in row['regions']]==[
            ('left',[0,h-rh,rw,h]),('right',[w-rw,h-rh,w,h])],'Review regions changed')
        for region in row['regions']:checked_pin(root,region['image'])
    prior_packets=read(root/'artifacts/coallocation-review-v1/readings.json')
    prior=[r for p in prior_packets for r in p['review']['sources']+p['review'].get('historical_comparisons',[])]+previous
    require(len(prior)==191,'Changed saved literal review inventory')
    values=[context[k] for k in ('catalog','policy','exclusions')]+[
        read(base/'overlay.json'),read(base/'routing.json'),read(base/'dispositions.json'),queue,probes,readings,prior]
    return old,values


def prepare(root,output):
    target=contained(root,output)
    require(target.parent==root/'configs/reviews' and not target.exists(),'Use a new scope configuration')
    config={'schema_version':'ocr-control-scope-config/3.0','task':'P14.1b.7b',
        'parent':PARENT,'screening':SCREEN,'probes':PROBES,'readings':READINGS,'previous_protocol':PREVIOUS_PROTOCOL,
        'code':[pin(root,p) for p in CODE]}
    inputs(root,config);target.write_bytes(encoded(config));return pin(root,output)


def documents(root,config_pin,verified,progress=False):
    old,values=verified;result=build_scope_v3(*values)
    overlay=values[3];probes=values[7];readings=values[8]
    files={};evidence=[];body=['<!doctype html><html lang="ru"><meta charset="utf-8"><title>Проверка 25 источников OCR</title>',
        '<style>body{font:17px system-ui;max-width:1100px;margin:2em auto}img{max-width:100%}pre{white-space:pre-wrap}section{margin:3em 0}</style>',
        '<h1>Все 25 оставшихся источников</h1><p>23 источника двух групп зарезервированы до разметки; два остаются pending. Глобальная независимость не подтверждена.</p>',
        '<p><a href="report.json">Результаты</a> · <a href="metadata-audit.json">Сопоставления</a> · <a href="dispositions.json">Все 383 PNG</a></p>']
    for row,reading in zip(probes['sources'],readings['sources'],strict=True):
        i=row['index'];section=[f'<section><h2>{i:02d}: {escape(row["input_path"])}</h2>']
        for kind,record in [('overview',row['overview']),*[(r['name'],r['image']) for r in row['regions']]]:
            name=f'images/{i:02d}-{kind}.png';files[name]=checked_pin(root,record).read_bytes()
            section.append(f'<img loading="lazy" src="{name}" alt="{kind}">')
        for region in row['regions']:
            probe={'rank':i,**{k:row[k] for k in ('input_path','input_sha256','preparation','source_size_px')},
                   'title_box_source_px':region['box_source_px'],'title_image':region['image']}
            proof=old['pixels'](root,probe)
            require(same(proof['tile_provenance'],region['tile_provenance']),'Prepared region provenance differs')
            evidence.append({**proof,'name':region['name']})
        section.append('<pre>'+escape(json.dumps(reading,ensure_ascii=False,indent=2))+'</pre></section>');body.extend(section)
        if progress and (i+1)%5==0:print(json.dumps({'source_pixels_replayed':i+1,'total':25}),flush=True)
    report={**result['report'],'config':config_pin,'parent':PARENT,'scope':'limited_relationship_review_and_future_reservations',
        'original_pixel_regions_replayed':len(evidence),
        'distinct_original_regions':len({(r['source']['path'],tuple(r['box_source_px'])) for r in evidence}),
        'opaque_RGBA_source_count':len({r['source']['path'] for r in evidence if r['mode']=='original_opaque_RGBA_RGB_channels'}),
        'actual_bytes_authenticated':True,'source_modified':False,'training_started':False,'validation_test_labels_released':False}
    result['report']=report;result['overlay']=overlay;result['readings']=readings;result['evidence']=evidence
    result['protocol-draft'].update(config=config_pin,review_readings=READINGS,review_probes=PROBES,
        previous_six_group_protocol=PREVIOUS_PROTOCOL,images_inspected_before_this_protocol=True,
        line_annotations_created_before_this_protocol=False,model_inference_before_this_protocol=False)
    for name,value in result.items():files[name+'.json']=encoded(value)
    files['index.html']=(''.join(body)+'</html>\n').encode('utf-8')
    return files,report


def build(root,config_pin,output):
    target=contained(root,output)
    require(target.parent==root/'artifacts' and not target.exists(),'Use a new direct artifacts directory')
    config=read(checked_pin(root,config_pin));verified=inputs(root,config)
    files,report=documents(root,config_pin,verified,True);inputs(root,config)
    target.mkdir()
    for name,data in files.items():
        path=contained(target,name);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    manifest={'schema_version':'ocr-control-scope-manifest/3.0','config':config_pin,
        'files':[pin(target,name) for name in sorted(files)]}
    (target/'manifest.json').write_bytes(encoded(manifest));return report


def verify(root,record):
    base,manifest=package(root,record)
    require(set(manifest)=={'schema_version','config','files'} and manifest['schema_version']=='ocr-control-scope-manifest/3.0','Unexpected scope manifest')
    config=read(checked_pin(root,manifest['config']))
    files,report=documents(root,manifest['config'],inputs(root,config),True)
    require(manifest['files']==[pin(base,name) for name in sorted(files)],'Scope manifest differs from replay')
    require(all((base/name).read_bytes()==data for name,data in files.items()),'Scope document differs from original evidence')
    return report


def frozen_protocol(root,scope_record,created):
    base,_=package(root,scope_record);draft=read(base/'protocol-draft.json')
    require(draft['ready_to_freeze'] is True and len(draft['reservations'])==2,'Limited protocol scope is unavailable')
    timestamp=datetime.fromisoformat(created)
    review=read(checked_pin(root,READINGS))
    require(timestamp.tzinfo is not None and timestamp>=datetime.fromisoformat(review['reviewed_at_utc']),'Protocol predates relationship review')
    return {**deepcopy(draft),'schema_version':'ocr-control-protocol/3.0','scope_manifest':scope_record,'frozen_at_utc':created}


def freeze(root,scope_record,output):
    target=contained(root,output)
    require(target.parent==root/'configs/reviews' and not target.exists(),'Use a new control protocol path')
    verify(root,scope_record)
    result=frozen_protocol(root,scope_record,datetime.now(timezone.utc).isoformat())
    target.write_bytes(encoded(result));return pin(root,output)


def verify_protocol(root,record):
    protocol=read(checked_pin(root,record));verify(root,protocol['scope_manifest'])
    expected=frozen_protocol(root,protocol['scope_manifest'],protocol['frozen_at_utc'])
    require(same(protocol,expected),'Frozen control protocol differs from reviewed scope')
    return {'reserved_group_count':len(protocol['reservations']),
        'reserved_source_count':sum(len(r['source_ids']) for r in protocol['reservations']),
        'labels_released':False,'independent_evaluation':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--workspace',type=Path,default=Path.cwd())
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--output',default='configs/reviews/ocr-control-scope-v3.json')
    p=sub.add_parser('build');p.add_argument('--config',default='configs/reviews/ocr-control-scope-v3.json');p.add_argument('--config-sha256',required=True);p.add_argument('--output',required=True)
    for name in ('verify','freeze'):
        p=sub.add_parser(name);p.add_argument('--manifest',required=True);p.add_argument('--manifest-sha256',required=True)
        if name=='freeze':p.add_argument('--output',default='configs/reviews/ocr-control-protocol-v3.json')
    p=sub.add_parser('verify-protocol');p.add_argument('--protocol',default='configs/reviews/ocr-control-protocol-v3.json');p.add_argument('--protocol-sha256',required=True)
    args=parser.parse_args();root=args.workspace.resolve()
    if args.command=='prepare':result=prepare(root,args.output)
    elif args.command=='build':result=build(root,{'path':args.config,'sha256':args.config_sha256},args.output)
    elif args.command=='verify-protocol':result=verify_protocol(root,{'path':args.protocol,'sha256':args.protocol_sha256})
    elif args.command=='freeze':result=freeze(root,{'path':args.manifest,'sha256':args.manifest_sha256},args.output)
    else:result=verify(root,{'path':args.manifest,'sha256':args.manifest_sha256})
    print(json.dumps(result,ensure_ascii=True))


if __name__=='__main__':main()
