"""Preregistered partial native-field comparison; unknown page content unscored."""
from .dataset_layout_v1 import checked,read,encoded
from .native_merge_v1 import bounds,overlap
from .ocr_baseline_metrics_v1 import normalize_text_v1,score_text_v1
from .joint_region_comparison_v1 import correct_relation

def match(texts,predictions):
    options=[]
    for t in texts:
        b=t['bbox'];core=[b[0]+2,b[1]+2,b[2]-2,b[3]-2];ca=(core[2]-core[0])*(core[3]-core[1]);ta=(b[2]-b[0])*(b[3]-b[1])
        for p in predictions:
            q=bounds(p['polygon_px']);inter=max(0,min(core[2],q[2])-max(core[0],q[0]))*max(0,min(core[3],q[3])-max(core[1],q[1]))
            pa=(q[2]-q[0])*(q[3]-q[1]);iou,_=overlap(b,q)
            if inter/ca>=.8 and pa/ta<=3:options.append((-iou,-p['confidence'],t['id'],p['id']))
    result={};used=set()
    for _,_,t,p in sorted(options):
        if t not in result and p not in used:result[t]=p;used.add(p)
    return result

def canonical(result):
    return {k:result[k] for k in ('texts','geometry','arrows','associations','aliases','text_conflicts','seam_review',
                                 'deferred_retries','pixel_coverage_complete','ocr_windows_complete','geometry_search_incomplete')}

def evaluate(root,folder,config):
    texts=read(root/'artifacts/joint-native-labels-v1/texts.json');links=read(root/'artifacts/joint-native-labels-v1/links.json')
    rows=[];saved={};details=[]
    for job in config['jobs']:
        base=folder/job['id'];runtime=read(base/'process.json');request=read(checked(root,job['request']))
        path=base/'result/observations.json';result=read(path) if path.exists() else None
        source=next(s for s in config['sources'] if s['id']==job['source_id']);region=request['region'] or [0,0,*source['size_px']]
        targets=[t for t in texts if t['source_id']==source['id'] and region[0]<=t['bbox'][0]<t['bbox'][2]<=region[2] and region[1]<=t['bbox'][1]<t['bbox'][3]<=region[3]]
        predictions=result['texts'] if result else [];assign=match(targets,predictions);by_id={p['id']:p for p in predictions}
        entries=[]
        for t in targets:
            p=by_id.get(assign.get(t['id']));raw=p['raw'] if p else ''
            entries.append({'id':t['id'],'reference':t['text'],'prediction':raw,'assigned_id':p['id'] if p else None,
                'raw':score_text_v1(t['text'],raw),'normalized':score_text_v1(normalize_text_v1(t['text']),normalize_text_v1(raw))})
        relations=[]
        for link in [l for l in links if l['text_id'] in assign]:
            proposed=[a for a in (result['associations'] if result else []) if a['source_text_id']==assign[link['text_id']] and a['status']=='proposed']
            correct=[a for a in proposed if correct_relation(link,{'kind':a['local_evidence']['kind'],'native_anchors':a['anchors_px']})]
            relations.append({'id':link['id'],'correct':bool(correct),'wrong_proposed':len(proposed)-len(correct)})
        stats={'targets':len(targets),'assigned':len(assign)}
        for name in ('raw','normalized'):
            den=sum(e[name]['reference_characters'] for e in entries)
            stats[name]={'exact':sum(e[name]['exact_match'] for e in entries),'cer':sum(e[name]['distance'] for e in entries)/den if den else None}
        calls=read(base/'result/execution.json')['model_calls'] if (base/'result/execution.json').exists() else None
        rows.append({'id':job['id'],'source_id':source['id'],'mode':job['mode'],'scope':job['scope'],'repeat':job['repeat'],
            'text_metrics':stats,'relations':relations,'correct_known_links':sum(r['correct'] for r in relations),
            'model_calls':calls,'window_count':len(result['windows']) if result else 0,'runtime':runtime,
            'retry_count':result['retry_window_count'] if result else None,'deferred_retries':result['deferred_retries'] if result else None,
            'geometry_search_incomplete':result['geometry_search_incomplete'] if result else True,
            'ocr_complete':result['ocr_windows_complete'] if result else False,
            'geometry_count':len(result['geometry']) if result else 0,'text_conflicts':len(result['text_conflicts']) if result else 0})
        details.append({'job':job['id'],'texts':entries});saved[job['id']]=result
    repeats={}
    for j in config['jobs']:
        if j['repeat']!=0:continue
        other=next(x for x in config['jobs'] if x['source_id']==j['source_id'] and x['scope']==j['scope'] and x['mode']==j['mode'] and x['repeat']==1)
        a,b=saved[j['id']],saved[other['id']]
        repeats[j['id']]=a is not None and b is not None and canonical(a)==canonical(b)
    result={'task':'P18.1b.1','policy':config['policy'],'matching':config['matching'],'runs':rows,'forward_reverse_equal':repeats,
        'summary':{'successful_workers':sum(r['runtime']['budget_pass'] for r in rows),'assigned_workers':len(rows),
            'model_calls':sum(r['model_calls'] or 0 for r in rows),'all_order_invariant':all(repeats.values()),
            'source_pixels_resized':False,'service_profile_changed':False,'accepted_geometry':False},
        'scope':'partial selected native fields, diagnostic validation; whole-region CV may internally reduce analysis size and reports it',
        'unknown_content':'not negative; no dense geometry or global OCR precision','physical_geometry_approved':False,'production_admitted':False}
    (folder/'details.json').write_bytes(encoded(details));(folder/'report.json').write_bytes(encoded(result));return result
