"""Replay-only repair: unresolved/unsupported associations have no anchors.

The v1 runtime/protocol and every saved inference result stay byte-identical.
Only the evaluator is versioned: [] is valid for an unresolved relation and
must remain [] in native space, rather than aborting the entire evaluation.
"""
from .dataset_layout_v1 import checked, encoded, read
from .joint_region_comparison_v1 import (
    FOLDER,LABELS,METHODS,POLICY,native_points,polygon_box,deduplicate,spatial_match,
    max_coverage,leader_coverage,correct_relation,character_preserved,summarize_texts,
    intersection,area)
from .ocr_baseline_metrics_v1 import normalize_text_v1,score_text_v1

def association_points(anchors,box,factor):
    if anchors==[]:return []
    return native_points(anchors,box,factor)

def evaluate(root,config,*,association_overrides=None,output_suffix='v2'):
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
                    relation_output=(association_overrides[job['id']][case['id']][crop['id']]
                        if association_overrides is not None else out['associations'])
                    associations[uid]=[{**a,'native_anchors':association_points(a['anchors_px'],box,f)}
                        for a in relation_output['associations'] if a['source_line_id']==line['id']]
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
    report={'task':'P09.3e.2a','evaluator_version':2,
        'repair':'v1 aborted on empty anchors of unresolved relation; no producer/model/labels/policy change or extra model calls',
        'scope':'targeted overview-scale validation diagnostics, not dense native-page or physical CAD quality',
        'policy':POLICY,'runs':reports,'repeat_target_outputs_equal':repeats,
        'production_admitted':False,'yolo_decision':'remains rejected under P09.3e.2',
        'notes':['Legacy-grid ROI-only is an ablation; current pipeline control is whole_page.',
                 'Five selected positive relations and five non-dimension controls only; unknown relations not negatives.',
                 'Normalized character retention is alignment-dependent, not glyph detector accuracy.',
                 'Neighbour overlap is a routing-risk diagnostic; whole-page processing intentionally includes all views.']}
    if association_overrides is not None:
        report['association_profile']='web-demo contour-association-v3 on identical saved prepared pixels/OCR/geometry/arrows; no deskew or inference rerun'
    (root/FOLDER/f'details-{output_suffix}.json').write_bytes(encoded(all_rows))
    (root/FOLDER/f'report-{output_suffix}.json').write_bytes(encoded(report))
    return report
