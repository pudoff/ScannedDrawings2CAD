"""Pure child-side preparation and derivative creation; no database writes."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import random
import ezdxf
from PIL import Image,ImageDraw
from .dataset_layout_v1 import encoded
from .graph_correction_v1 import prepare_graph_correction,accept_graph_correction,canonical_sha256
from .exporter_v2 import export_dxf_v2
from .graph_v2 import read_graph_v2
from .service_profiles_v2.storage import Storage
from .service_profiles_v2.config import Settings
from .service_v1.storage import digest
from .web_correction_contract_v1 import published,validate_edit


def write(path,value):path.write_bytes(encoded(value))


def graph_previews(before,after,folder):
    """One fixed view-local frame; these are proposal graphs, never DXF renders."""
    entities=[*before['primitives'],*after['primitives']];points=[]
    for entity in entities:
        if entity['kind']=='line':points.extend([entity['start'],entity['end']])
        elif entity['kind']=='circle':
            x,y=entity['center'];r=entity['radius'];points.extend([[x-r,y-r],[x+r,y+r]])
    if not points:raise ValueError('No supported geometry to preview')
    left=min(p[0] for p in points);right=max(p[0] for p in points)
    bottom=min(p[1] for p in points);top=max(p[1] for p in points)
    scale=340/max(right-left,top-bottom,1e-6)
    def xy(p):return (200+(p[0]-(left+right)/2)*scale,200-(p[1]-(bottom+top)/2)*scale)
    for name,graph in [('before',before),('after',after)]:
        image=Image.new('RGB',(400,400),'white');draw=ImageDraw.Draw(image)
        for e in graph['primitives']:
            if e['kind']=='line':draw.line([xy(e['start']),xy(e['end'])],fill='#087baf',width=2)
            elif e['kind']=='circle':
                x,y=e['center'];r=e['radius'];draw.ellipse([xy((x-r,y+r)),xy((x+r,y-r))],outline='#087baf',width=2)
        image.save(folder/(name+'.png'))


def prepare(files,run,page,edit):
    edit,target=validate_edit(files,run,page,edit)
    prefix,entries,load=published(files,run,page)
    result={'status':'ready_for_review','request':edit.model_dump(mode='json'),
        'parent_version':run['version'],'page_index':page,'before':target['value'],'after':edit.new_value,
        'target_id':edit.target_id,'view_id':edit.view_id,'kind':edit.kind,
        'source_accuracy_approved':False,'training_eligible':False,'automatic_quality_credit':False,
        'original_scan_ink_modified':False,'physical_scale_approved_by_this_edit':False}
    private={'summary':result}
    if edit.kind=='scalar':
        graph=read_graph_v2(load(prefix+'-reviewed-graph.json'))
        request={'parent_graph_sha256':canonical_sha256(graph),'view_id':edit.view_id,'constraint_id':edit.target_id,
            'expected_old_value':edit.expected_old_value,'new_value':edit.new_value,
            'parameter_evidence_ref':edit.evidence_ref,'max_point_displacement_mm':edit.max_point_displacement_mm,
            'max_radius_change_mm':edit.max_radius_change_mm}
        bindings=load(prefix+'-correction-bindings.json')
        core=prepare_graph_correction(graph,request,bindings)
        private.update(core=core,core_request=request,bindings=bindings)
        if core['status']!='needs_branch_review':
            result.update(status='rejected',reason=core['reason'])
        else:
            result.update(core_proposal_sha256=core['proposal_sha256'],touched_entity_ids=core['touched_entity_ids'],
                dimension_updates=core['dimension_updates'],before_boundaries=core['before_boundaries'],
                after_boundaries=core['after_boundaries'],preview_kind='view_local_graph_not_DXF',
                preview_view_id=edit.view_id)
    else:
        result.update(touched_entity_ids=[edit.target_id],dimension_updates=[],
            preview_kind='reading_metadata_only',
            effect='Меняется проверенное чтение/метаданные. Краска скана и физическая геометрия сохраняются.')
    # Bind approval to semantics, source context and the implementation, not temporary paths/timing.
    result['implementation_sha256']=digest(Path(__file__))
    result['proposal_sha256']=canonical_sha256(result)
    return private


def metadata_dxf(path,manual):
    doc=ezdxf.readfile(path)
    name='CAD_RECOVERY_MANUAL_V1'
    record=doc.rootdict.get(name)
    if record is None:record=doc.rootdict.add_xrecord(name)
    text=json.dumps(manual,ensure_ascii=True,sort_keys=True,separators=(',',':'))
    record.reset([(1,text[i:i+240]) for i in range(0,len(text),240)])
    doc.saveas(path)
    check=ezdxf.readfile(path);audit=check.audit()
    if audit.errors or audit.fixes:raise ValueError('Corrected DXF failed roundtrip audit')
    reread=''.join(tag.value for tag in check.rootdict.get(name).tags if tag.code==1)
    if json.loads(reread)!=manual:raise ValueError('DXF metadata roundtrip failed')
    return {'dxf_sha256':digest(path),'audit_errors':0,'audit_fixes':0,'metadata_updated':True}


def build(request):
    files=Storage(Settings(Path(request['storage']).resolve(),'unused'))
    run=request['parent'];page=request['page'];edit=request['edit'];folder=Path(request['folder'])
    output=folder/'output';output.mkdir()
    source=files.path(request['input_path'])
    if digest(source)!=run['input_sha256']:raise ValueError('Actual original input changed')
    proposal=prepare(files,run,page,edit);summary=proposal['summary']
    prefix,entries,load=published(files,run,page)
    if request['stage']=='preview':
        if summary['status']=='ready_for_review' and edit['kind']=='scalar':
            before=load(prefix+'-reviewed-graph.json');after=proposal['core']['candidate_graph']
            selected=edit['view_id']
            before={**before,'primitives':[e for e in before['primitives'] if e['view_id']==selected]}
            after={**after,'primitives':[e for e in after['primitives'] if e['view_id']==selected]}
            graph_previews(before,after,output)
        write(output/'proposal.json',proposal);write(output/'report.json',summary)
        return summary
    approval=request['approval']
    if summary['status']!='ready_for_review' or approval['proposal_sha256']!=summary['proposal_sha256']:
        raise ValueError('Stale recomputed correction proposal')
    # Copy all evidence into a new, unpublished attempt; no hard links to old versions.
    parent_path,parent_manifest=files.manifest(run)
    for record in parent_manifest.get('inventory',parent_manifest['files']):
        relative=Path(record['path'])
        if relative.as_posix()=='output/evidence-inventory.json':continue
        if edit['kind']=='scalar' and relative.as_posix()=='output/'+prefix+'.dxf':continue
        if relative.parts[0]!='output':raise ValueError('Unexpected parent evidence path')
        src=files.path((parent_path.parent/relative).relative_to(files.root).as_posix())
        if src.stat().st_size!=record['size_bytes'] or digest(src)!=record['sha256']:raise ValueError('Parent evidence changed')
        dest=output/relative.relative_to('output');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
        if digest(dest)!=record['sha256']:raise ValueError('Evidence changed during copy')
    manual=load(prefix+'-manual-graph.json',{'schema_version':'manual-review-graph/1.0','readings':{},'title_fields':{},'history':[]})
    record={'proposal_id':request['proposal_id'],'value':edit['new_value'],'previous_value':edit['expected_old_value'],
        'review_ref':approval['review_ref'],'actor':request['actor'],'parent_version':run['version'],
        'source_manifest_sha256':run['manifest']['sha256'],'proposal_sha256':summary['proposal_sha256']}
    manual['history'].append({**record,'kind':edit['kind'],'target_id':edit['target_id'],'view_id':edit.get('view_id')})
    manual.update(source_sha256=run['input_sha256'],page_index=page,automatic_quality_credit=False,
        physical_scale_approved_by_this_edit=False,original_scan_ink_modified=False)
    export={}
    if edit['kind']=='scalar':
        core_approval={'proposal_sha256':proposal['core']['proposal_sha256'],'review_ref':approval['review_ref'],
            'scope':'reviewed_correction','branch_reviewed':approval['branch_reviewed'],
            'derived_dimensions_reviewed':approval['derived_dimensions_reviewed']}
        corrected=accept_graph_correction(load(prefix+'-reviewed-graph.json'),proposal['core_request'],proposal['bindings'],core_approval)
        write(output/(prefix+'-reviewed-graph.json'),corrected.model_dump(mode='json'))
        export=export_dxf_v2(corrected,output/(prefix+'.dxf'))
        checked={r['id'] for r in proposal['core']['stored_geometry_checks']}
        unresolved=[issue for issue in export['issues'] if not (issue['code']=='constraints_require_separate_verification'
            and set(issue['entity_ids'])<=checked)]
        if export['status']=='failed' or unresolved:raise ValueError('Corrected graph export has unresolved issues')
        from .demo_export_v1 import render_dxf
        random.seed(1816);render_dxf(output/(prefix+'.dxf'),output/(prefix+'-dxf.png'))
        export['manual_metadata']=metadata_dxf(output/(prefix+'.dxf'),manual)
        write(output/(prefix+'-manual-export.json'),export)
    else:
        collection='readings' if edit['kind']=='ocr_reading' else 'title_fields'
        manual[collection][edit['target_id']]=record
        if edit['kind']=='title_field' and prefix+'-graph.json' in entries:
            graph=load(prefix+'-graph.json')
            graph.setdefault('title_block',{})[edit['target_id']]={'value':edit['new_value'],
                'status':'verified','source_ref':'manual-correction:'+request['proposal_id']}
            write(output/(prefix+'-graph.json'),graph)
        if prefix+'.dxf' in entries:
            export=metadata_dxf(output/(prefix+'.dxf'),manual)
    if export:write(output/(prefix+'-manual-export.json'),export)
    write(output/(prefix+'-manual-graph.json'),manual)
    report=load('report.json');row=report['pages'][page]
    row['manual_correction']={**record,'kind':edit['kind'],'target_id':edit['target_id'],
        'automatic_quality_credit':False,'original_scan_ink_modified':False,'geometry_overlay_available':edit['kind']!='scalar'}
    if export:
        if edit['kind']=='scalar':
            row['demo_export']={'mode':'reviewed_geometry','empty':not bool(corrected.primitives),
                'geometry_entity_count':len(corrected.primitives),'dxf_sha256':digest(output/(prefix+'.dxf'))}
        else:row.setdefault('demo_export',{}).update(dxf_sha256=export['dxf_sha256'])
    row['status']='needs_review';report['status']='needs_review'
    report['manual_corrections_applied']=True;report['automatic_quality_credit']=False
    write(output/'report.json',report)
    write(output/(prefix+'-correction-proposal.json'),proposal)
    write(output/(prefix+'-correction-approval.json'),{'approval':approval,'actor':request['actor']})
    result=deepcopy(parent_manifest['summary']);result.update(status='needs_review',manual_corrections_applied=True,automatic_quality_credit=False)
    result['pages'][page]=deepcopy(row)
    files.manifest(run)
    if digest(source)!=run['input_sha256']:raise ValueError('Source changed during correction')
    return result
