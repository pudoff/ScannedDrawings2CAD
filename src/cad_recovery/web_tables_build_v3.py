"""Safe table previews and source-bound text-only successors in owned scratch."""
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import ExitStack
from collections import Counter
import shutil
import ezdxf
import numpy as np

from .dataset_layout_v1 import read,encoded,digest
from .service_profiles_v9.storage import Storage
from .web_tables_contract_v3 import binding,parse_request
from .web_tables_context_v3 import describe,validate_edit
from .web_text_selection_v1 import evidence,public_graph,apply_selection
from .table_text_reconcile_v1 import reconcile as reconcile_text
from . import web_tables_build_v2 as previous
from . import native_table_export_v2 as safe


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(encoded(value))


def finalize(request,summary,text_report):
    folder=Path(request['folder']);output=folder/'output';index=request['page'];prefix=f'page-{index+1:03d}'
    from .annotation_style_v1 import render_review_dxf
    render_review_dxf(output/f'{prefix}.dxf',output/f'{prefix}-dxf.png')
    report=read(output/'report.json');page=next(p for p in report['pages'] if p['page_index']==index)
    detail=page['table_refinement'];detail.update(extension_binding=request['table_binding'],
        text_review_available=True,text_actions=text_report['actions'],
        manual_text_associations=len(text_report.get('manual_associations',[])),
        automatic_quality_credit=False,reviewer=request['edit']['reviewer'])
    doc=ezdxf.readfile(output/f'{prefix}.dxf')
    page['demo_export'].update(dxf_sha256=digest(output/f'{prefix}.dxf'),
        preview_sha256=digest(output/f'{prefix}-dxf.png'),table_refinement=detail,
        modelspace_counts=dict(Counter(e.dxftype() for e in doc.modelspace())),
        layer_counts=dict(Counter(e.dxf.layer for e in doc.modelspace())))
    report['table_refinement']=detail
    report['processing_execution']=dict(parent=report.get('processing_execution'),
        interface_contract='safe_manual_table_text/1.0',extension_binding=request['table_binding'])
    write(output/f'{prefix}-text-review.json',text_report)
    write(output/f'{prefix}-demo-export.json',page['demo_export']);write(output/f'{prefix}-report.json',page)
    write(output/'report.json',report)
    aliaspath=output/f'{prefix}-table-aliases.json'
    if aliaspath.exists():
        aliases=read(aliaspath)
        for handle,value in aliases['aliases'].items():
            entity=doc.entitydb.get(handle)
            if entity and entity.dxftype() in ('TEXT','MTEXT') and not value['status'].startswith('retired'):
                value.update(status='retained_with_text_decision',target_handles=[handle],layer=entity.dxf.layer)
        write(aliaspath,aliases)
    summary.update(has_changes=True,table_refinement=detail,
        pages=[{k:v for k,v in p.items() if k in ('page_index','status','issues','demo_export','table_refinement')} for p in report['pages']])
    write(output/'profile-artifacts.json',dict(schema_version='profile-artifacts/table-text/1.0',
        profile_sha256=report['processing_profile']['sha256'],extension_binding_sha256=request['table_binding']['sha256'],
        files=[dict(name=p.name,sha256=digest(p),size_bytes=p.stat().st_size) for p in sorted(output.iterdir())
               if p.is_file() and p.name!='profile-artifacts.json' and p.suffix in ('.json','.png','.dxf')]))
    write(folder/'completion.json',summary)
    return summary


def build(request,*,engine=None,progress=None):
    if request['table_binding']!=binding():raise ValueError('Table extension changed')
    edit=parse_request(deepcopy(request['edit']));folder=Path(request['folder']);output=folder/'output'
    files=Storage(SimpleNamespace(root=Path(request['storage']).resolve()))
    files.path(folder.relative_to(files.root).as_posix())
    prefix=f"page-{request['page']+1:03d}";slot=evidence(output,request['page'])
    if edit.text_selection:
        context,_=describe(files,request['document'],request['parent'],request['page']);validate_edit(context,edit)
        parent_path,_=files.manifest(request['parent'])
        sourcepath,source_manifest=files.manifest(dict(status='completed',manifest=request['text_source_manifest']))
        if request['text_source_manifest']['sha256']!=edit.text_selection.preview_sha256:
            raise ValueError('Selection source preview changed')
        output.mkdir()
        for item in source_manifest['inventory']:
            if item['path']=='output/evidence-inventory.json':continue
            rel=Path(item['path']);src=files.path((sourcepath.parent/rel).relative_to(files.root).as_posix())
            dest=output/rel.relative_to('output');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
        private=folder/'private';private.mkdir();target=private/'selected.dxf'
        text=apply_selection(parent_path.parent/'output'/f'{prefix}.dxf',sourcepath.parent/'output',request['page'],target,
            edit.text_selection.model_dump(mode='json'),edit.reviewer)
        shutil.copyfile(target,output/f'{prefix}.dxf')
        summary=deepcopy(request['text_source_summary'])
        summary['table_refinement']['proposal_id']=request['proposal_id']
        report=read(output/'report.json');page=next(p for p in report['pages'] if p['page_index']==request['page'])
        page['table_refinement']['proposal_id']=request['proposal_id'];write(output/'report.json',report)
        write(output/f'{prefix}-text-choice.json',dict(source_preview=request['text_source_manifest'],
            reviewer=edit.reviewer,choice=edit.text_selection.model_dump(mode='json'),automatic_quality_credit=False))
        # The source evidence remains fixed so further edits can be based on
        # the original alternatives, with the whole selection sent explicitly.
        result=finalize(request,summary,text)
        if digest(files.path(request['document']['input_path']))!=request['document']['input_sha256']:
            raise ValueError('Original source changed')
        return result

    original_export=safe.previous_export
    def captured_export(*args,**kwargs):
        value=original_export(*args,**kwargs)
        slot.mkdir(parents=True,exist_ok=True);shutil.copyfile(args[3],slot/'candidate.dxf')
        return value
    def captured_reconcile(baseline,composed,target,*,graph,native_to_DXF,**kwargs):
        graph=public_graph(graph)
        context,_=describe(files,request['document'],request['parent'],request['page'])
        write(slot/'graph.json',graph)
        write(slot/'registration.json',dict(native_to_DXF=np.asarray(native_to_DXF).tolist(),
            native_to_working=context['native_to_working'],working_size_px=context['working_size_px']))
        value=reconcile_text(baseline,composed,target,graph=graph,native_to_DXF=native_to_DXF)
        write(slot/'review.json',value)
        return value
    with ExitStack() as stack:
        for name,value in [('binding',binding),('parse_request',parse_request),('describe',describe),
                           ('validate_edit',validate_edit),('export',safe.export)]:
            stack.enter_context(patch.object(previous,name,value))
        stack.enter_context(patch.object(safe,'previous_export',captured_export))
        stack.enter_context(patch.object(safe,'reconcile',captured_reconcile))
        if progress:
            analyze=previous.analyze
            def measured(*args,**kwargs):
                with progress.phase('table_structure_OCR'):return analyze(*args,**kwargs)
            stack.enter_context(patch.object(previous,'analyze',measured))
        summary=previous.build(request,engine=engine)
    if (slot/'review.json').exists():return finalize(request,summary,read(slot/'review.json'))
    return summary
