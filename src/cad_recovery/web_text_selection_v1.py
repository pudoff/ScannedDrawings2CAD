"""Immutable table text evidence, public alternatives and explicit associations."""
from pathlib import Path
import hashlib
import numpy as np
import ezdxf
from .dataset_layout_v1 import read,encoded,digest
from .table_text_reconcile_v1 import reconcile,selection_binding,SELECTED,visible


def evidence(output,page):
    return Path(output)/'text-evidence'/f'page-{page+1:03d}'


def public_graph(graph):
    from copy import deepcopy
    value=deepcopy(graph)
    for table in value['tables']:
        for ref in table.get('masks',{}).values():ref.pop('path',None)
    return value


def load(output,page):
    folder=evidence(output,page)
    if not (folder/'review.json').exists():return None
    graph,registration,review=(read(folder/name) for name in ('graph.json','registration.json','review.json'))
    if digest(folder/'candidate.dxf')!=review['binding']['composition_sha256']:
        raise ValueError('Text candidate changed')
    return folder,graph,registration,review


def box_working(box,matrix):
    l,t,r,b=box;p=np.array([[l,t,1],[r,b,1]])@np.asarray(matrix).T
    return [*p[:,:2].min(0).tolist(),*p[:,:2].max(0).tolist()]


def relevant_unassigned(review,graph):
    result=[]
    for item in review['unassigned_baseline']:
        l,t,r,b=item['glyph_box_px']
        if any(max(l,table['box_px'][0])<min(r,table['box_px'][2]) and
               max(t,table['box_px'][1])<min(b,table['box_px'][3]) for table in graph['tables']):
            result.append(item)
    return result


def packet(output,page,offset=0,limit=40):
    data=load(output,page)
    if data is None:return dict(enabled=False,reason='Этот просмотр создан прежним режимом. Подготовьте новый для выбора текста.')
    folder,graph,registration,review=data;matrix=registration['native_to_working']
    rows=[dict(kind='cell',**r) for r in review['decisions'] if r['baseline'] or r['candidate']]
    rows.extend(dict(kind='unassigned',**r) for r in relevant_unassigned(review,graph))
    rows.sort(key=lambda r:0 if r['kind']=='unassigned' else 1 if r.get('action')=='conflict_keep_baseline' else 2)
    chosen=[]
    for r in rows[offset:offset+limit]:
        if r['kind']=='cell':
            item={k:r.get(k) for k in ('kind','region_id','cell_id','action','reason','ownership_ambiguous')}
            for name in ('baseline','candidate'):
                item[name]=[{k:v[k] for k in ('handle','raw','active')}|dict(box_px=box_working(v['glyph_box_px'],matrix)) for v in r[name]]
        else:
            item={k:r[k] for k in ('kind','handle','raw','active')}
            item['box_px']=box_working(r['glyph_box_px'],matrix)
        chosen.append(item)
    saved=Path(output)/f'page-{page+1:03d}-text-choice.json'
    saved_value=read(saved) if saved.exists() else {}
    previous=saved_value.get('choice')
    return dict(enabled=True,review_sha256=digest(folder/'review.json'),rows=chosen,previous_choice=previous,
        previous_reviewer=saved_value.get('reviewer'),
        offset=offset,total=len(rows),has_more=offset+limit<len(rows),
        working_size_px=registration['working_size_px'],automatic_quality_credit=False)


def apply_selection(baseline,output,page,target,selection,reviewer):
    folder,graph,registration,review=load(output,page)
    if digest(folder/'review.json')!=selection['review_sha256']:raise ValueError('Text review changed')
    if selection_binding(baseline,folder/'candidate.dxf',graph,registration['native_to_DXF'])!=review['binding']:
        raise ValueError('Text source/baseline/frame changed')
    choices=dict(schema_version='table-text-choices/1.0',binding=review['binding'],reviewer=reviewer,
        selections=selection['selections'])
    available={r['handle']:r for r in relevant_unassigned(review,graph)}
    seen=set();associations=[];w,h=registration['working_size_px']
    # Validate all associations before changing any document or saved output.
    for row in selection['associations']:
        handle=row['handle'];box=row['box_px']
        if handle in seen or handle not in available:raise ValueError('Unknown or duplicate unassigned text')
        seen.add(handle);l,t,r,b=box
        if any(type(v) is not int for v in box) or not 0<=l<r<=w or not 0<=t<b<=h:
            raise ValueError('Association outside working page')
        glyph=box_working(available[handle]['glyph_box_px'],registration['native_to_working'])
        if not (l<=glyph[0] and t<=glyph[1] and glyph[2]<=r and glyph[3]<=b):
            raise ValueError('Draw an area containing the complete text')
        associations.append(dict(handle=handle,box_px=box,raw=available[handle]['raw'],reviewer=reviewer,
            kind='explicit_text_area_association',cell_structure_confirmed=False))
    report=reconcile(baseline,folder/'candidate.dxf',target,graph=graph,
        native_to_DXF=registration['native_to_DXF'],choices=choices)
    if associations:
        doc=ezdxf.readfile(target)
        for row in associations:doc.entitydb[row['handle']].dxf.layer=SELECTED
        doc.saveas(target);audit=ezdxf.readfile(target).audit()
        if audit.errors or audit.fixes:raise ValueError('Selected text CAD requires repair')
        report['dxf_sha256']=digest(target)
    report['manual_associations']=associations
    report['automatic_quality_credit']=False
    target.with_suffix('.json').write_bytes(encoded(report))
    return report
