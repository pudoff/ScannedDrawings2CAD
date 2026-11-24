"""Acceptance-only replay controls; never installed into a processing profile."""
from copy import deepcopy
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import ezdxf
from .dataset_layout_v1 import read,encoded,digest
from .native_merge_v1 import points,vectors
from .native_tiled_v1 import assemble
from .annotation_style_v1 import replace_reviewed_ink,export_styled_review,raster_ink_from_dxf
from .noise_baseline_v1 import retention_metrics
from .exporter_v2 import semantic_entity_snapshot


def require(condition,message):
    if not condition:raise ValueError(message)


def write(path,value):path.write_bytes(encoded(value))


def reassemble_window_reports(output):
    """Independent file adapter: reconstruct raw observations from sealed producer reports."""
    stored=read(output/'private/page-001/observations.json');rows=[]
    for window in stored['windows']:
        folder=output/'private/page-001'/window['id'];matrix=window['processed_to_native']
        prefix=stored['source_sha256']+'/'+window['id'];width=window['box'][2]-window['box'][0];height=window['box'][3]-window['box'][1]
        geometry=read(folder/'page-001-geometry.json');ocr=read(folder/'page-001-ocr.json')
        arrows=read(folder/'page-001-arrows.json');associations=read(folder/'page-001-dimension-associations.json')
        texts=[];geometries=[];arrow_rows=[]
        for line in ocr['lines']:
            p=np.asarray(line['polygon_px']);margin=float(min(p[:,0].min(),p[:,1].min(),width-p[:,0].max(),height-p[:,1].max()))
            texts.append({'uid':prefix+'/'+line['id'],'raw':line['raw'],'confidence':line['confidence'],
                'polygon_px':points(line['polygon_px'],matrix),'edge_margin_px':margin,
                'parsed':next((n['parsed'] for n in ocr['notations'] if n['source_line_id']==line['id']),None)})
        for original in geometry['candidates']:
            g={k:deepcopy(v) for k,v in original.items() if k not in ('source_ref','id')}
            g.update(uid=prefix+'/'+original['id'],source_window_id=window['id'],
                evidence_coordinate_space='origin window pixel centres/IDs; main geometry mapped to native source')
            if g['kind']=='line':g['points_px']=points(g['points_px'],matrix)
            else:g['center_px']=points([g['center_px']],matrix)[0]
            geometries.append(g)
        for original in arrows['candidates']:
            arrow_rows.append({'uid':prefix+'/'+original['id'],'tip_px':points([original['tip_px']],matrix)[0],
                'direction':vectors([original['direction']],matrix)[0],'confidence':original['confidence']})
        rows.append({**window,'uid_prefix':prefix,'texts':texts,'geometry':geometries,'arrows':arrow_rows,
            'local_associations':associations['associations'],'ocr_complete':ocr['coverage']['complete'],
            'geometry_issues':geometry['issues']})
    return rows,stored


def real_seam_replay(output,target):
    target.mkdir();rows,stored=reassemble_window_reports(output)
    def replay(order):return assemble(order,stored['source_sha256'],stored['region'],deferred_retries=stored['deferred_retries'])
    forward=replay(rows);reverse=replay(list(reversed(rows)))
    alternating=replay(rows[::2]+rows[1::2])
    require(forward==reverse==alternating,'Window arrival order changed merged observations')
    for key,value in forward.items():require(stored[key]==value,'Stored native kernel differs from raw report replay: '+key)
    require(forward['accepted_objects']==0,'Seam replay admitted geometry')
    aliases=forward['aliases'];geo={g['id'] for g in forward['geometry']};arrows={a['id'] for a in forward['arrows']}
    text={t['id'] for t in forward['texts']}
    for relation in forward['associations']:
        require(relation['source_text_id'] in text,'Missing text after dedup')
        for key,ids in relation['references'].items():require(set(ids)<=(arrows if key=='arrow_candidate_ids' else geo),'Dangling geometry alias')
    result={'status':'passed','window_count':len(rows),'arrival_orders':3,
        'exact_packet_sha256':hashlib.sha256(encoded(forward)).hexdigest(),
        'raw_texts':sum(len(r['texts']) for r in rows),'merged_texts':len(forward['texts']),
        'raw_arrows':sum(len(r['arrows']) for r in rows),'merged_arrows':len(forward['arrows']),
        'raw_geometry':sum(len(r['geometry']) for r in rows),'merged_geometry':len(forward['geometry']),
        'text_conflicts':len(forward['text_conflicts']),'partial_arc_review':sum(r['kind']=='arc' for r in forward['seam_review']),
        'accepted':0,'extra_model_calls':0,'pixel_windows_verified_during_original_worker':True}
    write(target/'replayed.json',forward);write(target/'report.json',result);return result


def synthetic_seam_contract():
    """Controlled detector outputs distinguish dedup from unsupported semantic assembly."""
    boxes=[[0,0,260,120],[100,0,360,120],[200,0,460,120]];rows=[]
    for i,box in enumerate(boxes):
        uid=f'window-{i}';g=[{'uid':uid+'/long','kind':'line','points_px':[[box[0]+5,80],[box[2]-5,80]]},
            {'uid':uid+'/cross','kind':'line','points_px':[[230,5],[230,110]]},
            {'uid':uid+'/arc','kind':'arc','center_px':[230,50],'radius_px':30.,'start_angle_deg':30.,'sweep_angle_deg':80.}]
        # Native coordinates are intentional here; all three windows witness an overlap object.
        t=[{'uid':uid+'/text','raw':'Ø12','confidence':.9,'polygon_px':[[210,20],[250,20],[250,40],[210,40]],'edge_margin_px':10}]
        a=[{'uid':uid+'/arrow','tip_px':[230,80],'direction':[1.,0.],'confidence':.9}]
        rows.append({'id':uid,'box':box,'geometry':g,'texts':t,'arrows':a,'local_associations':[],
            'proof':{},'native_to_processed':np.eye(3).tolist(),'processed_to_native':np.eye(3).tolist(),
            'ocr_complete':True,'geometry_issues':[],'geometry_analysis':{}})
    # Keep distinct text fragments and conflicting readings; no GT-directed concatenation.
    rows[1]['texts'] += [{'uid':'window-1/conflict','raw':'012','confidence':.8,
        'polygon_px':[[210,20],[250,20],[250,40],[210,40]],'edge_margin_px':10}]
    for index,raw,left,right in [(0,'Ø',210,222),(2,'12',225,250)]:
        rows[index]['texts'].append({'uid':f'window-{index}/fragment','raw':raw,'confidence':.7,
            'polygon_px':[[left,20],[right,20],[right,40],[left,40]],'edge_margin_px':5})
    rows[1]['geometry'] += [{'uid':'window-1/partial','kind':'arc','center_px':[230,50],
        'radius_px':30.,'start_angle_deg':110.,'sweep_angle_deg':50.}]
    result=assemble(rows,'a'*64,[0,0,460,120]);reverse=assemble(rows[::-1],'a'*64,[0,0,460,120])
    require(result==reverse,'Synthetic order-dependent merge')
    aliases=result['aliases'];require(len({aliases['geometry'][f'window-{i}/long'] for i in range(3)})==1,'Long line fragmented')
    require(len(result['arrows'])==1 and len(result['texts'])==4,'Overlap text/arrow dedup or split raw changed')
    require({r['raw'] for r in result['texts']}=={'Ø12','012','Ø','12'},'Fragments were concatenated or dropped')
    require(result['text_conflicts'] and sum(r['kind']=='arc' for r in result['seam_review'])==2,'Conflict/partial arcs silently accepted')
    require(len([g for g in result['geometry'] if g['kind']=='line'])==2,'Crossing merged incorrectly')
    return {'status':'passed','kind':'controlled observations, not detector accuracy','packet':result,
        'long_line_windows':3,'dimension_text_conflicts_retained':len(result['text_conflicts']),'raw_fragments_retained':2,'arrow_duplicates_merged':3,
        'partial_arcs_retained':2,'intersecting_lines_separate':True,'model_calls':0}


def noise_and_style(root,automatic,target):
    target.mkdir();original=np.asarray(Image.open(automatic/'page-001-binary.png').convert('L'))==0
    noise=np.asarray(Image.open(root/'artifacts/noise-baseline-v3/noise-baseline-ink.png').convert('L'))>0
    useful=np.asarray(Image.open(root/'artifacts/noise-baseline-v3/useful-baseline-ink.png').convert('L'))>0
    unknown=np.asarray(Image.open(root/'artifacts/noise-baseline-v3/unknown-baseline-ink.png').convert('L'))>0
    old=np.asarray(Image.open(root/'artifacts/annotation-style-v5/real/original.png').convert('L'))==0
    require(np.array_equal(original,old),'Current draft changed frozen difficult-JPEG baseline')
    retention=retention_metrics(original,original,noise,useful,unknown)
    require(retention['useful_removed_px']==0 and retention['noise_removed_px']==0,'Unapproved filter became active')
    protected=np.asarray(Image.open(root/'artifacts/annotation-style-v5/real/protected.png').convert('L'))>0
    review=read(root/'configs/reviews/annotation-style-v2.json')
    derivative,removed,items,ledger=replace_reviewed_ink(original,protected,review,review['source_sha256'])
    require(len(items)==4 and int(removed.sum())==63 and not (removed&protected).any(),'Reviewed annotation replacement changed')
    require(not (derivative&removed).any() and np.array_equal(derivative|removed,original),'Double raster footprint')
    export=export_styled_review(automatic/'page-001.dxf',target/'styled.dxf',original,derivative,items)
    doc=ezdxf.readfile(target/'styled.dxf');require(np.array_equal(raster_ink_from_dxf(doc,original.shape),derivative),'Styled mask replay')
    source_doc=ezdxf.readfile(automatic/'page-001.dxf')
    def untouched(doc):return [semantic_entity_snapshot(e) for e in doc.modelspace() if e.dxf.layer not in
        ('SCAN_INK_DRAFT','ANNOTATION_LINES_REVIEW','ANNOTATION_HEADS_REVIEW')]
    require(untouched(doc)==untouched(source_doc),'Styling changed unrelated geometry/text')
    ambiguous=deepcopy(review);ambiguous['items']=[ambiguous['items'][0]];ambiguous['items'][0]['kind']='unknown'
    refused=replace_reviewed_ink(original,protected,ambiguous,review['source_sha256'])
    require(not refused[2] and not refused[1].any() and np.array_equal(refused[0],original),'Unknown role erased ink')
    result={'status':'passed','automatic_noise_retention':retention,'reviewed_replacement_pixels':63,
        'protected_pixels_changed':0,'reviewed_elements':4,'double_raster_footprint':False,
        'ambiguous_role_refused':True,'style_export':export,'integrated_automatic_styling':False,
        'physical_accuracy_approved':False,'manual_TrueView_acceptance':False}
    write(target/'ledger.json',ledger);write(target/'report.json',result);return result
