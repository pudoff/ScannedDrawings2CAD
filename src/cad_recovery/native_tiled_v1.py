"""Bounded, source-coordinate CV/OCR from exact prepared PNG samples.

Research kernel for P18.1b.1; no accepted graph, DXF, or service profile change.
"""
from copy import deepcopy
from pathlib import Path
import time
import numpy as np
from PIL import Image,ImageDraw
from .dataset_layout_v1 import digest,encoded,read
from .family_catalog_v2 import require
from .raster_regions import read_prepared_region
from .native_merge_v1 import points,vectors,bounds,geometry_box,stable_id,merge,conflicts

POLICY={'version':'native-tiled-observations/1.0','tile_size':1536,'overlap':256,'retry_margin':12,'max_retries':8,
 'max_windows':256,'max_geometry':50000,'max_texts':15000,'max_arrows':15000,
 'max_whole_region_pixels':12_000_000,'auto_deskew':False,'rotation_deg':0,
 'search_profile':'regional_masked','ocr_tile_size':1536,'ocr_overlap':192,'ocr_budget_seconds':240,
 'association':'contour-association-v3','physical_geometry_approved':False,'production_admitted':False}

def window_boxes(box,side=1536,overlap=256):
    require(type(side) is int and 128<=side<=1536 and type(overlap) is int and 0<=overlap<side,'Invalid tile parameters')
    x0,y0,x1,y1=box;require(all(type(v) is int for v in box) and x0<x1 and y0<y1,'Invalid region bounds')
    def starts(length):
        return sorted(set([0,*range(0,max(1,length-side+1),side-overlap),max(0,length-side)]))
    return [[x0+x,y0+y,min(x0+x+side,x1),min(y0+y+side,y1)] for y in starts(y1-y0) for x in starts(x1-x0)]

def window_id(box):return 'window-'+'-'.join(f'{n:05d}' for n in box)

def retry_boxes(observations,region,existing):
    """Source-blind retries from observations touching an internal crop edge."""
    proposed={};x0,y0,x1,y1=region;side=POLICY['tile_size']
    for r in observations:
        b=r['box']
        objects=[('text',bounds(t['polygon_px'])) for t in r['texts']]
        objects += [('curve',geometry_box(g)) for g in r['geometry'] if g['kind'] in ('arc','circle')]
        for kind,g in objects:
            if ((b[0]>x0 and g[0]<=b[0]+12) or (b[1]>y0 and g[1]<=b[1]+12) or
                (b[2]<x1 and g[2]>=b[2]-12) or (b[3]<y1 and g[3]>=b[3]-12)):
                cx=(g[0]+g[2])/2;cy=(g[1]+g[3])/2
                left=max(x0,min(int(cx-side/2),max(x0,x1-side)));top=max(y0,min(int(cy-side/2),max(y0,y1-side)))
                box=[left,top,min(left+side,x1),min(top+side,y1)]
                if tuple(box) not in existing:proposed[tuple(box)]=min(proposed.get(tuple(box),2),0 if kind=='text' else 1)
    ranked=sorted(proposed,key=lambda b:(proposed[b],b))
    return [list(b) for b in ranked[:POLICY['max_retries']]],len(ranked)-min(len(ranked),POLICY['max_retries'])

def analyze_window(prepared,box,folder,engine):
    from .preprocessing import prepare_scan
    from .geometry_detection import detect_geometry
    from .ocr import recognize_text
    from .arrow_detection import detect_arrows
    from .dimension_association_v3 import associate_dimensions_v3
    folder.mkdir(parents=True,exist_ok=False);pixels,proof=read_prepared_region(prepared,box,max_pixels=12_000_000)
    source=folder/'source.png';Image.fromarray(pixels).save(source);del pixels
    prep=prepare_scan(source,folder,auto_deskew=False)
    require(prep['transforms']['processed_to_input']==np.eye(3).tolist(),'Unexpected processed-image transformation')
    gray=folder/prep['artifacts']['grayscale'];geometry=detect_geometry(gray,folder,search_profile=POLICY['search_profile'])
    title=next((r['bbox_px'] for r in prep['regions'] if r['id']=='title-block-candidate'),None)
    ocr=recognize_text(gray,folder,engine=engine,processed_to_input=prep['transforms']['processed_to_input'],
                       title_bbox=title,tile_size=1536,overlap=192,time_budget_seconds=240)
    arrows=detect_arrows(gray,folder,geometry=geometry,text_polygons=[t['polygon_px'] for t in ocr['lines']])
    association=associate_dimensions_v3(gray,folder,geometry=geometry,ocr=ocr,arrows=arrows)
    matrix=np.array([[1,0,box[0]],[0,1,box[1]],[0,0,1]])
    identifier=window_id(box);uid_prefix=proof['source_sha256']+'/'+identifier;texts=[];geometries=[];arrow_rows=[]
    for line in ocr['lines']:
        b=bounds(line['polygon_px']);margin=min(b[0],b[1],box[2]-box[0]-b[2],box[3]-box[1]-b[3])
        texts.append({'uid':uid_prefix+'/'+line['id'],'raw':line['raw'],'confidence':line['confidence'],
           'polygon_px':points(line['polygon_px'],matrix),'edge_margin_px':margin,
           'parsed':next((n['parsed'] for n in ocr['notations'] if n['source_line_id']==line['id']),None)})
    for g in geometry['candidates']:
        row={k:deepcopy(v) for k,v in g.items() if k not in ('source_ref','id')};row['uid']=uid_prefix+'/'+g['id']
        row['source_window_id']=identifier
        row['evidence_coordinate_space']='origin window pixel centres/IDs; main geometry mapped to native source'
        if row['kind']=='line':row['points_px']=points(row['points_px'],matrix)
        else:row['center_px']=points([row['center_px']],matrix)[0]
        geometries.append(row)
    for a in arrows['candidates']:
        arrow_rows.append({'uid':uid_prefix+'/'+a['id'],'tip_px':points([a['tip_px']],matrix)[0],
                          'direction':vectors([a['direction']],matrix)[0],'confidence':a['confidence']})
    return {'id':identifier,'uid_prefix':uid_prefix,'box':box,'proof':proof,'native_to_processed':[[1,0,-box[0]],[0,1,-box[1]],[0,0,1]],
        'processed_to_native':matrix.tolist(),'texts':texts,'geometry':geometries,'arrows':arrow_rows,
        'local_associations':association['associations'],'ocr_complete':ocr['coverage']['complete'],
        'geometry_analysis':geometry['analysis'],'geometry_issues':geometry['issues']}

def assemble(observations,source_sha,region,*,deferred_retries=0):
    observations=sorted(observations,key=lambda r:r['id'])
    # Exact rectangle-union coverage without allocating a full-page bitmap.
    xs=sorted({region[0],region[2],*(v for r in observations for v in (r['box'][0],r['box'][2]))})
    covered=True
    for left,right in zip(xs,xs[1:]):
        if right<=region[0] or left>=region[2]:continue
        intervals=sorted((r['box'][1],r['box'][3]) for r in observations if r['box'][0]<=left and r['box'][2]>=right)
        end=region[1]
        for lo,hi in intervals:
            if lo>end:break
            end=max(end,hi)
        covered &= end>=region[3]
    require(covered,'Incomplete native window coverage')
    text_rows=[x for r in observations for x in r['texts']];geometry_rows=[x for r in observations for x in r['geometry']]
    arrow_rows=[x for r in observations for x in r['arrows']]
    require(len(text_rows)<=15000 and len(geometry_rows)<=50000 and len(arrow_rows)<=15000,'Native observation budget exceeded')
    texts,ta,tc=merge(text_rows,'text');geometry,ga,gc=merge(geometry_rows,'geometry');arrows,aa,ac=merge(arrow_rows,'arrow')
    relations=[]
    for row in observations:
        for a in row['local_associations']:
            prefix=row.get('uid_prefix',row['id'])
            source_text=prefix+'/'+a['source_line_id'];native_anchors=points(a['anchors_px'],row['processed_to_native']) if a['anchors_px'] else []
            references={}
            for key in ('measured_candidate_ids','dimension_line_candidate_ids','extension_line_candidate_ids','arrow_candidate_ids'):
                aliases=aa if key=='arrow_candidate_ids' else ga
                references[key]=[aliases[prefix+'/'+v] for v in a.get(key,[])]
            value={'source_window_id':row['id'],'local_association_id':a['id'],'source_text_id':ta[source_text],
                'anchors_px':native_anchors,'references':references,'status':a['status'],'accepted':False,
                'local_evidence':a,'local_evidence_coordinate_space':'processed window; use processed_to_native for points and its linear part for directions'}
            value['id']=stable_id('association',[source_sha,row['id'],a['id']]);relations.append(value)
    # Association proposals across windows remain distinct hypotheses with
    # explicit aliases. They are never silently promoted through deduplication.
    seam=[]
    for t in texts:
        if t['edge_margin_px']<=12:
            seam.append({'kind':'text','id':t['id'],'reason':'representative touches window boundary; retries may not resolve it','accepted':False})
    for g in geometry:
        if g['kind']=='arc':seam.append({'kind':'arc','id':g['id'],'reason':'partial observed sweep; no automatic arc stitching','accepted':False})
    return {'schema_version':'native-observations/1.0','source_sha256':source_sha,'region':region,
        'coordinate_system':'original encoded PNG pixel centres; half-open crop edges; no EXIF/resize/deskew applied',
        'texts':texts,'geometry':geometry,'arrows':arrows,'associations':sorted(relations,key=lambda a:a['id']),
        'aliases':{'texts':ta,'geometry':ga,'arrows':aa},'text_conflicts':conflicts(texts),'seam_review':seam,
        'deferred_retries':deferred_retries,'pixel_coverage_complete':True,
        'ocr_windows_complete':all(r['ocr_complete'] for r in observations),
        'geometry_search_incomplete':any(i['code'] in ('geometry_candidate_limit','skeleton_iteration_limit') for r in observations for i in r['geometry_issues']),
        'physical_geometry_approved':False,'production_admitted':False,'accepted_objects':0,
        'merge_checks':{'text':tc,'geometry':gc,'arrow':ac},
        'windows':[{'id':r['id'],'box':r['box'],'proof':r['proof'],'native_to_processed':r['native_to_processed'],
            'processed_to_native':r['processed_to_native'],'geometry_analysis':r['geometry_analysis']} for r in observations]}

def run(prepared,output,*,engine,region=None,mode='tiled',reverse_order=False):
    prepared=Path(prepared);output=Path(output);require(not output.exists(),'Choose new native output directory')
    report=read(prepared/'preparation.json');require(report['status']=='prepared' and report['source_integrity_validated'],'Unvalidated preparation')
    width,height=report['source_size_px'];region=[0,0,width,height] if region is None else region
    require(len(region)==4 and all(type(v) is int for v in region) and 0<=region[0]<region[2]<=width and 0<=region[1]<region[3]<=height,'Invalid native region')
    require(mode in ('tiled','whole_region'),'Unknown mode')
    require(mode!='whole_region' or (region[2]-region[0])*(region[3]-region[1])<=12_000_000,'Whole-region baseline exceeds native pixel limit; use tiled mode')
    boxes=window_boxes(region) if mode=='tiled' else [region]
    require(len(boxes)+8<=256,'Native window budget exceeded')
    output.mkdir(parents=True);started=time.perf_counter();observations=[]
    try:
        for box in reversed(boxes) if reverse_order else boxes:
            observations.append(analyze_window(prepared,box,output/window_id(box),engine))
        retries,deferred=retry_boxes(observations,region,{tuple(b) for b in boxes}) if mode=='tiled' else ([],0)
        for box in reversed(retries) if reverse_order else retries:
            observations.append(analyze_window(prepared,box,output/window_id(box),engine))
        result=assemble(observations,report['source_sha256'],region,deferred_retries=deferred)
        result.update(status='needs_review',policy=POLICY,mode=mode,source_size_px=[width,height],
            preparation_sha256=digest(prepared/'preparation.json'),base_window_count=len(boxes),retry_window_count=len(retries),
            elapsed_seconds=time.perf_counter()-started,source_original_rehashed=False,
            limitations=['Prepared tiles checked on access; original source must be authenticated by caller.',
                'This kernel consumes native RGB8 prepared PNG only; service/EXIF/PDF integration is a separate step.',
                'Association contexts stay window-local; arbitrary long dimensions and partial arcs may remain unresolved.',
                'No accepted graph or editable DXF is produced by this observation kernel.'])
        (output/'observations.json').write_bytes(encoded(result))
        return result
    except Exception as error:
        (output/'failure.json').write_bytes(encoded({'status':'failed','processed_windows':len(observations),'error':type(error).__name__+': '+str(error),'partial_outputs_admitted':False}))
        raise
