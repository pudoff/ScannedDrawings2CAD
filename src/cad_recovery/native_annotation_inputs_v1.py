"""Bind current OCR and optional review dimensions to the exact working raster."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .cad_object_inputs_v1 import load_observations
from .dataset_layout_v1 import pin,checked
from .grid_role_revision_v1 import sha


def load_inputs(root,observations,ocr_path,*,review_path=None):
    root = Path(root).resolve(); ocr_path = Path(ocr_path).resolve()
    packet,protection,refs,extra = load_observations(root,observations)
    refs['ocr'] = pin(root,ocr_path.relative_to(root).as_posix())
    ocr = json.loads(checked(root,refs['ocr']).read_text(encoding='utf8'))
    source = (ocr_path.parent/ocr['source']).resolve(); source.relative_to(root)
    refs['ocr_source'] = pin(root,source.relative_to(root).as_posix())
    if refs['ocr_source']['sha256']!=ocr['source_sha256']: raise ValueError('OCR source SHA differs')
    with Image.open(source) as image:
        original = image.convert('L')
        if list(original.size)!=ocr['processed_size_px'] or list(original.size)!=packet['summary']['input_size_px']:
            raise ValueError('OCR source/frame differs from CV input')
        original.thumbnail((max(packet['gray'].shape),max(packet['gray'].shape)),Image.Resampling.LANCZOS)
        if not np.array_equal(np.array(original),packet['gray']): raise ValueError('OCR thumbnail pixels differ')
    height,width = packet['gray'].shape; ow,oh = ocr['processed_size_px']
    matrix = np.asarray(packet['summary']['input_to_working'],float)
    expected = np.array([[width/ow,0,(width/ow-1)/2],[0,height/oh,(height/oh-1)/2],[0,0,1]])
    if matrix.shape!=(3,3) or not np.array_equal(matrix,expected): raise ValueError('OCR pixel-centre transform differs')
    lines = []; identities = set()
    if len(ocr['lines'])>10000: raise ValueError('OCR line budget exceeded')
    for row in ocr['lines']:
        if row['id'] in identities: raise ValueError('Duplicate OCR line ID')
        identities.add(row['id']); p = np.asarray(row['polygon_px'],float)
        if p.shape!=(4,2) or not np.isfinite(p).all(): raise ValueError('Finite OCR quadrilateral required')
        if row.get('source_ref',{}).get('sha256',ocr['source_sha256'])!=ocr['source_sha256']:
            raise ValueError('OCR line belongs to another image')
        mapped = p@matrix[:2,:2].T+matrix[:2,2]
        if np.any(mapped<-.5) or np.any(mapped[:,0]>width-.5) or np.any(mapped[:,1]>height-.5):
            raise ValueError('OCR outside working frame')
        raw = row.get('raw','')
        if not isinstance(raw,str) or len(raw)>4096: raise ValueError('Invalid OCR string')
        if any(ord(c)<32 and c not in '\n\r\t' for c in raw): raise ValueError('Unsupported OCR control character')
        confidence = row.get('confidence')
        if confidence is not None and (not isinstance(confidence,(int,float)) or not np.isfinite(confidence) or not 0<=confidence<=1):
            raise ValueError('Invalid OCR confidence')
        lines.append({**deepcopy(row),'polygon_px':mapped.tolist(),'original_polygon_px':p.tolist(),
            'status':'automatic_draft','physical_geometry_approved':False})
    review = {'items':[]}
    if review_path is not None:
        refs['review'] = pin(root,Path(review_path).resolve().relative_to(root).as_posix())
        review = json.loads(checked(root,refs['review']).read_text(encoding='utf8'))
        if review['source_pixel_sha256']!=sha(packet['gray']) or review['size_px']!=[width,height]:
            raise ValueError('Dimension review belongs to another raster')
    text_lookup = {r['id']:r for r in lines}; corrected = set()
    for item in review.get('text_items',[]):
        identity = item['ocr_line_id']; raw = item['raw']
        if identity not in text_lookup or identity in corrected: raise ValueError('Foreign or duplicate text review')
        if item.get('evidence_mode') not in ('agent_reviewed_draft','user_reviewed_draft') or not item.get('reviewer'):
            raise ValueError('Explicit text reviewer required')
        if not isinstance(raw,str) or len(raw)>4096 or any(ord(c)<32 and c not in '\n\r\t' for c in raw):
            raise ValueError('Invalid corrected text')
        row = text_lookup[identity]; row['original_ocr_raw'] = row['raw']; row['raw'] = raw
        row['text_review'] = deepcopy(item); row['status'] = item['evidence_mode']
        direction = item.get('reading_direction','polygon_baseline'); p = np.asarray(row['polygon_px'])
        orders = {'polygon_baseline':[0,1,2,3],'bottom_to_top':[3,0,1,2],
                  'top_to_bottom':[1,2,3,0],'reverse_baseline':[2,3,0,1]}
        if direction not in orders: raise ValueError('Unsupported reviewed reading direction')
        row['polygon_px'] = p[orders[direction]].tolist(); corrected.add(identity)
    folder = Path(observations)
    for name in ('heads.json','chains.json'):
        if (folder/name).is_file(): refs[name] = pin(root,(folder/name).resolve().relative_to(root).as_posix())
    heads = json.loads((folder/'heads.json').read_text(encoding='utf8')) if (folder/'heads.json').is_file() else []
    chains = json.loads((folder/'chains.json').read_text(encoding='utf8')) if (folder/'chains.json').is_file() else []
    if len({h['id'] for h in heads})!=len(heads): raise ValueError('Duplicate source head ID')
    items = review.get('items',[])
    if not isinstance(items,list) or any(not isinstance(r,dict) for r in items):
        raise ValueError('Dimension review items must be objects')
    graph = json.loads(checked(root,refs['graph.json']).read_text(encoding='utf8'))
    return packet,protection,{'lines':lines,'dimensions':items,'heads':heads,'chains':chains,'graph':graph,
        'binding':{'source_pixel_sha256':sha(packet['gray']),'ocr':refs['ocr'],'ocr_source':refs['ocr_source'],
            'ocr_to_working':matrix.tolist(),'review':refs.get('review')}},refs,extra
