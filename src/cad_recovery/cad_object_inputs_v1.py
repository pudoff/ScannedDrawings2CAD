"""Load current CV observations with explicit source and interval bindings."""
from copy import deepcopy
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from .dataset_layout_v1 import pin, checked


def load_observations(root,folder):
    root,folder = Path(root).resolve(),Path(folder).resolve()
    folder.relative_to(root)
    names = ['gray.png','roles.png','geometry.json','graph.json','summary.json']
    refs = {name:pin(root,(folder/name).relative_to(root).as_posix()) for name in names}
    def read(name): return json.loads(checked(root,refs[name]).read_text(encoding='utf-8'))
    gray = np.array(Image.open(checked(root,refs['gray.png'])).convert('L'))
    roles = np.array(Image.open(checked(root,refs['roles.png'])))
    geometry = read('geometry.json'); graph = read('graph.json'); summary = read('summary.json')
    if summary.get('ground_truth_used') is not False: raise ValueError('Image-only observation provenance required')
    if summary['size_px']!=list(gray.shape[::-1]): raise ValueError('Observation frame differs')
    if roles.shape!=gray.shape or roles.dtype!=np.uint8: raise ValueError('Role frame differs')
    lookup = {c['id']:c for c in geometry}; intervals = {key:[] for key in lookup}
    ids = set()
    for p in graph['proposals']:
        if p['id'] in ids: raise ValueError('Duplicate graph interval ID')
        ids.add(p['id']); parent = lookup.get(p['parent_primitive_id'])
        if parent is None or p['path_id']!=parent['path_id'] or p['source_observation_id']!=parent['source_observation_id']:
            raise ValueError('Graph/source parent differs')
        low,high = p['source_index_interval']; a,b = parent['source_index_interval']
        if not a<=low<=high<=b or p['points_px']!=parent['support_points'][low-a:high-a+1]:
            raise ValueError('Graph interval support differs from geometry')
        intervals[parent['id']].append((low,high,p['id']))
    enriched = deepcopy(geometry)
    for c in enriched:
        spans = sorted(intervals[c['id']])
        if not spans or spans[0][0]!=c['source_index_interval'][0] or spans[-1][1]!=c['source_index_interval'][1] or any(
            a[1]!=b[0] for a,b in zip(spans,spans[1:])):
            raise ValueError('Graph intervals do not partition candidate')
        c['interval_ids'] = [identity for _,_,identity in spans]
    # Preserve source-supported head ink, even before P13.5 can bind it to a
    # dimension. This is protection, not promotion of a diagnostic head.
    protected = np.zeros_like(gray,bool); extra = {}
    if (folder/'chains.json').is_file() and (folder/'heads.json').is_file():
        for name in ('chains.json','heads.json'): refs[name] = pin(root,(folder/name).relative_to(root).as_posix())
        used = {identity for chain in read('chains.json') for identity in chain.get('head_ids',[])}
        preserved = []; _,ink = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
        for head in read('heads.json'):
            polygon = np.asarray(head.get('polygon_px',[]),float)
            if polygon.shape!=(3,2) or not np.isfinite(polygon).all(): continue
            if np.any(polygon<0) or np.any(polygon[:,0]>=gray.shape[1]) or np.any(polygon[:,1]>=gray.shape[0]): continue
            mask = np.zeros_like(gray); cv2.fillPoly(mask,[np.rint(polygon).astype(np.int32)],255)
            area = int(np.count_nonzero(mask))
            if not 3<=area<=4096 or np.mean(ink[mask>0]>0)<.7: continue
            protected |= mask>0
            preserved.append(head['id'])
        extra['linked_heads_protected'] = len(used)
        extra['observed_heads_preserved_as_ink'] = preserved
    for record in refs.values(): checked(root,record)
    return {'gray':gray,'roles':roles,'geometry':enriched,'summary':summary,'heads':[]},protected,refs,extra
