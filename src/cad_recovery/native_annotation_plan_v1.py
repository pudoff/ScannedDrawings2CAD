"""Owned OCR components and explicitly sourced, unitless dimension drafts."""
from collections import Counter
from copy import deepcopy
import cv2
import numpy as np
from .cad_object_export_v1 import _ownership
from .grid_role_revision_v1 import sha
from .part_admission_v1 import object_sha


def chain_dimensions(packet,annotations):
    """Consume already bound chain observations; do not infer missing OCR links."""
    rows = []; rejected = []; geometry = {c['id']:c for c in packet['geometry']}
    intervals = {i:c['id'] for c in geometry.values() for i in c.get('interval_ids',[c['interval_id']] if c.get('interval_id') else [])}
    heads = {h['id']:h for h in annotations['heads']}
    for chain in annotations.get('chains',[]):
        identity = 'chain-'+object_sha(chain)[:20]
        try:
            text = chain.get('ocr_line_ids',[])
            if not text: raise ValueError('chain_awaiting_explicit_OCR_binding')
            feet = np.asarray(chain['feet_px'],float)
            if feet.shape!=(2,2) or len(chain['head_ids'])!=2: raise ValueError('incomplete_dimension_chain')
            refs = list(chain['shaft_interval_ids'])+[i for group in chain['extension_interval_ids'] for i in group]
            selected = sorted({intervals[i] for i in refs})
            tips = np.asarray([heads[i]['tip_px'] for i in chain['head_ids']],float)
            anchor_ids = [c['id'] for c in geometry.values() if c['id'] not in selected and any(
                np.linalg.norm(np.asarray(c['support_points'])-p,axis=1).min()<=2 for p in feet)]
            rows.append({'id':identity,'kind':'aligned','points_px':feet.tolist(),'placement_px':tips.mean(0).tolist(),
                'source_candidate_ids':selected,'anchor_source_candidate_ids':anchor_ids,
                'head_ids':chain['head_ids'],'ocr_line_ids':text,'evidence_mode':'automatic_chain_draft',
                'origin_chain_sha256':object_sha(chain)})
        except (KeyError,TypeError,ValueError) as error:
            rejected.append({'id':identity,'reason':str(error),'status':'needs_review'})
    return rows,rejected


def polygon_mask(shape,polygon):
    p = np.asarray(polygon,float)
    if p.shape!=(4,2) or not np.isfinite(p).all() or abs(cv2.contourArea(p.astype(np.float32)))<1:
        raise ValueError('Nondegenerate OCR quadrilateral required')
    mask = np.zeros(shape,np.uint8); cv2.fillConvexPoly(mask,np.rint(p).astype(np.int32),255)
    return mask>0


def dimension_plans(packet,annotations):
    geometry = {c['id']:c for c in packet['geometry']}; heads = {h['id']:h for h in annotations['heads']}
    lines = {r['id']:r for r in annotations['lines']}; accepted = []; rejected = []; used = set(); ids = set()
    if len(annotations['dimensions'])>128: raise ValueError('Dimension budget exceeded')
    for value in annotations['dimensions']:
        row = deepcopy(value)
        try:
            if row['id'] in ids: raise ValueError('duplicate_dimension_id')
            ids.add(row['id'])
            mode = row['evidence_mode']
            if mode=='automatic_chain_draft':
                source = next((c for c in annotations.get('chains',[]) if object_sha(c)==row.get('origin_chain_sha256')),None)
                if source is None or source.get('feet_px')!=row['points_px'] or source.get('head_ids')!=row.get('head_ids'):
                    raise ValueError('automatic_chain_binding_differs')
            elif mode not in ('agent_reviewed_draft','user_reviewed_draft') or not row.get('reviewer'):
                raise ValueError('dimension_requires_explicit_review_or_verified_adapter')
            if row['kind'] not in ('linear','aligned','radius','diameter','angular','leader'): raise ValueError('unsupported_dimension_kind')
            points = np.asarray(row['points_px'],float)
            count = len(points) if row['kind']=='leader' and 2<=len(points)<=16 else 3 if row['kind']=='angular' else 2
            if points.shape!=(count,2) or not np.isfinite(points).all(): raise ValueError('invalid_dimension_points')
            height,width = packet['gray'].shape
            if np.any(points<0) or np.any(points[:,0]>=width) or np.any(points[:,1]>=height): raise ValueError('dimension_outside_source')
            placement = np.asarray(row['placement_px'],float)
            if placement.shape!=(2,) or not np.isfinite(placement).all() or np.any(placement<0) or placement[0]>=width or placement[1]>=height:
                raise ValueError('invalid_dimension_placement')
            if np.linalg.norm(points[0]-points[1])<1: raise ValueError('degenerate_dimension')
            u,v = (points[1]-points[0],points[-1]-points[0])
            if row['kind']=='angular' and (np.linalg.norm(v)<1 or abs(u[0]*v[1]-u[1]*v[0])<1e-6):
                raise ValueError('degenerate_angle')
            sources = row.get('source_candidate_ids',[]); anchors = row.get('anchor_source_candidate_ids',[])
            for values in (sources,anchors,row.get('head_ids',[]),row.get('ocr_line_ids',[])):
                if not isinstance(values,list) or any(not isinstance(i,str) for i in values) or len(values)!=len(set(values)):
                    raise ValueError('duplicate_or_invalid_annotation_reference')
            if row['kind']=='leader' and len(row.get('head_ids',[]))!=1: raise ValueError('standalone_leader_needs_one_observed_head')
            if not np.isfinite(row.get('text_height_px',16)) or not 0<float(row.get('text_height_px',16))<=max(height,width):
                raise ValueError('invalid_annotation_text_height')
            if not anchors or not set(sources+anchors)<=geometry.keys(): raise ValueError('foreign_dimension_source')
            if set(sources)&used: raise ValueError('dimension_source_already_owned')
            if not set(row.get('head_ids',[]))<=heads.keys() or not set(row.get('ocr_line_ids',[]))<=lines.keys():
                raise ValueError('foreign_dimension_head_or_text')
            q = np.concatenate([np.asarray(geometry[i]['support_points'],float) for i in anchors])
            if row['kind'] in ('linear','aligned','angular','leader'):
                if any(np.linalg.norm(q-point,axis=1).min()>2 for point in points): raise ValueError('dimension_anchor_not_source_supported')
            else:
                radius = np.linalg.norm(points[1]-points[0])
                if np.max(np.abs(np.linalg.norm(q-points[0],axis=1)-radius))>2:
                    raise ValueError('dimension_circle_not_source_supported')
            row.update(source_members=[{'candidate_id':i,'source_observation_id':geometry[i]['source_observation_id'],
                'source_index_interval':geometry[i]['source_index_interval'],'interval_ids':geometry[i].get('interval_ids',[])}
                for i in sorted(set(sources+anchors))],units='px' if row['kind']!='angular' else 'degrees',
                physical_geometry_approved=False,source_pixel_sha256=sha(packet['gray']))
            used.update(sources); accepted.append(row)
        except (KeyError,TypeError,ValueError) as error:
            rejected.append({'id':row.get('id'),'reason':str(error),'status':'needs_review'})
    # Duplicate text claims would create two native displays of the same notation.
    claims = Counter(i for row in accepted for i in row.get('ocr_line_ids',[]))
    duplicate = {i for i,count in claims.items() if count>1}
    if duplicate:
        retained = []
        for row in accepted:
            if duplicate.intersection(row.get('ocr_line_ids',[])):
                rejected.append({'id':row['id'],'reason':'dimension_text_already_owned','status':'needs_review'})
            else: retained.append(row)
        accepted = retained
    return accepted,rejected


def compile_annotations(packet,annotations):
    gray,roles = packet['gray'],packet['roles']; _,raw = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    if annotations['binding']['source_pixel_sha256']!=sha(gray): raise ValueError('Annotation binding differs from source')
    generated,chain_rejections = chain_dimensions(packet,annotations)
    normalized = {**annotations,'dimensions':annotations['dimensions']+generated}
    ink = raw>0; dimensions,rejected = dimension_plans(packet,normalized)
    rejected += chain_rejections
    consumed = {i for d in dimensions if d['kind']!='leader' for i in d.get('ocr_line_ids',[])}
    _,labels,stats,_ = cv2.connectedComponentsWithStats(raw,connectivity=8)
    sizes = stats[:,cv2.CC_STAT_AREA]; text_plans = []; claims = Counter(); protection = np.zeros_like(ink)
    known_labels = set(map(int,np.unique(labels[ink & np.isin(roles,[1,2,3,4,5,7,8,9])])))
    for row in annotations['lines']:
        if not row['raw'].strip(): continue
        mask = polygon_mask(gray.shape,row['polygon_px']); protection |= mask
        selected,count = np.unique(labels[mask & ink],return_counts=True)
        full = [int(i) for i,n in zip(selected,count) if n==sizes[i]]
        partial = [int(i) for i,n in zip(selected,count) if n!=sizes[i]]
        # A known nontext role is a conflict, not permission to erase geometry.
        known_conflict = bool(known_labels.intersection(full))
        plan = {'line':deepcopy(row),'full_components':full,'partial_components':partial,
            'conflict':known_conflict,'consumed_by_dimension':row['id'] in consumed}
        p = np.asarray(row['polygon_px'],float)
        plan['orientation_uncertain'] = '\n' not in row['raw'] and len(row['raw'])>2 and np.linalg.norm(p[3]-p[0])>1.5*np.linalg.norm(p[1]-p[0])
        text_plans.append(plan); claims.update(full)
    replacement = np.zeros_like(ink); text_replacement = np.zeros_like(ink)
    for plan in text_plans:
        shared = any(claims[i]>1 for i in plan['full_components'])
        safe = not plan['conflict'] and not shared and bool(plan['full_components'])
        owned = np.isin(labels,plan['full_components']) if safe else np.zeros_like(ink)
        visible = safe and (not plan['partial_components'] or plan['consumed_by_dimension']) and not plan['orientation_uncertain']
        plan['visible'] = bool(visible)
        plan['reason'] = None if visible else 'orientation_requires_review' if plan['orientation_uncertain'] else 'source_component_conflict_or_partial_ownership'
        if visible: text_replacement |= owned
    candidates = {c['id']:c for c in packet['geometry']}; heads = {h['id']:h for h in annotations['heads']}
    dim_masks = []
    for row in dimensions:
        selected = row.get('source_candidate_ids',[])
        virtual = {'objects':[{'support_points':candidates[i]['support_points'],
            'source_members':[{'candidate_id':i}]} for i in selected]}
        _,owned,_ = _ownership(gray,virtual,packet['geometry'],roles==6)
        for identity in row.get('head_ids',[]):
            p = np.asarray(heads[identity]['polygon_px'],float)
            if p.shape!=(3,2) or not np.isfinite(p).all(): raise ValueError('Invalid reviewed head')
            mask = np.zeros_like(gray); cv2.fillPoly(mask,[np.rint(p).astype(np.int32)],255)
            area = mask>0
            if not area.any() or np.mean(ink[area])<.7: raise ValueError('Reviewed head lacks source ink')
            owned |= area & ink & (roles!=6)
        dim_masks.append(owned); protection |= owned; replacement |= owned
    replacement |= text_replacement
    return {'text':text_plans,'dimensions':dimensions,'dimension_rejections':rejected,
        'protection':protection,'replacement':replacement,'text_replacement':text_replacement,
        'dimension_masks':dim_masks,'consumed_ocr_ids':sorted(consumed),
        'source_sha256':sha(gray),'source_immutable':True}
