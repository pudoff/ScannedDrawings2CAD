"""Explicit, content-bound reviewed observations -> DrawingGraph 0.2 / P13.

This library accepts already reviewed contracts. HTTP inspection does not
expose this acceptance operation; applying edits belongs to P18.1e.
"""
from copy import deepcopy
import hashlib
import math
import numpy as np
from .dataset_layout_v1 import encoded
from .graph_v2 import (DrawingGraphV2,View,Page,Evidence,LineV2,CircleV2,ArcV2,TextV2,
                      DimensionV2,ConstraintV2)
from .schema import Source
from .observation_graph_v1 import matrix,transform
from .graph_constraint_solver_v2 import solve_graph_view


def fingerprint(observations):
    return hashlib.sha256(encoded(observations)).hexdigest()


def materialize(observations, *, expected_sha256, page, view, object_reviews=None,
                association_reviews=(), constraints=()):
    """No boolean approve-all, inferred mm, inferred view or automatic constraint."""
    if fingerprint(observations)!=expected_sha256:raise ValueError('Stale observation review')
    page=Page.model_validate(page);view=View.model_validate(view)
    if page.size_source=='unknown' or view.source_to_view is None or view.evidence.status!='verified' or not view.evidence.review_ref:
        raise ValueError('Reviewed page and source-to-view frame required')
    reviews=deepcopy(object_reviews or {});objects={o['id']:o for o in observations['objects']}
    if not set(reviews)<=set(objects):raise ValueError('Unknown reviewed observation ID')
    if reviews and not view.physical_scale_confirmed:raise ValueError('Accepting objects requires confirmed physical scale')
    for oid,review in reviews.items():
        if set(review)!={'review_ref','role'} or not isinstance(review['review_ref'],str) or not review['review_ref'].strip():
            raise ValueError('Each accepted observation needs an explicit review reference')
        if review['role'] not in ('part_geometry','annotation','text'):raise ValueError('Explicit reviewed role required')
        if (objects[oid]['kind']=='text')!=(review['role']=='text'):raise ValueError('Reviewed role does not match observation kind')
        if objects[oid]['kind'] not in ('geometry','text'):raise ValueError('Arrow observations cannot become part primitives')
        source=objects[oid]['observation']
        if 'annotation_support_only' in (source.get('role'),source.get('classification')) and review['role']=='part_geometry':
            raise ValueError('Annotation support cannot become measured part geometry')
    affine=matrix(view.source_to_view);linear=affine[:2,:2];scale=math.sqrt(abs(float(np.linalg.det(linear))))
    similarity=np.allclose(linear.T@linear,np.eye(2)*scale*scale,rtol=1e-9,atol=1e-9)
    primitives=[];texts=[]
    for oid,o in objects.items():
        if o['kind'] not in ('geometry','text'):continue
        reviewed=reviews.get(oid);accepted=reviewed is not None and reviewed['role'] in ('part_geometry','text')
        ev=Evidence(source_ref=oid,method='human_verified' if reviewed else ('ocr' if o['kind']=='text' else 'geometry_fit'),
                    status='verified' if reviewed else 'observed',review_ref=reviewed['review_ref'] if reviewed else None)
        common={'id':oid,'view_id':view.id,'status':'accepted' if accepted else 'proposed','evidence':ev}
        g=o['observation']
        if g.get('role')=='annotation_support_only' or g.get('classification')=='annotation_support_only':continue
        if o['kind']=='text':
            p=np.array(transform(g['polygon_px'],affine));edges=np.roll(p,-1,axis=0)-p
            axis=edges[np.argmax(np.linalg.norm(edges,axis=1))];normal=np.array([-axis[1],axis[0]])/np.linalg.norm(axis)
            texts.append(TextV2(**common,raw=g['raw'],position=p.mean(axis=0),height=max(.001,float(np.ptp(p@normal))),
                                rotation_deg=math.degrees(math.atan2(axis[1],axis[0]))));continue
        if g['kind']=='line':
            start,end=transform(g['points_px'],affine);primitives.append(LineV2(**common,start=start,end=end))
        elif g['kind'] in ('circle','arc'):
            if not similarity:raise ValueError('A circular observation needs a similarity source-to-view transform')
            center=transform([g['center_px']],affine)[0];radius=g['radius_px']*scale
            if g['kind']=='circle':primitives.append(CircleV2(**common,center=center,radius=radius))
            else:
                angles=[]
                for angle in (g['start_angle_deg'],g['start_angle_deg']+g['sweep_angle_deg']):
                    v=linear@np.array([math.cos(math.radians(angle)),math.sin(math.radians(angle))])
                    angles.append(math.degrees(math.atan2(v[1],v[0])))
                if np.linalg.det(linear)*g['sweep_angle_deg']<0:angles.reverse()
                primitives.append(ArcV2(**common,center=center,radius=radius,start_angle_deg=angles[0],end_angle_deg=angles[1]))
        else:raise ValueError('Unsupported geometry cannot be silently promoted')
    dimensions=[];bindings={};associations={a['id']:a for a in observations['associations']}
    geometries={p.id:p for p in primitives};seen=set()
    for review in association_reviews:
        if set(review)!={'association_id','alternative_index','review_ref','value','unit'}:raise ValueError('Explicit dimension fields required')
        identifier=review['association_id']
        if identifier in seen or identifier not in associations:raise ValueError('Unknown or duplicate reviewed association')
        seen.add(identifier);row=associations[identifier];index=review['alternative_index']
        if not view.physical_scale_confirmed:raise ValueError('Dimension acceptance requires confirmed units and scale')
        if not isinstance(review['review_ref'],str) or not review['review_ref'].strip():raise ValueError('Dimension review reference required')
        if index is None:
            if row['producer_status']!='proposed':raise ValueError('Unresolved relation requires an explicitly selected alternative')
            choice=row['active']
        else:
            if type(index)is not int or not 0<=index<len(row['alternatives']):raise ValueError('Invalid alternative selection')
            choice=row['alternatives'][index]
        if choice['unresolved_references']:raise ValueError('Cannot accept unresolved provenance IDs')
        kind=row['dimension_kind'];unit='degree' if kind=='angular' else 'mm'
        if kind not in ('linear','radius','diameter','angular') or review['unit']!=unit:raise ValueError('Dimension type/unit mismatch')
        nominal=row['nominal_reading'];value=review['value']
        if type(value) not in (int,float) or not math.isfinite(value) or value<=0 or nominal is None or not math.isclose(value,nominal,rel_tol=1e-12):
            raise ValueError('Accept an observed reading explicitly; text corrections require a new review version')
        refs=choice['references'];measured=refs['measured_candidate_ids']
        if not measured or not refs['dimension_line_candidate_ids'] or not refs['arrow_candidate_ids']:
            raise ValueError('Complete measured/shaft/arrow bindings required')
        if kind=='linear' and (len(refs['arrow_candidate_ids'])<2 or len(refs['extension_line_candidate_ids'])<2):
            raise ValueError('Linear dimension needs both arrows and witnesses')
        if any(v not in objects for values in refs.values() for v in values):raise ValueError('Referenced observation is absent')
        if any(g not in geometries or geometries[g].status!='accepted' for g in measured):
            raise ValueError('Every measured primitive must be explicitly accepted as part geometry')
        anchor=transform(choice['anchors_px'],affine)
        if len(anchor)!=(3 if kind=='angular' else 2):raise ValueError('Incomplete anchors')
        text=objects[row['source_text_id']]['observation'];placement=np.array(transform(text['polygon_px'],affine)).mean(axis=0)
        angle=0.
        if kind=='linear':
            observed_axis=choice['evidence'].get('measurement_axis_px')
            if observed_axis is None:raise ValueError('Linear measurement axis is not observed')
            axis=linear@matrix(choice['local_to_observation'])[:2,:2]@np.array(observed_axis,float)
            if axis.shape!=(2,) or not np.isfinite(axis).all() or np.linalg.norm(axis)<1e-12:raise ValueError('Invalid measurement axis')
            angle=math.degrees(math.atan2(axis[1],axis[0]))
        dim=DimensionV2(id='dimension:'+row['source_id'],view_id=view.id,status='accepted',
            evidence=Evidence(source_ref=identifier,method='human_verified',status='verified',review_ref=review['review_ref']),
            kind=kind,nominal=value,raw_text=row['raw'],points=anchor,placement=placement,measured_entity_ids=measured,angle_deg=angle)
        dimensions.append(dim)
        bindings[dim.id]={'observation_content_sha256':expected_sha256,'association_id':identifier,
            'source_text_id':row['source_text_id'],'selected_alternative':index,'references':deepcopy(refs),
            'anchors_source_px':deepcopy(choice['anchors_px']),'anchors_view':anchor,
            'anchor_ids':deepcopy(choice['anchor_ids']),'producer_anchor_ids':deepcopy(choice['producer_anchor_ids']),
            'anchor_id_origin':choice['anchor_id_origin'],
            'source_to_view':affine.tolist(),'view_id':view.id,'review_ref':review['review_ref'],
            'producer_evidence':deepcopy(choice['evidence']),'local_to_observation':choice['local_to_observation'],
            'original_relation_snapshot':deepcopy(row)}
    approved=[]
    for value in constraints:
        c=ConstraintV2.model_validate(value)
        if c.status!='accepted' or c.evidence.status!='verified' or not c.evidence.review_ref or c.view_id!=view.id:
            raise ValueError('Solver accepts only explicitly reviewed constraints in this view')
        if any(g not in geometries or geometries[g].status!='accepted' for g in c.entity_ids):
            raise ValueError('Constraint primitive is not accepted')
        approved.append(c)
    return DrawingGraphV2(source=Source(path='source-bound-observations',sha256=observations['source_sha256'],
        page_index=observations['page_index'],method='reviewed_observation_adapter/1.0'),page=page,views=[view],
        primitives=primitives,texts=texts,dimensions=dimensions,constraints=approved,
        provenance={'observation_adapter':'reviewed_observation_adapter/1.0','observation_content_sha256':expected_sha256,
            'observation_bindings':bindings,'object_reviews':reviews,'automatic_acceptance':False})


def solver_preview(graph,view_id):
    """P13 proposes only; never changes the supplied graph or its observations."""
    snapshot=graph.model_dump(mode='json')
    result=solve_graph_view(graph,view_id)
    if graph.model_dump(mode='json')!=snapshot:raise RuntimeError('Solver mutated reviewed graph')
    return result
