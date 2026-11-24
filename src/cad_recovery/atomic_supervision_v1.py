"""Evaluate atomic candidates AFTER extraction; preserve legacy held channels."""
from copy import deepcopy

from .object_supervision_protocol_v1 import pair_channels, role_channels
from .role_relation_release_v1 import assemble_patch


def attach(reading,gray,graph,rows,pixel_fit_targets,previous_source,policy):
    result=assemble_patch(reading,gray,graph,rows,pixel_fit_targets,policy)
    old_count=graph['legacy_node_count'];old_ids=set(graph['node_ids'][:old_count])
    old_objects={o['id']:o for o in previous_source['source_objects']}
    for obj in result['source_objects']:
        if set(obj['nodes'])&old_ids!=set(old_objects[obj['id']]['nodes']):
            raise ValueError('Legacy object matching changed')
    roles,role_provenance=role_channels(pixel_fit_targets,result['source_objects'],graph['node_ids'])
    legacy_pairs=deepcopy(previous_source['review']['relations'])
    known={tuple(sorted(r['nodes'])) for r in legacy_pairs}
    new_positive=[]
    for row in result['review']['relations']:
        if row['relation']=='same_primitive' and tuple(sorted(row['nodes'])) not in known:
            new_positive.append(row)
    score_input=deepcopy(result)
    score_input['review']['relations']=legacy_pairs+new_positive
    old_events={e['id']:e for e in previous_source['source_contacts']}
    for event in score_input['source_contacts']:
        event['candidate_pair_available']=old_events[event['id']]['candidate_pair_available']
    pairs=pair_channels(score_input,graph,[r['points_px'] for r in rows])
    pairs['legacy_evaluation_pairs']=legacy_pairs
    new_keys={tuple(sorted(r['nodes'])) for r in new_positive}
    for row in pairs['fit_pairs']:
        if tuple(sorted(row['nodes'])) in new_keys:
            row['origin']='new_atomic_candidates_inside_same_reviewed_primitive'
    result['pairs']=pairs
    result['additional_same_primitive_fit_pairs']=len(new_positive)
    result['role_fit_provenance']=role_provenance
    return result,roles
