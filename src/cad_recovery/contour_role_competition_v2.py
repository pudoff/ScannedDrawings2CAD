"""Resolve only an unconfirmed grid alternative against a bound open witness."""
from copy import deepcopy
import numpy as np
from .contour_role_competition_v1 import guard as previous,POLICY as PRIOR
from .part_admission_v1 import object_sha
from .grid_role_revision_v1 import sha

POLICY={**PRIOR,'version':'contour-role-competition/2.0',
    'weak_grid_alternative_requires_independent_confirmation':True,
    'positive_requirement':'complete_source_bound_native_pen_checked_open_section',
    'frame_annotation_and_certified_grid_veto_preserved':True}


def resolved_context(gray,graph,owner,proposed,context,positive):
    audit={'relaxed_track_ids':[],'relaxed_owner_numbers':[],'reason':'no_bound_positive_witness'}
    if positive is None:return context,audit
    if (not positive.get('complete') or positive.get('policy',{}).get('version')!='open-section-witness/5.0'
            or positive.get('source_sha256')!=sha(gray) or positive.get('graph_sha256')!=object_sha(graph)
            or positive.get('source_owner_sha256')!=sha(owner)
            or positive.get('original_sha256')!=sha(gray)
            or not np.array_equal(positive.get('mask'),proposed)):
        raise ValueError('Positive witness is incomplete or belongs to another source/graph/mask')
    items=graph['proposals'];numbers={p['id']:i for i,p in enumerate(items,1)}
    tracks={t['id']:t for t in graph['tracks']};certified=set()
    competition=context.get('competition',{})
    if not context.get('complete') or not competition.get('complete',False):
        audit['reason']='negative_context_incomplete';return context,audit
    for network in competition.get('networks',[]):
        if network.get('qualified'):
            certified.update(network['source_owner_numbers'])
    accepted={tuple(c['boundary_track_ids']) for c in positive.get('native_checks',[]) if c['accepted']}
    boundaries=set()
    for witness in positive.get('witnesses',[]):
        if tuple(witness['boundary_track_ids']) not in accepted:continue
        ns=set(witness['source_owner_numbers'])
        if any(type(n) is not int or not 1<=n<=len(items) for n in ns):raise ValueError('Foreign positive owner')
        for identity in witness['boundary_track_ids']:
            if identity not in tracks:raise ValueError('Foreign positive boundary track')
            current={numbers[i] for i in tracks[identity]['interval_ids']}
            if not current<=ns:raise ValueError('Positive boundary has unbound owners')
            boundaries|=current
    output=deepcopy(context)
    for line in output['context']['lines']:
        if line.get('role')!='unknown' or set(line['alternatives'])!={'part_contour','title_grid'}:continue
        if line['id'] not in tracks:raise ValueError('Foreign competing track')
        owners={numbers[i] for i in tracks[line['id']]['interval_ids']}
        if not owners or not owners<=boundaries or owners&certified:continue
        line['alternatives']=[a for a in line['alternatives'] if a!='title_grid']
        # This is a negative-guard resolution, never a new PART assignment.
        audit['relaxed_track_ids'].append(line['id']);audit['relaxed_owner_numbers']+=sorted(owners)
    audit['relaxed_owner_numbers']=sorted(set(audit['relaxed_owner_numbers']))
    audit['reason']='source_bound_witness_vs_unconfirmed_grid_alternative'
    return output,audit


def guard(gray,graph,source_owner,ambiguous_owner,proposed,context,*,positive=None):
    original=object_sha([graph,context]);mask_sha=sha(proposed)
    resolved,audit=resolved_context(gray,graph,source_owner,proposed,context,positive)
    result=previous(gray,graph,source_owner,ambiguous_owner,proposed,resolved)
    if original!=object_sha([graph,context]) or mask_sha!=sha(proposed):raise ValueError('Competition resolution mutated input')
    return {**result,'policy':deepcopy(POLICY),'positive_resolution':audit,'positive_role_created':False}
