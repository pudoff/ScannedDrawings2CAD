"""Preserve the rejected grid experiment while fixing external-context abstention."""
from .competitive_grid_admission_v2 import analyze as source_context
from .contour_grid_competition_v1 import analyze
from .grid_role_revision_v1 import cv_authority


def inference(gray, packet, proposed):
    authority = cv_authority(gray, packet['roles'], packet['files']['roles'])
    context = source_context(gray, packet['graph'], packet['roles'], packet['heads'],
                             packet['source_owner'], packet['ambiguous_owner_mask'], authority=authority)
    result = analyze(gray, packet['graph'], packet['source_owner'], packet['ambiguous_owner_mask'], proposed,
        frame_owner_numbers=context['frame_conflict_owner_numbers'],
        other_role_owner_numbers=context['other_conflict_owner_numbers'])
    if not context['source_immutable']:
        raise ValueError('Source context mutated its inputs')
    if not context['complete']:
        result['complete'] = False
        # Previous local evidence is retained for diagnosis, not counted as an
        # effective GRID veto once the whole proposal must be abandoned.
        result['discarded_local_grid_veto_pixels'] = result['grid_veto_pixels']
        result['mask'][:] = 0
        result['veto_mask'][:] = 0
        result['grid_veto_pixels'] = 0
        result['withdrawn_pixels'] = int((proposed > 0).sum())
        result['external_context_abstained_pixels'] = result['withdrawn_pixels']
        result['audit']['limits'].append('incomplete_source_context')
    metadata = {key: value for key, value in context.items()
                if key not in ('previous', 'old_candidate', 'old_owners', 'candidates', 'owners')}
    return result, metadata
