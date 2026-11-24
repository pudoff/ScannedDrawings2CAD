"""Resource completeness includes entities skipped before draw callbacks.

The capture mixin records resource skips separately from the bounded generic
frontend log. The metadata gate is pure: it never renders, reads or changes a
source document, and never invents missing INSERT provenance.
"""
from copy import deepcopy

from .reference_resources import IncompleteReferenceError

METHOD = 'drawn-and-skipped-native-resource-completeness-v1'
MAX_RESOURCE_SKIP_RECORDS = 4096
RESOURCE_TYPES = frozenset(('IMAGE', 'OLE2FRAME'))


class ResourceSkipCaptureMixin:
    """Capture every resource skip before the legacy generic message limit."""

    def __init__(self, *args, **kwargs):
        self.resource_skips = []
        self.resource_skip_count = 0
        super().__init__(*args, **kwargs)

    def skip_entity(self, entity, msg):
        if entity.dxftype() in RESOURCE_TYPES:
            if self.resource_skip_count >= MAX_RESOURCE_SKIP_RECORDS:
                raise ValueError('Complete resource skip audit exceeds its explicit record limit')
            native = entity.origin_of_copy or entity
            # virtual_entities can skip a non-copyable child before normal
            # visibility filtering. Resolve it in the active INSERT context.
            properties = self.ctx.resolve_all(entity)
            visible = bool(properties.is_visible)
            self.resource_skip_count += 1
            self.resource_skips.append({
                'entity_type': entity.dxftype(), 'entity_handle': entity.dxf.get('handle'),
                'native_handle': native.dxf.get('handle'),
                'native_owner_handle': native.dxf.get('owner'),
                'insert_path': None,
                'provenance_status': 'native_handle_and_owner_from_frontend_skip',
                'source_sha256': self._resource_source_sha,
                'native_skip_reason': str(msg), 'resource_required': visible,
                'resolved_layer': properties.layer,
                'visibility': 'visible' if visible else 'hidden',
            })
        return super().skip_entity(entity, msg)


def apply_resource_completeness(metadata, *, allow_incomplete_reference=False):
    """Return checked metadata; reject incomplete references by default.

    Full v5 skip capture takes precedence over the truncated generic log.
    Legacy metadata without full capture is conservative when that log is
    truncated: resource completeness is then unknown and cannot verify a pair.
    """
    if not isinstance(allow_incomplete_reference, bool):
        raise ValueError('allow_incomplete_reference must be an explicit boolean')
    if 'resource_completeness_policy' in metadata:
        raise ValueError('Resource completeness policy is already applied')
    result = deepcopy(metadata)
    source_sha = result['source_sha256']
    callback_complete = result['reference_complete']
    if not isinstance(callback_complete, bool):
        raise ValueError('Callback reference completeness must be boolean')
    observations = result['resource_observations']
    callback_issues = []
    for observation in observations:
        if observation.get('source_sha256') != source_sha:
            raise ValueError('Resource callback source provenance differs')
        required, loaded = observation.get('resource_required'), observation.get('loaded')
        if not isinstance(required, bool) or not isinstance(loaded, bool):
            raise ValueError('Resource callback completeness flags must be boolean')
        if required and not loaded:
            callback_issues.append(observation)

    generic_count, generic_skips = result['skipped_entity_count'], result['skipped_entities']
    if (isinstance(generic_count, bool) or not isinstance(generic_count, int)
            or generic_count < len(generic_skips)):
        raise ValueError('Generic skip audit counts are invalid')
    truncated = generic_count > len(generic_skips)
    full_capture = result.get('resource_skip_audit_complete') is True
    gaps = []
    if full_capture:
        skips = result['resource_skips']
        skip_count = result['resource_skip_count']
        if (isinstance(skip_count, bool) or not isinstance(skip_count, int)
                or skip_count != len(skips) or not 0 <= skip_count <= MAX_RESOURCE_SKIP_RECORDS):
            raise ValueError('Full resource skip audit counts are invalid')
    else:
        skips = [{'entity_type': row['type'], 'entity_handle': row['handle'],
                  'native_handle': row['handle'], 'native_owner_handle': None,
                  'insert_path': None, 'provenance_status': 'legacy_frontend_skip_handle_only',
                  'source_sha256': source_sha, 'native_skip_reason': row['reason'],
                  'resource_required': row['reason'] != 'invisible',
                  'resolved_layer': None, 'visibility': 'unknown' if row['reason'] != 'invisible' else 'hidden'}
                 for row in generic_skips if row['type'] in RESOURCE_TYPES]
        if truncated:
            gaps.append({'reason': 'resource_skip_audit_truncated',
                         'status': 'resource_completeness_unknown', 'reference_complete': False,
                         'source_sha256': source_sha, 'unrecorded_skip_count': generic_count-len(generic_skips)})
    hidden_count = 0
    for skip in skips:
        if (skip['entity_type'] not in RESOURCE_TYPES or skip['source_sha256'] != source_sha
                or not isinstance(skip['resource_required'], bool)):
            raise ValueError('Resource skip provenance or flags are invalid')
        if not skip['resource_required']:
            hidden_count += 1
            continue
        issue = {**skip, 'observation_kind': 'skipped_before_resource_callback',
                 'reason': 'native_resource_skipped_before_callback',
                 'status': 'incomplete_reference', 'reference_complete': False,
                 'loaded': False, 'diagnostic_placeholder': False}
        observations.append(issue)
        gaps.append(issue)
    complete = callback_complete and not callback_issues and not gaps
    result.update(callback_reference_complete=callback_complete,
                  reference_complete=complete,
                  reference_usage='comparison_candidate' if complete else 'diagnostic_only_incomplete_reference',
                  incomplete_reference_explicitly_allowed=allow_incomplete_reference,
                  resource_gaps=gaps,
                  resource_completeness_policy={
                      'method': METHOD, 'full_resource_skip_audit': full_capture,
                      'generic_skip_log_truncated': truncated,
                      'callback_observation_count': len(metadata['resource_observations']),
                      'captured_resource_skip_count': len(skips),
                      'hidden_resource_skip_count': hidden_count,
                      'required_resource_skip_count': len(skips)-hidden_count,
                      'unknown_resource_audit_gap_count': sum(gap['status'] == 'resource_completeness_unknown' for gap in gaps),
                      'max_resource_skip_records': MAX_RESOURCE_SKIP_RECORDS,
                      'insert_path_policy': 'Skip callbacks do not establish the INSERT path; null is not an empty direct-modelspace path.'})
    if not complete and not allow_incomplete_reference:
        issues = [*callback_issues, *gaps]
        if not issues:
            issues = [{'reason': 'upstream_reference_incomplete', 'source_sha256': source_sha}]
        raise IncompleteReferenceError(issues)
    return result
