"""Conservative source relationships from authenticated, assistant-read stamps.

This pure builder does not authenticate pixels or establish semantic families.
It preserves the full train registry and never admits a control sample. Exact
shared references, an explicitly incomplete reference and the 2959 suffix hold
retain distinct evidence and dispositions.
"""
from collections import Counter, defaultdict
from copy import deepcopy

from .coallocation_policy_v1 import build_coallocation_v1, validate_assignments_v1
from .family_catalog_v2 import canonical_sha, relative_path, require, stable_digest
from .ocr_control_selection_v2 import PROTOCOL_V2, _commitments


TRAIN_ANCHORS = {
    '362.03.2963СО': 'clear_sum_dataset/vse_png/1539588_01.png',
    '362.03.2965СО': 'clear_sum_dataset/vse_png/1579170.04 _Нов.png',
    '362.03.422ЦН1': 'clear_sum_dataset/vse_png/626278_04.png',
}
PARTIAL_REFERENCE = '362.03.2965'
PARTIAL_ANCHOR = TRAIN_ANCHORS['362.03.2965СО']
SUFFIX_HOLD = frozenset({'2959СО', '362.03.2959СО'})
READING_FIELDS = {'rank', 'id', 'input_path', 'input_sha256', 'title_image', 'title_designation',
                  'title_name', 'related_designations', 'decision', 'notes', 'visual_checks'}
FLAGS = {'split': None, 'annotation_approved': False, 'training_eligible': False,
         'semantic_family_approved': False, 'independent_evaluation': False}


def _same(left, right):
    return stable_digest(left) == stable_digest(right)


def _pin(value):
    require(isinstance(value, dict) and set(value) == {'path', 'sha256'}, 'Expected exact evidence pin')
    relative_path(value['path']); canonical_sha(value['sha256'])
    return deepcopy(value)


def _pins(values):
    by_path = {}
    for value in values:
        checked = _pin(value)
        require(checked['path'] not in by_path or by_path[checked['path']] == checked['sha256'],
                'Conflicting evidence revisions')
        by_path[checked['path']] = checked['sha256']
    return [{'path': path, 'sha256': sha} for path, sha in sorted(by_path.items())]


def _queue_context(queue, catalog, policy, exclusions, old_overlay, task_splits):
    require(isinstance(queue, dict) and queue.get('schema_version') == 'ocr-control-queue/2.0'
            and _same(queue.get('protocol'), PROTOCOL_V2), 'Expected frozen V2 selection queue')
    require(all(queue.get(key) is False for key in ('OCR_run', 'annotation_approved', 'training_eligible',
        'semantic_family_approved', 'independent_evaluation', 'source_modified', 'global_split_policy_modified'))
        and queue.get('split') is None, 'Queue cannot approve model use or change source/split')
    for name, value in [('catalog', catalog), ('policy', policy), ('exclusions', exclusions),
                        ('overlay', old_overlay), ('task_splits', task_splits)]:
        require(queue['input_content_sha256'][name] == stable_digest(value), 'Queue context revision differs: '+name)
    rebuilt = build_coallocation_v1(catalog, policy, exclusions, old_overlay['constraints'])
    require(_same(rebuilt, old_overlay), 'Previous overlay is not reproducible')
    commitments = _commitments(task_splits, catalog, policy, exclusions, old_overlay)
    require(_same(queue['inherited_train_commitments'], task_splits['train_commitments']), 'Queue changed inherited commitments')
    files = {row['path']: row for row in catalog['files']}
    old_groups = {row['group_id']: row for row in old_overlay['groups']}
    entries, components, seen_paths, seen_groups = {}, {}, set(), set()
    for component in queue['components']:
        cid = component['id']; gids = component['overlay_group_ids']
        require(cid not in components and isinstance(gids, list) and gids == sorted(set(gids)) and gids,
                'Duplicate or invalid queue component')
        require(not seen_groups.intersection(gids) and set(gids) <= set(old_groups), 'Component overlay partition differs')
        expected = sorted(path for gid in gids for path in old_groups[gid]['members'])
        require(component['member_paths'] == expected and not seen_paths.intersection(expected), 'Component omits whole-group members')
        require(cid == 'ocr-hint-component-'+stable_digest(gids)[:24], 'Component identity differs')
        require(component['split'] is None and component['independent_evaluation'] is False, 'Component cannot assign splits')
        seen_groups.update(gids); seen_paths.update(expected); components[cid] = component
    require(seen_groups == set(old_groups) and seen_paths == set(files), 'Incomplete queue group partition')
    for row in queue['entries']:
        identifier = row['id']; path = row['input_path']; component = components.get(row['component_id'])
        require(identifier not in entries and path in files and files[path]['sha256'] == row['input_sha256']
                and files[path]['kind'] == 'image' and component is not None and path in component['member_paths'],
                'Queue source identity or membership differs')
        require(row['coallocation_group_id'] == old_overlay['file_to_group'][path]
                and _same(row['rank'], component['rank']) and row['exclusion_reasons'] == component['exclusion_reasons'],
                'Queue source component state differs')
        require(row['eligibility'] == ('blocked' if component['exclusion_reasons'] else 'pending_review')
                and row['split'] is None and row['independent_evaluation'] is False, 'Queue source admission differs')
        entries[identifier] = row
    require(len({row['input_path'] for row in entries.values()}) == len(entries), 'Repeated queue source path')
    for component in components.values():
        members = sorted((row for row in entries.values() if row['component_id'] == component['id']),
                         key=lambda row: (row['input_sha256'], row['input_path']))
        require(component['candidate_ids'] == [row['id'] for row in members]
                and component['representative_id'] == (members[0]['id'] if members else None), 'Component representative or sibling list differs')
    ordered = [components[cid] for cid in queue['ordered_component_ids']]
    require(len(ordered) == len({row['id'] for row in ordered}) and
            {row['id'] for row in ordered} == {row['id'] for row in components.values() if row['candidate_ids'] and not row['exclusion_reasons']},
            'Ranked queue must cover every eligible component')
    require(all(type(row['rank']) is int for row in ordered) and [row['rank'] for row in ordered] == list(range(len(ordered))),
            'Ranked components must have contiguous integer ranks')
    old_members = {path for commitment in commitments.values() for path in commitment['member_paths']}
    return files, entries, components, ordered, old_members


def _readings(readings, ordered, entries, reading_pins):
    require(isinstance(readings, list) and len(readings) == len(ordered), 'Missing representative readings')
    expected = {row['representative_id']: row['rank'] for row in ordered}
    require(isinstance(reading_pins, dict) and set(reading_pins) == set(expected), 'Reading evidence inventory differs')
    observed = {}
    for row in readings:
        require(isinstance(row, dict) and READING_FIELDS <= set(row) <= READING_FIELDS|{'additional_viewed_images'}, 'Unexpected reading fields')
        identifier = row['id']; candidate = entries.get(identifier)
        require(identifier in expected and identifier not in observed and type(row['rank']) is int
                and row['rank'] == expected[identifier] and candidate is not None
                and all(row[key] == candidate[key] for key in ('input_path', 'input_sha256')), 'Reading identity or source revision differs')
        _pin(row['title_image']); _pin(reading_pins[identifier])
        require(isinstance(row.get('additional_viewed_images', []), list), 'Additional views must be a list')
        _pins(row.get('additional_viewed_images', []))
        require(isinstance(row['visual_checks'], dict) and set(row['visual_checks']) == {'title_seen', 'overview_seen'}
                and row['visual_checks']['title_seen'] is True and type(row['visual_checks']['overview_seen']) is bool, 'Reading requires explicit title inspection')
        require(all(row[key] is None or isinstance(row[key], str) and row[key].strip() for key in ('title_designation', 'title_name')),
                'Invalid title observation')
        require(isinstance(row['notes'], list) and row['notes'] and all(isinstance(note, str) and note.strip() for note in row['notes']), 'Missing reading limits')
        refs = row['related_designations']
        require(isinstance(refs, list) and all(isinstance(ref, str) and ref.strip() for ref in refs)
                and len(refs) == len(set(refs)), 'Invalid reference observations')
        decision = 'known_train_link' if set(refs).intersection(TRAIN_ANCHORS) else 'needs_further_review'
        require(row['decision'] == decision, 'Reading decision disagrees with literal complete reference')
        observed[identifier] = row
    require(set(observed) == set(expected), 'Reading registry omits or repeats a ranked representative')
    return observed


def build_relations_v2(queue, catalog, policy, exclusions, old_overlay, readings, selection_pin, reading_pins, *, task_splits):
    """Propagate reviewed relationships through entire groups, without admission.

    All snapshots, authored image evidence and pins are authenticated by the
    wrapper. Counts follow the complete supplied registry; real-corpus wrappers
    additionally enforce the pinned 80 representatives / 383 sources / 90 rows.
    """
    files, entries, components, ordered, old_members = _queue_context(queue, catalog, policy, exclusions, old_overlay, task_splits)
    observed = _readings(readings, ordered, entries, reading_pins)
    selection_pin = _pin(selection_pin)
    _pins([selection_pin, *reading_pins.values(), *[row['title_image'] for row in observed.values()],
           *[pin for row in observed.values() for pin in row.get('additional_viewed_images', [])]])
    by_reference = defaultdict(list)
    for row in observed.values():
        for reference in row['related_designations']:
            by_reference[reference].append(row)
    constraints = deepcopy(old_overlay['constraints'])
    added = []

    def add(identifier, members, basis, evidence, reason):
        members = sorted(set(members))
        require(len(members) >= 2, 'Relationship constraint must bind at least two paths')
        added.append({'id': identifier, 'member_paths': members, 'basis': basis,
                      'evidence': _pins(evidence), 'reason': reason})

    for component in ordered:
        if len(component['overlay_group_ids']) > 1:
            add('control-v2-hint-'+component['id'].removeprefix('ocr-hint-component-'), component['member_paths'],
                'unresolved_series_hold', [selection_pin], 'Keep the complete frozen filename-hint component together; semantic family remains unapproved.')
    for reference, anchor in TRAIN_ANCHORS.items():
        rows = by_reference[reference]
        if rows:
            require(anchor in old_members and anchor in files, 'Known reference anchor is outside inherited train commitments')
            add('control-v2-reference-'+stable_digest(reference)[:16], [anchor, *[row['input_path'] for row in rows]],
                'reviewed_shared_reference', [selection_pin, *[reading_pins[row['id']] for row in rows]],
                'Literal reviewed shared reference '+reference+' connects the representative sources to an inherited train anchor; field meaning and semantic family are not asserted.')
    partial = by_reference[PARTIAL_REFERENCE]
    if partial:
        require(PARTIAL_ANCHOR in old_members and PARTIAL_ANCHOR in files, 'Partial-reference anchor is outside inherited train commitments')
        add('control-v2-partial-reference-2965', [PARTIAL_ANCHOR, *[row['input_path'] for row in partial]],
            'unresolved_series_hold', [selection_pin, *[reading_pins[row['id']] for row in partial]],
            'The literal reading 362.03.2965 lacks the final letters. Preserve this unresolved possible train connection without completing the transcription.')
    suffix_rows = [row for row in observed.values() if set(row['related_designations']).intersection(SUFFIX_HOLD)]
    if len(suffix_rows) > 1:
        add('control-v2-unresolved-suffix-2959', [row['input_path'] for row in suffix_rows], 'unresolved_series_hold',
            [selection_pin, *[reading_pins[row['id']] for row in suffix_rows]],
            'Keep literal 2959СО and 362.03.2959СО observations together conservatively; missing prefixes and field meaning remain uncertain, with no train or evaluation admission.')
    require(not {row['id'] for row in constraints}.intersection(row['id'] for row in added), 'New constraint identity collides with previous evidence')
    overlay = build_coallocation_v1(catalog, policy, exclusions, constraints+added)
    core = [{key: row[key] for key in ('id', 'source_path', 'source_sha256', 'image_sha256', 'split')}
            for row in task_splits['assignments']]
    routing = validate_assignments_v1(catalog, policy, exclusions, overlay, core)
    groups = {row['group_id']: row for row in overlay['groups']}
    train_groups = {overlay['file_to_group'][path] for path in old_members}
    commitments = [{'coallocation_group_id': gid, 'member_paths': groups[gid]['members'], 'split': 'train',
        'reason': 'Preserve every inherited train member and conservatively coallocated relatives; no new OCR labels are admitted.',
        'semantic_family_approved': False, 'independent_evaluation': False} for gid in sorted(train_groups)]
    expanded_members = {path for row in commitments for path in row['member_paths']}
    require(old_members <= expanded_members and {row['coallocation_group_id'] for row in routing['assignments']} <= train_groups,
            'Expanded routing lost an inherited train commitment')
    dispositions = []
    for row in sorted(entries.values(), key=lambda row: row['input_path']):
        component = components[row['component_id']]; gid = overlay['file_to_group'][row['input_path']]
        representative = observed.get(component['representative_id'])
        if row['eligibility'] == 'blocked':
            status, reason = 'blocked_previous_metadata', 'Existing metadata exclusion remains in force.'
        elif gid in train_groups:
            if representative['decision'] == 'known_train_link':
                status, reason = 'blocked_reviewed_train_link', 'A complete reviewed reference joins this whole component to inherited train.'
            else:
                status, reason = 'blocked_unresolved_train_link', 'A conservative unresolved relationship joins this whole component to inherited train.'
        else:
            status, reason = 'pending_relationship_review', 'No control admission: relationships and transcription remain unresolved.'
        dispositions.append({'id': row['id'], 'input_path': row['input_path'], 'input_sha256': row['input_sha256'],
            'component_id': component['id'], 'representative_id': component['representative_id'],
            'base_group_id': row['group_id'], 'coallocation_group_id': gid, 'status': status, 'reason': reason,
            'representative_review': None if representative is None else _pin(reading_pins[representative['id']]), **FLAGS})
    counts = Counter(row['status'] for row in dispositions)
    pending_groups = sorted({row['coallocation_group_id'] for row in dispositions if row['status'] == 'pending_relationship_review'})
    report = {'schema_version': 'ocr-control-relations-report/2.0', 'scope': 'relationship_review_without_control_or_label_admission',
        'reviewed_representative_count': len(observed),
        'exact_known_train_reference_representatives': sum(row['decision'] == 'known_train_link' for row in observed.values()),
        'needs_further_review_representatives': sum(row['decision'] == 'needs_further_review' for row in observed.values()),
        'partial_train_reference_representatives': len(partial), 'suffix_2959_representatives': len(suffix_rows),
        'source_count': len(dispositions), 'disposition_counts': {name: counts[name] for name in (
            'blocked_previous_metadata', 'blocked_reviewed_train_link', 'blocked_unresolved_train_link', 'pending_relationship_review')},
        'previous_constraint_count': len(constraints), 'added_constraint_count': len(added),
        'inherited_sample_count': len(core), 'new_sample_count': 0,
        'inherited_committed_member_count': len(old_members), 'expanded_committed_member_count': len(expanded_members),
        'expanded_train_group_count': len(train_groups), 'remaining_pending_group_count': len(pending_groups),
        'remaining_pending_group_ids': pending_groups, 'six_group_quota_available': len(pending_groups) >= 6,
        'control_release_created': False, 'actual_bytes_authenticated': False, 'OCR_run': False,
        'source_modified': False, 'global_split_policy_modified': False, **FLAGS,
        'limitations': ['Literal code readings are assistant observations, not proof of original font code points or field meaning.',
            'Previously blocked metadata components retain their exclusion without new whole-component merges that could conflict with history.',
            'Only selected representative stamps were reviewed; sibling relationships propagate conservatively without claiming every sibling was visually inspected.',
            'Potential remaining groups are not approved independent families or validation/test samples.',
            'The wrapper must authenticate every source, crop, input snapshot and review pin.']}
    routing = {**routing, 'inherited_sample_ids': [row['id'] for row in task_splits['assignments']],
               'inherited_train_commitments': deepcopy(task_splits['train_commitments']), 'train_commitments': commitments}
    return {'overlay': overlay, 'routing': routing, 'report': report, 'dispositions': dispositions}
