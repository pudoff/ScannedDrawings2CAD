"""Source-only OCR routing under the frozen family/split-v2 restrictions.

Routing is separate from label quality. These pure functions never authenticate
filesystem bytes, approve a native target or authorize model training. The
release consumer must pin the catalog/policy, rehash the actual source/crops,
and separately establish the OCR transcription review.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .family_catalog_v2 import canonical_sha, relative_path, require, stable_digest

SPLITS = {'train', 'validation', 'test'}


class OCRAssignmentV1(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, revalidate_instances='always')
    id: str = Field(min_length=1, max_length=256)
    task: Literal['ocr_recognition']
    source_path: str = Field(min_length=1, max_length=4096)
    source_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    split: Literal['train', 'validation', 'test']
    image_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')


def _index(rows, key, name):
    require(isinstance(rows, list), 'Expected '+name+' inventory')
    result = {row[key]: row for row in rows}
    require(len(result) == len(rows), 'Duplicate '+name+' identity')
    return result


def _checked_context(catalog, policy, exclusions):
    require(catalog['schema_version'] == policy['schema_version'] == '2.0' and
            catalog['purpose'] == 'lineage_catalog_without_training_approval' and
            policy['purpose'] == 'split_policy_pending_label_approval' and
            policy['catalog_content_sha256'] == stable_digest(catalog),
            'OCR routing needs the matching frozen catalog/policy')
    require(catalog['training_eligible'] is False and policy['training_eligible'] is False and
            catalog['historical_split_modified'] is False and policy['historical_split_modified'] is False and
            set(policy['splits']) == SPLITS and all(rows == [] for rows in policy['splits'].values()),
            'The frozen global split policy must remain observation-only and empty')
    files = _index(catalog['files'], 'path', 'source file')
    groups = _index(catalog['groups'], 'group_id', 'lineage group')
    reservations = _index(policy['group_reservations'], 'group_id', 'group reservation')
    require(set(groups) == set(reservations), 'Group reservation inventory differs')
    require(policy['historical_reservations'] == catalog['historical_reservations'],
            'Historical reservations changed between catalog and policy')
    members, by_sha, family_groups = {}, defaultdict(set), defaultdict(set)
    for identifier, group in groups.items():
        require(type(group['train_only']) is bool and group['historical_reserved_split'] in SPLITS | {None},
                'Invalid group restrictions')
        require(group['split'] is None and group['training_eligible'] is False,
                'Catalog group cannot supply training approval')
        require(not group['train_only'] or group['historical_reserved_split'] in (None, 'train'),
                'Train-only group conflicts with a historical holdout')
        require(reservations[identifier] == {'group_id': identifier,
                'reserved_split': group['historical_reserved_split'], 'train_only': group['train_only']},
                'Group reservation differs from catalog')
        family_ids = group['known_family_ids']
        require(isinstance(family_ids, list) and all(isinstance(x, str) and x for x in family_ids)
                and len(family_ids) == len(set(family_ids)), 'Invalid family coallocation identities')
        require(isinstance(group['confirmed_native_template_family_ids'], list) and
                set(group['confirmed_native_template_family_ids']) <= set(family_ids),
                'Confirmed native family lies outside group identities')
        for family in family_ids:
            family_groups[family].add(identifier)
        require(isinstance(group['members'], list) and bool(group['members']), 'A group needs explicit members')
        for path in group['members']:
            require(path in files and path not in members, 'Group members must form a unique file partition')
            members[path] = identifier
    require(set(members) == set(files), 'Group membership omits registered files')
    require(all(len(ids) == 1 for ids in family_groups.values()),
            'A known family spanning separate groups needs a reviewed merged catalog')
    for path, row in files.items():
        relative_path(path); canonical_sha(row['sha256'])
        require(row['group_id'] == members[path], 'File points to a different lineage group')
        group = groups[row['group_id']]
        require(type(row['train_only']) is bool and row['train_only'] == group['train_only'] and
                row['historical_reserved_split'] == group['historical_reserved_split'] and
                row['training_eligible'] is False and row['split'] is None,
                'File restrictions differ from lineage or contain unsupported approval')
        by_sha[row['sha256']].add(row['group_id'])
    require(all(len(ids) == 1 for ids in by_sha.values()),
            'Identical source bytes in different groups require a reviewed merged catalog')
    for edge in catalog.get('lineage_edges', []):
        require(edge['a'] in files and edge['b'] in files and members[edge['a']] == members[edge['b']],
                'A declared lineage edge crosses groups')
    for reservation in catalog['historical_reservations']:
        split = reservation['reserved_split']
        require(split in SPLITS and reservation['label_approval_inferred'] is False,
                'Historical allocation is not label approval')
        original = reservation['original_record']
        for prefix in ('input', 'target'):
            path = original[prefix+'_path']
            require(path in files and files[path]['sha256'] == original[prefix+'_sha256'],
                    'Historical source revision is missing or changed')
            require(groups[members[path]]['historical_reserved_split'] == split,
                    'Historical holdout reservation was dropped or changed')
    pairs = _index(catalog['pairs'], 'pair_id', 'candidate pair')
    dispositions = _index(policy['candidate_dispositions'], 'pair_id', 'candidate disposition')
    require(set(pairs) == set(dispositions), 'Policy candidate inventory differs')
    for identifier, pair in pairs.items():
        for prefix in ('input', 'target'):
            path = pair[prefix+'_path']
            require(path in files and files[path]['sha256'] == pair[prefix+'_sha256'] and
                    files[path]['group_id'] == pair['group_id'], 'Pair revision or lineage differs from files')
        require(pair['target_status'] in ('excluded', 'not_approved') and
                pair['annotation_approved'] is False and pair['training_eligible'] is False and pair['split'] is None,
                'Catalog pair cannot grant OCR or native-target approval')
        group = groups[pair['group_id']]
        require(pair['train_only'] == group['train_only'] and type(pair['train_only']) is bool and
                pair['group_reserved_split'] == group['historical_reserved_split'] and
                (pair['family_id'] is None or pair['family_id'] in group['known_family_ids']),
                'Pair family or restrictions differ from the group')
        if pair.get('family_status') == 'confirmed_template_from_pinned_native_label':
            require(pair['family_id'] in group['confirmed_native_template_family_ids'],
                    'Source pair claims a native family missing from the group evidence')
        expected = {key: pair[key] for key in ('pair_id', 'input_path', 'target_path', 'group_id', 'target_status', 'pending_reasons')}
        expected.update(split=None, training_eligible=False)
        require(dispositions[identifier] == expected, 'Candidate disposition differs from catalog')
    if exclusions is not None:
        require(policy['exclusions_content_sha256'] == stable_digest(exclusions) and
                exclusions['schema_version'] == '1.0' and exclusions['purpose'] == 'training_target_exclusions' and
                exclusions['training_eligible'] is False and exclusions['default_status'] == 'not_approved' and
                exclusions['split_modified'] is False,
                'Native exclusion policy changed')
        excluded = _index(exclusions['exclusions'], 'target_sha256', 'excluded native revision')
        for sha in excluded:
            canonical_sha(sha)
            require(excluded[sha]['scope'] == 'native_dxf_revision_and_derived_labels', 'Unknown native exclusion scope')
        for pair in pairs.values():
            require((pair['target_status'] == 'excluded') == (pair['target_sha256'] in excluded),
                    'Native exclusion disposition disagrees with current policy')
    return files, groups, pairs


def _derive(context, source_path, source_sha256):
    files, groups, pairs = context
    relative_path(source_path); canonical_sha(source_sha256)
    require(source_path in files, 'Unknown OCR source')
    source = files[source_path]
    require(source['sha256'] == source_sha256, 'OCR source revision differs from catalog')
    require(source['kind'] == 'image' and source['role'] == 'drawing' and PurePosixPath(source_path).suffix.lower() == '.png',
            'This release version accepts original PNG sources only')
    group = groups[source['group_id']]
    source_pairs = [p for p in pairs.values() if p['input_path'] == source_path]
    # These are routing identities, never semantic truth inferred for the PNG.
    families = sorted(group['known_family_ids'])
    known_source_family = any(p.get('family_status') == 'confirmed_template_from_pinned_native_label'
                              and p.get('family_id') in families for p in source_pairs)
    allowed = set(SPLITS)
    if group['train_only'] or not known_source_family:
        allowed &= {'train'}
    if group['historical_reserved_split'] is not None:
        allowed &= {group['historical_reserved_split']}
    return {'source_path': source_path, 'source_sha256': source_sha256,
            'group_id': group['group_id'], 'family_ids': families,
            'family_identity_scope': 'coallocation_restrictions_not_semantic_label_approval',
            'train_only': group['train_only'],
            'development_train_only_due_to_unreviewed_family': not known_source_family,
            'historical_reserved_split': group['historical_reserved_split'],
            'allowed_splits': sorted(allowed), 'lineage_member_count': len(group['members']),
            'native_target_dispositions': [
                {'pair_id': p['pair_id'], 'target_path': p['target_path'], 'target_sha256': p['target_sha256'],
                 'status': p['target_status'], 'annotation_approved': False,
                 'scope': 'informational_native_snapshot_not_an_OCR_label_source'}
                for p in sorted(source_pairs, key=lambda p: p['pair_id'])],
            'native_target_approved': False, 'physical_geometry_approved': False,
            'label_quality_approved': False, 'training_eligible': False,
            'routing_scope': 'partial_source_OCR_fields_only_other_group_labels_unchanged'}


def derive_source_lineage_v1(catalog, policy, source_path, source_sha256, *, exclusions=None):
    """Derive original PNG restrictions from authenticated caller-owned inputs.

    ``source_path`` is dataset-relative, matching ``catalog.files.path``.
    An excluded/unknown native target remains excluded/unknown and is not a
    label source. Independently reviewed PNG text may have its own task release.
    """
    return _derive(_checked_context(catalog, policy, exclusions), source_path, source_sha256)


def validate_ocr_assignments_v1(catalog, policy, records, *, exclusions=None):
    """Check a complete proposed OCR allocation without mutating global splits.

    Records contain exactly id, task='ocr_recognition', source_path,
    source_sha256, image_sha256 and split. Actual image-byte hashes and label
    review must be authenticated by the consumer. Pass the complete task
    release, including previously allocated records, to detect cross-release
    crop leakage; separate calls do not establish a global registry.
    """
    require(isinstance(records, list) and len(records) <= 100000, 'Expected a bounded OCR assignment list')
    context = _checked_context(catalog, policy, exclusions)
    parsed = [OCRAssignmentV1.model_validate(r.model_dump() if isinstance(r, OCRAssignmentV1) else r)
              for r in records]
    require(len({r.id for r in parsed}) == len(parsed), 'Duplicate OCR sample identity')
    owners, image_groups, derived, rows = {}, {}, {}, []
    for record in parsed:
        key = record.source_path, record.source_sha256
        if key not in derived:
            derived[key] = _derive(context, *key)
        lineage = derived[key]
        require(record.split in lineage['allowed_splits'],
                'OCR split violates train-only, family review or historical reservation')
        group = lineage['group_id']
        require(record.image_sha256 not in image_groups or image_groups[record.image_sha256] == group,
                'Cross-group duplicate OCR images require deduplication or explicit reviewed coallocation')
        image_groups[record.image_sha256] = group
        tokens = [('group', group), ('source', record.source_path), ('source_hash', record.source_sha256),
                  ('image_hash', record.image_sha256)]+[('family', family) for family in lineage['family_ids']]
        for token in tokens:
            require(token not in owners or owners[token] == record.split, 'OCR lineage or exact-content leakage across splits')
            owners[token] = record.split
        rows.append({**record.model_dump(), 'group_id': group, 'family_ids': lineage['family_ids'],
                     'train_only': lineage['train_only'],
                     'historical_reserved_split': lineage['historical_reserved_split']})
    return {'schema_version': 'ocr-routing/1.0', 'status': 'routing_consistent_label_review_required',
            'task': 'ocr_recognition', 'catalog_content_sha256': stable_digest(catalog),
            'policy_content_sha256': stable_digest(policy), 'records': rows,
            'source_lineages': sorted(derived.values(), key=lambda row: row['source_path']),
            'sample_count': len(rows), 'counts_by_split': {s: sum(r['split'] == s for r in rows) for s in sorted(SPLITS)},
            'global_split_policy_modified': False, 'native_target_approved': False,
            'physical_geometry_approved': False, 'label_quality_approved': False,
            'training_eligible': False,
            'scope': 'task_routing_only_not_an_instruction_or_authorization_to_train'}
