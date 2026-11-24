"""Hash-based exclusions for incomplete native targets, never training approval.

This separate contract is intended for future label/split consumers. A target
absent from this exclusion list remains unapproved, including changed revisions.
"""
from collections import defaultdict
import hashlib
from pathlib import Path, PurePosixPath
import re


SCOPE = 'native_dxf_revision_and_derived_labels'
DECISION = 'exclude_incomplete_native_target'
MANIFEST_KEYS = {'schema_version', 'purpose', 'decision_report', 'exclusions',
                 'default_status', 'training_eligible', 'split_modified'}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _text(value, label, *, limit=256):
    _require(isinstance(value, str) and 0 < len(value) <= limit
             and value.strip() == value and not any(ord(c) < 32 for c in value),
             label+' must be a nonempty, trimmed string without control characters')
    return value


def _sha(value):
    _require(isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value),
             'Expected a canonical SHA-256 digest')
    return value


def _pin(value):
    _require(isinstance(value, dict) and set(value) == {'path', 'sha256'},
             'Decision report requires an exact path and SHA-256 pin')
    path = _text(value['path'], 'Decision report path', limit=4096)
    parts = PurePosixPath(path)
    _require(not parts.is_absolute() and not any(p in {'.', '..'} for p in parts.parts)
             and parts.as_posix() == path and path != '.' and '\\' not in path and ':' not in path,
             'Decision report path must be a normalized relative POSIX path')
    return {'path': path, 'sha256': _sha(value['sha256'])}


def _strings(values, label):
    _require(isinstance(values, list) and 0 < len(values) <= 10000,
             label+' must be a nonempty bounded list')
    checked = [_text(value, label) for value in values]
    _require(len(checked) == len(set(checked)), label+' contains duplicate values')
    return checked


def build_exclusion_manifest(decision_report_pin, cases):
    """Group explicit exclusions by native content hash, regardless of pathname."""
    report = _pin(decision_report_pin)
    _require(isinstance(cases, list) and 0 < len(cases) <= 10000,
             'Expected a nonempty bounded list of resource decisions')
    grouped = defaultdict(lambda: {'case_ids': set(), 'reason_codes': set()})
    seen = set()
    for case in cases:
        _require(isinstance(case, dict) and case.get('decision') == DECISION,
                 'Every resource decision must explicitly exclude its incomplete native target')
        identifier = _text(case.get('case_id'), 'Case identifier')
        _require(identifier not in seen, 'Repeated resource case identifier')
        seen.add(identifier)
        target = _sha(case.get('target_sha256'))
        reasons = _strings(case.get('reason_codes'), 'Reason codes')
        grouped[target]['case_ids'].add(identifier)
        grouped[target]['reason_codes'].update(reasons)
    return {'schema_version': '1.0', 'purpose': 'training_target_exclusions',
            'decision_report': report,
            'exclusions': [{'target_sha256': target, 'case_ids': sorted(data['case_ids']),
                            'reason_codes': sorted(data['reason_codes']), 'scope': SCOPE}
                           for target, data in sorted(grouped.items())],
            'default_status': 'not_approved', 'training_eligible': False, 'split_modified': False}


def _validate_manifest(manifest):
    _require(isinstance(manifest, dict) and set(manifest) == MANIFEST_KEYS,
             'Unexpected exclusion manifest fields')
    _require(manifest['schema_version'] == '1.0'
             and manifest['purpose'] == 'training_target_exclusions'
             and manifest['default_status'] == 'not_approved'
             and manifest['training_eligible'] is False and manifest['split_modified'] is False,
             'Exclusion manifest cannot approve targets or alter split assignments')
    _pin(manifest['decision_report'])
    entries = manifest['exclusions']
    _require(isinstance(entries, list) and 0 < len(entries) <= 10000,
             'Expected a nonempty bounded exclusion inventory')
    indexed, case_ids = {}, set()
    for entry in entries:
        _require(isinstance(entry, dict)
                 and set(entry) == {'target_sha256', 'case_ids', 'reason_codes', 'scope'}
                 and entry['scope'] == SCOPE, 'Invalid target exclusion scope or fields')
        target = _sha(entry['target_sha256'])
        _require(target not in indexed, 'Duplicate target SHA in exclusion manifest')
        identifiers = _strings(entry['case_ids'], 'Case identifiers')
        _require(not case_ids.intersection(identifiers), 'Case identifier belongs to multiple targets')
        case_ids.update(identifiers)
        _strings(entry['reason_codes'], 'Reason codes')
        indexed[target] = entry
    return indexed


def screen_training_target(path, manifest):
    """Hash actual target bytes and return excluded or not_approved, never allowed.

    The caller must establish the manifest's pinned report provenance, as the
    accompanying CLI does. This function has no filename-based allow fallback.
    """
    exclusions = _validate_manifest(manifest)
    target = Path(path)
    before = target.stat()
    with target.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    after = target.stat()
    _require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
             'Target changed while its content was being checked')
    found = exclusions.get(digest)
    return {'target_sha256': digest, 'status': 'excluded' if found else 'not_approved',
            'reason_codes': list(found['reason_codes']) if found else [],
            'training_eligible': False, 'annotation_approved': False, 'split': None}
