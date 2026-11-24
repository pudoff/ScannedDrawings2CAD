"""Compare explicitly reviewed variant facts without approving training labels.

This module neither reads filenames nor infers facts from visual similarity.
The caller must establish source-pixel and native-CAD provenance separately.
Numbers are normalized authored readings, not physical CAD measurements.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
import re


METHOD = 'explicit-reviewed-variant-facts-v1'
REQUIRED_FIELDS = ('designation', 'displayed_length', 'displayed_length_tolerance',
                   'mass_g', 'shaft_break')
OPTIONAL_FIELDS = ('strength_class',)
FACT_FIELDS = frozenset((*REQUIRED_FIELDS, *OPTIONAL_FIELDS))
MAX_CANDIDATES = 256


def _number(value, field):
    try:
        finite = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f'{field} must be a finite number, not a boolean')
    if value < 0 or (field != 'displayed_length_tolerance' and value == 0):
        raise ValueError(f'{field} must be positive (tolerance may be zero)')


def _facts(value):
    if not isinstance(value, dict) or not set(value) <= FACT_FIELDS:
        raise ValueError('Facts must be a map of explicitly supported variant fields')
    for field, reading in value.items():
        if field in ('displayed_length', 'displayed_length_tolerance', 'mass_g'):
            _number(reading, field)
        elif field == 'shaft_break':
            if type(reading) is not bool:
                raise ValueError('shaft_break must be a boolean')
        elif field == 'designation':
            if (not isinstance(reading, str) or len(reading) > 128
                    or re.fullmatch(r'M[1-9]\d*-[1-9]\d*', reading, flags=re.ASCII) is None):
                raise ValueError('designation must be normalized as Mdigits-digits')
        elif field == 'strength_class':
            if (not isinstance(reading, str) or len(reading) > 128
                    or re.fullmatch(r'[1-9]\d*\.\d', reading, flags=re.ASCII) is None):
                raise ValueError('strength_class must be a normalized digit.digit string')
    return deepcopy(value)


def _native_conflicts(value):
    if not isinstance(value, list) or len(value) > 256:
        raise ValueError('native_conflicts must be a bounded list')
    for item in value:
        if not ((isinstance(item, str) and item.strip()) or (isinstance(item, dict) and item)):
            raise ValueError('Native conflicts must contain nonempty strings or JSON objects')
        try:
            encoded = json.dumps(item, ensure_ascii=True, allow_nan=False)
        except (ValueError, TypeError, OverflowError) as error:
            raise ValueError('Native conflicts must be finite JSON data') from error
        if len(encoded) > 16384:
            raise ValueError('Native conflict exceeds its size limit')
        # JSON objects with non-string keys silently change identity on writing.
        if isinstance(item, dict) and json.loads(encoded) != item:
            raise ValueError('Native conflicts must round-trip as JSON objects')
    return deepcopy(value)


def _unapproved():
    return {'training_eligible': False, 'annotation_approved': False, 'split': None}


def evaluate_variant_candidates(source_facts: dict, candidates: list[dict]) -> dict:
    """Return contradictions, unknowns and a decision scoped to this candidate set.

    An absent key is unknown evidence. Every known contradiction rejects that
    candidate even if other fields agree. Missing required facts, incomplete
    references or native conflicts prevent support. No numeric tolerance or
    similarity weighting is inferred, and no result approves physical scale.
    """
    source = _facts(source_facts)
    if not isinstance(candidates, list) or not 2 <= len(candidates) <= MAX_CANDIDATES:
        raise ValueError('Variant comparison requires 2..256 explicit candidates')
    prepared, identifiers = [], set()
    for candidate in candidates:
        if not isinstance(candidate, dict) or set(candidate) != {'id', 'facts', 'reference_complete', 'native_conflicts'}:
            raise ValueError('Candidate must contain only id, facts, reference_complete and native_conflicts')
        identifier = candidate['id']
        if (not isinstance(identifier, str) or not identifier.strip() or len(identifier) > 256
                or any(ord(character) < 32 for character in identifier)):
            raise ValueError('Candidate id must be a nonempty bounded string')
        if identifier in identifiers:
            raise ValueError('Candidate ids must be unique')
        identifiers.add(identifier)
        complete = candidate['reference_complete']
        if type(complete) is not bool:
            raise ValueError('reference_complete must be a boolean')
        prepared.append((identifier, _facts(candidate['facts']), complete,
                         _native_conflicts(candidate['native_conflicts'])))
    results = []
    for identifier, facts, complete, native_conflicts in prepared:
        matching, contradictions = [], []
        for field in (*REQUIRED_FIELDS, *OPTIONAL_FIELDS):
            if field not in source or field not in facts:
                continue
            if source[field] == facts[field]:
                matching.append(field)
            else:
                contradictions.append({'field': field, 'source_value': source[field],
                                       'candidate_value': facts[field]})
        missing = [field for field in REQUIRED_FIELDS if field not in source or field not in facts]
        reasons = []
        if contradictions:
            reasons.append('source_fact_contradiction')
        if missing:
            reasons.append('missing_required_evidence')
        if not complete:
            reasons.append('incomplete_reference')
        if native_conflicts:
            reasons.append('native_conflict_requires_review')
        status = ('candidate_rejected_for_reviewed_variant' if contradictions else
                  'unresolved' if reasons else 'supported_candidate_for_object_review')
        results.append({'id': identifier, 'status': status, 'facts': facts,
                        'matching_fields': matching, 'contradictions': contradictions,
                        'missing_required_fields': missing,
                        'missing_source_fields': [field for field in REQUIRED_FIELDS if field not in source],
                        'missing_candidate_fields': [field for field in REQUIRED_FIELDS if field not in facts],
                        'uncompared_optional_fields': [field for field in OPTIONAL_FIELDS
                                                      if field not in source or field not in facts],
                        'reference_complete': complete, 'native_conflicts': native_conflicts,
                        'reasons': reasons, **_unapproved()})
    supported = [row['id'] for row in results if row['status'] == 'supported_candidate_for_object_review']
    selected = None
    if len(supported) > 1:
        status = 'ambiguous'
    elif len(supported) == 1 and all(row['id'] == supported[0] or
            row['status'] == 'candidate_rejected_for_reviewed_variant' for row in results):
        status, selected = 'distinguished_within_reviewed_candidates', supported[0]
    else:
        status = 'unresolved'
    return {'schema_version': '1.0', 'method': METHOD, 'status': status,
            'selected_candidate_id': selected, 'supported_candidate_ids': supported,
            'source_facts': source, 'required_fields': list(REQUIRED_FIELDS),
            'optional_fields': list(OPTIONAL_FIELDS), 'candidates': results,
            'candidate_set_scope': 'Only the explicitly supplied reviewed candidates; not an exhaustive search.',
            'numeric_comparison': 'Exact equality of normalized authored readings; no fitted tolerance.',
            'physical_scale_confirmed': False, **_unapproved()}
