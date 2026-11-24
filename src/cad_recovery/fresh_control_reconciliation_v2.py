"""Field-level reconciliation of five known inventory/selection metadata files."""
from .fresh_control_metadata_v1 import subjects

REGISTRIES={'artifacts/role-admission-labels-v1/groups.json',
            'artifacts/role-admission-labels-v1-final/groups.json'}
INTAKE='artifacts/cv-stage2-1-8-intake-v1.json'
OCR_AUDIT='artifacts/ocr-fresh-holdout-v1/source-audit.json'
OCR_SELECTION='configs/reviews/ocr-fresh-holdout-selection-v1.json'
PATHS=REGISTRIES|{INTAKE,OCR_AUDIT,OCR_SELECTION}


def reconcile(path,data,index):
    """Return actual assignment/exclusion subjects, not every item in a pool ledger."""
    all_ids=subjects(data,index)
    if path in REGISTRIES:
        if (data['schema_version']!='role-admission-groups/1.0' or data['declared_splits_changed']
                or data['independent_evaluation'] or 'labels' in data or 'predictions' in data):
            raise ValueError('Group registry is not the expected metadata-only schema')
        return set(),all_ids,'group_constraints_only_no_source_run_or_label_assignment'
    if path==INTAKE:
        if (data['opened_reserved_sources'] or data['independent_control_released']
                or not isinstance(data['read_only_index'],dict)
                or any(r['scope']!='catalog_only_candidate_for_pair_family_and_ground_truth_review' for r in data['pairs'])):
            raise ValueError('Intake no longer declares a read-only catalogue')
        return set(),all_ids,'explicit_read_only_intake_no_control_release'
    if path==OCR_AUDIT:
        if any(r['disposition'] not in ('selected','excluded','eligible_deferred') for r in data):
            raise ValueError('Unknown OCR pool disposition')
        used=subjects([r for r in data if r['disposition'] in ('selected','excluded')],index)
        return used,all_ids-used,'selected_and_previously_excluded_held_out; eligible_deferred_is_pool_metadata'
    if path==OCR_SELECTION:
        if data['schema_version']!='ocr-fresh-holdout-selection/1.0' or data['OCR_run'] or data['DXF_read_for_transcription']:
            raise ValueError('Unexpected selection recipe')
        used=subjects(data['selected_sources'],index)
        return used,all_ids-used,'only_selected_sources_reserved; source_audit_and_exclusion_ledger_are_metadata'
    raise ValueError('No reviewed field-level reconciliation for this file')
