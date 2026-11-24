"""Preserve unresolved ancestry flags while clearing catalogue-only references."""
from .fresh_control_metadata_v1 import subjects
from .fresh_control_reconciliation_v2 import PATHS,INTAKE,reconcile as previous


def reconcile(path,data,index):
    if path!=INTAKE:return previous(path,data,index)
    allowed_scopes={'catalog_only_candidate_for_pair_family_and_ground_truth_review','resolution_pending'}
    allowed_flags={'visual_pair_needs_resolution','related_split_requires_review','preserve_existing_ocr_review_or_reservation'}
    if (data['opened_reserved_sources'] or data['independent_control_released']
            or not isinstance(data['read_only_index'],dict)
            or any(r['scope'] not in allowed_scopes or set(r['review_flags'])-allowed_flags for r in data['pairs'])):
        raise ValueError('Unreviewed intake scope or flag')
    all_ids=subjects(data,index)
    held=[r for r in data['pairs'] if set(r['review_flags'])&
        {'related_split_requires_review','preserve_existing_ocr_review_or_reservation'}]
    used=subjects(held,index)
    return used,all_ids-used,'inventory_only_not_usage; explicit_related_split_and_prior_OCR_reservations_remain_unresolved'
