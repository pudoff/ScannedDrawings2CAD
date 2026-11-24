"""Deterministic source-fresh OCR holdout; no model or DXF transcription."""
from .family_catalog_v2 import require

POLICY={'source_count':12,'source_order':'ascending input SHA-256, then relative path; first twelve eligible; no replacement',
    'exclude':'all declared train/validation source hashes/groups, all previous OCR test source hashes/groups, including unresolved reviewed slots',
    'fields':['drawing_id','part_name','dimension_1','dimension_2'],
    'dimensions':'two topmost dimension annotation fields in the sheet, leftmost on visually shared row ties; keep illegible/absent slots',
    'physical_lines':'one target per physical reading line/component; do not stitch separate tolerances',
    'literal_convention':'Russian words Cyrillic; U and brands Latin; Latin x for printed numeric cross; ASCII hyphen for minus; preserve punctuation/decimal separator; never infer missing glyphs',
    'pixel_policy':'original RGB pixels only; right-angle rotation recorded; no text/rule cleanup, thresholding or synthesis',
    'transcription':'visual by implementing assistant, no OCR/DXF/filename targets; human_verified=false',
    'native_quarantine':'preserve all pairing/native/geometry exclusions; OCR admission requires its own pixel/label review',
    'new_evaluation_now':False,'old_test_inference_allowed':False,'independent_evaluation':False,'semantic_family_approved':False}


def reviewed_test_sources(labels):
    slots=[s for s in labels.get('new_field_ledger',[]) if s['split']=='test']
    if isinstance(labels.get('previous_labels'),dict):slots+=reviewed_test_sources(labels['previous_labels'])
    return slots


def choose_sources(sources,labels,rows):
    require(len({r['input'] for r in sources})==len(sources),'Duplicate source inventory')
    by={r['input']:r for r in sources};reviewed=reviewed_test_sources(labels)
    reviewed_names={r['source_path'] for r in reviewed}|{r['source_path'] for r in rows if r['split']=='test'}
    for slot in reviewed:
        require(slot['source_path'] in by and by[slot['source_path']]['declared_split']=='test'
            and by[slot['source_path']]['input_sha256']==slot['source_sha256'],'Reviewed source rebound')
    blocked=[r for r in sources if r['declared_split'] in ('train','validation') or r['input'] in reviewed_names]
    hashes={r['input_sha256'] for r in blocked};groups={r['group_id'] for r in blocked}
    audit=[];eligible=[]
    for r in sorted((r for r in sources if r['declared_split']=='test'),key=lambda r:(r['input_sha256'],r['input'])):
        reasons=[]
        if r['input'] in reviewed_names:reasons.append('previous_test_review_or_inference')
        if r['input_sha256'] in hashes:reasons.append('excluded_source_hash')
        if r['group_id'] in groups:reasons.append('excluded_conservative_group')
        if r['source_kind']!='png' or r['prepared_status']!='prepared_original_pixels':reasons.append('missing_original_pixel_preparation')
        if not reasons:eligible.append(r)
        audit.append({'input':r['input'],'input_sha256':r['input_sha256'],'group_id':r['group_id'],
            'visual_status':r['visual_status'],'excluded_native_target':r['excluded_native_target'],'exclusion_reasons':reasons})
    require(len(eligible)>=POLICY['source_count'],'Not enough fresh eligible sources')
    selected=eligible[:POLICY['source_count']];names={r['input'] for r in selected}
    for r in audit:r['disposition']='excluded' if r['exclusion_reasons'] else 'selected' if r['input'] in names else 'eligible_deferred'
    return selected,audit,{'eligible':len(eligible),'previously_reviewed_test_sources':len(reviewed_names),
        'excluded_hashes':sorted(hashes),'excluded_groups':sorted(groups)}
