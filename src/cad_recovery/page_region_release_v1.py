"""Scoped partial page-region labels with authenticated native evidence.

These are positive context boxes. Unlabelled page space never becomes negative
background, and no detector/dense/physical training admission is implied.
"""
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from .dataset_layout_v1 import checked, encoded, pin, read
from .family_catalog_v2 import require
from .page_region_review_v1 import (INPUTS, POLICY, READING, context,
    authenticated_source, native, reading, rectangle, translations, tiles)

ACCEPTANCE = 'configs/reviews/page-region-acceptance-v1.json'
USES = {'train':['positive_roi_classification'],
        'validation':['diagnostic_roi_classification','diagnostic_positive_box_localization']}


def verified_review(root, *, pixels=True):
    """Screen every source before any native image is opened; replay evidence."""
    require(type(pixels) is bool, 'Explicit boolean replay mode required')
    _, sources, excluded = context(root)
    require(not excluded and read(checked(root, INPUTS))['sources'] == sources, 'Screened selection changed')
    decisions = reading(root, sources)
    accepted = read(checked(root, pin(root, ACCEPTANCE)))
    require(accepted['reviewer'] == POLICY['reviewer'] and accepted['human_verified'] is False
            and accepted['accepted_scope'] == 'positive_semantic_context_boxes', 'Review authority changed')
    for name in ('complete_ocr_context','pixel_mask_authority','full_page_exhaustive',
                 'frame_approved','physical_geometry_approved','production_admitted'):
        require(accepted[name] is False, 'Unreviewed authority escalation')
    evidence = read(checked(root, accepted['evidence']))
    require(evidence['selection'] == INPUTS and evidence['reading'] == pin(root,READING)
            and evidence['model_calls'] == 0, 'Evidence lineage changed')
    rows = evidence['regions']; ids = [r['id'] for r in rows]
    require(ids == accepted['reviewed_region_ids'] and len(set(ids)) == len(ids)
            and len(rows) == len(decisions['regions']), 'Incomplete visual review')
    require(set(accepted['observation_overrides']) <= set(ids), 'Unknown observation override')
    by_id = {s['id']:s for s in sources}
    for source in sources:
        authenticated_source(root, source)
    tile_count = 0; pixel_count = 0
    for row, decision in zip(rows,decisions['regions']):
        require(all(row[k] == v for k,v in decision.items()), 'Region changed after visual review')
        source = by_id[row['source_id']]; box = row['box']
        require(row['polygon'] == rectangle(box,source['size_px']) and row['transform'] == translations(box), 'Invalid region coordinates')
        expected = tiles(box)
        require([t['box'] for t in row['tiles']] == expected, 'Gap, overlap or reordered native coverage')
        for tile in row['tiles']:
            require(tile['transform'] == translations(tile['box']), 'Invalid tile transform')
            path = checked(root,tile['asset']); tile_count += 1
            x,y,r,b = tile['box']; pixel_count += (r-x)*(b-y)
            if pixels:
                original, proof = native(root,source,tile['box'])
                with Image.open(path) as im:
                    require(im.mode == 'RGB' and im.size == (r-x,b-y), 'Changed native RGB dimensions')
                    require(np.array_equal(np.asarray(im),original), 'Native evidence pixels differ')
                require(tile['native_proof'] == proof, 'Changed native evidence provenance')
        checked(root,row['preview'])
    witnesses = read(checked(root,accepted['witnesses']))
    require([w['id'] for w in witnesses] == accepted['observed_native_witness_ids'], 'Unreviewed native witness')
    for witness in witnesses:
        source = by_id[witness['source_id']]; rectangle(witness['box'],source['size_px'])
        path = checked(root,witness['image'])
        if pixels:
            original, proof = native(root,source,witness['box'])
            with Image.open(path) as im:
                require(im.mode == 'RGB' and np.array_equal(np.asarray(im),original), 'Native witness pixels differ')
            require(witness['native_proof'] == proof, 'Native witness provenance differs')
    return sources, decisions, accepted, rows, {'native_tiles':tile_count,'native_pixels':pixel_count,
                                               'native_witnesses':len(witnesses),'pixels_replayed':pixels}


def payload(sources, decisions, accepted, reviewed):
    by_id = {s['id']:s for s in sources}; records = []
    for row in reviewed:
        source = by_id[row['source_id']]
        records.append({'id':row['id'], 'source_id':source['id'], 'source':source['source'],
                        'split':source['split'], 'group_id':source['group_id'],
                        'source_size_px':source['size_px'], 'role':row['role'],
                        'bbox':row['box'], 'polygon':row['polygon'], 'transform':row['transform'],
                        'proposal_id':source['id']+'-region-'+str(row['proposal']),
                        'reviewer':POLICY['reviewer'], 'human_verified':False,
                        'observation':accepted['observation_overrides'].get(row['id'],row['observation']),
                        'semantics':'context_envelope_with_dimensions_and_possible_neighbouring_ink',
                        'dense_mask':False, 'complete_ocr_context':False,
                        'physical_geometry_approved':False, 'evidence':accepted['evidence']})
    dispositions = []
    for s in sources:
        for i, p in enumerate(s['proposals'],1):
            found = [r['id'] for r in records if r['proposal_id'] == p['id']]
            unknown = [r for r in decisions['unknown_proposals'] if r['source_id'] == s['id'] and r['proposal'] == i]
            dispositions.append({'proposal_id':p['id'], 'original_box':p['box'], 'original_role':p['role'],
                                 'status':'accepted_after_review' if found else 'unknown_outside_taxonomy',
                                 'released_regions':found, 'reason':None if found else unknown[0]['reason']})
    summary = {'regions':len(records), 'sources':len(sources), 'groups':len({s['group_id'] for s in sources}),
               'by_role':dict(Counter(r['role'] for r in records)),
               'by_split':{split:{'sources':sum(s['split']==split for s in sources),
                    'groups':len({s['group_id'] for s in sources if s['split']==split}),
                    'regions':sum(r['split']==split for r in records),
                    'by_role':dict(Counter(r['role'] for r in records if r['split']==split))}
                    for split in ('train','validation')}, 'original_proposals':len(dispositions),
               'unknown_proposals':len(decisions['unknown_proposals']), 'disputed_regions':0}
    admission = {'schema_version':'page-region-admission/1.0','reviewer':POLICY['reviewer'],
        'human_verified':False,'scope':'positive_semantic_context_boxes',
        'allowed_uses':{split:list(purposes) for split,purposes in USES.items()},
        'forbidden_uses':['full_page_detector_training','dense_pixel_training','negative_background_sampling',
                          'independent_evaluation','physical_targets','production'],
        'unlabelled_policy':'unknown_never_negative','full_page_exhaustive':False,
        'frame_admitted':False,'table_validation_admitted':False,'independent_families_confirmed':False,
        'physical_geometry_approved':False,'production_admitted':False,
        'evaluation_limits':'Known positives only: IoU/recall on these context envelopes; no full-page precision/AP, no OCR quality claim.',
        'class_gaps':decisions['class_gaps']}
    return {'records.json':records,'sources.json':sources,'dispositions.json':dispositions,
            'summary.json':summary,'admission.json':admission}


def build(root, folder):
    require(not folder.exists() and folder.parent == root/'artifacts', 'Choose new label release directory')
    sources, decisions, accepted, rows, replay = verified_review(root)
    values = payload(sources, decisions, accepted, rows)
    folder.mkdir()
    for name,value in values.items():
        (folder/name).write_bytes(encoded(value))
    manifest = {'schema_version':'page-region-label-release/1.0','policy':POLICY,
                'selection':INPUTS,'reading':pin(root,READING),'acceptance':pin(root,ACCEPTANCE),
                'evidence':accepted['evidence'],'witnesses':accepted['witnesses'],
                'replay':replay,'model_calls':0,'closed_test_used':False,
                'files':{name:pin(root,(folder/name).relative_to(root).as_posix()) for name in values}}
    (folder/'manifest.json').write_bytes(encoded(manifest))
    return pin(root,(folder/'manifest.json').relative_to(root).as_posix())


def load(root, release, *, split, purpose, replay_pixels=False):
    """Fail closed on data/admission edits; callers explicitly select one scope."""
    require(split in ('train','validation'), 'Closed or unknown split')
    require(purpose in USES[split], 'Use outside scoped admission')
    manifest = read(checked(root,release))
    require(manifest['schema_version'] == 'page-region-label-release/1.0'
            and manifest['policy'] == POLICY and manifest['selection'] == INPUTS
            and manifest['reading'] == pin(root,READING) and manifest['acceptance'] == pin(root,ACCEPTANCE)
            and manifest['closed_test_used'] is False and manifest['model_calls'] == 0, 'Release provenance changed')
    sources, decisions, accepted, reviewed, _ = verified_review(root,pixels=replay_pixels)
    require(manifest['evidence'] == accepted['evidence'] and manifest['witnesses'] == accepted['witnesses'], 'Release evidence changed')
    expected = payload(sources,decisions,accepted,reviewed)
    require(set(manifest['files']) == set(expected), 'Incomplete release')
    values = {name:read(checked(root,p)) for name,p in manifest['files'].items()}
    require(values == expected, 'Release differs from reviewed scope or labels')
    require(purpose in values['admission.json']['allowed_uses'][split], 'Use outside scoped admission')
    return [r for r in values['records.json'] if r['split'] == split]
