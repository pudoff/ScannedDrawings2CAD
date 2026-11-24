"""Reserve a limited, source-disjoint OCR control scope after complete review.

Reservations are not label release, semantic-family approval or proof of global
independence. Literal identity retrieval and exact-content checks retain all
matches; generic part names and standards are not unique drawing identities.
"""
from collections import Counter
from copy import deepcopy
from pathlib import PurePosixPath
import re
import unicodedata

from .coallocation_policy_v1 import validate_assignments_v1
from .family_catalog_v2 import require, stable_digest

FLAGS={'annotation_approved':False,'training_eligible':False,'independent_evaluation':False,
       'semantic_family_approved':False,'OCR_run':False,'new_OCR_labels':0}
RULES={
    'id':'ocr-control-limited-scope/3.0',
    'purpose':'limited_pilot_title_OCR_control_not_external_acceptance',
    'revision_reason':'V2 six-distinct-group quota unavailable after complete relationship screening',
    'previous_six_group_protocol_modified':False,
    'selection':'all remaining whole groups with every source reviewed as an identified detail drawing and no concrete external identity/content link in the reviewed scope',
    'ordering':'minimum inherited V2 component rank, then coallocation group id',
    'target_group_count':2,'split_by_group_index':['validation','test'],
    'all_sources_and_relatives_reserved':True,
    'annotation_scope':['one main title designation per source','one title part name per source',
                        'one adjacent drawing reference per source if present'],
    'missing_fields':'retain an absent candidate; no invented label',
    'uncertain_fields':'retain uncertainty with no OCR target; do not complete from filenames or sibling drawings',
    'line_candidate_ledger':'every scoped field of every reserved source, including absent and quarantined fields',
    'quality_or_OCR_error_based_source_replacement':False,
    'new_cross_split_link':'block the whole affected group; freeze a new protocol before continuing',
    'release_prerequisites':['all scoped candidates accounted for','at least one accepted line per reserved source',
                             'original pixel and transformation replay','whole-group routing including all 90 inherited train lines',
                             'explicit scoped relationship evidence and current train protection inherited'],
    'evaluation':['raw CER and exact line match','fixed existing normalization separately',
                  'per source and per group counts, coverage and quarantined fields',
                  'one group per split; no estimate of between-family generalization',
                  'validation for choices; test inference only after experiment settings are frozen'],
    'limitations':['Images and relationship readings were inspected before this revised protocol.',
                   'No new OCR predictions, line labels or model tuning preceded this protocol.',
                   'Only the stated image regions, catalog metadata and saved literal readings were reviewed.',
                   'An unseen external relationship remains possible; global semantic independence is unapproved.',
                   'Shared organization, drafting style, generic names and standards remain possible across splits.'],
    'training':False,'independent_evaluation':False,'labels_released':False,
}


def canonical(value):
    """Retrieval only: fold formatting, Latin C/O and ambiguous З/3 glyphs."""
    value=unicodedata.normalize('NFC',value).casefold().translate(str.maketrans({'c':'с','o':'о','з':'3'}))
    return ''.join(ch for ch in value if ch.isalnum())


def identity_keys(value, *, reference=False):
    require(isinstance(value,str) and bool(value.strip()),'Empty literal identity')
    keys={'literal:'+canonical(value)}
    numeric=re.match(r'^(\d{5,})(?=\D|$)',value)
    if numeric:keys.add('numeric-root:'+numeric[1])
    project=re.match(r'^([^\W\d_3]+|[Зз3]{2}П)\.(\d+)\.(\d+)',value,re.I)
    if project:keys.add('project:'+canonical(project[0]))
    if reference:
        suffix=re.search(r'(\d{4})(?:[A-Za-zА-Яа-я]+)?$',value)
        if suffix:keys.add('unresolved-reference-suffix:'+suffix[1])
    return keys


def file_keys(path):
    stem=PurePosixPath(path).stem.strip()
    return identity_keys(stem)|{'unresolved-reference-suffix:'+m[1]
        for m in re.finditer(r'(?<!\d)(\d{4})(?!\d)',stem)}


def prior_literals(row):
    title=row.get('title_designation')
    if title is None:title=row.get('title_observation',{}).get('designation')
    refs=[r if isinstance(r,str) else r['text'] for r in row.get('related_designations',[])]
    return ([title] if title else []),refs


def literal_keys(row):
    titles,refs=prior_literals(row)
    return set().union(*(identity_keys(value,reference=is_ref)
        for values,is_ref in ((titles,False),(refs,True)) for value in values))


def build_scope_v3(catalog,policy,exclusions,overlay,routing,dispositions,queue,probes,readings,prior_rows):
    """Pure scope review; the caller authenticates pinned bytes and source pixels."""
    # The inherited overlay is reproduced by the established routing validator.
    validate_assignments_v1(catalog,policy,exclusions,overlay,[])
    files={r['path']:r for r in catalog['files']};groups={r['group_id']:r for r in overlay['groups']}
    require(len(files)==len(catalog['files']),'Duplicate catalog file')
    require(isinstance(dispositions,list) and len({r['id'] for r in dispositions})==len(dispositions),'Duplicate disposition')
    original={r['id']:r for r in queue['entries']}
    require(len(original)==len(queue['entries']) and {r['id'] for r in dispositions}==set(original),'Incomplete inherited disposition registry')
    for row in dispositions:
        source=original[row['id']]
        require(row['input_path']==source['input_path'] and row['input_sha256']==source['input_sha256']
            and row['coallocation_group_id']==overlay['file_to_group'][row['input_path']],'Disposition source differs')
        require(row['split'] is None and row['annotation_approved'] is False
            and row['training_eligible'] is False and row['independent_evaluation'] is False,'Inherited disposition cannot admit samples')
    pending=[r for r in dispositions if r['status']=='pending_relationship_review']
    require(pending,'No pending scope to review')
    require(probes['source_count']==len(probes['sources'])==len(pending),'Incomplete remaining-source probes')
    expected={r['id']:r for r in pending};probe_by_id={r['id']:r for r in probes['sources']}
    require(len(probe_by_id)==len(pending) and set(probe_by_id)==set(expected),'Probe source inventory differs')
    require(readings['schema_version']=='ocr-control-scope-readings/3.0'
        and readings['reviewer']=='assistant' and readings['human_verified'] is False
        and readings['OCR_run'] is False and readings['training_started'] is False
        and readings['new_OCR_labels']==0 and type(readings['new_OCR_labels']) is int
        and readings['independent_evaluation'] is False,'Invalid scope review claims')
    observed=readings['sources'];require(isinstance(observed,list) and len(observed)==len(pending),'Incomplete source review')
    require([r['index'] for r in observed]==list(range(len(pending))) and all(type(r['index']) is int for r in observed),'Changed review order')
    require([r['id'] for r in observed]==[r['id'] for r in pending],'Changed pending source order')
    required={'index','id','input_path','input_sha256','group_id','title_designation','title_name','related_designations',
              'document_kind','decision','visual_checks','viewed_images','overview_contact','uncertainties','notes'}
    for row in observed:
        require(set(row)==required,'Unexpected review fields')
        old=expected[row['id']];probe=probe_by_id[row['id']]
        require(row['input_path']==old['input_path']==probe['input_path']
            and row['input_sha256']==old['input_sha256']==probe['input_sha256']
            and row['group_id']==old['coallocation_group_id']==probe['group_id'],'Review source binding differs')
        require(row['input_path'] in files and files[row['input_path']]['sha256']==row['input_sha256']
            and files[row['input_path']]['kind'] in ('image','pdf')
            and overlay['file_to_group'][row['input_path']]==row['group_id'],'Reviewed source differs from catalog')
        require(row['visual_checks']=={'overview_seen':True,'left_seen':True,'right_seen':True}
            and all(v is True for v in row['visual_checks'].values()),'Incomplete visual review')
        require(row['viewed_images']==[probe['overview'],*[r['image'] for r in probe['regions']]]
            and row['overview_contact']==probes['contacts'][row['index']//5],'Changed viewed evidence')
        require(row['decision'] in ('no_known_cross_split_link_in_reviewed_scope','needs_further_review','known_train_link'),'Unknown review decision')
        require(row['document_kind'] in ('detail_drawing','reference_diagram','unidentified_detail_fragment'),'Unknown document kind')
        require(all(row[key] is None or isinstance(row[key],str) and row[key].strip() for key in ('title_designation','title_name')),'Invalid title observation')
        for key in ('related_designations','notes','uncertainties'):
            require(isinstance(row[key],list) and all(isinstance(v,str) and v.strip() for v in row[key]),'Invalid reading list')
        require(len(row['related_designations'])==len(set(row['related_designations'])),'Repeated literal reference')
        require(bool(row['notes']),'Missing review rationale')
    require(isinstance(prior_rows,list) and len({r['input_path'] for r in prior_rows})==len(prior_rows),'Duplicate prior source reading')
    for row in prior_rows:
        require(row['input_path'] in files and files[row['input_path']]['sha256']==row['input_sha256'],'Prior literal source differs')
    core=[{k:r[k] for k in ('id','source_path','source_sha256','image_sha256','split')} for r in routing['assignments']]
    inherited=validate_assignments_v1(catalog,policy,exclusions,overlay,core)
    require(stable_digest(inherited['assignments'])==stable_digest(routing['assignments'])
        and all(r['split']=='train' for r in core),'Inherited assignments changed')
    train_ids={r['coallocation_group_id'] for r in inherited['assignments']}
    train_members={p for gid in train_ids for p in groups[gid]['members']}
    require({p for c in routing['train_commitments'] for p in c['member_paths']}==train_members
        and {c['coallocation_group_id'] for c in routing['train_commitments']}==train_ids,'Train protections differ')
    require(len(routing['train_commitments'])==len(train_ids) and all(c['split']=='train'
        and c['member_paths']==groups[c['coallocation_group_id']]['members']
        for c in routing['train_commitments']),'Changed whole-group train commitment')
    rows_by_group={gid:[r for r in observed if r['group_id']==gid] for gid in sorted({r['group_id'] for r in observed})}
    keys_by_prior={r['input_path']:literal_keys(r) for r in prior_rows}
    file_identity={p:file_keys(p) for p in files}
    audits=[];reviews=[]
    for gid,rows in rows_by_group.items():
        group=groups[gid];members=set(group['members']);paths={r['input_path'] for r in rows}
        full_sources={p for p in members if files[p]['kind'] in ('image','pdf')}
        known_keys=set().union(*(literal_keys(r) for r in rows))
        name_matches=sorted(p for p,k in file_identity.items() if known_keys&k)
        literal_matches=sorted(p for p,k in keys_by_prior.items() if known_keys&k)
        hashes={files[p]['sha256'] for p in members}
        content_matches=sorted(p for p in files if files[p]['sha256'] in hashes)
        external={p for p in name_matches+literal_matches+content_matches if p not in members}
        missing=sorted(full_sources-paths)
        reasons=[]
        if missing:reasons.append('unreviewed_source_relative')
        if external:reasons.append('concrete_external_identity_or_content_link')
        if gid in train_ids or members&train_members:reasons.append('inherited_train_protection')
        if not {'validation','test'}<=set(group['routing_allowed_splits']):reasons.append('historical_or_routing_restriction')
        if any(r['decision']!='no_known_cross_split_link_in_reviewed_scope' for r in rows):reasons.append('unresolved_review_decision')
        if any(r['document_kind']!='detail_drawing' or not r['title_designation'] for r in rows):reasons.append('unidentified_document_provenance')
        # Generic names are reported, never interpreted as unique drawing IDs.
        names={canonical(r['title_name']) for r in rows if r['title_name']}
        generic=sorted(r['input_path'] for r in prior_rows if canonical(r.get('title_name') or r.get('title_observation',{}).get('name') or '') in names and r['input_path'] not in members)
        audits.append({'group_id':gid,'catalog_filename_identity_matches':name_matches,
            'saved_literal_identity_matches':literal_matches,'whole_group_exact_content_matches':content_matches,
            'external_identity_or_content_matches':sorted(external),'unreviewed_source_relatives':missing,
            'generic_title_name_matches_outside_group':generic,'generic_name_match_is_family_proof':False})
        ranks=[r['rank'] for r in queue['entries'] if r['input_path'] in paths]
        require(ranks and all(type(r) is int for r in ranks),'Missing inherited component ranks')
        reviews.append({'group_id':gid,'source_ids':[r['id'] for r in rows],'source_paths':sorted(paths),
            'member_paths':sorted(members),'first_inherited_rank':min(ranks),'blocking_reasons':reasons,
            'scoped_relationship_review_passed':not reasons,'retained_unresolved_hold_ids':group['unresolved_hold_ids'],
            'independent_evaluation':False,'semantic_family_approved':False})
    eligible=sorted((r for r in reviews if not r['blocking_reasons']),key=lambda r:(r['first_inherited_rank'],r['group_id']))
    ready=len(eligible)==RULES['target_group_count'];reservations=[]
    if ready:
        for index,group in enumerate(eligible):
            reservations.append({**deepcopy(group),'reserved_split':RULES['split_by_group_index'][index],
                'status':'reserved_for_future_scoped_control_pending_line_annotation',**FLAGS})
    reserved_split={r['group_id']:r['reserved_split'] for r in reservations}
    prospective=[{'id':'scope-reservation-'+stable_digest(path)[:24],'source_path':path,
        'source_sha256':files[path]['sha256'],'split':group['reserved_split']}
        for group in reservations for path in group['member_paths']]
    combined=validate_assignments_v1(catalog,policy,exclusions,overlay,core+prospective)
    revised=[]
    for old in dispositions:
        row=deepcopy(old)
        if row['id'] in expected:
            row['status']='reserved_pending_line_annotation' if row['coallocation_group_id'] in reserved_split else 'pending_relationship_review'
            row['reserved_split']=reserved_split.get(row['coallocation_group_id'])
            row['reason']=('Scoped review and whole-group reservation only; OCR targets and release remain pending.'
                if row['reserved_split'] else 'Relationship review remains pending; no control reservation.')
        revised.append(row)
    protocol={'schema_version':'ocr-control-protocol-draft/3.0','task':'P14.1b.7',
        'rules':deepcopy(RULES),'ready_to_freeze':ready,'reservations':reservations,
        'train_commitments':deepcopy(routing['train_commitments']),
        'inherited_train_sample_ids':[r['id'] for r in inherited['assignments']],
        'release_created':False,**FLAGS}
    report={'schema_version':'ocr-control-scope-report/3.0','reviewed_source_count':len(observed),
        'reviewed_group_count':len(reviews),'catalog_file_count':len(files),'prior_literal_source_count':len(prior_rows),
        'whole_relative_file_count':sum(len(r['member_paths']) for r in reviews),
        'scoped_candidate_group_count':len(eligible),'reserved_group_count':len(reservations),
        'reserved_source_count':sum(len(r['source_ids']) for r in reservations),
        'reserved_relative_file_count':len(prospective),'remaining_pending_source_count':sum(r['status']=='pending_relationship_review' for r in revised),
        'inherited_train_sample_count':len(core),'protected_train_file_count':len(train_members),
        'external_identity_or_content_match_count':sum(len(r['external_identity_or_content_matches']) for r in audits),
        'disposition_counts':dict(Counter(r['status'] for r in revised)),
        'limited_protocol_ready':ready,'six_group_quota_available':False,'control_release_created':False,
        'actual_bytes_authenticated':False,'global_split_policy_modified':False,**FLAGS}
    return {'report':report,'metadata-audit':audits,'group-reviews':reviews,'reservations':reservations,
            'reservation-routing':combined,'dispositions':revised,'protocol-draft':protocol}
