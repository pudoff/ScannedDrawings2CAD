"""Metadata-only freshness screening; absence of a collision never approves ancestry."""
from collections import defaultdict
import re
from .role_interval_dataset_v1 import canonical

IDENTITY_KEYS={'source','input','path','input_path','source_path','target','target_path',
    'sha256','source_sha256','input_sha256','target_sha256','file','native_dxf',
    'sources','inputs','source_paths','paths','members','target_paths'}
SKIP_VALUES={'points','points_px','vertices','coordinates','features','probabilities','confusion',
    'text','transcript','literal','reading_text','pixels','tensor','weights'}
METADATA_PREFIXES=('dataset','current-prepared','clear-raster-prepared','family-split',
    'source-coallocation','geometry-lineage','role-family-survey','resource-disposition',
    'pair-','ocr-current-release','ocr-control-release','cv-stage2-family','coallocation-',
    'architecture-data-readiness')
SKIP_DIRS={'deps','code','workspace','.cache','tiles','rasters','rgb-prepared','rgba-prepared',
    'site-packages','node_modules','.git','__pycache__','documents'}


def key_path(value):
    value=value.replace('\\','/').strip()
    if '/dataset/' in value.casefold():value=value[value.casefold().rfind('/dataset/')+1:]
    return canonical(value)


def geometry_groups(report):
    signatures=defaultdict(set)
    for r in report['records']:
        if r['status']!='complete_supported_geometry' or r.get('problems') or r.get('geometry_entities',0)<=0:continue
        if not re.fullmatch('[0-9a-f]{64}',r.get('signature','')):continue
        signatures[r['signature']].update(r['paths'])
        signatures[r['signature']].update(p['input'] for p in r.get('pairs',[]))
    return [dict(id='native-signature-'+s[:20],members=sorted(paths),
        scope='conservative_exclusion_only_not_semantic_family_approval')
        for s,paths in sorted(signatures.items()) if len(paths)>1]


def working_groups(cases):
    groups=defaultdict(set)
    for case in cases:
        groups[case['group']].add(case['source']['path'])
        target=case.get('native_DXF_for_scoring_only')
        if target:groups[case['group']].add(target['path'])
    return [dict(id='previous-working-'+name,members=sorted(paths),
        scope='prior_development_group_not_new_independence') for name,paths in sorted(groups.items())]


def identity_index(sources):
    paths=defaultdict(set);hashes=defaultdict(set)
    for row in sources:
        for value in (row['input'],row['target']):paths[key_path(value)].add(row['id'])
        for value in (row['input_sha256'],row['target_sha256']):hashes[value].add(row['id'])
    return paths,hashes


def subjects(data,index):
    """Extract identity fields only; text/geometry values never become selection features."""
    paths,hashes=index;found=set()
    def walk(value):
        if isinstance(value,dict):
            for k,v in value.items():
                if k.casefold() in SKIP_VALUES:continue
                if k.casefold() in IDENTITY_KEYS and isinstance(v,str):
                    found.update(hashes.get(v.lower(),()));found.update(paths.get(key_path(v),()))
                elif k.casefold() in IDENTITY_KEYS and isinstance(v,list):
                    for item in v:
                        if isinstance(item,str):
                            found.update(hashes.get(item.lower(),()));found.update(paths.get(key_path(item),()))
                        elif isinstance(item,(dict,list)):walk(item)
                elif isinstance(v,(dict,list)):walk(v)
        elif isinstance(value,list):
            for v in value:
                if isinstance(v,(dict,list)):walk(v)
    walk(data)
    return found


def history_kind(relative):
    parts=relative.replace('\\','/').split('/');name=parts[-1].casefold()
    root=parts[1].casefold() if parts[0]=='artifacts' and len(parts)>1 else ''
    if root.startswith(METADATA_PREFIXES):return 'inventory_or_lineage_metadata_only'
    if name in ('execution.json','prediction.json','result.json','status.json','request.json'):
        return 'previous_execution_or_assigned_inference'
    if any(s in name for s in ('reading','labels','record','train.jsonl','validation.jsonl','test.jsonl')):
        return 'previous_review_or_supervision_record'
    if any(s in name for s in ('selection','requests','controls','protocol','cases','batch','progress')):
        return 'previous_assignment_or_reservation'
    return 'unresolved_identity_reference'


def disposition(row,decision,history,catalog):
    reasons=list(decision['reasons'])
    if row['declared_split']!='test':reasons.append('not_original_test')
    if decision['development_related_family']:reasons.append('previous_development_family')
    if catalog.get('visual_status')=='verified':reasons.append('previous_verified_pair_content_review')
    positive=[r for r in history if r['kind']!='inventory_or_lineage_metadata_only']
    if positive:reasons.append('prior_use_or_identity_reference_requires_review')
    return dict(id=row['id'],source=dict(path='dataset/'+row['input'],sha256=row['input_sha256']),
        target=dict(path='dataset/'+row['target'],sha256=row['target_sha256']),
        declared_split=row['declared_split'],canonical_group=decision['canonical_family'],
        exclusions=sorted(set(reasons)),metadata_clear=not reasons,
        history=positive,inventory_reference_count=len(history)-len(positive),
        pair_content_approved=catalog.get('pair_content_approved',False),
        family_provenance_confirmed=False,prior_external_variants_confirmed_unused=False,
        independent_admission_approved=False,pixels_may_be_opened=False)


def choose(rows,count=6):
    """Fixed metadata order, different provisional roots; never uses recognition scores."""
    selected=[];groups=set()
    for row in sorted(rows,key=lambda r:(r['source']['sha256'],r['source']['path'])):
        if not row['metadata_clear'] or row['canonical_group'] in groups:continue
        groups.add(row['canonical_group']);selected.append(row)
        if len(selected)==count:break
    return selected


def admission(rows,confirmations,*,candidate_component_passed):
    """Future confirmations are source-bound; missing facts remain missing."""
    family=set();eligible=[];missing={}
    by={r['source']['sha256']:r for r in confirmations}
    if len(by)!=len(confirmations):raise ValueError('Duplicate confirmation source')
    for row in rows:
        c=by.get(row['source']['sha256']);problems=[]
        if not row['metadata_clear']:problems.append('metadata_history_exclusion')
        if c is None:problems.append('source_provenance_and_family_not_confirmed')
        else:
            if c['source']!=row['source']:problems.append('confirmation_identity_mismatch')
            if (c.get('authority') not in ('user','customer') or not c.get('evidence')
                    or not c.get('family_id') or not c.get('drawing_designation')
                    or c.get('revision') in (None,'unknown','') or not c.get('origin_description')):
                problems.append('incomplete_source_provenance')
            if c.get('train_validation_development_or_prior_variants_used') is not False:
                problems.append('external_history_not_confirmed_unused')
        if problems:missing[row['id']]=problems
        else:eligible.append(row['id']);family.add(c['family_id'])
    approved=len(eligible)>=6 and len(family)>=3
    return dict(metadata_approved_sources=len(eligible),confirmed_families=len(family),
        required_sources=6,required_families=3,missing=missing,
        source_provenance_admission_approved=approved,
        candidate_component_passed=bool(candidate_component_passed),
        control_content_may_be_opened=approved and candidate_component_passed,
        independent_quality_acceptance=False)
