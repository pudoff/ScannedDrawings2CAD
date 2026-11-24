"""Observed stage history and processing domains, without changing v3 decisions.

A crop is a processing domain, never proof of a drawing view or physical scale.
Each geometry assignment keeps its IDs; no sorted ID/anchor zipping is allowed.
"""
from collections import defaultdict
from copy import deepcopy
import hashlib

from .dataset_layout_v1 import encoded
from .dimension_chain_audit_v1 import audit_dimension_chains
from .notation import parse_notation
from .notation_recovery_v1 import POLICY, project

MAX_REFERENCES=50000
ROLE_KEYS=('measured_candidate_ids','dimension_line_candidate_ids','extension_line_candidate_ids','arrow_candidate_ids')


def trace(source_image, *, geometry, ocr, arrows, associations, domain):
    if set(domain)!={'source','box','context_image'}:
        raise ValueError('Explicit authenticated processing domain required')
    if (not isinstance(domain['box'],list) or len(domain['box'])!=4
            or not all(type(x)is int for x in domain['box'])
            or not 0<=domain['box'][0]<domain['box'][2]
            or not 0<=domain['box'][1]<domain['box'][3]):
        raise ValueError('Invalid native domain box')
    if len(ocr['lines'])>POLICY['maximum_lines']:
        raise ValueError('Trace line budget exceeded')
    if [domain['box'][2]-domain['box'][0],domain['box'][3]-domain['box'][1]]!=geometry['processed_size_px']:
        raise ValueError('Domain extent mismatch')
    # Existing auditor freshly hashes the prepared image and authenticates every
    # producer's source hash/size/version. It also validates geometric incidence.
    audit=audit_dimension_chains(source_image,geometry=geometry,ocr=ocr,arrows=arrows,associations=associations)
    domain_id='domain-'+hashlib.sha256(encoded(domain)).hexdigest()[:20]
    matrix=[[1,0,domain['box'][0]],[0,1,domain['box'][1]],[0,0,1]]
    geometry_ids={p['id'] for p in geometry['candidates']};arrow_ids={p['id'] for p in arrows['candidates']}
    notations={p['source_line_id']:p for p in ocr['notations']}
    by_text=defaultdict(list)
    assignments=defaultdict(list);references=0
    for association in associations['associations']:
        by_text[association['source_line_id']].append(association)
        for choice_id,choice in [('selected',association)]+[(f'alternative-{i:03d}',a) for i,a in enumerate(association['alternatives'])]:
            for key in ROLE_KEYS:
                for identifier in choice.get(key,[]):
                    references+=1
                    if references>MAX_REFERENCES:raise ValueError('Trace reference budget exceeded')
                    allowed=arrow_ids if key=='arrow_candidate_ids' else geometry_ids
                    if identifier not in allowed:raise ValueError('Dangling producer geometry reference')
                    assignments[identifier].append({'association_id':association['id'],'choice_id':choice_id,'role':key})
    rows=[]
    for line in ocr['lines']:
        notation=notations.get(line['id']);aa=by_text[line['id']]
        parsed=notation['parsed'] if notation else parse_notation(line['raw'])
        history=[{'stage':'observed_text','observation_id':line['id'],'raw':line['raw'],
                  'confidence':line['confidence'],'parent_ids':line.get('parent_observation_ids',[]),
                  'hypothesis':deepcopy(line.get('hypothesis'))},
                 {'stage':'notation','status':parsed['status'],'kind':parsed.get('kind'),
                  'notation_id':notation['id'] if notation else None,
                  'notation_status':notation['status'] if notation else None}]
        for a in aa:
            history.append({'stage':'association','id':a['id'],'status':a['status'],
                'reasons':deepcopy(a['reasons']),'diagnostics':deepcopy(a['evidence']),
                'alternative_count':len(a['alternatives']),
                'checks':associations.get('search_checks_by_notation',{}).get(a['notation_id'])})
        if not aa:history.append({'stage':'association','status':'no_record',
                                  'meaning':'producer emitted no association; no search-failure reason inferred'})
        rows.append({'text_id':line['id'],'processing_domain_id':domain_id,'view_id':None,
                     'polygon_native_px':project(line['polygon_px'],matrix),'history':history,
                     'original_observation':deepcopy(line.get('original_observation')),
                     'accepted':False})
    return {'schema_version':'dimension-recovery-trace/1.0','status':'needs_review',
            'processing_domain':{'id':domain_id,**deepcopy(domain),'context_to_native':matrix,
                'meaning':'processing crop only; not a semantic view or accepted dimension domain'},
            'texts':rows,'geometry_assignments':dict(assignments),
            'multiple_assignments':{k:v for k,v in assignments.items() if len(v)>1},
            'chain_audit':audit,'reference_count':references,
            'physical_geometry_approved':False,'accepted_geometry':False,'producer_decisions_changed':False}
