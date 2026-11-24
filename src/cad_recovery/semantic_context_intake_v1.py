"""Finite metadata-only intake and an explicit source-review contract.

This is a protocol, not images, annotations, a trained model or permission to fit.
"""
from collections import Counter, defaultdict
import math
import re
from pathlib import PurePosixPath

REGIONS = ('view','detail','table','title_block','notes')
ROLES = {
    1: ('PART','Visible part boundary; thickness alone is not evidence. Split geometric LINE/ARC at true primitive changes, not auxiliary crossings.'),
    2: ('DIMENSION','Dimension stroke/arc supported by visible arrow or dimension context; text transcription may remain unknown; not a native CAD DIMENSION.'),
    3: ('EXTENSION','Projection from geometry reference towards dimension; do not relabel contour or axis by proximity to an arrow.'),
    4: ('LEADER','Annotation pointer supported by note/callout context. Separate geometric pieces keep one leader parent, not one composite LINE.'),
    5: ('AXIS','One observed axis with its visible dash runs and gaps; individual dashes are not independent axes.'),
    6: ('TEXT_INK','Text ink only; grid remains grid. One logical text-line parent, not independent glyph atoms. No claim of correct OCR.'),
    7: ('FRAME','Outer sheet border, distinguished from title/table grid by full-page context. Shared border/grid pixels remain multi-role.'),
    8: ('TITLE_GRID','Title-block or table ruling, excluding text. Record parent region. Shared FRAME/TITLE_GRID is not forced into one class.'),
    9: ('HATCH','Section hatch strokes with a hatch-region parent, not any parallel lines. Parent/source weighting must retain correlation.'),
}
POLICY = dict(maximum_sources=8, maximum_sources_per_group=2, maximum_contexts=32,
              contexts_per_source=4, maximum_source_pixels=64_000_000,
              maximum_inventory_regions_per_source=128,
              maximum_overview_side=2048, maximum_context_overview_side=1536, native_tile_side=1024,
              selection='round robin distinct conservative groups ordered by their smallest source SHA/path; two rounds; no filename or score preferences',
              source_scope='declared train PNG only, outside development/closed/reserved groups and prior graph-source hashes',
              annotation_scope='full-page region inventory plus up to three complete semantic contexts; no fixed central crop',
              source_origin='unknown until evidence; folder/name/appearance do not prove CAD export or real scan',
              automatic_replacement=False, automatic_expansion=False, images_decoded=False, fit_authorized=False)
SLOTS = ('full_page','geometric_context','annotation_context','boundary_context')


def select(inventory, decisions, used_hashes):
    admitted, rejected = [], []
    for row in inventory:
        if row.get('declared_split') != 'train' or row.get('role') != 'input' or not row['path'].lower().endswith('.png'):
            continue
        source = dict(path='dataset/'+row['path'],sha256=row['sha256'])
        d = decisions[row['sha256']]
        reasons = list(d['reasons'])
        if d['blocked'] and not reasons: reasons.append('blocked')
        if d['development_related_family']: reasons.append('development_related')
        if row['sha256'] in used_hashes: reasons.append('previous_graph_source')
        record = dict(source=source,family=d['canonical_family'])
        if reasons: rejected.append(dict(**record,reasons=sorted(set(reasons))))
        else: admitted.append(record)
    admitted.sort(key=lambda r:(r['source']['sha256'],r['source']['path']))
    groups = defaultdict(list)
    seen = set()
    for row in admitted:
        if row['source']['sha256'] in seen: continue
        seen.add(row['source']['sha256']);groups[row['family']].append(row)
    order = sorted(groups,key=lambda g:(groups[g][0]['source']['sha256'],groups[g][0]['source']['path'],g))
    selected = []
    for round_index in range(POLICY['maximum_sources_per_group']):
        for group in order:
            if round_index < len(groups[group]) and len(selected) < POLICY['maximum_sources']:
                selected.append(dict(**groups[group][round_index],selection_round=round_index+1))
    return selected, admitted, rejected


def box(value, size):
    if (not isinstance(value,list) or len(value)!=4 or any(type(v) is not int for v in value)
            or not 0<=value[0]<value[2]<=size[0] or not 0<=value[1]<value[3]<=size[1]):
        raise ValueError('Integer half-open source box required')
    return value


def context_assignments(regions, size):
    """Annotation windows only. Never hand these GT windows to inference workers."""
    def key(r):
        a,b,c,d = r['box']
        return b,a,(c-a)*(d-b),r['id']
    known = sorted([r for r in regions if r['label_status']=='known'],key=key)
    geometric = [r for r in known if r['roles'][0] in ('view','detail')]
    annotations = [r for role in ('table','title_block','notes') for r in known if r['roles'][0]==role]
    rows = [dict(slot='full_page',status='assigned',box=[0,0,*size],region_ids=[],purpose='complete coarse region inventory; partial stroke truth stays separately scoped')]
    for slot,chosen in [('geometric_context',geometric[:1]),('annotation_context',annotations[:1])]:
        rows.append(dict(slot=slot,status='assigned' if chosen else 'absent',
                         box=chosen[0]['box'] if chosen else None,region_ids=[r['id'] for r in chosen]))
    # A full geometric region plus its closest complete annotation region retains
    # border/title context; never select windows using model confidence/errors.
    boundary = []
    if geometric and annotations:
        first = geometric[0]
        def distance(r):
            a,b,c,d = first['box'];e,f,g,h = r['box']
            return ((a+c-e-g)**2+(b+d-f-h)**2,key(r))
        boundary = [first,min(annotations,key=distance)]
    if boundary:
        b = [min(r['box'][0] for r in boundary),min(r['box'][1] for r in boundary),
             max(r['box'][2] for r in boundary),max(r['box'][3] for r in boundary)]
        duplicate = any(r['status']=='assigned' and r['box']==b for r in rows)
        rows.append(dict(slot='boundary_context',status='already_covered' if duplicate else 'assigned',box=b,
                         region_ids=[r['id'] for r in boundary]))
    else:
        rows.append(dict(slot='boundary_context',status='absent',box=None,region_ids=[]))
    return rows


def source_template(case):
    return dict(source_id=case['id'],source=case['source'],family=case['family'],source_size_px=case['source_size_px'],
                status='pending',reviewed_entire_page=False,region_inventory_complete=False,
                origin=dict(kind='unknown',evidence=[],appearance='unreviewed',independence_confirmed=False),
                overview=None,regions=[],objects=[],contacts=[],unknown_areas=[],review_notes='',
                context_slots=[dict(slot=s,status='pending',box=[0,0,*case['source_size_px']] if s=='full_page' else None,region_ids=[]) for s in SLOTS],
                authority='assistant_source_review',human_verified=False,physical_geometry_approved=False,
                physical_units_confirmed=False,derived_from_native_DXF=False,dense_labels_approved=False)


def validate_overview(value, size):
    """Integer BOX-average overview: separate pixel-centre and pixel-edge maps."""
    factor=math.ceil(max(size)/POLICY['maximum_overview_side'])
    if value.get('factor')!=factor or value.get('size_px')!=[math.ceil(v/factor) for v in size]:
        raise ValueError('Frozen integer overview sampling changed')
    ref=value.get('image',{})
    path=ref.get('path','')
    if (set(ref)!={'path','sha256'} or not re.fullmatch('[0-9a-f]{64}',ref['sha256']) or not path
            or '\\' in path or ':' in path or PurePosixPath(path).is_absolute() or '..' in PurePosixPath(path).parts):
        raise ValueError('Workspace relative overview image binding required')
    expected=dict(source_to_overview_edges=[[1/factor,0,0],[0,1/factor,0],[0,0,1]],
                  overview_to_source_edges=[[factor,0,0],[0,factor,0],[0,0,1]],
                  source_to_overview_centres=[[1/factor,0,(1/factor-1)/2],[0,1/factor,(1/factor-1)/2],[0,0,1]],
                  overview_to_source_centres=[[factor,0,(factor-1)/2],[0,factor,(factor-1)/2],[0,0,1]])
    for key,matrix in expected.items():
        actual=value.get(key)
        if (not isinstance(actual,list) or len(actual)!=3 or any(not isinstance(row,list) or len(row)!=3 for row in actual)
                or any(not isinstance(v,(int,float)) or not math.isfinite(v) for row in actual for v in row)
                or any(abs(actual[i][j]-matrix[i][j])>1e-12 for i in range(3) for j in range(3))):
            raise ValueError('Centre/edge source transform differs')
    return True


def validate_annotation(value, cases):
    """Check annotation structure and completeness, never silently authorize fit."""
    if value.get('schema_version')!='semantic-context-reading/1.0':raise ValueError('Unknown annotation schema')
    rows = value.get('sources',[])
    by_id = {c['id']:c for c in cases}
    if len(rows)!=len(cases) or {r['source_id'] for r in rows}!=set(by_id):raise ValueError('Entire fixed source assignment required')
    complete = 0
    for row in rows:
        case = by_id[row['source_id']];size=case['source_size_px']
        if any(row[k]!=case[k] for k in ('source','family','source_size_px')):raise ValueError('Source/family binding changed')
        if (row.get('authority')!='assistant_source_review' or row.get('human_verified') is not False
                or any(row.get(k) is not False for k in ('physical_geometry_approved','physical_units_confirmed','derived_from_native_DXF','dense_labels_approved'))):
            raise ValueError('Protocol admits raster semantic review only, no CAD/physical/dense approval')
        origin = row['origin']
        if origin['kind'] not in ('unknown','clean_cad_export','real_scan','photograph','synthetic_distortion'):
            raise ValueError('Unknown origin stratum')
        if origin['kind']!='unknown' and not origin['evidence']:raise ValueError('Appearance/folder is not origin evidence')
        for evidence in origin['evidence']:
            if (evidence.get('authority') not in ('user_origin_statement','source_provenance_manifest','reproducible_renderer_manifest','synthetic_transform_manifest')
                    or evidence.get('source_sha256')!=case['source']['sha256']
                    or set(evidence.get('reference',{}))!={'path','sha256'}):
                raise ValueError('Origin requires source-bound provenance, not visual appearance')
        if origin['independence_confirmed'] is not False:raise ValueError('No independent-family admission in train intake')
        if origin['kind']=='synthetic_distortion' and not origin.get('parent_source'):
            raise ValueError('Synthetic derivatives retain original source parent/group')
        regions = row['regions']
        if len(regions)>POLICY['maximum_inventory_regions_per_source']:raise ValueError('Region inventory budget; no silent truncation')
        ids = {r['id'] for r in regions}
        if len(ids)!=len(regions):raise ValueError('Duplicate region ID')
        for region in regions:
            box(region['box'],size)
            status=region['label_status'];roles=region['roles']
            if status not in ('known','ambiguous','unknown') or len(set(roles))!=len(roles) or not set(roles)<=set(REGIONS):
                raise ValueError('Explicit region roles/ambiguity required')
            if status=='known' and (len(roles)!=1 or len(region.get('evidence','').strip())<8):
                raise ValueError('Known region needs one supported semantic label')
        objects = {o['id']:o for o in row['objects']}
        if len(objects)!=len(row['objects']):raise ValueError('Duplicate source object')
        for obj in objects.values():
            box(obj['box'],size)
            if (any(type(r) is not int for r in obj['roles']) or not set(obj['roles'])<=set(ROLES)
                    or len(set(obj['roles']))!=len(obj['roles'])):raise ValueError('Invalid object roles')
            if obj['label_status'] not in ('known','ambiguous','unknown','multi_role'):raise ValueError('Explicit object ambiguity required')
            if obj['label_status']=='known' and (len(obj['roles'])!=1 or not obj.get('complete_visible_primitive')):
                raise ValueError('Known whole object must be complete and single-role')
            if obj['label_status']=='multi_role' and len(obj['roles'])<2:raise ValueError('Shared semantics need multiple role bits')
            target=obj['roles'][0] if obj['label_status']=='known' else 255
            if obj.get('supervision_target')!=target:raise ValueError('Ambiguous/shared/unknown role must stay IGNORE255')
            if obj.get('physical_geometry_approved') is not False:raise ValueError('Physical regression is not approved')
            if not set(obj.get('parent_region_ids',[]))<=ids:raise ValueError('Unknown object region')
            if not obj.get('source_unit_id') or not obj.get('correlation_parent_id'):
                raise ValueError('Original source unit and correlation parent required')
        contact_ids=set()
        for contact in row['contacts']:
            if contact['id'] in contact_ids:raise ValueError('Repeated source contact')
            contact_ids.add(contact['id'])
            names=contact['objects']
            if len(names)!=2 or names[0]==names[1] or not set(names)<=set(objects):raise ValueError('Contact needs two reviewed source objects')
            if contact['relation'] not in ('same_primitive','dimension_extension','reviewed_unrelated'):raise ValueError('Unknown relation')
            if contact['relation']=='dimension_extension' and [objects[n]['roles'] for n in names]!=[[2],[3]]:
                raise ValueError('DIM/EXT direction requires explicit single source roles')
            if contact['relation']=='dimension_extension' and any(objects[n]['label_status']!='known' for n in names):
                raise ValueError('DIM/EXT contacts cannot assert unknown source roles')
            if contact['relation']=='same_primitive' and len({objects[n]['source_unit_id'] for n in names})!=1:
                raise ValueError('Same primitive needs one original source identity')
        if row['status']=='complete':
            if not row['reviewed_entire_page'] or not row['region_inventory_complete'] or len(row['review_notes'].strip())<8:
                raise ValueError('Full-page review/coverage proof required')
            if row['overview'] is None:raise ValueError('A source-bound overview and transforms are required')
            validate_overview(row['overview'],size)
            if row['context_slots']!=context_assignments(regions,size):raise ValueError('Deterministic complete-region windows changed')
            if any(not set(o.get('parent_region_ids',[]))<=ids for o in objects.values()):raise ValueError('Missing parent region')
            complete+=1
        elif row['status']!='pending':raise ValueError('Pending or complete review status required')
    return dict(fixed_sources=len(rows),complete_sources=complete,pending_sources=len(rows)-complete,
                structural_annotation_valid=True,annotation_complete=complete==len(rows),
                fit_authorized=False,dense_training_admitted=False,independent_evaluation=False)
