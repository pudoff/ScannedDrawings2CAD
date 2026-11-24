"""Native facts for additional role admission; no physical or family truth inferred."""
from collections import defaultdict
import math
import re

import ezdxf
from ezdxf.math import Matrix44

from .dataset_layout_v1 import checked
from .dimension_attachment_candidates import collect_attachment_candidates
from .dimension_field_decisions import canonical_annotation, join_native_display, parse_reviewed_annotation
from .dimension_field_native import native_geometry_snapshot
from .dxf_semantics import _dimension
from .family_catalog_v2 import require, stable_digest
from .object_review import _binding

ROLE_SUPPORT_RULES = {
    'chamfer_size': ({'chamfer_outer_corner', 'chamfer_flat_corner'},),
    'flange_width': ({'collar_outer_face', 'collar_inner_face'},),
    'drill_depth': ({'hole_mouth_endface', 'full_diameter_drill_depth'},),
    'plate_thickness': ({'sheet_wall_first', 'sheet_wall_second'},),
    'hole_diameter': ({'hole_wall_first', 'hole_wall_second'},
                      {'hole_circle_endpoint_first', 'hole_circle_endpoint_second'}),
}


def survey_native(root, catalog, progress=None):
    """Screen all non-test native text, including block definitions, without rendering."""
    by_sha = defaultdict(list)
    for source in catalog['entries']:
        if source['declared_split'] in ('train', 'validation'):
            by_sha[source['target_sha256']].append(source)
    rows = []
    for index, (sha, sources) in enumerate(sorted(by_sha.items())):
        reference = min(sources, key=lambda s: s['target'])
        row = {'sha256': sha, 'paths': sorted({s['target'] for s in sources}),
               'splits': sorted({s['declared_split'] for s in sources})}
        path = checked(root/'dataset', {'path': reference['target'], 'sha256': sha})
        try:
            doc = ezdxf.readfile(path)
            require(len(doc.entitydb) <= 200_000, 'Native survey entity budget exceeded')
            row['texts'] = sorted({e.plain_text() for e in doc.entitydb.values()
                                  if e.is_alive and e.dxftype() in ('TEXT', 'MTEXT')})
            row['dimensions'] = sorted({str(e.dxf.get('text', '')) for e in doc.modelspace().query('DIMENSION')})
            row['status'] = 'read'
        except (OSError, ValueError, ArithmeticError, ezdxf.DXFError) as error:
            row.update(status='unreadable', error=type(error).__name__+': '+str(error)[:200])
        rows.append(row)
        if progress and index % 100 == 0:
            progress(index+1, len(by_sha))
    return rows


def annotation_neighbours(survey, sha, limit=6):
    """A reproducible candidate screen, explicitly insufficient for family approval."""
    pattern = r'(?:[Rr\u2205\u00d8n\ue90d])?\d+(?:[.,]\d+)?(?:\*)?'
    def tokens(row):
        return {s for s in row.get('texts', []) if re.fullmatch(pattern, s)}
    base = tokens(next(r for r in survey if r['sha256'] == sha))
    ranked = []
    for row in survey:
        if row['sha256'] == sha or row['status'] != 'read':
            continue
        other = tokens(row)
        common = base & other
        if len(common) >= 4:
            ranked.append({'sha256': row['sha256'], 'paths': row['paths'], 'splits': row['splits'],
                           'common_tokens': sorted(common), 'score': len(common)/len(base | other)})
    return sorted(ranked, key=lambda r: (-r['score'], -len(r['common_tokens']), r['paths'][0]))[:limit]


def compose_groups(overlay, survey, constraints):
    """Coallocate whole parents for this release only; never change declared splits."""
    groups = {g['id']: g for g in overlay['groups']}
    membership = {name: g['id'] for g in groups.values() for name in g['members']}
    require(len(membership) == sum(len(g['members']) for g in groups.values()), 'Repeated parent member')
    parent = {key: key for key in groups}
    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key
    applied = []
    require(len({c['id'] for c in constraints}) == len(constraints), 'Repeated family constraint')
    for constraint in constraints:
        require(constraint['basis'] in ('shared_project_development_pool', 'visually_related_validation_designs'), 'Unknown coallocation basis')
        paths = constraint['targets']
        require(paths and len(paths) == len(set(paths)) and all(p in membership for p in paths), 'Invalid coallocation targets')
        if constraint['basis'] == 'shared_project_development_pool':
            prefix = constraint['printed_project_prefix']
            require(prefix and paths == sorted({p for r in survey if r['status'] == 'read'
                    and any(t.strip().startswith(prefix) for t in r['texts']) for p in r['paths']}), 'Incomplete printed project pool')
        ids = sorted({membership[p] for p in paths})
        splits = {s for key in ids for s in groups[key]['declared_splits']}
        require(len(ids) >= 2 and len(splits) == 1 and not any(groups[k]['cross_split'] for k in ids), 'Constraint crosses splits or lacks parents')
        require((constraint['basis'] == 'shared_project_development_pool' and splits == {'train'})
                or (constraint['basis'] == 'visually_related_validation_designs' and splits == {'validation'}), 'Wrong constraint scope')
        for key in ids[1:]:
            a, b = sorted((find(ids[0]), find(key)))
            parent[b] = a
        applied.append({**constraint, 'parent_group_ids': ids, 'semantic_family_approved': False})
    components = defaultdict(list)
    for key in sorted(groups):
        components[find(key)].append(key)
    mapping, result = {}, []
    for ids in components.values():
        key = ids[0] if len(ids) == 1 else 'role-pool-'+stable_digest(ids)[:20]
        result.append({'id': key, 'parent_group_ids': ids,
            'members': sorted(p for old in ids for p in groups[old]['members']),
            'declared_splits': groups[ids[0]]['declared_splits'], 'cross_split': False,
            'semantic_family_approved': False})
        mapping.update({old: key for old in ids})
    require(sum(len(g['members']) for g in result) == len(membership), 'Parent membership lost')
    return {'schema_version': 'role-admission-groups/1.0', 'groups': sorted(result, key=lambda g: g['id']),
            'parent_mapping': mapping, 'constraints': applied, 'independent_evaluation': False,
            'declared_splits_changed': False, 'parent_groups_split': False}


def inspect_slot(doc, slot, decision):
    """Re-read a reviewed DIMENSION and each explicitly selected finite support."""
    require(decision['slot_id'] == slot['id'], 'Decision identity differs')
    entity = doc.entitydb.get(decision['dimension_handle'])
    require(entity is not None and entity.dxftype() == 'DIMENSION'
            and entity.dxf.owner == doc.modelspace().block_record_handle, 'Expected direct modelspace dimension')
    native = _dimension(entity, doc, '*Model_Space', [], Matrix44())
    require(native['kind'] in ('linear', 'aligned', 'diameter', 'radius'), 'Unsupported dimension kind')
    require(tuple(entity.dxf.extrusion) == (0., 0., 1.), 'Unsupported dimension plane')
    pieces = [t['text'] for t in native['stored_display_texts']]
    require(pieces == decision['native_display_pieces'], 'Native display changed')
    source_text = ';'.join(slot['transcription_lines'])
    mapping = decision['display_comparison']
    require(mapping in ('canonical_text_agrees', 'font_glyph_reviewed', 'private_use_unresolved'), 'Unknown display comparison')
    if mapping == 'canonical_text_agrees':
        require(canonical_annotation(join_native_display(pieces)) == canonical_annotation(source_text), 'Source/native display conflict')
    elif mapping == 'font_glyph_reviewed':
        # Mapping is an explicit visual assertion for this object, never a global n->diameter substitution.
        require(decision['glyph_evidence'] and any('n' in p for p in pieces), 'Missing font glyph evidence')
        require(canonical_annotation(join_native_display([p.replace('n', 'Ø') for p in pieces]).replace('\n', ';'))
                == canonical_annotation(source_text), 'Reviewed glyph mapping disagrees with source')
    else:
        require(decision['role_decision'] == 'excluded_taxonomy' and slot['state'] == 'unsupported_role_taxonomy'
                and any('\ue90d' in p for p in pieces), 'Unresolved private glyph cannot approve role')
    measurement = native['measurement_raw']
    factor = native['style_effective'].get('dimlfac')
    require(type(measurement) in (int, float) and math.isfinite(measurement) and measurement > 0
            and type(factor) in (int, float) and math.isfinite(factor) and factor > 0, 'Invalid native measurement/factor')
    if decision['role_decision'] == 'accepted_categorical_role':
        require(slot['state'] == 'reviewed_candidate' and decision['role'] == slot['candidate_role'], 'Unreviewed role promotion')
        require(decision['role'] in ROLE_SUPPORT_RULES and
                {s['role_observation'] for s in decision['supports'].values()} in ROLE_SUPPORT_RULES[decision['role']],
                'Role does not match explicit support observations')
        nominal = parse_reviewed_annotation(source_text)['nominal']
        require(math.isclose(nominal, measurement*factor, rel_tol=0, abs_tol=1e-6), 'Native numeric mismatch')
    else:
        require(decision['role_decision'] == 'excluded_taxonomy' and decision['role'] is None
                and slot['state'] == 'unsupported_role_taxonomy', 'Unsupported rejection')
    incidence = collect_attachment_candidates(doc, [native])['dimensions'][0]
    supports = decision['supports']
    require(set(supports) == set(incidence['points']), 'Every required anchor needs an explicit support')
    bindings = []
    for field, support in supports.items():
        candidates = incidence['points'][field]['candidates']
        require(any(c['geometry_handle'] == support['handle'] and c['relation'] == support['relation'] for c in candidates),
                'Selected geometry is not an exact finite candidate')
        geometry = doc.entitydb[support['handle']]
        binding = _binding(geometry, native['definition_points_modelspace'][field], support['relation'], {})
        require(binding['exact_finite_geometry_attachment'] is True, 'Infinite support cannot enter categorical release')
        bindings.append({'definition_field': field, 'role_observation': support['role_observation'],
                         'identity': {'handle': support['handle'], 'insert_path': []},
                         'native_geometry': native_geometry_snapshot(geometry), 'binding': binding})
    return {'slot_id': slot['id'], 'dimension_handle': entity.dxf.handle, 'native_dimension': native,
        'selected_bindings': bindings, 'numeric_comparison': ('agrees_in_native_units_with_dimlfac'
            if decision['role_decision'] == 'accepted_categorical_role' else 'not_required_for_excluded_role'),
        'display_comparison': mapping, 'physical_geometry_approved': False,
        'coordinate_regression_approved': False, 'physical_scale_confirmed': False}
