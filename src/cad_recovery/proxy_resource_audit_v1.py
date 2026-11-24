"""Read-only proxy ownership and explicit dependency graph; payloads stay opaque."""
from collections import Counter, defaultdict, deque
import hashlib
import math
import re

from ezdxf.lldxf.tagwriter import TagCollector
from ezdxf.lldxf.types import is_pointer_code

from .dimension_field_native import inspect_drawing_native
from .family_catalog_v2 import require

MAX_ENTITIES = 200_000
MAX_TAGS = 2_000_000
MAX_BINARY_BYTES = 16 * 1024**2
PROXIES = {'ACAD_PROXY_OBJECT', 'ACAD_PROXY_ENTITY'}
SPECIFICATION = 'https://help.autodesk.com/cloudhelp/2020/ENU/AutoCAD-DXF/files/GUID-F59F0EC3-D34D-4C1A-91AC-7FDA569EF016.htm'


def handle(value):
    require(isinstance(value, str) and re.fullmatch(r'[0-9A-Fa-f]+', value) is not None, 'Invalid native handle')
    return value.upper()


def closure(seeds, edges, nodes):
    """Follow declared non-owner references and explicit render containment only."""
    outgoing = defaultdict(list)
    for edge in edges:
        if edge['kind'] != 'owner':
            outgoing[edge['from']].append(edge['to'])
    found, missing = set(seeds), set()
    queue = deque(sorted(seeds))
    while queue:
        current = queue.popleft()
        require(len(found) <= MAX_ENTITIES, 'Dependency graph budget exceeded')
        for target in outgoing[current]:
            if target not in nodes:
                missing.add(target)
            elif target not in found:
                found.add(target)
                queue.append(target)
    return sorted(found), sorted(missing)


def owner_chain(start, entities):
    rows, seen, current = [], set(), start
    while current != '0':
        if current in seen:
            return {'status': 'cycle', 'chain': rows, 'block_record': None}
        if current not in entities:
            return {'status': 'missing_owner', 'chain': rows, 'block_record': None}
        require(len(rows) < 64, 'Proxy ownership depth exceeded')
        seen.add(current)
        entity = entities[current]
        kind = entity.dxftype()
        row = {'handle': current, 'entity_type': kind}
        if kind == 'BLOCK_RECORD':
            row['name'] = entity.dxf.name
            rows.append(row)
            return {'status': 'block_record', 'chain': rows, 'block_record': current}
        rows.append(row)
        owner = entity.dxf.get('owner')
        if not owner:
            return {'status': 'no_block_owner', 'chain': rows, 'block_record': None}
        current = handle(owner)
    return {'status': 'root_without_block', 'chain': rows, 'block_record': None}


def inspect_document(doc, selected_handles):
    """Inventory facts without opening resources, exploding, auditing or saving DXF."""
    entities = {handle(e.dxf.handle): e for e in doc.entitydb.values() if e.is_alive}
    require(len(entities) <= MAX_ENTITIES and len(entities) == sum(e.is_alive for e in doc.entitydb.values()),
            'Invalid or oversized native entity inventory')
    selected = sorted({handle(h) for h in selected_handles})
    require(selected and all(h in entities for h in selected), 'Unknown selected local entity')
    classes = [{'class_id': 500+i, 'name': c.dxf.name, 'cpp_class_name': c.dxf.cpp_class_name,
                'application_name': c.dxf.app_name} for i, c in enumerate(doc.classes)]
    class_index = {c['class_id']: c for c in classes}
    edges, records, proxy_rows, problems = [], [], [], []
    total_tags = total_binary = 0
    for identifier, entity in sorted(entities.items()):
        tags = TagCollector.dxftags(entity)
        total_tags += len(tags)
        require(total_tags <= MAX_TAGS, 'Native tag budget exceeded')
        kind, groups, subclass, tag_rows = entity.dxftype(), [], None, []
        for index, tag in enumerate(tags):
            value = tag.value
            if isinstance(value, bytes):
                total_binary += len(value)
                require(total_binary <= MAX_BINARY_BYTES, 'Native binary budget exceeded')
                stored = {'size_bytes': len(value), 'sha256': hashlib.sha256(value).hexdigest()}
            elif isinstance(value, (str, int)):
                stored = value
            elif isinstance(value, float):
                require(math.isfinite(value), 'Nonfinite native tag')
                stored = value
            else:
                stored = [float(v) for v in value]
                require(all(math.isfinite(v) for v in stored), 'Nonfinite native point tag')
            tag_rows.append([tag.code, stored])
            if tag.code == 102:
                if isinstance(value, str) and value.startswith('{'):
                    groups.append(value)
                elif value == '}' and groups:
                    groups.pop()
            if tag.code == 100:
                subclass = value
            if is_pointer_code(tag.code):
                target = handle(value)
                if target == '0':
                    continue
                relation = ('owner' if tag.code == 330 and subclass is None and not groups
                            and target == entity.dxf.get('owner', '0').upper() else
                            'reactor' if '{ACAD_REACTORS' in groups else
                            'extension_dictionary' if '{ACAD_XDICTIONARY' in groups else 'pointer')
                edges.append({'from': identifier, 'to': target, 'kind': relation, 'group_code': tag.code, 'tag_index': index})
        snapshot = {'handle': identifier, 'entity_type': kind, 'owner': entity.dxf.get('owner'),
                    'tags_sha256': hashlib.sha256(repr(tag_rows).encode('utf-8')).hexdigest()}
        records.append(snapshot)
        if kind in PROXIES:
            class_ids = [t.value for t in tags if t.code == 91]
            proxy_ids = [t.value for t in tags if t.code == 90]
            class_id = class_ids[0] if len(class_ids) == 1 else None
            proxy_rows.append({**snapshot, 'tags': tag_rows, 'proxy_class_ids': proxy_ids,
                'application_class': class_index.get(class_id), 'application_class_id_raw': class_id,
                'owner_trace': owner_chain(identifier, entities),
                'binary_payload_decoded': False})

    def edge(source, target, kind):
        edges.append({'from': source, 'to': target, 'kind': kind, 'group_code': None, 'tag_index': None})

    # Structural links are explicit and read-only: no virtual_entities/copy/transform.
    for block in doc.blocks:
        record = block.block_record_handle
        for entity in block:
            edge(record, entity.dxf.handle, 'block_member')
    for identifier, entity in sorted(entities.items()):
        kind = entity.dxftype()
        block_names = []
        if kind == 'INSERT':
            block_names.append((entity.dxf.name, 'insert_block'))
            for attribute in entity.attribs:
                edge(identifier, attribute.dxf.handle, 'insert_attribute')
        elif kind == 'DIMENSION':
            name = entity.dxf.get('geometry')
            if name:
                block_names.append((name, 'dimension_graphics_block'))
            else:
                problems.append({'handle': identifier, 'problem': 'missing_dimension_graphics_block'})
            name = entity.dxf.dimstyle
            if name in doc.dimstyles:
                edge(identifier, doc.dimstyles.get(name).dxf.handle, 'dimension_style')
            else:
                problems.append({'handle': identifier, 'problem': 'missing_dimension_style'})
        elif kind == 'DIMSTYLE':
            for key in ('dimblk', 'dimblk1', 'dimblk2', 'dimldrblk'):
                name = entity.dxf.get(key, '')
                if name:
                    block_names.append((name, 'dimension_arrow_block'))
        for name, relation in block_names:
            if name in doc.blocks:
                edge(identifier, doc.blocks[name].block_record_handle, relation)
            else:
                problems.append({'handle': identifier, 'problem': 'missing_block', 'name': name})
    edges.sort(key=lambda e: (e['from'], e['to'], e['kind'], e['group_code'] or -1, e['tag_index'] or -1))
    roots = sorted(layout.block_record_handle for layout in doc.layouts)
    active, active_missing = closure(roots, edges, entities)
    explicit_edges = [e for e in edges if e['group_code'] is not None]
    local, local_missing = closure(selected, explicit_edges, entities)
    proxy_handles = {p['handle'] for p in proxy_rows}
    proxy_dependencies, proxy_missing = closure(proxy_handles, edges, entities)
    resource_inventory = inspect_drawing_native(doc)
    proxy_roots = sorted({p['owner_trace']['block_record'] for p in proxy_rows if p['owner_trace']['block_record']})
    return {'schema_version': 'proxy-resource-native-audit/1.0', 'entities': records, 'classes': classes,
        'edges': edges, 'proxies': proxy_rows, 'resources': resource_inventory,
        'layouts': [{'name': l.name, 'block_record': l.block_record_handle, 'entity_count': len(l)} for l in doc.layouts],
        'active_reference_closure': {'roots': roots, 'handles': active, 'missing_handles': active_missing,
            'proxy_handles': sorted(proxy_handles.intersection(active))},
        'selected_reference_closure': {'roots': selected, 'handles': local, 'missing_handles': local_missing,
            'proxy_handles': sorted(proxy_handles.intersection(local))},
        'proxy_reference_closure': {'handles': proxy_dependencies, 'missing_handles': proxy_missing,
            'selected_component_intersection': sorted(set(local).intersection(proxy_dependencies))},
        'problems': problems, 'summary': {'live_entities': len(entities), 'proxy_objects': sum(p['entity_type'] == 'ACAD_PROXY_OBJECT' for p in proxy_rows),
            'proxy_entities': sum(p['entity_type'] == 'ACAD_PROXY_ENTITY' for p in proxy_rows),
            'proxy_owner_block_records': proxy_roots, 'tags': total_tags, 'binary_bytes': total_binary,
            'proxy_class_counts': dict(sorted(Counter(p['application_class']['name'] if p['application_class'] else 'UNRESOLVED' for p in proxy_rows).items()))},
        'graph_policy': 'Declared pointer tags excluding owner traversal, plus block/INSERT/DIMENSION render dependencies. Binary payloads are not decoded.',
        'specification': SPECIFICATION, 'source_modified': False, 'native_target_complete': False,
        'physical_geometry_approved': False, 'coordinate_regression_approved': False, 'independent_evaluation': False}


def local_admission(audit, allowed_classes):
    """Component-only gate; absence of found references never approves whole DXF."""
    proxies = audit['proxies']
    resources = audit['resources']
    conditions = {
        'has_proxy_objects_to_review': bool(proxies),
        'only_non_graphical_proxy_objects': bool(proxies) and all(p['entity_type'] == 'ACAD_PROXY_OBJECT' for p in proxies),
        'all_proxy_classes_explicitly_reviewed': all(p['application_class'] is not None
            and p['application_class']['name'] in allowed_classes and p['proxy_class_ids'] == [499] for p in proxies),
        'proxy_owners_terminate_at_block_records': all(p['owner_trace']['status'] == 'block_record' for p in proxies),
        'proxy_blocks_outside_active_layout_dependencies': not set(audit['summary']['proxy_owner_block_records']).intersection(audit['active_reference_closure']['handles']),
        'active_dependencies_proxy_free': not audit['active_reference_closure']['proxy_handles'],
        'selected_component_proxy_free': not audit['selected_reference_closure']['proxy_handles'],
        'no_declared_proxy_path_into_selected_component': not audit['proxy_reference_closure']['selected_component_intersection'],
        'active_and_selected_references_resolve': not audit['active_reference_closure']['missing_handles'] and not audit['selected_reference_closure']['missing_handles'],
        'no_structural_dependency_problems': not audit['problems'],
        'no_other_native_resources_or_xrefs': not resources['xref_blocks'] and
            all(r['entity_type'] == 'ACAD_PROXY_OBJECT' for r in resources['resource_entities']),
    }
    return {'decision': 'candidate_for_local_categorical_review' if all(conditions.values()) else 'retain_role_exclusion',
            'conditions': conditions, 'failed_conditions': sorted(k for k, v in conditions.items() if not v),
            'whole_native_target_approved': False, 'binary_payload_decoded': False,
            'training_eligible': False, 'independent_evaluation': False, 'physical_geometry_approved': False}
