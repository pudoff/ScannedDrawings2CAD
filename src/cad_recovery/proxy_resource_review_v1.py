"""Pinned P14.1b.11 review, preserving the existing resource exclusion."""
from collections import defaultdict, deque
from html import escape
from importlib.metadata import version
from pathlib import Path

import ezdxf

from .dataset_layout_v1 import checked, encoded, inside, pin, read
from .family_catalog_v2 import require
from .proxy_resource_audit_v1 import inspect_document, local_admission
from .role_admission_release_v1 import verify as verify_parent
from .source_coallocation_v1 import package

CODE = ('src/cad_recovery/proxy_resource_audit_v1.py', 'src/cad_recovery/proxy_resource_review_v1.py',
        'scripts/review_proxy_resources_v1.py', 'src/cad_recovery/role_admission_release_v1.py',
        'src/cad_recovery/dimension_field_native.py', 'src/cad_recovery/dataset_layout_v1.py', 'requirements.lock')
FALSE_FLAGS = ('human_verified', 'training_eligible', 'physical_geometry_approved',
               'coordinate_regression_approved', 'independent_evaluation', 'label_release_changed', 'test_used')


def dependency_path(audit, roots, target):
    """Shortest witness in the declared graph; it is not proof of a render dependency."""
    outgoing = defaultdict(list)
    for edge in audit['edges']:
        if edge['kind'] != 'owner':
            outgoing[edge['from']].append(edge)
    queue, previous = deque(sorted(roots)), {h: None for h in roots}
    while queue:
        current = queue.popleft()
        if current == target:
            result = []
            while previous[current] is not None:
                edge = previous[current]
                result.append(edge)
                current = edge['from']
            return list(reversed(result))
        for edge in outgoing[current]:
            if edge['to'] not in previous:
                previous[edge['to']] = edge
                queue.append(edge['to'])
    return None


def context(root, config_pin):
    config = read(checked(root, config_pin))
    require(config['schema_version'] == 'proxy-resource-review-config/1.0'
            and config['task'] == 'P14.1b.11', 'Unknown proxy review configuration')
    require(config['policy'] == 'preserve_global_native_resource_exclusion/1.0', 'This audit cannot replace the admission policy')
    require(all(config[f] is False for f in FALSE_FLAGS), 'Proxy audit cannot approve labels or physical targets')
    require(config['code'] == [pin(root, n) for n in CODE] and config['ezdxf_version'] == version('ezdxf'),
            'Proxy review implementation/environment changed')
    parent_report, roles, texts = verify_parent(root, config['parent_manifest'])
    folder, _ = package(root, config['parent_manifest'], 'role-admission-manifest/1.0')
    selection = read(checked(root, config['selection']))
    source = next(s for s in selection['sources'] if s['id'] == config['source_id'])
    require(source['id'] == 'validation-3' and source['declared_split'] == 'validation', 'Review is scoped to the selected validation source')
    slot_id = config['slot_id']
    require(slot_id == source['id']+'-1', 'Only the diameter-15 slot is reviewed')
    key = 'additional-'+slot_id
    require(all(r['id'] != key for r in roles), 'Role exclusion already differs from parent')
    text = next(r for r in texts if r['id'] == key)
    require(text['split'] == 'validation' and text['source_sha256'] == source['input_sha256'], 'Reviewed raster lineage differs')
    labels = read(folder/'labels.json')
    excluded = next(r for r in labels['new_role_exclusions'] if r['slot_id'] == slot_id)
    require(excluded['state'] == 'deferred_native_resources', 'Missing resource exclusion in verified parent')
    fact = next(r for r in read(folder/'native-evidence.json') if r['slot_id'] == slot_id)
    require(fact['source_id'] == source['id'] and fact['native_target'] == {'path': 'dataset/'+source['target'], 'sha256': source['target_sha256']},
            'Native review lineage differs')
    reading = read(checked(root, config['reading']))
    require(reading['schema_version'] == 'proxy-resource-reading/1.0' and reading['reviewer'] == 'assistant'
            and reading['slot_id'] == slot_id and reading['decision'] == 'retain_role_exclusion', 'This snapshot preserves the parent exclusion')
    require(all(reading[f] is False for f in FALSE_FLAGS), 'Review authority/scope changed')
    require(len(reading['observations']) >= 4 and all(isinstance(s, str) and len(s.strip()) >= 12 for s in reading['observations']),
            'Missing explicit proxy observations')
    require(reading['allowed_proxy_classes'] == sorted(set(reading['allowed_proxy_classes'])) and reading['allowed_proxy_classes'],
            'Expected an explicit class shortlist')
    return config, source, fact, reading, folder, parent_report, text


def materialize(root, config_pin):
    config, source, fact, reading, parent, parent_report, text = context(root, config_pin)
    target_pin = {'path': source['target'], 'sha256': source['target_sha256']}
    checked(root/'dataset', {'path': source['input'], 'sha256': source['input_sha256']})
    doc = ezdxf.readfile(checked(root/'dataset', target_pin))
    selected = [fact['dimension_handle']] + [b['identity']['handle'] for b in fact['selected_bindings']]
    audit = inspect_document(doc, selected)
    checked(root/'dataset', target_pin)
    require(doc.entitydb[fact['dimension_handle']].dxftype() == 'DIMENSION' and
            {doc.entitydb[b['identity']['handle']].dxftype() for b in fact['selected_bindings']} == {'CIRCLE'},
            'The locally reviewed DIMENSION/CIRCLE identities changed')
    assessment = local_admission(audit, reading['allowed_proxy_classes'])
    require(assessment['decision'] == 'retain_role_exclusion', 'New facts require a separate review, not silent reapproval')
    paths = {h: dependency_path(audit, audit['active_reference_closure']['roots'], h)
             for h in audit['active_reference_closure']['missing_handles']}
    kinds = {r['handle']: r['entity_type'] for r in audit['entities']}
    missing = [{'handle': h, 'incoming_references': [e for e in audit['edges'] if e['to'] == h],
                'witness_path': paths[h], 'witness_source_types': [kinds[e['from']] for e in paths[h] or []],
                'qualification': 'Includes notification/reactor and dictionary links; not a claim that this style is drawn.'}
               for h in sorted(paths)]
    decision = {'schema_version': 'proxy-resource-decision/1.0', 'slot_id': config['slot_id'],
        'decision': reading['decision'], 'assessment': assessment, 'missing_references': missing,
        'selected_native_evidence': fact, 'parent_manifest': config['parent_manifest'],
        'native_target_approved': False, **{f: False for f in FALSE_FLAGS}}
    report = {'schema_version': 'proxy-resource-review-report/1.0', 'config': config_pin,
        'source': {'input': source['input'], 'input_sha256': source['input_sha256'],
                   'target': source['target'], 'target_sha256': source['target_sha256']},
        'summary': audit['summary'], 'decision': reading['decision'],
        'active_declared_reference_nodes': len(audit['active_reference_closure']['handles']),
        'selected_component_nodes': len(audit['selected_reference_closure']['handles']),
        'missing_declared_reference_handles': sorted(paths),
        'parent_label_counts': {k: parent_report[k] for k in ('roles_by_split', 'text_fields_by_split')},
        'parent_manifest': config['parent_manifest'], 'no_new_label_release': True,
        **{f: False for f in FALSE_FLAGS}}
    # Copy the already reviewed raster evidence exactly; there is no new pixel annotation.
    assets = {'audit.json': encoded(audit), 'reading.json': encoded(reading),
              'decision.json': encoded(decision), 'report.json': encoded(report),
              'query.png': (parent/text['query_image']).read_bytes(),
              'context.png': (parent/'additional/contexts'/('validation-3.png')).read_bytes()}
    assets['index.html'] = ('<!doctype html><meta charset="utf-8"><title>P14.1b.11 proxy review</title>'
        '<style>body{font:16px sans-serif;margin:2rem}img{max-width:100%;max-height:220px}pre{white-space:pre-wrap}</style>'
        '<h1>Drawing 110 — resource exclusion retained</h1><p>The local DIMENSION/CIRCLE component is standard. '
        'The full native target remains unapproved; no labels or split changed.</p><img src="query.png" alt="Original query">'
        '<p><a href="context.png">Source context</a> · <a href="audit.json">Native graph</a> · '
        '<a href="decision.json">Decision and witness paths</a></p><pre>'+escape(str(report))+'</pre>'
        +''.join('<p>'+escape(s)+'</p>' for s in reading['observations'])).encode('utf-8')
    return assets, report


def build(workspace, config_pin, output):
    root = Path(workspace).resolve()
    folder = inside(root, output)
    require(not folder.exists() and not folder.resolve().is_relative_to((root/'dataset').resolve()), 'Choose a fresh output outside dataset')
    assets, report = materialize(root, config_pin)
    folder.mkdir(parents=True, exist_ok=False)
    for name, data in assets.items():
        with inside(folder, name).open('xb') as stream:
            stream.write(data)
    with (folder/'manifest.json').open('xb') as stream:
        stream.write(encoded({'schema_version': 'proxy-resource-review-manifest/1.0', 'config': config_pin,
                              'files': [pin(folder, n) for n in sorted(assets)]}))
    return {'manifest': pin(root, (folder/'manifest.json').relative_to(root).as_posix()), 'report': report}


def verify(workspace, manifest_pin):
    root = Path(workspace).resolve()
    folder, manifest = package(root, manifest_pin, 'proxy-resource-review-manifest/1.0')
    assets, report = materialize(root, manifest['config'])
    require({r['path'] for r in manifest['files']} == set(assets), 'Proxy audit inventory differs')
    for name, data in assets.items():
        require((folder/name).read_bytes() == data, 'Proxy audit differs from reviewed facts: '+name)
    return report
