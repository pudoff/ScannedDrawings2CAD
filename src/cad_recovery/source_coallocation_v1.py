"""Reviewed source-format links restrict coallocation; they never approve labels."""
from collections import Counter
from copy import deepcopy
from html import escape
import os
from pathlib import Path

from .dataset_layout_v1 import checked, encoded, inside, pin, read, verify_layout_snapshot
from .family_catalog_v2 import require, stable_digest

CODE = ('src/cad_recovery/source_coallocation_v1.py', 'scripts/review_source_coallocation_v1.py',
        'src/cad_recovery/dataset_layout_v1.py', 'src/cad_recovery/family_catalog_v2.py')
DEPENDENCIES = {'queue_manifest', 'audit_manifest', 'history_manifest', 'layout_manifest',
                'catalog', 'sources', 'readings'}
FALSE_FLAGS = ('training_eligible', 'physical_geometry_approved', 'semantic_family_approved', 'independent_evaluation')


def package(root, record, schema):
    path = checked(root, record)
    manifest = read(path)
    require(manifest['schema_version'] == schema, 'Unexpected package schema')
    names = [item['path'] for item in manifest['files']]
    require(len(names) == len(set(n.casefold() for n in names)), 'Repeated package file')
    actual = {p.relative_to(path.parent).as_posix() for p in path.parent.rglob('*') if p.is_file()}
    require(actual == set(names)|{'manifest.json'} and 'manifest.json' not in names, 'Package inventory differs')
    for item in manifest['files']:
        checked(path.parent, item)
    return path.parent, manifest


def validate_readings(queue, reading):
    require(reading['schema_version'] == 'source-coallocation-readings/1.0', 'Unknown reading schema')
    require(reading['reviewer'] == 'assistant' and reading['human_verified'] is False, 'Changed review authority')
    require(all(reading[flag] is False for flag in FALSE_FLAGS), 'Review cannot approve labels or independence')
    candidates = {c['id']: c for c in queue['candidates']}
    require(len(candidates) == len(queue['candidates']), 'Repeated queue candidate')
    decisions = reading['decisions']
    require(len(decisions) == len(candidates) and {d['candidate_id'] for d in decisions} == set(candidates),
            'Every queued candidate needs exactly one decision')
    for decision in decisions:
        candidate = candidates[decision['candidate_id']]
        require(decision['decision'] in ('same_drawing_for_joint_assignment', 'deferred'), 'Unknown source decision')
        require(decision['reviewed_images'] == [e['image'] for e in candidate['source_evidence']],
                'Reviewed source images differ from queue')
        require(isinstance(decision['drawing_identifier'], str) and decision['drawing_identifier'].strip(), 'Missing printed identifier')
        require(isinstance(decision['part_name'], str) and decision['part_name'].strip(), 'Missing printed part name')
        require(isinstance(decision['observations'], list) and len(decision['observations']) >= 3
                and all(isinstance(s, str) and len(s.strip()) >= 8 for s in decision['observations']),
                'Insufficient explicit source observations')
        require(isinstance(decision['differences'], list) and decision['differences'], 'Missing difference/uncertainty observation')
        require(candidate['declared_splits'] == ['train'] and candidate['cross_split_candidate'] is False,
                'This reviewed overlay is train-only')
    return {d['candidate_id']: d for d in decisions}


def compose(base, queue, reading):
    """Union whole parent groups and retain all inherited records and constraints."""
    require(base['schema_version'] == 'history-coallocation/1.0', 'Unknown parent groups')
    require(base['independent_evaluation'] is False, 'Parent independence cannot be reinterpreted')
    decisions = validate_readings(queue, reading)
    groups = {g['id']: g for g in base['groups']}
    require(len(groups) == len(base['groups']), 'Repeated parent group')
    membership = {}
    for g in groups.values():
        require(g['cross_split'] is False and g['semantic_family_approved'] is False, 'Conflicted or differently approved parent group')
        require(len(g['members']) == len(set(g['members'])) and g['members'], 'Invalid parent membership')
        for name in g['members']:
            require(name not in membership, 'Repeated parent member')
            membership[name] = g['id']
    parent = {key: key for key in groups}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    constraints = deepcopy(base['constraints'])
    added = []
    for candidate in queue['candidates']:
        names = candidate['targets']+candidate['inputs']
        require(names and all(name in membership for name in names), 'Unknown candidate source/target')
        ids = sorted({membership[name] for name in names})
        require(ids == candidate['history_group_ids'] and len(ids) > 1, 'Candidate parent groups differ')
        require(all(groups[key]['declared_splits'] == ['train'] for key in ids), 'Cannot reassign non-train parent')
        if decisions[candidate['id']]['decision'] == 'deferred':
            continue
        for key in ids[1:]:
            a, b = sorted((find(ids[0]), find(key)))
            parent[b] = a
        added.append({'id': candidate['id'], 'basis': 'assistant_reviewed_same_drawing_across_source_formats',
            'parent_history_group_ids': ids, 'members': sorted(names),
            'semantic_family_approved': False, 'physical_geometry_approved': False})
    require(len({c['id'] for c in constraints+added}) == len(constraints)+len(added), 'Repeated inherited/new constraint')
    components = {}
    for key in sorted(groups):
        components.setdefault(find(key), []).append(groups[key])
    result = []
    mapping = []
    for originals in components.values():
        if len(originals) == 1:
            group = deepcopy(originals[0])
        else:
            ids = sorted(g['id'] for g in originals)
            group = {'id': 'source-group-'+stable_digest(ids)[:20],
                'members': sorted(name for g in originals for name in g['members']),
                'declared_splits': ['train'], 'cross_split': False, 'semantic_family_approved': False,
                'parent_group_ids': sorted({key for g in originals for key in g['parent_group_ids']}),
                'parent_history_group_ids': ids, 'inherited_parent_groups': deepcopy(originals)}
        result.append(group)
        mapping.extend({'parent_group_id': g['id'], 'group_id': group['id']} for g in originals)
    return {'schema_version': 'source-coallocation/1.0', 'groups': sorted(result, key=lambda g: g['id']),
        'parent_group_mapping': sorted(mapping, key=lambda row: row['parent_group_id']),
        'constraints': constraints+added, 'inherited_constraint_count': len(constraints),
        'new_constraint_count': len(added), 'training_eligible': False, 'physical_geometry_approved': False,
        'semantic_family_approved': False, 'independent_evaluation': False,
        'declared_splits_changed': False, 'parent_groups_split': False}


def context(workspace, config_pin):
    root = Path(workspace).resolve()
    config = read(checked(root, config_pin))
    require(config['schema_version'] == 'source-coallocation-config/1.0' and config['task'] == 'P08.4c', 'Unknown review config')
    require(config['code'] == [pin(root, name) for name in CODE], 'Source review code changed')
    require(set(config['dependencies']) == DEPENDENCIES, 'Incomplete source review dependencies')
    require(all(config[flag] is False for flag in FALSE_FLAGS), 'Config cannot approve labels or independence')
    paths = {key: checked(root, value) for key, value in config['dependencies'].items()}
    queue_dir, _ = package(root, config['dependencies']['queue_manifest'], 'geometry-lineage-review-manifest/1.0')
    audit_dir, _ = package(root, config['dependencies']['audit_manifest'], 'geometry-lineage-manifest/1.0')
    history_dir, _ = package(root, config['dependencies']['history_manifest'], 'dataset-history-manifest/1.0')
    queue, audit, base = read(queue_dir/'queue.json'), read(audit_dir/'report.json'), read(history_dir/'groups.json')
    require(queue['schema_version'] == 'geometry-lineage-review-queue/1.0'
            and queue['training_eligible'] is False and queue['independent_evaluation'] is False
            and queue['groups_changed'] is False, 'Changed queue scope')
    require(queue['dependencies'] == {key: value for key, value in (
        ('audit_manifest', config['dependencies']['audit_manifest']), ('catalog', config['dependencies']['catalog']),
        ('sources', config['dependencies']['sources']), ('history_groups', pin(root, (history_dir/'groups.json').relative_to(root).as_posix())))},
        'Queue lineage differs')
    candidates = {c['id']: c for c in audit['candidates'] if len(c['history_group_ids']) > 1}
    require(len(queue['candidates']) == len(candidates) and {c['id'] for c in queue['candidates']} == set(candidates),
            'Changed candidate scope')
    catalog = {row['input']: row for row in read(paths['catalog'])['entries']}
    sources = {row['input']: row for row in read(paths['sources'])}
    for c in queue['candidates']:
        require(all(c[key] == value for key, value in candidates[c['id']].items()), 'Queue differs from native audit')
        require(len(c['source_evidence']) == len(c['inputs'])
                and {e['input'] for e in c['source_evidence']} == set(c['inputs']), 'Incomplete source previews')
        for e in c['source_evidence']:
            pair = catalog[e['input']]
            require(e['declared_split'] == pair['declared_split'] == 'train'
                    and e['input_sha256'] == pair['input_sha256'] and pair['target'] in c['targets']
                    and pair['target_sha256'] in c['target_sha256'], 'Source/target routing differs')
            checked(root, {'path': 'dataset/'+e['input'], 'sha256': e['input_sha256']})
            checked(root, e['image'])
            if e['kind'] == 'existing_source_overview':
                require(e['image'] in sources[e['input']]['prepared_assets'], 'Unknown source overview')
                checked(root, sources[e['input']]['preparation'])
            else:
                require(e['kind'] == 'existing_pdf_preview', 'Unknown source evidence')
                vis = pair['visual_evidence']
                source = next(a for a in vis['artifacts'] if a['path'].endswith('/source.png'))
                expected = {'path': 'artifacts/dataset-content-v1/'+vis['phase']+'/'+source['path'], 'sha256': source['sha256']}
                require(e['image'] == expected, 'Unknown PDF source preview')
    verify_layout_snapshot(root, config['dependencies']['layout_manifest'])
    reading = read(paths['readings'])
    validate_readings(queue, reading)
    return root, config, queue, base, reading


def summary(base, reading, overlay):
    return {'reviewed_candidates': len(reading['decisions']), 'accepted_source_links': overlay['new_constraint_count'],
        'deferred_candidates': sum(d['decision'] == 'deferred' for d in reading['decisions']),
        'parent_groups': len(base['groups']), 'conservative_groups': len(overlay['groups']),
        'files_covered': sum(len(g['members']) for g in overlay['groups']),
        'cross_split_groups': sum(g['cross_split'] for g in overlay['groups']),
        'inherited_constraints': len(base['constraints']),
        'groups_by_split': dict(sorted(Counter(','.join(g['declared_splits']) or 'unassigned' for g in overlay['groups']).items())),
        'training_eligible': False, 'semantic_family_approved': False, 'physical_geometry_approved': False,
        'independent_evaluation': False, 'declared_splits_changed': False, 'training_started': False}


def build(workspace, config_pin, output):
    root = Path(workspace).resolve()
    target = inside(root, output)
    require(not target.exists() and not target.resolve().is_relative_to((root/'dataset').resolve()), 'Choose a new output outside dataset')
    root, config, queue, base, reading = context(root, config_pin)
    overlay = compose(base, queue, reading)
    stats = summary(base, reading, overlay)
    target.mkdir(parents=True, exist_ok=False)
    (target/'groups.json').write_bytes(encoded(overlay))
    (target/'reviews.json').write_bytes(encoded(reading))
    (target/'report.json').write_bytes(encoded({'schema_version': 'source-coallocation-report/1.0',
        'config': config_pin, 'summary': stats,
        'purpose': 'additive joint-assignment constraints for future releases; existing releases remain historical'}))
    decision_map = {d['candidate_id']: d for d in reading['decisions']}
    sections = []
    for c in queue['candidates']:
        d = decision_map[c['id']]
        figures = []
        for e in c['source_evidence']:
            link = os.path.relpath(root/e['image']['path'], target).replace('\\', '/')
            figures.append('<figure><a href="'+escape(link, quote=True)+'"><img loading="lazy" src="'+escape(link, quote=True)+'"></a><figcaption>'+escape(e['input'])+'</figcaption></figure>')
        sections.append('<section><h2>'+escape(d['drawing_identifier']+' — '+d['part_name'])+'</h2><p>'+escape(d['decision'])+
            '</p><ul>'+''.join('<li>'+escape(note)+'</li>' for note in d['observations'])+'</ul><p>'+escape('; '.join(d['differences']))+'</p><div class="pair">'+''.join(figures)+'</div></section>')
    (target/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>P08.4c source review</title>'
        '<style>body{font:16px sans-serif;margin:2rem}section{border-top:1px solid #aaa;padding:1rem 0}.pair{display:flex;gap:1rem}figure{width:48%;margin:0}img{width:100%}figcaption{overflow-wrap:anywhere}@media(max-width:700px){.pair{display:block}figure{width:100%}}</style>'
        '<h1>P08.4c: проверенные связи исходных PNG/PDF</h1><p>Совместное назначение файлов. '
        'Физические цели, полные DXF и независимость семейств не утверждены.</p><pre>'+escape(str(stats))+'</pre>'+''.join(sections), encoding='utf-8')
    manifest = {'schema_version': 'source-coallocation-manifest/1.0', 'config': config_pin,
        'files': [pin(target, name) for name in ('groups.json', 'reviews.json', 'report.json', 'index.html')]}
    (target/'manifest.json').write_bytes(encoded(manifest))
    return {'manifest': pin(root, (target/'manifest.json').relative_to(root).as_posix()), 'summary': stats}


def load_overlay(workspace, manifest_pin):
    root = Path(workspace).resolve()
    directory, manifest = package(root, manifest_pin, 'source-coallocation-manifest/1.0')
    _, _, queue, base, reading = context(root, manifest['config'])
    overlay = read(directory/'groups.json')
    require(overlay == compose(base, queue, reading), 'Overlay differs from reviewed parent composition')
    require(read(directory/'reviews.json') == reading, 'Published reviews differ')
    report = read(directory/'report.json')
    require(report['schema_version'] == 'source-coallocation-report/1.0'
            and report['config'] == manifest['config'] and report['summary'] == summary(base, reading, overlay),
            'Published summary differs')
    return overlay
