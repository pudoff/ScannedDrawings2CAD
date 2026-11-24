"""Replay content-screening evidence into a current, unapproved pair catalog."""
from collections import Counter, defaultdict
import html
from pathlib import Path
import re
import unicodedata

from .dataset_layout_v1 import checked, digest, encoded, inside, pin, read
from .dataset_content_v1 import PHASES, bind_pairs, inputs, load_phase, stable
from .family_catalog_v2 import require

DESIGNATION = re.compile(r'(?<![\w.])(?:(?:[A-ZА-ЯЁ]{1,8}\.)?\d{3}(?:\.\d{2,4}){2,5}|\d{6,7}(?:[_.]\d{2}){1,3})(?:[A-ZА-ЯЁ]{1,4})?(?![\w.])', re.IGNORECASE)
CODE = ['src/cad_recovery/dataset_content_review_v1.py', 'scripts/assemble_dataset_content_v1.py']


def _index(rows, key):
    require(len({r[key] for r in rows}) == len(rows), 'Repeated evidence key')
    return {r[key]: r for r in rows}


def validate_phases(audit, pairs, phases):
    files = {r['path']: r for r in audit['files']}
    for phase, extension in (('native', '.dxf'), ('pdf', '.pdf')):
        expected = {r['sha256'] for r in files.values() if r['declared_split'] and Path(r['path']).suffix.lower() == extension}
        actual = _index(phases[phase], 'sha256')
        require(set(actual) == expected, 'Incomplete ' + phase + ' evidence')
        for sha, row in actual.items():
            require(row['path'] in files and files[row['path']]['sha256'] == sha
                    and Path(row['path']).suffix.lower() == extension and row['status'] in ('ok', 'error'), 'Invalid native/PDF identity')
    for phase in ('controls', 'pdf_pairs'):
        expected = {p['input']: p for p in pairs if (p['declared_split'] in ('validation', 'test') if phase == 'controls' else p['bundle'] == 'train_pdf')}
        actual = _index(phases[phase], 'input')
        require(set(actual) == set(expected), 'Incomplete visual pair evidence')
        for source, row in actual.items():
            pair = expected[source]
            require(all(row[k] == pair[k] for k in ('input', 'target', 'bundle', 'declared_split'))
                    and row['input_sha256'] == files[source]['sha256']
                    and row['target_sha256'] == files[pair['target']]['sha256'], 'Pair evidence rebound to another source')
            require(row['training_eligible'] is False and row['target_approved'] is False
                    and row['status'] in ('verified', 'quarantine', 'incomplete_reference'), 'Pair screening cannot approve targets')
            if row['status'] == 'verified':
                require(not row['errors'] and row['comparison']['status'] == 'verified'
                        and row['reference_metadata']['reference_complete'] is True, 'Incomplete reference claims visual verification')
    return files


def build_relationships(audit, pairs, native, pdf):
    files = {r['path']: r for r in audit['files']}
    owners = {p: p for p in files}
    links = []

    def find(p):
        while owners[p] != p:
            owners[p] = owners[owners[p]]
            p = owners[p]
        return p

    def join(paths, basis, evidence):
        paths = sorted(set(paths))
        if len(paths) < 2:
            return
        for p in paths[1:]:
            a, b = find(paths[0]), find(p)
            owners[max(a, b)] = min(a, b)
        links.append({'paths': paths, 'basis': basis, 'evidence': evidence, 'semantic_equivalence_approved': False})

    for group in audit['exact_duplicates']:
        join(group['paths'], 'identical_file_bytes', group['sha256'])
    for group in audit['inherited_relationships']:
        join(group['current_paths'], 'historical_group_by_all_matching_hashes', group['historical_group_id'])
    for pair in pairs:
        join([pair['input'], pair['target']], 'candidate_pair_coallocation_only', pair['basis'])
    signatures, designations, literal = defaultdict(list), defaultdict(list), defaultdict(list)
    for path, row in files.items():
        text = []
        if row['role'] == 'target':
            record = native[row['sha256']]
            if record['status'] == 'ok':
                if record.get('geometry_signature'):
                    signatures[record['geometry_signature']].append(path)
                text = record['texts']
        elif row['bundle'] == 'train_pdf':
            record = pdf[row['sha256']]
            if record['status'] == 'ok':
                text = [line for page in record['pages'] for line in page['text'].splitlines()]
        for value in text:
            value = ' '.join(unicodedata.normalize('NFC', value).split())
            for match in DESIGNATION.finditer(value):
                designations[match[0].casefold()].append(path)
            if len(value) >= 12 and len(re.findall('[A-Za-zА-Яа-яЁё]', value)) >= 8:
                literal[value.casefold()].append(path)
    for key, paths in sorted(signatures.items()):
        join(paths, 'selected_native_geometry_signature_relationship_hint', key)
    for key, paths in sorted(designations.items()):
        join(paths, 'shared_literal_designation_candidate', key)
    weak = []
    for value, paths in sorted(literal.items()):
        paths = sorted(set(paths))
        splits = {files[p]['declared_split'] for p in paths}
        if 2 <= len(paths) <= 12 and len(splits) > 1:
            weak.append({'literal_text': value, 'paths': paths, 'basis': 'shared_substantial_literal_text_only',
                         'coallocation_applied': False, 'family_approved': False})
    members = defaultdict(list)
    for path in files:
        members[find(path)].append(path)
    groups = []
    for paths in sorted(members.values(), key=lambda ps: min(ps)):
        paths.sort()
        splits = sorted({files[p]['declared_split'] for p in paths if files[p]['declared_split']})
        groups.append({'id': 'content-group-' + stable(paths)[:20], 'members': paths, 'declared_splits': splits,
                       'cross_split': len(splits) > 1, 'training_eligible': False, 'family_approved': False})
    return {'groups': groups, 'links': links, 'cross_split_groups': [g for g in groups if g['cross_split']],
            'weak_cross_split_text_hints': weak, 'independent_evaluation': False,
            'scope': 'all pairs and exact/specified native geometry and literal text hints; not exhaustive semantic family matching',
            'geometry_signature_limitations': 'Selected native primitives and INSERT placements only, six decimal rounding; text/layers/handles excluded, not geometric or CAD equivalence.',
            'literal_text_limitations': 'DXF/PDF literals are candidate relations; they do not approve PNG readings. Shared names/materials/standards can be incidental.'}


def catalog(audit, pairs, phases, handoff):
    files = validate_phases(audit, pairs, phases)
    native, pdf = (_index(phases[p], 'sha256') for p in ('native', 'pdf'))
    visuals = {r['input']: r for phase in ('controls', 'pdf_pairs') for r in phases[phase]}
    previous = defaultdict(list)
    for row in handoff['entries']:
        previous[row['input_sha256'], row['target_sha256']].append(row)
    entries = []
    for pair in pairs:
        source, target = files[pair['input']], files[pair['target']]
        native_record = native[target['sha256']]
        inherited = previous.get((source['sha256'], target['sha256']), []) if pair['bundle'] == 'train_png' else []
        visual = visuals.get(pair['input'])
        require(visual is not None or inherited, 'A pair has neither current screening nor matched prior evidence')
        entry = {**pair, 'input_sha256': source['sha256'], 'target_sha256': target['sha256'],
                 'native_status': native_record['status'], 'native_resource_status': native_record.get('resource_completeness', 'unavailable'),
                 'native_audit_errors': len(native_record.get('audit_errors', [])),
                 'native_audit_fixes': len(native_record.get('audit_fixes', [])),
                 'excluded_native_target': target['historically_excluded_native_target'],
                 'visual_status': visual['status'] if visual else 'inherited_' + inherited[0]['visual_status'],
                 'execution_errors': visual['errors'] if visual else [],
                 'previous_pair_evidence': [r['pair_report'] for r in inherited],
                 'visual_evidence': None if visual is None else {'phase': 'controls' if pair['declared_split'] != 'train' else 'pdf_pairs',
                                                               'artifacts': visual['artifacts']},
                 'pair_content_approved': False, 'training_eligible': False, 'annotation_approved': False}
        if pair['bundle'] == 'train_pdf':
            p = pdf[source['sha256']]
            entry['pdf_status'] = p['status']
            entry['pdf_page_classes'] = [x['classification'] for x in p.get('pages', [])]
        entry['status'] = ('excluded_native_target' if entry['excluded_native_target'] else
                           'requires_execution_review' if entry['native_status'] != 'ok' or entry['execution_errors'] else
                           'requires_pair_and_semantic_review')
        entries.append(entry)
    relationships = build_relationships(audit, pairs, native, pdf)
    pdf_files = [r for r in files.values() if r['bundle'] == 'train_pdf' and r['role'] == 'input']
    native_files = [r for r in files.values() if r['role'] == 'target']
    summary = {'pair_count': len(entries), 'pairs_by_bundle': dict(Counter(p['bundle'] for p in pairs)),
               'explicit_alias_count': sum(p['basis'] != 'unique_layout_stem' for p in pairs),
               'native_file_count': len(native_files), 'unique_native_hashes': len(native),
               'native_status_by_file': dict(Counter(native[r['sha256']]['status'] for r in native_files)),
               'native_resource_status_by_file': dict(Counter(native[r['sha256']].get('resource_completeness', 'unavailable') for r in native_files)),
               'pdf_file_count': len(pdf_files), 'unique_pdf_hashes': len(pdf),
               'pdf_status_by_file': dict(Counter(pdf[r['sha256']]['status'] for r in pdf_files)),
               'pdf_page_classes_by_file': dict(Counter(page['classification'] for r in pdf_files for page in pdf[r['sha256']].get('pages', []))),
               'visual_status_by_bundle': {b: dict(Counter(r['visual_status'] for r in entries if r['bundle'] == b)) for b in sorted({p['bundle'] for p in pairs})},
               'inherited_pair_reviews': sum(bool(r['previous_pair_evidence']) for r in entries),
               'visual_execution_errors_by_bundle': {b: sum(bool(r['execution_errors']) for r in entries if r['bundle'] == b) for b in sorted({p['bundle'] for p in pairs})},
               'excluded_target_files': sum(r['excluded_native_target'] for r in entries),
               'content_group_count': len(relationships['groups']), 'cross_split_content_groups': len(relationships['cross_split_groups']),
               'weak_cross_split_text_hints': len(relationships['weak_cross_split_text_hints']),
               'training_eligible': False, 'independent_evaluation': False, 'training_started': False}
    return {'schema_version': 'dataset-content-catalog/1.0', 'task': 'P08.9b', 'entries': entries,
            'summary': summary, 'relationships': relationships,
            'limitations': ['Visual verification is a fixed modelspace screen, not full sheet, physical geometry or label approval.',
                            'Quarantine retains failures, incomplete references, other layouts and rendering limitations for the next release.',
                            'Directory split declarations are retained even when conservative relationship conflicts require exclusions before evaluation.',
                            'Missing old sources remain missing; previous reviews are reused only for exact input and target hashes.']}


def gallery(entries, phase_name='dataset-content-v1'):
    rows = ['<!doctype html><html lang="ru"><meta charset="utf-8"><title>Содержимое датасетов</title>',
            '<style>body{font:16px system-ui;max-width:1450px;margin:auto}img{width:32%;vertical-align:top}section{margin:2em 0}pre{white-space:pre-wrap}</style>',
            '<h1>Проверка пар новой структуры</h1><p>Кандидаты и визуальная диагностика; метки и независимость не утверждены.</p>',
            '<label>Набор <select id="bundle" onchange="showRows()"><option value="">Все</option>' + ''.join('<option>'+b+'</option>' for b in sorted({r['bundle'] for r in entries})) + '</select></label>']
    for row in entries:
        rows.append('<section data-bundle="'+html.escape(row['bundle'], quote=True)+'"><h2>'+html.escape(row['input'])+'</h2><pre>'+html.escape(row['target']+'\n'+row['visual_status'])+'</pre>')
        evidence = row['visual_evidence']
        if evidence:
            for image in evidence['artifacts']:
                if image['path'].endswith('.png'):
                    url = '../'+phase_name+'/'+evidence['phase']+'/'+image['path']
                    rows.append('<a href="'+html.escape(url, quote=True)+'"><img loading="lazy" src="'+html.escape(url, quote=True)+'" alt="'+html.escape(Path(image['path']).name)+'"></a>')
        else:
            for record in row['previous_pair_evidence']:
                rows.append('<a href="../'+html.escape(record['path'].removeprefix('artifacts/'), quote=True)+'">Прежняя проверка по тем же SHA</a>')
        rows.append('</section>')
    rows.append('<script>function showRows(){const b=document.getElementById("bundle").value;document.querySelectorAll("section").forEach(e=>e.hidden=!!(b&&e.dataset.bundle!==b));}</script></html>')
    return ''.join(rows).encode('utf-8')


def phase_inventory(base, phase, records):
    directory = base/phase
    manifest = read(directory/'manifest.json')
    expected = {'manifest.json', manifest['runtime']['path'], *(r['path'] for r in manifest['records'])}
    for row in records:
        for artifact in row.get('artifacts', []):
            expected.add(artifact['path'])
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()}
    require(actual == expected, 'Unexpected phase files')


def assemble(workspace, config_pin, phase_dir, output):
    root = Path(workspace).resolve()
    config, audit, pairs = inputs(root, config_pin)
    base = inside(root, phase_dir)
    target = inside(root, output)
    require(base.parent == root/'artifacts' and target.parent == root/'artifacts' and not target.exists(), 'Choose a new direct artifacts directory')
    phases = {phase: load_phase(root, base, phase, config_pin) for phase in PHASES}
    for phase, records in phases.items():
        phase_inventory(base, phase, records)
    handoff = read(checked(root, config['previous_handoff']))
    for row in handoff['entries']:
        for key in ('pair_report', 'preparation', 'run_report'):
            checked(root, row[key])
    result = catalog(audit, pairs, phases, handoff)
    documents = {'catalog.json': encoded(result), 'summary.json': encoded(result['summary']),
                 'relationships.json': encoded(result['relationships']), 'index.html': gallery(result['entries'], base.name)}
    dependencies = [pin(root, (base/p/'manifest.json').relative_to(root).as_posix()) for p in PHASES]
    target.mkdir()
    for name, value in documents.items():
        (target/name).write_bytes(value)
    manifest = {'schema_version': 'dataset-content-review/1.0', 'config': config_pin, 'phases': dependencies,
                'code': [pin(root, name) for name in CODE], 'files': [pin(target, name) for name in sorted(documents)]}
    (target/'manifest.json').write_bytes(encoded(manifest))
    return result['summary']


def verify_review(workspace, manifest_pin):
    root = Path(workspace).resolve()
    path = checked(root, manifest_pin)
    manifest = read(path)
    require(manifest['schema_version'] == 'dataset-content-review/1.0'
            and manifest['code'] == [pin(root, p) for p in CODE], 'Changed review code')
    for p in manifest['phases']:
        checked(root, p)
    config, audit, pairs = inputs(root, manifest['config'])
    phase_paths = {read(inside(root, p['path']))['phase']: inside(root, p['path']).parent.parent for p in manifest['phases']}
    require(set(phase_paths) == set(PHASES) and len(manifest['phases']) == len(PHASES), 'Missing review phases')
    phases = {p: load_phase(root, base, p, manifest['config']) for p, base in phase_paths.items()}
    require(len(set(phase_paths.values())) == 1, 'Phases come from different packages')
    for phase, records in phases.items():
        phase_inventory(phase_paths[phase], phase, records)
    handoff = read(checked(root, config['previous_handoff']))
    for row in handoff['entries']:
        for key in ('pair_report', 'preparation', 'run_report'):
            checked(root, row[key])
    result = catalog(audit, pairs, phases, handoff)
    files = {'catalog.json': encoded(result), 'summary.json': encoded(result['summary']),
             'relationships.json': encoded(result['relationships']), 'index.html': gallery(result['entries'], next(iter(phase_paths.values())).name)}
    require(manifest['files'] == [pin(path.parent, n) for n in sorted(files)]
            and all(inside(path.parent, n).read_bytes() == value for n, value in files.items()), 'Saved content catalog differs from evidence')
    require(sorted(p.name for p in path.parent.iterdir()) == sorted([*files, 'manifest.json']), 'Unexpected review files')
    return result['summary']
