"""Deterministic raster review intake; this package cannot supply training labels."""
from collections import Counter
from html import escape
from io import BytesIO
from pathlib import Path

from PIL import Image

from .dataset_layout_v1 import checked, encoded, inside, pin, read
from .dataset_preparation_v1 import crop_source
from .family_catalog_v2 import require
from .source_coallocation_v1 import load_overlay, package

CODE = ('src/cad_recovery/role_source_expansion_v1.py',
        'scripts/prepare_role_source_expansion_v1.py',
        'src/cad_recovery/dataset_preparation_v1.py',
        'src/cad_recovery/source_coallocation_v1.py',
        'src/cad_recovery/dataset_layout_v1.py')
FALSE_FLAGS = ('human_verified', 'training_eligible', 'physical_geometry_approved',
               'coordinate_regression_approved', 'independent_evaluation', 'test_used')
ROLES = {'chamfer_size', 'flange_width', 'hole_diameter', 'drill_depth', 'plate_thickness'}
STATES = {'reviewed_candidate', 'unsupported_role_taxonomy', 'absent', 'deferred'}
SELECTION = ('min-source-SHA representative of each eligible group, first four by SHA '
             'per train/validation; no readability replacement')


def select_sources(overlay, sources, catalog, excluded_groups):
    """Choose representatives before inspecting text; never rebalance for readability."""
    membership = {}
    for group in overlay['groups']:
        for name in group['members']:
            require(name not in membership, 'Repeated source group membership')
            membership[name] = group
    entries = {c['input']: c for c in catalog['entries']}
    require(len(entries) == len(catalog['entries']), 'Repeated catalog input')
    require(len({s['input'] for s in sources}) == len(sources), 'Repeated source input')
    pools = {'train': {}, 'validation': {}}
    for source in sources:
        group = membership[source['input']]
        split = source['declared_split']
        if (source['source_kind'] != 'png' or split not in pools
                or group['id'] in excluded_groups or group['cross_split']
                or group['declared_splits'] != [split]
                or entries[source['input']]['visual_status'] not in ('verified', 'inherited_verified')):
            continue
        pool = pools[split]
        key = (source['input_sha256'], source['input'])
        old = pool.get(group['id'])
        if old is None or key < (old['input_sha256'], old['input']):
            pool[group['id']] = source
    selected = []
    for split, pool in pools.items():
        require(len(pool) >= 4, 'Insufficient distinct source groups')
        for index, source in enumerate(sorted(pool.values(), key=lambda s: (s['input_sha256'], s['input']))[:4], 1):
            selected.append({'id': f'{split}-{index}', 'declared_split': split,
                'group_id': membership[source['input']]['id'], 'source_record': source,
                **{k: source[k] for k in ('input', 'input_sha256', 'target', 'target_sha256',
                                         'preparation', 'source_size_px')},
                'overview': next(p for p in source['prepared_assets'] if p['path'].endswith('/overview.png'))})
    return selected, {split: len(pool) for split, pool in pools.items()}


def valid_box(box, size):
    return (isinstance(box, list) and len(box) == 4 and all(type(v) is int for v in box)
            and 0 <= box[0] < box[2] <= size[0] and 0 <= box[1] < box[3] <= size[1]
            and (box[2]-box[0])*(box[3]-box[1]) <= 16_777_216)


def observations(row):
    require(isinstance(row['observations'], list) and len(row['observations']) >= 2
            and all(isinstance(s, str) and len(s.strip()) >= 10 for s in row['observations']),
            'Missing explicit raster observations')


def validate_reading(selected, reading):
    require(reading['schema_version'] == 'role-source-expansion-reading/1.0'
            and reading['reviewer'] == 'assistant', 'Unknown raster review authority')
    require(all(reading[f] is False for f in FALSE_FLAGS), 'Intake cannot approve training or independence')
    sources = {s['id']: s for s in selected}
    assessments = reading['source_assessments']
    require(len(assessments) == len(sources) and {a['source_id'] for a in assessments} == set(sources),
            'Every selected source needs one assessment')
    kinds = {}
    for assessment in assessments:
        observations(assessment)
        require(assessment['drawing_kind'] in ('component_drawing', 'purchased_items_list', 'assembly_drawing'),
                'Unknown source kind')
        kinds[assessment['source_id']] = assessment['drawing_kind']
    contexts = {c['id']: c for c in reading['contexts']}
    require(len(contexts) == len(reading['contexts']), 'Repeated context')
    for c in contexts.values():
        require(c['source_id'] in sources and valid_box(c['source_box'], sources[c['source_id']]['source_size_px']),
                'Invalid source context box')
        require(c['id'].startswith(c['source_id']), 'Context identity differs from source')
    slots = reading['slots']
    expected = {s+'-'+str(n) for s in sources for n in (1, 2)}
    require(len(slots) == len(expected) and {r['id'] for r in slots} == expected, 'Exactly two slots per source required')
    used = set()
    for row in slots:
        source = row['source_id']
        require(source in sources and row['id'] in (source+'-1', source+'-2'), 'Changed slot source')
        require(row['training_eligible'] is False and row['human_verified'] is False, 'Unapproved slot authority')
        require(row['state'] in STATES and isinstance(row['reason'], str) and row['reason'].strip(), 'Unknown slot outcome')
        observations(row)
        if row['state'] in ('absent', 'deferred'):
            require(row['context_id'] is None and row['query_source_box'] is None
                    and row['candidate_role'] is None and row['transcription_lines'] == []
                    and row['rotation_cw'] == 0, 'Excluded slot cannot carry a label or image')
            require((kinds[source], row['state']) in (('purchased_items_list', 'absent'), ('assembly_drawing', 'deferred')),
                    'Exclusion differs from source assessment')
            continue
        require(kinds[source] == 'component_drawing', 'Part role requires component drawing')
        require(row['context_id'] in contexts, 'Missing query context')
        c = contexts[row['context_id']]
        box = row['query_source_box']
        require(c['source_id'] == source and valid_box(box, sources[source]['source_size_px']), 'Invalid query source box')
        require(c['source_box'][0] <= box[0] < box[2] <= c['source_box'][2]
                and c['source_box'][1] <= box[1] < box[3] <= c['source_box'][3], 'Query outside context')
        require(type(row['rotation_cw']) is int and row['rotation_cw'] in (0, 90), 'Unsupported query rotation')
        lines = row['transcription_lines']
        require(isinstance(lines, list) and 1 <= len(lines) <= 2
                and all(isinstance(s, str) and s.strip() and '\n' not in s and '\r' not in s for s in lines), 'Invalid visible transcription')
        require((row['state'] == 'reviewed_candidate' and row['candidate_role'] in ROLES)
                or (row['state'] == 'unsupported_role_taxonomy' and row['candidate_role'] is None), 'Unsupported role assignment')
        used.add(c['id'])
    require(used == set(contexts), 'Unreviewed context')
    for hypothesis in reading['family_hypotheses']:
        ids = hypothesis['source_ids']
        require(len(ids) >= 2 and len(ids) == len(set(ids)) and set(ids) <= set(sources)
                and hypothesis['status'] == 'unresolved', 'Intake cannot approve family independence')
        observations(hypothesis)


def context(root, config_pin):
    config = read(checked(root, config_pin))
    require(config['schema_version'] == 'role-source-expansion-config/1.0'
            and config['task'] == 'P14.1b.10a', 'Unknown intake configuration')
    require(all(config[f] is False for f in FALSE_FLAGS), 'Changed intake scope')
    require(config['code'] == [pin(root, name) for name in CODE], 'Intake code changed')
    selection = read(checked(root, config['selection']))
    require(selection['schema_version'] == 'role-source-expansion-selection/1.0'
            and selection['task'] == config['task'] and selection['selection'] == SELECTION, 'Changed selection protocol')
    require(all(selection[f] is False for f in ('training_eligible', 'independent_evaluation', 'test_used')), 'Changed selection scope')
    overlay = load_overlay(root, selection['source_overlay'])
    parent, _ = package(root, selection['parent_role_manifest'], 'semantic-role-manifest/2.0')
    groups = sorted({r['group_id'] for r in read(parent/'labels.json')['samples']})
    require(selection['excluded_current_role_groups'] == groups, 'Current role groups must be wholly excluded')
    selected, counts = select_sources(overlay, read(checked(root, selection['source_dispositions'])),
                                      read(checked(root, selection['catalog'])), groups)
    require(selection['sources'] == selected and selection['group_candidate_counts'] == counts, 'Source selection differs from frozen order')
    reading = read(checked(root, config['reading']))
    validate_reading(selected, reading)
    return config, selection, reading


def png(image):
    stream = BytesIO()
    image.save(stream, format='PNG')
    return stream.getvalue()


def materialize(root, config_pin):
    config, selection, reading = context(root, config_pin)
    sources = {s['id']: s for s in selection['sources']}
    assets = {'selection.json': encoded(selection)}
    images, published_contexts = {}, []
    for source in sources.values():
        assets['overviews/'+source['id']+'.png'] = checked(root, source['overview']).read_bytes()
    for c in reading['contexts']:
        pixels, provenance = crop_source(root, sources[c['source_id']]['source_record'], c['source_box'])
        im = Image.fromarray(pixels)
        images[c['id']] = im
        name = 'contexts/'+c['id']+'.png'
        assets[name] = png(im)
        published_contexts.append({**c, 'image': name, 'crop_provenance': provenance})
    published_slots = []
    for row in reading['slots']:
        name = None
        if row['context_id'] is not None:
            c = next(c for c in reading['contexts'] if c['id'] == row['context_id'])
            x, y = c['source_box'][:2]
            b = row['query_source_box']
            im = images[c['id']].crop((b[0]-x, b[1]-y, b[2]-x, b[3]-y))
            if row['rotation_cw'] == 90:
                im = im.transpose(Image.Transpose.ROTATE_270)
            name = 'queries/'+row['id']+'.png'
            assets[name] = png(im)
        published_slots.append({**row, 'query_image': name})
    report = {'schema_version': 'role-source-expansion-report/1.0', 'config': config_pin,
        'sources_by_split': dict(Counter(s['declared_split'] for s in sources.values())),
        'slot_outcomes': dict(Counter(r['state'] for r in published_slots)),
        'candidate_roles_by_split': dict(Counter(sources[r['source_id']]['declared_split'] for r in published_slots if r['candidate_role'])),
        'contexts': len(published_contexts), 'queries': sum(r['query_image'] is not None for r in published_slots),
        'training_labels_released': 0, **{f: False for f in FALSE_FLAGS}}
    assets['review.json'] = encoded({**reading, 'selection': config['selection'],
        'contexts': published_contexts, 'slots': published_slots})
    assets['report.json'] = encoded(report)
    cards = []
    for s in sources.values():
        assessment = next(a for a in reading['source_assessments'] if a['source_id'] == s['id'])
        cards.append('<section><h2>'+escape(s['id']+' — '+s['input'])+'</h2><p>'+escape(assessment['drawing_kind'])
                     +'</p><img class="overview" src="overviews/'+s['id']+'.png" alt="Source overview">')
        cards.extend('<p>'+escape(o)+'</p>' for o in assessment['observations'])
        for row in (r for r in published_slots if r['source_id'] == s['id']):
            cards.append('<h3>'+escape(row['id']+' — '+row['state'])+'</h3><p>'+escape(row['reason'])+'</p>')
            if row['query_image']:
                cards.append('<p>'+escape(' / '.join(row['transcription_lines']))+' | '+escape(row['candidate_role'] or 'UNSUPPORTED')+'</p>'
                    '<img class="query" src="'+row['query_image']+'" alt="Query"><p><a href="contexts/'+row['context_id']+'.png">Context</a></p>')
            cards.extend('<p>'+escape(o)+'</p>' for o in row['observations'])
        cards.append('</section>')
    assets['index.html'] = ('<!doctype html><meta charset="utf-8"><title>Additional source review</title>'
        '<style>body{font:16px sans-serif;margin:2rem}section{border-top:1px solid #aaa;padding:1rem}.overview{max-width:100%;max-height:550px}.query{max-width:100%;max-height:200px}</style>'
        '<h1>P14.1b.10a — raster candidates</h1><p>No training release. Family independence and native role admission are pending.</p>'
        '<pre>'+escape(str(report))+'</pre>'+''.join(cards)).encode('utf-8')
    return assets, report


def build(workspace, config_pin, output):
    root = Path(workspace).resolve()
    target = inside(root, output)
    require(not target.exists() and not target.resolve().is_relative_to((root/'dataset').resolve()), 'Choose a fresh output outside dataset')
    assets, report = materialize(root, config_pin)
    target.mkdir(parents=True, exist_ok=False)
    for name, data in assets.items():
        path = inside(target, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(data)
    with (target/'manifest.json').open('xb') as stream:
        stream.write(encoded({'schema_version': 'role-source-expansion-manifest/1.0', 'config': config_pin,
                              'files': [pin(target, name) for name in sorted(assets)]}))
    return {'manifest': pin(root, (target/'manifest.json').relative_to(root).as_posix()), 'report': report}


def verify(workspace, manifest_pin):
    root = Path(workspace).resolve()
    folder, manifest = package(root, manifest_pin, 'role-source-expansion-manifest/1.0')
    assets, report = materialize(root, manifest['config'])
    require({r['path'] for r in manifest['files']} == set(assets), 'Intake inventory differs')
    for name, data in assets.items():
        require((folder/name).read_bytes() == data, 'Intake differs from reviewed pixels/recipe: '+name)
    return report
