"""Read-only admission audit for partial segmentation and separate VLM targets.

No inference, training, target rewriting or closed-test image access. Package
members are verified selectively, so an OCR manifest never opens its test set.
"""
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image

from .dataset_layout_v1 import checked, encoded, inside, pin, read
from .family_catalog_v2 import require
from .hard_raster_runs_v1 import screen
from .pixel_role_release_v1 import POLICY as MASK_POLICY, dilate, reviewed_masks
from .raster_regions import read_prepared_region
from .segmentation_revision_v1 import training_tiles

PACKAGES = {
    'masks': ('pixel-role-expansion-v3', 'ba7836cfa899633e3291f95c6715fbd9b64418322e2ff14377e4143dbb320746'),
    'unet': ('segmentation-revision-v2', '9e5b8ec7a6bc72407b940092032e5c80704eac1e3a3baf4494afd8cd8f827491'),
    'cv': ('segmentation-revision-assessment-v2', 'c50b6f6defaa7f63c44af40c3336665698131497fa18042c630b6a47eeb929e4'),
    'ocr': ('ocr-current-release-v2', '6f04f9239843c89f35d0b5b5a176a8a137a0f4e0baa91fafbfad0639b8d7cea7'),
    'glyphs': ('engineering-glyph-release-v1', 'b72c5093c6a0fd774d7d33d41b6cc0bc49542e8b5b0aff12958dad26eb0f16d9'),
    'components': ('angular-components-inputs-v1', '9b1080084e4cdbae3275ed8e0273b91b1f1f1469293d159d03a2a0ec372b603b'),
    'angular': ('real-angular-inputs-v1', 'fc1258009841106b8e1097f94183f7e44975e7af580feb4bc33e067b6b8115fb'),
}
METADATA = (
    'artifacts/role-admission-labels-v1-final/groups.json',
    'artifacts/ocr-current-release-v2/source-dispositions.json',
    'artifacts/ocr-control-release-v2/task-splits.json',
    'artifacts/ocr-current-release-v2/task-splits.json',
    'configs/reviews/ocr-fresh-holdout-selection-v1.json',
)
CODE = (
    'src/cad_recovery/architecture_data_readiness_v1.py',
    'scripts/audit_architecture_data_v1.py',
    'src/cad_recovery/dataset_layout_v1.py',
    'src/cad_recovery/family_catalog_v2.py',
    'src/cad_recovery/hard_raster_runs_v1.py',
    'src/cad_recovery/pixel_role_release_v1.py',
    'src/cad_recovery/pixel_role_draft_v1.py',
    'src/cad_recovery/raster_regions.py',
    'src/cad_recovery/segmentation_revision_v1.py',
    'src/cad_recovery/segmentation_experiment_v1.py',
)
POLICY = {
    'task': 'P10.2a.4', 'classes': MASK_POLICY['classes'], 'ignore': 255,
    'mask_recipe': MASK_POLICY, 'maximum_patch_side': 1024,
    'monochrome_stratum': 'nonneutral ink fraction <= 0.01; not proof of scan provenance',
    'known_pixels_only': True, 'validation': 'diagnostic_only_no_model_selection',
    'closed_test_used': False, 'models_executed': False, 'training_started': False,
    'human_verified': False, 'independent_evaluation': False,
    'dense_ground_truth': False, 'physical_geometry_approved': False,
    'production_admitted': False, 'vlm_training_admitted': False,
    'geometry_metrics': 'assigned native ink and reviewed arrow controls only; no physical accuracy',
    'vlm_scope': 'separate OCR literals, glyph diagnostics, component layout and unresolved abstentions',
}


class Package:
    """Authenticate metadata first; verify only explicitly requested members."""

    def __init__(self, root, record):
        self.root = root
        self.record = record
        self.folder = checked(root, record).parent
        self.manifest = read(self.folder / 'manifest.json')
        rows = self.manifest['files']
        self.members = {r['path']: r for r in rows}
        require(len(rows) == len(self.members), 'Duplicate package member')
        self.accessed = set()

    def path(self, name):
        require(name in self.members, 'Unsigned package member: ' + name)
        require(not name.startswith('test') and '/test/' not in name, 'Closed test access forbidden')
        path = checked(self.folder, self.members[name])
        self.accessed.add(name)
        return path

    def json(self, name):
        return read(self.path(name))

    def workspace_image(self, record):
        path = inside(self.root, record['path'])
        name = path.relative_to(self.folder).as_posix()
        require(self.members.get(name) == {'path': name, 'sha256': record['sha256']}, 'Image not in signed package')
        return self.path(name)


def freeze(root, destination):
    require(not destination.exists(), 'Choose a new config')
    config = {'schema_version': 'architecture-data-readiness-config/1.0', 'policy': POLICY,
              'packages': {k: {'path': 'artifacts/' + v[0] + '/manifest.json', 'sha256': v[1]}
                           for k, v in PACKAGES.items()},
              'metadata': [pin(root, p) for p in METADATA], 'code': [pin(root, p) for p in CODE]}
    for record in config['packages'].values(): checked(root, record)
    destination.write_bytes(encoded(config))
    return pin(root, destination.relative_to(root).as_posix())


class Boundary:
    def __init__(self, groups, sources, registries, fresh):
        self.groups, self.registries, self.fresh = groups, registries, fresh
        self.membership = {p: g for g in groups for p in g['members']}
        require(len(self.membership) == sum(len(g['members']) for g in groups), 'Ambiguous source ancestry')
        self.sources = {s['input']: s for s in sources}
        self.by_hash = defaultdict(list)
        for s in sources: self.by_hash[s['input_sha256']].append(s)

    def check(self, source, split, identifier):
        require(split in ('train', 'validation'), 'Only train/diagnostic validation allowed')
        name = source['path'].removeprefix('dataset/')
        require(name in self.membership and name in self.sources, 'Unknown source ancestry')
        require(self.sources[name]['input_sha256'] == source['sha256'], 'Source hash metadata mismatch')
        group = self.membership[name]
        require(not group['cross_split'] and group['declared_splits'] == [split], 'Cross-split source group')
        result = screen([{'id': identifier, 'declared_split': split, 'input': name,
                          'source_sha256': source['sha256']}], self.groups, self.registries, self.fresh)[0]
        return result

    def by_source_hash(self, sha, split, identifier):
        matches = self.by_hash.get(sha, [])
        require(matches, 'Unresolved real source hash')
        results = [self.check({'path': 'dataset/' + s['input'], 'sha256': sha}, split, identifier)
                   for s in matches]
        require(len({r['group_id'] for r in results}) == 1, 'Ambiguous hash ancestry')
        return results[0]


def validate_arrays(rgb, labels, bits, state):
    """Check safety independently of corridor reconstruction, including unknown."""
    require(rgb.dtype == labels.dtype == state.dtype == np.uint8 and bits.dtype.kind in 'ui', 'Wrong mask dtype')
    require(rgb.ndim == 3 and rgb.shape[2] == 3 and rgb.shape[:2] == labels.shape == bits.shape == state.shape,
            'Mismatched pixel grids')
    require(np.isin(labels, [*range(10), 255]).all() and ((bits >= 0) & (bits <= 511)).all() and (state <= 3).all(), 'Invalid mask values')
    gray = np.asarray(Image.fromarray(rgb).convert('L'))
    count = sum(((bits >> i) & 1).astype(np.uint8) for i in range(9))
    require(((state == 3) | (state == np.where(count == 0, 0, np.where(count == 1, 1, 2)))).all(), 'Role bits/state disagree')
    ambiguous = dilate((state == 2) | (state == 3), 2)
    require((labels[ambiguous] == 255).all(), 'Ambiguous/unknown pixels became targets')
    for role in range(1, 10):
        selected = labels == role
        require(((bits[selected] == 1 << (role - 1)) & (state[selected] == 1) & (gray[selected] < 240)).all(),
                'Foreground target is not single-role native ink')
    background = labels == 0
    require(not (background & (dilate(gray < 250, 3) | (bits != 0))).any(), 'Unassigned ink/near-ink became background')
    return {'known_pixels': int((labels != 255).sum()), 'ignore_pixels': int((labels == 255).sum()),
            'unassigned_ink_pixels': int(((gray < 240) & (labels == 255)).sum()),
            'explicit_unknown_pixels': int((state == 3).sum()),
            'multi_role_pixels': int((state == 2).sum()),
            'class_pixels': np.bincount(labels.ravel(), minlength=256)[:10].tolist()}


def native_box(row):
    width, height = row['size_px']
    require(all(type(v) is int and 0 < v <= 1024 for v in (width, height)), 'Unbounded patch')
    matrix = row['patch_to_source']
    x, y = matrix[0][2], matrix[1][2]
    require(type(x) is int and type(y) is int and min(x, y) >= 0, 'Invalid native origin')
    require(matrix == [[1, 0, x], [0, 1, y], [0, 0, 1]] and
            row['source_to_patch'] == [[1, 0, -x], [0, 1, -y], [0, 0, 1]], 'Non-native crop transform')
    box = [x, y, x + width, y + height]
    require(row['review'].get('source_box') in (None, box), 'Review/native box mismatch')
    return box


def overlap_inventory(rows):
    result = []
    for a, b in combinations(rows, 2):
        if a['source']['sha256'] != b['source']['sha256']: continue
        x, y = a['source_box'], b['source_box']
        area = max(0, min(x[2], y[2]) - max(x[0], y[0])) * max(0, min(x[3], y[3]) - max(x[1], y[1]))
        if area: result.append({'a': a['id'], 'b': b['id'], 'overlap_pixels': area,
                                'independent_samples': False})
    return result


def audit_masks(root, package, boundary, unet, cv):
    rows = package.json('records.json')
    require(len(rows) == 25 and len({r['id'] for r in rows}) == 25, 'Changed patch inventory')
    admission = package.json('admission.json')
    require(admission['train'] == 'experimental_partial_supervision_only' and
            admission['validation'] == 'diagnostic_only_no_training_or_model_selection' and
            all(admission[k] is False for k in ('human_verified', 'dense_segmentation_admitted',
                'production_admitted', 'independent_evaluation', 'physical_geometry_approved')), 'Changed target admission')
    for split in ('train', 'validation'):
        import json
        lines = [json.loads(s) for s in package.path(split + '.jsonl').read_text('utf-8').splitlines() if s]
        require(lines == [r for r in rows if r['split'] == split], 'Split JSONL differs from records')
    # Boundary screening of EVERY source finishes before the first image read.
    screened = [boundary.check(r['source'], r['split'], r['id']) for r in rows]
    require(not any(s['blocked'] for s in screened), 'Closed segmentation source/group')
    require(all(s['group_id'] == r['group_id'] for s, r in zip(screened, rows)), 'Group reassignment')
    group_splits, hash_splits = defaultdict(set), defaultdict(set)
    for r in rows:
        group_splits[r['group_id']].add(r['split']); hash_splits[r['source']['sha256']].add(r['split'])
        require(r['task'] == 'partial_pixel_role_supervision' and r['partial_targets'] is True and
                r['training_eligible'] == (r['split'] == 'train') and
                r['diagnostic_only'] == (r['split'] == 'validation') and
                all(r[k] is False for k in ('dense_ground_truth', 'independent_evaluation', 'physical_geometry_approved')),
                'Patch authority/split changed')
    require(all(len(s) == 1 for s in [*group_splits.values(), *hash_splits.values()]), 'Train/validation leakage')
    ids = {r['id'] for r in rows}
    for evaluation in unet['evaluations'].values():
        require(len(evaluation['patches']) == len(ids) and {r['id'] for r in evaluation['patches']} == ids,
                'Saved U-Net pixel scope changed')
    cv_ids = {r['id'] for r in cv['CV']}
    require(len(cv_ids) == len(cv['CV']) and cv_ids <= ids, 'Saved CV scope changed')
    results, arrays, verified_sources = [], {}, set()
    for row in rows:
        box = native_box(row)
        source = boundary.sources[row['source']['path'].removeprefix('dataset/')]
        if row['source']['sha256'] not in verified_sources:
            checked(root, row['source']); verified_sources.add(row['source']['sha256'])
        prep_path = checked(root, source['preparation'])
        require(read(prep_path)['source_sha256'] == row['source']['sha256'], 'Preparation/source mismatch')
        native, proof = read_prepared_region(prep_path.parent, box, max_pixels=1024**2)
        data = {}
        for key in ('image', 'labels', 'role-bits', 'state'):
            with Image.open(package.path(row['paths'][key])) as image:
                require(key != 'image' or image.mode == 'RGB', 'Patch must retain RGB')
                data[key] = np.asarray(image).copy()
        rgb, labels = data['image'], data['labels']
        require(list(rgb.shape[:2][::-1]) == row['size_px'] and np.array_equal(rgb, native), 'Resized/changed native pixels')
        stats = validate_arrays(rgb, labels, data['role-bits'], data['state'])
        rebuilt = reviewed_masks(Image.fromarray(rgb), row['annotations'], row['unknown_regions'], row['review']['background_regions'])
        for key, old in [('labels', 'labels'), ('role-bits', 'bits'), ('state', 'state')]:
            require(np.array_equal(data[key], rebuilt[old]), 'Mask recipe/annotation mismatch: ' + row['id'])
        gray = np.asarray(Image.fromarray(rgb).convert('L')); ink = gray < 240
        chroma = rgb.max(axis=2).astype(int) - rgb.min(axis=2).astype(int) > 20
        fraction = float((chroma & ink).sum() / max(1, ink.sum()))
        strata = 'monochrome' if fraction <= .01 else 'colored'
        quantiles = [np.percentile(gray[labels == c], [10, 50, 90]).tolist() if (labels == c).any() else None for c in range(10)]
        results.append({'id': row['id'], 'split': row['split'], 'source': row['source'], 'group_id': row['group_id'],
            'source_box': box, 'size_px': row['size_px'], 'image': pin(root, package.path(row['paths']['image']).relative_to(root).as_posix()),
            'labels': pin(root, package.path(row['paths']['labels']).relative_to(root).as_posix()),
            'source_to_patch': row['source_to_patch'], 'patch_to_source': row['patch_to_source'],
            'native_pixels_equal': True, 'source_original_rehashed': True,
            'native_tiles_used': proof['used_tile_count'], 'stratum': strata,
            'nonneutral_ink_fraction': fraction, 'class_gray_quantiles_10_50_90': quantiles, **stats,
            'saved_unet_pixel_variants': sorted(unet['evaluations']),
            'saved_cv_variants': sorted(next(r['scores'] for r in cv['CV'] if r['id'] == row['id'])) if row['id'] in cv_ids else [],
            'missing_for_P16_1e': [] if row['id'] in cv_ids else ['original_CV_and_guided_CV_downstream_diagnostics']})
        arrays[row['id']] = (rgb, labels)
    coverage = []
    for split in ('train', 'validation'):
        for stratum in ('all', 'monochrome', 'colored'):
            selected = [r for r in results if r['split'] == split and (stratum == 'all' or r['stratum'] == stratum)]
            for index, name in enumerate(POLICY['classes']):
                support = [r for r in selected if r['class_pixels'][index]]
                coverage.append({'split': split, 'stratum': stratum, 'class_id': index, 'class': name,
                    'pixels': sum(r['class_pixels'][index] for r in selected), 'patches': [r['id'] for r in support],
                    'source_count': len({r['source']['sha256'] for r in support}),
                    'groups': sorted({r['group_id'] for r in support}), 'supported': bool(support)})
    tile_inventory = training_tiles(rows, arrays, True)
    pools = [sum(t['supervised_counts'][0] >= 512 and sum(t['supervised_counts'][1:]) == 0 for t in tile_inventory)]
    pools += [sum(t['supervised_counts'][c] >= 24 for t in tile_inventory) for c in range(1, 10)]
    require(all(pools), 'Missing train sampling class/background pool')
    require(all(r['pixels'] > 0 for r in coverage if r['stratum'] == 'all'), 'Missing split class support')
    return {'rows': results, 'class_coverage': coverage, 'overlapping_ROIs': overlap_inventory(results),
            'boundary': screened, 'train_tiles': len(tile_inventory), 'train_sampling_pool_sizes': pools,
            'source_count': len(verified_sources), 'partial_comparison_admitted': True}


def audit_vlm(root, packages, boundary):
    """Inventory references without combining crops into independent observations."""
    import json
    items = []
    ocr = packages['ocr']
    for split in ('train', 'validation'):
        for row in map(json.loads, ocr.path(split + '.jsonl').read_text('utf-8').splitlines()):
            source = {'path': 'dataset/' + row['source_path'], 'sha256': row['source_sha256']}
            result = boundary.check(source, split, row['id'])
            image = ocr.members[row['image']]
            require(image['sha256'] == row['image_sha256'] and row['split'] == split, 'OCR image/split changed')
            if not result['blocked']: ocr.path(row['image'])
            items.append({'corpus': 'ocr', 'id': row['id'], 'split': split, 'group_id': result['group_id'],
                'original_group_id': row['group_id'], 'source': source, 'field': row['field'],
                'image': {'path': (ocr.folder / row['image']).relative_to(root).as_posix(), 'sha256': image['sha256']},
                'literal': None if result['blocked'] else row['text'], 'exact_target': not result['blocked'],
                'blocked': result['blocked'], 'reasons': result['reasons'],
                'scope': 'excluded_historical_closed_source' if result['blocked'] else 'literal_diagnostic',
                'new_vlm_training_admitted': False})
    glyphs = packages['glyphs']; labels = glyphs.json('labels.json')
    require(labels['training_admission'] == 'not_released', 'Changed glyph training admission')
    contexts = {c['id']: c for c in labels['contexts']}
    for row in labels['glyphs']:
        context = contexts[row['context_id']]; split = row['split']
        require(split == context['split'], 'Glyph/context split mismatch')
        if split in ('train', 'validation'):
            result = boundary.by_source_hash(context['source_sha256'], split, row['id'])
        else:
            require(split in ('development', 'synthetic'), 'Unexpected glyph split')
            synthetic = context['origin'] == 'controlled_font' or context['group_id'] == 'procedural-nontext'
            require((split == 'synthetic') == synthetic, 'Synthetic/real confusion')
            if split == 'development':
                require(context['source_sha256'] ==
                        'd3baaca90795c9ccb85e3bcfa71245bb2182996c74132bc410e3ce2653e4ca14', 'Unknown development glyph source')
            result = {'group_id': row['group_id'], 'blocked': False, 'reasons': []}
        nontext = row['proposal_method'] == 'controlled_geometry' and row['class'] in ('circle_geometry', 'crossed_geometry')
        exact = row['state'] == 'accepted_diagnostic' and not nontext
        require((exact and isinstance(row['literal'], str)) or (nontext and row['literal'] is None) or
                (row['state'] == 'unresolved' and row['literal'] is None and row['class'] is None), 'Guessed unresolved glyph')
        if not result['blocked']:
            glyphs.path(row['image']['path']); glyphs.path(context['released_image']['path'])
        items.append({'corpus': 'glyph', 'id': row['id'], 'context_id': row['context_id'], 'split': split,
            'group_id': result['group_id'], 'original_group_id': row['group_id'],
            'source_sha256': context.get('source_sha256'), 'image': {'path': (glyphs.folder / row['image']['path']).relative_to(root).as_posix(),
                'sha256': row['image']['sha256']}, 'literal': row['literal'], 'class': row['class'],
            'field': row['semantic_role_context'], 'exact_target': exact,
            'scope': 'nontext_geometry_negative' if nontext else ('controlled_font_only' if split == 'synthetic' else
                     ('literal_diagnostic' if exact else 'abstention_only')),
            'blocked': result['blocked'], 'reasons': result['reasons'], 'new_vlm_training_admitted': False})
    for key in ('components', 'angular'):
        package = packages[key]
        for row in package.json('cases.json'):
            split = row['split']
            if split in ('train', 'validation'):
                result = boundary.check(row['source'], split, row['id'])
            else:
                require(split == 'development' and row['source']['sha256'] ==
                        'd3baaca90795c9ccb85e3bcfa71245bb2182996c74132bc410e3ce2653e4ca14', 'Unknown development source')
                result = {'group_id': row['group_id'], 'blocked': False, 'reasons': []}
            require(row['physical_geometry_approved'] is False and row['training_eligible'] is False,
                    'Component diagnostics cannot approve physical/training targets')
            # Components reference the context in the angular package.
            if not result['blocked']: packages['angular'].workspace_image(row['image'])
            observations = row['components'] if key == 'components' else [row]
            for observation in observations:
                exact = observation.get('reading_status') == 'accepted_diagnostic' if key == 'components' else False
                if key == 'components' and not result['blocked']: package.workspace_image(observation['image'])
                items.append({'corpus': key, 'id': row['id'] + '/' + observation['id'] if key == 'components' else row['id'],
                    'context_id': row['id'], 'source_box': row['source_box'], 'context_to_source': row['context_to_source'],
                    'label_record': {'file': pin(root, package.path('cases.json').relative_to(root).as_posix()), 'case_id': row['id']},
                    'split': split, 'group_id': result['group_id'], 'source': row['source'],
                    'image': observation['image'], 'literal': observation.get('reference') if exact else None,
                    'exact_target': exact, 'scope': ('component_literal_and_layout' if exact else 'abstention_only')
                        if key == 'components' else 'pixel_geometry_candidate_diagnostic_only',
                    'expected_edges': row.get('expected_edges', []),
                    'expected_points_px': row.get('expected_points_px'), 'geometry_state': row.get('reviewed_geometry_state'),
                    'blocked': result['blocked'], 'reasons': result['reasons'],
                    'new_vlm_training_admitted': False})
    duplicates = defaultdict(list)
    for row in items: duplicates[row['image']['sha256']].append([row['corpus'], row['id']])
    fields = {s: dict(Counter(r['field'] for r in items if r['corpus'] == 'ocr' and r['split'] == s and not r['blocked']))
              for s in ('train', 'validation')}
    return {'schema_version': 'architecture-vlm-inventory/1.0', 'items': items,
        'counts_by_corpus_split': dict(Counter(r['corpus'] + '/' + r['split'] for r in items)),
        'usable_counts_by_corpus_split': dict(Counter(r['corpus'] + '/' + r['split'] for r in items if not r['blocked'])),
        'blocked_counts_by_corpus_split': dict(Counter(r['corpus'] + '/' + r['split'] for r in items if r['blocked'])),
        'blocked_count': sum(r['blocked'] for r in items), 'ocr_fields': fields,
        'identical_image_groups': [v for v in duplicates.values() if len(v) > 1],
        'independent_evaluation': False, 'new_vlm_training_admitted': False,
        'count_warning': 'Glyphs/components/lines overlap in source and context. Counts must not be summed as independent cases.',
        'gaps': ['No independent new difficult-scan evaluation set.',
                 'No confirmed material/person validation fields in this OCR release.',
                 'Unknown isolated glyphs remain abstention controls, never exact literals.',
                 'No candidate-ID VLM selection contract yet; geometry truth does not guarantee candidate availability.',
                 'Freeze VLM model/license/resources/prompt/abstention policy before evaluation in P16.4a.']}


def build(root, config_pin):
    config = read(checked(root, config_pin))
    require(config['schema_version'] == 'architecture-data-readiness-config/1.0' and config['policy'] == POLICY,
            'Changed audit policy')
    require(config['code'] == [pin(root, p) for p in CODE] and
            config['metadata'] == [pin(root, p) for p in METADATA], 'Changed audit implementation/metadata')
    expected = {k: {'path': 'artifacts/' + v[0] + '/manifest.json', 'sha256': v[1]} for k, v in PACKAGES.items()}
    require(config['packages'] == expected, 'Changed upstream releases')
    packages = {k: Package(root, p) for k, p in config['packages'].items()}
    groups, sources, registry1, registry2, fresh = [read(checked(root, p)) for p in config['metadata']]
    fresh_sources = [{'source_path': s['input'], 'source_sha256': s['input_sha256']} for s in fresh['selected_sources']]
    boundary = Boundary(groups['groups'], sources, [registry1, registry2], fresh_sources)
    for key in ('unet', 'cv'):
        protocol = read(checked(root, packages[key].manifest['config']))
        require(protocol['labels'] == expected['masks'], 'Saved baseline used different targets')
    unet = packages['unet'].json('summary.json'); cv = packages['cv'].json('report.json')
    require(unet['closed_test_used'] is False and cv['closed_test_used'] is False and
            unet['checkpoint_selected_before_validation'] is True, 'Invalid saved baseline selection')
    masks = audit_masks(root, packages['masks'], boundary, unet, cv)
    vlm = audit_vlm(root, packages, boundary)
    report = {'schema_version': 'architecture-data-readiness/1.0', 'policy': POLICY,
        'patch_counts': dict(Counter(r['split'] for r in masks['rows'])), 'source_count': masks['source_count'],
        'groups_by_split': {s: sorted({r['group_id'] for r in masks['rows'] if r['split'] == s}) for s in ('train', 'validation')},
        'strata_counts': dict(Counter(r['split'] + '/' + r['stratum'] for r in masks['rows'])),
        'train_tiles': masks['train_tiles'], 'train_sampling_pool_sizes': masks['train_sampling_pool_sizes'],
        'known_pixels': sum(r['known_pixels'] for r in masks['rows']),
        'ignore_pixels': sum(r['ignore_pixels'] for r in masks['rows']),
        'overlapping_ROIs': masks['overlapping_ROIs'], 'boundary': masks['boundary'],
        'saved_unet_pixel_ROIs': len(masks['rows']),
        'saved_cv_ROIs': sum(bool(r['saved_cv_variants']) for r in masks['rows']),
        'missing_downstream_ROIs': [r['id'] for r in masks['rows'] if r['missing_for_P16_1e']],
        'minimum_new_annotation_for_partial_experiment': 0,
        'limitations': ['Partial assistant-reviewed corridors and background only; no dense view/page truth.',
            'Disjoint conservative groups are not confirmed independent semantic families.',
            'Color/brightness may encode roles. Grayscale alone is not a leakage control.',
            'Monochrome stratum does not confirm physical scan provenance.',
            'Original source bytes rehashed; crop equality checked against pinned native preparation tiles, not a new full PNG decode.',
            'CV has no semantic role labels: compare downstream assigned-ink/arrow diagnostics, not invented role IoU.',
            'Missing classes within a stratum are unsupported, never zero error.',
            'No independent quality conclusion, physical accuracy or production admission.']}
    admission = {'schema_version': 'architecture-data-admission/1.0', 'task': 'P10.2a.4',
        'partial_CV_UNet_SegFormer_comparison_admitted': masks['partial_comparison_admitted'],
        'train': 'experimental_partial_supervision_only', 'validation': 'diagnostic_only_no_training_or_model_selection',
        'dense_segmentation_admitted': False, 'region_training_admitted': False, 'arrow_training_admitted': False,
        'VLM_inventory_ready': True, 'new_VLM_training_admitted': False,
        'physical_geometry_approved': False, 'independent_evaluation': False, 'production_admitted': False,
        'next_task': 'P16.1e', 'next_protocol': ['Reuse verified cached measurements; execute only missing/incompatible baselines.',
            'Freeze shared ROI/coordinates/masks and downstream metric definitions before new inference.',
            'Fix primary variant and final update before diagnostic validation; no checkpoint/parameter selection.',
            'Report per-class and per-stratum support; ignore=255 is never background.',
            'SegFormer and VLM licenses, weights and resource qualification are still future work.']}
    metrics = {'pixel_roles': 'known pixels only; 10x10 confusion, per-class IoU; foreground macro excludes unsupported roles',
               'arrow_controls': pin(root, packages['masks'].path('arrow-controls.json').relative_to(root).as_posix()),
               'background_false_foreground': 'only reviewed label=0 denominator',
               'thin_islands': 'reviewed components >=24 pixels, extent >=16; radius <=4; connected retention >=0.8 (existing acceptance protocol)',
               'CV_downstream': 'same native RGB, reviewed role ink coverage band=3 and visible arrow/negative controls; not full CAD topology',
               'OCR_VLM': 'separate literal CER/exact, sign confusion, unknown abstention and false corrections; no mask-as-text targets',
               'resources': 'new model/adapter must measure time and host/device memory; old U-Net GPU qualification does not transfer'}
    access = {k: {'manifest': p.record, 'members_read': sorted(p.accessed)} for k, p in packages.items()}
    return {'report.json': report, 'admission.json': admission,
            'roi-matrix.json': {'metrics': metrics, 'rows': masks['rows']},
            'class-coverage.json': masks['class_coverage'], 'vlm-inventory.json': vlm,
            'access-ledger.json': access}


def write(root, config_pin, destination):
    require(not destination.exists(), 'Choose a new output directory')
    outputs = build(root, config_pin)
    destination.mkdir(parents=True)
    for name, value in outputs.items(): (destination / name).write_bytes(encoded(value))
    manifest = {'schema_version': 'architecture-data-readiness-manifest/1.0', 'config': config_pin,
                'files': [pin(destination, name) for name in sorted(outputs)]}
    (destination / 'manifest.json').write_bytes(encoded(manifest))
    return outputs['report.json']


def verify(root, config_pin, destination):
    manifest = read(destination / 'manifest.json')
    require(manifest['schema_version'] == 'architecture-data-readiness-manifest/1.0' and manifest['config'] == config_pin,
            'Changed result protocol')
    expected = build(root, config_pin)
    require({r['path'] for r in manifest['files']} == set(expected), 'Changed result inventory')
    for record in manifest['files']:
        require(checked(destination, record).read_bytes() == encoded(expected[record['path']]), 'Replay differs')
    return {'verified': True, 'replayed_files': len(expected), 'models_executed': False, 'closed_test_used': False}
