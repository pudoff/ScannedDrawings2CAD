"""Prepare source-screened CV controls without executing inference or converters."""
from collections import Counter, defaultdict
import hashlib
import html
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .architecture_data_readiness_v1 import Boundary, Package, native_box, validate_arrays
from .cv_comparison_fixtures_v1 import FAMILIES, scene
from .dataset_layout_v1 import checked, encoded, pin, read
from .family_catalog_v2 import require
from .page_region_review_v1 import READINESS, PARENT
from .pixel_role_release_v1 import POLICY as MASK_POLICY
from .raster_regions import read_prepared_region

PREVIOUS = 'artifacts/e2e-acceptance-release-v1.json'
PREVIOUS_SHA = '1f4633bcc460f6aa587243940b15201abf13888880b7e7992979bfba9d7e2667'
CODE = ['src/cad_recovery/cv_comparison_protocol_v1.py', 'src/cad_recovery/cv_comparison_fixtures_v1.py',
        'scripts/prepare_cv_comparison_v1.py', 'src/cad_recovery/geometry_detection.py',
        'src/cad_recovery/circle_search_v2.py', 'src/cad_recovery/arrow_detection.py',
        'src/cad_recovery/connector_guard_v1.py', 'src/cad_recovery/architecture_data_readiness_v1.py',
        'src/cad_recovery/raster_regions.py', 'src/cad_recovery/pixel_role_release_v1.py',
        'src/cad_recovery/stroke_curve_control_v1.py', 'parallel_project/pipeline.py']

POLICY = {
    'task': 'P15.2f.2', 'version': 'cv-comparison-protocol/1.0',
    'models_executed': False, 'training': False, 'closed_test_used': False,
    'reference_dxf_used': False, 'human_verified': False, 'independent_evaluation': False,
    'physical_geometry_approved': False, 'production_admitted': False,
    'split_policy': 'source family before crops/scale/width derivatives; validation diagnostic only',
    'parameter_selection': 'train families only; freeze configuration SHA before validation',
    'coordinate_system': 'original crop pixel centers; x right y down; no independent bbox alignment',
    'common_inputs': 'identical gray PNG bytes and common partial text-mask bytes for both routes',
    'text_modes': ['untouched', 'common_reviewed_text_mask'],
    'text_mask_authority': 'oracle partial localization diagnostic; not OCR or deployable segmentation',
    'real_unknown': 'unassigned ink is unknown, never a negative or precise geometry target',
    'geometry_modes': ['native_crop', 'half_resolution_crop'],
    'large_sheets': 'native source crops through authenticated tiles; no full-sheet inference in this protocol',
    'millimeters': 'not_evaluable_mm until independently confirmed pixel-to-model transform',
    'baseline': {'geometry': 'geometry_detection.detect_geometry(search_profile=regional_masked)',
        'arrows': 'current arrow_detection; connector_guard_v1 reference retained separately',
        'model_units': 'pixels; no A4/DPI/nominal-derived scale'},
    'parallel_reference': {'functions': ['detect_arrows', 'stage7_lines'],
        'execution': 'copy pinned pipeline.py into per-attempt directory; set private WORK; child process only',
        'forbidden': ['run', 'stage2_calibrate', 'ocr_all_boxes', 'stage11_dxf', 'stage12_qc'],
        'fixed_pixel_parameters': 'raw reference baseline only; scale sensitivity must be reported',
        'automatic_role_classifier': 'not admitted: requires separate adaptation of A4/scale rules in P10.2c.3b'},
    'budgets': {'maximum_variants_per_experiment': 4, 'maximum_case_seconds': 60,
        'maximum_case_rss_bytes': 1073741824, 'maximum_case_pixels': 4194304,
        'maximum_child_output_bytes': 64*1024**2, 'cpu_threads': 1,
        'maximum_total_cpu_minutes_per_experiment': 30, 'repeats': 2,
        'timeouts': 'failure, no zero-score or success substitution'},
    'metrics': {
        'matching': 'one-to-one type-aware maximum matching; fragmented and duplicate coverage reported separately',
        'geometry': ['symmetric centerline distance p50/p95/max', 'endpoint error', 'radius and swept-angle error',
                     'missed branches', 'false bridges', 'duplicate lines', 'wrong primitive types'],
        'arrows': ['tip error in native px', 'signed direction error degrees', 'scoped precision/recall'],
        'roles': 'confusion matrix and macro F1 on single-role reviewed ink only; coverage/abstentions separately',
        'topology': ['holes and nesting', 'connected components', 'intentional gaps', 'T/X degree'],
        'separate_stages': ['extraction', 'simplification', 'DXF roundtrip'],
        'raster_overlap': 'auxiliary only; cannot replace geometry or topology',
        'real_partial_guides': 'corridor coverage only; no exact endpoint/radius GT inferred from role annotations',
        'aggregation': 'per family and stratum first; crops/scales/width variants not independent samples'},
    'gates_before_predictions': {
        'analytical_centerline_p95_px': 2.0, 'analytical_endpoint_px': 3.0,
        'analytical_radius_px': 2.0, 'analytical_sweep_deg': 5.0,
        'real_directed_tip_tolerance_native_px': 8, 'direction_tolerance_deg': 20,
        'synthetic_geometry_precision_min': 0.98, 'synthetic_role_macro_f1_min': 0.90,
        'maximum_new_false_bridges': 0, 'maximum_lost_holes_or_branches': 0,
        'maximum_removed_protected_or_unknown_pixels': 0,
        'maximum_scoped_real_arrow_regressions': 0,
        'minimum_family_macro_error_relative_improvement': 0.10,
        'alternative_minimum_recall_absolute_improvement': 0.02,
        'maximum_latency_ratio_to_baseline': 1.5,
        'acceptance': 'all applicable no-regression/resource gates plus error or recall improvement; scope by experiment',
        'zero_denominator': 'not_evaluable, never fabricated 0 or 100%',
        'authority': 'development decision thresholds, not customer physical tolerances'},
}


def boundary(root):
    config = read(checked(root, READINESS))
    groups, sources, a, b, fresh = [read(checked(root, p)) for p in config['metadata']]
    return config, Boundary(groups['groups'], sources, [a, b],
        [{'source_path': s['input'], 'source_sha256': s['input_sha256']} for s in fresh['selected_sources']])


def validate_split_groups(rows):
    groups = defaultdict(set)
    for row in rows:
        require(row['split'] in ('train', 'validation'), 'Closed or invalid split')
        groups[row['group_id']].add(row['split'])
    require(all(len(v) == 1 for v in groups.values()), 'Family leakage across splits')


def parallel_screen(root, guard):
    result = []
    for path in sorted((root/'parallel_project').glob('*.png')):
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        hits = guard.by_hash.get(sha, [])
        item = {'alias': pin(root, path.relative_to(root).as_posix()),
                'matches': [{'path': 'dataset/'+s['input'], 'sha256': sha,
                             'declared_split': s['declared_split']} for s in hits]}
        if not hits:
            item.update(disposition='excluded', reason='unknown source ancestry')
        else:
            try:
                screenings = [guard.check({'path': 'dataset/'+s['input'], 'sha256': sha}, s['declared_split'], path.stem) for s in hits]
                require(len({s['group_id'] for s in screenings}) == 1, 'Ambiguous alias family')
                item.update(screening=screenings, group_id=screenings[0]['group_id'],
                    disposition='excluded' if any(s['blocked'] for s in screenings) else 'viewed_train_development_only',
                    geometry_training_eligible=False, pair_approved=all(s.get('family_approved') for s in hits))
            except ValueError as exc:
                item.update(disposition='excluded', reason=str(exc))
        result.append(item)
    return result


def write(folder, name, data):
    path = folder/name; path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(data if isinstance(data, bytes) else encoded(data))


def png(folder, name, array):
    path = folder/name; path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        Image.fromarray(array).save(stream, format='PNG')


def variants(folder, identifier, gray, text_mask):
    result = []
    height, width = gray.shape
    for mode, factor in [('native_crop', 1), ('half_resolution_crop', .5)]:
        size = [max(1, int(width*factor)), max(1, int(height*factor))]
        sx, sy = size[0]/width, size[1]/height
        require(sx == sy, 'Comparison resize must be isotropic')
        if factor == 1:
            image, mask = gray, text_mask
        else:
            image = np.asarray(Image.fromarray(gray).resize(size, Image.Resampling.BOX))
            # Conservative OR pooling; small protected glyph pixels must not disappear.
            mask = np.asarray(Image.fromarray(text_mask).resize(size, Image.Resampling.BOX)) > 0
            mask = mask.astype(np.uint8)*255
        base = f'cases/{identifier}/{mode}'
        png(folder, base+'/gray.png', image); png(folder, base+'/text-mask.png', mask)
        matrix = [[sx, 0, (sx-1)/2], [0, sy, (sy-1)/2], [0, 0, 1]]
        result.append({'mode': mode, 'size_px': size, 'gray': base+'/gray.png',
            'common_text_mask': base+'/text-mask.png', 'native_to_input': matrix,
            'input_to_native': np.linalg.inv(matrix).tolist(), 'artificial_upscale': False})
    return result


def reviewed_aliases(root, folder, cases, reviewed):
    require(reviewed.get('human_verified') is False and reviewed.get('physical_geometry_approved') is False,
            'Alias reading cannot grant physical or customer approval')
    by_id = {r['id']: r for r in reviewed['cases']}
    require(set(by_id) == {'parallel-star-top', 'parallel-star-stamp'}, 'Missing or duplicate alias review')
    require(len(reviewed['cases']) == len(by_id), 'Duplicate alias reading')
    result = []
    for case in cases:
        if case['id'] not in by_id:
            result.append(case); continue
        row = by_id[case['id']]
        require(row['source_sha256'] == case['source']['sha256'] and row['source_box'] == case['source_box'],
                'Alias reading source/crop rebound')
        require(row['image'] == pin(root, (folder/case['variants'][0]['gray']).relative_to(root).as_posix()),
                'Alias reading pixels changed')
        with Image.open(checked(root, row['image'])) as im:
            gray = np.asarray(im).copy()
        def valid_point(point):
            return (isinstance(point, list) and len(point) == 2 and np.isfinite(point).all()
                    and 0 <= point[0] < gray.shape[1] and 0 <= point[1] < gray.shape[0])
        controls = row['arrows'] + row['geometry_guides'] + row['negative_regions']
        require(len({c['id'] for c in controls}) == len(controls), 'Duplicate alias control')
        for arrow in row['arrows']:
            require(valid_point(arrow['tip']) and len(arrow['direction']) == 2
                    and np.isfinite(arrow['direction']).all() and np.linalg.norm(arrow['direction']) > 0,
                    'Invalid reviewed arrow')
        for guide in row['geometry_guides']:
            require(len(guide['points']) >= 2 and all(valid_point(p) for p in guide['points'])
                    and guide['exact_regression_target'] is False, 'Invalid partial guide')
        for negative in row['negative_regions']:
            x,y,right,bottom = negative['box']
            require(0 <= x < right <= gray.shape[1] and 0 <= y < bottom <= gray.shape[0], 'Invalid negative ROI')
        mask = np.zeros_like(gray)
        for box in row['text_regions']:
            x,y,right,bottom = box
            require(all(type(v) is int for v in box) and 0 <= x < right <= gray.shape[1]
                    and 0 <= y < bottom <= gray.shape[0], 'Invalid reviewed text region')
            mask[y:bottom, x:right] = (gray[y:bottom, x:right] < 240).astype(np.uint8)*255
        amended = {**case, 'scope': 'viewed_train_alias_reviewed_partial', 'reading': row,
                   'labels_authority': 'assistant pixel review, approximate guides, no dense truth'}
        amended['variants'] = []
        for variant in case['variants']:
            size = variant['size_px']
            scaled = mask if size == case['size_px'] else (np.asarray(Image.fromarray(mask).resize(size, Image.Resampling.BOX)) > 0).astype(np.uint8)*255
            name = f'cases/{case["id"]}/{variant["mode"]}/reviewed-text-mask.png'
            png(folder, name, scaled)
            amended['variants'].append({**variant, 'common_text_mask': name})
        result.append(amended)
    return result


def prepare(root, folder):
    root, folder = root.resolve(), folder.resolve()
    require(folder.parent == root/'artifacts' and not folder.exists(), 'Choose new artifacts directory')
    checked(root, {'path': PREVIOUS, 'sha256': PREVIOUS_SHA})
    config, guard = boundary(root)
    package = Package(root, PARENT)
    rows = package.json('records.json')
    screenings = [guard.check(r['source'], r['split'], r['id']) for r in rows]
    require(not any(s['blocked'] for s in screenings), 'Blocked source before pixel access')
    require(all(s['group_id'] == r['group_id'] for s, r in zip(screenings, rows)), 'Changed family assignment')
    validate_split_groups(rows)
    aliases = parallel_screen(root, guard)  # Complete all ancestry checks before any image opens.
    folder.mkdir()
    write(folder, 'screening.json', {'metadata': config['metadata'], 'parallel_aliases': aliases,
        'patch_sources': [{'id': r['id'], 'source': r['source'], 'screening': s} for r, s in zip(rows, screenings)],
        'all_screened_before_pixels': True})
    cases = []; source_seen = set()
    for row in rows:
        source = guard.sources[row['source']['path'].removeprefix('dataset/')]
        if row['source']['sha256'] not in source_seen:
            checked(root, row['source']); source_seen.add(row['source']['sha256'])
        prep = checked(root, source['preparation'])
        require(read(prep)['source_sha256'] == row['source']['sha256'], 'Preparation rebound')
        box = native_box(row)
        rgb, proof = read_prepared_region(prep.parent, box, max_pixels=1024**2)
        arrays = {}
        for key in ('image', 'labels', 'role-bits', 'state'):
            path = package.path(row['paths'][key])
            with Image.open(path) as im:
                arrays[key] = np.asarray(im).copy()
            write(folder, f'cases/{row["id"]}/parent-{key}.png', path.read_bytes())
        require(np.array_equal(arrays['image'], rgb), 'Source/patch pixels differ')
        stats = validate_arrays(rgb, arrays['labels'], arrays['role-bits'], arrays['state'])
        text_id = MASK_POLICY['classes'].index('text')
        text_mask = (arrays['labels'] == text_id).astype(np.uint8)*255
        gray = np.asarray(Image.fromarray(rgb).convert('L'))
        geometry = [a for a in row['annotations'] if a['kind'] in ('polyline', 'circle')]
        case = {'id': row['id'], 'split': row['split'], 'group_id': row['group_id'], 'source': row['source'],
            'scope': 'partial_real_reviewed_ink', 'dense_geometry_truth': False,
            'size_px': row['size_px'], 'source_box': box, 'source_to_native': row['source_to_patch'],
            'native_to_source': row['patch_to_source'], 'preparation': source['preparation'],
            'native_tiles_used': proof['used_tile_count'], 'annotations': row['annotations'],
            'geometry_guides': geometry, 'unknown_regions': row['unknown_regions'],
            'arrow_controls': row['review'].get('arrow_controls', []),
            'labels_authority': 'inherited assistant reviewed partial role corridors; not exact CAD',
            'mask_stats': stats, 'variants': variants(folder, row['id'], gray, text_mask)}
        cases.append(case)
    # Additional viewed-source examples: exact train aliases, observation only, no pair/scale approval.
    star = next((a for a in aliases if a['alias']['path'].endswith('/звездочка.png')), None)
    require(star and star['disposition'] != 'excluded', 'Star alias is unavailable')
    star_source = star['matches'][0]; src = guard.sources[star_source['path'].removeprefix('dataset/')]
    checked(root, {'path': star_source['path'], 'sha256': star_source['sha256']})
    prep = checked(root, src['preparation'])
    require(read(prep)['source_sha256'] == star_source['sha256'], 'Star preparation rebound')
    for identifier, box in [('parallel-star-top', [2000, 350, 3024, 1374]),
                            ('parallel-star-stamp', [3600, 5900, 4624, 6924])]:
        rgb, proof = read_prepared_region(prep.parent, box, max_pixels=1024**2)
        gray = np.asarray(Image.fromarray(rgb).convert('L'))
        cases.append({'id': identifier, 'split': 'train', 'group_id': star['group_id'],
            'source': {'path': star_source['path'], 'sha256': star_source['sha256']},
            'scope': 'viewed_train_alias_pending_scoped_reading', 'dense_geometry_truth': False,
            'size_px': [1024,1024], 'source_box': box,
            'source_to_native': [[1,0,-box[0]],[0,1,-box[1]],[0,0,1]],
            'native_to_source': [[1,0,box[0]],[0,1,box[1]],[0,0,1]],
            'native_tiles_used': proof['used_tile_count'], 'pair_approved': False,
            'variants': variants(folder, identifier, gray, np.zeros_like(gray))})
    for family in FAMILIES:
        for width in (3, 7, 11):
            gray, text_mask, labels = scene(family, width)
            identifier = f'analytic-{family}-w{width}'
            cases.append({'id': identifier, 'split': labels['split'], 'group_id': labels['group_id'],
                'scope': 'analytical_control', 'size_px': [768,512], 'dense_geometry_truth': True,
                'labels': labels, 'variants': variants(folder, identifier, gray, text_mask)})
    validate_split_groups(cases)
    write(folder, 'cases.json', cases)
    write(folder, 'inherited-stroke-reading.json', checked(root, pin(root, 'configs/reviews/stroke-curve-control-reading-v1.json')).read_bytes())
    noise_manifest = read(root/'artifacts/noise-baseline-v3/manifest.json')
    noise_names = ['protocol.json', 'noise-baseline-ink.png', 'useful-baseline-ink.png', 'unknown-baseline-ink.png']
    noise_pins = [{'path': 'artifacts/noise-baseline-v3/'+n,
                   'sha256': noise_manifest['files']['artifacts/noise-baseline-v3/'+n]} for n in noise_names]
    for item in noise_pins: checked(root, item)
    write(folder, 'known-folds-regression.json', {'source': pin(root,'IMG_20261109_160629_212.jpg'),
        'scope': 'existing viewed development pixel preservation control; no new physical labels',
        'inputs': [pin(root, 'artifacts/e2e-acceptance-v1/known-folds-jpeg/automatic/'+name)
                   for name in ('page-001-processed.png', 'page-001-binary.png', 'page-001-preprocessing.json')],
        'masks': noise_pins, 'protected_useful_pixels': 4203, 'noise_pixels': 205,
        'unknown_pixels': 1, 'geometry_error_metrics': 'not_evaluable'})
    write(folder, 'selection.json', {'policy': POLICY, 'class_names': MASK_POLICY['classes'],
        'parents': [READINESS, PARENT, pin(root,'configs/reviews/stroke-curve-control-reading-v1.json'),
                    pin(root,'artifacts/noise-baseline-v3/manifest.json')],
        'case_counts': dict(Counter(c['scope'] for c in cases)),
        'source_family_counts': {s:len({c['group_id'] for c in cases if c['split']==s}) for s in ('train','validation')},
        'baseline_code': [pin(root, p) for p in CODE],
        'dependency_check': {'skimage_available': importlib.util.find_spec('skimage') is not None,
            'parallel_execution_qualification': 'required before first candidate run; no environment changes here'}})
    cards = ['<!doctype html><meta charset="utf-8"><title>CV comparison inputs</title>',
        '<style>body{font:16px sans-serif}img{max-width:48%;vertical-align:top}section{border-top:1px solid;padding:1rem}</style>',
        '<h1>P15.2f.2 — fixed CV inputs</h1><p>Partial real labels and analytical controls; no model results.</p>']
    for case in cases:
        cards.append('<section><h2>'+html.escape(case['id'])+' / '+case['split']+'</h2>'+''.join(
            '<img src="'+v['gray']+'" alt="'+v['mode']+'">' for v in case['variants'])+'</section>')
    write(folder, 'index.html', ''.join(cards).encode('utf-8'))
    return {'cases': len(cases), 'selection': pin(root, (folder/'selection.json').relative_to(root).as_posix())}


def seal(root, folder, reading):
    require(not (folder/'protocol.json').exists(), 'Already sealed')
    cases = read(folder/'cases.json'); selection = read(folder/'selection.json'); reviewed = read(checked(root, reading))
    require({r['id'] for r in reviewed['cases']} == {'parallel-star-top', 'parallel-star-stamp'}, 'Missing alias review')
    for row in selection['baseline_code']: checked(root, row)
    validate_split_groups(cases)
    effective = reviewed_aliases(root, folder, cases, reviewed)
    write(folder, 'effective-cases.json', effective)
    files = {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in folder.rglob('*') if p.is_file()}
    protocol = {'schema': 'cv-comparison-protocol/1.0', 'task': 'P15.2f.2',
        'created_before_candidate_inference': True, 'policy': POLICY, 'files': files,
        'selection': selection, 'additional_reading': reading, 'baseline_runs': 0,
        'case_index': 'effective-cases.json',
        'next_task': 'P10.2c.3a', 'scope_complete': 'data and preregistration only; no quality admission'}
    write(folder, 'protocol.json', protocol)
    return verify(root, folder, pin(root,(folder/'protocol.json').relative_to(root).as_posix())['sha256'])


def verify(root, folder, expected_sha):
    checked(root, {'path': (folder/'protocol.json').relative_to(root).as_posix(), 'sha256': expected_sha})
    protocol = read(folder/'protocol.json')
    require(protocol['policy'] == POLICY and protocol['baseline_runs'] == 0, 'Changed protocol authority')
    for name, sha in protocol['files'].items(): checked(folder, {'path': name, 'sha256': sha})
    for record in protocol['selection']['baseline_code']+protocol['selection']['parents']+[protocol['additional_reading']]:
        checked(root, record)
    inventory = set(protocol['files']) | {'protocol.json'}
    require({p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()} == inventory,
            'Unregistered or missing package file')
    cases = read(folder/protocol['case_index']); validate_split_groups(cases)
    require(len(cases) == 45, 'Changed case inventory')
    _, guard = boundary(root)
    screening = read(folder/'screening.json')
    for alias in screening['parallel_aliases']:
        checked(root, alias['alias'])
    for case in cases:
        if 'source' in case:
            result = guard.check(case['source'], case['split'], case['id'])
            require(not result['blocked'] and result['group_id'] == case['group_id'], 'Source no longer admitted')
            checked(root, case['source'])
        if 'preparation' in case: checked(root, case['preparation'])
    noise = read(folder/'known-folds-regression.json')
    for record in [noise['source'], *noise['inputs'], *noise['masks']]: checked(root, record)
    for case in cases:
        for variant in case['variants']:
            a,b = np.asarray(variant['native_to_input']), np.asarray(variant['input_to_native'])
            require(np.allclose(a@b, np.eye(3)), 'Invalid coordinate roundtrip')
    return {'status': 'prepared_and_sealed', 'cases': len(cases), 'input_variants': sum(len(c['variants']) for c in cases),
        'protocol': pin(root, (folder/'protocol.json').relative_to(root).as_posix()),
        'candidate_runs': 0, 'closed_test_used': False, 'next_task': 'P10.2c.3a'}
