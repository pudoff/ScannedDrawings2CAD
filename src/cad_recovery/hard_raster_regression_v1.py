"""Sealed pixel references and saved-observation diagnostics; no inference or training."""
from collections import Counter
from html import escape
from io import BytesIO
from pathlib import Path

import ezdxf
import numpy as np
from PIL import Image

from .dataset_layout_v1 import checked, encoded, inside, pin, read
from .family_catalog_v2 import require
from .hard_raster_metrics_v1 import POLICY, score_saved, transform_points, validate_references
from .service_v1.config import ProcessingOptions, Settings
from .source_coallocation_v1 import package

CODE = ('src/cad_recovery/hard_raster_metrics_v1.py', 'src/cad_recovery/hard_raster_regression_v1.py',
        'scripts/prepare_hard_raster_regression_v1.py', 'src/cad_recovery/ocr_baseline_metrics_v1.py',
        'src/cad_recovery/dataset_layout_v1.py', 'requirements.lock')
FALSE_FLAGS = ('human_verified', 'training_eligible', 'physical_geometry_approved',
               'coordinate_regression_approved', 'independent_evaluation', 'models_executed',
               'closed_ocr_tests_executed', 'closed_acceptance_ten_allowed')


def validate_protocol(root, protocol):
    require(protocol['schema_version'] == 'hard-raster-regression-protocol/1.0' and protocol['task'] == 'P15.2a', 'Unknown regression protocol')
    require(protocol['matching_policy'] == POLICY and protocol['cohort'] == 'viewed_development_regression_only', 'Changed scoring/split policy')
    require(all(protocol[f] is False for f in ('training_eligible', 'independent_evaluation', 'physical_geometry_approved',
            'closed_acceptance_ten_allowed', 'new_inference_allowed_in_this_step')), 'Protocol cannot approve inference, physical labels or closed testing')
    for record in protocol['current_profile_code']:
        checked(root, record)
    manifest = read(checked(root, protocol['model_manifest']))
    require(len(protocol['models']) == len(manifest['models']) == 3, 'Missing model pins')
    for model in manifest['models']:
        record = next(r for r in protocol['models'] if r['path'].endswith('/'+model['filename']))
        require(record['sha256'] == model['sha256'], 'Model differs from pinned OCR manifest')
        checked(root, record)
    for key, expected_profile in (('worker_baseline', 'baseline'), ('worker_geometry', 'geometry_observations')):
        row = protocol['profiles'][key]
        options = ProcessingOptions.model_validate({k: row[k] for k in ('profile', 'render_dpi', 'image_dpi', 'rotation_deg')})
        require(options.profile == expected_profile and row['ocr'] is False and row['associate'] is False,
                'Worker does not implement OCR/association in this protocol')
    require(protocol['budgets']['worker_timeout_seconds'] == Settings.__dataclass_fields__['timeout_seconds'].default,
            'Worker timeout changed')


def context(root, config_pin):
    config = read(checked(root, config_pin))
    require(config['schema_version'] == 'hard-raster-regression-config/1.0' and config['task'] == 'P15.2a', 'Unknown regression config')
    require(config['code'] == [pin(root, n) for n in CODE], 'Regression implementation changed')
    require(all(config[f] is False for f in FALSE_FLAGS), 'This packet is neither training nor independent acceptance')
    refs = read(checked(root, config['references']))
    validate_references(refs)
    checked(root, refs['source'])
    snapshot = read(checked(root, config['legacy_snapshot']))
    require(snapshot['schema_version'] == 'hard-raster-legacy-snapshot/1.0' and snapshot['model_run'] is False, 'Unknown historical snapshot')
    base = inside(root, snapshot['folder'])
    names = {r['path'] for r in snapshot['files']}
    require(len(names) == len(snapshot['files']) and names == {p.relative_to(base).as_posix() for p in base.rglob('*') if p.is_file()},
            'Historical snapshot inventory changed')
    for record in snapshot['files']:
        checked(base, record)
    require(read(base/'checks.json')['source_sha256'] == refs['source']['sha256'], 'Historical source differs')
    protocol = read(checked(root, config['protocol']))
    validate_protocol(root, protocol)
    cases = read(checked(root, config['prospective_cases']))
    require(cases['schema_version'] == 'hard-raster-prospective-cases/1.0'
            and cases['test_sources_selected'] is False and cases['new_inference_allowed'] is False, 'Invalid prospective cohort')
    selection = read(checked(root, cases['selection']))
    require(len(cases['cases']) == len(selection['sources'])
            and {c['id'] for c in cases['cases']} == {s['id'] for s in selection['sources']}, 'Prospective case ledger lost assigned sources')
    folder, _ = package(root, cases['constraint_manifest'], 'role-admission-manifest/1.0')
    membership = {p: g for g in read(folder/'groups.json')['groups'] for p in g['members']}
    for case in cases['cases']:
        source = next(s for s in selection['sources'] if s['id'] == case['id'])
        require(case['input'] == source['input'] and case['source_sha256'] == source['input_sha256']
                and case['declared_split'] == source['declared_split'] in ('train', 'validation')
                and case['group_id'] == membership[source['input']]['id'], 'Prospective source/split/group changed')
        require(case['requires_closed_ocr_boundary_screen'] is True and case['model_run_allowed_now'] is False
                and case['independent_evaluation'] is False, 'Unscreened case cannot enter inference')
        checked(root/'dataset', {'path': case['input'], 'sha256': case['source_sha256']})
    return config, refs, snapshot, base, protocol, cases


def png(im):
    stream = BytesIO(); im.save(stream, format='PNG'); return stream.getvalue()


def overlay(refs):
    w, h = refs['source_size_px']
    colors = {'part_contour': '#009b4d', 'axis_or_construction': '#9933cc', 'sheet_frame': '#005eff',
              'title_grid': '#005eff', 'dimension_stroke': '#ff6b00', 'leader_stroke': '#ff6b00',
              'dimension_arc_guide': '#ff6b00', 'fold_artifact': '#dc2626'}
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}"><image href="source.png" width="{w}" height="{h}"/>']
    for field in refs['texts']:
        x, y, r, b = field['box']
        color = '#0891b2' if field['state'] == 'readable' else '#6b7280'
        parts.append(f'<rect x="{x}" y="{y}" width="{r-x}" height="{b-y}" fill="none" stroke="{color}" stroke-width=".7"><title>'
                     +escape(field['id']+' — '+(field['transcription'] or field['state']))+'</title></rect>')
    for stroke in refs['strokes']:
        value = ' '.join(str(x)+','+str(y) for x, y in stroke['points'])
        dash = ' stroke-dasharray="3 2"' if stroke['state'] == 'partly_occluded' else ''
        parts.append('<polyline points="'+value+'" fill="none" stroke="'+colors[stroke['role']]+'" stroke-width=".8"'+dash+'><title>'
                     +escape(stroke['id']+' — '+stroke['role']+' — '+stroke['state'])+'</title></polyline>')
    for arrow in refs['arrows']:
        x, y = arrow['tip']
        parts.append(f'<circle cx="{x}" cy="{y}" r="{arrow["radius_px"]}" fill="none" stroke="#dc2626" stroke-width=".7"><title>'
                     +escape(arrow['id'])+'</title></circle>')
    return (''.join(parts)+'</svg>').encode('utf-8')


def materialize(root, config_pin):
    config, refs, snapshot, base, protocol, cases = context(root, config_pin)
    analysis = base/'analysis'
    ocr, geometry = read(analysis/'page-001-ocr.json'), read(analysis/'page-001-geometry.json')
    arrows = read(analysis/'page-001-arrows.json')
    associations, graph = read(analysis/'page-001-dimension-associations.json'), read(analysis/'page-001-graph.json')
    prep = read(analysis/'page-001-preprocessing.json')
    require(ocr['processed_to_input'] == prep['transforms']['processed_to_input'], 'Stored pixel transforms disagree')
    for line in ocr['lines']:
        require(np.allclose(transform_points(line['polygon_px'], ocr['processed_to_input']), line['polygon_input_px'], rtol=0, atol=1e-6),
                'OCR observation does not map to original pixels')
    for item in (geometry, arrows, associations):
        require(item['source_sha256'] == ocr['source_sha256'] and item['processed_size_px'] == ocr['processed_size_px'], 'Mixed historical coordinate frames')
    with Image.open(checked(root, refs['source'])) as source:
        require(list(source.size) == refs['source_size_px'], 'Original raster size changed')
        image = source.convert('RGB')
    with Image.open(analysis/'page-001-source.png') as historic:
        require(historic.size == image.size and historic.convert('RGB').tobytes() == image.tobytes(), 'Historical decoded pixels differ from JPEG')
    require(ocr['engine']['model_manifest_sha256'] == protocol['model_manifest']['sha256'], 'Historical model manifest differs')
    for model in ocr['engine']['models']:
        require(any(m['path'].endswith('/'+model['filename']) and m['sha256'] == model['sha256'] for m in protocol['models']),
                'Historical OCR weights differ')
    dxf = ezdxf.readfile(base/'convert/page-001.dxf')
    counts = dict(Counter(e.dxftype() for e in dxf.modelspace()))
    require(counts == read(base/'checks.json')['dxf_modelspace_counts'], 'Saved DXF inventory differs')
    scores = score_saved(refs, ocr, geometry, arrows, associations, graph, counts)
    report = {'schema_version': 'hard-raster-regression-report/1.0', 'config': config_pin,
        'reference_counts': {k: len(refs[k]) for k in ('texts', 'symbols', 'strokes', 'arrows', 'links')},
        'historical_scores': scores['summary'], 'source_sha256': refs['source']['sha256'],
        'historical_internal_elapsed_seconds': {'convert': read(base/'convert/report.json')['elapsed_seconds'],
                                                'analysis': read(base/'analysis/report.json')['elapsed_seconds']},
        'historical_code_identity': snapshot['historical_code_identity'],
        'prospective_case_count': len(cases['cases']), 'prospective_case_statuses': dict(Counter(c['review_scope'] for c in cases['cases'])),
        'historical_predictions_scored': True, 'current_profiles_executed': False,
        'pixel_evaluation_only': True, **{f: False for f in FALSE_FLAGS}}
    assets = {'references.json': encoded(refs), 'protocol.json': encoded(protocol), 'prospective-cases.json': encoded(cases),
        'legacy-snapshot.json': encoded(snapshot), 'scores.json': encoded(scores), 'report.json': encoded(report),
        'source.png': png(image), 'overlay.svg': overlay(refs)}
    for field in refs['texts']:
        query = image.crop(field['box'])
        if field['rotation_cw']:
            query = query.transpose({90: Image.Transpose.ROTATE_270, 180: Image.Transpose.ROTATE_180, 270: Image.Transpose.ROTATE_90}[field['rotation_cw']])
        assets['queries/'+field['id']+'.png'] = png(query)
    for symbol in refs['symbols']:
        assets['symbols/'+symbol['id']+'.png'] = png(image.crop(symbol['box']))
    for name, file in (('ocr.json', 'page-001-ocr.json'), ('geometry.json', 'page-001-geometry.json'),
                       ('arrows.json', 'page-001-arrows.json'), ('associations.json', 'page-001-dimension-associations.json'),
                       ('graph.json', 'page-001-graph.json')):
        assets['historical/'+name] = (analysis/file).read_bytes()
    assets['historical/result.dxf'] = (base/'convert/page-001.dxf').read_bytes()
    rows = {r['id']: r for r in scores['text_rows']}
    cards = []
    for field in refs['texts']:
        result = rows.get(field['id'])
        prediction = result['prediction'] if result else 'UNSCORED: '+field['state']
        cards.append('<section><h3>'+escape(field['id'])+'</h3><img src="queries/'+field['id']+'.png" alt="Original query">'
            '<p>Reference: '+escape(field['transcription'] or field['state'])+'</p><p>Saved observation: '+escape(prediction)+'</p></section>')
    assets['index.html'] = ('<!doctype html><meta charset="utf-8"><title>P15.2a hard-raster regression</title>'
        '<style>body{font:16px sans-serif;margin:2rem}.overlay{width:min(100%,650px);height:auto}section{border-top:1px solid #aaa;padding:1rem}section img{max-height:120px;max-width:100%;image-rendering:pixelated}pre{white-space:pre-wrap}</style>'
        '<h1>P15.2a — viewed development pixel references</h1><p>Assistant-reviewed, partial page scope. '
        'No new inference, training, physical accuracy or independent test.</p><p>Current worker has no OCR/association stage; '
        'its output must not be compared with analyze-dimensions as the same route.</p>'
        '<object class="overlay" type="image/svg+xml" data="overlay.svg"></object><p><a href="source.png">Original RGB</a> · '
        '<a href="scores.json">Saved-observation scores</a> · <a href="protocol.json">Frozen profiles and budgets</a></p>'
        '<pre>'+escape(str(report))+'</pre>'+''.join(cards)).encode('utf-8')
    return assets, report


def build(workspace, config_pin, output):
    root = Path(workspace).resolve(); folder = inside(root, output)
    require(not folder.exists() and not folder.resolve().is_relative_to((root/'dataset').resolve()), 'Choose a fresh output outside dataset')
    assets, report = materialize(root, config_pin)
    folder.mkdir(parents=True, exist_ok=False)
    for name, data in assets.items():
        path = inside(folder, name); path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream: stream.write(data)
    with (folder/'manifest.json').open('xb') as stream:
        stream.write(encoded({'schema_version': 'hard-raster-regression-manifest/1.0', 'config': config_pin,
                              'files': [pin(folder, n) for n in sorted(assets)]}))
    return {'manifest': pin(root, (folder/'manifest.json').relative_to(root).as_posix()), 'report': report}


def verify(workspace, manifest_pin):
    root = Path(workspace).resolve(); folder, manifest = package(root, manifest_pin, 'hard-raster-regression-manifest/1.0')
    assets, report = materialize(root, manifest['config'])
    require({r['path'] for r in manifest['files']} == set(assets), 'Regression packet inventory differs')
    for name, data in assets.items():
        require((folder/name).read_bytes() == data, 'Regression packet differs from source pixels/protocol: '+name)
    return report
