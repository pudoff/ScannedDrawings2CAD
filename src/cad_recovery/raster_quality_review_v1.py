"""Pixel-quality packet on a sealed viewed cohort; never executes an OCR model."""
from collections import Counter
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import numpy as np
from PIL import Image

from .dataset_layout_v1 import checked, encoded, inside, pin, read
from .family_catalog_v2 import require
from .hard_raster_regression_v1 import png
from .hard_raster_runs_v1 import verify as verify_runs, environment
from .raster_quality_v1 import POLICY, REASONS, analyze_pixels
from .source_coallocation_v1 import package

PARENT = {'path': 'artifacts/hard-raster-runs-final-v2/manifest.json',
          'sha256': '43c0c58a0de2751ff216182c2b6590afb17c17540869883bd1d9f4c5ce57b03b'}
CODE = ('src/cad_recovery/raster_quality_v1.py', 'src/cad_recovery/raster_quality_review_v1.py',
        'scripts/diagnose_raster_quality_v1.py', 'requirements.lock')
FALSE_FLAGS = ('models_executed', 'closed_ocr_tests_executed', 'training_started',
               'physical_geometry_approved', 'independent_evaluation', 'automatic_acceptance')
COMPARISON = {
    'schema_version': 'bounded-preprocessing-comparison-contract/1.0', 'execution_task': 'P15.2c',
    'executed_here': False, 'cohort': 'entire fixed ledger plus separately screened real scans; freeze extension before inference',
    'arms': [
        {'id': 'original', 'operation': 'original RGB pixels, existing baseline profile'},
        {'id': 'upscale-2x', 'operation': 'OpenCV INTER_CUBIC, exactly 2x on both axes',
         'source_to_derivative_pixel_centers': [[2, 0, .5], [0, 2, .5], [0, 0, 1]]},
        {'id': 'local-contrast', 'operation': 'CLAHE on grayscale; clipLimit=2.0; tileGridSize=[8,8]',
         'source_to_derivative_pixel_centers': [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}],
    'pixel_budget': 25_000_000, 'overflow': 'retain assigned refusal; no source replacement or silent smaller upscale',
    'producer_profiles': 'same frozen CLI/worker stages, model weights and CPU settings; record each changed input derivative',
    'ground_truth': 'unchanged literal fields in original source coordinates; partial/unknown never empty correct targets',
    'rec_only': 'upright physical lines scored separately; reviewed rotations are not automatic orientation',
    'selection': 'validation only, complete literal-field exact must strictly improve with no increased misses or lost assigned signs; otherwise keep original; refusals preclude complete-coverage claim',
    'metrics': ['paired field exact', 'CER with failures', 'assigned sign retention', 'missed fields', 'quantity candidates',
                'actual DXF objects by matching stages', 'whole child-tree walltime and memory'],
    'budgets': {'worker_seconds': 180, 'analysis_seconds': 600, 'memory_MiB': 4096},
    'quality_proxy_alone_cannot_select_arm': True, 'threshold_tuning_on_closed_tests': False,
    'lost_pixel_reconstruction_claim': False, 'physical_accuracy_claim': False,
    'new_family_independence_claim': False, 'training_allowed': False,
}


def inputs(root):
    verify_runs(root, PARENT)
    base, manifest = package(root, PARENT, 'hard-raster-runs-manifest/1.0')
    config = read(checked(root, manifest['config']))
    refs = read(root/'configs/reviews/hard-raster-references-v1.json')
    return base, config, refs


def freeze(root, output):
    _, parent, _ = inputs(root)
    config = {'schema_version': 'raster-quality-review-config/1.0', 'task': 'P09.3a',
              'created_before_measurement_utc': datetime.now(timezone.utc).isoformat(),
              'parent': PARENT, 'policy': POLICY, 'comparison': COMPARISON,
              'code': [pin(root, p) for p in CODE], 'environment': environment(),
              'reference_scope': 'all ten contexts and all 34 assigned text fields; includes three partial/unknown fields',
              'cases': parent['cases'], **{f: False for f in FALSE_FLAGS}}
    path = inside(root, output); require(not path.exists(), 'Choose a new quality config')
    path.write_bytes(encoded(config)); return pin(root, output)


def authenticate(root, config_pin):
    config = read(checked(root, config_pin))
    require(config['schema_version'] == 'raster-quality-review-config/1.0' and config['task'] == 'P09.3a', 'Unknown quality config')
    require(config['policy'] == POLICY and config['comparison'] == COMPARISON and config['parent'] == PARENT, 'Changed diagnostic/comparison policy')
    require(config['code'] == [pin(root, p) for p in CODE] and config['environment'] == environment(), 'Quality code/environment changed')
    require(all(config[f] is False for f in FALSE_FLAGS), 'Quality packet cannot approve model runs, physical targets or acceptance')
    base, parent, refs = inputs(root)
    require(config['cases'] == parent['cases'], 'Lost/replaced/relabeled source regions')
    require(all(not r['blocked'] for r in parent['boundary']), 'Closed-source boundary changed')
    return config, base, refs


def turn(image, degrees):
    return image.copy() if not degrees else image.transpose({90: Image.Transpose.ROTATE_270,
        180: Image.Transpose.ROTATE_180, 270: Image.Transpose.ROTATE_90}[degrees])


def associations(regions):
    """Descriptive cross-tabs after measurement, not a trained risk predictor."""
    result = {}
    for stratum in sorted({r['stratum'] for r in regions}):
        result[stratum] = {}
        for split in sorted({r['split'] for r in regions if r['stratum'] == stratum}):
            selected = [r for r in regions if r['stratum'] == stratum and r['split'] == split and r['outcome'] is not None]
            comparisons = {}
            for flag in sorted(REASONS):
                arms = {}
                for present in (True, False):
                    part = [r for r in selected if (flag in r['measurement']['flags']) == present]
                    arms['with_flag' if present else 'without_flag'] = {
                        'fields': len(part), 'det_rec_misses': sum(not r['outcome']['detected'] for r in part),
                        'det_rec_wrong': sum(not r['outcome']['det_rec_exact'] for r in part),
                        'rec_only_wrong': sum(not r['outcome']['rec_only_exact'] for r in part)}
                comparisons[flag] = arms
            result[stratum][split] = comparisons
    return {'interpretation': 'viewed development cross-tabs only; no causal or predictive accuracy claim', 'strata': result}


def overlay(case, regions, guides):
    w, h = case['size_px']
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}"><image href="source.png" width="{w}" height="{h}"/>']
    for r in regions:
        x, y, right, bottom = r['context_box']
        color = '#b45309' if r['measurement']['flags'] else '#0369a1'
        parts.append(f'<rect x="{x}" y="{y}" width="{right-x}" height="{bottom-y}" fill="none" stroke="{color}" stroke-width="1.5"><title>'
            +escape(r['field_id']+' | '+', '.join(r['measurement']['flags']))+'</title></rect>')
    for g in guides:
        points = ' '.join(str(x)+','+str(y) for x, y in g['points'])
        parts.append('<polyline points="'+points+'" fill="none" stroke="#a21caf" stroke-width="2"><title>'+escape(g['id']+' manual artifact guide')+'</title></polyline>')
    return (''.join(parts)+'</svg>').encode('utf-8')


def render(report):
    parts = ['<!doctype html><html lang="ru"><meta charset="utf-8"><title>Качество областей — P09.3a</title>',
        '<style>body{font:16px system-ui;margin:2rem;max-width:1200px}article{border-top:1px solid #bbb;padding:1rem 0}img{max-width:100%;max-height:180px;image-rendering:pixelated}.context{max-height:480px}pre{white-space:pre-wrap}table{border-collapse:collapse}td,th{padding:.5rem;border:1px solid #bbb}.muted{color:#555}</style>',
        '<h1>Диагностика качества областей</h1><p>Просмотренный набор разработки. Индикаторы не подтверждают читаемость, физический масштаб или точность CAD.</p>',
        '<p><a href="report.json">Измерения и ограничения</a> · <a href="associations.json">Связь с прежними ошибками</a> · <a href="comparison-contract.json">Условия будущего сравнения</a></p>',
        '<label><input id="flagged" type="checkbox"> Только области с пиксельными предупреждениями</label>']
    for context in report['contexts']:
        parts.append('<details><summary>'+escape(context['id'])+' — исходный контекст</summary><img class="context" src="contexts/'+context['id']+'/overlay.svg"></details>')
    for region in report['regions']:
        m = region['measurement']; directory = 'regions/'+region['id']
        parts += ['<article data-flagged="'+str(bool(m['flags'])).lower()+'"><h2>'+escape(region['id'])+'</h2>',
            '<p>Исходник: '+escape(region['source']['path'])+'; bbox '+escape(str(region['source_box']))+'. '+escape(region['reference_state'])+'</p>',
            '<p>Оригинальные пиксели</p><img src="'+directory+'/native.png"><p>Поворот для измерения: '+str(m['rotation_cw'])+'° по ручной разметке</p><img src="'+directory+'/upright.png">',
            '<p>Высота компонент, медиана: '+escape(str(m['component_height_median_px']))+' px; контраст: '+str(m['robust_contrast_8bit'])+'/255; дисперсия лапласиана: '+str(m['laplacian_variance_8bit'])+'</p>']
        parts.append('<ul>'+''.join('<li>'+escape(s)+'</li>' for s in m['reasons'])+'</ul>' if m['reasons'] else '<p>Пиксельные пороги не сработали; это не автоматический допуск.</p>')
        if region['reference_state'] != 'readable': parts.append('<p>Целевой текст не утверждён, exact-оценка не выполняется.</p>')
        if region['outcome']:
            o = region['outcome']
            parts.append('<p>Сохранённый baseline: det+rec '+escape(repr(o['det_rec_prediction']))+'; rec-only '+escape(repr(o['rec_only_prediction']))+'. Эталон: '+escape(repr(o['reference']))+'</p>')
        parts.append('<p class="muted">Причина размытия, складки и степень сжатия по этим индикаторам не установлены.</p></article>')
    parts.append('<script>document.querySelector("#flagged").addEventListener("change",e=>{document.querySelectorAll("article").forEach(a=>{a.hidden=e.target.checked&&a.dataset.flagged!=="true"})})</script></html>')
    return ''.join(parts).encode('utf-8')


def materialize(root, config_pin):
    config, base, refs = authenticate(root, config_pin)
    saved = read(base/'report.json')
    # Join observations only after the pixel kernel runs; they cannot set a flag.
    outcomes = {(t['case_id'], r['id']): r for t in saved['tasks'] if t['text'] for r in t['text']['rows']}
    recognition = {(r['case_id'], r['id']): r for r in saved['rec_only_rows']}
    contexts, regions, assets = [], [], {}
    for case in config['cases']:
        cid = case['id']; path = base/'inputs'/(cid+'.png')
        with Image.open(path) as image: image = image.convert('RGB')
        with Image.open(root/case['source']['path']) as source:
            codec, full_size = source.format, list(source.size)
        require(list(image.size) == case['size_px'], 'Context pixels changed')
        guides = [s for s in refs['strokes'] if s['role'] == 'fold_artifact'] if cid == 'jpeg' else []
        context = {'id': cid, 'source': case['source'], 'source_size_px': full_size, 'source_box': case['source_box'],
                   'stratum': case['stratum'], 'split': case['split'], 'source_codec': codec,
                   'measurement': analyze_pixels(np.array(image), source_box=case['source_box'], codec=codec, scope='context', manual_artifacts=guides)}
        contexts.append(context)
        assets['contexts/'+cid+'/source.png'] = path.read_bytes()
        local = []
        for field in case['texts']:
            ident = cid+'--'+field['id']; x, y, right, bottom = field['box']
            box = [x+case['source_box'][0], y+case['source_box'][1], right+case['source_box'][0], bottom+case['source_box'][1]]
            native = image.crop(field['box'])
            measured = analyze_pixels(np.array(native), source_box=box, rotation_cw=field['rotation_cw'], codec=codec, manual_artifacts=guides)
            row = {'id': ident, 'case_id': cid, 'field_id': field['id'], 'source': case['source'],
                'source_box': box, 'context_box': field['box'], 'source_size_px': full_size,
                'reference_state': field['state'], 'stratum': case['stratum'], 'split': case['split'],
                'measurement': measured, 'outcome': None}
            if field['state'] == 'readable':
                o, rec = outcomes[(cid, field['id'])], recognition[(cid, field['id'])]
                require(o['reference'] == rec['reference'] == field['transcription'], 'Saved reference changed')
                row['outcome'] = {'reference': o['reference'], 'detected': o['detected'],
                    'det_rec_exact': o['raw_exact'], 'rec_only_exact': rec['raw_exact'],
                    'det_rec_prediction': o['prediction'], 'rec_only_prediction': rec['prediction']}
            regions.append(row); local.append(row)
            directory = 'regions/'+ident+'/'
            assets[directory+'native.png'] = png(native)
            assets[directory+'upright.png'] = png(turn(native, field['rotation_cw']))
            assets[directory+'measurement.json'] = encoded(row)
        assets['contexts/'+cid+'/overlay.svg'] = overlay(case, local, guides)
    summary = {'contexts': len(contexts), 'regions': len(regions),
        'reference_states': dict(Counter(r['reference_state'] for r in regions)),
        'region_flags': dict(Counter(f for r in regions for f in r['measurement']['flags'])),
        'context_flags': dict(Counter(f for c in contexts for f in c['measurement']['flags'])),
        'manual_review_required_regions': len(regions),
        'pixel_flags_without_baseline_error': sum(bool(r['measurement']['flags']) and r['outcome'] is not None and r['outcome']['det_rec_exact'] for r in regions),
        'baseline_errors_without_pixel_flags': sum(not r['measurement']['flags'] and r['outcome'] is not None and not r['outcome']['det_rec_exact'] for r in regions)}
    report = {'schema_version': 'raster-quality-review-report/1.0', 'config': config_pin,
        'summary': summary, 'contexts': contexts, 'regions': regions, 'policy': POLICY,
        'no_flags_is_not_acceptance': True, 'physical_accuracy': None,
        'threshold_validation': 'descriptive on viewed development; no independently calibrated quality labels',
        'preprocessing_variants_executed': False, **{f: False for f in FALSE_FLAGS}}
    assets.update({'report.json': encoded(report), 'associations.json': encoded(associations(regions)),
                   'comparison-contract.json': encoded(COMPARISON), 'policy.json': encoded(POLICY), 'index.html': render(report)})
    return report, assets


def build(root, config_pin, output):
    report, assets = materialize(root, config_pin)
    target = inside(root, output); require(target.parent == root/'artifacts' and not target.exists(), 'Choose new artifacts folder')
    target.mkdir(parents=True)
    for name, value in assets.items():
        path = target/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(value)
    manifest = {'schema_version': 'raster-quality-review-manifest/1.0', 'config': config_pin,
                'files': [pin(target, p) for p in sorted(assets)]}
    (target/'manifest.json').write_bytes(encoded(manifest))
    return {'manifest': pin(root, output+'/manifest.json'), 'summary': report['summary']}


def verify(root, manifest_pin):
    base, manifest = package(root, manifest_pin, 'raster-quality-review-manifest/1.0')
    report, expected = materialize(root, manifest['config'])
    require({p.relative_to(base).as_posix() for p in base.rglob('*') if p.is_file()} == set(expected)|{'manifest.json'}, 'Quality inventory changed')
    require({p['path'] for p in manifest['files']} == set(expected), 'Manifest inventory changed')
    for name, value in expected.items(): require((base/name).read_bytes() == value, 'Quality artifact differs from original pixels/policy: '+name)
    return {'manifest': manifest_pin, 'summary': report['summary'], 'models_reexecuted': False}
