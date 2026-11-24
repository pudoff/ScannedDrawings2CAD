"""Manual table review composed with the unchanged profile v6 CAD output."""
from pathlib import Path
import json

import numpy as np
from PIL import Image

from .dataset_layout_v1 import digest, encoded, read
from .table_structure_v1 import pixel_sha
from .table_cad_export_v1 import export_reviewed


def page_inputs(source, baseline, page_index, *, allow_partial=False):
    """Bind the actual preprocessed surface, not an independently resized image."""
    baseline = Path(baseline)
    report = read(baseline/'report.json')
    from .profile_contract_v6 import resolve_profile
    complete = report['processing_profile'] == resolve_profile('native_cad_layers_draft')
    common = report.get('common_input', {})
    if complete:
        if digest(source) != report['source_sha256']:
            raise ValueError('Baseline belongs to another source')
    elif allow_partial:
        # A failed frozen CAD run may have persisted a source-ink fallback.
        # It can diagnose this component, but is never a successful core run.
        from .profile_contract_v4 import resolve_profile as old_binding
        if report['processing_profile'] != old_binding('native_cad_draft'):
            raise ValueError('Unrecognized partial baseline binding')
        common = read(baseline.with_name(baseline.name+'-input-v2')/'manifest.json')
        prepared = baseline.with_name(baseline.name+'-input-v2')/'working.png'
        if (digest(source) != common['sha256'] or digest(prepared) != common['analysis_sha256']
                or report['source_sha256'] != common['analysis_sha256']):
            raise ValueError('Incomplete baseline/source chain differs')
        inner = read(baseline/'input-adaptation.json')
        common = {**common, 'source_pixel_to_analysis_px':
                  (np.array(inner['source_pixel_to_analysis_px']) @
                   common['source_pixel_to_analysis_px']).tolist()}
    else:
        raise ValueError('Baseline incomplete; explicit diagnostic allowance required')
    pages = [p for p in report['pages'] if p['page_index'] == page_index]
    if len(pages) != 1:
        raise ValueError('Requested page is not in baseline')
    page = pages[0]
    prefix = f'page-{page_index+1:03d}'
    gray_path = baseline/'private'/prefix/'experimental/gray.png'
    if digest(gray_path) != page['preprocessing']['artifact_sha256']['grayscale']:
        raise ValueError('Working pixels differ from the recorded preprocessing artifact')
    gray = np.array(Image.open(gray_path).convert('L'))
    exported = page['demo_export']
    if not complete and exported.get('experimental_fallback', {}).get('reason') != 'Incomplete source paths':
        raise ValueError('Partial baseline is not the recorded path-limit fallback')
    if exported.get('working_size_px', exported['source_size_px']) != list(gray.shape[::-1]):
        raise ValueError('CAD and working raster sizes differ')
    dxf = baseline/exported['dxf']
    if digest(dxf) != exported['dxf_sha256']:
        raise ValueError('Baseline CAD changed')
    if common.get('mode') == 'bounded_pdf_passthrough':
        # Page pixels are already rendered by the frozen core. PDF-point to
        # source-pixel mapping is kept separately; never infer physical units.
        input_matrix = np.eye(3)
        source_space = 'rendered_page_pixels'
    else:
        input_matrix = np.array(common['source_pixel_to_analysis_px'])
        source_space = 'original_raster_pixels'
    prepared = np.array(page['preprocessing']['transforms']['input_to_processed'])
    experimental = np.array(page['experimental_cv'].get('input_to_working', np.eye(3)))
    matrix = experimental @ prepared @ input_matrix
    binding = dict(source_sha256=digest(source), page_index=page_index,
                   gray_file_sha256=digest(gray_path), working_pixel_sha256=pixel_sha(gray),
                   working_size_px=list(gray.shape[::-1]), baseline_sha256=digest(dxf),
                   source_coordinate_space=source_space, source_to_working=matrix.tolist(),
                   baseline_profile_sha256=report['processing_profile']['sha256'],
                   baseline_complete=complete, allow_partial_baseline=allow_partial)
    return gray, dxf, binding


def prepare_review(source, baseline, output, *, page_index=0, allow_partial=False):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    gray, _, binding = page_inputs(source, baseline, page_index, allow_partial=allow_partial)
    output.mkdir(parents=True)
    Image.fromarray(gray).save(output/'working.png')
    template = dict(schema_version='table-region-review/1.0', **binding,
                    page_kind='mixed', reviewer='', regions=[])
    (output/'review-template.json').write_bytes(encoded(template))
    template_file = Path(__file__).with_name('table_review_ui_v1.html')
    # Template data is encoded as a JS string literal, including HTML-sensitive
    # characters, rather than interpolated into HTML as markup.
    data = json.dumps(template, ensure_ascii=True).replace('<', '\\u003c')
    html = template_file.read_text(encoding='utf-8').replace('__REVIEW_JSON__', data)
    (output/'review.html').write_text(html, encoding='utf-8')
    return template


def convert_review(source, baseline, review_file, output, *, models=None, render=False, recognizer=None,
                   no_ocr=False):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    review = read(review_file)
    page_index = review.get('page_index')
    if type(page_index) is not int or page_index < 0:
        raise ValueError('Explicit page index required')
    gray, base, binding = page_inputs(source, baseline, page_index,
                                    allow_partial=review.get('allow_partial_baseline') is True)
    for key, value in binding.items():
        # Early v1 review templates predate these two diagnostic metadata
        # fields. A COMPLETE baseline can supply them without weakening any
        # source/size/pixel/profile/transform check. Partial mode stays opt-in.
        if key in ('baseline_complete', 'allow_partial_baseline') and key not in review and binding['baseline_complete']:
            review[key] = value
        if review.get(key) != value:
            raise ValueError(f'Review/baseline binding changed: {key}')
    engine = None
    if recognizer is None and not no_ocr:
        from .ocr_engine import LocalOCREngine
        engine = LocalOCREngine(models or Path(__file__).resolve().parents[2]/'artifacts/models/ppocr-v5')
        recognizer = engine.recognize
    output.mkdir(parents=True)
    # Persist the explicit override before entering the new producer.
    (output/'review-journal.json').write_bytes(encoded(dict(
        schema_version='table-review-journal/1.0', original_source_sha256=digest(source),
        baseline=binding, explicit_override=review, automatic_routing=False,
        baseline_processing_retained=True)))
    report = export_reviewed(gray, review, digest(source), base, output/f'page-{page_index+1:03d}.dxf',
        recognizer=recognizer, source_to_working=binding['source_to_working'], render=render,
        ocr_provenance=engine.provenance if engine else dict(mode='disabled' if no_ocr else 'injected'))
    if digest(source) != binding['source_sha256'] or digest(base) != binding['baseline_sha256']:
        raise ValueError('Source or frozen baseline changed during table conversion')
    return report
