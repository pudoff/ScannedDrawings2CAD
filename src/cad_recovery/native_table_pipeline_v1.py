"""Explicit native table regions over native or registered legacy CAD frames."""
from pathlib import Path
import time

import numpy as np

from .dataset_layout_v1 import read, digest, encoded
from .native_sheet_source_v1 import NativePage, prepare
from .native_table_structure_v1 import analyze
from .native_table_export_v1 import export


def native_source(source, baseline, output):
    """Reuse only authenticated full-resolution surfaces; never stale predictions."""
    source, baseline, output = Path(source), Path(baseline), Path(output)
    manifest = baseline/'source/manifest.json'
    if manifest.exists():
        data = read(manifest)
        if data['source']['sha256'] != digest(source): raise ValueError('Native baseline belongs to another input')
        for record in data['pages']:
            if digest(record['surface']) != record['surface_sha256']: raise ValueError('Native surface changed')
        if output.exists(): raise FileExistsError(output)
        output.mkdir(parents=True)
        (output/'manifest.json').write_bytes(encoded(data))
        return data
    return prepare(source, output)


def baseline_binding(source, baseline, page_index, page):
    baseline = Path(baseline); report = read(baseline/'report.json')
    if report.get('schema_version') == 'native-cad-pipeline/2.0':
        if report['source']['sha256'] != digest(source): raise ValueError('Native CAD source mismatch')
        from .native_cad_pipeline_v2 import binding
        if report['adapter_binding'] != binding(): raise ValueError('Native CAD adapter changed')
        rows = [p for p in report['pages'] if p['page_index'] == page_index]
        if len(rows) != 1 or rows[0]['size_px'] != [page.width, page.height]: raise ValueError('Native page frame mismatch')
        old = rows[0]; dxf = baseline/f'page-{page_index+1:03d}.dxf'
        if digest(dxf) != old['dxf_sha256']: raise ValueError('Native CAD file changed')
        return dxf, dict(mode='native_CAD_source_frame', working_size_px=old['size_px'],
            native_to_working=np.eye(3).tolist(), source_sha256=digest(source), dxf_sha256=digest(dxf),
            baseline_core_search_complete=old['core_search_complete'],
            outside_table_native_resolution=True, whole_page_semantics_accepted=False)
    from .table_review_dispatch_v1 import page_inputs
    _, dxf, prior = page_inputs(source, baseline, page_index, allow_partial=True)
    transform = np.asarray(prior['source_to_working']) @ np.linalg.inv(page.matrix)
    return dxf, dict(mode='legacy_working_frame_with_native_table_evidence',
        working_size_px=prior['working_size_px'], native_to_working=transform.tolist(),
        source_sha256=digest(source), dxf_sha256=digest(dxf),
        baseline_core_search_complete=prior['baseline_complete'],
        outside_table_native_resolution=False, whole_page_semantics_accepted=False,
        legacy_binding=prior)


def prepare_review(source, baseline, previous_review, output):
    """Register a prior explicit visual ROI against exact source pixels."""
    output = Path(output)
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    previous = read(previous_review)
    if digest(source) != previous['source_sha256']: raise ValueError('Prior review source changed')
    data = native_source(source, baseline, output/'source')
    index = previous['page_index']; record = data['pages'][index]
    page = NativePage(record)
    try:
        _, binding = baseline_binding(source, baseline, index, page)
        matrix = np.asarray(previous['source_to_working']) @ np.linalg.inv(page.matrix)
        inverse = np.linalg.inv(matrix)
        regions = []
        for region in previous['regions']:
            l, t, r, b = region['box_px']
            corners = np.array([[l, t, 1], [r, t, 1], [r, b, 1], [l, b, 1]]) @ inverse.T
            corners = corners[:, :2]/corners[:, 2, None]
            low = np.floor(corners.min(0)).astype(int); high = np.ceil(corners.max(0)).astype(int)
            box = [max(0, int(low[0])), max(0, int(low[1])), min(page.width, int(high[0])), min(page.height, int(high[1]))]
            regions.append({**region, 'box_px': box,
                'registration_evidence':dict(previous_working_box=region['box_px'],
                    native_to_previous_working=matrix.tolist(), outward_rounding=True)})
        review = dict(schema_version='native-table-review/1.0', source_sha256=digest(source),
            source_surface_sha256=record['surface_sha256'], size_px=record['size_px'], page_index=index,
            page_kind=previous['page_kind'], reviewer=previous['reviewer'], regions=regions,
            prior_review=dict(path=str(Path(previous_review).resolve()), sha256=digest(previous_review)),
            baseline=binding)
        (output/'review.json').write_bytes(encoded(review))
        return review
    finally: page.close()


def run(source, baseline, review_path, source_manifest, output, *, engine):
    output = Path(output)
    if output.exists(): raise FileExistsError(output)
    review = read(review_path); source_sha = digest(source)
    data = read(source_manifest)
    if data['source']['sha256'] != source_sha: raise ValueError('Source manifest mismatch')
    record = data['pages'][review['page_index']]
    if digest(record['surface']) != record['surface_sha256']: raise ValueError('Source surface changed')
    page = NativePage(record)
    started = time.perf_counter()
    try:
        dxf, bound = baseline_binding(source, baseline, review['page_index'], page)
        if bound != review['baseline']: raise ValueError('Reviewed baseline binding changed')
        output.mkdir(parents=True)
        (output/'review-journal.json').write_bytes(encoded(dict(review=review,
            source_sha256=source_sha, review_sha256=digest(review_path), baseline=bound,
            automatic_routing=False, source_manifest_sha256=digest(source_manifest))))
        graph = analyze(page, review, source_sha, output/'structure', engine=engine)
        result = export(page, graph, dxf, output/f"page-{review['page_index']+1:03d}.dxf",
                        bound['native_to_working'], bound['working_size_px'])
        if digest(source) != source_sha or digest(dxf) != bound['dxf_sha256']:
            raise ValueError('Source or baseline changed during table composition')
        result.update(baseline=bound, graph='structure/structure.json',
            source_manifest=str(Path(source_manifest).resolve()), source_manifest_sha256=digest(source_manifest),
            ocr_calls=graph['ocr_calls'], native_table_evidence=True,
            whole_page_native_CAD_accepted=False, elapsed_seconds=time.perf_counter()-started)
        (output/'report.json').write_bytes(encoded(result))
        return result
    except BaseException as error:
        if output.exists(): (output/'failure.json').write_bytes(encoded(dict(error=str(error), production_admitted=False)))
        raise
    finally: page.close()
