"""Screen source ancestry before reading whole-page overviews; no frame labels."""
from pathlib import Path
import numpy as np
from PIL import Image
from cad_recovery.dataset_layout_v1 import checked, pin
from cad_recovery.architecture_data_readiness_v1 import Boundary
from cad_recovery.page_region_review_v1 import READINESS
from run_topology_robust_v1 import read, write

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'artifacts/frame-page-sources-v1'


def build():
    BASE.mkdir(exist_ok=False)
    readiness = read(checked(ROOT, READINESS))
    groups, sources, one, two, fresh = [read(checked(ROOT, p)) for p in readiness['metadata']]
    checker = Boundary(groups['groups'], sources, [one, two],
        [{'source_path': r['input'], 'source_sha256': r['input_sha256']} for r in fresh['selected_sources']])
    manifest = read(ROOT/'artifacts/pixel-role-expansion-v3/manifest.json')
    record_pin = next(p for p in manifest['files'] if p['path'] == 'records.json')
    records_pin = {'path': 'artifacts/pixel-role-expansion-v3/records.json', 'sha256': record_pin['sha256']}
    patches = read(checked(ROOT, records_pin))
    by_sha = {r['input_sha256']: r for r in sources}
    screened = []
    # Finish all metadata screens before decoding any overview.
    for patch in patches:
        result = checker.check(patch['source'], patch['split'], patch['id'])
        if result['blocked']: raise ValueError('Blocked source cannot enter frame context')
        source = by_sha[patch['source']['sha256']]
        if source['declared_split'] != patch['split']: raise ValueError('Reassigned source split')
        screened.append((patch, source, result))
    pages = {}; rows = []
    for patch, source, result in screened:
        identity = patch['source']['sha256']
        if identity not in pages:
            report_path = checked(ROOT, source['preparation']); preparation = read(report_path)
            if preparation['source_sha256'] != identity or not preparation['source_integrity_validated'] or preparation['source_modified']:
                raise ValueError('Preparation identity/integrity mismatch')
            checked(ROOT, patch['source'])
            asset = preparation['overview']
            path = (report_path.parent/asset['path']).relative_to(ROOT).as_posix()
            overview = {'path': path, 'sha256': asset['sha256']}; checked(ROOT, overview)
            with Image.open(ROOT/path) as image:
                if list(image.size) != asset['size_px'] or image.width*image.height > 4194304:
                    raise ValueError('Bounded authenticated overview required')
            matrix = np.asarray(asset['overview_to_source'], float)
            if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
                raise ValueError('Invertible pixel-centre transform required')
            pages[identity] = {'source': overview, 'size_px': asset['size_px'], 'source_size_px': preparation['source_size_px'],
                               'overview_to_source': asset['overview_to_source'], 'original_source': patch['source'],
                               'preparation': source['preparation'], 'kind': 'authenticated_page_overview',
                               'source_verified': True, 'sampling': asset['method']}
        page = pages[identity]
        translation = np.asarray(patch['patch_to_source'], float)
        inverse = np.asarray(patch['source_to_patch'], float)
        if not np.allclose(translation @ inverse, np.eye(3), atol=1e-12): raise ValueError('Patch transforms do not invert')
        x, y = translation[:2, 2]; width, height = patch['size_px']
        if not (0 <= x < x+width <= page['source_size_px'][0] and 0 <= y < y+height <= page['source_size_px'][1]):
            raise ValueError('ROI outside original source')
        rows.append({'case': patch['id'], 'source_sha256': identity, 'page_source': {**page,
                     'source_to_patch': patch['source_to_patch'], 'patch_to_source': patch['patch_to_source'],
                     'patch_bbox': [int(x), int(y), int(x+width), int(y+height)]},
                     'screening': result, 'original_split': patch['split']})
    write(BASE/'catalog.json', {'readiness': READINESS, 'metadata': readiness['metadata'], 'records': records_pin,
          'rows': rows, 'source_count': len(pages), 'frame_GT_or_annotation_bbox_read': False,
          'all_sources_screened_before_image_reads': True, 'closed_test_used': False, 'independent_evaluation': False})
    print({'patches': len(rows), 'pages': len(pages), 'catalog': pin(ROOT, 'artifacts/frame-page-sources-v1/catalog.json')}, flush=True)


if __name__ == '__main__': build()
