"""Source-screened review of native semantic page regions; no model inference."""
from collections import Counter
from pathlib import Path

from PIL import Image
import numpy as np

from .architecture_data_readiness_v1 import Package, Boundary
from .dataset_layout_v1 import checked, encoded, pin, read
from .family_catalog_v2 import require
from .raster_regions import read_prepared_region

PREVIOUS = {'path': 'artifacts/conditional-ocr-release-v1.json',
            'sha256': '5142a0ad3d3271cc349e3e952d3cca6649cb645da9ec65bcca1fa744ba1c827d'}
READINESS = {'path': 'configs/reviews/architecture-data-readiness-v5.json',
             'sha256': '56e53362b7e633e61de341f254250353ca519b238c68977bdd21f07d602834df'}
PARENT = {'path': 'artifacts/pixel-role-expansion-v3/manifest.json',
          'sha256': 'ba7836cfa899633e3291f95c6715fbd9b64418322e2ff14377e4143dbb320746'}
POLICY = {'id': 'page-region-review/1.0', 'classes': ['view', 'detail', 'title_block', 'table', 'frame'],
          'maximum_sources': 6, 'maximum_regions': 64, 'native_tile_side': 1024,
          'maximum_native_tiles': 512, 'maximum_region_pixels': 67108864,
          'maximum_total_region_pixels': 268435456, 'reviewer': 'assistant_pixel_review',
          'human_verified': False, 'dense_ground_truth': False, 'closed_test_used': False,
          'independent_evaluation': False, 'physical_geometry_approved': False,
          'production_admitted': False}


def preserve(root):
    previous = read(checked(root, PREVIOUS))
    values = previous['prior_protected_pins'] | previous['files'] | {PREVIOUS['path']: PREVIOUS['sha256']}
    for name, sha in values.items():
        checked(root, {'path': name, 'sha256': sha})
    return values


def context(root):
    readiness = read(checked(root, READINESS))
    require(readiness['packages']['masks'] == PARENT, 'Changed region parent')
    package = Package(root, PARENT)
    pages = package.json('page-regions.json'); selection = package.json('selection.json')
    values = [read(checked(root, p)) for p in readiness['metadata']]
    groups, sources, registry1, registry2, fresh = values
    boundary = Boundary(groups['groups'], sources, [registry1, registry2],
        [{'source_path': s['input'], 'source_sha256': s['input_sha256']} for s in fresh['selected_sources']])
    by_id = {s['id']: s for s in selection['sources']}
    require(len(by_id) == len(pages) == 6 and sum(len(p['regions']) for p in pages) == 22, 'Changed coarse region inventory')
    approved = []; excluded = []
    # Screen ALL metadata before opening even the overview of any source.
    for page in pages:
        source = by_id[page['source_id']]; s = source['source_record']
        require(page['source'] == {'path': 'dataset/'+s['input'], 'sha256': s['input_sha256']}
                and page['source_size_px'] == s['source_size_px'], 'Coarse source rebound')
        result = boundary.check(page['source'], s['declared_split'], page['source_id'])
        require(result['group_id'] == source['group_id'], 'Group ancestry changed')
        item = {'id': page['source_id'], 'source': page['source'], 'split': s['declared_split'],
                'group_id': source['group_id'], 'size_px': s['source_size_px'],
                'preparation': s['preparation'], 'proposals': page['regions'], 'screening': result}
        (excluded if result['blocked'] else approved).append(item)
    return package, approved, excluded


def authenticated_source(root, source):
    checked(root, source['source'])
    prep = checked(root, source['preparation']); report = read(prep)
    require(report['source_sha256'] == source['source']['sha256'] and report['source_size_px'] == source['size_px']
            and report['source_integrity_validated'] is True and report['source_modified'] is False
            and report['status'] == 'prepared', 'Native preparation mismatch')
    return prep, report


def native(root, source, box, *, maximum_pixels=1048576):
    path = checked(root, source['preparation'])
    require(read(path)['source_sha256'] == source['source']['sha256'], 'Native source mismatch')
    pixels, proof = read_prepared_region(path.parent, box, max_pixels=maximum_pixels)
    return pixels, proof


def review_inputs(root, folder):
    require(not folder.exists() and folder.parent == root / 'artifacts', 'Choose new review directory')
    prior = preserve(root); package, sources, excluded = context(root)
    folder.mkdir(); overview_rows = []
    for source in sources:
        prep, report = authenticated_source(root, source)
        name = 'overviews/'+source['id']+'.png'
        # The historical copy is pinned independently of the preparation pointer.
        path = package.path(name); original = report['overview']
        overview_pin = {'path': (prep.parent / original['path']).relative_to(root).as_posix(), 'sha256': original['sha256']}
        checked(root, overview_pin)
        require(pin(root, path.relative_to(root).as_posix())['sha256'] == original['sha256'], 'Overview pixels differ')
        with Image.open(path) as image:
            size = list(image.size)
        overview_rows.append({'source_id': source['id'], 'overview': pin(root, path.relative_to(root).as_posix()),
                              'size_px': size, 'source_to_overview': original['source_to_overview'],
                              'overview_to_source': original['overview_to_source'], 'sampling': original['method']})
    value = {'schema_version': 'page-region-review-inputs/1.0', 'policy': POLICY,
             'previous': PREVIOUS, 'parent': PARENT, 'readiness': READINESS,
             'sources': sources, 'excluded_sources': excluded, 'overviews': overview_rows,
             'coarse_regions': 22, 'protected_before_review': len(prior), 'model_calls': 0}
    (folder / 'selection.json').write_bytes(encoded(value))
    return {'selection': pin(root, (folder / 'selection.json').relative_to(root).as_posix()),
            'sources': len(sources), 'excluded': len(excluded), 'splits': dict(Counter(s['split'] for s in sources))}


READING = 'configs/reviews/page-region-reading-v2.json'
INPUTS = {'path': 'artifacts/page-region-review-inputs-v1/selection.json',
          'sha256': 'd2b855ace04836f42b6c5be1b5cb84d6023849cfd75578b7cadf73488cc12eb3'}


def rectangle(box, size):
    require(isinstance(box, list) and len(box) == 4
            and all(type(v) is int for v in box), 'Integer pixel-edge box required')
    x, y, right, bottom = box
    require(0 <= x < right <= size[0] and 0 <= y < bottom <= size[1], 'Box outside native page')
    require((right-x)*(bottom-y) <= POLICY['maximum_region_pixels'], 'Region pixel budget exceeded')
    return [[x, y], [right, y], [right, bottom], [x, bottom]]


def translations(box):
    x, y = box[:2]
    return {'quarter_turns_ccw': 0, 'source_to_region': [[1,0,-x],[0,1,-y],[0,0,1]],
            'region_to_source': [[1,0,x],[0,1,y],[0,0,1]],
            'coordinates': 'pixel_edges; centres use the same translation; no resize or rotation'}


def tiles(box, side=1024):
    x0, y0, x1, y1 = box
    return [[x,y,min(x+side,x1),min(y+side,y1)]
            for y in range(y0,y1,side) for x in range(x0,x1,side)]


def reading(root, sources):
    value = read(checked(root, pin(root, READING)))
    require(value['selection'] == INPUTS and value['reviewer'] == POLICY['reviewer']
            and value['human_verified'] is False and value['disputed'] == [], 'Review authority mismatch')
    by_id = {s['id']: s for s in sources}; used = Counter(); total = 0; count = 0
    require(0 < len(value['regions']) <= POLICY['maximum_regions'], 'Region count budget exceeded')
    for row in value['regions']:
        source = by_id[row['source_id']]
        require(row['role'] in POLICY['classes'] and row['role'] != 'frame', 'Unsupported area supervision')
        require(type(row['proposal']) is int and 1 <= row['proposal'] <= len(source['proposals']), 'Unknown proposal')
        rectangle(row['box'], source['size_px'])
        require(bool(row['observation']), 'Missing review observation')
        used[source['id'],row['proposal']] += 1
        x,y,r,b = row['box']; total += (r-x)*(b-y); count += len(tiles(row['box']))
    for row in value['unknown_proposals']:
        require((row['source_id'],row['proposal']) not in used and row['reason'], 'Ambiguous proposal disposition')
        used[row['source_id'],row['proposal']] += 1
    expected = {(s['id'], i) for s in sources for i in range(1,len(s['proposals'])+1)}
    require(set(used) == expected, 'Missing or invented proposal dispositions')
    require(total <= POLICY['maximum_total_region_pixels'] and count <= POLICY['maximum_native_tiles'], 'Native review budget exceeded')
    return value


def extract_review(root, folder):
    """Materialize exact native tiles and bounded previews, never a dense mask."""
    require(not folder.exists() and folder.parent == root/'artifacts', 'Choose new evidence directory')
    package, sources, excluded = context(root)
    require(not excluded, 'Review selection changed')
    require(read(checked(root, INPUTS))['sources'] == sources, 'Review inputs changed')
    decisions = reading(root, sources); by_id = {s['id']: s for s in sources}
    for source in sources:
        authenticated_source(root, source)
    folder.mkdir(); (folder/'tiles').mkdir(); (folder/'previews').mkdir()
    rows = []
    for i, row in enumerate(decisions['regions'], 1):
        source = by_id[row['source_id']]; box = row['box']; x0,y0,x1,y1 = box
        factor = max(1, int(np.ceil(max(x1-x0,y1-y0)/1100)))
        # Piecewise nearest display is explicitly only an inspection preview.
        preview = Image.new('RGB', (int(np.ceil((x1-x0)/factor)),int(np.ceil((y1-y0)/factor))), 'white')
        entries = []
        for j, tile_box in enumerate(tiles(box)):
            pixels, proof = native(root, source, tile_box)
            name = f'tiles/region-{i:02d}-{j:03d}.png'; path = folder/name
            Image.fromarray(pixels).save(path)
            tx,ty,tr,tb = tile_box
            # Globally aligned sample positions avoid seams in the display.
            sx = (-(tx-x0)) % factor; sy = (-(ty-y0)) % factor
            reduced = pixels[sy::factor,sx::factor]
            if reduced.size:
                preview.paste(Image.fromarray(reduced), ((tx-x0+sx)//factor,(ty-y0+sy)//factor))
            entries.append({'asset': pin(root,path.relative_to(root).as_posix()), 'box': tile_box,
                            'transform': translations(tile_box), 'native_proof': proof})
        preview_name = f'previews/region-{i:02d}.png'; preview.save(folder/preview_name)
        rows.append({'id': f'region-{i:02d}', **row, 'polygon': rectangle(box,source['size_px']),
                     'transform': translations(box), 'tiles': entries,
                     'preview': pin(root,(folder/preview_name).relative_to(root).as_posix()),
                     'preview_sampling': {'method': 'nearest_native_integer_grid', 'factor': factor,
                                          'source_origin': [x0,y0], 'annotation_authority': False}})
    value = {'schema_version': 'page-region-native-evidence/1.0', 'selection': INPUTS,
             'reading': pin(root,READING), 'regions': rows, 'model_calls': 0}
    (folder/'evidence.json').write_bytes(encoded(value))
    return pin(root,(folder/'evidence.json').relative_to(root).as_posix())
