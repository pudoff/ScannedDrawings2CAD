"""Fresh, bounded pixel observations with explicit stage limits and provenance."""
from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path
import platform

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import connector_guard_v1 as geometry
from . import axis_dash_search_v1 as axes
from . import open_arc_search_v1 as arcs
from . import arrow_path_recovery_v1 as paths
from . import arrow_head_validation_v1 as heads
from .dataset_layout_v1 import digest, encoded, inside, pin, read
from .family_catalog_v2 import require
from .raster_regions import read_prepared_region

POLICY = {'id': 'native-cv-experimental/1.0', 'maximum_source_bytes': 50*1024**2,
    'maximum_direct_source_pixels': 25000000, 'maximum_analysis_pixels': 16777216,
    'maximum_text_bytes': 2*1024**2, 'maximum_text_polygons': 512, 'maximum_polygon_vertices': 32,
    'geometry_profile': 'regional_masked', 'axis_profile': 'local_contrast_components',
    'geometry_stroke_field_is_already_native': True, 'axes_feed_arrow_paths': False,
    'segmentation_filter_used': False, 'original_samples_resampled': False,
    'continuous_axis_inferred': False, 'physical_geometry_approved': False,
    'production_admitted': False, 'CAD_promotions': 0, 'exhaustive_recovery_claimed': False}
CODE = ['native_cv_profile_v1', 'connector_guard_v1', 'geometry_detection', 'circle_search_v2',
    'axis_dash_search_v1', 'open_arc_search_v1', 'arrow_path_recovery_v1', 'arrow_head_validation_v1',
    'arrow_head_completion_v1', 'arrow_detection', 'raster_regions', 'dataset_layout_v1', 'family_catalog_v2']


def code_pins():
    root = Path(__file__).resolve().parents[2]
    return [pin(root, 'src/cad_recovery/'+n+'.py') for n in CODE]


def box_bounds(box, size):
    require(isinstance(box, (list, tuple)) and len(box) == 4 and all(type(v) is int for v in box),
            'ROI must have four integer half-open bounds')
    require(0 <= box[0] < box[2] <= size[0] and 0 <= box[1] < box[3] <= size[1], 'ROI outside source')
    require((box[2]-box[0])*(box[3]-box[1]) <= POLICY['maximum_analysis_pixels'], 'ROI pixel budget exceeded')
    return list(box)


def load_pixels(*, image=None, prepared=None, roi=None):
    require((image is None) != (prepared is None), 'Choose exactly one image or preparation directory')
    if prepared is not None:
        require(roi is not None, 'Prepared sources require a native ROI')
        rgb, meta = read_prepared_region(Path(prepared), roi, max_pixels=POLICY['maximum_analysis_pixels'])
        return rgb, {'mode': 'prepared_roi', 'path': str(Path(prepared).resolve()),
            'sha256': meta['source_sha256'], 'size_px': meta['source_size_px'], 'roi': meta['source_box_px'],
            'native_to_source': meta['region_to_source'], 'source_to_native': meta['source_to_region'],
            'original_rehashed': False, 'integrity': meta}
    path = Path(image).resolve()
    require(path.is_file() and 0 < path.stat().st_size <= POLICY['maximum_source_bytes'], 'Image byte budget or missing file')
    sha = digest(path)
    with Image.open(path) as header:
        require(header.format in ('PNG', 'JPEG') and getattr(header, 'n_frames', 1) == 1, 'Expected one PNG/JPEG frame')
        require(header.width*header.height <= POLICY['maximum_direct_source_pixels'],
                'Large source: use validated preparation tiles and --roi')
        size = list(header.size); mode = header.mode
        require(mode in ('RGB', 'L', 'RGBA'), 'Expected RGB, grayscale or opaque RGBA image')
        box = box_bounds(roi if roi is not None else [0, 0, *size], size)
        header.verify()
    with Image.open(path) as source:
        if mode == 'RGBA':
            require(source.getchannel('A').getextrema() == (255, 255), 'Nonopaque source needs an explicit separate preparation')
        require('transparency' not in source.info, 'Transparent source needs separate preparation')
        rgb = np.array(source.crop(tuple(box)).convert('RGB'))
    require(digest(path) == sha, 'Input changed while decoding')
    x, y = box[:2]
    return rgb, {'mode': 'direct_roi' if roi is not None else 'direct_image', 'path': str(path), 'sha256': sha,
        'size_px': size, 'roi': box, 'native_to_source': [[1, 0, x], [0, 1, y], [0, 0, 1]],
        'source_to_native': [[1, 0, -x], [0, 1, -y], [0, 0, 1]], 'original_rehashed': True,
        'integrity': {'decoded_source_sha256_verified': True, 'RGB_samples_preserved': True,
                      'EXIF_orientation_applied': False}}


def load_text(path, source):
    if path is None: return [], {'provided': False, 'OCR_executed': False}
    path = Path(path).resolve()
    require(path.is_file() and 0 < path.stat().st_size <= POLICY['maximum_text_bytes'], 'Text evidence byte budget')
    sha = digest(path); value = read(path)
    require(value['source_sha256'] == source['sha256'] and value['source_size_px'] == source['size_px']
        and value['coordinate_space'] == 'original_source_pixels', 'Text evidence source/frame mismatch')
    polygons = value['polygons']; require(isinstance(polygons, list) and len(polygons) <= POLICY['maximum_text_polygons'], 'Text polygon count budget')
    matrix = np.asarray(source['source_to_native'], float); result = []
    for polygon in polygons:
        p = np.asarray(polygon, float)
        require(p.ndim == 2 and p.shape[1] == 2 and 3 <= len(p) <= POLICY['maximum_polygon_vertices']
            and np.isfinite(p).all() and np.abs(p).max() <= 1e7, 'Invalid source OCR polygon')
        result.append((p @ matrix[:2, :2].T+matrix[:2, 2]).tolist())
    require(digest(path) == sha, 'Text evidence changed while reading')
    return result, {'provided': True, 'path': str(path), 'sha256': sha,
                    'polygons': len(result), 'OCR_executed': False, 'authority': 'saved proposal exclusions only'}


def stage_statuses(g, axis, arc, path, head):
    analysis = g['analysis']; limits = []
    for key, value in analysis.items():
        if (key.endswith('_limit_reached') and value is True) or key == 'validation_queue_dropped' and value > 0:
            limits.append(key)
    if analysis.get('hough_segments', 0) > 2500: limits.append('hough_line_budget')
    if analysis.get('proposal_exclusions', {}).get('mask_records_truncated'): limits.append('mask_records_truncated')
    for issue in g['issues']:
        if issue['code'] in ('geometry_candidate_limit', 'skeleton_iteration_limit'): limits.append(issue['code'])
    output = {'geometry': {'search_incomplete': bool(limits), 'limits': sorted(set(limits)),
                           'issues': deepcopy(g['issues']), 'analysis_scale': analysis['scale']}}
    for name, value, reasons in [('axis', axis, axis['analysis']['limits'] +
            (['axis_group_budget'] if axis['analysis']['group_limit_reached'] else [])),
        ('open_arc', arc, arc['analysis']['limits']), ('arrow_path', path, path['analysis']['limits']),
        ('head_seed', head['raw_auxiliary'], head['raw_auxiliary']['analysis']['limits']),
        ('head_shape', head['validated_auxiliary'], head['validated_auxiliary']['limits']),
        ('attached_head', head['attached'], head['attached']['analysis']['limits'])]:
        require(type(value['search_incomplete']) is bool, 'Missing upstream incompleteness flag')
        output[name] = {'search_incomplete': value['search_incomplete'] or bool(reasons), 'limits': sorted(set(reasons))}
    return output


def observed_support_ids(value):
    result = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ('geometry_ids', 'supporting_geometry_ids', 'observed_junction_lines', 'occluding_line_ids'):
                result.extend(item)
            elif key in ('path_id', 'geometry_id') and item is not None: result.append(item)
            else: result.extend(observed_support_ids(item))
    elif isinstance(value, list):
        for item in value: result.extend(observed_support_ids(item))
    return sorted(set(result))


def observations(g, axis, arc, head, source, native_sha):
    groups = [('geometry', 'geometry', g['candidates']), ('axis', 'axis', axis['candidates']),
              ('open_arc', 'arc', arc['candidates']), ('arrow_head', 'arrow', head['candidates'])]
    raw_map = {}; ids = {}; output = []
    for stage, prefix, candidates in groups:
        for i, c in enumerate(candidates, 1):
            require(c['id'] not in raw_map, 'Ambiguous cross-stage producer ID')
            identity = prefix+'-'+str(i).zfill(5); raw_map[c['id']] = identity; ids[(stage, c['id'])] = identity
    for stage, _, candidates in groups:
        for c in candidates:
            supports = observed_support_ids(c.get('evidence', {}))
            require(all(v in raw_map for v in supports), 'Dangling supporting observation ID')
            fields = {k: deepcopy(v) for k, v in c.items() if k not in ('id', 'source_ref', 'evidence')}
            output.append({**fields, 'id': ids[(stage, c['id'])],
                'producer_ref': {'stage': stage, 'id': c['id']}, 'producer_evidence': deepcopy(c.get('evidence', {})),
                'supporting_observation_ids': [raw_map[v] for v in supports],
                'source_ref': {'sha256': source['sha256'], 'native_image_sha256': native_sha,
                    'coordinate_space': 'native_analysis_pixels', 'native_to_source': source['native_to_source']},
                'acceptance': 'unverified_pixel_observation', 'physical_geometry_approved': False})
    patterns = []
    for group in axis['groups']:
        members = [c for c in axis['candidates'] if c['evidence']['group_id'] == group['id']]
        patterns.append({'id': group['id'], 'observation_ids': [raw_map[c['id']] for c in members],
                         'producer_evidence': deepcopy(group), 'continuous_axis_inferred': False})
    return output, patterns


def draw_overlay(rgb, candidates):
    image = Image.fromarray(rgb); image.thumbnail((1600, 1600)); draw = ImageDraw.Draw(image)
    sx, sy = image.width/rgb.shape[1], image.height/rgb.shape[0]
    for c in candidates:
        if 'polygon_px' in c: points = c['polygon_px']+[c['polygon_px'][0]]; color = '#dd3232'
        elif c.get('kind') == 'line':
            points = c['points_px']; color = '#a642d6' if c['producer_ref']['stage'] == 'axis' else '#0085ac'
        else:
            start, sweep = c.get('start_angle_deg', 0), c.get('sweep_angle_deg', 360)
            a = np.radians(np.linspace(start, start+sweep, 128))
            points = (np.asarray(c['center_px'])+c['radius_px']*np.c_[np.cos(a), np.sin(a)]).tolist(); color = '#00a050'
        draw.line([(x*sx, y*sy) for x, y in points], fill=color, width=1)
    stream = BytesIO(); image.save(stream, format='PNG'); return stream.getvalue(), [sx, sy]


def analyze(output, *, image=None, prepared=None, roi=None, text_evidence=None):
    output = Path(output).resolve()
    require(not output.exists() and not output.is_symlink(), 'Experimental CV never overwrites an output directory')
    rgb, source = load_pixels(image=image, prepared=prepared, roi=roi)
    polygons, exclusion = load_text(text_evidence, source)
    producer_code = code_pins()
    cv2.setNumThreads(4); cv2.setRNGSeed(20261110)
    output.mkdir(parents=True); completed = []
    try:
        Image.fromarray(rgb).save(output/'native.png'); native_sha = digest(output/'native.png')
        (output/'source.json').write_bytes(encoded(source)); gray = np.asarray(Image.fromarray(rgb).convert('L'))
        g = geometry.detect_geometry(output/'native.png', output/'geometry', search_profile=POLICY['geometry_profile'])
        g.pop('timings', None); (output/'geometry'/g['artifacts']['report']).write_bytes(encoded(g)); completed.append('geometry')
        # geometry_detection already returns this width in native processed pixels.
        stroke = g['analysis']['estimated_stroke_width_px']
        require(np.isfinite(stroke) and 1 <= stroke <= 24, 'Native stroke estimate outside supported range; use an appropriate ROI')
        axis = axes.search(gray, profile=POLICY['axis_profile']); completed.append('axis')
        arc = arcs.search(gray, stroke_px=stroke, lines=g['candidates'], text_polygons=polygons); completed.append('open_arc')
        path = paths.search(gray, lines=g['candidates'], arcs=arc['candidates'], stroke_px=stroke, text_polygons=polygons); completed.append('arrow_path')
        head = heads.search(gray, baseline=path['candidates'], lines=g['candidates'], stroke_px=stroke, text_polygons=polygons); completed.append('arrow_head')
        for name, value in [('axis', axis), ('open-arcs', arc), ('arrow-paths', path), ('arrow-heads', head)]:
            (output/(name+'.json')).write_bytes(encoded(value))
        candidates, patterns = observations(g, axis, arc, head, source, native_sha)
        statuses = stage_statuses(g, axis, arc, path, head)
        overlay, scale = draw_overlay(rgb, candidates); (output/'overlay.png').write_bytes(overlay)
        require(code_pins() == producer_code, 'Producer code changed during analysis')
        report = {'schema_version': 'native-cv-report/1.0', 'profile': POLICY, 'status': 'needs_review',
            'source': source, 'native_image': {'path': 'native.png', 'sha256': native_sha},
            'native_size_px': [rgb.shape[1], rgb.shape[0]], 'native_stroke_width_px': stroke,
            'coordinate_system': 'native pixel centres; x right, y down; angles clockwise from +x',
            'source_coordinate_system': 'original input pixel centres, with explicit native_to_source transform',
            'exclusions': exclusion, 'observations': candidates, 'axis_patterns': patterns,
            'stage_statuses': statuses, 'search_incomplete': any(v['search_incomplete'] for v in statuses.values()),
            'incomplete_stages': [k for k, v in statuses.items() if v['search_incomplete']],
            'preview_scale_xy': scale, 'code': producer_code,
            'runtime_versions': {'python': platform.python_version(), 'opencv': cv2.__version__, 'numpy': np.__version__},
            'training_executed': False, 'OCR_executed': False, 'neural_inference_executed': False,
            'production_admitted': False, 'physical_geometry_approved': False, 'CAD_promotions': 0,
            'limitations': ['Separate experimental observations, no default pipeline switch or CAD acceptance.',
                'Stage limits are aggregated; absence of limits is not proof of exhaustive recovery.',
                'Axes retain finite ink members; no gap is joined and no complete axis/circle is invented.',
                'Producer evidence keeps its original stage-local IDs; supporting_observation_ids names the unified IDs.',
                'Prepared ROI integrity covers metadata/intersecting tiles; original source rehash status is explicit.']}
        (output/'report.json').write_bytes(encoded(report))
        files = sorted(p.relative_to(output).as_posix() for p in output.rglob('*') if p.is_file())
        (output/'manifest.json').write_bytes(encoded({'schema_version': 'native-cv-manifest/1.0',
            'profile': POLICY['id'], 'files': [pin(output, n) for n in files]}))
        return report
    except Exception as error:
        (output/'failure.json').write_bytes(encoded({'status': 'failed', 'completed_stages': completed,
            'error_type': type(error).__name__, 'message': str(error), 'production_admitted': False}))
        raise


def verify(output):
    folder = Path(output).resolve(); manifest = read(folder/'manifest.json')
    require(manifest['schema_version'] == 'native-cv-manifest/1.0' and manifest['profile'] == POLICY['id'], 'Wrong native CV packet')
    expected = {'native.png', 'source.json', 'geometry/page-001-geometry.json', 'geometry/page-001-geometry-preview.png',
        'axis.json', 'open-arcs.json', 'arrow-paths.json', 'arrow-heads.json', 'overlay.png', 'report.json'}
    require({r['path'] for r in manifest['files']} == expected and
        {p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()} == expected | {'manifest.json'},
        'Changed native CV inventory')
    for r in manifest['files']:
        require(pin(folder, r['path']) == r, 'Changed native CV file')
    report = read(folder/'report.json')
    require(report['profile'] == POLICY and report['code'] == code_pins() and report['production_admitted'] is False
        and report['physical_geometry_approved'] is False and report['CAD_promotions'] == 0
        and all(report[k] is False for k in ('training_executed', 'OCR_executed', 'neural_inference_executed')),
        'Changed native CV scope/code')
    ids = {v['id'] for v in report['observations']}
    require(len(ids) == len(report['observations']) and all(set(v['supporting_observation_ids']) <= ids for v in report['observations']),
            'Dangling/duplicate unified observation IDs')
    source = read(folder/'source.json')
    require(source == report['source'] and report['native_image'] == pin(folder, 'native.png'), 'Changed source provenance')
    box = box_bounds(source['roi'], source['size_px']); x, y = box[:2]
    require(source['native_to_source'] == [[1, 0, x], [0, 1, y], [0, 0, 1]] and
        source['source_to_native'] == [[1, 0, -x], [0, 1, -y], [0, 0, 1]] and
        report['native_size_px'] == [box[2]-box[0], box[3]-box[1]], 'Changed crop coordinate transform')
    with Image.open(folder/'native.png') as picture:
        require(list(picture.size) == report['native_size_px'] and picture.mode == 'RGB', 'Changed native raster shape')
    g, axis, arc, path, head = [read(folder/n) for n in ('geometry/page-001-geometry.json',
        'axis.json', 'open-arcs.json', 'arrow-paths.json', 'arrow-heads.json')]
    require(g['source_sha256'] == report['native_image']['sha256'] and
        g['processed_size_px'] == report['native_size_px'] and
        report['native_stroke_width_px'] == g['analysis']['estimated_stroke_width_px'], 'Changed native geometry frame/stroke')
    statuses = stage_statuses(g, axis, arc, path, head)
    current, patterns = observations(g, axis, arc, head, source, report['native_image']['sha256'])
    require(report['observations'] == current and report['axis_patterns'] == patterns and
        report['stage_statuses'] == statuses and report['incomplete_stages'] ==
        [k for k, v in statuses.items() if v['search_incomplete']] and
        report['search_incomplete'] == any(v['search_incomplete'] for v in statuses.values()),
        'Changed stage composition or hidden upstream limit')
    return report
