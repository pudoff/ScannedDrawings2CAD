"""Deterministic pixel indicators; no OCR, acceptance gate or physical inference.

These operational review heuristics are deliberately distinct from calibrated
blur/fold/compression classifiers. Text and model predictions are not inputs.
"""
import cv2
import numpy as np


POLICY = {
    'schema_version': 'raster-quality-policy/1.0',
    'threshold_status': 'operational_review_heuristics_not_calibrated_quality_gate',
    'thresholds_chosen_after_viewed_baseline': True,
    'thresholds_frozen_before_new_preprocessing_comparison': True,
    'short_context_side_px': 512,
    'small_component_median_height_px': 12,
    'low_robust_contrast_8bit': 64,
    'minimum_ink_span_8bit': 8,
    'low_laplacian_variance_8bit': 80,
    'background_tile_spread_8bit': 24,
    'background_grid': 4,
    'context_analysis_max_side_px': 1024,
    'component_min_area_px': 3,
    'component_min_height_px': 2,
    'component_max_width_height_ratio': 5,
    'component_max_height_width_ratio': 12,
    'long_stroke_span_fraction': .8,
    'long_stroke_thickness_fraction': .08,
    'merged_component_area_fraction': .2,
    'merged_component_span_fraction': .8,
    'maximum_component_records': 100,
    'jpeg_boundary_period_px': 8,
    'jpeg_boundary_ratio': 1.5,
    'jpeg_boundary_excess_8bit': 2,
    'jpeg_min_boundary_lines': 4,
    'manual_artifact_margin_px': 3,
    'physical_scale_inferred': False,
    'automatic_orientation_implemented': False,
    'automatic_fold_classifier_implemented': False,
    'automatic_quality_acceptance': False,
}

REASONS = {
    'small_context': 'Короткая сторона контекста меньше 512 px; DPI не подтверждает детализацию.',
    'small_ink_components': 'Медиана высот подходящих тёмных компонент меньше 12 px; мелкие знаки требуют проверки.',
    'low_robust_contrast': 'Разность 99-го и 1-го перцентилей яркости меньше 64/255.',
    'insufficient_ink_support': 'Не выделены подходящие тёмные компоненты; это не доказательство отсутствия текста.',
    'low_high_frequency_detail': 'Низкая дисперсия лапласиана; возможны размытие, бледность или малая доля штрихов.',
    'uneven_background_proxy': 'Разброс фоновых 90-х перцентилей плиток больше 24/255; причина не установлена.',
    'periodic_boundary_jumps_proxy': 'На глобальной сетке 8 px повышены перепады; это не подтверждённая JPEG-блочность.',
    'long_stroke_in_region': 'Через текстовую область проходит длинная тонкая компонента; возможна линия или часть знака.',
    'large_connected_ink_proxy': 'Крупная связная компонента занимает обе оси области; возможны соединённые штрихи.',
    'reviewed_artifact_near_region': 'Область пересекает коридор ранее вручную отмеченного артефакта.',
}


def require(value, message):
    if not value: raise ValueError(message)


def number(value):
    return round(float(value), 6)


def upright_matrix(source_box, rotation_cw):
    """Original source pixel centers to upright query centers, no interpolation."""
    require(isinstance(source_box, (list, tuple)) and len(source_box) == 4, 'Invalid source box')
    left, top, right, bottom = source_box
    require(all(type(x) is int for x in source_box) and right > left >= 0 and bottom > top >= 0, 'Invalid source box')
    require(type(rotation_cw) is int and rotation_cw in (0, 90, 180, 270), 'Only right-angle rotations')
    w, h = right-left, bottom-top
    rotations = {0: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                 90: [[0, -1, h-1], [1, 0, 0], [0, 0, 1]],
                 180: [[-1, 0, w-1], [0, -1, h-1], [0, 0, 1]],
                 270: [[0, 1, 0], [-1, 0, w-1], [0, 0, 1]]}
    translate = np.array([[1, 0, -left], [0, 1, -top], [0, 0, 1]])
    forward = np.array(rotations[rotation_cw]) @ translate
    inverse = np.rint(np.linalg.inv(forward)).astype(int)
    return {'source_to_upright': forward.tolist(), 'upright_to_source': inverse.tolist(),
            'pixel_convention': 'centers; boxes are exclusive at right/bottom',
            'upright_size_px': [h, w] if rotation_cw in (90, 270) else [w, h]}


def source_component_box(box, inverse):
    x, y, right, bottom = box
    corners = np.array([[x, y, 1], [right-1, y, 1], [right-1, bottom-1, 1], [x, bottom-1, 1]])
    points = (np.asarray(inverse) @ corners.T).T[:, :2]
    return [int(points[:, 0].min()), int(points[:, 1].min()),
            int(points[:, 0].max())+1, int(points[:, 1].max())+1]


def boundary_jumps(gray, origin):
    """Preserve original grid phase; drawing edges can also cause this proxy."""
    boundary, interior, lines = [], [], 0
    for axis, start in ((1, origin[0]), (0, origin[1])):
        values = np.abs(np.diff(gray.astype(np.int16), axis=axis)).astype(float)
        mask = (np.arange(1, gray.shape[axis])+start) % POLICY['jpeg_boundary_period_px'] == 0
        lines += int(mask.sum())
        if axis == 1:
            boundary.append(values[:, mask].ravel()); interior.append(values[:, ~mask].ravel())
        else:
            boundary.append(values[mask, :].ravel()); interior.append(values[~mask, :].ravel())
    boundary = np.concatenate(boundary); interior = np.concatenate(interior)
    b = float(boundary.mean()) if boundary.size else None
    other = float(interior.mean()) if interior.size else None
    valid = lines >= POLICY['jpeg_min_boundary_lines'] and b is not None and other is not None and other > 0
    return {'boundary_lines': lines, 'boundary_mean_abs_jump': number(b) if b is not None else None,
            'interior_mean_abs_jump': number(other) if other is not None else None,
            'ratio': number(b/other) if valid else None,
            'excess_8bit': number(b-other) if valid else None,
            'grid_origin_px': list(origin), 'diagnostic_only': True}


def artifact_hits(box, guides):
    margin = POLICY['manual_artifact_margin_px']
    x, y, right, bottom = box; hits = []
    for guide in guides:
        for a, b in zip(guide['points'], guide['points'][1:]):
            a, b = tuple(round(float(v)) for v in a), tuple(round(float(v)) for v in b)
            intersects, _, _ = cv2.clipLine((x-margin, y-margin, right-x+margin*2, bottom-y+margin*2), a, b)
            if intersects:
                hits.append(guide['id']); break
    return hits


def analyze_pixels(rgb, *, source_box, rotation_cw=0, codec='PNG', scope='text', manual_artifacts=()):
    """Measure an original rectangular RGB crop, optionally using known rotation.

    No transcript, OCR confidence, DPI, class target or error flag is accepted.
    Scope=context measures a declared reduced analysis image without glyph claims.
    """
    require(isinstance(rgb, np.ndarray) and rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3,
            'Expected uint8 RGB pixels')
    height, width = rgb.shape[:2]
    require(min(width, height) > 0 and width*height <= 25_000_000, 'Invalid pixel budget')
    require(scope in ('text', 'context') and (scope != 'text' or width*height <= 1_000_000), 'Unsupported analysis scope/region size')
    require(source_box[2]-source_box[0] == width and source_box[3]-source_box[1] == height, 'Pixels and source box disagree')
    transform = upright_matrix(source_box, rotation_cw)
    # Work only on a private copy. Rotation changes neither values nor resolution.
    native = cv2.cvtColor(np.array(rgb, copy=True), cv2.COLOR_RGB2GRAY)
    gray = np.rot90(native, -(rotation_cw//90)).copy()
    scale = min(1., POLICY['context_analysis_max_side_px']/max(gray.shape)) if scope == 'context' else 1.
    if scale < 1:
        gray = cv2.resize(gray, (max(1, round(gray.shape[1]*scale)), max(1, round(gray.shape[0]*scale))), interpolation=cv2.INTER_AREA)
    p1, p50, p99 = np.percentile(gray, [1, 50, 99]); span = float(p99-p1)
    laplacian = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    if span < POLICY['minimum_ink_span_8bit']: ink[:] = 0
    grid = POLICY['background_grid']; backgrounds = []
    for row in np.array_split(gray, min(grid, gray.shape[0]), axis=0):
        for tile in np.array_split(row, min(grid, gray.shape[1]), axis=1):
            backgrounds.append(float(np.percentile(tile, 90)))
    background_spread = float(np.percentile(backgrounds, 90)-np.percentile(backgrounds, 10))
    flags = []
    if scope == 'context' and min(width, height) < POLICY['short_context_side_px']: flags.append('small_context')
    if span < POLICY['low_robust_contrast_8bit']: flags.append('low_robust_contrast')
    if laplacian < POLICY['low_laplacian_variance_8bit']: flags.append('low_high_frequency_detail')
    if background_spread > POLICY['background_tile_spread_8bit']: flags.append('uneven_background_proxy')
    components = []; rejected = []; heights = []; long_count = merged_count = 0
    if scope == 'text':
        _, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        h, w = gray.shape
        for x, y, cw, ch, area in stats[1:]:
            x, y, cw, ch, area = (int(v) for v in (x, y, cw, ch, area))
            span_limit = POLICY['long_stroke_span_fraction']; thickness = POLICY['long_stroke_thickness_fraction']
            long = (cw >= span_limit*w and ch <= max(2, thickness*h)) or (ch >= span_limit*h and cw <= max(2, thickness*w))
            merged_span = POLICY['merged_component_span_fraction']
            merged = area >= POLICY['merged_component_area_fraction']*w*h and cw >= merged_span*w and ch >= merged_span*h
            long_count += int(long); merged_count += int(merged)
            eligible = area >= POLICY['component_min_area_px'] and ch >= POLICY['component_min_height_px']
            eligible = eligible and cw/ch <= POLICY['component_max_width_height_ratio'] and ch/cw <= POLICY['component_max_height_width_ratio'] and not long
            record = {'upright_box': [x, y, x+cw, y+ch], 'source_box': source_component_box([x, y, x+cw, y+ch], transform['upright_to_source']), 'area_px': area}
            if eligible: heights.append(ch); components.append(record)
            else: rejected.append(record)
        if not heights: flags.append('insufficient_ink_support')
        elif float(np.median(heights)) < POLICY['small_component_median_height_px']: flags.append('small_ink_components')
        if long_count: flags.append('long_stroke_in_region')
        if merged_count: flags.append('large_connected_ink_proxy')
    block = boundary_jumps(native, source_box[:2]) if codec.upper() == 'JPEG' else None
    if block and block['ratio'] is not None and block['ratio'] >= POLICY['jpeg_boundary_ratio'] and block['excess_8bit'] >= POLICY['jpeg_boundary_excess_8bit']:
        flags.append('periodic_boundary_jumps_proxy')
    hits = artifact_hits(source_box, manual_artifacts)
    if hits: flags.append('reviewed_artifact_near_region')
    limit = POLICY['maximum_component_records']
    return {'schema_version': 'raster-quality-measurement/1.0', 'scope': scope,
        'native_size_px': [width, height], 'analysis_size_px': [gray.shape[1], gray.shape[0]],
        'analysis_scale_xy': [gray.shape[1]/transform['upright_size_px'][0], gray.shape[0]/transform['upright_size_px'][1]],
        'rotation_cw': rotation_cw, 'rotation_authority': 'reviewed_reference_only_not_automatic', 'transform': transform,
        'luminance_percentiles_1_50_99': [number(v) for v in (p1, p50, p99)],
        'robust_contrast_8bit': number(span), 'laplacian_variance_8bit': number(laplacian),
        'ink_fraction_otsu': number(np.mean(ink != 0)), 'background_tile_spread_8bit': number(background_spread),
        'component_height_median_px': number(np.median(heights)) if heights else None,
        'eligible_component_count': len(heights), 'components': components[:limit],
        'rejected_component_count': len(rejected), 'rejected_components': rejected[:limit],
        'component_records_truncated': len(components) > limit or len(rejected) > limit,
        'long_stroke_components': long_count, 'large_connected_components': merged_count,
        'jpeg_container': codec.upper() == 'JPEG', 'periodic_grid_proxy': block,
        'reviewed_artifact_ids': hits, 'fold_presence_from_pixels': 'unknown',
        'blur_classification': 'unknown', 'compression_severity': 'unknown',
        'flags': sorted(flags), 'reasons': [REASONS[f] for f in sorted(flags)],
        'manual_review_required': True, 'physical_scale_inferred': False,
        'semantic_region_classification': 'not_implemented', 'quality_acceptance': None}
