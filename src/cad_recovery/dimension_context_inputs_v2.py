"""Prepare bounded automatic observations without inventing fits at crop edges."""
from copy import deepcopy
import shutil

import numpy as np
from PIL import Image

from .dataset_layout_v1 import checked, pin, read, encoded
from .grid_role_revision_v1 import cv_authority, sha
from .native_dimension_roles_v1 import _authority
from .part_admission_v1 import object_sha
from .source_arrow_observation_v1 import checked_matrix, transform


def working_bounds(matrix, source_box, size):
    matrix = checked_matrix(matrix)
    if not np.allclose(matrix[:2, :2], np.diag(np.diag(matrix[:2, :2]))) or np.any(np.diag(matrix)[:2] <= 0):
        raise ValueError('This crop adapter requires positive axis-aligned registration')
    inverse = np.linalg.inv(matrix)
    low = np.maximum(0, np.ceil(transform([source_box[:2]], inverse)[0])).astype(int)
    high = np.minimum(size, np.floor(transform([[source_box[2]-1, source_box[3]-1]], inverse)[0])+1).astype(int)
    if np.any(high <= low):
        raise ValueError('Input contains no working pixel centres')
    local = matrix.copy()
    local[:2, 2] += matrix[:2, :2] @ low - source_box[:2]
    return low, high, local


def _points(row):
    points = np.asarray(row.get('support_points', row.get('points_px', [])), float)
    if points.ndim != 2 or points.shape[1] != 2 or not len(points) or not np.isfinite(points).all():
        raise ValueError('Finite nonempty observation support required')
    return points


def clip_geometry(geometry, low, high, source_sha):
    low, high = np.asarray(low), np.asarray(high)
    result = deepcopy(geometry)
    candidates, intervals, cut, fragments = [], [], [], []
    for row in geometry['candidates']:
        points = _points(row)
        selected = ((points >= low).all(1) & (points < high).all(1))
        if not selected.any():
            continue
        if not selected.all():
            # Keep its observed ink below, but do not turn a partial circle into
            # a circle or bridge separated supports with a newly fitted line.
            cut.append(row['id'])
            continue
        local = deepcopy(row)
        local['support_points'] = (points-low).tolist()
        for field in ('points_px', 'center_px'):
            if field in local:
                local[field] = (np.asarray(local[field])-low).tolist()
        candidates.append(local)
    for row in geometry.get('ownership_intervals', []):
        points = _points(row)
        selected = np.flatnonzero((points >= low).all(1) & (points < high).all(1))
        if not len(selected):
            continue
        runs = np.split(selected, np.flatnonzero(np.diff(selected) != 1)+1)
        for i, run in enumerate(runs):
            local = deepcopy(row)
            partial = len(run) != len(points)
            if partial:
                local.update(id=f"{row['id']}/crop/{i}", parent_interval_id=row['id'],
                             parent_support_range=[int(run[0]), int(run[-1])+1])
                fragments.append(local['id'])
            local['support_points'] = (points[run]-low).tolist()
            intervals.append(local)
    result.update(source_sha256=source_sha, processed_size_px=(high-low).tolist(),
                  candidates=candidates, ownership_intervals=intervals,
                  complete=geometry.get('complete') is True and not cut,
                  source='bounded_automatic_observations_no_boundary_refitting')
    audit = dict(typed_before=len(geometry['candidates']), typed_retained=len(candidates),
                 boundary_typed_withheld=cut, intervals_before=len(geometry.get('ownership_intervals', [])),
                 intervals_retained=len(intervals), interval_fragments=fragments,
                 observation_complete=result['complete'], new_fit=False)
    return result, audit


def save(root, folder, name, value):
    path = folder/name
    with path.open('xb') as stream:
        stream.write(encoded(value))
    return pin(root, path.relative_to(root).as_posix())


def prepare(root, folder, base, request):
    """Only the declared source/mode/box and automatic producer are consumed."""
    source = checked(root, request['source'])
    if base['source'] != request['source']:
        raise ValueError('Automatic observations belong to another source')
    obs = base['observations']
    matrix = checked_matrix(base['working_to_source'])
    with Image.open(source) as image:
        size = image.size
        box = request['source_input_box']
        if len(box) != 4 or not all(type(x) is int for x in box):
            raise ValueError('Integer source box required')
        x, y, right, bottom = box
        if not (0 <= x < right <= size[0] and 0 <= y < bottom <= size[1]):
            raise ValueError('Input box outside source')
        mode = request['input_mode']
        if mode not in ('source_page', 'standalone_crop'):
            raise ValueError('Explicit input mode required')
        if mode == 'source_page' and box != [0, 0, *size]:
            raise ValueError('Source-page request must expose the entire page')
        expected = [[1, 0, x], [0, 1, y], [0, 0, 1]]
        if request['local_to_source'] != expected:
            raise ValueError('Request coordinate transform differs')
        original_pixels = image.convert('L').crop(box)
    geometry = read(checked(root, obs['geometry.json']))
    with Image.open(checked(root, obs['source.png'])) as image:
        low, high, local_matrix = working_bounds(matrix, box, image.size)
        working = np.array(image.convert('L').crop([*low, *high]))
    with Image.open(checked(root, obs['roles.png'])) as image:
        if list(image.size) != geometry['processed_size_px']:
            raise ValueError('Role frame differs from geometry')
        roles = np.array(image.convert('L').crop([*low, *high]))
    if roles.shape != working.shape:
        raise ValueError('Working and role frames differ')
    folder.mkdir(parents=True, exist_ok=False)
    if mode == 'source_page':
        shutil.copyfile(source, folder/'input.png')
    else:
        original_pixels.convert('RGB').save(folder/'input.png')
    Image.fromarray(working).save(folder/'working.png')
    Image.fromarray(roles).save(folder/'input-roles.png')
    source_ref = pin(root, (folder/'input.png').relative_to(root).as_posix())
    working_ref = pin(root, (folder/'working.png').relative_to(root).as_posix())
    local_geometry, audit = clip_geometry(geometry, low, high, working_ref['sha256'])
    geometry_ref = save(root, folder, 'geometry.json', local_geometry)
    code = [*base['automatic_input_code'], pin(root, 'src/cad_recovery/dimension_context_inputs_v2.py')]
    for ref in code:
        checked(root, ref)
    origin = dict(schema_version='automatic-dimension-role-input/1.0',
                  working_gray_sha256=sha(working), roles_sha256=sha(roles),
                  original_source_sha256=source_ref['sha256'], reviewed_input=False,
                  code=code, input_role_file=pin(root, (folder/'input-roles.png').relative_to(root).as_posix()),
                  geometry=geometry_ref, geometry_object_sha256=object_sha(local_geometry),
                  original_automatic_source=base['source'], automatic_observations=obs,
                  working_origin=low.tolist(), source_input_box=box,
                  scope='bounded_automatic_pipeline_input_not_reviewed_truth')
    origin_ref = save(root, folder, 'automatic-role-origin.json', origin)
    protected = np.zeros_like(roles)
    authority = cv_authority(working, roles, origin_ref, protected=protected)
    validate_authority(root, working, roles, source_ref, local_geometry, authority)
    ocr = dict(source_sha256=working_ref['sha256'], processed_size_px=list(working.shape[::-1]),
               lines=[], notations=[], availability='not_run')
    evidence = save(root, folder, 'prepared.json', dict(request=request, source=source_ref,
        working=working_ref, geometry=geometry_ref, origin=origin_ref, clipping=audit,
        working_to_local_source=local_matrix.tolist(), new_full_CV_run=False,
        raw_OCR_not_executed=True, authority_checked=True))
    return dict(original=folder/'input.png', working=folder/'working.png', geometry=local_geometry,
                roles=roles, gray=working, ocr=ocr, working_to_source=local_matrix,
                authority=authority, protected=protected, evidence=evidence)


def validate_authority(root, gray, roles, source_ref, geometry, authority):
    checked(root, source_ref)
    permitted = _authority(gray, roles, authority, root, source_ref['sha256'])
    producer = read(checked(root, authority['producer']))
    bound_geometry = read(checked(root, producer['geometry']))
    if object_sha(geometry) != producer['geometry_object_sha256'] or geometry != bound_geometry:
        raise ValueError('Local authority geometry differs')
    return permitted
