"""Exclusive snapshots and checked loading of one source/model/frame CV bundle."""
from pathlib import Path
from copy import deepcopy
import json
import sys

import cv2
import numpy as np
from PIL import Image, __version__ as pillow_version

from .dataset_layout_v1 import checked, pin
from .grid_role_revision_v1 import sha
from .part_admission_v1 import object_sha
from .geometry_bundle_v1 import build, inspect, matrix, select, validate_geometry

FILES = {'raw_source': 'raw-source.json', 'proposed_geometry': 'proposed-geometry.json', 'source': 'source.json',
         'geometry': 'geometry.json', 'graph': 'graph.json', 'heads': 'heads.json',
         'head_observations': 'head-observations.json', 'chains': 'links.json'}
IMAGES = {'roles': 'roles.png', 'source_owner': 'source-owner.png', 'role_owner': 'role-owner.png',
          'ambiguous_owner_mask': 'ambiguous-owner.png', 'raw_ink': 'raw-ink.png',
          'preprocessing_excluded_ink': 'preprocessing-excluded-ink.png'}


def serializable(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {k: serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [serializable(v) for v in value]
    return value


def write(path, value):
    with path.open('x', encoding='utf-8') as f: json.dump(serializable(value), f, ensure_ascii=False, allow_nan=False, indent=2)


def model_binding(root, contract):
    """The expected model comes from the caller's pinned contract, not a bundle."""
    path = checked(root, contract)
    data = json.loads(path.read_text(encoding='utf-8'))
    required = {'src/cad_recovery/geometry_bundle_v1.py', 'src/cad_recovery/geometry_bundle_io_v1.py'}
    if not required <= {p['path'] for p in data['code']}: raise ValueError('Incomplete bundle producer code binding')
    for p in data['code']: checked(root, p)
    checked(root, data['runtime_manifest'])
    import skimage
    return {'contract': deepcopy(contract), 'code_sha256': object_sha(data['code']), 'skimage': skimage.__version__,
            'numpy': np.__version__, 'pillow': pillow_version,
            'runtime_manifest': data['runtime_manifest'], 'opencv': cv2.__version__,
            'python': list(sys.version_info[:3]), 'configuration': data['configuration']}


def capture(root, view, folder, *, model_contract):
    root = Path(root).resolve(); folder = Path(folder).resolve()
    folder.relative_to(root)
    if folder.exists(): raise FileExistsError(folder)
    source_file = checked(root, view['gray']); models = model_binding(root, model_contract)
    with Image.open(source_file) as image:
        if image.width*image.height > 4194304: raise ValueError('Bundle source image is too large')
        gray = np.array(image.convert('L'))
    packet = build(gray, input_to_native=view['input_to_native'], models=models)
    folder.mkdir(parents=True, exist_ok=False)
    Image.fromarray(gray).save(folder/'input.png')
    refs = {'input': pin(root, (folder/'input.png').relative_to(root).as_posix())}
    for key, name in FILES.items():
        write(folder/name, packet[key]); refs[key] = pin(root, (folder/name).relative_to(root).as_posix())
    for key, name in IMAGES.items():
        Image.fromarray(packet[key]).save(folder/name); refs[key] = pin(root, (folder/name).relative_to(root).as_posix())
    counts = {key: len(packet[key]) for key in ('source', 'geometry', 'heads', 'chains')}
    descriptor = {k: packet[k] for k in ('revision', 'prefix', 'input_sha256', 'input_size_px', 'input_to_native', 'native_to_input',
        'models', 'selection', 'policy', 'inspection', 'complete', 'dependency_state', 'dependent_graph_rebuilt',
        'cached_dependent_inputs_used', 'source_pixels_erased', 'production_changed', 'physical_geometry_approved')}
    descriptor.update(schema='geometry-bundle/1.0', input_file=deepcopy(view['gray']), files=refs, counts=counts,
        owner_interval_ids=[p['id'] for p in packet['graph']['proposals']],
        ownership={'ambiguous_source_owner': 0, 'unassigned_role_owner': 0,
                   'array_frame': 'input_pixel_centres', 'role_owner_is_partial': True},
        source_ownership_work=packet['source_ownership_work'],
        excluded_input_ink_pixels=int(np.count_nonzero(packet['preprocessing_excluded_ink'])),
        observations_require_review=True, new_OCR_ML_calls=0, training=False)
    descriptor['content_sha256'] = object_sha(descriptor)
    write(folder/'bundle.json', descriptor)
    return pin(root, (folder/'bundle.json').relative_to(root).as_posix())


def load(root, record, view, *, model_contract):
    root = Path(root).resolve(); path = checked(root, record)
    descriptor = json.loads(path.read_text(encoding='utf-8'))
    expected = descriptor.pop('content_sha256')
    if object_sha(descriptor) != expected: raise ValueError('Bundle descriptor content differs')
    if descriptor['schema'] != 'geometry-bundle/1.0': raise ValueError('Unsupported geometry bundle schema')
    if descriptor['models'] != model_binding(root, model_contract): raise ValueError('Bundle producer model differs')
    checked(root, view['gray'])
    if descriptor['input_file'] != view['gray']: raise ValueError('Bundle input file binding differs')
    if not np.array_equal(matrix(view['input_to_native']), descriptor['input_to_native']): raise ValueError('Bundle native frame differs')
    if not np.array_equal(np.linalg.inv(matrix(view['input_to_native'])), descriptor['native_to_input']): raise ValueError('Bundle inverse native frame differs')
    refs = descriptor['files']
    for value in refs.values(): checked(root, value)
    gray = np.array(Image.open(checked(root, view['gray'])).convert('L'))
    saved = np.array(Image.open(checked(root, refs['input'])).convert('L'))
    if sha(gray) != descriptor['input_sha256'] or not np.array_equal(saved, gray): raise ValueError('Bundle decoded pixels differ')
    if descriptor['input_size_px'] != list(gray.shape[::-1]): raise ValueError('Bundle input dimensions differ')
    packet = {**descriptor, **{k: json.loads(checked(root, refs[k]).read_text(encoding='utf-8')) for k in FILES}}
    for k in IMAGES: packet[k] = np.array(Image.open(checked(root, refs[k])))
    for field in ('source_owner', 'role_owner'):
        a = packet[field]
        if not np.issubdtype(a.dtype, np.integer) or a.min(initial=0) < 0 or a.max(initial=0) > 65535:
            raise ValueError('Invalid unsigned owner values')
        packet[field] = a.astype(np.uint16)
    expected_selection = select(gray, packet['raw_source'], packet['proposed_geometry'],
        input_to_native=view['input_to_native'], models=descriptor['models'], extraction_complete=descriptor['selection']['extraction_complete'])
    for field in ('revision', 'prefix', 'source', 'geometry', 'selection'):
        if object_sha(packet[field]) != object_sha(expected_selection[field]): raise ValueError('Bundle selected geometry differs: '+field)
    validate_geometry(gray, packet['source'], packet['geometry'])
    if descriptor['owner_interval_ids'] != [p['id'] for p in packet['graph']['proposals']]: raise ValueError('Bundle owner index differs')
    observed = inspect(gray, packet, view['input_to_native'])
    if not observed['valid'] or observed != descriptor['inspection']: raise ValueError('Bundle graph binding differs')
    if not descriptor['dependent_graph_rebuilt'] or descriptor['cached_dependent_inputs_used']: raise ValueError('Stale dependency handoff')
    if descriptor['complete'] != all(descriptor['dependency_state'].values()): raise ValueError('Incomplete dependent graph reported as complete')
    raw_ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    from .stroke_roles_v1 import foreground
    excluded = np.where((raw_ink > 0) & (foreground(gray) == 0), 255, 0).astype(np.uint8)
    if not np.array_equal(raw_ink, packet['raw_ink']) or not np.array_equal(excluded, packet['preprocessing_excluded_ink']):
        raise ValueError('Original input ink was not preserved')
    if descriptor['physical_geometry_approved'] or descriptor['production_changed']: raise ValueError('Unreviewed geometry cannot acquire approval')
    return packet


def structural_inputs(root, record, view, *, model_contract):
    packet = load(root, record, view, model_contract=model_contract)
    if not packet['complete']: raise ValueError('Incomplete geometry bundle cannot be handed to a role consumer')
    return {k: packet['files'][v] for k, v in [('graph', 'graph'), ('heads', 'heads'), ('links', 'chains'), ('roles', 'roles')]}
