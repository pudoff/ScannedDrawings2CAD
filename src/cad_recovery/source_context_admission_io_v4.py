"""Source/model-bound positive-role cache with independently replayable content."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .dataset_layout_v1 import checked, pin
from .geometry_bundle_io_v2 import load as load_parent
from .grid_role_revision_v1 import cv_authority
from .part_admission_v1 import object_sha
from .source_context_admission_v4 import analyze

FIELDS = ('previous', 'candidate', 'owners', 'old_candidate', 'old_owners')
CODE = {'src/cad_recovery/'+s+'.py' for s in ('source_interval_context_v1', 'positive_grid_witness_v3',
        'source_context_admission_v4', 'source_context_admission_io_v4')}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, data):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(data, f, sort_keys=True, allow_nan=False)


def check_model(root, model):
    contract = read(checked(root, model))
    if not CODE <= {p['path'] for p in contract['code']}:
        raise ValueError('Incomplete source-context producer contract')
    for ref in contract['code']:
        checked(root, ref)


def inference(root, parent, view, parent_model):
    packet = load_parent(root, parent, view, model_contract=parent_model)
    gray = np.array(Image.open(checked(root, view['gray'])).convert('L'))
    authority = cv_authority(gray, packet['roles'], packet['files']['roles'])
    result = analyze(gray, packet['graph'], packet['roles'], packet['heads'],
                     packet['source_owner'], packet['ambiguous_owner_mask'], authority=authority)
    return packet, result


def capture(root, parent, view, folder, *, parent_model, model):
    root, folder = Path(root).resolve(), Path(folder).resolve()
    folder.relative_to(root)
    if folder.exists():
        raise FileExistsError(folder)
    check_model(root, model)
    packet, result = inference(root, parent, view, parent_model)
    folder.mkdir(parents=True)
    files = {}
    for field in FIELDS:
        path = folder/(field+'.png')
        a = result[field].astype(np.uint16) if 'owners' in field else result[field]
        Image.fromarray(a).save(path); files[field] = pin(root, path.relative_to(root).as_posix())
    metadata = {k: v for k, v in result.items() if k not in FIELDS}
    write(folder/'context.json', metadata); files['context'] = pin(root, (folder/'context.json').relative_to(root).as_posix())
    descriptor = {'schema': 'source-context-admission/4.0', 'parent': parent,
                  'parent_model': parent_model, 'model': model,
                  'view': {k: view[k] for k in ('gray', 'input_to_native')}, 'files': files,
                  'graph_sha256': object_sha(packet['graph']), 'chains_sha256': object_sha(packet['chains']),
                  'metadata_sha256': object_sha(metadata)}
    descriptor['content_sha256'] = object_sha(descriptor)
    write(folder/'context-admission.json', descriptor)
    cache = {k: files[k] for k in ('previous', 'candidate', 'owners')}
    return pin(root, (folder/'context-admission.json').relative_to(root).as_posix()), cache, packet, result


def load(root, record, parent, view, *, parent_model, model):
    descriptor = read(checked(root, record)); content = descriptor.pop('content_sha256')
    if content != object_sha(descriptor) or descriptor['schema'] != 'source-context-admission/4.0':
        raise ValueError('Invalid positive-context manifest')
    if descriptor['parent'] != parent or descriptor['parent_model'] != parent_model or descriptor['model'] != model:
        raise ValueError('Positive context rebound to another source or model')
    if descriptor['view'] != {k: view[k] for k in ('gray', 'input_to_native')}:
        raise ValueError('Positive context frame differs')
    check_model(root, model)
    packet, expected = inference(root, parent, view, parent_model)
    if descriptor['graph_sha256'] != object_sha(packet['graph']) or descriptor['chains_sha256'] != object_sha(packet['chains']):
        raise ValueError('Positive context references stale dependencies')
    for field in FIELDS:
        actual = np.array(Image.open(checked(root, descriptor['files'][field])))
        if not np.array_equal(actual, expected[field]):
            raise ValueError('Positive-context replay differs: '+field)
    metadata = read(checked(root, descriptor['files']['context']))
    if descriptor['metadata_sha256'] != object_sha(metadata) or object_sha(metadata) != object_sha({k: v for k, v in expected.items() if k not in FIELDS}):
        raise ValueError('Positive-context metadata differs')
    return descriptor, packet, expected
