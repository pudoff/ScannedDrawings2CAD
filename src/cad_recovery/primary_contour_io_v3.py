"""Replayed source/model/frame binding for primary proposals and baseline inputs."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .dataset_layout_v1 import checked, pin
from .geometry_bundle_io_v2 import load as parent_load
from .grid_role_revision_v1 import cv_authority
from .part_admission_v1 import object_sha
from .primary_contour_v3 import analyze
from .competitive_grid_admission_v2 import analyze as baseline_context

REQUIRED = {'src/cad_recovery/'+n+'.py' for n in ('primary_contour_v3', 'primary_contour_io_v3',
             'competitive_grid_witness_v2', 'competitive_grid_admission_v2',
             'source_context_admission_v1', 'source_interval_context_v1')}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, data):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(data, f, sort_keys=True, allow_nan=False)


def check_model(root, model):
    contract = read(checked(root, model))
    if not REQUIRED <= {r['path'] for r in contract['code']}:
        raise ValueError('Incomplete primary contour producer')
    for ref in contract['code']:
        checked(root, ref)


def inference(root, parent, view, parent_model):
    packet = parent_load(root, parent, view, model_contract=parent_model)
    gray = np.array(Image.open(checked(root, view['gray'])).convert('L'))
    authority = cv_authority(gray, packet['roles'], packet['files']['roles'])
    baseline = baseline_context(gray, packet['graph'], packet['roles'], packet['heads'],
                                packet['source_owner'], packet['ambiguous_owner_mask'], authority=authority)
    primary = analyze(gray, packet['graph'], packet['roles'], packet['source_owner'],
                      packet['ambiguous_owner_mask'], packet['heads'], authority=authority, source_paths=packet['source'])
    return packet, baseline, primary


def arrays(baseline, primary):
    return {**{k: baseline[k] for k in ('previous', 'old_candidate', 'old_owners')},
            'agreement': baseline['candidates']['agreement'], 'agreement_owners': baseline['owners']['agreement'],
            'primary_mask': primary['mask']}


def metadata(baseline, primary):
    return {'baseline': {k: v for k, v in baseline.items() if k not in ('previous', 'old_candidate', 'old_owners', 'candidates', 'owners')},
            'primary': {k: v for k, v in primary.items() if k != 'mask'}}


def capture(root, parent, view, folder, *, parent_model, model):
    root, folder = Path(root).resolve(), Path(folder).resolve(); folder.relative_to(root)
    if folder.exists():
        raise FileExistsError(folder)
    check_model(root, model); packet, baseline, primary = inference(root, parent, view, parent_model)
    folder.mkdir(parents=True); files = {}
    for key, value in arrays(baseline, primary).items():
        path = folder/(key+'.png')
        Image.fromarray(value.astype(np.uint16) if 'owners' in key else value).save(path)
        files[key] = pin(root, path.relative_to(root).as_posix())
    write(folder/'evidence.json', metadata(baseline, primary))
    files['metadata'] = pin(root, (folder/'evidence.json').relative_to(root).as_posix())
    descriptor = {'schema': 'primary-contour/3.0', 'parent': parent, 'parent_model': parent_model, 'model': model,
                  'view': {k: view[k] for k in ('gray', 'input_to_native')}, 'files': files,
                  'graph_sha256': object_sha(packet['graph']), 'chains_sha256': object_sha(packet['chains']),
                  'metadata_sha256': object_sha(metadata(baseline, primary))}
    descriptor['content_sha256'] = object_sha(descriptor); write(folder/'primary.json', descriptor)
    return pin(root, (folder/'primary.json').relative_to(root).as_posix()), packet, baseline, primary


def load(root, record, parent, view, *, parent_model, model):
    descriptor = read(checked(root, record)); checksum = descriptor.pop('content_sha256')
    if checksum != object_sha(descriptor) or descriptor['schema'] != 'primary-contour/3.0':
        raise ValueError('Invalid primary manifest')
    if descriptor['parent'] != parent or descriptor['parent_model'] != parent_model or descriptor['model'] != model:
        raise ValueError('Rebound source or producer')
    if descriptor['view'] != {k: view[k] for k in ('gray', 'input_to_native')}:
        raise ValueError('Rebound source frame')
    check_model(root, model); packet, baseline, primary = inference(root, parent, view, parent_model)
    if descriptor['graph_sha256'] != object_sha(packet['graph']) or descriptor['chains_sha256'] != object_sha(packet['chains']):
        raise ValueError('Stale dependencies')
    for key, value in arrays(baseline, primary).items():
        if not np.array_equal(value, np.array(Image.open(checked(root, descriptor['files'][key])))):
            raise ValueError('Primary replay differs: '+key)
    actual = read(checked(root, descriptor['files']['metadata']))
    if object_sha(actual) != descriptor['metadata_sha256'] or object_sha(actual) != object_sha(metadata(baseline, primary)):
        raise ValueError('False primary evidence')
    return descriptor, packet, baseline, primary
