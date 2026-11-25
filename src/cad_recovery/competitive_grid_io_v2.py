"""Checked two-arm caches with replay over an immutable geometric parent."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .dataset_layout_v1 import checked, pin
from .geometry_bundle_io_v2 import load as parent_load
from .grid_role_revision_v1 import cv_authority
from .part_admission_v1 import object_sha
from .competitive_grid_admission_v2 import analyze, ARMS

REQUIRED = {'src/cad_recovery/'+n+'.py' for n in ('competitive_grid_witness_v2',
             'competitive_grid_admission_v2', 'competitive_grid_io_v2', 'positive_grid_witness_v3',
             'positive_grid_witness_v1', 'source_context_admission_v1', 'source_interval_context_v1')}


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, data):
    with Path(path).open('x', encoding='utf-8') as f: json.dump(data, f, sort_keys=True, allow_nan=False)


def check_model(root, model):
    contract = read(checked(root, model))
    if not REQUIRED <= {r['path'] for r in contract['code']}: raise ValueError('Incomplete competing-role producer')
    for ref in contract['code']: checked(root, ref)


def inference(root, parent, view, parent_model):
    packet = parent_load(root, parent, view, model_contract=parent_model)
    gray = np.array(Image.open(checked(root, view['gray'])).convert('L'))
    authority = cv_authority(gray, packet['roles'], packet['files']['roles'])
    result = analyze(gray, packet['graph'], packet['roles'], packet['heads'], packet['source_owner'],
                     packet['ambiguous_owner_mask'], authority=authority)
    return packet, result


def arrays(result):
    return {**{k: result[k] for k in ('previous', 'old_candidate', 'old_owners')},
            **{arm: result['candidates'][arm] for arm in ARMS},
            **{arm+'_owners': result['owners'][arm] for arm in ARMS}}


def metadata(result): return {k: v for k, v in result.items() if k not in ('previous', 'old_candidate', 'old_owners', 'candidates', 'owners')}


def capture(root, parent, view, folder, *, parent_model, model):
    root, folder = Path(root).resolve(), Path(folder).resolve(); folder.relative_to(root)
    if folder.exists(): raise FileExistsError(folder)
    check_model(root, model); packet, result = inference(root, parent, view, parent_model)
    folder.mkdir(parents=True); files = {}
    for key, a in arrays(result).items():
        path = folder/(key+'.png'); Image.fromarray(a.astype(np.uint16) if 'owners' in key else a).save(path)
        files[key] = pin(root, path.relative_to(root).as_posix())
    write(folder/'evidence.json', metadata(result)); files['metadata'] = pin(root, (folder/'evidence.json').relative_to(root).as_posix())
    d = {'schema': 'competitive-grid/2.0', 'parent': parent, 'parent_model': parent_model, 'model': model,
         'view': {k: view[k] for k in ('gray', 'input_to_native')}, 'files': files,
         'graph_sha256': object_sha(packet['graph']), 'chains_sha256': object_sha(packet['chains']),
         'metadata_sha256': object_sha(metadata(result))}
    d['content_sha256'] = object_sha(d); write(folder/'competition.json', d)
    caches = {arm: {'previous': files['previous'], 'candidate': files[arm], 'owners': files[arm+'_owners']} for arm in ARMS}
    return pin(root, (folder/'competition.json').relative_to(root).as_posix()), caches, packet, result


def load(root, record, parent, view, *, parent_model, model):
    d = read(checked(root, record)); content = d.pop('content_sha256')
    if content != object_sha(d) or d['schema'] != 'competitive-grid/2.0': raise ValueError('Invalid competition manifest')
    if d['parent'] != parent or d['parent_model'] != parent_model or d['model'] != model: raise ValueError('Rebound source or producer')
    if d['view'] != {k: view[k] for k in ('gray', 'input_to_native')}: raise ValueError('Rebound source frame')
    check_model(root, model); packet, result = inference(root, parent, view, parent_model)
    if d['graph_sha256'] != object_sha(packet['graph']) or d['chains_sha256'] != object_sha(packet['chains']): raise ValueError('Stale dependencies')
    for key, a in arrays(result).items():
        if not np.array_equal(a, np.array(Image.open(checked(root, d['files'][key])))): raise ValueError('Competition replay differs: '+key)
    actual = read(checked(root, d['files']['metadata']))
    if object_sha(actual) != d['metadata_sha256'] or object_sha(actual) != object_sha(metadata(result)): raise ValueError('False competition evidence')
    return d, packet, result
