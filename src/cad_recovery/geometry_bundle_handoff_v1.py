"""Pass only checked current dependencies to consumers; never inherit old caches."""
from copy import deepcopy
from .geometry_bundle_io_v1 import load


def consumer_view(root, record, expected_view, *, model_contract):
    packet = load(root, record, expected_view, model_contract=model_contract)
    if not packet['complete']: raise ValueError('Incomplete bundle requires review, not automatic handoff')
    files = packet['files']
    # Construct a new view rather than overlaying the old one. In particular,
    # admission_cache/comparison/cache/bank may refer to a rejected geometry.
    view = {'gray': deepcopy(expected_view['gray']), 'input_to_native': packet['input_to_native'],
        'geometry_bundle': deepcopy(record), 'geometry_revision': packet['revision'],
        'structural_inputs': {key: files[field] for key, field in
            [('graph', 'graph'), ('heads', 'heads'), ('links', 'chains'), ('roles', 'roles')]},
        'geometry_snapshot': {'source': files['source'], 'geometry': files['geometry'], 'arm': 'direct',
            'input_sha256': packet['input_sha256'], 'model_contract': deepcopy(model_contract),
            'input_to_native': packet['input_to_native'], 'selected_mode': packet['selection']['mode']},
        'admission_must_be_rebuilt': True, 'physical_geometry_approved': False}
    if 'mode' in expected_view: view['mode'] = expected_view['mode']
    return view
