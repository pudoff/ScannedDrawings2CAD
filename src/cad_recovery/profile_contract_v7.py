"""Version the common export repair; old profile bindings remain executable."""
import hashlib
from pathlib import Path

from . import profile_contract_v6 as previous
from .dataset_layout_v1 import checked, digest, encoded, read

ROOT = previous.ROOT
REGISTRY = ROOT/'configs/cad-export-guard-v1.json'
ProcessingOptions, ProfileError = previous.ProcessingOptions, previous.ProfileError
DEFINITIONS, NATIVE, REVIEW, CANDIDATE = previous.DEFINITIONS, previous.NATIVE, previous.REVIEW, previous.CANDIDATE
verify_models, verify_cv_runtime = previous.verify_models, previous.verify_cv_runtime


def resolve_profile(name):
    core = previous.resolve_profile(name)
    if name != CANDIDATE:
        return core
    registry = read(REGISTRY)
    for ref in registry['code'] + registry['qualification']:
        checked(ROOT, ref)
    if registry['guard'] != 'source-binding/1.0':
        raise ProfileError('profile_contract_changed', 'Unexpected source binding guard')
    body = {**core, 'schema_version': 'processing-profile/7.0', 'core_binding': core,
        'definition': {**core['definition'], 'export_safety': registry['guard']},
        'code': {**core['code'], **{r['path']: r['sha256'] for r in registry['code']}},
        'export_safety': {'version': registry['guard'], 'source_intervals_repaired': False,
            'withheld_ink': 'residual', 'CV_OCR_changed': False},
        'export_guard_registry_sha256': digest(REGISTRY)}
    body.pop('sha256', None)
    return {**body, 'sha256': hashlib.sha256(encoded(body)).hexdigest()}


def validate_binding(name, binding):
    if binding and binding.get('schema_version') != 'processing-profile/7.0':
        return previous.validate_binding(name, binding)
    value = resolve_profile(name)
    if value != binding:
        raise ProfileError('profile_contract_changed', 'Queued export safety binding changed')
    return value
