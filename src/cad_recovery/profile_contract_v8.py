"""Bind qualified compatible intake separately from unchanged CV/OCR/export."""
import hashlib

from . import profile_contract_v7 as previous
from .dataset_layout_v1 import checked, digest, encoded, read

ROOT = previous.ROOT
REGISTRY = ROOT/'configs/raster-input-compat-web-v1.json'
ProcessingOptions, ProfileError = previous.ProcessingOptions, previous.ProfileError
DEFINITIONS, NATIVE, REVIEW, CANDIDATE = previous.DEFINITIONS, previous.NATIVE, previous.REVIEW, previous.CANDIDATE
verify_models, verify_cv_runtime = previous.verify_models, previous.verify_cv_runtime


def resolve_profile(name):
    core = previous.resolve_profile(name)
    registry = read(REGISTRY)
    if registry['adapter'] != 'bounded-raster-compat/1.0':
        raise ProfileError('profile_contract_changed', 'Unexpected compatible input adapter')
    for pin in registry['code'] + registry['qualification']:
        checked(ROOT, pin)
    body = {**core, 'schema_version': 'processing-profile/8.0', 'core_binding': core,
        'definition': {**core['definition'], 'input_adapter': registry['adapter']},
        'code': {**core['code'], **{r['path']:r['sha256'] for r in registry['code']}},
        'input_compatibility': dict(version=registry['adapter'], original_bytes_retained=True,
            explicit_derivative=True, CV_OCR_export_changed=False),
        'compatible_input_registry_sha256': digest(REGISTRY)}
    body.pop('sha256', None)
    return {**body, 'sha256': hashlib.sha256(encoded(body)).hexdigest()}


def validate_binding(name, binding):
    if binding and binding.get('schema_version') != 'processing-profile/8.0':
        return previous.validate_binding(name, binding)
    value = resolve_profile(name)
    if value != binding:
        raise ProfileError('profile_contract_changed', 'Queued compatible-input binding changed')
    return value
