"""Explicit opt-in experimental demo binding; legacy contracts remain unchanged."""
import hashlib
from pathlib import Path
from typing import Literal
from . import profile_contract_v2 as previous
from .dataset_layout_v1 import checked, digest, encoded, read

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = Path(__file__).parent/'data/processing-profiles-v3.json'
EXPERIMENTAL = 'experimental_draft'
EXPERIMENT = {'path':'artifacts/native-frame-profile-v3/plan.json',
    'sha256':'2e829327f4f0f55ac55d234fc14a5214c0fccaec4ba95e9484b625c72be98523'}
ADAPTER_CODE = (
    'src/cad_recovery/experimental_cv_v1.py','src/cad_recovery/experimental_export_v1.py',
    'src/cad_recovery/profile_contract_v3.py','src/cad_recovery/profile_pipeline_v3.py',
    'src/cad_recovery/service_profiles_v8/api.py','src/cad_recovery/service_profiles_v8/queue.py',
    'src/cad_recovery/service_profiles_v8/worker.py','scripts/service_profile_pipeline_v3.py',
    'scripts/run_profiles_v10.py','artifacts/cv-skeleton-runtime-v1/installed-files.json')
ProfileError = previous.ProfileError
verify_models = previous.verify_models
DEFINITION = {**previous.DEFINITIONS['presentation_draft'], 'revision': '3.0',
    'geometry': True, 'export': 'experimental_native_geometry_layered_residual_DXF',
    'cv': 'joint_shape+joint_heads+no_part+native_mid', 'cv_max_pixels': 4_000_000,
    'cv_max_side': 2000, 'automatic_roles_verified': False, 'production_admitted': False,
    'quality_gate_override': 'user_requested_demo_only_2026-11-15'}
DEFINITIONS = {**previous.DEFINITIONS, EXPERIMENTAL: DEFINITION}


class ProcessingOptions(previous.ProcessingOptions):
    profile: Literal['baseline','geometry_observations','presentation_draft','native_observations','experimental_draft'] = EXPERIMENTAL


def resolve_profile(name):
    if name != EXPERIMENTAL:
        return previous.resolve_profile(name)
    parent = previous.resolve_profile('presentation_draft')
    registry = read(REGISTRY)
    plan = read(checked(ROOT, EXPERIMENT))
    required = set(ADAPTER_CODE) | {p['path'] for p in plan['code']}
    if (registry.get('definition') != DEFINITION or registry.get('experiment') != EXPERIMENT
            or not required <= set(registry.get('code',{}))):
        raise ProfileError('profile_contract_changed', 'Experimental demo registry changed')
    if any(registry['code'][p['path']] != p['sha256'] for p in plan['code']):
        raise ProfileError('profile_contract_changed', 'Frozen experiment code changed')
    for path, sha in registry['code'].items():
        checked(ROOT, {'path':path,'sha256':sha})
    body = {'schema_version':'processing-profile/3.0', 'name':EXPERIMENTAL,
        'definition':DEFINITION, 'code':registry['code'], 'registry_sha256':digest(REGISTRY),
        'ocr':parent['ocr'], 'dependencies':parent['dependencies'],
        'legacy_binding_sha256':parent['sha256'], 'physical_geometry_approved':False,
        'production_admitted':False, 'experiment':registry['experiment']}
    checked(ROOT, registry['experiment'])
    return {**body, 'sha256':hashlib.sha256(encoded(body)).hexdigest()}


def validate_binding(name, binding):
    value = resolve_profile(name)
    if value != binding:
        raise ProfileError('profile_contract_changed', 'Queued profile binding changed; create a new version')
    return value


def verify_cv_runtime():
    """Check the small, consumed optional runtime, without hashing the historical archive."""
    manifest = read(ROOT/'artifacts/cv-skeleton-runtime-v1/installed-files.json')
    for path, sha in manifest['installed_static_files'].items():
        checked(ROOT, {'path':path,'sha256':sha})
    return len(manifest['installed_static_files'])
