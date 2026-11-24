"""An opt-in CAD/layer demo profile; legacy profile bindings remain exact."""
import hashlib
from pathlib import Path
from typing import Literal
from . import profile_contract_v4 as previous
from .dataset_layout_v1 import checked, digest, encoded, read

ROOT = previous.ROOT
REGISTRY = Path(__file__).parent/'data/processing-profiles-v5.json'
NATIVE, REVIEW = previous.NATIVE, previous.REVIEW
CANDIDATE = 'native_cad_layers_draft'
ProfileError = previous.ProfileError
verify_models, verify_cv_runtime = previous.verify_models, previous.verify_cv_runtime
DEFINITION = {**previous.DEFINITION, 'revision':'5.0',
    'export':'whole_primitives_OCR_source_confirmed_FRAME', 'review_example':False,
    'quality_gate_override':'limited_demo_9.1.1_9.1.2_2026-11-20'}
DEFINITIONS = {**previous.DEFINITIONS, CANDIDATE:DEFINITION}
NEW_CODE = ('src/cad_recovery/profile_contract_v5.py','src/cad_recovery/profile_pipeline_v5.py',
    'src/cad_recovery/service_profiles_v10/__init__.py','src/cad_recovery/service_profiles_v10/api.py',
    'src/cad_recovery/service_profiles_v10/worker.py','src/cad_recovery/service_profiles_v10/queue.py',
    'scripts/service_profile_pipeline_v5.py','scripts/run_profiles_v12.py')


class ProcessingOptions(previous.ProcessingOptions):
    profile:Literal['baseline','geometry_observations','presentation_draft','native_observations',
        'experimental_draft','native_cad_draft','native_cad_review_demo','native_cad_layers_draft'] = NATIVE


def resolve_profile(name):
    if name != CANDIDATE:
        return previous.resolve_profile(name)
    base = previous.resolve_profile(NATIVE)
    registry = read(REGISTRY)
    if registry['definition'] != DEFINITION:
        raise ProfileError('profile_contract_changed','Candidate definition differs')
    if not set(NEW_CODE) <= registry['code'].keys():
        raise ProfileError('profile_contract_changed','Candidate adapter binding incomplete')
    for path, checksum in registry['code'].items():
        checked(ROOT, dict(path=path,sha256=checksum))
    for ref in registry['qualification']:
        checked(ROOT, ref)
    qualification = {Path(ref['path']).name:read(checked(ROOT,ref)) for ref in registry['qualification']}
    if not qualification['verification.json']['limited_demo_layer_passed']:
        raise ProfileError('profile_contract_changed','Layer candidate was not qualified')
    required = {r['path']:r['sha256'] for r in qualification['protocol.json']['code']}
    if any(registry['code'].get(p) != s for p,s in required.items()):
        raise ProfileError('profile_contract_changed','Qualified candidate code differs')
    body = {**base, 'schema_version':'processing-profile/5.0', 'name':name,
        'definition':DEFINITION, 'baseline_binding_sha256':base['sha256'],
        'code':{**base['code'], **registry['code']}, 'registry_sha256':digest(REGISTRY),
        'qualification':registry['qualification'], 'review_example':None,
        'production_admitted':False, 'manual_CAD_verified':False}
    body.pop('sha256',None)
    return {**body, 'sha256':hashlib.sha256(encoded(body)).hexdigest()}


def validate_binding(name, binding):
    value = resolve_profile(name)
    if value != binding:
        raise ProfileError('profile_contract_changed','Queued CAD candidate changed; create a new version')
    return value
