"""Native CAD demo profiles; old documents retain their exact old bindings."""
import hashlib
import os
from pathlib import Path
from typing import Literal
from . import profile_contract_v3 as previous
from .dataset_layout_v1 import checked,digest,encoded,read

ROOT = previous.ROOT
REGISTRY = Path(__file__).parent/'data/processing-profiles-v4.json'
NATIVE = 'native_cad_draft'
REVIEW = 'native_cad_review_demo'
NATIVE_PROFILES = (NATIVE,REVIEW)
ProfileError = previous.ProfileError
verify_models = previous.verify_models
verify_cv_runtime = previous.verify_cv_runtime
RELEASE = {'path':'artifacts/native-annotations-release-v1.json',
    'sha256':'7b77dedfd1904d1ed30e46e2441ab9dd4da677037eff234d77b25df17a8cfec7'}
DEFINITION = {**previous.DEFINITION,'revision':'4.0','export':'native_objects_TEXT_MTEXT_DIMENSION_draft',
    'annotations':'source_bound_components_and_unitless_dimensions','native_frame':'working_pixel_centres',
    'review_example':False,'quality_gate_override':'user_requested_CAD_demo_2026-11-17'}
DEFINITIONS = {**previous.DEFINITIONS,NATIVE:DEFINITION,
    REVIEW:{**DEFINITION,'review_example':True,'review_policy':'explicit_matching_source_agent_example_only'}}
NEW_CODE = ('src/cad_recovery/profile_contract_v4.py','src/cad_recovery/profile_pipeline_v4.py',
    'src/cad_recovery/native_comparison_v1.py','src/cad_recovery/service_profiles_v9/__init__.py',
    'src/cad_recovery/service_profiles_v9/api.py','src/cad_recovery/service_profiles_v9/queue.py',
    'src/cad_recovery/service_profiles_v9/worker.py','src/cad_recovery/service_profiles_v9/storage.py',
    'scripts/service_profile_pipeline_v4.py','scripts/run_profiles_v11.py')


class ProcessingOptions(previous.ProcessingOptions):
    profile:Literal['baseline','geometry_observations','presentation_draft','native_observations',
                    'experimental_draft','native_cad_draft','native_cad_review_demo'] = NATIVE


def resolve_profile(name):
    if name not in NATIVE_PROFILES: return previous.resolve_profile(name)
    parent = previous.resolve_profile(previous.EXPERIMENTAL)
    registry = read(REGISTRY); release = read(checked(ROOT,RELEASE))
    required = {r['path']:r['sha256'] for r in release['files'] if r['path'].startswith('src/')}
    prior = read(checked(ROOT,release['prior_release']))
    required.update({r['path']:r['sha256'] for r in prior['files'] if r['path'].startswith('src/')})
    if registry['definitions']!={n:DEFINITIONS[n] for n in NATIVE_PROFILES} or registry['release']!=RELEASE:
        raise ProfileError('profile_contract_changed','Native CAD registry definition differs')
    if not (set(NEW_CODE)|set(required)) <= registry['code'].keys():
        raise ProfileError('profile_contract_changed','Native CAD code binding incomplete')
    if any(registry['code'][p]!=s for p,s in required.items()):
        raise ProfileError('profile_contract_changed','Released annotation code differs')
    example = next(r for r in release['files'] if r['path']=='artifacts/native-annotations-v1/star-review.json')
    if registry['review_example']!=example: raise ProfileError('profile_contract_changed','Prepared review example differs')
    for path,sha in registry['code'].items(): checked(ROOT,{'path':path,'sha256':sha})
    checked(ROOT,registry['review_example'])
    font = Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts/cambria.ttc'
    if not font.is_file(): raise ProfileError('annotation_font_missing','Native CAD demo requires installed Cambria')
    from fontTools.ttLib import TTFont
    face = TTFont(font,fontNumber=0)
    try:
        if not {0x410,0x44f,0xd8,0x2300,0xb1,0xb0}<=face.getBestCmap().keys():
            raise ProfileError('annotation_font_missing','Annotation font lacks required engineering glyphs')
    finally: face.close()
    body = {'schema_version':'processing-profile/4.0','name':name,'definition':DEFINITIONS[name],
        'code':registry['code'],'registry_sha256':digest(REGISTRY),'annotation_release':RELEASE,
        'review_example':registry['review_example'] if name==REVIEW else None,
        'legacy_binding_sha256':parent['sha256'],'ocr':parent['ocr'],'dependencies':parent['dependencies'],
        'annotation_font':{'file':'cambria.ttc','sha256':digest(font),'glyph_coverage_verified':True},
        'physical_geometry_approved':False,'production_admitted':False}
    return {**body,'sha256':hashlib.sha256(encoded(body)).hexdigest()}


def validate_binding(name,binding):
    value = resolve_profile(name)
    if value!=binding: raise ProfileError('profile_contract_changed','Queued CAD profile changed; create a new version')
    return value
