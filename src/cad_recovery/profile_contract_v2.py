"""Add a native observation profile without changing any v1 binding."""
import hashlib
from pathlib import Path
from typing import Literal
from . import profile_contract_v1 as historical
from .dataset_layout_v1 import digest,encoded,read,checked
from .family_catalog_v2 import require

ROOT=Path(__file__).resolve().parents[2]
REGISTRY=Path(__file__).parent/'data/processing-profiles-v2.json'
ProfileError=historical.ProfileError
NATIVE='native_observations'
DEFINITION={'revision':'2.0','prepare':'streaming native with EXIF/quarter-turn chain','geometry':True,'ocr':True,
    'arrows':True,'association':'contour-association-v3','input_reduction':False,'export':'observations_only_no_DXF',
    'formats':['PNG','JPEG','PDF'],'png_large':'RGB8 noninterlaced','jpeg_or_other_png_max_pixels':25_000_000,
    'pdf_max_pages':5,'pdf_page_max_render_pixels':25_000_000,'max_native_pixels':300_000_000,
    'max_retry_windows':16,'max_windows_per_page':256}
DEFINITIONS={**historical.DEFINITIONS,NATIVE:DEFINITION}
CODE=['src/cad_recovery/profile_contract_v2.py','src/cad_recovery/native_input_v2.py',
 'src/cad_recovery/native_profile_kernel_v2.py','src/cad_recovery/profile_pipeline_v2.py',
 'src/cad_recovery/service_profiles_v2/storage.py','src/cad_recovery/service_profiles_v2/worker.py',
 'src/cad_recovery/service_profiles_v2/api.py','src/cad_recovery/service_profiles_v2/queue.py',
 'src/cad_recovery/service_profiles_v2/budget.py','scripts/process_drawing_v2.py',
 'src/cad_recovery/service_profiles_v2/config.py',
 'scripts/service_profile_pipeline_v2.py','scripts/run_profiles_v2.py',
 'src/cad_recovery/service_profiles_v2/assets/index.html','src/cad_recovery/service_profiles_v2/assets/app.js',
 'src/cad_recovery/native_tiled_v1.py','src/cad_recovery/native_merge_v1.py','src/cad_recovery/raster_regions.py',
 'src/cad_recovery/large_raster.py','src/cad_recovery/png_stream.py']

class ProcessingOptions(historical.ProcessingOptions):
    profile:Literal['baseline','geometry_observations','presentation_draft','native_observations']='presentation_draft'

def resolve_profile(name):
    if name!=NATIVE:return historical.resolve_profile(name)
    parent=historical.resolve_profile('presentation_draft')
    registry=read(REGISTRY)
    if registry.get('definition')!=DEFINITION or set(registry.get('code',{}))!=set(CODE):
        raise ProfileError('profile_contract_changed','Native registry definition differs')
    for path,sha in registry['code'].items():checked(ROOT,{'path':path,'sha256':sha})
    body={'schema_version':'processing-profile/2.0','name':NATIVE,'definition':DEFINITION,
        'code':registry['code'],'registry_sha256':digest(REGISTRY),'ocr':parent['ocr'],
        'dependencies':parent['dependencies'],'legacy_binding_sha256':parent['sha256'],
        'physical_geometry_approved':False,'production_admitted':False}
    return {**body,'sha256':hashlib.sha256(encoded(body)).hexdigest()}

def validate_binding(name,binding):
    current=resolve_profile(name)
    if current!=binding:raise ProfileError('profile_contract_changed','Queued profile binding changed; create an explicit new version')
    return current

verify_models=historical.verify_models
