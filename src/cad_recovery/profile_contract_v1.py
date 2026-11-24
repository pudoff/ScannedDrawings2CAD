"""Versioned local processing profiles shared by CLI, queue and worker."""
from copy import deepcopy
import hashlib
import importlib.metadata
import json
from pathlib import Path

from .service_demo_v1.config import ProcessingOptions

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = Path(__file__).parent/'data/processing-profiles-v1.json'
MODELS = ROOT/'artifacts/models/ppocr-v5'
DEFINITIONS = {
    'baseline': {'revision': '1.0', 'prepare': 'rotation_only', 'geometry': False,
        'ocr': False, 'arrows': False, 'association': None, 'input_reduction': False,
        'export': 'pdf_observations_or_empty_raster'},
    'geometry_observations': {'revision': '1.0', 'prepare': 'always', 'geometry': True,
        'ocr': False, 'arrows': False, 'association': None, 'input_reduction': False,
        'export': 'pdf_observations_or_empty_raster'},
    'presentation_draft': {'revision': '1.0', 'prepare': 'always', 'geometry': True,
        'ocr': True, 'arrows': True, 'association': 'contour-association-v3',
        'input_reduction': True, 'export': 'binary_SOLID_or_pdf_vector_review'},
}
LIMITS = {'source_bytes': 50*1024*1024, 'pdf_pages': 20, 'native_pixels': 25_000_000,
    'presentation_source_pixels': 100_000_000, 'presentation_reduced_pixels': 4_000_000,
    'render_dpi': [36,300], 'image_dpi': [36,1200], 'rotation_deg': [0,90,180,270],
    'ocr_windows': 64, 'ocr_observations': 10000, 'ocr_window_pixels': 1536*1536,
    'ocr_overlap_px': 192, 'ocr_soft_page_seconds': 240., 'scan_SOLID_rectangles': 250000,
    'physical_geometry_approved': False, 'training_eligible': False}
DEPENDENCIES = ('ezdxf','pydantic','pypdf','pypdfium2','pillow','numpy','opencv-python',
                'PyMuPDF','rapidocr','onnxruntime')
CODE_FILES = ('src/cad_recovery/profile_contract_v1.py','src/cad_recovery/profile_pipeline_v1.py',
    'src/cad_recovery/service_profiles_v1/queue.py','src/cad_recovery/service_profiles_v1/worker.py',
    'src/cad_recovery/service_profiles_v1/app.py','scripts/process_drawing_v1.py',
    'scripts/service_profile_pipeline_v1.py','scripts/run_profiles_v1.py',
    'src/cad_recovery/pipeline.py','src/cad_recovery/intake.py','src/cad_recovery/preprocessing.py',
    'src/cad_recovery/geometry_detection.py','src/cad_recovery/ocr_engine.py','src/cad_recovery/ocr.py',
    'src/cad_recovery/notation.py','src/cad_recovery/arrow_detection.py',
    'src/cad_recovery/circle_search_v2.py','src/cad_recovery/title_recognition.py',
    'src/cad_recovery/stroke_evidence.py',
    'src/cad_recovery/dimension_association_v3.py','src/cad_recovery/contour_topology_v3.py',
    'src/cad_recovery/dimension_roles.py','src/cad_recovery/demo_pipeline_v1.py',
    'src/cad_recovery/demo_input_v1.py','src/cad_recovery/demo_export_v1.py',
    'src/cad_recovery/exporter.py','src/cad_recovery/schema.py','src/cad_recovery/tolerances.py',
    'src/cad_recovery/service_demo_v1/config.py','src/cad_recovery/service_v1/config.py',
    'src/cad_recovery/service_v1/queue.py','src/cad_recovery/service_v1/storage.py',
    'src/cad_recovery/service_v1/process.py','src/cad_recovery/service_demo_v1/api.py',
    'src/cad_recovery/web_v1/reviews.py','src/cad_recovery/data/ocr-models-v1.json',
    'src/cad_recovery/__init__.py','scripts/run_demo.py')


class ProfileError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def encoded(value):
    return (json.dumps(value,ensure_ascii=True,sort_keys=True,allow_nan=False,separators=(',',':'))+'\n').encode()


def load_registry():
    registry = json.loads(REGISTRY.read_text(encoding='utf-8'))
    if registry.get('schema_version') != 'processing-profiles/1.0' or registry.get('definitions') != DEFINITIONS or registry.get('limits') != LIMITS:
        raise ProfileError('profile_contract_changed','Processing profile definitions or limits changed')
    if set(registry.get('code',{})) != set(CODE_FILES) or set(registry.get('dependencies',{})) != set(DEPENDENCIES):
        raise ProfileError('profile_contract_changed','Incomplete profile code or dependency binding')
    for name, expected in registry['code'].items():
        path = (ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or digest(path) != expected:
            raise ProfileError('profile_contract_changed',f'Processing code binding differs: {name}')
    for name, expected in registry['dependencies'].items():
        if importlib.metadata.version(name) != expected:
            raise ProfileError('profile_environment_changed',f'Processing dependency version differs: {name}')
    manifest_path=Path(__file__).parent/'data/ocr-models-v1.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    expected_ocr={'id':manifest['id'],'manifest_sha256':digest(manifest_path),'provider':'CPUExecutionProvider',
        'models':[{k:item[k] for k in ('role','filename','sha256')} for item in manifest['models']]}
    if registry.get('ocr') != expected_ocr:
        raise ProfileError('profile_contract_changed','OCR profile binding differs from the pinned manifest')
    return registry


def resolve_profile(name):
    registry = load_registry()
    if name not in DEFINITIONS:
        raise ProfileError('profile_unknown','Unknown processing profile')
    body = {'schema_version':'processing-profile/1.0','name':name,
        'definition':deepcopy(DEFINITIONS[name]),'limits':deepcopy(LIMITS),
        'registry_sha256':digest(REGISTRY),'code':registry['code'],
        'dependencies':registry['dependencies'],
        'ocr':registry['ocr'] if DEFINITIONS[name]['ocr'] else None,
        'physical_geometry_approved':False,'production_admitted':False}
    return dict(body,sha256=hashlib.sha256(encoded(body)).hexdigest())


def validate_binding(name, binding):
    current = resolve_profile(name)
    if binding != current:
        raise ProfileError('profile_contract_changed','Queued profile binding differs; create a new version explicitly')
    return current


def verify_models(binding, directory=None):
    if binding['ocr'] is None:
        return None
    root = Path(directory or MODELS).resolve()
    for item in binding['ocr']['models']:
        path = root/item['filename']
        if not path.is_file():
            raise ProfileError('profile_model_missing',f'Local OCR model is missing: {item["filename"]}')
        if path.is_symlink() or not path.resolve().is_relative_to(root) or digest(path) != item['sha256']:
            raise ProfileError('profile_model_mismatch',f'Local OCR model hash differs: {item["filename"]}')
    return root
