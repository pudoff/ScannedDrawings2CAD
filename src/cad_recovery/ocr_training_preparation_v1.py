"""Train-only OCR export and evidence for P16.2a; never starts training."""
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess

from .dataset_layout_v1 import checked, encoded, inside, pin, read
from .family_catalog_v2 import require
from .ocr_current_release_v1 import verify_current_ocr
from .ocr_release_v3 import _package

RELEASE = {'path': 'artifacts/ocr-current-release-v1/manifest.json',
           'sha256': 'faae846b483d94e5a6e4ab1c8cc77270f72598334ce64fae35e695e556f7928a'}
PARENT = {'path': 'artifacts/ocr-baseline-final-v4.json',
          'sha256': '010fc7e6e84f141c1aaabdec94381601f9dc09ebae426c1ddeb7ce7832be611f'}
CODE = ['src/cad_recovery/ocr_training_preparation_v1.py', 'scripts/prepare_ocr_training_v1.py']
MODEL_MANIFEST = 'src/cad_recovery/data/ocr-models-v1.json'
METHOD = {
    'status': 'candidate_not_a_runnable_experiment', 'component': 'recognition_only',
    'candidate': 'eslav_PP-OCRv5_mobile_rec', 'upstream_reference_tag': 'v3.3.2',
    'seed_candidate': 20261108, 'optimizer_and_schedule': None,
    'epochs': None, 'batch_size': None, 'learning_rate': None,
    'selection': 'validation only after task coverage and rec-only zero-shot baseline',
    'test_policy': 'no training, calibration, label editing or hyperparameter selection',
    'text_policy': 'literal UTF-8; no dictionary adaptation from validation or test',
    'export_policy': 'all accepted train rows retained including within-train duplicates',
    'native_targets': 'not exported; no DXF/PDF targets approved by this task',
}


def prepare_config(workspace, output):
    root = Path(workspace).resolve()
    checked(root, PARENT); checked(root, RELEASE)
    model_manifest = read(inside(root, MODEL_MANIFEST))
    config = {'schema_version': 'ocr-training-preparation-config/1.0', 'task': 'P16.2a.1',
              'release': RELEASE, 'parent': PARENT, 'model_manifest': pin(root, MODEL_MANIFEST),
              'models': [pin(root, 'artifacts/models/ppocr-v5/'+m['filename']) for m in model_manifest['models']],
              'code': [pin(root, name) for name in CODE], 'method': METHOD}
    target = inside(root, output)
    require(not target.exists(), 'Choose a new preparation config')
    target.write_bytes(encoded(config))
    return pin(root, output)


def environment_snapshot():
    packages = {}
    for name in ('paddlepaddle', 'paddlepaddle-gpu', 'paddleocr', 'torch', 'onnx',
                 'onnxruntime-training', 'onnxruntime', 'rapidocr'):
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = None
    gpu = {'available_command': bool(shutil.which('nvidia-smi')), 'returncode': None, 'stdout': '', 'stderr': ''}
    if gpu['available_command']:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
                                capture_output=True, text=True, timeout=15, check=False)
        gpu.update(returncode=result.returncode, stdout=result.stdout.strip(), stderr=result.stderr.strip())
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'packages': packages, 'gpu_query': gpu, 'gpu_training_verified': False,
            'scope': 'current inference interpreter; availability is not a training smoke test'}


def model_alphabet(root, config):
    """Read ONNX metadata without recognition or any customer pixels."""
    import onnxruntime as ort
    paths = [checked(root, p) for p in config['models']]
    rec = next(p for p in paths if p.name == 'eslav_PP-OCRv5_rec_mobile.onnx')
    session = ort.InferenceSession(str(rec), providers=['CPUExecutionProvider'])
    characters = session.get_modelmeta().custom_metadata_map['character'].splitlines()
    require(characters and all(len(c) == 1 for c in characters)
            and len(set(characters)) == len(characters), 'Unexpected ONNX character metadata')
    # RapidOCR's CTC decoder adds blank at zero and a trailing space.
    classes = session.get_outputs()[0].shape[-1]
    require(classes == len(characters)+2, 'ONNX alphabet/output class mismatch')
    for p in config['models']: checked(root, p)
    return {'source': 'pinned ONNX character metadata, not a verified Paddle checkpoint dictionary',
            'characters': characters, 'implicit_space': True, 'ctc_blank_index': 0,
            'output_classes': classes, 'training_dictionary_compatibility_verified': False}


def train_documents(rows, base, alphabet):
    """Pure routing/export policy; caller authenticates the release first."""
    require(rows and len({r['id'] for r in rows}) == len(rows), 'Missing or duplicated OCR row')
    owners = {key: defaultdict(set) for key in ('image_sha256', 'source_sha256', 'group_id')}
    by_hash = defaultdict(list)
    for row in rows:
        require(row['split'] in ('train', 'validation', 'test') and row['task'] == 'ocr_recognition', 'Invalid OCR routing')
        require(isinstance(row['text'], str) and row['text'] and not any(c in row['text'] for c in '\t\r\n\x00'),
                'Unsafe Paddle label delimiter or empty target')
        for key in owners: owners[key][row[key]].add(row['split'])
        by_hash[row['image_sha256']].append(row)
    require(all(len(s) == 1 for values in owners.values() for s in values.values()), 'OCR train/control overlap')
    require(all(len({r['text'] for r in group}) == 1 for group in by_hash.values()), 'Identical crop has conflicting targets')
    train = sorted((r for r in rows if r['split'] == 'train'), key=lambda r: r['id'])
    require(train, 'Empty train export')
    files, records, lines = {}, [], []
    for i, row in enumerate(train):
        # Generated ASCII filenames avoid path delimiters and customer filename hints.
        name = f'images/train-{i:04d}.png'
        data = inside(base, row['image']).read_bytes()
        require(hashlib.sha256(data).hexdigest() == row['image_sha256'], 'Changed train crop')
        files[name] = data
        records.append({**row, 'image': name})
        lines.append(name+'\t'+row['text']+'\n')
    files['train.txt'] = ''.join(lines).encode('utf-8')
    files['train.jsonl'] = b''.join(json.dumps(r, ensure_ascii=False, sort_keys=True).encode('utf-8')+b'\n' for r in records)
    allowed = set(alphabet['characters'])|{' '}
    chars = Counter(''.join(r['text'] for r in train))
    report = {'train': {'rows': len(train), 'unique_images': len({r['image_sha256'] for r in train}),
                        'sources': len({r['source_sha256'] for r in train}), 'groups': len({r['group_id'] for r in train}),
                        'field_counts': dict(Counter(r['field'] for r in train)),
                        'maximum_text_length': max(len(r['text']) for r in train),
                        'rows_above_upstream_length_25': [r['id'] for r in train if len(r['text']) > 25],
                        'duplicates_retained': len(train)-len({r['image_sha256'] for r in train}),
                        'historical_control_rows_now_train': sum(r.get('historical_split') in ('validation', 'test') for r in train)},
              'train_alphabet': {'unique_characters': len(chars), 'unsupported_characters':
                                [{'character': c, 'codepoint': f'U+{ord(c):04X}', 'occurrences': chars[c]}
                                 for c in sorted(set(chars)-allowed)]},
              'control_counts_not_exported': dict(Counter(r['split'] for r in rows if r['split'] != 'train')),
              'validation_coverage': {}}
    val = [r for r in rows if r['split'] == 'validation']
    report['validation_coverage'] = {
        'rows': len(val), 'sources': len({r['source_sha256'] for r in val}),
        'real_names': sum(r['field'] == 'part_name' and not r['name_is_identifier_placeholder'] for r in val),
        'dimension_lines': sum(r['field'] not in ('part_name', 'drawing_id') for r in val),
        'identifier_placeholders': sum(r['name_is_identifier_placeholder'] for r in val)}
    return files, report


def readiness(report, environment):
    """A closed gate until the remaining preparation tasks have actual evidence."""
    reasons = ['native_training_checkpoint_and_dictionary_not_verified',
               'dedicated_training_environment_not_smoke_tested',
               'recognition_only_zero_shot_baseline_not_measured',
               'training_hyperparameters_and_export_equivalence_not_frozen']
    if not any(environment['packages'].get(p) for p in ('paddlepaddle', 'paddlepaddle-gpu')):
        reasons.append('paddle_not_installed_in_current_interpreter')
    if environment['platform'].startswith('Windows') and tuple(map(int, environment['python'].split('.')[:2])) > (3, 13):
        reasons.append('current_python_outside_documented_windows_paddle_range')
    if not report['validation_coverage']['real_names'] or not report['validation_coverage']['dimension_lines']:
        reasons.append('validation_has_no_real_names_or_dimension_lines')
    if report['train_alphabet']['unsupported_characters']:
        reasons.append('literal_train_characters_outside_inference_alphabet')
    if report['train']['rows_above_upstream_length_25']:
        reasons.append('upstream_max_text_length_25_would_filter_accepted_train_lines')
    return {'training_ready': False, 'reasons': reasons, 'training_started': False,
            'checkpoint_downloaded': False, 'packages_installed': False,
            'validation_test_exported_for_training': False, 'independent_evaluation': False}


def _documents(workspace, config_pin, environment):
    root = Path(workspace).resolve(); config = read(checked(root, config_pin))
    require(config['schema_version'] == 'ocr-training-preparation-config/1.0' and config['task'] == 'P16.2a.1'
            and config['release'] == RELEASE and config['parent'] == PARENT and config['method'] == METHOD,
            'Changed preparation protocol')
    require([p['path'] for p in config['code']] == CODE, 'Incomplete preparation code pins')
    for p in config['code']: checked(root, p)
    parent = read(checked(root, PARENT))
    require(parent['training_started'] is False, 'Unexpected trained parent')
    manifest = read(checked(root, config['model_manifest']))
    expected = [{'path': 'artifacts/models/ppocr-v5/'+m['filename'], 'sha256': m['sha256']} for m in manifest['models']]
    require(config['model_manifest']['path'] == MODEL_MANIFEST and config['models'] == expected, 'Changed model identities')
    source_report, rows, base = verify_current_ocr(root, RELEASE)
    alphabet = model_alphabet(root, config)
    files, report = train_documents(rows, base, alphabet)
    report.update(schema_version='ocr-training-preparation/1.0', task='P16.2a.1', config=config_pin,
                  source_release=RELEASE, gate=readiness(report, environment),
                  pending_control_sources=source_report['pending_control_source_count'],
                  unresolved_field_slots=source_report['field_decision_counts']['unresolved'],
                  pdf_training_lines=0, pixel_transformations='none; accepted PNG bytes copied verbatim')
    files.update({'report.json': encoded(report), 'environment.json': encoded(environment),
                  'inference-alphabet.json': encoded(alphabet), 'method-candidate.json': encoded(METHOD)})
    return files, report


def build_preparation(workspace, config_pin, output):
    root = Path(workspace).resolve(); target = inside(root, output)
    require(target.parent == root/'artifacts' and not target.exists(), 'Choose a new direct artifacts directory')
    files, report = _documents(root, config_pin, environment_snapshot())
    target.mkdir()
    for name, data in files.items():
        path = inside(target, name); path.parent.mkdir(exist_ok=True, parents=True); path.write_bytes(data)
    (target/'manifest.json').write_bytes(encoded({'schema_version': 'ocr-training-preparation-manifest/1.0',
        'config': config_pin, 'files': [pin(target, name) for name in sorted(files)]}))
    return report


def verify_preparation(workspace, manifest_pin):
    root = Path(workspace).resolve(); path, manifest = _package(root, manifest_pin)
    require(manifest['schema_version'] == 'ocr-training-preparation-manifest/1.0', 'Unknown preparation manifest')
    files, report = _documents(root, manifest['config'], read(path.parent/'environment.json'))
    require(manifest['files'] == [pin(path.parent, name) for name in sorted(files)]
            and all(inside(path.parent, name).read_bytes() == value for name, value in files.items()), 'Preparation differs from replay')
    return report


def require_training_ready(report):
    require(report['gate']['training_ready'] is True, 'Training prerequisites incomplete: '+', '.join(report['gate']['reasons']))
