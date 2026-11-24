"""Pin and verify the isolated CPU Paddle smoke test, never corpus training."""
import json
import math
from pathlib import Path
import subprocess

from .dataset_layout_v1 import checked, encoded, inside, pin, read
from .family_catalog_v2 import require
from .ocr_release_v3 import _package

PARENT = {'path': 'artifacts/ocr-training-preparation-final-v1.json',
          'sha256': 'eead28875e6d458944432ebcbd029c408b22d9792d23aa882a4a51cb187c49bb'}
BASE = 'artifacts/ocr-training-setup-v1'
SOURCE = BASE+'/source/PaddleOCR-95dc316a8490a350c1d1fcf3bc80bf83d4e52450'
COMMIT = '95dc316a8490a350c1d1fcf3bc80bf83d4e52450'
CODE = ['src/cad_recovery/ocr_training_runtime_v1.py', 'scripts/smoke_ocr_training_native_v1.py',
        'scripts/check_ocr_training_runtime_v1.py']
FIXED_ASSETS = {
    BASE+'/python-3.12.12.tar.gz': '3c8b9b10a933909c98b9916297e2093b24a9c2abaa23df1c2622c2bfe052cb94',
    BASE+'/paddleocr-95dc316.tar.gz': 'e6022165ffe4d4fc4cc1fd7a7083f6b03e5b7b28f2a6ac71855875be804ce2c0',
    BASE+'/retry/eslav_PP-OCRv5_mobile_rec_pretrained.pdparams': '7960917674b33127072d56e5f3e3bc9f4f819f36538d5deedeca212c96b3f45c'}
CONFIG_PATH = 'configs/reviews/ocr-training-runtime-v1.json'


def tree_files(root, name):
    base = inside(root, name)
    return sorted(p.relative_to(root).as_posix() for p in base.rglob('*') if p.is_file()
                  and '__pycache__' not in p.parts and p.suffix not in ('.pyc', '.pyo'))


def freeze_runtime(workspace, output=CONFIG_PATH):
    root = Path(workspace).resolve(); checked(root, PARENT)
    for name, sha in FIXED_ASSETS.items(): checked(root, {'path': name, 'sha256': sha})
    dictionary = SOURCE+'/ppocr/utils/dict/ppocrv5_eslav_dict.txt'
    upstream = SOURCE+'/configs/rec/PP-OCRv5/multi_language/eslav_PP-OCRv5_mobile_rec.yml'
    trees = [SOURCE+'/ppocr', BASE+'/runtime/python', BASE+'/venv']
    files = sorted(set(CODE+[dictionary, upstream, SOURCE+'/LICENSE',
        'artifacts/ocr-training-preparation-v1/inference-alphabet.json',
        'configs/ocr-training-runtime-v1.lock', BASE+'/install-report.json', BASE+'/install-report-setuptools.json', BASE+'/install-report-tqdm.json',
        BASE+'/checkpoint-http-head.json', BASE+'/model-card.json', BASE+'/model-card-README.md',
        BASE+'/setup-audit.json', BASE+'/downloads.json', BASE+'/retry/downloads.json',
        *FIXED_ASSETS, *[name for tree in trees for name in tree_files(root, tree)]]))
    config = {'schema_version': 'ocr-training-runtime-config/1.0', 'task': 'P16.2a.2', 'parent': PARENT,
              'source_root': SOURCE, 'source_commit': COMMIT, 'dictionary': dictionary, 'upstream_config': upstream,
              'checkpoint': BASE+'/retry/eslav_PP-OCRv5_mobile_rec_pretrained.pdparams',
              'inference_alphabet': 'artifacts/ocr-training-preparation-v1/inference-alphabet.json',
              'python': BASE+'/venv/Scripts/python.exe', 'source_trees': trees,
              'inputs': [pin(root, name) for name in files], 'seed': 20261108,
              'device': 'cpu', 'smoke_learning_rate': 0.0001, 'synthetic_optimizer_steps': 1,
              'customer_samples': 0, 'corpus_training_started': False, 'gpu_training_verified': False}
    target = inside(root, output); require(not target.exists(), 'Choose a new runtime config')
    target.write_bytes(encoded(config)); return pin(root, output)


def validate_runtime_config(root, config_pin):
    config = read(checked(root, config_pin))
    require(config['schema_version'] == 'ocr-training-runtime-config/1.0' and config['task'] == 'P16.2a.2'
            and config['parent'] == PARENT and config['source_commit'] == COMMIT
            and config['device'] == 'cpu' and config['seed'] == 20261108
            and config['synthetic_optimizer_steps'] == 1 and config['customer_samples'] == 0
            and config['corpus_training_started'] is False and config['gpu_training_verified'] is False,
            'Changed runtime smoke scope')
    require(config['source_root'] == SOURCE and config['python'] == BASE+'/venv/Scripts/python.exe'
            and config['dictionary'] == SOURCE+'/ppocr/utils/dict/ppocrv5_eslav_dict.txt'
            and config['upstream_config'] == SOURCE+'/configs/rec/PP-OCRv5/multi_language/eslav_PP-OCRv5_mobile_rec.yml'
            and config['checkpoint'] == BASE+'/retry/eslav_PP-OCRv5_mobile_rec_pretrained.pdparams'
            and config['inference_alphabet'] == 'artifacts/ocr-training-preparation-v1/inference-alphabet.json'
            and config['source_trees'] == [SOURCE+'/ppocr', BASE+'/runtime/python', BASE+'/venv']
            and config['smoke_learning_rate'] == 0.0001, 'Changed smoke dependency routing')
    checked(root, PARENT)
    names = [p['path'] for p in config['inputs']]
    require(len(names) == len(set(names)) and set(CODE).issubset(names), 'Incomplete/duplicated runtime pins')
    for p in config['inputs']: checked(root, p)
    for name, sha in FIXED_ASSETS.items():
        require({'path': name, 'sha256': sha} in config['inputs'], 'Changed public asset identity')
    for tree in config['source_trees']:
        require(set(tree_files(root, tree)) == {n for n in names if n.startswith(tree+'/')}, 'Runtime/source inventory changed')
    for key in ('python', 'checkpoint', 'dictionary', 'upstream_config', 'inference_alphabet'):
        require(config[key] in names, 'Unpinned runtime dependency')
    return config


def validate_smoke_report(report, config, config_pin):
    require(report['schema_version'] == 'ocr-training-native-smoke/1.0' and report['task'] == 'P16.2a.2'
            and report['config_sha256'] == config_pin['sha256'] and report['device'] == 'cpu'
            and report['source_commit'] == COMMIT and report['seed'] == config['seed'], 'Wrong smoke report identity')
    require(not report['missing'] and not report['unexpected'] and not report['shape_mismatches']
            and report['all_loaded_values_equal'] is True and report['loaded_state_entries'] > 0,
            'Partial checkpoint load')
    require(report['dictionary_order_matches_inference'] is True and report['dictionary_characters'] == 517
            and report['ctc_output_classes'] == 519 and report['nrtr_actual_output_classes'] == 523,
            'Wrong output alphabet')
    require(report['synthetic_optimizer_steps'] == 1 and report['updated_parameter_count'] > 0
            and report['gradient_parameter_count'] > 0 and report['model_and_optimizer_reload_equal'] is True
            and report['reloaded_forward_max_abs_difference'] == 0, 'Smoke step or reload failed')
    require(set(report['losses']) == {'CTCLoss', 'NRTRLoss', 'loss'}
            and all(math.isfinite(v) and v >= 0 for v in report['losses'].values())
            and math.isfinite(report['elapsed_seconds']) and report['elapsed_seconds'] > 0,
            'Non-finite or incomplete smoke measurements')
    require(set(report['gradient_branches']) == {'backbone.', 'head.ctc_', 'head.gtc_head.'}
            and all(v > 0 for v in report['gradient_branches'].values()), 'Missing gradient branch')
    require(report['compiled_with_cuda'] is False and report['python'].startswith('3.12.12 ')
            and report['executable'] == config['python'] and report['input_shape'] == [2, 3, 48, 320]
            and report['output_shape'] == [2, 40, 519] and report['smoke_learning_rate'] == config['smoke_learning_rate'],
            'Wrong native backend or synthetic input shape')
    versions = {r['name'].lower().replace('_', '-'): r['version'] for r in report['packages']}
    require(len(versions) == len(report['packages']) and versions.get('paddlepaddle') == '3.2.2'
            and versions.get('numpy') == '2.2.6', 'Wrong or ambiguous native package inventory')
    require(report['pretrained_state_sha256'] != report['updated_state_sha256'], 'Unchanged model state')
    require(report['customer_samples_read'] == 0 and report['corpus_training_started'] is False
            and report['test_used'] is False and report['production_model_updated'] is False
            and report['gpu_training_verified'] is False, 'Smoke report claims another scope')


def run_runtime(workspace, config_pin, output):
    root = Path(workspace).resolve(); config = validate_runtime_config(root, config_pin)
    target = inside(root, output); require(target.parent == root/'artifacts' and not target.exists(), 'Choose new artifacts output')
    command = [str(inside(root, config['python'])), str(inside(root, CODE[1])), '--workspace', str(root),
               '--config', str(checked(root, config_pin)), '--config-sha256', config_pin['sha256'], '--output', str(target)]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, encoding='utf-8', errors='replace', check=False, timeout=300)
    require(result.returncode == 0, 'Native smoke failed: '+result.stderr[-8000:])
    validate_runtime_config(root, config_pin)
    report = read(target/'report.json'); validate_smoke_report(report, config, config_pin)
    checked(target, {'path': 'synthetic-step.pdparams', 'sha256': report['saved_model_sha256']})
    checked(target, {'path': 'synthetic-step.pdopt', 'sha256': report['saved_optimizer_sha256']})
    (target/'stdout.txt').write_text(result.stdout, encoding='utf-8')
    (target/'stderr.txt').write_text(result.stderr, encoding='utf-8')
    names = sorted(p.relative_to(target).as_posix() for p in target.rglob('*') if p.is_file())
    manifest = {'schema_version': 'ocr-training-runtime-manifest/1.0', 'config': config_pin,
                'files': [pin(target, name) for name in names]}
    (target/'manifest.json').write_bytes(encoded(manifest)); return report


def verify_runtime(workspace, manifest_pin):
    root = Path(workspace).resolve(); path, manifest = _package(root, manifest_pin)
    require(manifest['schema_version'] == 'ocr-training-runtime-manifest/1.0', 'Unknown runtime manifest')
    config = validate_runtime_config(root, manifest['config'])
    require({p['path'] for p in manifest['files']} == {'report.json', 'synthetic-step.pdparams', 'synthetic-step.pdopt', 'stdout.txt', 'stderr.txt'}, 'Wrong smoke inventory')
    report = read(path.parent/'report.json'); validate_smoke_report(report, config, manifest['config'])
    checked(path.parent, {'path': 'synthetic-step.pdparams', 'sha256': report['saved_model_sha256']})
    checked(path.parent, {'path': 'synthetic-step.pdopt', 'sha256': report['saved_optimizer_sha256']})
    require(json.loads((path.parent/'stdout.txt').read_text('utf-8').splitlines()[-1]) == report, 'Saved native stdout/report differ')
    return report
