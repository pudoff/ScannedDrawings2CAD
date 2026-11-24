"""Freeze a research-only SegFormer experiment after synthetic qualification."""
from pathlib import Path
import math

from .dataset_layout_v1 import checked, encoded, pin, read
from .family_catalog_v2 import require
from .segformer_model_v1 import POLICY as MODEL

BASELINE = {'path': 'artifacts/architecture-baseline-release-v1.json',
            'sha256': '69baaf578b7b58df816312becbe8423293423cc80c5def3ab6b6be6ab20758d9'}
ASSETS = 'artifacts/segformer-assets-v1'
EXTRA = 'artifacts/environments/segformer-v1'
GPU_PYTHON = 'artifacts/environments/gpu-training-v1/Scripts/python.exe'
CODE = ['src/cad_recovery/segformer_model_v1.py', 'src/cad_recovery/segformer_protocol_v1.py',
        'scripts/probe_segformer_v1.py', 'scripts/prepare_segformer_protocol_v1.py',
        'src/cad_recovery/segmentation_revision_v1.py', 'src/cad_recovery/segmentation_experiment_v1.py',
        'src/cad_recovery/gpu_training_probe_v1.py', 'src/cad_recovery/gpu_inference_v1.py',
        'src/cad_recovery/hard_raster_runs_v1.py', 'src/cad_recovery/service_v1/process.py']


def environment(root):
    import importlib.metadata
    report = read(root / ASSETS / 'install-report.json')
    distributions = {d.metadata['Name'].lower().replace('_', '-'): d for d in
                     importlib.metadata.distributions(path=[str(root / EXTRA)])}
    result = []
    for row in report['install']:
        name, version = row['metadata']['name'], row['metadata']['version']
        key = name.lower().replace('_', '-')
        require(key in distributions and distributions[key].version == version, 'Incomplete or changed isolated install: ' + name)
        record = Path(distributions[key]._path) / 'RECORD'
        result.append({'name': name, 'version': version, 'download': row['download_info'],
                       'RECORD': pin(root, record.relative_to(root).as_posix())})
    names = ('transformers/models/segformer/modeling_segformer.py',
             'transformers/models/segformer/configuration_segformer.py')
    return {'packages': result, 'install_report': pin(root, ASSETS + '/install-report.json'),
            'model_code': [pin(root, EXTRA + '/' + name) for name in names],
            'base_GPU_environment': pin(root, 'configs/gpu-training-v1.json'),
            'base_GPU_lock': pin(root, 'configs/gpu-training-v1.lock')}


def freeze_probe(root, output):
    require(not output.exists(), 'Choose a new qualification config')
    downloads = read(root / ASSETS / 'downloads.json')
    require(downloads['revision'] == MODEL['revision'], 'Changed model revision')
    assets = {'config': pin(root, ASSETS + '/config.json'), 'weights': pin(root, ASSETS + '/pytorch_model.bin'),
              'license': pin(root, ASSETS + '/nvidia-LICENSE.txt')}
    require(assets['weights']['sha256'] == MODEL['weights_sha256'], 'Changed ImageNet weights')
    config = {'schema_version': 'segformer-probe-config/1.0', 'task': 'P16.3b.1',
        'model': MODEL, 'assets': assets, 'environment': environment(root),
        'code': [pin(root, p) for p in CODE], 'dependency_directory': EXTRA,
        'download_evidence': pin(root, ASSETS + '/downloads.json'),
        'resources': {'seconds': 900, 'host_MiB': 6144, 'allocated_MiB': 4096, 'reserved_MiB': 6144},
        'synthetic_only': True, 'customer_samples': 0, 'dataset_training_started': False}
    output.write_bytes(encoded(config))
    return pin(root, output.relative_to(root).as_posix())


def probe(root, config_pin, output):
    from .hard_raster_runs_v1 import execute
    config = read(checked(root, config_pin))
    require(config['model'] == MODEL and config['environment'] == environment(root), 'Changed probe model/environment')
    require(config['code'] == [pin(root, p) for p in CODE], 'Changed probe implementation')
    require(not output.exists(), 'Choose a fresh qualification run')
    output.mkdir(parents=True)
    command = [str(root / GPU_PYTHON), '-u', str(root / 'scripts/probe_segformer_v1.py'),
        '--config', config_pin['path'], '--sha256', config_pin['sha256'],
        '--output', (output / 'result').relative_to(root).as_posix()]
    runtime = execute(root, command, output / 'process', config['resources']['seconds'], config['resources']['host_MiB'])
    (output / 'runtime.json').write_bytes(encoded(runtime))
    require(runtime['budget_pass'], 'Synthetic probe failed; retain logs and do not admit dataset training')
    result = read(output / 'result/report.json')
    require(result['qualified'] is True and result['customer_samples'] == 0, 'Invalid synthetic qualification')
    return result


def validate_qualification(config, qualification, runtime):
    """Reject a report from another checkpoint or an incomplete synthetic run."""
    require(qualification['qualified'] is True and runtime['budget_pass'] is True,
            'Unqualified SegFormer capacity')
    require(qualification['model'] == config['assets'] and qualification['torch'] == '2.14.1+cu130'
            and qualification['transformers'] == '4.57.1' and qualification['cuda'] == '13.0',
            'Qualification model/backend mismatch')
    require(qualification['customer_samples'] == 0 and qualification['dataset_training_started'] is False
            and qualification['quality_evaluation_executed'] is False and qualification['production_admitted'] is False,
            'Qualification must use synthetic data only')
    cases = qualification['results']
    require(len(cases) == 2 and [c['repeat'] for c in cases] == [0, 1], 'Missing synthetic repeat')
    require(cases[0]['initial'] == cases[1]['initial'] and cases[0]['final'] == cases[1]['final']
            and cases[0]['losses'] == cases[1]['losses'], 'Synthetic repeat differs')
    for case in cases:
        require(case['initial'] != case['final'] and case['checkpoint_resume_equal'] is True
                and len(case['losses']) == 2 and all(math.isfinite(v) for v in case['losses']),
                'Incomplete synthetic updates/resume')
        require(case['allocated_peak_MiB'] <= 4096 and case['reserved_peak_MiB'] <= 6144,
                'Unqualified memory budget')
        require(case['inference'] == [{'finite': True, 'shape': [1, 10, s, s], 'side': s} for s in (256, 512)],
                'Incomplete native inference probes')


def freeze_experiment(root, probe_config, result_folder, output):
    require(not output.exists(), 'Choose a new experiment protocol')
    qualification = read(result_folder / 'result/report.json')
    runtime = read(result_folder / 'runtime.json')
    config = read(checked(root, probe_config))
    validate_qualification(config, qualification, runtime)
    require(config['model'] == MODEL and config['code'] == [pin(root, p) for p in CODE] and
            config['environment'] == environment(root), 'Changed qualified model/code/environment')
    release = read(checked(root, BASELINE))
    old_path = 'configs/reviews/segmentation-revision-v2.json'
    old = read(checked(root, {'path': old_path, 'sha256': '90656866e73dabdc3a31915597aaa4a8a7ee0b4529b81dcb6a386c943d6681c3'}))
    from .segmentation_experiment_v1 import class_weights
    train_ids = old['training_patch_ids']
    require(len(train_ids) == 15 and len(old['inventory']['balanced']) == 485 and
            len(old['schedules']['balanced']) == 120, 'Changed train-only sampling recipe')
    value = {'schema_version': 'segformer-experiment/1.0', 'task': 'P16.3b.2',
        'model': MODEL, 'qualification_config': probe_config,
        'qualification': pin(root, (result_folder / 'result/report.json').relative_to(root).as_posix()),
        'qualification_runtime': pin(root, (result_folder / 'runtime.json').relative_to(root).as_posix()),
        'baseline_release': BASELINE,
        'baseline_manifest': {'path': 'artifacts/architecture-baseline-v1/manifest.json',
                              'sha256': release['files']['artifacts/architecture-baseline-v1/manifest.json']},
        'labels': old['labels'], 'environment': config['environment'], 'assets': config['assets'],
        'initial_seed': 20261110, 'updates': 120, 'batch': 4, 'optimizer': 'AdamW',
        'learning_rate': .0003, 'weight_decay': .01, 'AMP': False, 'TF32': False,
        'deterministic_algorithms': True, 'primary': 'mit-b0-balanced-contrast',
        'variants': ['mit-b0-balanced-contrast', 'mit-b0-balanced-contrast-repeat'],
        'checkpoint': 'fixed final update120 only; no diagnostic-validation checkpoint selection',
        'training_patch_ids': train_ids, 'inventory': old['inventory']['balanced'],
        'schedule': old['schedules']['balanced'], 'train_class_counts': old['train_class_counts'],
        'class_weights': class_weights(old['train_class_counts']).tolist(),
        'input': {'tile_side': 256, 'inference_stride': 192, 'native_samples_unscaled': True,
            'contrast': old['policy']['contrast'], 'mode': 'gray replicated to RGB',
            'normalization_mean': [.485, .456, .406], 'normalization_std': [.229, .224, .225],
            'label_reduction': False, 'padding': 'white RGB / ignore255 target',
            'inference': 'overlap-average restored native logits before argmax; all25 fixed ROI'},
        'resources': {'seconds_per_child': 900, 'total_train_child_seconds': 1800,
            'host_MiB': 6144, 'allocated_MiB': 4096, 'reserved_MiB': 6144, 'device_MiB': 8192},
        'necessary_partial_gates': {'each_foreground_IoU_min': .5,
            'background_false_foreground_max': .01, 'thin_axis_connected_80_min': .8,
            'no_per_ROI_CV_ink_regression': True, 'no_per_ROI_arrow_or_negative_regression': True,
            'repeat_parameters_and_predictions_equal': True, 'resource_pass': True},
        'comparison': {'pixel_ROIs': 25, 'CV_geometry_ROIs': 25, 'arrow_ROIs': 13,
            'CV_profile': 'regional_masked', 'primitive_coverage_band_px': 3,
            'CV_semantic_IoU': None, 'evaluation': 'known labels only; per split, color stratum, class and source group',
            'unsupported_strata': 'null, never zero error',
            'pretraining_confound': 'Both encoders ImageNet-1K; different weights, encoder, decoder and normalization layers. This is a pipeline comparison, not isolated causal architecture attribution.'},
        'closed_test_used': False, 'validation_model_selection': False, 'dataset_training_started': False,
        'physical_geometry_approved': False, 'independent_evaluation': False, 'production_admitted': False,
        'license_scope': 'research_or_evaluation_only; no commercial profile or redistribution admission',
        'VLM_capacity_qualified': False, 'next_action': 'implement/run frozen P16.3b.2; preserve baseline and service profiles'}
    output.write_bytes(encoded(value))
    return pin(root, output.relative_to(root).as_posix())
