"""Run only with the isolated Paddle interpreter. Synthetic tensors, no dataset."""
import argparse
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time


def digest(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def state_digest(state):
    result = hashlib.sha256()
    def visit(value):
        if isinstance(value, dict):
            for name, item in sorted(value.items()): result.update(name.encode()); visit(item)
        elif hasattr(value, 'numpy') or hasattr(value, 'tobytes'):
            array = value.numpy() if hasattr(value, 'numpy') else value
            result.update(str(array.dtype).encode()); result.update(str(array.shape).encode()); result.update(array.tobytes())
        else: result.update(json.dumps(value, sort_keys=True).encode())
    visit(state)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); root = args.workspace.resolve()
    config_path = args.config.resolve()
    if digest(config_path) != args.config_sha256: raise ValueError('Changed smoke config')
    config = json.loads(config_path.read_text('utf-8'))
    for record in config['inputs']:
        if digest(root/record['path']) != record['sha256']: raise ValueError('Changed smoke input: '+record['path'])
    target = args.output.resolve()
    if target.parent != root/'artifacts' or target.exists(): raise ValueError('Choose a new artifacts output')
    os.environ['OMP_NUM_THREADS'] = '4'; os.environ['MKL_NUM_THREADS'] = '4'
    os.environ['FLAGS_use_mkldnn'] = '0'
    # Paddle 3.2.2's dataset import hardcodes expanduser('~')/.cache.
    # Redirect only this child process profile; the host profile is untouched.
    cache_profile = root/'artifacts/ocr-training-setup-v1/cache-profile'
    cache_profile.mkdir(exist_ok=True)
    os.environ['USERPROFILE'] = str(cache_profile)
    sys.path.insert(0, str(root/config['source_root']))
    import numpy as np
    import paddle
    import yaml
    from ppocr.modeling.architectures import build_model
    from ppocr.losses import build_loss

    started = time.perf_counter()
    paddle.set_device('cpu'); paddle.seed(config['seed']); np.random.seed(config['seed']); random.seed(config['seed'])
    dictionary = (root/config['dictionary']).read_text('utf-8').splitlines()
    alphabet = json.loads((root/config['inference_alphabet']).read_text('utf-8'))
    if dictionary != alphabet['characters']: raise ValueError('Training/inference dictionary order differs')
    if len(dictionary) != len(set(dictionary)) or any(len(c) != 1 for c in dictionary): raise ValueError('Invalid dictionary')
    upstream = yaml.safe_load((root/config['upstream_config']).read_text('utf-8'))
    architecture = deepcopy(upstream['Architecture']); n_ctc = len(dictionary)+2
    architecture['Head']['out_channels_list'] = {'CTCLabelDecode': n_ctc, 'NRTRLabelDecode': n_ctc+3}
    original = paddle.load(str(root/config['checkpoint']))
    weights = {k: v for k, v in original.items() if k != 'StructuredToParameterName@@'}
    def make_model():
        with paddle.utils.unique_name.guard(): return build_model(deepcopy(architecture))
    model = make_model(); actual = model.state_dict()
    missing = sorted(set(actual)-set(weights)); unexpected = sorted(set(weights)-set(actual))
    mismatches = [k for k in set(actual)&set(weights) if list(actual[k].shape) != list(weights[k].shape)]
    if missing or unexpected or mismatches: raise ValueError(json.dumps({'missing': missing, 'unexpected': unexpected, 'shape_mismatches': mismatches}))
    model.set_state_dict(weights)
    for name in weights:
        np.testing.assert_array_equal(model.state_dict()[name].numpy(), weights[name].numpy() if hasattr(weights[name], 'numpy') else weights[name])
    pretrained_digest = state_digest(model.state_dict())
    data = np.random.default_rng(config['seed']).uniform(-1, 1, (2, 3, 48, 320)).astype('float32')
    x = paddle.to_tensor(data); texts = ['A1', '\u04112']; vocabulary = dictionary+[' ']
    max_length = upstream['Global']['max_text_length']
    ctc = np.zeros((2, max_length), dtype='int64'); nrtr = ctc.copy()
    for i, text in enumerate(texts):
        ctc[i, :len(text)] = [vocabulary.index(c)+1 for c in text]
        nrtr[i, :len(text)+2] = [2]+[vocabulary.index(c)+4 for c in text]+[3]
    batch = [x, paddle.to_tensor(ctc), paddle.to_tensor(nrtr),
             paddle.to_tensor(np.array([2, 2], dtype='int64')), paddle.to_tensor(np.ones(2, dtype='float32'))]
    model.eval()
    with paddle.no_grad(): before = model(x).numpy()
    if before.shape[-1] != n_ctc or not np.isfinite(before).all(): raise ValueError('Invalid pretrained forward')
    parameters_before = {name: p.numpy().copy() for name, p in model.named_parameters() if not p.stop_gradient}
    optimizer = paddle.optimizer.Adam(learning_rate=config['smoke_learning_rate'], parameters=model.parameters())
    criterion = build_loss(deepcopy(upstream['Loss']))
    model.train(); prediction = model(x, data=batch[1:]); losses = criterion(prediction, batch)
    scalars = {k: float(v.numpy().item()) for k, v in losses.items()}
    if not all(np.isfinite(v) for v in scalars.values()): raise ValueError('Non-finite synthetic loss')
    losses['loss'].backward()
    gradients = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            grad = parameter.grad.numpy()
            if not np.isfinite(grad).all(): raise ValueError('Non-finite gradient: '+name)
            gradients[name] = float(np.linalg.norm(grad))
    for prefix in ('backbone.', 'head.ctc_', 'head.gtc_head.'):
        if not any(k.startswith(prefix) and value > 0 for k, value in gradients.items()): raise ValueError('Missing gradient branch: '+prefix)
    optimizer.step(); optimizer.clear_grad()
    changed = [name for name, p in model.named_parameters() if not p.stop_gradient and not np.array_equal(parameters_before[name], p.numpy())]
    if not changed: raise ValueError('No parameter updated')
    model.eval()
    with paddle.no_grad(): after = model(x).numpy()
    if not np.isfinite(after).all(): raise ValueError('Non-finite trained forward')
    target.mkdir()
    paddle.save(model.state_dict(), str(target/'synthetic-step.pdparams'))
    paddle.save(optimizer.state_dict(), str(target/'synthetic-step.pdopt'))
    restored = make_model()
    restored.set_state_dict(paddle.load(str(target/'synthetic-step.pdparams')))
    restored_optimizer = paddle.optimizer.Adam(learning_rate=config['smoke_learning_rate'], parameters=restored.parameters())
    restored_optimizer.set_state_dict(paddle.load(str(target/'synthetic-step.pdopt')))
    if state_digest(restored.state_dict()) != state_digest(model.state_dict()): raise ValueError('Reloaded model differs')
    if state_digest(restored_optimizer.state_dict()) != state_digest(optimizer.state_dict()): raise ValueError('Reloaded optimizer differs')
    restored.eval()
    with paddle.no_grad(): reloaded = restored(x).numpy()
    np.testing.assert_array_equal(after, reloaded)
    for record in config['inputs']:
        if digest(root/record['path']) != record['sha256']: raise ValueError('Smoke input changed during execution')
    report = {'schema_version': 'ocr-training-native-smoke/1.0', 'task': 'P16.2a.2',
              'config_sha256': args.config_sha256, 'device': paddle.get_device(),
              'compiled_with_cuda': paddle.is_compiled_with_cuda(), 'gpu_training_verified': False,
              'python': sys.version, 'executable': Path(sys.executable).resolve().relative_to(root).as_posix(),
              'source_commit': config['source_commit'], 'seed': config['seed'],
              'dictionary_order_matches_inference': True, 'dictionary_characters': len(dictionary),
              'ctc_output_classes': n_ctc, 'nrtr_actual_output_classes': int(prediction['gtc'].shape[-1]),
              'loaded_state_entries': len(weights), 'missing': missing, 'unexpected': unexpected,
              'shape_mismatches': mismatches, 'all_loaded_values_equal': True,
              'synthetic_input_sha256': hashlib.sha256(data.tobytes()).hexdigest(), 'synthetic_texts': texts,
              'input_shape': list(data.shape), 'output_shape': list(after.shape), 'losses': scalars,
              'gradient_parameter_count': len(gradients), 'updated_parameter_count': len(changed),
              'gradient_branches': {prefix: sum(k.startswith(prefix) and value > 0 for k, value in gradients.items())
                                    for prefix in ('backbone.', 'head.ctc_', 'head.gtc_head.')},
              'synthetic_optimizer_steps': 1, 'smoke_learning_rate': config['smoke_learning_rate'],
              'pretrained_state_sha256': pretrained_digest, 'updated_state_sha256': state_digest(model.state_dict()),
              'optimizer_state_sha256': state_digest(optimizer.state_dict()),
              'saved_model_sha256': digest(target/'synthetic-step.pdparams'),
              'saved_optimizer_sha256': digest(target/'synthetic-step.pdopt'),
              'model_and_optimizer_reload_equal': True, 'reloaded_forward_max_abs_difference': float(np.max(np.abs(after-reloaded))),
              'elapsed_seconds': time.perf_counter()-started,
              'packages': sorted([{'name':d.metadata['Name'],'version':d.version}
                                  for d in importlib.metadata.distributions(path=[str(Path(sys.prefix)/'Lib/site-packages')])],
                                 key=lambda r:r['name'].lower()),
              'customer_samples_read': 0, 'corpus_training_started': False, 'test_used': False,
              'production_model_updated': False, 'remaining_training_gates': ['P16.2a.3 validation', 'P16.2a.4 experiment freeze']}
    (target/'report.json').write_text(json.dumps(report,ensure_ascii=True,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True))


if __name__ == '__main__': main()
