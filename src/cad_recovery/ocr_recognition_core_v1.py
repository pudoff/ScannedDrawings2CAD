"""Fixed recognition tensors, literal CTC decoding and train label/selection policy."""
from copy import deepcopy
import hashlib
import math
import os
from pathlib import Path
import sys

import cv2
import numpy as np

from .ocr_baseline_metrics_v1 import aggregate_scores_v1

KINDS = ('legacy_identifier', 'name_literal', 'dimension_literal')
PREPROCESS = {'shape_chw': [3, 48, 320], 'input': 'original RGB crop converted to contiguous BGR',
    'resize': 'cv2.INTER_LINEAR; width=min(320,ceil(48*w/h)); full image, no clipping',
    'normalization': 'float32 CHW /255; subtract 0.5; divide by 0.5',
    'padding': 'right zero in normalized space', 'valid_ratio_mask': False,
    'decode': 'greedy CTC: collapse adjacent timestep classes then drop blank=0; no confidence filtering or text repair'}
TRAINING = {'seed': 20261108, 'device': 'cpu', 'epochs': 20, 'batch_size': 8, 'drop_last': False,
    'num_workers': 0, 'cpu_threads': 4, 'optimizer': 'Adam', 'learning_rate': 0.00001,
    'beta1': 0.9, 'beta2': 0.999, 'weight_decay': 0.00003, 'schedule': 'constant',
    'loss': 'upstream MultiLoss CTCLoss+NRTRLoss', 'max_text_length': 25,
    'parameters': 'all native trainable parameters; native BatchNorm train behavior',
    'augmentation': 'none', 'sampler': 'numpy default_rng(seed), new permutation each epoch; every accepted train row exactly once',
    'selection': 'include epoch zero; strict improvement of mean raw CER across three unique-image validation strata; dimension raw CER must not increase; no extra errors/empty; ties prefer earlier epoch',
    'test_policy': 'no test inference before checkpoint selection and export acceptance; one final evaluation in P16.2b',
    'production_update': 'manual project decision after comparison; training runner never replaces deployed ONNX'}

EXPORT_BUDGET = {'max_absolute_probability_error':0.0005,'max_relative_l2_error':0.0001,
    'all_timestep_classes_identical':True,'original_strict_rtol':0.0001,'original_strict_atol':0.00001,
    'revision':'Explicit engineering budget after v2/v3 numeric diagnosis; original strict result retained. Not a blind numeric preregistration.'}


def compare_export(reference,actual):
    a=np.asarray(reference);b=np.asarray(actual)
    if a.shape!=b.shape or a.ndim!=2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Invalid probability comparison')
    difference=a.astype('float64')-b.astype('float64')
    maximum=float(np.abs(difference).max())
    relative=float(np.linalg.norm(difference)/max(np.linalg.norm(a.astype('float64')),np.finfo('float64').tiny))
    identical=bool(np.array_equal(a.argmax(-1),b.argmax(-1)))
    return {'allclose':maximum<=EXPORT_BUDGET['max_absolute_probability_error']
                and relative<=EXPORT_BUDGET['max_relative_l2_error'] and identical,
            'max_abs':maximum,'relative_l2':relative,'timestep_classes_equal':identical,
            'original_strict_allclose':bool(np.allclose(a,b,rtol=0.0001,atol=0.00001))}


def tensor_rgb(rgb):
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) <= 0:
        raise ValueError('Expected nonempty RGB uint8 crop')
    h, w = rgb.shape[:2]; width = min(320, math.ceil(48*w/h))
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    scaled = cv2.resize(bgr, (width, 48), interpolation=cv2.INTER_LINEAR).astype('float32')
    scaled = scaled.transpose((2, 0, 1))/255; scaled -= 0.5; scaled /= 0.5
    result = np.zeros((3, 48, 320), dtype='float32'); result[:, :, :width] = scaled
    return result, {'original_hw': [h, w], 'resized_width': width, 'valid_ratio': width/320,
                    'width_compressed_to_limit': math.ceil(48*w/h) > 320,
                    'tensor_sha256': hashlib.sha256(result.tobytes()).hexdigest()}


def decode_ctc(probabilities, characters):
    a = np.asarray(probabilities)
    if a.ndim != 2 or a.shape[1] != len(characters)+2 or not np.isfinite(a).all():
        raise ValueError('Invalid CTC output or alphabet')
    vocab = ['']+list(characters)+[' ']; indices = a.argmax(1)
    selected = [i for i, idx in enumerate(indices) if idx != 0 and (i == 0 or idx != indices[i-1])]
    return ''.join(vocab[int(indices[i])] for i in selected), float(np.mean([a[i, indices[i]] for i in selected])) if selected else 0.0


def encode_targets(texts, characters, max_length=25):
    vocab = {c: i for i, c in enumerate(list(characters)+[' '])}
    ctc = np.zeros((len(texts), max_length), dtype='int64'); nrtr = ctc.copy(); lengths = []
    for i, text in enumerate(texts):
        if not isinstance(text, str) or not text or len(text) > max_length-2 or any(c not in vocab for c in text):
            raise ValueError('Target would be skipped/truncated or contains an unsupported literal character')
        ctc[i, :len(text)] = [vocab[c]+1 for c in text]
        nrtr[i, :len(text)+2] = [2]+[vocab[c]+4 for c in text]+[3]; lengths.append(len(text))
    return ctc, nrtr, np.array(lengths, dtype='int64')


def epoch_batches(count, rng, batch_size=8):
    if count <= 0 or batch_size <= 0: raise ValueError('Empty or invalid training batch policy')
    order = rng.permutation(count)
    return [order[i:i+batch_size].tolist() for i in range(0, count, batch_size)]


def selection_score(rows, predictions):
    by = {p['id']: p for p in predictions}
    if len(by) != len(predictions) or set(by) != {r['id'] for r in rows}: raise ValueError('Incomplete checkpoint predictions')
    representatives = {}
    for row in sorted(rows, key=lambda r: r['id']): representatives.setdefault(row['image_sha256'], row)
    scores = {}
    for kind in KINDS:
        part = [r for r in representatives.values() if r['text_kind'] == kind]
        if not part: raise ValueError('Checkpoint selection needs all three validation strata')
        summary = aggregate_scores_v1([{'id':r['id'],'source_id':r['source_id'],'reference':r['text'],
                    'prediction':by[r['id']]['prediction'],'status':by[r['id']]['status']} for r in part])
        scores[kind] = {'cer':summary['raw']['cer'],'distance':summary['raw']['distance'],
                       'reference_characters':summary['raw']['reference_characters'],'exact_matches':summary['raw']['exact_matches'],
                       'count':len(part),'bad_statuses':sum(by[r['id']]['status'] != 'ok' for r in part)}
    return {'macro_raw_cer':sum(v['cer'] for v in scores.values())/3,
            'dimension_raw_cer':scores['dimension_literal']['cer'], 'by_kind':scores,
            'bad_statuses':sum(v['bad_statuses'] for v in scores.values())}


def improves(candidate, best, baseline):
    return (candidate['macro_raw_cer'] < best['macro_raw_cer']
            and candidate['dimension_raw_cer'] <= baseline['dimension_raw_cer']
            and candidate['bad_statuses'] <= baseline['bad_statuses'])


def native_environment(root, source_root):
    os.environ['OMP_NUM_THREADS'] = '4'; os.environ['MKL_NUM_THREADS'] = '4'; os.environ['FLAGS_use_mkldnn'] = '0'
    profile = Path(root)/'artifacts/ocr-recognition-setup-v1/cache-profile'; profile.mkdir(exist_ok=True)
    os.environ['USERPROFILE'] = str(profile)
    sys.path.insert(0, str(Path(root)/source_root))


def native_model(root, runtime):
    import paddle
    import yaml
    from ppocr.modeling.architectures import build_model
    config = yaml.safe_load((Path(root)/runtime['upstream_config']).read_text('utf-8'))
    chars = (Path(root)/runtime['dictionary']).read_text('utf-8').splitlines()
    architecture = deepcopy(config['Architecture'])
    architecture['Head']['out_channels_list'] = {'CTCLabelDecode':len(chars)+2, 'NRTRLabelDecode':len(chars)+5}
    with paddle.utils.unique_name.guard(): model = build_model(architecture)
    weights = paddle.load(str(Path(root)/runtime['checkpoint']))
    weights.pop('StructuredToParameterName@@', None); state = model.state_dict()
    if set(weights) != set(state) or any(list(weights[k].shape) != list(state[k].shape) for k in state):
        raise ValueError('Native checkpoint keys/shapes differ')
    model.set_state_dict(weights)
    for k in state: np.testing.assert_array_equal(model.state_dict()[k].numpy(), weights[k].numpy() if hasattr(weights[k], 'numpy') else weights[k])
    model.eval(); return model, chars, config
