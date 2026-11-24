"""Architecture and acceptance for synthetic GPU-training qualification only."""
import hashlib
import json
import math
from pathlib import Path

POLICY = {
    'version': 'resnet18-unet-training-probe/1.0',
    'torch': '2.14.1+cu130', 'torchvision': '0.29.1+cu130', 'cuda': '13.0',
    'architecture': 'ResNet18 encoder, transposed-convolution U-Net decoder 256/128/64/32/16, six logits',
    'pretrained_weights': False, 'input_channels': 3, 'classes': 6,
    'optimizer': 'AdamW', 'learning_rate': .001, 'weight_decay': .01,
    'loss': 'mean softmax cross entropy via explicit log_softmax and one_hot; deterministic CUDA-compatible expression',
    'steps': 2, 'seed': 20261110, 'dtype': 'float32', 'AMP': False, 'TF32': False,
    'deterministic_algorithms': True, 'reload_atol': 1e-6, 'reload_rtol': 1e-5,
    'timeout_seconds': 300, 'host_memory_MiB': 6144,
    'max_allocated_MiB': 10240, 'max_reserved_MiB': 12288,
    'whole_device_memory_ceiling_MiB': 13312,
    'cases': [
        {'id': 's256-b1', 'side': 256, 'batch': 1, 'required': True, 'checkpoint': False},
        {'id': 's512-b1', 'side': 512, 'batch': 1, 'required': True, 'checkpoint': False},
        {'id': 's512-b2', 'side': 512, 'batch': 2, 'required': True, 'checkpoint': True},
        {'id': 's512-b4', 'side': 512, 'batch': 4, 'required': False, 'checkpoint': False},
        {'id': 's1024-b1', 'side': 1024, 'batch': 1, 'required': True, 'checkpoint': False},
        {'id': 's1024-b2', 'side': 1024, 'batch': 2, 'required': False, 'checkpoint': False},
        {'id': 's1024-b4', 'side': 1024, 'batch': 4, 'required': False, 'checkpoint': False},
        {'id': 's512-b2-repeat', 'side': 512, 'batch': 2, 'required': True, 'checkpoint': True},
    ],
    'synthetic_only': True, 'customer_samples': 0, 'closed_tests_executed': False,
    'quality_improvement_claimed': False, 'paddle_gpu_qualified': False,
    'transformer_or_VLM_capacity_qualified': False,
    'capacity_scope': 'largest passing tested batch per side; not an exhaustive maximum or a guarantee for another model',
}


def state_digest(state):
    """Stable digest of names, shapes, dtypes and values, not archive metadata."""
    result = hashlib.sha256()
    def visit(value):
        if isinstance(value, dict):
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
                result.update(str(key).encode('utf-8'))
                visit(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
        elif hasattr(value, 'numpy') or hasattr(value, 'tobytes'):
            array = value.detach().cpu().numpy() if hasattr(value, 'detach') else value
            result.update(str(array.dtype).encode())
            result.update(str(array.shape).encode())
            result.update(array.tobytes())
        else:
            result.update(json.dumps(value, sort_keys=True).encode())
    visit(state)
    return result.hexdigest()


def assess(rows):
    by_id = {row['id']: row for row in rows}
    assigned = {case['id'] for case in POLICY['cases']}
    if len(by_id) != len(rows) or set(by_id) != assigned:
        raise ValueError('Lost/repeated assigned training capacity case')
    required_pass = all(by_id[c['id']]['status'] == 'pass' for c in POLICY['cases'] if c['required'])
    errors = [r['id'] for r in rows if r['status'] not in ('pass', 'capacity_exceeded')]
    capacities = {}
    for case in POLICY['cases']:
        if by_id[case['id']]['status'] == 'pass':
            capacities[str(case['side'])] = max(capacities.get(str(case['side']), 0), case['batch'])
    repeat = by_id['s512-b2']['status'] == by_id['s512-b2-repeat']['status'] == 'pass' and by_id['s512-b2'].get('final_parameters_sha256') == by_id['s512-b2-repeat'].get('final_parameters_sha256')
    return {'qualified': required_pass and not errors and repeat, 'required_cases_pass': required_pass,
            'repeat_state_equal': repeat, 'unexpected_errors': errors, 'largest_tested_passing_batch': capacities,
            'scope': POLICY['capacity_scope'], 'paddle_gpu_qualified': False,
            'transformer_or_VLM_capacity_qualified': False, 'quality_improvement_claimed': False}


def validate_case_result(case, row):
    """A saved pass must still contain finite CUDA training and reload evidence."""
    if row['id'] != case['id'] or row.get('side') != case['side'] or row.get('batch') != case['batch']:
        raise ValueError('Changed assigned model shape or batch')
    if row['status'] != 'pass':
        return
    if row.get('torch') != POLICY['torch'] or row.get('cuda') != POLICY['cuda'] or row.get('dtype') != 'float32' or row.get('AMP') is not False:
        raise ValueError('Changed training precision/framework')
    if row.get('input_device') != 'cuda:0' or row.get('target_device') != 'cuda:0' or row.get('fp32_precision') != {'matmul': 'ieee', 'convolution': 'ieee'}:
        raise ValueError('Missing CUDA FP32 training placement')
    if not row.get('parameters_changed') or row.get('initial_parameters_sha256') == row.get('final_parameters_sha256'):
        raise ValueError('No parameter update evidence')
    if not 0 < row['allocated_peak_MiB'] <= POLICY['max_allocated_MiB'] or not row['allocated_peak_MiB'] <= row['reserved_peak_MiB'] <= POLICY['max_reserved_MiB']:
        raise ValueError('Training memory ceiling exceeded')
    if len(row['steps']) != POLICY['steps']:
        raise ValueError('Lost training update')
    names = row['parameter_tensor_names']
    if not names or len(set(names)) != len(names):
        raise ValueError('Invalid parameter inventory')
    for step in row['steps']:
        if not math.isfinite(step['loss']) or not math.isfinite(step['cuda_event_ms']) or step['cuda_event_ms'] <= 0:
            raise ValueError('Nonfinite loss or missing CUDA interval')
        gradients = step['gradients']
        if {g['name'] for g in gradients} != set(names) or len(gradients) != len(names):
            raise ValueError('Missing/repeated trainable gradient')
        if not all(g['device'] == 'cuda:0' and g['dtype'] == 'torch.float32' and g['finite'] is True and math.isfinite(g['norm']) and g['norm'] >= 0 for g in gradients):
            raise ValueError('Nonfinite/CPU/non-FP32 gradient')
        if not any(g['norm'] > 0 for g in gradients):
            raise ValueError('All gradients are zero')
    if case['checkpoint']:
        saved = row['checkpoint_probe']
        if not saved or not all(saved[k] is True for k in ('reload_state_equal', 'reload_prediction_allclose', 'resume_state_equal')) or saved['reference_resume_sha256'] != saved['restored_resume_sha256']:
            raise ValueError('Missing successful checkpoint/resume evidence')


def build_model():
    """No downloads: architecture only; classification head is not registered."""
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torchvision.models import resnet18

    def double_conv(inputs, outputs):
        return nn.Sequential(nn.Conv2d(inputs, outputs, 3, padding=1, bias=False), nn.GroupNorm(8, outputs), nn.ReLU(),
                             nn.Conv2d(outputs, outputs, 3, padding=1, bias=False), nn.GroupNorm(8, outputs), nn.ReLU())

    class Up(nn.Module):
        def __init__(self, inputs, skip, outputs):
            super().__init__()
            self.up = nn.ConvTranspose2d(inputs, outputs, 2, stride=2)
            self.conv = double_conv(skip+outputs, outputs)

        def forward(self, x, skip):
            x = self.up(x)
            x = x[..., :skip.shape[-2], :skip.shape[-1]]
            right, bottom = skip.shape[-1]-x.shape[-1], skip.shape[-2]-x.shape[-2]
            if right or bottom:
                x = F.pad(x, (0, right, 0, bottom))
            return self.conv(torch.cat((x, skip), dim=1))

    class ResNet18UNetProbe(nn.Module):
        def __init__(self):
            super().__init__()
            encoder = resnet18(weights=None)
            self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
            self.pool = encoder.maxpool
            self.layer1, self.layer2 = encoder.layer1, encoder.layer2
            self.layer3, self.layer4 = encoder.layer3, encoder.layer4
            self.up3, self.up2 = Up(512, 256, 256), Up(256, 128, 128)
            self.up1, self.up0 = Up(128, 64, 64), Up(64, 64, 32)
            self.final = nn.Sequential(nn.ConvTranspose2d(32, 16, 2, stride=2), double_conv(16, 16), nn.Conv2d(16, POLICY['classes'], 1))

        def forward(self, x):
            height, width = x.shape[-2:]
            stem = self.stem(x)
            first = self.layer1(self.pool(stem))
            second = self.layer2(first)
            third = self.layer3(second)
            bottom = self.layer4(third)
            x = self.up0(self.up1(self.up2(self.up3(bottom, third), second), first), stem)
            return self.final(x)[..., :height, :width]

    return ResNet18UNetProbe()


def segmentation_probe_loss(logits, targets):
    """Same unweighted mean cross-entropy, without nondeterministic CUDA NLL2d."""
    import torch
    from torch.nn import functional as F
    labels = F.one_hot(targets, num_classes=logits.shape[1]).movedim(-1, 1).to(logits.dtype)
    return -(torch.log_softmax(logits, dim=1)*labels).sum(dim=1).mean()
