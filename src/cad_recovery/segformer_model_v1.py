"""Local ImageNet MiT-B0 encoder, new ten-role head and native-grid loss.

Weights retain NVIDIA's research/evaluation-only restriction. This module
does not download assets, run a dataset or change a service profile.
"""
from pathlib import Path

from .dataset_layout_v1 import checked, read
from .family_catalog_v2 import require

POLICY = {
    'model_id': 'nvidia/mit-b0', 'revision': '80983a413c30d36a39c20203974ae7807835e2b4',
    'weights_sha256': '4af8348c2a802bf76115d34797ff5ce1d9f110bb8593c22d9b66d8bd7fa227fc',
    'implementation': 'transformers==4.57.1', 'pretraining': 'ImageNet-1K classification encoder only',
    'backbone': 'MiT-B0', 'depths': [2, 2, 2, 2], 'hidden_sizes': [32, 64, 160, 256],
    'num_labels': 10, 'ignore_index': 255, 'label_reduction': False,
    'head': 'new randomly initialized SegFormer MLP decoder; no ADE20K/Cityscapes head',
    'logit_restore': 'bilinear to target native size, align_corners=False; never interpolate labels',
    'loss': 'existing train-weighted masked CE, ignore255, normalized by sum of known pixel weights',
    'license_scope': 'research_or_evaluation_only', 'production_admitted': False,
}


def build_model(root, assets):
    import torch
    from transformers import SegformerConfig, SegformerForSemanticSegmentation
    source = read(checked(root, assets['config']))
    require(source['depths'] == POLICY['depths'] and source['hidden_sizes'] == POLICY['hidden_sizes'], 'Wrong encoder architecture')
    config = SegformerConfig(**source)
    config.id2label = {i: 'role_' + str(i) for i in range(10)}
    config.label2id = {v: k for k, v in config.id2label.items()}
    config.semantic_loss_ignore_index = 255
    model = SegformerForSemanticSegmentation(config)
    require(assets['weights']['sha256'] == POLICY['weights_sha256'], 'Wrong initial checkpoint')
    state = torch.load(checked(root, assets['weights']), map_location='cpu', weights_only=True)
    discarded = sorted(k for k in state if not k.startswith('segformer.'))
    require(discarded == ['classifier.bias', 'classifier.weight'], 'Unexpected classification checkpoint payload')
    encoder = {k.removeprefix('segformer.'): v for k, v in state.items() if k.startswith('segformer.')}
    model.segformer.load_state_dict(encoder, strict=True)
    return model


def restored_logits(logits, size):
    from torch.nn import functional as F
    require(len(size) == 2 and all(type(v) is int and v > 0 for v in size), 'Invalid target grid')
    return F.interpolate(logits, size=size, mode='bilinear', align_corners=False)


def partial_loss(logits, targets, weights):
    from .segmentation_experiment_v1 import masked_loss
    return masked_loss(restored_logits(logits, tuple(targets.shape[-2:])), targets, weights)
