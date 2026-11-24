"""A fresh offline GPU child for one frozen SegFormer variant."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HOME'] = str(ROOT / 'artifacts/.cache/huggingface')

if __name__ == '__main__':
    from cad_recovery.dataset_layout_v1 import inside
    from cad_recovery.gpu_inference_v1 import deny_python_network
    from cad_recovery.segformer_training_v1 import train
    parser = argparse.ArgumentParser(__doc__)
    for name in ('config', 'sha256', 'variant', 'output'): parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    with deny_python_network():
        train(ROOT, {'path': args.config, 'sha256': args.sha256}, args.variant, inside(ROOT, args.output))
