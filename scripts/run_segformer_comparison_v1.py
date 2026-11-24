"""Freeze/run/replay the fixed SegFormer comparison with existing CV/U-Net."""
import argparse
from pathlib import Path
from cad_recovery.dataset_layout_v1 import inside, pin
from cad_recovery.segformer_comparison_v1 import freeze, run, verify, downstream


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('action', choices=['freeze', 'run', 'cv', 'verify'])
    p.add_argument('--config', default='configs/reviews/segformer-comparison-v1.json')
    p.add_argument('--output', default='artifacts/segformer-comparison-v1')
    a = p.parse_args(); root = Path(__file__).resolve().parents[1]
    config, output = inside(root, a.config), inside(root, a.output)
    if a.action == 'freeze': print(freeze(root, config)); return
    record = pin(root, a.config)
    if a.action == 'cv': downstream(root, record, output)
    elif a.action == 'verify': print(verify(root, record, output))
    else:
        result = run(root, record, output)
        print(result['decision'], result['necessary_partial_gates'])


if __name__ == '__main__': main()
