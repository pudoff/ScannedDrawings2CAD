"""Prepare and qualify SegFormer without training on customer drawings."""
import argparse
from pathlib import Path
from cad_recovery.dataset_layout_v1 import inside, pin
from cad_recovery.segformer_protocol_v1 import freeze_experiment, freeze_probe, probe


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('action', choices=['freeze-probe', 'probe', 'freeze-experiment'])
    p.add_argument('--config', default='configs/reviews/segformer-probe-v2.json')
    p.add_argument('--output', default='artifacts/segformer-qualification-v2')
    p.add_argument('--experiment', default='configs/reviews/segformer-experiment-v1.json')
    a = p.parse_args(); root = Path(__file__).resolve().parents[1]
    config, output = inside(root, a.config), inside(root, a.output)
    if a.action == 'freeze-probe': print(freeze_probe(root, config))
    elif a.action == 'probe': print(probe(root, pin(root, a.config), output))
    else: print(freeze_experiment(root, pin(root, a.config), output, inside(root, a.experiment)))


if __name__ == '__main__': main()
