"""P16.4b: freeze before customer calls, compare twice, or replay without models."""
import argparse
from pathlib import Path

from cad_recovery.dataset_layout_v1 import inside, pin
from cad_recovery.vlm_comparison_v1 import freeze, run, verify


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('action', choices=['freeze', 'run', 'verify'])
    parser.add_argument('--config', default='configs/reviews/vlm-comparison-v1.json')
    parser.add_argument('--catalog', default='artifacts/vlm-comparison-inputs-v1')
    parser.add_argument('--output', default='artifacts/vlm-comparison-v1')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.action == 'freeze':
        print(freeze(root, inside(root, args.config), inside(root, args.catalog)))
    else:
        print((run if args.action == 'run' else verify)(root, pin(root, args.config), inside(root, args.output)))


if __name__ == '__main__':
    main()
