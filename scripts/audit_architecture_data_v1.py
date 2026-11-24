"""Freeze, run or replay the P10.2a.4 read-only readiness audit."""
import argparse
from pathlib import Path

from cad_recovery.architecture_data_readiness_v1 import freeze, verify, write
from cad_recovery.dataset_layout_v1 import inside, pin


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('action', choices=['freeze', 'run', 'verify'])
    parser.add_argument('--config', default='configs/reviews/architecture-data-readiness-v5.json')
    parser.add_argument('--output', default='artifacts/architecture-data-readiness-v3')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = inside(root, args.config)
    if args.action == 'freeze':
        print(freeze(root, config))
    elif args.action == 'run':
        report = write(root, pin(root, args.config), inside(root, args.output))
        print({k: report[k] for k in ['patch_counts', 'source_count', 'saved_unet_pixel_ROIs', 'saved_cv_ROIs']})
    else:
        print(verify(root, pin(root, args.config), inside(root, args.output)))


if __name__ == '__main__': main()
