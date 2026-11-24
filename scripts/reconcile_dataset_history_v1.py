"""Reconcile historical unmatched records with the current pinned dataset."""
import argparse
import json
from pathlib import Path
from cad_recovery.dataset_history_v1 import build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path.cwd())
    parser.add_argument('--config', default='configs/reviews/dataset-history-v1.json')
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = build(args.workspace, {'path': args.config, 'sha256': args.config_sha256}, args.output)
    print(json.dumps(result, ensure_ascii=True))


if __name__ == '__main__':
    main()
