"""Freeze or execute a phase of the dataset content audit."""
import argparse
import json
from pathlib import Path
from cad_recovery.dataset_content_v1 import prepare_config, run_phase


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--output', default='configs/reviews/dataset-content-v1.json')
    p = sub.add_parser('run')
    p.add_argument('--config', default='configs/reviews/dataset-content-v1.json')
    p.add_argument('--config-sha256', required=True)
    p.add_argument('--phase', choices=('native', 'pdf', 'controls', 'pdf_pairs'), required=True)
    p.add_argument('--output', default='artifacts/dataset-content-v1')
    args = parser.parse_args()
    result = (prepare_config(args.workspace, args.output) if args.command == 'prepare' else
              run_phase(args.workspace, {'path': args.config, 'sha256': args.config_sha256}, args.output, args.phase))
    print(json.dumps(result, ensure_ascii=True))


if __name__ == '__main__':
    main()
