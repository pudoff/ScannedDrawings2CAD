"""Assemble or replay the current dataset content catalog."""
import argparse
import json
from pathlib import Path
from cad_recovery.dataset_content_review_v1 import assemble, verify_review


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, default=Path.cwd())
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('build')
    a.add_argument('--config-sha256', required=True)
    a.add_argument('--config', default='configs/reviews/dataset-content-v1.json')
    a.add_argument('--phase-dir', default='artifacts/dataset-content-v1')
    a.add_argument('--output', required=True)
    v = sub.add_parser('verify')
    v.add_argument('--manifest', required=True)
    v.add_argument('--manifest-sha256', required=True)
    args = p.parse_args()
    result = (assemble(args.workspace, {'path': args.config, 'sha256': args.config_sha256}, args.phase_dir, args.output)
              if args.command == 'build' else verify_review(args.workspace, {'path': args.manifest, 'sha256': args.manifest_sha256}))
    print(json.dumps(result, ensure_ascii=True))


if __name__ == '__main__':
    main()
