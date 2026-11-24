"""Prepare, build or verify P14.1b.4 pending OCR control-source packets."""
import argparse
import json
from pathlib import Path
from cad_recovery.ocr_control_review_v1 import (prepare_control_review_config_v1,
                                              run_control_review_v1, verify_control_review_v1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path.cwd())
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare')
    prepare.add_argument('--output', default='configs/reviews/ocr-control-review-v1.json')
    build = commands.add_parser('build')
    build.add_argument('--config', default='configs/reviews/ocr-control-review-v1.json')
    build.add_argument('--config-sha256', required=True)
    build.add_argument('--output', required=True)
    verify = commands.add_parser('verify')
    verify.add_argument('--manifest', required=True)
    verify.add_argument('--manifest-sha256', required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare_control_review_config_v1(args.workspace, args.output)
    elif args.command == 'build':
        result = run_control_review_v1(args.workspace, {'path': args.config, 'sha256': args.config_sha256}, args.output)['summary']
    else:
        result = verify_control_review_v1(args.workspace, {'path': args.manifest, 'sha256': args.manifest_sha256})['summary']
    print(json.dumps(result, ensure_ascii=True))


if __name__ == '__main__':
    main()
