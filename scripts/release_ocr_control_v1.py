"""Prepare, build or verify the complete OCR registry with grouped controls."""
import argparse
import json
from pathlib import Path

from cad_recovery.ocr_control_release_v1 import prepare_control_config_v1, run_control_release_v1, load_ocr_control_v1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path.cwd())
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare')
    prepare.add_argument('--protocol', default='configs/reviews/ocr-control-lines-protocol-v1.json')
    prepare.add_argument('--protocol-sha256', required=True)
    prepare.add_argument('--output', default='configs/reviews/ocr-control-release-v1.json')
    build = commands.add_parser('build')
    build.add_argument('--config', default='configs/reviews/ocr-control-release-v1.json')
    build.add_argument('--config-sha256', required=True)
    build.add_argument('--output', required=True)
    verify = commands.add_parser('verify')
    verify.add_argument('--manifest', required=True)
    verify.add_argument('--manifest-sha256', required=True)
    verify.add_argument('--split', required=True, choices=('train', 'validation', 'test'))
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare_control_config_v1(args.workspace, {'path': args.protocol, 'sha256': args.protocol_sha256}, args.output)
    elif args.command == 'build':
        result = run_control_release_v1(args.workspace, {'path': args.config, 'sha256': args.config_sha256}, args.output)
    else:
        loaded = load_ocr_control_v1(args.workspace, {'path': args.manifest, 'sha256': args.manifest_sha256}, split=args.split)
        result = {key: loaded[key] for key in ('task', 'split', 'count', 'scope', 'independent_evaluation')}
    print(json.dumps(result, ensure_ascii=True))


if __name__ == '__main__':
    main()
