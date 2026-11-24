"""Prepare, build or authenticate the six-source development OCR extension."""
import argparse
import json
from pathlib import Path
from cad_recovery.ocr_release_v3 import prepare_ocr_config_v3, run_ocr_release_v3, load_ocr_training_v3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path.cwd())
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare'); prepare.add_argument('--output', default='configs/reviews/ocr-release-v3.json')
    build = commands.add_parser('build'); build.add_argument('--config', default='configs/reviews/ocr-release-v3.json')
    build.add_argument('--config-sha256', required=True); build.add_argument('--output', required=True)
    verify = commands.add_parser('verify'); verify.add_argument('--manifest', required=True); verify.add_argument('--manifest-sha256', required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare_ocr_config_v3(args.workspace, args.output)
    elif args.command == 'build':
        result = run_ocr_release_v3(args.workspace, {'path': args.config, 'sha256': args.config_sha256}, args.output)
    else:
        result = load_ocr_training_v3(args.workspace, {'path': args.manifest, 'sha256': args.manifest_sha256})['report']
    print(json.dumps(result, ensure_ascii=True))


if __name__ == '__main__':
    main()
