"""Build or verify a pinned source-only Rib OCR release."""
import argparse
import json
from pathlib import Path

from cad_recovery.rib_ocr_release_v1 import run_rib_ocr_release_v1, load_rib_ocr_training_v1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path.cwd())
    commands = parser.add_subparsers(dest='command', required=True)
    build = commands.add_parser('build')
    build.add_argument('--config', default='configs/reviews/rib-ocr-release-v1.json')
    build.add_argument('--config-sha256', required=True)
    build.add_argument('--output', required=True)
    verify = commands.add_parser('verify')
    verify.add_argument('--manifest', required=True)
    verify.add_argument('--manifest-sha256', required=True)
    args = parser.parse_args()
    if args.command == 'build':
        report = run_rib_ocr_release_v1(args.workspace, {'path': args.config, 'sha256': args.config_sha256}, args.output)
        result = {key: report[key] for key in ('status', 'sample_count', 'task_split_counts', 'training_started')}
    else:
        report = load_rib_ocr_training_v1(args.workspace, {'path': args.manifest, 'sha256': args.manifest_sha256})
        result = {key: report[key] for key in ('task', 'split', 'count', 'training_started')}
    print(json.dumps(result, ensure_ascii=True))


if __name__ == '__main__':
    main()
