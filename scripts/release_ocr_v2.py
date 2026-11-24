"""Build/verify the combined reviewed OCR development release."""
import argparse
import json
from pathlib import Path
from cad_recovery.ocr_release_v2 import run_ocr_release_v2, load_ocr_training_v2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path.cwd())
    subs = parser.add_subparsers(dest='command', required=True)
    build = subs.add_parser('build')
    build.add_argument('--config', default='configs/reviews/ocr-release-v2.json')
    build.add_argument('--config-sha256', required=True)
    build.add_argument('--output', required=True)
    verify = subs.add_parser('verify')
    verify.add_argument('--manifest', required=True)
    verify.add_argument('--manifest-sha256', required=True)
    args = parser.parse_args()
    if args.command == 'build':
        result = run_ocr_release_v2(args.workspace, {'path': args.config, 'sha256': args.config_sha256}, args.output)
        result = {k: result[k] for k in ('status', 'annotation_count', 'sample_count', 'task_split_counts', 'training_started')}
    else:
        result = load_ocr_training_v2(args.workspace, {'path': args.manifest, 'sha256': args.manifest_sha256})
        result = {k: result[k] for k in ('count', 'task', 'split', 'training_started')}
    print(json.dumps(result, ensure_ascii=True))


if __name__ == '__main__':
    main()
