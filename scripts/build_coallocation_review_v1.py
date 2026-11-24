"""Prepare, build or independently replay the additive P14.1b.5 overlay."""
import argparse
import json
from pathlib import Path

from cad_recovery.coallocation_review_v1 import (prepare_coallocation_config_v1,
                                               run_coallocation_review_v1, verify_coallocation_review_v1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path.cwd())
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare')
    prepare.add_argument('--output', default='configs/reviews/coallocation-review-v1.json')
    build = commands.add_parser('build')
    build.add_argument('--config', default='configs/reviews/coallocation-review-v1.json')
    build.add_argument('--config-sha256', required=True)
    build.add_argument('--output', required=True)
    verify = commands.add_parser('verify')
    verify.add_argument('--manifest', required=True)
    verify.add_argument('--manifest-sha256', required=True)
    args = parser.parse_args()

    def progress(index, total, path):
        if index == 1 or index % 10 == 0 or index == total:
            print(json.dumps({'source': index, 'total': total, 'path': path}, ensure_ascii=True), flush=True)

    if args.command == 'prepare':
        result = prepare_coallocation_config_v1(args.workspace, args.output)
    elif args.command == 'build':
        result = run_coallocation_review_v1(args.workspace, {'path': args.config, 'sha256': args.config_sha256}, args.output, progress=progress)
    else:
        result = verify_coallocation_review_v1(args.workspace, {'path': args.manifest, 'sha256': args.manifest_sha256}, progress=progress)
    print(json.dumps(result, ensure_ascii=True))


if __name__ == '__main__':
    main()
