"""Register or verify the explicit train/validation/test directory layout."""
import argparse
import json
from pathlib import Path

from cad_recovery.dataset_layout_v1 import build_layout_snapshot, verify_layout_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path.cwd())
    commands = parser.add_subparsers(dest='command', required=True)
    build = commands.add_parser('build')
    build.add_argument('--config', default='configs/dataset-layout-v1.json')
    build.add_argument('--output', required=True)
    verify = commands.add_parser('verify')
    verify.add_argument('--manifest', required=True)
    verify.add_argument('--manifest-sha256', required=True)
    args = parser.parse_args()
    result = (build_layout_snapshot(args.workspace, args.config, args.output) if args.command == 'build'
              else verify_layout_snapshot(args.workspace, {'path': args.manifest, 'sha256': args.manifest_sha256}))
    print(json.dumps(result, ensure_ascii=True))


if __name__ == '__main__':
    main()
