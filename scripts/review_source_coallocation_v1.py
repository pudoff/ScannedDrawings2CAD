"""Build or verify an immutable, assistant-reviewed source coallocation overlay."""
import argparse
import json
import os
from pathlib import Path

from cad_recovery.source_coallocation_v1 import build, load_overlay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest='command', required=True)
    create = sub.add_parser('build')
    create.add_argument('--config', default='configs/reviews/source-coallocation-v1.json')
    create.add_argument('--config-sha256', required=True)
    create.add_argument('--output', required=True)
    verify = sub.add_parser('verify')
    verify.add_argument('--manifest', required=True)
    verify.add_argument('--manifest-sha256', required=True)
    args = parser.parse_args()
    os.environ.setdefault('XDG_CACHE_HOME', str(args.workspace/'artifacts/.cache'))
    if args.command == 'build':
        result = build(args.workspace, {'path': args.config, 'sha256': args.config_sha256}, args.output)
    else:
        overlay = load_overlay(args.workspace, {'path': args.manifest, 'sha256': args.manifest_sha256})
        result = {'verified': True, 'groups': len(overlay['groups']), 'new_constraints': overlay['new_constraint_count']}
    print(json.dumps(result, ensure_ascii=True))


if __name__ == '__main__':
    main()
