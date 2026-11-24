"""Build/verify the expanded categorical role release without changing V1."""
import argparse
import json
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest='command', required=True)
    create = sub.add_parser('build')
    create.add_argument('--config', default='configs/reviews/semantic-roles-v2.json')
    create.add_argument('--config-sha256', required=True)
    create.add_argument('--output', required=True)
    verify = sub.add_parser('verify')
    verify.add_argument('--manifest', required=True)
    verify.add_argument('--manifest-sha256', required=True)
    args = parser.parse_args()
    os.environ.setdefault('XDG_CACHE_HOME', str(args.workspace/'artifacts/.cache'))
    from cad_recovery.semantic_role_release_v2 import build, load_training
    if args.command == 'build':
        result = build(args.workspace, {'path': args.config, 'sha256': args.config_sha256}, args.output)
    else:
        result = load_training(args.workspace, {'path': args.manifest, 'sha256': args.manifest_sha256})
        result.pop('records')
    print(json.dumps(result, ensure_ascii=True))


if __name__ == '__main__':
    main()
