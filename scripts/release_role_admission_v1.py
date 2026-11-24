"""Build/verify additional text and categorical role labels without training."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest='command', required=True)
    create = sub.add_parser('build')
    create.add_argument('--config', default='configs/reviews/role-admission-v1-final.json')
    create.add_argument('--config-sha256', required=True)
    create.add_argument('--output', required=True)
    check = sub.add_parser('verify')
    check.add_argument('--manifest', required=True)
    check.add_argument('--manifest-sha256', required=True)
    args = parser.parse_args()
    os.environ.setdefault('XDG_CACHE_HOME', str(args.workspace/'artifacts/.cache'))
    from cad_recovery.role_admission_release_v1 import build, verify
    if args.command == 'build':
        result = build(args.workspace, {'path': args.config, 'sha256': args.config_sha256}, args.output)
    else:
        result = verify(args.workspace, {'path': args.manifest, 'sha256': args.manifest_sha256})[0]
    print(json.dumps(result, ensure_ascii=True))


if __name__ == '__main__':
    main()
