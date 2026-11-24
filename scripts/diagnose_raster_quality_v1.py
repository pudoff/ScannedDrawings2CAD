"""Freeze/build/verify development pixel-quality diagnostics without inference."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    f = commands.add_parser('freeze'); f.add_argument('--output', required=True)
    b = commands.add_parser('build'); b.add_argument('--config', required=True); b.add_argument('--config-sha256', required=True); b.add_argument('--output', required=True)
    v = commands.add_parser('verify'); v.add_argument('--manifest', required=True); v.add_argument('--manifest-sha256', required=True)
    args = parser.parse_args(); root = Path.cwd(); os.environ.setdefault('XDG_CACHE_HOME', str(root/'artifacts/.cache'))
    from cad_recovery.raster_quality_review_v1 import freeze, build, verify
    if args.command == 'freeze': result = freeze(root, args.output)
    elif args.command == 'build': result = build(root, {'path': args.config, 'sha256': args.config_sha256}, args.output)
    else: result = verify(root, {'path': args.manifest, 'sha256': args.manifest_sha256})
    print(json.dumps(result, ensure_ascii=True))


if __name__ == '__main__': main()
