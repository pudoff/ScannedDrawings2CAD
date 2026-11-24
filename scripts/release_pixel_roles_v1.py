"""Freeze, build and verify reviewed partial-pixel experiment labels."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from cad_recovery import pixel_role_release_v1 as release


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['freeze','build','verify'])
    parser.add_argument('--reading')
    parser.add_argument('--config')
    parser.add_argument('--manifest')
    parser.add_argument('--sha256')
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.command == 'freeze':
        result = release.freeze(ROOT, args.reading, ROOT/args.output)
    elif args.command == 'build':
        result = release.build(ROOT, {'path': args.config, 'sha256': args.sha256}, args.output)
    else:
        report, rows = release.verify(ROOT, {'path': args.manifest, 'sha256': args.sha256})
        result = {'verified': True, 'patches': len(rows), 'admission': report['admission']}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
