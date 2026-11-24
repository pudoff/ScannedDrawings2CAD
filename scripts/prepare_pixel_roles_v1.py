"""Create/replay partial pixel-role review packets; never train a model."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT/'artifacts/.cache'))
from cad_recovery.pixel_role_draft_v1 import freeze, build, verify


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    f = commands.add_parser('freeze'); f.add_argument('--output', type=Path, required=True)
    b = commands.add_parser('build'); b.add_argument('--config', required=True); b.add_argument('--sha256', required=True); b.add_argument('--output', required=True)
    v = commands.add_parser('verify'); v.add_argument('--manifest', required=True); v.add_argument('--sha256', required=True)
    args = parser.parse_args()
    if args.action == 'freeze': result = freeze(ROOT, ROOT/args.output)
    elif args.action == 'build':
        result = build(ROOT, {'path': args.config, 'sha256': args.sha256}, args.output)
        result = {'manifest': result['manifest'], 'patches': result['report']['patch_rows'], 'training_eligible': False}
    else:
        summary, rows = verify(ROOT, {'path': args.manifest, 'sha256': args.sha256})
        result = {'verified': True, 'patches': len(rows), 'training_eligible': False, 'models_executed': False}
    print(json.dumps(result))
