"""Freeze or prepare current-layout drawings without approving labels."""
import argparse
import json
from pathlib import Path
from cad_recovery.dataset_preparation_v1 import prepare_config, prepare_current, verify_preparation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, default=Path.cwd())
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('prepare-config'); a.add_argument('--output', default='configs/reviews/current-preparation-v1.json')
    a = sub.add_parser('run'); a.add_argument('--config', default='configs/reviews/current-preparation-v1.json')
    a.add_argument('--config-sha256', required=True); a.add_argument('--output', required=True)
    a = sub.add_parser('verify'); a.add_argument('--manifest', required=True); a.add_argument('--manifest-sha256', required=True)
    args = p.parse_args()
    if args.command == 'prepare-config': result = prepare_config(args.workspace, args.output)
    elif args.command == 'run': result = prepare_current(args.workspace, {'path': args.config, 'sha256': args.config_sha256}, args.output)
    else:
        rows, _ = verify_preparation(args.workspace, {'path': args.manifest, 'sha256': args.manifest_sha256})
        result = {'count': len(rows)}
    print(json.dumps(result, ensure_ascii=True))


if __name__ == '__main__': main()
