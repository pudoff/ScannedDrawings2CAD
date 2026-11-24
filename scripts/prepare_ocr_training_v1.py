"""Prepare or verify train-only OCR inputs; check exits 2 while training is unready."""
import argparse
import json
from pathlib import Path
from cad_recovery.ocr_training_preparation_v1 import prepare_config, build_preparation, verify_preparation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, default=Path.cwd())
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('freeze'); a.add_argument('--output', default='configs/reviews/ocr-training-preparation-v1.json')
    a = sub.add_parser('build'); a.add_argument('--config', default='configs/reviews/ocr-training-preparation-v1.json')
    a.add_argument('--config-sha256', required=True); a.add_argument('--output', required=True)
    for command in ('verify', 'check'):
        a = sub.add_parser(command); a.add_argument('--manifest', required=True); a.add_argument('--manifest-sha256', required=True)
    args = p.parse_args()
    if args.command == 'freeze': result = prepare_config(args.workspace, args.output)
    elif args.command == 'build': result = build_preparation(args.workspace, {'path': args.config, 'sha256': args.config_sha256}, args.output)
    else: result = verify_preparation(args.workspace, {'path': args.manifest, 'sha256': args.manifest_sha256})
    print(json.dumps(result, ensure_ascii=True))
    return 2 if args.command == 'check' and not result['gate']['training_ready'] else 0


if __name__ == '__main__': raise SystemExit(main())
