"""Freeze/run/verify isolated CPU Paddle smoke; full dataset training is separate."""
import argparse
import json
from pathlib import Path
from cad_recovery.ocr_training_runtime_v1 import freeze_runtime, run_runtime, verify_runtime


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--workspace', type=Path, default=Path.cwd())
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('freeze'); a.add_argument('--output', default='configs/reviews/ocr-training-runtime-v1.json')
    a = sub.add_parser('run'); a.add_argument('--config', default='configs/reviews/ocr-training-runtime-v1.json')
    a.add_argument('--config-sha256', required=True); a.add_argument('--output', required=True)
    a = sub.add_parser('verify'); a.add_argument('--manifest', required=True); a.add_argument('--manifest-sha256', required=True)
    args = p.parse_args()
    if args.command == 'freeze': result = freeze_runtime(args.workspace, args.output)
    elif args.command == 'run': result = run_runtime(args.workspace, {'path': args.config, 'sha256': args.config_sha256}, args.output)
    else: result = verify_runtime(args.workspace, {'path': args.manifest, 'sha256': args.manifest_sha256})
    print(json.dumps(result,ensure_ascii=True))


if __name__ == '__main__': main()
