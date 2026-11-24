"""Prepare, run or verify the fixed P16.1a development OCR diagnostic."""
import argparse
import json
from pathlib import Path
from cad_recovery.ocr_baseline_v1 import prepare_config_v1, run_baseline_v1, verify_baseline_v1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path.cwd())
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare')
    prepare.add_argument('--output', default='configs/reviews/ocr-baseline-v1.json')
    run = commands.add_parser('run')
    run.add_argument('--config', default='configs/reviews/ocr-baseline-v1.json')
    run.add_argument('--config-sha256', required=True)
    run.add_argument('--output', required=True)
    verify = commands.add_parser('verify')
    verify.add_argument('--manifest', required=True)
    verify.add_argument('--manifest-sha256', required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare_config_v1(args.workspace, args.output)
    elif args.command == 'run':
        result = run_baseline_v1(args.workspace, {'path': args.config, 'sha256': args.config_sha256}, args.output)
    else:
        result = verify_baseline_v1(args.workspace, {'path': args.manifest, 'sha256': args.manifest_sha256})
    if args.command != 'prepare':
        result = {**{k: result[k] for k in ('sample_count', 'status_counts')},
                  **{mode: {k: result[mode][k] for k in ('reference_characters', 'distance', 'cer', 'exact_matches')}
                     for mode in ('raw', 'normalized')}}
    print(json.dumps(result, ensure_ascii=True))


if __name__ == '__main__':
    main()
