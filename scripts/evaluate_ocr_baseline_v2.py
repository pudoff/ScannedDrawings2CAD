"""Prepare, run or verify the 90-line P16.1b OCR diagnostic."""
import argparse
import json
from pathlib import Path
from cad_recovery.ocr_baseline_v2 import prepare_config_v2, run_baseline_v2, verify_baseline_v2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path.cwd())
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare'); prepare.add_argument('--output', default='configs/reviews/ocr-baseline-v2.json')
    run = commands.add_parser('run'); run.add_argument('--config', default='configs/reviews/ocr-baseline-v2.json')
    run.add_argument('--config-sha256', required=True); run.add_argument('--output', required=True)
    verify = commands.add_parser('verify'); verify.add_argument('--manifest', required=True); verify.add_argument('--manifest-sha256', required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare_config_v2(args.workspace, args.output)
    else:
        report = run_baseline_v2(args.workspace, {'path': args.config, 'sha256': args.config_sha256}, args.output) if args.command == 'run' else verify_baseline_v2(args.workspace, {'path': args.manifest, 'sha256': args.manifest_sha256})
        result = {'sample_count': report['sample_count'], 'status_counts': report['status_counts'],
            **{mode: {key: report[mode][key] for key in ('reference_characters', 'distance', 'cer', 'exact_matches')} for mode in ('raw', 'normalized')},
            'inherited_changed_count': report['regression']['changed_count']}
    print(json.dumps(result, ensure_ascii=True))


if __name__ == '__main__':
    main()
