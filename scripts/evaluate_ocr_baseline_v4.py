"""Freeze, run or replay P16.1d on the current OCR release."""
import argparse
import json
from pathlib import Path
from cad_recovery.ocr_baseline_v4 import prepare_config_v4, run_baseline_v4, verify_baseline_v4


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--workspace', type=Path, default=Path.cwd())
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('prepare'); a.add_argument('--output', default='configs/reviews/ocr-baseline-v4.json')
    a = sub.add_parser('run'); a.add_argument('--config', default='configs/reviews/ocr-baseline-v4.json')
    a.add_argument('--config-sha256', required=True); a.add_argument('--output', required=True)
    a = sub.add_parser('verify'); a.add_argument('--manifest', required=True); a.add_argument('--manifest-sha256', required=True)
    args = p.parse_args()
    if args.command == 'prepare': result = prepare_config_v4(args.workspace, args.output)
    else:
        report = (run_baseline_v4(args.workspace, {'path': args.config, 'sha256': args.config_sha256}, args.output)
                  if args.command == 'run' else verify_baseline_v4(args.workspace, {'path': args.manifest, 'sha256': args.manifest_sha256}))
        result = {s: {'sample_count': v['overall']['sample_count'], 'status_counts': v['overall']['status_counts'],
                      **{mode: {k: v['overall'][mode][k] for k in ('exact_matches', 'distance', 'reference_characters', 'cer')}
                         for mode in ('raw', 'normalized')}} for s, v in report['by_split'].items()}
    print(json.dumps(result, ensure_ascii=True))


if __name__ == '__main__': main()
