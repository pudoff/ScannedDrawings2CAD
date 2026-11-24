"""Freeze, run or verify P16.1c validation/test OCR diagnostics."""
import argparse
import json
from pathlib import Path

from cad_recovery.ocr_baseline_v3 import prepare_config_v3, run_baseline_v3, verify_baseline_v3


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--workspace',type=Path,default=Path.cwd())
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--output',default='configs/reviews/ocr-baseline-v3.json')
    p=sub.add_parser('run');p.add_argument('--config',default='configs/reviews/ocr-baseline-v3.json');p.add_argument('--config-sha256',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('verify');p.add_argument('--manifest',required=True);p.add_argument('--manifest-sha256',required=True)
    args=parser.parse_args()
    if args.command=='prepare':result=prepare_config_v3(args.workspace,args.output)
    else:
        report=(run_baseline_v3(args.workspace,{'path':args.config,'sha256':args.config_sha256},args.output)
            if args.command=='run' else verify_baseline_v3(args.workspace,{'path':args.manifest,'sha256':args.manifest_sha256}))
        result={split:{'sample_count':part['overall']['sample_count'],'status_counts':part['overall']['status_counts'],
            **{mode:{k:part['overall'][mode][k] for k in ('exact_matches','reference_characters','distance','cer')}
                for mode in ('raw','normalized')}} for split,part in report['by_split'].items()}
    print(json.dumps(result,ensure_ascii=True))


if __name__=='__main__':main()
