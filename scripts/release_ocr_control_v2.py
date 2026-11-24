"""Prepare, build or replay the limited title OCR release with explicit splits."""
import argparse
import json
from pathlib import Path

from cad_recovery.ocr_control_release_v2 import prepare_control_config_v2, run_control_release_v2, load_ocr_control_v2


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--workspace',type=Path,default=Path.cwd())
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--output',default='configs/reviews/ocr-control-release-v2.json')
    p=sub.add_parser('build');p.add_argument('--config',default='configs/reviews/ocr-control-release-v2.json');p.add_argument('--config-sha256',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('verify');p.add_argument('--manifest',required=True);p.add_argument('--manifest-sha256',required=True);p.add_argument('--split',required=True,choices=['train','validation','test'])
    args=parser.parse_args()
    if args.command=='prepare':result=prepare_control_config_v2(args.workspace,args.output)
    elif args.command=='build':result=run_control_release_v2(args.workspace,{'path':args.config,'sha256':args.config_sha256},args.output)
    else:
        loaded=load_ocr_control_v2(args.workspace,{'path':args.manifest,'sha256':args.manifest_sha256},split=args.split)
        result={k:loaded[k] for k in ('task','split','count','scope','independent_evaluation')}
    print(json.dumps(result,ensure_ascii=True))


if __name__=='__main__':main()
