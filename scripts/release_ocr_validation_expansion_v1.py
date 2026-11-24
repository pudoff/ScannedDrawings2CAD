"""Freeze/build/replay P16.2a.3 validation labels; no OCR or corpus training."""
import argparse
import json
from pathlib import Path
from cad_recovery.ocr_validation_expansion_v1 import prepare_config,build_release,verify_release


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace',type=Path,default=Path.cwd())
    sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('freeze');a.add_argument('--output',default='configs/reviews/ocr-validation-expansion-release-v1.json')
    a=sub.add_parser('build');a.add_argument('--config',default='configs/reviews/ocr-validation-expansion-release-v1.json')
    a.add_argument('--config-sha256',required=True);a.add_argument('--output',required=True)
    a=sub.add_parser('verify');a.add_argument('--manifest',required=True);a.add_argument('--manifest-sha256',required=True)
    a=p.parse_args()
    if a.command=='freeze':result=prepare_config(a.workspace,a.output)
    elif a.command=='build':result=build_release(a.workspace,{'path':a.config,'sha256':a.config_sha256},a.output)
    else:result=verify_release(a.workspace,{'path':a.manifest,'sha256':a.manifest_sha256})[0]
    print(json.dumps(result,ensure_ascii=True))


if __name__=='__main__':main()
