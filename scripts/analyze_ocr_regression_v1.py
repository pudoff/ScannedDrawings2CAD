"""Freeze/run/check the validation-only first-training regression diagnosis."""
import argparse,json
from pathlib import Path
from cad_recovery.ocr_regression_v1 import freeze,run,verify


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace',type=Path,default=Path.cwd());s=p.add_subparsers(dest='command',required=True)
    a=s.add_parser('freeze');a.add_argument('--output',default='configs/reviews/ocr-regression-probe-v1.json')
    a=s.add_parser('run');a.add_argument('--config',required=True);a.add_argument('--config-sha256',required=True);a.add_argument('--output',required=True)
    a=s.add_parser('verify');a.add_argument('--manifest',required=True);a.add_argument('--manifest-sha256',required=True)
    a=p.parse_args()
    if a.command=='freeze':r=freeze(a.workspace,a.output)
    elif a.command=='run':r=run(a.workspace,{'path':a.config,'sha256':a.config_sha256},a.output)
    else:
        report=verify(a.workspace,{'path':a.manifest,'sha256':a.manifest_sha256})
        r={'task':report['task'],'test_model_calls':0,'arms':{k:v['selection_score']['macro_raw_cer'] for k,v in report['bn_probe'].items()}}
    print(json.dumps(r,ensure_ascii=True))


if __name__=='__main__':main()
