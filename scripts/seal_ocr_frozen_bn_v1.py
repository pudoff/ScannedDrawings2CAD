"""Seal/verify the next validation-only experiment; never starts training."""
import argparse,json
from pathlib import Path
from cad_recovery.ocr_frozen_bn_experiment_v1 import seal,verify_protocol


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace',type=Path,default=Path.cwd());s=p.add_subparsers(dest='command',required=True)
    a=s.add_parser('seal');a.add_argument('--diagnosis',required=True);a.add_argument('--diagnosis-sha256',required=True)
    a.add_argument('--output',default='configs/reviews/ocr-frozen-bn-experiment-v1.json')
    a=s.add_parser('verify');a.add_argument('--config',required=True);a.add_argument('--config-sha256',required=True)
    a=p.parse_args()
    if a.command=='seal':r=seal(a.workspace,{'path':a.diagnosis,'sha256':a.diagnosis_sha256},a.output)
    else:r=verify_protocol(a.workspace,{'path':a.config,'sha256':a.config_sha256})
    print(json.dumps(r,ensure_ascii=True))


if __name__=='__main__':main()
