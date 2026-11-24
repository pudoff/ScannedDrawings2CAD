"""Qualify/verify frozen-BN training; no test evaluation command exists."""
import argparse,json
from pathlib import Path
from cad_recovery.ocr_frozen_bn_run_v1 import qualify,verify
from cad_recovery.dataset_layout_v1 import encoded,pin


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace',type=Path,default=Path.cwd());s=p.add_subparsers(dest='command',required=True)
    a=s.add_parser('qualify');a.add_argument('--training',default='artifacts/ocr-frozen-bn-training-v1');a.add_argument('--output',required=True)
    a=s.add_parser('verify');a.add_argument('--manifest',required=True);a.add_argument('--manifest-sha256',required=True);a.add_argument('--output',required=True)
    a=p.parse_args()
    if a.command=='qualify':r=qualify(a.workspace,a.training,a.output)
    else:
        report=verify(a.workspace,{'path':a.manifest,'sha256':a.manifest_sha256});target=a.workspace/a.output
        if target.exists():raise ValueError('Choose a new verification output')
        target.write_bytes(encoded(report));r={'proof':pin(a.workspace,a.output),'selected_epoch':report['training_replay']['selected_epoch'],'test_model_calls':0}
    print(json.dumps(r,ensure_ascii=True))


if __name__=='__main__':main()
