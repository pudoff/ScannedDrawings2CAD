"""Qualify selected OCR weights, evaluate final test once, or verify offline."""
import argparse,json
from pathlib import Path
from cad_recovery.ocr_first_training_v1 import qualify,evaluate_test,verify_all
from cad_recovery.dataset_layout_v1 import encoded,pin


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace',type=Path,default=Path.cwd());s=p.add_subparsers(dest='command',required=True)
    a=s.add_parser('qualify');a.add_argument('--training',default='artifacts/ocr-first-training-v1');a.add_argument('--output',required=True)
    a=s.add_parser('test');a.add_argument('--qualification',required=True);a.add_argument('--qualification-sha256',required=True);a.add_argument('--output',required=True)
    a=s.add_parser('verify');a.add_argument('--qualification',required=True);a.add_argument('--qualification-sha256',required=True)
    a.add_argument('--test-manifest',required=True);a.add_argument('--test-sha256',required=True);a.add_argument('--output',required=True)
    a=p.parse_args()
    if a.command=='qualify':result=qualify(a.workspace,a.training,a.output)
    elif a.command=='test':result=evaluate_test(a.workspace,{'path':a.qualification,'sha256':a.qualification_sha256},a.output)
    else:
        result=verify_all(a.workspace,{'path':a.qualification,'sha256':a.qualification_sha256},{'path':a.test_manifest,'sha256':a.test_sha256})
        target=a.workspace/a.output
        if target.exists():raise ValueError('Choose a new verification output')
        target.write_bytes(encoded(result));result={'proof':pin(a.workspace,a.output),'selected_epoch':result['training_replay']['selected_epoch']}
    print(json.dumps(result,ensure_ascii=True))


if __name__=='__main__':main()
