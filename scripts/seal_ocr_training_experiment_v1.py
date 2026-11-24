"""Seal/check first OCR training protocol; this CLI never launches training."""
import argparse,json
from pathlib import Path
from cad_recovery.ocr_training_experiment_v1 import seal,verify_experiment


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace',type=Path,default=Path.cwd());s=p.add_subparsers(dest='command',required=True)
 a=s.add_parser('seal');a.add_argument('--baseline-manifest',required=True);a.add_argument('--baseline-sha256',required=True);a.add_argument('--output',default='configs/reviews/ocr-first-training-experiment-v1.json')
 a=s.add_parser('verify');a.add_argument('--experiment',required=True);a.add_argument('--experiment-sha256',required=True)
 a=p.parse_args()
 if a.command=='seal':r=seal(a.workspace,{'path':a.baseline_manifest,'sha256':a.baseline_sha256},a.output)
 else:r=verify_experiment(a.workspace,{'path':a.experiment,'sha256':a.experiment_sha256})
 print(json.dumps(r,ensure_ascii=True))


if __name__=='__main__':main()
