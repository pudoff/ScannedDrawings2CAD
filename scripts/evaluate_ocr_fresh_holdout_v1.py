"""Prepare, execute once, or verify the fresh holdout evaluation offline."""
import argparse
from pathlib import Path
from cad_recovery.dataset_layout_v1 import encoded,pin
from cad_recovery.ocr_fresh_evaluation_v1 import prepare_run,evaluate,verify_evaluation

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('prepare','evaluate','verify'))
    p.add_argument('config_or_manifest');p.add_argument('--output');a=p.parse_args();root=Path.cwd()
    if a.mode=='prepare':
        if not a.output:p.error('--output required for prepare')
        result=prepare_run(root,a.config_or_manifest,a.output)
    elif a.mode=='evaluate':result=evaluate(root,pin(root,a.config_or_manifest))
    else:
        r=verify_evaluation(root,pin(root,a.config_or_manifest))
        result={k:r[k] for k in ('task','actual_model_calls','metrics','timing','old_test_model_calls','independent_evaluation')}
    print(encoded(result).decode('ascii'))

if __name__=='__main__':main()
