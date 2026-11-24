"""Build or verify the new OCR holdout without model inference."""
import argparse
from pathlib import Path
from cad_recovery.dataset_layout_v1 import encoded,pin
from cad_recovery.ocr_fresh_holdout_v1 import build_release,verify_release,check_admission

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('build','verify','admission'))
    p.add_argument('config_or_manifest');p.add_argument('--output');a=p.parse_args();root=Path.cwd()
    frozen=pin(root,a.config_or_manifest)
    if a.mode=='build':
        if not a.output:p.error('--output is required for build')
        result=build_release(root,frozen,a.output)
    elif a.mode=='verify':result=verify_release(root,frozen)[0]
    else:result=check_admission(root,frozen)
    print(encoded(result).decode('ascii'))

if __name__=='__main__':main()
