"""Internal child for jobs bound to the shared processing contract."""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cad_recovery.profile_contract_v1 import ProfileError,digest
from cad_recovery.profile_pipeline_v1 import run_profile
from cad_recovery.service_v1.storage import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--request',type=Path,required=True);args=p.parse_args()
    try:
        request=json.loads(args.request.read_text(encoding='utf-8'))
        if set(request)!={'source','input_sha256','output','options','profile_binding'}:
            raise ProfileError('profile_contract_changed','Incomplete worker profile envelope')
        if digest(request['source'])!=request['input_sha256']:
            raise ProfileError('input_integrity','Input digest differs from uploaded source')
        run_profile(request['source'],request['output'],request['options'],
            models=os.environ.get('CAD_DEMO_OCR_MODELS'),expected_binding=request['profile_binding'])
    except Exception as error:
        write_json(args.request.parent/'status.json',{'ok':False,'code':getattr(error,'code','input_invalid'),
            'type':type(error).__name__,'message':str(error)[:1000]})
        raise
    write_json(args.request.parent/'status.json',{'ok':True})


if __name__=='__main__':main()
