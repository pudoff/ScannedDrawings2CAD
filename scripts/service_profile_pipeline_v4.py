"""Internal experimental child with the same integrity envelope as prior workers."""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cad_recovery.profile_contract_v4 import ProfileError
from cad_recovery.profile_pipeline_v4 import run_profile
from cad_recovery.service_v1.storage import digest,write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--request',type=Path,required=True)
    args=parser.parse_args()
    try:
        if args.request.stat().st_size>1024**2:
            raise ValueError('Request exceeds 1MiB')
        request=json.loads(args.request.read_text('utf-8'))
        if set(request)!={'source','input_sha256','output','options','profile_binding'}:
            raise ProfileError('profile_contract_changed','Invalid request envelope')
        if digest(Path(request['source']))!=request['input_sha256']:
            raise ProfileError('input_integrity','Source hash changed')
        run_profile(request['source'],request['output'],request['options'],
            models=os.environ.get('CAD_DEMO_OCR_MODELS'),expected_binding=request['profile_binding'])
    except Exception as error:
        write_json(args.request.parent/'status.json',{'ok':False,'code':getattr(error,'code','input_invalid'),'message':str(error)[:1000]})
        raise
    write_json(args.request.parent/'status.json',{'ok':True})


if __name__=='__main__':
    main()
