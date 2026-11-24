"""Internal v2 child; request integrity and binding are checked before analysis."""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cad_recovery.profile_contract_v2 import ProfileError
from cad_recovery.profile_pipeline_v2 import run_profile
from cad_recovery.service_v1.storage import digest,write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--request',type=Path,required=True);a=p.parse_args()
    try:
        if a.request.stat().st_size>1024**2:raise ValueError('Request exceeds 1MiB')
        r=json.loads(a.request.read_text('utf-8'))
        if set(r)!={'source','input_sha256','output','options','profile_binding'}:raise ProfileError('profile_contract_changed','Invalid request envelope')
        if digest(Path(r['source']))!=r['input_sha256']:raise ProfileError('input_integrity','Source hash changed')
        run_profile(r['source'],r['output'],r['options'],models=os.environ.get('CAD_DEMO_OCR_MODELS'),expected_binding=r['profile_binding'])
    except Exception as error:
        write_json(a.request.parent/'status.json',{'ok':False,'code':getattr(error,'code','input_invalid'),'message':str(error)[:1000]});raise
    write_json(a.request.parent/'status.json',{'ok':True})

if __name__=='__main__':main()
