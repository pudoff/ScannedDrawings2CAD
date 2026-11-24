"""CLI using the same dispatcher and immutable profile bindings as worker v2."""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cad_recovery.profile_contract_v9 import ProcessingOptions,DEFINITIONS,resolve_profile,ProfileError

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True);sub.add_parser('profiles')
    run=sub.add_parser('run');run.add_argument('input',type=Path);run.add_argument('--output',type=Path,required=True)
    run.add_argument('--profile',choices=DEFINITIONS,required=True);run.add_argument('--render-dpi',type=float,default=120.)
    run.add_argument('--image-dpi',type=float);run.add_argument('--rotation-deg',type=int,choices=(0,90,180,270),default=0)
    run.add_argument('--ocr-models',type=Path);run.add_argument('--expected-profile-sha256');a=p.parse_args()
    os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'artifacts/.cache'))
    try:
        if a.command=='profiles':print(json.dumps({'profiles':[resolve_profile(n) for n in DEFINITIONS]}));return 0
        binding=resolve_profile(a.profile)
        if a.expected_profile_sha256 and binding['sha256']!=a.expected_profile_sha256:raise ProfileError('profile_contract_changed','Unexpected profile binding')
        from cad_recovery.profile_pipeline_v9 import run_profile
        options=ProcessingOptions(profile=a.profile,render_dpi=a.render_dpi,image_dpi=a.image_dpi,rotation_deg=a.rotation_deg)
        report=run_profile(a.input,a.output,options,models=a.ocr_models,expected_binding=binding)
        print(json.dumps({'status':report['status'],'profile':a.profile,'profile_sha256':binding['sha256'],'page_count':report['page_count']},ensure_ascii=False));return 0
    except Exception as error:
        print(json.dumps({'status':'failed','code':getattr(error,'code','input_invalid'),'message':str(error)},ensure_ascii=False),file=sys.stderr);return 1

if __name__=='__main__':raise SystemExit(main())
