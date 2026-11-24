"""Explicit shared profiles; historical cad-recovery commands retain their scope."""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cad_recovery.profile_contract_v1 import DEFINITIONS,ProcessingOptions,ProfileError,resolve_profile


def main(argv=None):
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser(description=__doc__)
    commands=p.add_subparsers(dest='command',required=True)
    commands.add_parser('profiles',help='Print pinned profiles, models, stages and limits')
    run=commands.add_parser('run',help='Process with an explicitly selected shared profile')
    run.add_argument('input',type=Path);run.add_argument('--output',type=Path,required=True)
    run.add_argument('--profile',choices=DEFINITIONS,required=True)
    run.add_argument('--render-dpi',type=float,default=120.)
    run.add_argument('--image-dpi',type=float);run.add_argument('--rotation-deg',type=int,choices=(0,90,180,270),default=0)
    run.add_argument('--ocr-models',type=Path);run.add_argument('--expected-profile-sha256')
    args=p.parse_args(argv)
    os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'artifacts/.cache'))
    try:
        if args.command=='profiles':
            print(json.dumps({'profiles':[resolve_profile(name) for name in DEFINITIONS]}));return 0
        options=ProcessingOptions(profile=args.profile,render_dpi=args.render_dpi,
            image_dpi=args.image_dpi,rotation_deg=args.rotation_deg)
        binding=resolve_profile(args.profile)
        if args.expected_profile_sha256 and binding['sha256']!=args.expected_profile_sha256:
            raise ProfileError('profile_contract_changed','Selected profile SHA differs from the requested one')
        from cad_recovery.profile_pipeline_v1 import run_profile
        report=run_profile(args.input,args.output,options,models=args.ocr_models,expected_binding=binding)
        print(json.dumps({'status':report['status'],'output':str(args.output),'profile':args.profile,
            'profile_sha256':binding['sha256'],'page_count':report['page_count']},ensure_ascii=False))
        return 0
    except Exception as error:
        print(json.dumps({'status':'failed','code':getattr(error,'code','input_invalid'),
                          'message':str(error)},ensure_ascii=False),file=sys.stderr)
        return 1


if __name__=='__main__':raise SystemExit(main())
