"""CLI using the same bounded-input pipeline as the new Web worker."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cad_recovery.profile_pipeline_v6 import run_profile


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--profile',default='native_cad_layers_draft');p.add_argument('--models',type=Path,default=ROOT/'artifacts/models/ppocr-v5')
    p.add_argument('--rotation',type=int,default=0);a=p.parse_args()
    result=run_profile(a.source,a.output,dict(profile=a.profile,render_dpi=120.,image_dpi=None,rotation_deg=a.rotation),models=a.models)
    print('Completed:',a.output,'; original source:',result['source_sha256'])
