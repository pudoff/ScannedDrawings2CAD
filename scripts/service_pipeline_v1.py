"""Internal child: process one server-owned request, without reference targets."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cad_recovery.service_v1.config import ProcessingOptions
from cad_recovery.service_v1.storage import digest,write_json
from cad_recovery.pipeline import run_pipeline

def main():
    import json
    p=argparse.ArgumentParser();p.add_argument('--request',type=Path,required=True);a=p.parse_args()
    c=json.loads(a.request.read_text('utf-8'));options=ProcessingOptions.model_validate(c['options'])
    try:
        if digest(Path(c['source']))!=c['input_sha256']:raise ValueError('Input digest changed')
        run_pipeline(Path(c['source']),Path(c['output']),export=True,
            render_dpi=options.render_dpi,image_dpi=options.image_dpi,
            prepare=options.profile=='geometry_observations' or options.rotation_deg!=0,detect=options.profile=='geometry_observations',
            rotation_deg=options.rotation_deg,ocr=False,associate=False)
    except Exception as error:
        write_json(a.request.parent/'status.json',{'ok':False,'code':'input_invalid' if isinstance(error,ValueError) else 'processing_failed',
            'type':type(error).__name__,'message':str(error)[:1000]});raise
    write_json(a.request.parent/'status.json',{'ok':True})

if __name__=='__main__':main()
