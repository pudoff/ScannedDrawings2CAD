"""Same frozen profile conversion with an honest document-level phase."""
import argparse
from pathlib import Path
import os
import sys
ROOT=Path(__file__).resolve().parents[1]
for name in ('src','scripts','artifacts/service-runtime-v1/deps','artifacts/cv-skeleton-runtime-v1/deps'):
    sys.path.insert(0,str(ROOT/name))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--request',type=Path,required=True);a=p.parse_args()
    from cad_recovery.dataset_layout_v1 import read,encoded,digest
    from cad_recovery.runtime_progress_v1 import Progress
    progress=Progress(a.request.parent/'runtime-progress.json')
    try:
        with progress.phase('document_conversion'):
            request=read(a.request)
            if digest(request['source'])!=request['input_sha256']:raise ValueError('Source changed')
            from cad_recovery.profile_pipeline_v6 import run_profile
            run_profile(request['source'],request['output'],request['options'],
                        models=os.environ.get('CAD_DEMO_OCR_MODELS'),expected_binding=request['profile_binding'])
        progress.finish('completed');(a.request.parent/'status.json').write_bytes(encoded({'ok':True}))
    except BaseException as error:
        progress.finish('failed',error)
        (a.request.parent/'status.json').write_bytes(encoded(dict(ok=False,code=getattr(error,'code','processing_failed'))))
        raise
