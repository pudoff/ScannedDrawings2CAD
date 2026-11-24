"""Dedicated child: retained V14/V15 or explicit safe text extension."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
for name in ('src','scripts','artifacts/service-runtime-v1/deps','artifacts/cv-skeleton-runtime-v1/deps'):
    sys.path.insert(0,str(ROOT/name))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--request',type=Path,required=True);a=p.parse_args()
    from cad_recovery.dataset_layout_v1 import read,encoded
    from cad_recovery.runtime_progress_v1 import Progress
    from cad_recovery.web_tables_contract_v1 import binding as first
    from cad_recovery.web_tables_contract_v2 import binding as second
    request=read(a.request);progress=Progress(a.request.parent/'runtime-progress.json')
    try:
        with progress.phase('table_text_choice' if request['edit'].get('text_selection') else 'table_build'):
            sha=request['table_binding']['sha256']
            if sha==first()['sha256']:
                from cad_recovery.web_tables_build_v1 import build
                build(request)
            elif sha==second()['sha256']:
                from cad_recovery.web_tables_build_v2 import build
                build(request)
            else:
                from cad_recovery.web_tables_build_v3 import build
                build(request,progress=progress)
        progress.finish('completed');(a.request.parent/'status.json').write_bytes(encoded({'ok':True}))
    except BaseException as error:
        progress.finish('failed',error)
        (a.request.parent/'status.json').write_bytes(encoded(dict(ok=False,code='table_processing_failed')))
        raise
