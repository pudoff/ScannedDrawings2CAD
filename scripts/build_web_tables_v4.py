"""Same retained table producers; common phase, status and failure persistence."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
for name in ('src','scripts','artifacts/service-runtime-v1/deps','artifacts/cv-skeleton-runtime-v1/deps'):
    sys.path.insert(0,str(ROOT/name))
from cad_recovery.runtime_worker_v1 import child_main


def execute(request, progress):
    from cad_recovery.web_tables_contract_v1 import binding as first
    from cad_recovery.web_tables_contract_v2 import binding as second
    with progress.phase('table_text_choice' if request['edit'].get('text_selection') else 'table_build'):
        sha=request['table_binding']['sha256']
        if sha == first()['sha256']:
            from cad_recovery.web_tables_build_v1 import build
            build(request)
        elif sha == second()['sha256']:
            from cad_recovery.web_tables_build_v2 import build
            build(request)
        else:
            from cad_recovery.web_tables_build_v3 import build
            build(request, progress=progress)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--request',type=Path,required=True)
    child_main(p.parse_args().request, execute)
