"""Unchanged engineering correction builder inside supervised compute+seal."""
from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parents[1]
for p in ('src','scripts','artifacts/service-runtime-v1/deps','artifacts/cv-skeleton-runtime-v1/deps'):
    sys.path.insert(0,str(ROOT/p))
from cad_recovery.sealed_child_v1 import child_main
from cad_recovery.runtime_io_v1 import atomic_json,read_json
from cad_recovery.web_correction_build_v1 import build


def operation(request,progress):
    with progress.phase('engineering_correction',stage=request['stage']):
        summary=build(request)
        folder=Path(request['folder'])
        snapshot=read_json(folder/'execution-budget.json')
        snapshot['scope']='pre-seal resource snapshot; final diagnostics remain execution-budget.json'
        atomic_json(folder/'output/resource-budget.json',snapshot)
        atomic_json(folder/'completion.json',summary)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--request',type=Path,required=True)
    child_main(p.parse_args().request,operation)
