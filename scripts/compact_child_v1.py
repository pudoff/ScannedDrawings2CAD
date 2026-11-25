"""Activate explicit metadata adapters, then reuse the exact sealed operations."""
import argparse,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
base=Path(sys.base_prefix).resolve()
sys.path=[p for p in sys.path if p and Path(p).resolve().is_relative_to(base) and 'site-packages' not in p]
os.environ['XDG_CACHE_HOME']=str(ROOT/'artifacts/.cache')
for name in ('src','scripts','runtime/algorithm-deps','artifacts/service-runtime-v1/deps','artifacts/cv-skeleton-runtime-v1/deps'):
    sys.path.insert(0,str(ROOT/name))
from cad_recovery.compact_runtime_v1 import activate,binding
activate()
from cad_recovery.sealed_child_v3 import child_main
from cad_recovery.runtime_io_v1 import atomic_json


def operation(request,progress):
    folder=Path(request['folder']) if 'folder' in request else Path(request['output']).parent
    atomic_json(folder/'compact-runtime.json',binding())
    if 'stage' in request:
        from build_web_correction_v2 import operation as execute
    elif 'folder' in request:
        from cad_recovery.table_build_dispatch_v4 import build_table as execute
    else:
        if request['options']['profile']=='native_cad_review_demo':
            from cad_recovery.profile_contract_v1 import ProfileError
            raise ProfileError('profile_contract_changed','Prepared example requires the research runtime')
        from service_profile_diagnostic_v7 import execute
    execute(request,progress)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--request',type=Path,required=True)
    child_main(parser.parse_args().request,operation)
