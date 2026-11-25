"""Isolated compact checkout entry; no site/.pth or research paths are needed."""
import os,runpy,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
os.environ['PYTHONDONTWRITEBYTECODE']='1'
paths=[ROOT/'src',ROOT/'scripts',ROOT/'runtime/algorithm-deps',
       ROOT/'artifacts/service-runtime-v1/deps',ROOT/'artifacts/cv-skeleton-runtime-v1/deps']
for path in paths:
    if not path.is_dir():raise RuntimeError('Compact runtime directory missing: '+str(path))
base=Path(sys.base_prefix).resolve()
sys.path=[p for p in sys.path if p and Path(p).resolve().is_relative_to(base) and 'site-packages' not in p]
sys.path[:0]=[str(p) for p in reversed(paths)]
os.environ['PYTHONPATH']=os.pathsep.join(str(p) for p in paths)
os.environ['XDG_CACHE_HOME']=str(ROOT/'artifacts/.cache')
os.environ['PYTHONNOUSERSITE']='1'
from cad_recovery.compact_runtime_v1 import activate
activate()
runpy.run_path(str(ROOT/'scripts/run_compact_demo_v1.py'),run_name='__main__')
