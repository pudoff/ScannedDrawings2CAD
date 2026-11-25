"""Unchanged correction builder with optimization-safe child telemetry."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
for name in ('src','scripts','artifacts/service-runtime-v1/deps','artifacts/cv-skeleton-runtime-v1/deps'):
    sys.path.insert(0,str(ROOT/name))
from cad_recovery.sealed_child_v3 import child_main
from build_web_correction_v2 import operation

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--request',type=Path,required=True)
    child_main(parser.parse_args().request,operation)
