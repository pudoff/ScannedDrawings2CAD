"""Run opaque RGB requests in a fresh CPU process."""
from pathlib import Path
import sys
from cad_recovery.semantic_region_experiment_v1 import worker

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    worker(root,{'path':sys.argv[1],'sha256':sys.argv[2]},Path(sys.argv[3]),sys.argv[4])
