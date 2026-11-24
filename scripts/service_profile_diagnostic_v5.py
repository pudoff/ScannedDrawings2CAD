"""Supervised V23 dispatch; old bindings use their unchanged producer."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for name in ('src', 'scripts', 'artifacts/service-runtime-v1/deps', 'artifacts/cv-skeleton-runtime-v1/deps'):
    sys.path.insert(0, str(ROOT/name))
from cad_recovery.sealed_child_v1 import child_main


def execute(request, progress):
    from cad_recovery.dataset_layout_v1 import digest
    from cad_recovery.profile_pipeline_v7 import run_profile
    if digest(request['source']) != request['input_sha256']:
        raise ValueError('Source changed')
    with progress.phase('document_conversion'):
        run_profile(request['source'], request['output'], request['options'],
            models=os.environ.get('CAD_DEMO_OCR_MODELS'), expected_binding=request['profile_binding'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--request', type=Path, required=True)
    child_main(parser.parse_args().request, execute)
