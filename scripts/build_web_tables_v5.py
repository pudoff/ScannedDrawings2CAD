"""Retained legacy dispatch plus separate text-only publication."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
for name in ('src', 'scripts', 'artifacts/service-runtime-v1/deps', 'artifacts/cv-skeleton-runtime-v1/deps'):
    sys.path.insert(0, str(ROOT/name))
from cad_recovery.runtime_worker_v1 import child_main


def execute(request, progress):
    from cad_recovery.web_tables_contract_v4 import binding
    if request['table_binding']['sha256'] == binding()['sha256']:
        from cad_recovery.web_tables_build_v4 import build
        with progress.phase('text_only_choice' if request['edit'].get('text_selection') else 'table_OCR_evidence'):
            build(request, progress=progress)
    else:
        from build_web_tables_v4 import execute as previous
        previous(request, progress)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--request', type=Path, required=True)
    child_main(parser.parse_args().request, execute)
