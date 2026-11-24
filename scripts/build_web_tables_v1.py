"""Dedicated child for one leased table-review job; never writes database state."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
for name in ('src', 'scripts', 'artifacts/service-runtime-v1/deps', 'artifacts/cv-skeleton-runtime-v1/deps'):
    sys.path.insert(0, str(ROOT/name))
if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--request', type=Path, required=True)
    args = p.parse_args()
    from cad_recovery.dataset_layout_v1 import read
    from cad_recovery.web_tables_build_v1 import build
    build(read(args.request))
    from cad_recovery.dataset_layout_v1 import encoded
    (args.request.parent/'status.json').write_bytes(encoded({'ok': True}))
