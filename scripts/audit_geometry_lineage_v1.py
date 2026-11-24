"""Read-only geometry-lineage audit; output must be a new directory."""
import argparse
import json
import os
from pathlib import Path

from cad_recovery.geometry_lineage_v1 import build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path.cwd())
    parser.add_argument('--config', default='configs/reviews/geometry-lineage-v1.json')
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    os.environ.setdefault('XDG_CACHE_HOME', str(args.workspace/'artifacts/.cache'))
    result = build(args.workspace, {'path': args.config, 'sha256': args.config_sha256}, args.output,
                   progress=lambda n, total: print(f'Geometry fingerprints: {n}/{total}', flush=True))
    print(json.dumps(result, ensure_ascii=True))


if __name__ == '__main__':
    main()
