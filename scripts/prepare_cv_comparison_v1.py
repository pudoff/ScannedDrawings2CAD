"""Prepare, seal or verify the P15.2f.2 CV comparison inputs."""
import argparse
import json
from pathlib import Path

from cad_recovery.cv_comparison_protocol_v1 import prepare, seal, verify
from cad_recovery.dataset_layout_v1 import pin

ROOT = Path(__file__).resolve().parents[1]
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare','seal','verify'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reading', type=Path)
    parser.add_argument('--sha256')
    args = parser.parse_args(); folder = args.output.resolve()
    if args.mode == 'prepare': result = prepare(ROOT, folder)
    elif args.mode == 'seal':
        if args.reading is None: parser.error('--reading is required')
        result = seal(ROOT, folder, pin(ROOT,args.reading.resolve().relative_to(ROOT).as_posix()))
    else:
        if not args.sha256: parser.error('--sha256 is required')
        result = verify(ROOT, folder, args.sha256)
    print(json.dumps(result, ensure_ascii=False))
