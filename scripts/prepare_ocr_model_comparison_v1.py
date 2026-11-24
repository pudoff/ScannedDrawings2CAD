"""Freeze or verify P16.2i.1 data and model protocol without model inference."""
import argparse
import json
from pathlib import Path

from cad_recovery.dataset_layout_v1 import pin
from cad_recovery.ocr_model_comparison_protocol_v1 import freeze, verify

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.verify:
        result = verify(root, pin(root, args.protocol.resolve().relative_to(root).as_posix()))
    else:
        if args.data is None:
            parser.error('--data is required when freezing')
        result = freeze(root, args.protocol, args.data)
    print(json.dumps(result, ensure_ascii=False))
