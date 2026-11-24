"""Freeze the VLM adapter/catalog and run its separate synthetic qualification."""
import argparse
from pathlib import Path
from cad_recovery.dataset_layout_v1 import inside, pin
from cad_recovery.vlm_protocol_v1 import freeze, qualify, execute_requests


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('action', choices=['freeze', 'qualify', 'run'])
    parser.add_argument('--config', default='configs/reviews/vlm-adapter-v2.json')
    parser.add_argument('--catalog', default='artifacts/vlm-catalog-v2')
    parser.add_argument('--request', default='artifacts/vlm-catalog-v2/qualification-request.json')
    parser.add_argument('--output', default='artifacts/vlm-qualification-v2')
    args = parser.parse_args(); root = Path(__file__).resolve().parents[1]
    if args.action == 'freeze': print(freeze(root, inside(root, args.config), inside(root, args.catalog))); return
    config, request = pin(root, args.config), pin(root, args.request)
    if args.action == 'qualify': print(qualify(root, config, request, inside(root, args.output)))
    else: print(execute_requests(root, config, request, inside(root, args.output)))


if __name__ == '__main__': main()
