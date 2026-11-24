"""Owned, resource-bounded correction child process."""
import argparse
import json
from pathlib import Path
from cad_recovery.web_correction_build_v1 import build
from cad_recovery.dataset_layout_v1 import encoded

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--request',type=Path,required=True)
    args=parser.parse_args();request=json.loads(args.request.read_text(encoding='utf-8'))
    result=build(request);(Path(request['folder'])/'completion.json').write_bytes(encoded(result))
