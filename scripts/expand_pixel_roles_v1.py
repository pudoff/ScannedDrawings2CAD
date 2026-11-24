"""Select, release and replay the next partial pixel-role dataset."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cad_recovery import pixel_role_expansion_v1 as expansion
from cad_recovery.dataset_layout_v1 import encoded,pin

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['select','freeze','build','verify'])
    for name in ('output','selection','reading','config','manifest','sha256'):p.add_argument('--'+name)
    a=p.parse_args()
    if a.command=='select':
        target=ROOT/a.output
        if target.exists():raise ValueError('Choose new selection path')
        target.write_bytes(encoded(expansion.selection(ROOT)));value=pin(ROOT,a.output)
    elif a.command=='freeze':value=expansion.freeze(ROOT,a.selection,a.reading,ROOT/a.output)
    elif a.command=='build':value=expansion.build(ROOT,{'path':a.config,'sha256':a.sha256},a.output)
    else:
        report,rows=expansion.verify(ROOT,{'path':a.manifest,'sha256':a.sha256})
        value={'verified':True,'patches':len(rows),'sources':report['sources_by_split']}
    print(json.dumps(value))
