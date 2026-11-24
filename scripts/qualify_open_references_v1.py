"""Train-only source bank and qualification before opening fresh validation targets."""
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from cad_recovery.dataset_layout_v1 import checked, pin
from cad_recovery.source_chain_profile_v2 import analyze, diagnostic_heads
from cad_recovery.head_junction_ownership_v1 import apply
from cad_recovery.open_reference_chains_v1 import extend
from head_junction_inputs_v1 import prepare
from open_reference_inputs_v1 import reconstruct
from run_topology_robust_v1 import read, write, object_hash

ROOT = Path(__file__).resolve().parents[1]
CONTROLS = {'path': 'artifacts/open-reference-controls-v1/contract.json',
            'sha256': 'f94ed4b1c00f6b5e5f5e65735009926d8652a8c73475abc635530688bda7f200'}
BASE = ROOT / 'artifacts/open-reference-qualification-v1'


def bank():
    cv2.setNumThreads(1); BASE.mkdir(exist_ok=False)
    contract = read(checked(ROOT, CONTROLS)); records = []
    for job in contract['jobs']:
        if job['case']['split'] != 'train': continue
        for view in job['views']:
            gray, source, geometry, inherited, previous, local, heads, extra, calls = prepare(ROOT, view)
            advanced = diagnostic_heads(gray, heads)
            raw = analyze(gray, previous, inherited, local['local']['text_mask'], advanced['candidates'], diagnostic_heads=extra)
            packet = apply(gray, previous, inherited, raw, advanced['candidates'])
            folder = BASE / job['case']['id'] / view['mode']; folder.mkdir(parents=True)
            Image.fromarray(packet['pixel_roles']).save(folder / 'roles.png')
            write(folder / 'graph.json', {k: previous[k] for k in ('proposals', 'tracks', 'boundaries', 'hatch_bundles', 'stroke_px', 'search_incomplete')})
            write(folder / 'heads.json', advanced)
            write(folder / 'base.json', {k: v for k, v in packet.items() if k not in ('pixel_roles', 'pixel_interval_owner', 'junction_abstention_mask')})
            records.append({'case': job['case'], 'view': view, 'folder': folder.relative_to(ROOT).as_posix(),
                            'heads_incomplete': advanced['search_incomplete'], 'preparation': calls})
        print({'prepared': job['case']['id']}, flush=True)
    write(BASE / 'index.json', records)


def qualify(output):
    records = read(BASE / 'index.json'); results = []
    for record in records:
        folder = ROOT / record['folder']; gray = np.array(Image.open(checked(ROOT, record['view']['gray'])))
        previous = read(folder / 'graph.json'); heads = read(folder / 'heads.json')['candidates']; packet = read(folder / 'base.json')
        packet['pixel_roles'] = np.array(Image.open(folder / 'roles.png'))
        allowed = set(packet['head_ids_used']); used = [h for h in heads if h['id'] in allowed]
        before = object_hash([previous, heads]); result = extend(gray, previous, packet, used)
        assert before == object_hash([previous, heads])
        results.append({'case': record['case']['id'], 'mode': record['view']['mode'],
                        'expected': record['case']['labels']['expected_dimension_chains'],
                        'base': len(packet['dimension_chains']), 'new': result['open_reference_chains'],
                        'audit': result['open_reference_audit'], 'incomplete': result['search_incomplete'],
                        'rejections': result['open_reference_rejections']})
    write(ROOT / output, {'controls': CONTROLS, 'code': pin(ROOT, 'src/cad_recovery/open_reference_chains_v1.py'),
                          'train_only': True, 'rows': results})
    print({'views': len(results), 'new_chains': sum(len(r['new']) for r in results),
           'negative_new_chains': sum(len(r['new']) for r in results if not r['expected']),
           'incomplete': sum(r['incomplete'] for r in results)}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['bank', 'qualify']); parser.add_argument('--output')
    args = parser.parse_args()
    if args.mode == 'bank': bank()
    else: qualify(args.output)
