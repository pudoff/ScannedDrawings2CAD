"""Register adversarial geometry/QA truth before implementing robust predicates."""
import json
import math
from pathlib import Path

from cad_recovery.dataset_layout_v1 import pin

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'artifacts/topology-robust-v1'
FAMILIES = [('nested_boxes', 'train'), ('nested_circles', 'train'), ('near_nodes', 'train'), ('safe_simplification', 'train'),
            ('near_disjoint_boxes', 'validation'), ('contacts', 'validation'), ('non_simple', 'validation'), ('flatten_limit', 'validation')]


def line(name, a, b, owner=None):
    return {'id': name, 'path_id': owner or name, 'kind': 'line', 'points_px': [a, b]}


def box(name, x, y, size):
    return {'id': name, 'path_id': name, 'kind': 'polyline', 'closed': True,
            'points_px': [[x, y], [x+size, y], [x+size, y+size], [x, y+size]]}


def cases():
    output = []
    for family, split in FAMILIES:
        for variant, offset in enumerate((0., 1_000_000., 9_000_000.)):
            x, y = offset+5.123, offset+7.231
            expectation = {'guard_accepted': True, 'nesting_complete': True}
            if family == 'nested_boxes':
                a = [box('outer', x, y, 20), box('inner', x+.013, y+.013, 19.974)]
                expectation['depths'] = [0, 1]
            elif family == 'nested_circles':
                a = [{'id': name, 'path_id': name, 'kind': 'circle', 'center_px': [x+10, y+10], 'radius_px': r}
                     for name, r in [('outer', 5.), ('hole', 1.25)]]
                expectation['depths'] = [0, 1]
            elif family == 'near_nodes':
                a = [line('axis', [x-10, y], [x+10, y]), line('left', [x, y], [x, y+8]),
                     line('right', [x+1, y], [x+1, y-8])]
            elif family == 'safe_simplification':
                a = [{'id': 'source', 'path_id': 'edge', 'kind': 'polyline', 'closed': False,
                      'points_px': [[x, y], [x+3, y], [x+10, y]]}]
            elif family == 'near_disjoint_boxes':
                a = [box('left', x, y, 1), box('right', x+1.00003, y, 1)]
                expectation['depths'] = [0, 0]
            elif family == 'contacts':
                a = [box('left', x, y, 10), box('right', x+10, y+10, 5)]
                expectation = {'guard_accepted': False, 'nesting_complete': False, 'reason': 'boundary_contact'}
            elif family == 'non_simple':
                a = [box('outer', x, y, 20), box('hole', x+5, y+5, 5),
                     line('bridge', [x, y], [x+5, y+5])]
                expectation = {'guard_accepted': False, 'nesting_complete': False, 'reason': 'non_simple_boundary'}
            else:
                points = [[x+20*math.cos(i*2*math.pi/520), y+20*math.sin(i*2*math.pi/520)] for i in range(520)]
                a = [{'id': 'many', 'path_id': 'many', 'kind': 'polyline', 'points_px': points, 'closed': True}]
                expectation = {'guard_accepted': False, 'nesting_complete': False, 'reason': 'flattened_vertex_budget'}
            b = json.loads(json.dumps(a))
            evidence = None
            if family == 'near_nodes':
                # Both junctions move coherently; the source path identities are unchanged.
                for c in b:
                    c['points_px'] = [[p[0]+.4, p[1]] for p in c['points_px']]
                if variant == 2:
                    b[-1]['points_px'][0][1] += .01  # Deliberate lost branch, not a matching improvement.
                    expectation['guard_accepted'] = False
            elif family == 'safe_simplification':
                b = [line('new', [x, y], [x+10, y], 'edge')]
            output.append({'id': f'robust-control-{family}-v{variant}', 'family': family, 'split': split,
                'source': a, 'proposal': b, 'node_evidence': evidence, 'expected': expectation,
                'authority': 'analytic construction before robust implementation', 'physical_geometry_approved': False})
    return output


if __name__ == '__main__':
    BASE.mkdir(exist_ok=False)
    value = {'task': 'P15.2d.2', 'families': FAMILIES, 'cases': cases(),
             'generator': pin(ROOT, 'scripts/prepare_topology_robust_controls_v1.py'),
             'prepared_before_robust_implementation': True,
             'fixed_budgets': {'maximum_flat_vertices': 512, 'maximum_topology_predicates': 250000,
                              'maximum_nesting_checks': 100000, 'maximum_node_distance_native_px': 3.},
             'acceptance': 'all analytic expected decisions; unsafe/touch/non-simple/budget remain rejected; no geometry changes or production admission',
             'new_OCR_ML_calls': 0, 'new_geometry_extractions': 0, 'training': False, 'closed_test_used': False}
    path = BASE/'controls-contract.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(pin(ROOT, path.relative_to(ROOT).as_posix())))
