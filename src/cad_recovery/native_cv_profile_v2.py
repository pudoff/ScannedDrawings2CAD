"""Experimental native CV v2, retaining the immutable v1 composition contract."""
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path

from . import open_arc_search_v2
from .family_catalog_v2 import require
from .dataset_layout_v1 import read

BASE_SHA256 = '71b323aa9ae31104e9d229857b382bc1fe1cedccb2614ad2696a5f2843ca429e'
_path = Path(__file__).with_name('native_cv_profile_v1.py')
require(hashlib.sha256(_path.read_bytes()).hexdigest() == BASE_SHA256, 'Frozen native CV profile changed')
_spec = importlib.util.spec_from_file_location('cad_recovery._native_cv_kernel_v2', _path)
_kernel = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_kernel)
_kernel.POLICY = deepcopy(_kernel.POLICY)
_kernel.POLICY.update(id='native-cv-experimental/2.0', arc_profile=open_arc_search_v2.POLICY['version'])
_kernel.arcs = open_arc_search_v2
_kernel.CODE = list(_kernel.CODE)+['native_cv_profile_v2', 'open_arc_search_v2']
POLICY = _kernel.POLICY
analyze, code_pins = _kernel.analyze, _kernel.code_pins


def verify(output):
    report = _kernel.verify(output)
    arc = read(Path(output)/'open-arcs.json'); stats = arc['analysis']; queue = stats['queue']
    require(arc['schema_version'] == 'open-arc-search/2.0' and arc['policy'] == open_arc_search_v2.POLICY,
            'Changed native arc policy')
    require(len(queue) == stats['seed_count'] and [r['seed_index'] for r in queue] == list(range(len(queue)))
        and all(r['state'] in ('screen_rejected', 'refined', 'awaiting_refinement') for r in queue), 'Changed arc queue inventory')
    ranked = sorted((r for r in queue if r['priority'] is not None), key=lambda r: -r['priority'])
    require([r['rank'] for r in ranked] == list(range(1, len(ranked)+1)) and len(ranked) == stats['eligible_seeds']
        and sum(r['state'] == 'refined' for r in queue) == stats['refinements']
        and sum(r['state'] == 'awaiting_refinement' for r in queue) == stats['pending_refinements'], 'Changed arc queue accounting')
    accepted = {r['candidate_id']: r for r in queue if r.get('refinement_reason') == 'accepted'}
    require(set(accepted) == {c['id'] for c in arc['candidates']}, 'Changed accepted arc queue')
    for c in arc['candidates']:
        row = accepted[c['id']]; witnesses = c['evidence']['native_tangent_witnesses']
        require(row['seed'] == c['evidence']['seed'] and row['tangent_witnesses'] == witnesses
            and [w['endpoint'] for w in witnesses] == [i for i, m in enumerate(c['evidence']['endpoint_methods']) if m == 'straight_tangent'],
            'Changed native tangent evidence')
        require(all(w['accepted'] is True and 0 < w['sample_count'] and
            w['native_support'] == w['supported_samples']/w['sample_count'] and
            open_arc_search_v2.POLICY['minimum_tangent_native_support'] <= w['native_support'] <= 1 for w in witnesses),
            'Unsupported accepted native tangent')
    return report
