"""Experimental native CV with a finite event/diversity arc queue."""
import hashlib
import importlib.util
from pathlib import Path

from . import open_arc_search_v4 as arcs
from .dataset_layout_v1 import read
from .family_catalog_v2 import require

_path = Path(__file__).with_name('native_cv_profile_v1.py')
require(hashlib.sha256(_path.read_bytes()).hexdigest() ==
    '71b323aa9ae31104e9d229857b382bc1fe1cedccb2614ad2696a5f2843ca429e', 'Frozen CV kernel changed')
_spec = importlib.util.spec_from_file_location('cad_recovery._native_cv_kernel_v4', _path)
_core = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_core)
_core.POLICY.update(id='native-cv-experimental/4.0', arc_profile=arcs.POLICY['version'],
    arc_queue_profile='events_diverse')
_core.arcs = arcs
_core.CODE += ['open_arc_search_v2', 'open_arc_search_v3', 'open_arc_search_v4',
    'arc_final_validation_v1', 'native_cv_profile_v4']
POLICY, analyze, code_pins = _core.POLICY, _core.analyze, _core.code_pins


def verify(output):
    report = _core.verify(output); arc = read(Path(output)/'open-arcs.json')
    require(arc['schema_version'] == 'open-arc-search/3.0' and arc['policy'] == arcs.POLICY, 'Changed arc queue policy')
    stats = arc['analysis']; queue = stats['queue']
    require(len(queue) == stats['seed_count'] and [r['seed_index'] for r in queue] == list(range(len(queue))), 'Changed seed inventory')
    eligible = [r for r in queue if r['priority'] is not None]
    ranked = arcs.ordered(eligible, arcs.POLICY['queue_profile'])
    require([r['rank'] for r in ranked] == list(range(1, len(ranked)+1)) and len(ranked) == stats['eligible_seeds'], 'Changed queue order')
    require(sum(r['state'] == 'refined' for r in queue) == stats['refinements'] <= arcs.POLICY['maximum_refinements'] and
        sum(r['state'] == 'awaiting_refinement' for r in queue) == stats['pending_refinements'], 'Changed refinement accounting')
    accepted = {r['candidate_id']: r for r in queue if r.get('refinement_reason') == 'accepted'}
    require(set(accepted) == {c['id'] for c in arc['candidates']}, 'Changed accepted queue IDs')
    for curve in arc['candidates']:
        row = accepted[curve['id']]; geometry = arcs.previous.final_geometry(curve)
        require(geometry['accepted'] and row['final_geometry'] == curve['evidence']['final_geometry'] == geometry and
            row['seed'] == curve['evidence']['seed'], 'Changed final geometry evidence')
        check = curve['evidence']['native_centerline']
        require(row['native_centerline'] == check and check['accepted'] and check['sample_count'] >= 12 and
            0 <= check['rmse_px'] <= arcs.POLICY['maximum_seed_rmse_px'], 'Changed native centreline evidence')
        require(row['tangent_witnesses'] == curve['evidence']['native_tangent_witnesses'] and
            all(w['accepted'] for w in row['tangent_witnesses']), 'Changed native tangents')
    require(sum(stats['costs'][k] for k in ('screen_radial_samples', 'refinement_radial_samples')) <=
        arcs.POLICY['maximum_discovery_radial_samples'] and stats['costs']['screen_boundary_samples'] <=
        arcs.POLICY['maximum_screen_boundary_samples'] and stats['costs']['screen_line_tests'] <=
        arcs.POLICY['maximum_screen_line_tests'], 'Changed screen budget accounting')
    return report
