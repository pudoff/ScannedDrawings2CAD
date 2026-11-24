"""Native CV v3: final resolved arcs, retaining the frozen composition kernel."""
import hashlib
import importlib.util
from pathlib import Path

from . import open_arc_search_v3 as arcs
from .dataset_layout_v1 import read
from .family_catalog_v2 import require

BASE_SHA256 = '970bde0188bf93b4be1ac2b59541a22475b83e83ae14578b13efb80af7aede98'
_path = Path(__file__).with_name('native_cv_profile_v2.py')
require(hashlib.sha256(_path.read_bytes()).hexdigest() == BASE_SHA256, 'Frozen v2 CV profile changed')
_spec = importlib.util.spec_from_file_location('cad_recovery._native_cv_adapter_v3', _path)
_previous = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_previous)
_core = _previous._kernel
_core.POLICY.update(id='native-cv-experimental/3.0', arc_profile=arcs.POLICY['version'])
_core.arcs = arcs
_core.CODE += ['native_cv_profile_v3', 'open_arc_search_v3']
_previous.open_arc_search_v2 = arcs
POLICY, analyze, code_pins = _core.POLICY, _core.analyze, _core.code_pins


def verify(output):
    report = _previous.verify(output)
    arc = read(Path(output)/'open-arcs.json'); queue = arc['analysis']['queue']
    for curve in arc['candidates']:
        witness = arcs.final_geometry(curve)
        require(witness['accepted'] and curve['evidence']['final_geometry'] == witness,
                'Changed final arc geometry evidence')
        row = next(r for r in queue if r.get('candidate_id') == curve['id'])
        require(row['final_geometry'] == witness, 'Changed final arc queue evidence')
        check = curve['evidence']['native_centerline']
        require(row['native_centerline'] == check and check['accepted'] is True and
            check['sample_count'] >= 12 and 0 <= check['rmse_px'] <= arcs.POLICY['maximum_seed_rmse_px'],
            'Changed final native centerline evidence')
    decisions = [r for r in queue if 'final_geometry' in r]
    require(arc['analysis']['final_validation']['examined'] == len(decisions) and
        arc['analysis']['final_validation']['retained'] == len(arc['candidates']), 'Changed final validation counts')
    return report
