"""Reproducible JSON entry points for calibration and constraint proposals."""
import hashlib
import json
import math
from pathlib import Path

from .geometry_constraints_v2 import fit_view_scale, solve_constraints, apply_view_calibration
from .graph_constraint_solver_v2 import solve_graph_view
from .graph_v2 import read_graph_v2

MODULES = ('geometry_constraints_v2.py', 'graph_constraint_solver_v2.py', 'geometry_solver_run_v2.py', 'graph_v2.py', 'schema.py')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _load(path):
    path = Path(path).resolve()
    if path.stat().st_size > 16*1024**2:
        raise ValueError('Solver JSON input exceeds 16 MiB')
    def number(text):
        value = float(text)
        if not math.isfinite(value):
            raise ValueError('Nonfinite JSON numbers are not solver inputs')
        return value
    def pairs(items):
        result = dict(items)
        if len(result) != len(items):
            raise ValueError('Repeated JSON field in solver input')
        return result
    def invalid_constant(text):
        raise ValueError('Invalid JSON numeric constant: '+text)
    return json.loads(path.read_text(encoding='utf-8'), parse_float=number,
                      parse_constant=invalid_constant, object_pairs_hook=pairs)


def _write(path, data):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(data, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write('\n')


def run_geometry_solver(mode, source, output, *, graph_path=None, view_id=None):
    if mode not in ('calibrate', 'solve', 'graph'):
        raise ValueError('Choose calibrate, solve or graph')
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Keep previous solver artifacts; choose a new directory')
    if (mode == 'graph') != (view_id is not None) or (graph_path is not None and mode != 'calibrate'):
        raise ValueError('view_id belongs to graph mode; optional graph_path belongs to calibrate mode')
    pins = [{'path': str(source), 'sha256': sha(source)}]
    data = _load(source)
    calibrated = None
    if mode == 'calibrate':
        result = fit_view_scale(data)
        if graph_path is not None:
            graph_path = Path(graph_path).resolve()
            pins.append({'path': str(graph_path), 'sha256': sha(graph_path)})
            graph = read_graph_v2(_load(graph_path))
            if result['status'] == 'solved':
                calibrated = apply_view_calibration(graph, result)
    elif mode == 'solve':
        result = solve_constraints(data)
    else:
        result = solve_graph_view(data, view_id)
    for pin in pins:
        if sha(pin['path']) != pin['sha256']:
            raise ValueError('Solver input changed during processing')
    report = {'schema_version': '1.0', 'mode': mode, 'inputs': pins, 'result': result,
              'implementation': {name: sha(Path(__file__).parent/name) for name in MODULES},
              'source_modified': False, 'training_eligible': False,
              'calibrated_graph_written': calibrated is not None,
              'limitations': ['Evidence verification is supplied explicitly by the caller, not inferred from numeric text.',
                              'Local rank does not prove global uniqueness or physical correctness of a source drawing.',
                              'Constraint results are review proposals and do not rewrite the graph or dimension values.']}
    output.mkdir(parents=True, exist_ok=False)
    _write(output/'report.json', report)
    _write(output/'input-snapshot.json', data)
    if calibrated is not None:
        _write(output/'calibrated-graph.json', calibrated.model_dump(mode='json'))
    _write(output/'manifest.json', {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()})
    return report
