"""Versioned file runner for correction previews and explicit approvals."""
from pathlib import Path

import ezdxf

from .exporter_v2 import export_dxf_v2, semantic_entity_snapshot
from .geometry_solver_run_v2 import _load, _write, sha
from .graph_correction_v1 import prepare_graph_correction, accept_graph_correction
from .graph_v2 import read_graph_v2

MODULES = ('graph_correction_v1.py', 'graph_correction_run_v1.py', 'graph_v2.py',
           'graph_constraint_solver_v2.py', 'geometry_constraints_v2.py',
           'exporter_v2.py', 'topology_v2.py', 'geometry_solver_run_v2.py')


def run_graph_correction(mode, graph_path, request_path, bindings_path, output, *, approval_path=None):
    if mode not in ('preview', 'apply') or (mode == 'apply') != (approval_path is not None):
        raise ValueError('Apply requires an approval file; preview cannot use one')
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Choose a new correction directory; previous versions are preserved')
    paths = {key: Path(path).resolve() for key, path in (
        ('graph', graph_path), ('request', request_path), ('bindings', bindings_path))}
    if approval_path is not None:
        paths['approval'] = Path(approval_path).resolve()
    pins = {key: {'path': str(path), 'sha256': sha(path)} for key, path in paths.items()}
    inputs = {key: _load(path) for key, path in paths.items()}
    graph = read_graph_v2(inputs['graph'])
    source = Path(graph.source.path).resolve()
    if not source.is_file() or sha(source) != graph.source.sha256:
        raise ValueError('Actual source bytes must match the graph source SHA-256')
    pins['source'] = {'path': str(source), 'sha256': graph.source.sha256}
    preview = prepare_graph_correction(graph, inputs['request'], inputs['bindings'])
    corrected = None
    if mode == 'apply':
        corrected = accept_graph_correction(graph, inputs['request'], inputs['bindings'], inputs['approval'])
    report = {'schema_version': '1.0', 'mode': mode, 'status': preview['status'], 'inputs': pins,
              'proposal_sha256': preview.get('proposal_sha256'), 'source_modified': False,
              'source_accuracy_approved': False, 'training_eligible': False,
              'global_uniqueness_proven': False, 'cad_associativity_verified': False,
              'implementation': {name: sha(Path(__file__).parent/name) for name in MODULES}}
    for pin in pins.values():
        if sha(pin['path']) != pin['sha256']:
            raise ValueError('Correction input changed during validation')
    output.mkdir(parents=True, exist_ok=False)
    _write(output/'proposal.json', preview)
    for name, snapshot in inputs.items():
        _write(output/(name+'-input.json'), snapshot)
    if corrected is not None:
        _write(output/'corrected-graph.json', corrected.model_dump(mode='json'))
        for name, value in (('original', graph), ('corrected', corrected)):
            export = export_dxf_v2(value, output/(name+'.dxf'))
            _write(output/(name+'-export.json'), export)
            report[name+'_export_status'] = export['status']
            checked_ids = {row['id'] for row in preview['stored_geometry_checks']}
            unresolved = [issue for issue in export['issues'] if not (
                issue['code'] == 'constraints_require_separate_verification' and
                set(issue['entity_ids']) <= checked_ids)]
            report[name+'_export_unresolved_issues'] = unresolved
            saved = ezdxf.readfile(output/(name+'.dxf'))
            _write(output/(name+'-semantic.json'), [semantic_entity_snapshot(e) for e in saved.modelspace()])
        report['status'] = 'applied_reviewed_correction' if all(report[n+'_export_status'] != 'failed'
            and not report[n+'_export_unresolved_issues']
            for n in ('original', 'corrected')) else 'export_requires_review'
        report['review_scope'] = inputs['approval']['scope']
    if any(sha(pin['path']) != pin['sha256'] for pin in pins.values()):
        report['status'] = 'input_changed_during_run'
    _write(output/'report.json', report)
    _write(output/'manifest.json', {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()})
    return report
