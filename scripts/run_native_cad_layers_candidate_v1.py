"""9.1.2: five new full source-only runs against the frozen 9.1.1 results."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from cad_recovery.dataset_layout_v1 import checked, digest, pin, read
from run_native_cad_demo_candidate_v1 import write


def child(request_path):
    from cad_recovery.profile_pipeline_v4 import run_profile
    from cad_recovery.native_annotation_inputs_v1 import load_inputs
    from cad_recovery.native_cad_layers_candidate_v1 import export_candidate
    request = read(request_path); source = checked(ROOT, request['source'])
    output = ROOT/request['output']
    base = run_profile(source, output, request['options'], models=ROOT/'artifacts/models/ppocr-v5',
                       expected_binding=request['binding'])
    page = base['pages'][0]
    packet, protection, annotations, _, _ = load_inputs(ROOT,
        output/'private/page-001/experimental', output/page['artifacts']['ocr_report'])
    target = output/'candidate'; target.mkdir()
    report = export_candidate(packet, annotations, target/'page-001.dxf', protected=protection)
    report['candidate_scope'] = 'source_confirmed_frame_layers_not_Web_deployed'
    write(target/'report.json', report)
    checked(ROOT, request['source'])
    write(request_path.parent/'completed.json', dict(ok=True))


def run(output):
    from cad_recovery.hard_raster_runs_v1 import execute
    from cad_recovery.profile_contract_v4 import resolve_profile
    prior = ROOT/'artifacts/native-cad-demo-candidate-v1'
    baseline_protocol = read(prior/'protocol.json')
    for ref in baseline_protocol['code']+baseline_protocol['sources']:
        checked(ROOT, ref)
    if resolve_profile('native_cad_draft') != baseline_protocol['binding']:
        raise ValueError('Web baseline binding changed')
    output.mkdir(parents=True, exist_ok=False)
    code = [pin(ROOT, p.relative_to(ROOT).as_posix()) for p in sorted((ROOT/'src/cad_recovery').rglob('*.py'))]
    code.append(pin(ROOT, Path(__file__).relative_to(ROOT).as_posix()))
    plan = dict(task='9.1.2', prior_protocol=pin(ROOT, (prior/'protocol.json').relative_to(ROOT).as_posix()),
        prior_comparison=pin(ROOT, (prior/'comparison.json').relative_to(ROOT).as_posix()),
        sources=baseline_protocol['sources'], code=code, options=baseline_protocol['options'],
        binding=baseline_protocol['binding'], limits=dict(seconds=180, memory_MiB=1024),
        unchanged_frame_thresholds=True, scoring_inputs_forwarded=False, independent=False, web_changed=False)
    write(output/'protocol.json', plan)
    rows = []
    for index, source in enumerate(plan['sources']):
        folder = output/f'{index+1:02d}-C'; folder.mkdir()
        write(folder/'request.json', dict(source=source, output=(folder/'output').relative_to(ROOT).as_posix(),
            options=plan['options'], binding=plan['binding']))
        print(f'Start {index+1}/5: {source["path"]}', flush=True)
        process = execute(ROOT, [sys.executable, str(Path(__file__).resolve()), '--child', str(folder/'request.json')],
                          folder/'process', seconds=180, memory_MiB=1024)
        rows.append(dict(source=source, variant='C', folder=folder.relative_to(ROOT).as_posix(),
                         process=process, completed=(folder/'completed.json').is_file()))
        write(output/'progress.json', dict(cases=rows))
        print(f'Finish {index+1}/5: {process["failure"]}, {process["walltime_seconds"]:.1f}s', flush=True)
    for ref in code+plan['sources']:
        checked(ROOT, ref)
    write(output/'batch.json', dict(cases=rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--child', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.child:
        child(args.child)
    elif args.output:
        run(args.output.resolve())
    else:
        parser.error('Specify --child or --output')
