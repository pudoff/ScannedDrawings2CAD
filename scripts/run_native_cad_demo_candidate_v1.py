"""Finite 9.1.1 source-only A/candidate comparison; no service bindings changed."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from cad_recovery.dataset_layout_v1 import digest, pin, read


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def child(request_path):
    from cad_recovery.profile_pipeline_v4 import run_profile
    from cad_recovery.native_annotation_inputs_v1 import load_inputs
    from cad_recovery.native_cad_candidate_v1 import export_candidate
    request = read(request_path)
    source = ROOT/request['source']['path']
    if digest(source) != request['source']['sha256']:
        raise ValueError('Source changed')
    output = ROOT/request['output']
    baseline = run_profile(source, output, request['options'],
        models=ROOT/'artifacts/models/ppocr-v5', expected_binding=request['binding'])
    if request['variant'] == 'C':
        page = baseline['pages'][0]
        observations = output/'private/page-001/experimental'
        packet, protection, annotations, _, _ = load_inputs(
            ROOT, observations, output/page['artifacts']['ocr_report'])
        target = output/'candidate'
        target.mkdir()
        report = export_candidate(packet, annotations, target/'page-001.dxf', protected=protection)
        report['candidate_scope'] = 'geometry_adapter_with_existing_OCR_not_Web_deployed'
        write(target/'report.json', report)
    if digest(source) != request['source']['sha256']:
        raise ValueError('Source changed during inference')
    write(request_path.parent/'completed.json', {'ok': True})


def run(output):
    from cad_recovery.hard_raster_runs_v1 import execute
    from cad_recovery.profile_contract_v4 import resolve_profile
    output.mkdir(parents=True, exist_ok=False)
    sources = ['dataset/clear_sum_dataset/vse_png/звездочка.png',
               'dataset/test_png/чертеж131.png', 'dataset/test_png/чертеж133.png',
               'dataset/test_png/чертеж180.png', 'IMG_20261109_160629_212.jpg']
    binding = resolve_profile('native_cad_draft')
    options = dict(profile='native_cad_draft', render_dpi=120., image_dpi=None, rotation_deg=0)
    code = [pin(ROOT, p.relative_to(ROOT).as_posix()) for p in sorted((ROOT/'src/cad_recovery').rglob('*.py'))]
    code.append(pin(ROOT, Path(__file__).relative_to(ROOT).as_posix()))
    plan = dict(task='9.1.1', sources=[pin(ROOT, s) for s in sources], binding=binding,
                code=code, options=options, limits=dict(seconds=180, memory_MiB=1024),
                independent=False, training=False, web_changed=False,
                evaluation='whole-object existing CAD QA; annotation snapshots; all failures retained')
    write(output/'protocol.json', plan)
    rows = []
    for index, source in enumerate(plan['sources']):
        for variant in ('A', 'C'):
            folder = output/f'{index+1:02d}-{variant}'
            folder.mkdir()
            request = dict(source=source, variant=variant, binding=binding, options=options,
                           output=(folder/'output').relative_to(ROOT).as_posix())
            write(folder/'request.json', request)
            print(f'Start {index+1}/5 {variant}: {source["path"]}', flush=True)
            process = execute(ROOT, [sys.executable, str(Path(__file__).resolve()),
                '--child', str(folder/'request.json')], folder/'process', seconds=180, memory_MiB=1024)
            row = dict(source=source, variant=variant, folder=folder.relative_to(ROOT).as_posix(),
                       process=process, completed=(folder/'completed.json').is_file())
            rows.append(row)
            write(output/'progress.json', dict(cases=rows))
            print(f'Finish {index+1}/5 {variant}: {process["failure"]}, {process["walltime_seconds"]:.1f}s', flush=True)
    for ref in code+plan['sources']:
        if digest(ROOT/ref['path']) != ref['sha256']:
            raise ValueError('Pinned source/code changed')
    write(output/'batch.json', dict(cases=rows, protocol=pin(ROOT, (output/'protocol.json').relative_to(ROOT).as_posix())))


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
