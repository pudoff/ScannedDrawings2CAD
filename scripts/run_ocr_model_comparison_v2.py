"""Run the unchanged OCR experiment with corrected Windows progress transport."""
import argparse
import json
from pathlib import Path

from cad_recovery.dataset_layout_v1 import encoded, pin, read
from cad_recovery.ocr_model_comparison_execution_v2 import freeze, run, load
from cad_recovery.ocr_model_comparison_execution_v1 import preserve
from cad_recovery.ocr_model_comparison_evaluation_v1 import write_report, build_report

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['freeze', 'qualify', 'run', 'verify'])
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--admission', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.mode == 'freeze':
        print(json.dumps(freeze(root, args.config, args.output)))
    else:
        record = pin(root, args.config.resolve().relative_to(root).as_posix())
        if args.mode == 'qualify':
            runtime = run(root, record, args.output, qualification=True)
            payload = args.output/'result/qualification.json'
            admission = {'execution_contract': record, 'budget_pass': runtime['budget_pass'] and payload.exists(),
                'customer_calls': 0, 'runtime': pin(root, (args.output/'process/runtime.json').resolve().relative_to(root).as_posix()),
                'result': pin(root, payload.resolve().relative_to(root).as_posix()) if payload.exists() else None}
            (args.output/'admission.json').write_bytes(encoded(admission))
            print(json.dumps(admission))
        elif args.mode == 'run':
            if args.admission is None: parser.error('--admission is required')
            run(root, record, args.output, admission_pin=pin(root, args.admission.resolve().relative_to(root).as_posix()))
            config, _, _ = load(root, record)
            report = write_report(root, config, args.output)
            preserve(root)
            print(json.dumps(report['comparisons'], ensure_ascii=False))
        else:
            config, _, _ = load(root, record)
            assert (args.output/'report.json').read_bytes() == encoded(build_report(root, config, args.output)), 'Replay differs'
            print(json.dumps({'verified': True, 'model_calls': 0}))
