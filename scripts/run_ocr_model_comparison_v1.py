"""Prepare, qualify, run or replay the frozen CPU OCR comparison."""
import argparse
from pathlib import Path
import json

from cad_recovery.dataset_layout_v1 import checked, encoded, pin, read
from cad_recovery.ocr_model_comparison_execution_v1 import freeze, run, load, preserve
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
            result = read(args.output/'result/qualification.json') if (args.output/'result/qualification.json').exists() else None
            admission = {'execution_contract': record, 'budget_pass': runtime['budget_pass'] and result is not None,
                'customer_calls': 0, 'runtime': pin(root, (args.output/'process/runtime.json').resolve().relative_to(root).as_posix()),
                'result': pin(root, (args.output/'result/qualification.json').resolve().relative_to(root).as_posix()) if result else None}
            (args.output/'admission.json').write_bytes(encoded(admission))
            print(json.dumps(admission))
        elif args.mode == 'run':
            if args.admission is None: parser.error('--admission is required for customer comparison')
            admission = pin(root, args.admission.resolve().relative_to(root).as_posix())
            run(root, record, args.output, admission_pin=admission)
            config, _, _ = load(root, record)
            report = write_report(root, config, args.output)
            preserve(root)
            print(json.dumps(report['comparisons'], ensure_ascii=False))
        else:
            config, _, _ = load(root, record)
            expected = build_report(root, config, args.output)
            assert (args.output/'report.json').read_bytes() == encoded(expected), 'Replay differs'
            print(json.dumps({'verified': True, 'model_calls': 0, 'comparisons': expected['comparisons']}, ensure_ascii=False))
