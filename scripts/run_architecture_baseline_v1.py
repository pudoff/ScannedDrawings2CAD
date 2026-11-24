"""Freeze/run/replay the common partial CV, U-Net, OCR and signs baseline."""
import argparse
from pathlib import Path
from cad_recovery.architecture_baseline_v1 import freeze, new_cv, run, verify
from cad_recovery.dataset_layout_v1 import inside, pin


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('action', choices=['freeze', 'run', 'child', 'verify'])
    parser.add_argument('--config', default='configs/reviews/architecture-baseline-v1.json')
    parser.add_argument('--output', default='artifacts/architecture-baseline-v1')
    args = parser.parse_args(); root = Path(__file__).resolve().parents[1]
    config = inside(root, args.config); destination = inside(root, args.output)
    if args.action == 'freeze': print(freeze(root, config)); return
    record = pin(root, args.config)
    if args.action == 'child': new_cv(root, record, destination / 'new-cv')
    elif args.action == 'verify': print(verify(root, record, destination))
    else:
        report = run(root, record, destination)
        print({k: report[k] for k in ['ROI_count', 'CV_task_count', 'new_CV_tasks', 'OCR_validation_lines']})


if __name__ == '__main__': main()
