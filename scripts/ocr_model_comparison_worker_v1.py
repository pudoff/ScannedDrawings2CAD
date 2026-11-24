"""Pinned offline OCR worker. Receives models and anonymous images, no truth."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from cad_recovery.dataset_layout_v1 import checked, read, inside
from cad_recovery.gpu_inference_v1 import deny_python_network
from cad_recovery.ocr_model_comparison_protocol_v1 import ARMS, BACKEND, POLICY, require
from cad_recovery.ocr_model_comparison_runtime_v1 import run_batch, qualify

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', required=True)
    parser.add_argument('--contract-sha256', required=True)
    parser.add_argument('--request', required=True)
    parser.add_argument('--request-sha256', required=True)
    parser.add_argument('--qualify', action='store_true')
    args = parser.parse_args()
    with deny_python_network():
        contract = read(checked(ROOT, {'path': args.contract, 'sha256': args.contract_sha256}))
        require(set(contract) == {'schema_version', 'models', 'arms', 'backend', 'runtime_limits', 'code'}, 'Unexpected contract metadata')
        require(contract['arms'] == ARMS and contract['backend'] == BACKEND and
                contract['runtime_limits'] == {k: POLICY[k] for k in ('limits', 'precision', 'intra_op_threads', 'inter_op_threads', 'warm_repeats_per_case')},
                'Contract drift')
        for record in list(contract['models'].values())+contract['code']:
            checked(ROOT, record)
        job = read(checked(ROOT, {'path': args.request, 'sha256': args.request_sha256}))
        if args.qualify:
            require(set(job) == {'output'}, 'Qualification accepts no drawing inputs')
            qualify(ROOT, contract, inside(ROOT, job['output']))
        else:
            print(run_batch(ROOT, contract, job))
