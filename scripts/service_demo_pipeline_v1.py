"""Internal demo worker child; source and output are owned by the local service."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cad_recovery.demo_pipeline_v1 import run_demo_pipeline
from cad_recovery.service_v1.storage import digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args()
    from cad_recovery.service_demo_v1.config import ProcessingOptions
    request = json.loads(args.request.read_text('utf-8'))
    options = ProcessingOptions.model_validate(request['options'])
    try:
        if options.profile != 'presentation_draft':
            raise ValueError('Demo child requires presentation_draft')
        if digest(Path(request['source'])) != request['input_sha256']:
            raise ValueError('Input digest changed')
        run_demo_pipeline(request['source'], request['output'],
            ocr_models=Path(os.environ.get('CAD_DEMO_OCR_MODELS', ROOT / 'artifacts/models/ppocr-v5')),
            render_dpi=options.render_dpi, image_dpi=options.image_dpi, rotation_deg=options.rotation_deg)
    except Exception as error:
        write_json(args.request.parent / 'status.json', {'ok': False,
            'code': 'input_invalid' if isinstance(error, ValueError) else 'processing_failed',
            'type': type(error).__name__, 'message': str(error)[:1000]})
        raise
    write_json(args.request.parent / 'status.json', {'ok': True})


if __name__ == '__main__':
    main()
