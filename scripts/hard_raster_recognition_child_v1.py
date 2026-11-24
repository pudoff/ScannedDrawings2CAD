"""Recognition-only measurements on authorized, upright physical line crops."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser(); p.add_argument('--request', type=Path, required=True); args = p.parse_args()
    request = json.loads(args.request.read_text('utf-8'))
    from cad_recovery.ocr_engine import LocalOCREngine, _offline_backend_scope
    from rapidocr.ch_ppocr_rec.typings import TextRecInput
    from cad_recovery.updated_raster_batch import _peak_memory
    from cad_recovery.dataset_layout_v1 import encoded
    import numpy as np
    from PIL import Image
    engine = LocalOCREngine(Path(request['models'])); rows = []
    for item in request['queries']:
        path = Path(item['image']); started = time.perf_counter()
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']: raise ValueError('Changed query pixels')
        with Image.open(path) as image: bgr = np.array(image.convert('RGB'))[:, :, ::-1].copy()
        try:
            with _offline_backend_scope(): output = engine._backend.text_rec(TextRecInput(img=bgr, return_word_box=False))
            if len(output.txts) != 1: raise ValueError('Recognition output must contain one line')
            prediction = output.txts[0]
            row = {'id': item['id'], 'prediction': prediction, 'status': 'ok' if prediction else 'empty'}
        except Exception as error:
            row = {'id': item['id'], 'prediction': '', 'status': 'error', 'error': type(error).__name__+': '+str(error)}
        rows.append({**row, 'seconds': time.perf_counter()-started})
    Path(request['output']).write_bytes(encoded({'rows': rows, 'engine': engine.provenance, 'memory': _peak_memory(),
        'detector_executed': False, 'classifier_executed': False, 'weights_updated': False}))


if __name__ == '__main__': main()
